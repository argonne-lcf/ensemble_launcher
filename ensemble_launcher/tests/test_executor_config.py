"""Tests for per-executor configuration (``LauncherConfig.executor_configs``).

The interesting failure modes are all on the wire. A ``LauncherConfig`` is serialised with
``model_dump_json``, embedded in a dict that is passed through ``json.dumps(..., default=str)``,
written to a temp file, and rebuilt in a freshly-spawned child process with
``model_validate_json`` (``AsyncMaster.asdict`` / ``_launch_child`` / ``AsyncMaster.fromdict``).
Anything that survives an in-process round trip but not that one shows up as a wrong executor
argument inside a worker, far from the config that caused it -- so every test here goes through
:func:`_through_the_wire`.
"""

import json

import pytest

from ensemble_launcher.config import (
    AsyncLokyConfig,
    AsyncMPIConfig,
    ExecutorConfig,
    LauncherConfig,
    MPIConfig,
)

pytestmark = pytest.mark.core


def _through_the_wire(config: LauncherConfig) -> LauncherConfig:
    """Round-trip a config exactly the way launching a child process does."""
    child_dict = {"config": config.model_dump_json()}
    revived = json.loads(json.dumps(child_dict, default=str))
    return LauncherConfig.model_validate_json(revived["config"])


# Stand-ins for the runtime-derived arguments the orchestrator computes per node.
RUNTIME = {
    "logger": "a-logger",
    "gpu_selector": "ZE_AFFINITY_MASK",
    "return_stdout": False,
    "max_workers": 32,
}


# ------------------------------------------------------------------ #
#  The default path                                                    #
# ------------------------------------------------------------------ #


def test_no_configs_is_the_identity():
    """A config that sets no `executor_configs` must not perturb any executor."""
    config = _through_the_wire(LauncherConfig())
    assert config.executor_configs == {}
    assert config.executor_kwargs("async_loky", **RUNTIME) == RUNTIME


def test_unconfigured_executor_is_untouched():
    """Configuring one executor must not reach the others."""
    config = _through_the_wire(
        LauncherConfig(executor_configs={"async_loky": {"max_workers": 4}})
    )
    assert config.executor_kwargs("async_mpi", **RUNTIME) == RUNTIME
    assert config.executor_kwargs("async_loky", **RUNTIME)["max_workers"] == 4


# ------------------------------------------------------------------ #
#  Crossing the wire                                                   #
# ------------------------------------------------------------------ #


def test_subclass_survives_the_wire():
    """`SerializeAsAny` plus the key-keyed validator must rebuild the right subclass.

    Without `SerializeAsAny` pydantic serialises to the declared base and drops every
    subclass field, so `timeout` would simply vanish between master and worker.
    """
    config = _through_the_wire(
        LauncherConfig(executor_configs={"async_loky": {"timeout": 60.0}})
    )
    loky = config.executor_configs["async_loky"]
    assert isinstance(loky, AsyncLokyConfig)
    assert loky.timeout == 60.0


def test_unset_fields_do_not_clobber_runtime_values():
    """The regression this design exists to prevent.

    `model_dump_json` writes unset fields as explicit nulls, so a naive implementation ends up
    with every field in `model_fields_set` on the far side and overwrites each runtime
    argument with `None`.
    """
    config = _through_the_wire(
        LauncherConfig(executor_configs={"async_loky": {"max_workers": 4}})
    )
    kwargs = config.executor_kwargs("async_loky", **RUNTIME)
    assert kwargs["max_workers"] == 4
    assert kwargs["gpu_selector"] == "ZE_AFFINITY_MASK"
    assert kwargs["return_stdout"] is False
    assert kwargs["logger"] == "a-logger"
    assert "timeout" not in kwargs


def test_explicit_none_is_still_an_override():
    """Set-to-None and unset have to stay distinguishable."""
    config = _through_the_wire(
        LauncherConfig(executor_configs={"async_loky": {"max_workers": None}})
    )
    assert config.executor_kwargs("async_loky", **RUNTIME)["max_workers"] is None


def test_nested_model_stays_a_model():
    """`overrides()` must not flatten `MPIConfig` into a dict.

    `AsyncMPIExecutor` reads attributes off it (`nprocesses_flag` and friends) when building a
    launch command, so a dict here is an `AttributeError` at the first submit in a worker.
    """
    config = _through_the_wire(
        LauncherConfig(
            executor_configs={"async_mpi": {"mpi_config": MPIConfig(flavor="openmpi")}}
        )
    )
    mpi = config.executor_configs["async_mpi"]
    assert isinstance(mpi, AsyncMPIConfig)
    assert isinstance(mpi.mpi_config, MPIConfig)
    assert mpi.mpi_config.launcher == "mpirun"
    assert mpi.mpi_config.nprocesses_flag == "-np"
    assert isinstance(
        config.executor_kwargs("async_mpi", **RUNTIME)["mpi_config"], MPIConfig
    )


def test_serialization_is_idempotent():
    """Two hops must produce the same bytes as one -- children launch children."""
    config = LauncherConfig(
        executor_configs={"async_loky": {"timeout": 60.0, "max_workers": 4}}
    )
    once = _through_the_wire(config)
    twice = _through_the_wire(once)
    assert once.model_dump_json() == config.model_dump_json()
    assert twice.model_dump_json() == config.model_dump_json()


# ------------------------------------------------------------------ #
#  Precedence and tolerance                                            #
# ------------------------------------------------------------------ #


def test_config_beats_the_top_level_scalar():
    """A per-executor `gpu_selector` overrides the shared one."""
    config = _through_the_wire(
        LauncherConfig(
            gpu_selector="ZE_AFFINITY_MASK",
            executor_configs={"async_loky": {"gpu_selector": "CUDA_VISIBLE_DEVICES"}},
        )
    )
    kwargs = config.executor_kwargs("async_loky", gpu_selector=config.gpu_selector)
    assert kwargs["gpu_selector"] == "CUDA_VISIBLE_DEVICES"


def test_unknown_executor_name_falls_back_to_the_base():
    """An externally-registered executor is configurable without a config class."""
    config = _through_the_wire(
        LauncherConfig(executor_configs={"some_plugin": {"whatever": 7}})
    )
    entry = config.executor_configs["some_plugin"]
    assert type(entry) is ExecutorConfig
    assert config.executor_kwargs("some_plugin", **RUNTIME)["whatever"] == 7


def test_extra_fields_are_carried_through():
    """`extra="allow"` reaches a constructor argument with no field declared for it."""
    config = _through_the_wire(
        LauncherConfig(executor_configs={"async_loky": {"not_a_field": "x"}})
    )
    entry = config.executor_configs["async_loky"]
    assert isinstance(entry, AsyncLokyConfig)
    assert entry.model_extra == {"not_a_field": "x"}
    assert config.executor_kwargs("async_loky", **RUNTIME)["not_a_field"] == "x"


def test_rejects_a_value_that_is_neither_mapping_nor_config():
    with pytest.raises(Exception):
        LauncherConfig(executor_configs={"async_loky": 5})


# ------------------------------------------------------------------ #
#  Interaction with how the master forwards its config                 #
# ------------------------------------------------------------------ #


def test_model_copy_preserves_executor_configs():
    """`AsyncMaster` hands each child a `model_copy(update=...)` of its own config."""
    config = LauncherConfig(executor_configs={"async_loky": {"max_workers": 4}})
    child = config.model_copy(update={"task_executor_name": ["async_loky"]})
    revived = _through_the_wire(child)
    assert revived.task_executor_name == ["async_loky"]
    assert revived.executor_kwargs("async_loky", **RUNTIME)["max_workers"] == 4


def test_cluster_secret_survives():
    """The model serializer is on `ExecutorConfig`, not `LauncherConfig`.

    `cluster_secret` is set in `model_post_init` via `object.__setattr__`, so it is absent
    from `model_fields_set`. Filtering unset fields at the `LauncherConfig` level would drop
    it and break cluster mode.
    """
    config = LauncherConfig(
        cluster=True, executor_configs={"async_loky": {"max_workers": 4}}
    )
    assert config.cluster_secret is not None
    assert _through_the_wire(config).cluster_secret == config.cluster_secret
