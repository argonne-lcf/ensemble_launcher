"""Per-executor configuration models.

Executor constructor arguments come from two places that have to be kept apart.  Some are
derived at runtime by the orchestrator node doing the constructing -- the logger, the number
of cores in the node's own allocation, the ``(host, cpu) -> rank`` map an MPI launch needs --
and cannot be written down in advance.  The rest are choices a user makes, and until now there
was nowhere to put them: ``LauncherConfig`` carried a handful of top-level scalars
(``gpu_selector``, ``return_stdout``) which every executor shared, and anything more specific
had to be hardcoded at the construction site.

The models here cover the second kind, one subclass per registered executor, keyed by the name
the executor is registered under.  :meth:`LauncherConfig.executor_kwargs` merges them over the
runtime-derived arguments, so a config entry overrides a default without the orchestrator having
to know which executor cares about which argument.

This module deliberately imports nothing from ``ensemble_launcher.executors``: the executors
import *this* package (e.g. ``AsyncMPIExecutor`` needs :class:`MPIConfig`), so the name-to-class
table has to live on the config side to keep the two packages acyclic.
"""

from typing import Any, Dict, List, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, model_serializer

from .mpi_config import MPIConfig


class ExecutorConfig(BaseModel):
    """Base for every executor's configuration.

    Every field defaults to ``None`` and only the fields actually set by the caller are
    reported by :meth:`overrides`, so a config says "change this one thing" rather than
    restating an executor's whole signature.  ``extra="allow"`` lets a config reach a
    constructor argument that has no field here yet -- all the executors end their
    ``__init__`` in ``**kwargs``, so an unknown-but-valid argument still lands.
    """

    model_config = ConfigDict(extra="allow")

    gpu_selector: Optional[str] = None
    """Environment variable the executor sets to pin a task to its GPUs"""
    return_stdout: Optional[bool] = None
    """Whether a command task's stdout is captured and returned with the result"""

    @model_serializer(mode="wrap")
    def _serialize_only_set(self, handler) -> Dict[str, Any]:
        """Serialize the explicitly-set fields only.

        The config reaches a worker as JSON (``LauncherConfig.model_dump_json`` in the
        orchestrator's ``asdict``, then a temp file, then ``model_validate_json`` in the child).
        A plain dump writes every unset field as an explicit ``null``, so on the far side
        ``model_fields_set`` contains *everything* and :meth:`overrides` inverts into "clobber
        every runtime argument with ``None``".  Dropping unset fields here keeps the set/unset
        distinction across the wire, and still lets a caller pass an explicit ``None``.
        """
        return {
            key: value
            for key, value in handler(self).items()
            if key in self.model_fields_set
        }

    def overrides(self) -> Dict[str, Any]:
        """The explicitly-set fields, as keyword arguments for the executor's constructor.

        Reads the attributes rather than ``model_dump`` so nested models stay models: the MPI
        executors call methods on :class:`MPIConfig`, and a plain dict there is an
        ``AttributeError`` at the first submit inside a worker.  Iterating ``model_fields_set``
        also picks up any ``extra="allow"`` field for free.
        """
        return {name: getattr(self, name) for name in self.model_fields_set}


class PoolExecutorConfig(ExecutorConfig):
    """Shared by the executors backed by a worker pool.

    ``max_workers`` is not on :class:`ExecutorConfig` because the two MPI executors ignore it --
    they size each launch from the task's own resource request.
    """

    max_workers: Optional[int] = None
    """Workers in the pool.  Defaults, per executor, to something derived from the allocation."""


class AsyncProcessPoolConfig(PoolExecutorConfig):
    """Configuration for ``async_processpool`` (:class:`AsyncProcessPoolExecutor`)"""

    worker_method: Optional[str] = None
    """``"spawn"`` to build the pool on a spawn context, anything else for the default"""


class AsyncThreadPoolConfig(PoolExecutorConfig):
    """Configuration for ``async_threadpool`` (:class:`AsyncThreadPoolExecutor`)"""


class AsyncLokyConfig(PoolExecutorConfig):
    """Configuration for ``async_loky`` (:class:`AsyncLokyExecutor`)"""

    timeout: Optional[float] = None
    """Seconds an idle worker survives before loky reaps it"""


class AsyncPinnedConfig(PoolExecutorConfig):
    """Configuration for ``async_pinned`` (:class:`AsyncPinnedExecutor`)"""

    gpus: Optional[List[Union[int, str]]] = None
    """Devices to build one pinned pool each for.

    Defaults to the ids of the worker's own node allocation. Set it to restrict the executor
    to a subset, or to spell ids the allocation does not describe (Aurora's tiles, say).

    Spell the ids the way the system does, as ``SystemConfig.gpus`` has them: polaris
    describes its devices as ``[0, 1, 2, 3]`` and aurora as ``["0", ..., "11"]``. A grant
    carries the system's spelling and the executor matches it exactly, so aurora's devices
    written as ints would match no pool.
    """
    timeout: Optional[float] = None
    """Seconds an idle worker survives before loky reaps it"""


class AsyncMPIConfig(ExecutorConfig):
    """Configuration for ``async_mpi`` (:class:`AsyncMPIExecutor`)"""

    tmp_dir: Optional[str] = None
    """Directory for the hostfiles and rankfiles the executor writes per launch"""
    mpi_config: Optional[MPIConfig] = None
    """Launcher flavour and flag spellings"""


class AsyncMPIPoolConfig(ExecutorConfig):
    """Configuration for ``async_mpi_processpool`` (:class:`AsyncMPIPoolExecutor`)"""

    mpi_config: Optional[MPIConfig] = None
    """Launcher flavour and flag spellings"""


_EXECUTOR_CONFIGS: Dict[str, Type[ExecutorConfig]] = {}


def register_executor_config(name: str, config_class: Type[ExecutorConfig]) -> None:
    """Associate an executor's registered name with its configuration class"""
    _EXECUTOR_CONFIGS[name] = config_class


def get_executor_config_class(name: str) -> Type[ExecutorConfig]:
    """The configuration class for an executor name.

    Falls back to the base :class:`ExecutorConfig`, whose ``extra="allow"`` still carries
    arbitrary keyword arguments through to the constructor.  An executor registered from an
    external module therefore needs no config class of its own to be configurable.
    """
    return _EXECUTOR_CONFIGS.get(name, ExecutorConfig)


register_executor_config("async_processpool", AsyncProcessPoolConfig)
register_executor_config("async_threadpool", AsyncThreadPoolConfig)
register_executor_config("async_loky", AsyncLokyConfig)
register_executor_config("async_pinned", AsyncPinnedConfig)
register_executor_config("async_mpi", AsyncMPIConfig)
register_executor_config("async_mpi_processpool", AsyncMPIPoolConfig)


def coerce_executor_config(name: str, value: Any) -> ExecutorConfig:
    """Build the right :class:`ExecutorConfig` subclass for an executor name.

    Accepts a mapping, or an already-built config -- including one built as the wrong class,
    which happens when a :class:`LauncherConfig` is revalidated after a round trip through a
    base-class annotation.  Re-validating from :meth:`ExecutorConfig.overrides` preserves the
    set/unset distinction; ``model_dump`` would not.
    """
    config_class = get_executor_config_class(name)
    if isinstance(value, ExecutorConfig):
        if type(value) is config_class:
            return value
        return config_class(**value.overrides())
    if isinstance(value, dict):
        return config_class(**value)
    raise TypeError(
        f"executor_configs[{name!r}] must be a mapping or an ExecutorConfig, "
        f"got {type(value).__name__}"
    )


__all__ = [
    "ExecutorConfig",
    "PoolExecutorConfig",
    "AsyncProcessPoolConfig",
    "AsyncThreadPoolConfig",
    "AsyncLokyConfig",
    "AsyncPinnedConfig",
    "AsyncMPIConfig",
    "AsyncMPIPoolConfig",
    "register_executor_config",
    "get_executor_config_class",
    "coerce_executor_config",
]
