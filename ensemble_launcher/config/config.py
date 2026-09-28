import logging
import multiprocessing as mp
import secrets
from collections import Counter
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
)

from .executor_config import ExecutorConfig, coerce_executor_config
from .mpi_config import MPIConfig


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    nlevels: int = 1
    nchildren: int = 1
    leaf_nodes: int = 1
    strict_priority: bool = False  # If True, tasks are scheduled in strict priority order (no lower-priority task runs before a higher-priority one)


def _reject_duplicates(field_name: str, ids: list) -> list:
    dupes = sorted(
        {str(i) for i, n in Counter(ids).items() if n > 1}, key=str
    )
    if dupes:
        raise ValueError(
            f"SystemConfig.{field_name} contains duplicate ids: {dupes}. "
            "Repeating an id to oversubscribe a device is no longer supported here; "
            "express sharing per-task instead, e.g. Task(ngpus_per_process=0.25) "
            "to let 4 tasks share one GPU."
        )
    return ids


class SystemConfig(BaseModel):
    """Input configuration of the system"""

    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0
    cpus: List[int] = Field(default_factory=list)
    gpus: List[Union[str, int]] = Field(default_factory=list)

    @field_validator("cpus")
    @classmethod
    def _no_duplicate_cpus(cls, v: List[int]) -> List[int]:
        return _reject_duplicates("cpus", v)

    @field_validator("gpus")
    @classmethod
    def _no_duplicate_gpus(cls, v: List[Union[str, int]]) -> List[Union[str, int]]:
        return _reject_duplicates("gpus", v)


def get_system_config(name="aurora"):
    if name == "aurora":
        return SystemConfig(
            name="aurora",
            ncpus=102,
            ngpus=12,
            cpus=list(range(1, 52)) + list(range(53, 104)),
            gpus=list(map(str, range(12))),
        )
    elif name == "polaris":
        return SystemConfig(
            name="polaris",
            ncpus=32,
            ngpus=4,
            cpus=list(range(32)),
            gpus=list(range(4)),
        )
    else:
        raise NotImplementedError(f"unknown system {name}")


aurora_config = get_system_config("aurora")
polaris_config = get_system_config("polaris")


class LauncherConfig(BaseModel):
    """Configuration for launcher"""

    child_executor_name: str = "async_processpool"
    task_executor_name: Union[str, List[str]] = "async_processpool"
    comm_name: Literal["async_zmq"] = "async_zmq"
    report_interval: float = 10.0
    return_stdout: bool = False
    worker_logs: bool = False
    master_logs: bool = False
    sequential_child_launch: Optional[bool] = (
        None  ##If True, launch children one by one even for MPI executor
    )
    profile: Optional[Literal["perfetto"]] = (
        None  ##Enable profiling with event registry and Perfetto export for timeline visualization
    )
    gpu_selector: str = "ZE_AFFINITY_MASK"
    log_dir: str = "logs"  # Directory for log files
    log_level: int = logging.INFO
    children_scheduler_policy: str = (
        "simple_split_children_policy"  ##Policy to use for children scheduler
    )
    task_scheduler_policy: str = (
        "large_resource_policy"  ##Policy to use for children scheduler
    )
    policy_config: PolicyConfig = Field(default_factory=PolicyConfig)
    enable_workstealing: bool = (
        False  ##If True, master will listen for task requests from worker children
    )
    cluster: bool = False  # Eager result delivery + submit() API
    checkpoint_dir: Optional[str] = (
        None  # Directory for checkpoints; None disables checkpointing
    )
    heartbeat_interval: float = 1.0  # heart beat interval

    heartbeat_dead_threshold: float = (
        30.0  # Seconds before HB process declares a peer dead.
    )

    req_res: bool = True  # Enable ACK-based guaranteed delivery
    send_retries: int = 10  # Retry count on ACK timeout (req_res=True only)
    send_timeout: float = 1.0  # Per-attempt ACK timeout in seconds (req_res=True only)

    overload_orchestrator_core: bool = True  # Setting this to false reserves the first core of the head compute node for EL orchestrator

    restart_children_on_failure: bool = True

    result_buffer_size: int = (
        1000000  # max buffer size of the result queue in cluster mode
    )

    result_flush_interval: float = 5.0  # Flush result queues every fixed time

    task_buffer_size: int = 1000000  # max buffer size of the task queue per child

    cluster_secret: Optional[str] = None

    task_flush_interval: float = 5.0  # Flush task queues every fixed time

    task_request_size: Optional[int] = (
        None  # size of the task request in work stealing mode
    )

    task_request_interval: float = (
        0.5  # Seconds between periodic task requests in workstealing mode
    )

    mpi_config: MPIConfig = MPIConfig(
        flavor="mpich"
    )  ## Configuration to help build mpi options like -np, -ppn etc

    executor_configs: Dict[str, SerializeAsAny[ExecutorConfig]] = Field(
        default_factory=dict
    )
    """Per-executor constructor arguments, keyed by the executor's registered name.

    The arguments an executor is built with are otherwise either derived at runtime by the
    orchestrator node constructing it (its logger, its node's core count) or taken from the
    top-level scalars above, which every executor shares.  An entry here overrides both, so
    one executor can be configured without disturbing the others -- ``{"async_pinned":
    {"gpus": [0, 1, 2, 3]}}``.

    ``SerializeAsAny`` is required: annotated as the base class alone, pydantic would serialize
    to :class:`ExecutorConfig` and silently drop every subclass field on the way to a worker.
    """

    @field_validator("executor_configs", mode="before")
    @classmethod
    def _coerce_executor_configs(cls, value):
        """Build each entry as the subclass registered for its key.

        The dict key is the discriminator -- an executor name already identifies its
        configuration class -- so the entries need no tag field of their own.
        """
        if not isinstance(value, dict):
            return value
        return {
            name: coerce_executor_config(name, entry) for name, entry in value.items()
        }

    def executor_kwargs(self, name: str, **runtime: Any) -> Dict[str, Any]:
        """Constructor arguments for one executor.

        Args:
            name: The executor's registered name, e.g. ``"async_loky"``
            runtime: Arguments derived by the caller -- the logger, the node's core count,
                anything else that cannot be written into a config ahead of time
        Returns:
            ``runtime``, with this executor's configured fields merged over it.  With no
            entry for ``name`` this is ``runtime`` unchanged, so a config that sets no
            ``executor_configs`` behaves exactly as before.
        """
        kwargs = dict(runtime)
        config = self.executor_configs.get(name)
        if config is not None:
            kwargs.update(config.overrides())
        return kwargs

    def model_post_init(self, _) -> None:
        if self.cluster and self.cluster_secret is None:
            object.__setattr__(self, "cluster_secret", secrets.token_hex(16))

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the config"""
        lines = [f"{self.__class__.__name__}:"]
        for field_name, field_value in self.__dict__.items():
            lines.append(f"  {field_name}: {field_value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation"""
        return self.__str__()
