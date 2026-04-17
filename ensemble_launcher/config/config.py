import logging
import multiprocessing as mp
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field

from .mpi_config import MPIConfig


class PolicyConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    nlevels: int = 1
    nchildren: int = 1
    leaf_nodes: int = 1
    strict_priority: bool = False  # If True, tasks are scheduled in strict priority order (no lower-priority task runs before a higher-priority one)


class SystemConfig(BaseModel):
    """Input configuration of the system"""

    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0
    cpus: List[int] = Field(default_factory=list)
    gpus: List[Union[str, int]] = Field(default_factory=list)


def get_system_config(name="aurora"):
    if name == "aurora":
        return SystemConfig(
            name="aurora",
            ncpus=102,
            ngpus=12,
            cpus=list(range(1, 52)) + list(range(53, 104)),
            gpus=list(map(str, range(12))),
        )
    else:
        raise NotImplementedError(f"unknown system {name}")


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

    overload_orchestrator_core: bool = True  # Setting this to false reserves the first core of the head compute node for EL orchestrator

    restart_children_on_failure: bool = True

    result_buffer_size: int = (
        1000000  # max buffer size of the result queue in cluster mode
    )

    result_flush_interval: float = 5.0  # Flush result queues every fixed time

    task_buffer_size: int = 1000000  # max buffer size of the task queue per child

    task_flush_interval: float = 5.0  # Flush task queues every fixed time

    task_request_size: Optional[int] = (
        None  # size of the task request in work stealing mode
    )

    task_request_interval: float = (
        5.0  # Seconds between periodic task requests in workstealing mode
    )

    mpi_config: MPIConfig = MPIConfig(
        flavor="mpich"
    )  ## Configuration to help build mpi options like -np, -ppn etc

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the config"""
        lines = [f"{self.__class__.__name__}:"]
        for field_name, field_value in self.__dict__.items():
            lines.append(f"  {field_name}: {field_value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation"""
        return self.__str__()
