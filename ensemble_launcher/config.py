import logging
import multiprocessing as mp
from difflib import get_close_matches
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, Field


class SystemConfig(BaseModel):
    """Input configuration of the system"""

    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0
    cpus: List[int] = Field(default_factory=list)
    gpus: List[Union[str, int]] = Field(default_factory=list)


class LauncherConfig(BaseModel):
    """Configuration for launcher"""

    child_executor_name: str = "async_processpool"
    task_executor_name: str = "async_processpool"
    comm_name: Literal["async_zmq"] = "async_zmq"
    report_interval: float = 10.0
    nlevels: int = 1
    return_stdout: bool = False
    worker_logs: bool = False
    master_logs: bool = False
    nchildren: Optional[int] = (
        None  ##Setting this will fix the number of children at each level
    )
    sequential_child_launch: bool = (
        False  ##If True, launch children one by one even for MPI executor
    )
    profile: Optional[Literal["perfetto"]] = (
        None  ##Enable profiling with event registry and Perfetto export for timeline visualization
    )
    gpu_selector: str = "ZE_AFFINITY_MASK"
    log_level: int = logging.INFO
    use_mpi_ppn: bool = True  ##If True, use -ppn flag when launching MPI jobs
    children_scheduler_policy: str = (
        "greedy_children_policy"  ##Policy to use for children scheduler
    )
    enable_workstealing: bool = (
        False  ##If True, master will listen for task requests from worker children
    )
    cpu_binding_option: str = "--cpu-bind"
    cluster: bool = False  # Eager result delivery + submit() API
    checkpoint_dir: Optional[str] = None  # Directory for checkpoints; None disables checkpointing
    dead_child_factor: float = 3.0  # Child is declared dead if status age > dead_child_factor * report_interval
    max_parent_send_failures: int = 3  # Consecutive send failures before declaring parent dead

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the config"""
        lines = [f"{self.__class__.__name__}:"]
        for field_name, field_value in self.__dict__.items():
            lines.append(f"  {field_name}: {field_value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation"""
        return self.__str__()
