from pydantic import BaseModel, Field
import multiprocessing as mp
from typing import Literal, List, Union, Optional
from difflib import get_close_matches
import logging

class SystemConfig(BaseModel):
    """Input configuration of the system"""
    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0
    cpus: List[int] = Field(default_factory=list)
    gpus: List[Union[str, int]] = Field(default_factory=list)

class LauncherConfig(BaseModel):
    """Configuration for launcher"""
    child_executor_name: str = "multiprocessing"
    task_executor_name: str = "multiprocessing"
    comm_name: Literal["multiprocessing","zmq","dragon","async_zmq"] = "multiprocessing"
    report_interval: float = 10.0
    nlevels: int = 1
    return_stdout: bool = False
    worker_logs: bool = False
    master_logs: bool = False
    nchildren: Optional[int] = None ##Setting this will fix the number of children at each level
    sequential_child_launch: bool = False ##If True, launch children one by one even for MPI executor
    profile: Optional[Literal["perfetto"]] = None ##Enable profiling with event registry and Perfetto export for timeline visualization
    gpu_selector: str = "ZE_AFFINITY_MASK"
    log_level: int = logging.INFO
    use_mpi_ppn: bool = True ##If True, use -ppn flag when launching MPI jobs
    worker_scheduler_policy: str = "greedy_worker_policy" ##Policy to use for worker scheduler

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the config"""
        lines = [f"{self.__class__.__name__}:"]
        for field_name, field_value in self.__dict__.items():
            lines.append(f"  {field_name}: {field_value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation"""
        return self.__str__()