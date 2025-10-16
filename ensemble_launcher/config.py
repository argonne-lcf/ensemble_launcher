from pydantic import BaseModel, Field
import multiprocessing as mp
from typing import Literal, List, Union, Optional
from difflib import get_close_matches


class SystemConfig(BaseModel):
    """Input configuration of the system"""
    name: str
    ncpus: int = mp.cpu_count()
    ngpus: int = 0
    cpus: List[int] = Field(default_factory=list)
    gpus: List[Union[str, int]] = Field(default_factory=list)

class LauncherConfig(BaseModel):
    """Configuration for launcher"""
    child_executor_name: Literal["multiprocessing","dragon","mpi"] = "multiprocessing"
    task_executor_name: Literal["multiprocessing","dragon","mpi"] = "multiprocessing"
    comm_name: Literal["multiprocessing","zmq","dragon"] = "multiprocessing"
    report_interval: float = 10.0
    nlevels: int = 1
    return_stdout: bool = False
    worker_logs: bool = False
    master_logs: bool = False
    nchildren: Optional[int] = None ##Setting this will fix the number of children at each level
    profile: Optional[Literal["basic","timeline"]] = None ##Setting this will print output some profiling information like communication latency, execution time of each task for every nodes
    gpu_selector: str = "ZE_AFFINITY_MASK"

    def __str__(self) -> str:
        """Return a nicely formatted string representation of the config"""
        lines = [f"{self.__class__.__name__}:"]
        for field_name, field_value in self.__dict__.items():
            lines.append(f"  {field_name}: {field_value}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        """Return a detailed string representation"""
        return self.__str__()