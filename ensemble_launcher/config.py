from pydantic import BaseModel, Field
import multiprocessing as mp
from typing import Literal, List, Union
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