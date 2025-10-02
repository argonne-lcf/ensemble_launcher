from dataclasses import dataclass
from typing import Any

@dataclass
class Status:
    nrunning_tasks: int = 0
    nfailed_tasks: int = 0
    nsuccessful_tasks: int = 0
    nfree_cores: int = 0
    nfree_gpus: int = 0

    def __add__(self, other: Any) -> "Status":
        if not isinstance(other, Status):
            return NotImplementedError
        return Status(
            nrunning_tasks=self.nrunning_tasks + other.nrunning_tasks,
            nfailed_tasks=self.nfailed_tasks + other.nfailed_tasks,
            nsuccessful_tasks=self.nsuccessful_tasks + other.nsuccessful_tasks,
            nfree_cores=self.nfree_cores + other.nfree_cores,
            nfree_gpus=self.nfree_gpus + other.nfree_gpus,
        )

    __radd__ = __add__

class Result:
    def __init__(self, data: Any):
        self.data = data