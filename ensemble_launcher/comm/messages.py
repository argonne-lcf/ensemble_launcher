from dataclasses import dataclass
from typing import Any

@dataclass
class Status:
    nrunning_tasks: int
    nready_tasks: int
    nfailed_tasks: int
    nfinished_tasks: int
    nfree_cores: int
    nfree_gpus: int


class Result:
    def __init__(self, data: Any):
        self.data = data