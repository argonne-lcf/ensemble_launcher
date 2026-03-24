import enum
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, List, Optional

from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import JobResource


class StopType(enum.Enum):
    TERMINATE = "terminate"
    KILL = "kill"


##Base message class
@dataclass
class Message:
    sender: str = None
    receiver: str = None
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_id": self.message_id,
        }


@dataclass
class Status(Message):
    nrunning_tasks: int = 0
    nfailed_tasks: int = 0
    nsuccessful_tasks: int = 0
    nfree_cores: int = 0
    nfree_gpus: int = 0
    nremaining_tasks: int = 0
    ##performance metrics
    task_throughput: float = 0.0
    tag: str = ""

    def __add__(self, other: Any) -> "Status":
        if not isinstance(other, Status):
            raise TypeError(
                f"Cannot add Status and {type(other)}"
            )  # Should raise, not return
        return Status(
            sender=self.sender,
            receiver=self.receiver,
            nrunning_tasks=self.nrunning_tasks + other.nrunning_tasks,
            nfailed_tasks=self.nfailed_tasks + other.nfailed_tasks,
            nsuccessful_tasks=self.nsuccessful_tasks + other.nsuccessful_tasks,
            nfree_cores=self.nfree_cores + other.nfree_cores,
            nfree_gpus=self.nfree_gpus + other.nfree_gpus,
            nremaining_tasks=self.nremaining_tasks + other.nremaining_tasks,
            task_throughput=self.task_throughput + other.task_throughput,
        )

    __radd__ = __add__

    def to_file(self, fname: str):
        with open(fname, "w") as f:
            json.dump(
                {
                    "sender": self.sender,
                    "receiver": self.receiver,
                    "timestamp": self.timestamp.isoformat() if self.timestamp else None,
                    "message_id": self.message_id,
                    "nrunning_tasks": self.nrunning_tasks,
                    "nfailed_tasks": self.nfailed_tasks,
                    "nsuccessful_tasks": self.nsuccessful_tasks,
                    "nfree_cores": self.nfree_cores,
                    "nfree_gpus": self.nfree_gpus,
                },
                f,
                indent=2,
            )


@dataclass
class Result(Message):
    data: Any = None
    task_id: str = None
    success: bool = True
    exception: Optional[str] = None

    def to_dict(self):
        ret_dict = super().to_dict()
        ret_dict.update(
            {
                "data": self.data,
                "task_id": self.task_id,
                "success": self.success,
                "exception": self.exception,
            }
        )
        return ret_dict


@dataclass
class ResultBatch(Message):
    data: List[Result] = field(default_factory=list)

    def add_result(self, result: Result):
        self.data.append(result)

    def to_dict(self):
        return {r.task_id: r.to_dict() for r in self.data}

    def __add__(self, other) -> "ResultBatch":
        if not isinstance(other, ResultBatch):
            raise TypeError(
                f"Cannot add ResultBatch and {type(other)}"
            )  # Should raise, not return
        return ResultBatch(
            sender=self.sender, receiver=self.receiver, data=self.data + other.data
        )

    def __radd__(self, other) -> "ResultBatch":
        return self.__add__(other)


# I got lazy and didn't want to change the work done logic in the orchestrator.
@dataclass
class IResultBatch(Message):
    data: List[Result] = field(default_factory=list)

    def add_result(self, result: Result):
        self.data.append(result)

    def to_dict(self):
        return {r.task_id: r.to_dict() for r in self.data}

    def __add__(self, other) -> "ResultBatch":
        if not isinstance(other, ResultBatch):
            raise TypeError(
                f"Cannot add ResultBatch and {type(other)}"
            )  # Should raise, not return
        return ResultBatch(
            sender=self.sender, receiver=self.receiver, data=self.data + other.data
        )

    def __radd__(self, other) -> "ResultBatch":
        return self.__add__(other)


@dataclass
class TaskUpdate(Message):
    added_tasks: List[Task] = field(default_factory=list)  # Should be callable
    deleted_tasks: List[Task] = field(default_factory=list)  # Should be callable


@dataclass
class NodeUpdate(Message):
    nodes: Optional[JobResource] = None


@dataclass
class ResultAck(Message):
    pass


@dataclass
class Ready(Message):
    pass


@dataclass
class Stop(Message):
    type: Optional[StopType] = None


@dataclass
class TaskRequest(Message):
    ntasks: int = 0
    free_resources: Optional[JobResource] = None


@dataclass
class NodeRequest(Message):
    pass


all_messages = [
    Message,
    Status,
    Result,
    ResultBatch,
    TaskUpdate,
    NodeUpdate,
    ResultAck,
    Ready,
    Stop,
    TaskRequest,
    NodeRequest,
    IResultBatch,
]
