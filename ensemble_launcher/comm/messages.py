from dataclasses import dataclass, field
from typing import Any, List, Optional
from datetime import datetime
from ensemble_launcher.ensemble import Task
import enum
import json



class ActionType(enum.Enum):
    START = "start"
    WAIT = "wait"
    STOP = "stop"

##Base message class
@dataclass
class Message:
    sender:str = None
    receiver:str = None
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None

@dataclass    
class Status(Message):
    nrunning_tasks: int = 0
    nfailed_tasks: int = 0
    nsuccessful_tasks: int = 0
    nfree_cores: int = 0
    nfree_gpus: int = 0

    def __add__(self, other: Any) -> "Status":
        if not isinstance(other, Status):
            raise TypeError(f"Cannot add Status and {type(other)}")  # Should raise, not return
        return Status(
            sender=self.sender,
            receiver=self.receiver,
            nrunning_tasks=self.nrunning_tasks + other.nrunning_tasks,
            nfailed_tasks=self.nfailed_tasks + other.nfailed_tasks,
            nsuccessful_tasks=self.nsuccessful_tasks + other.nsuccessful_tasks,
            nfree_cores=self.nfree_cores + other.nfree_cores,
            nfree_gpus=self.nfree_gpus + other.nfree_gpus,
        )

    __radd__ = __add__

    def to_file(self,fname:str):
        with open(fname, 'w') as f:
            json.dump({
                'sender': self.sender,
                'receiver': self.receiver,
                'timestamp': self.timestamp.isoformat() if self.timestamp else None,
                'message_id': self.message_id,
                'nrunning_tasks': self.nrunning_tasks,
                'nfailed_tasks': self.nfailed_tasks,
                'nsuccessful_tasks': self.nsuccessful_tasks,
                'nfree_cores': self.nfree_cores,
                'nfree_gpus': self.nfree_gpus
            }, f, indent=2)

@dataclass
class Result(Message):
    data: Any = None
    task_id: str = None
    success: bool = True
    exception: Optional[str] = None

@dataclass
class TaskUpdate(Message):
    added_tasks: List[Task] = field(default_factory=list)  # Should be callable
    deleted_tasks: List[Task] = field(default_factory=list)  # Should be callable

@dataclass
class HeartBeat(Message):
    alive: bool = True

@dataclass
class Action(Message):
    type: Optional[ActionType] = None

@dataclass
class TaskRequest(Message):
    ntasks: int = 0