import enum
import json
import struct
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

import cloudpickle

from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import JobResource


class StopType(enum.Enum):
    TERMINATE = "terminate"
    KILL = "kill"


_MSG_REGISTRY: Dict[int, Type["Message"]] = {}


@dataclass
class Message:
    MSG_TYPE_ID: int = field(default=0, init=False, repr=False)

    sender: str = None
    receiver: str = None
    timestamp: datetime = field(default_factory=datetime.now)
    message_id: Optional[str] = None

    _packed: bool = field(default=False, init=False, repr=False)
    _raw_deep: Optional[bytes] = field(default=None, init=False, repr=False)

    def to_dict(self):
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "message_id": self.message_id,
        }

    # ------------------------------------------------------------------ #
    #  pack / unpack                                                      #
    # ------------------------------------------------------------------ #

    @property
    def is_packed(self) -> bool:
        return self._packed

    def _check_unpacked(self, field_name: str) -> None:
        if self._packed:
            raise RuntimeError(
                f"{type(self).__name__}.{field_name} not accessible in packed state "
                "— call unpack() first"
            )

    def pack(self):
        if self._packed:
            return self
        self._packed = True
        return self

    def unpack(self) -> None:
        if not self._packed:
            return
        self._packed = False

    # ------------------------------------------------------------------ #
    #  to_byte_array / from_byte_array  (zero-copy fast path)            #
    # ------------------------------------------------------------------ #

    def to_byte_array(self) -> list:
        if not self._packed:
            self.pack()
        shallow = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_packed", "_raw_deep", "MSG_TYPE_ID")
        }
        shallow_blob = cloudpickle.dumps(shallow)
        header = struct.pack("!HI", self.MSG_TYPE_ID, len(shallow_blob))
        if self._raw_deep is not None:
            return [header, shallow_blob, self._raw_deep]
        return [header, shallow_blob]

    @classmethod
    def from_byte_array(cls, frames: list) -> "Message":
        header = frames[0]
        (type_id,) = struct.unpack_from("!H", header, 0)
        msg_cls = _MSG_REGISTRY.get(type_id)
        if msg_cls is None:
            raise ValueError(f"Unknown message type ID: {type_id}")
        if msg_cls is not cls and msg_cls.from_byte_array is not Message.from_byte_array:
            return msg_cls.from_byte_array(frames)
        shallow_blob = frames[1]
        shallow_dict = cloudpickle.loads(shallow_blob)
        deep_blob = frames[2] if len(frames) > 2 else None
        msg = msg_cls.__new__(msg_cls)
        msg.__dict__.update(shallow_dict)
        msg._packed = True
        msg._raw_deep = deep_blob
        msg.MSG_TYPE_ID = type_id
        return msg

    # ------------------------------------------------------------------ #
    #  to_bytes / from_bytes  (compat wrappers)                          #
    # ------------------------------------------------------------------ #

    def to_bytes(self) -> bytes:
        return b"".join(self.to_byte_array())

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        (type_id,) = struct.unpack_from("!H", data, 0)
        msg_cls = _MSG_REGISTRY.get(type_id)
        if msg_cls is None:
            raise ValueError(f"Unknown message type ID: {type_id}")
        if msg_cls is not cls and msg_cls.from_bytes is not Message.from_bytes:
            return msg_cls.from_bytes(data)
        (shallow_len,) = struct.unpack_from("!I", data, 2)
        shallow_dict = cloudpickle.loads(data[6 : 6 + shallow_len])
        mv = memoryview(data)
        deep_blob = mv[6 + shallow_len :]
        msg = msg_cls.__new__(msg_cls)
        msg.__dict__.update(shallow_dict)
        msg._packed = True
        msg._raw_deep = deep_blob if len(deep_blob) > 0 else None
        msg.MSG_TYPE_ID = type_id
        return msg


@dataclass
class Status(Message):
    MSG_TYPE_ID: int = field(default=1, init=False, repr=False)

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
    MSG_TYPE_ID: int = field(default=2, init=False, repr=False)

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

    def pack(self):
        if self._packed:
            return self
        self._raw_deep = cloudpickle.dumps({"data": self.data})
        self.data = None
        self._packed = True
        return self

    def unpack(self) -> None:
        if not self._packed:
            return
        deep = cloudpickle.loads(self._raw_deep)
        self.data = deep["data"]
        self._raw_deep = None
        self._packed = False


@dataclass
class ResultBatch(Message):
    MSG_TYPE_ID: int = field(default=3, init=False, repr=False)

    data: List[Result] = field(default_factory=list)

    def add_result(self, result: Result):
        self.data.append(result)

    def to_dict(self):
        return {r.task_id: r.to_dict() for r in self.data}

    def __add__(self, other) -> "ResultBatch":
        if not isinstance(other, ResultBatch):
            raise TypeError(
                f"Cannot add ResultBatch and {type(other)}"
            )
        packed = self._packed or other._packed
        if packed:
            if not self._packed:
                self.pack()
            if not other._packed:
                other.pack()
        new = ResultBatch(
            sender=self.sender, receiver=self.receiver, data=self.data + other.data
        )
        new._packed = packed
        return new

    def __radd__(self, other) -> "ResultBatch":
        return self.__add__(other)

    def pack(self):
        if self._packed:
            return self
        for r in self.data:
            r.pack()
        self._packed = True
        return self

    def unpack(self) -> None:
        if not self._packed:
            return
        for r in self.data:
            r.unpack()
        self._packed = False

    def to_byte_array(self) -> list:
        if not self._packed:
            self.pack()
        items = []
        deep_parts = []
        offset = 0
        for r in self.data:
            deep = r._raw_deep or b""
            start = offset
            end = offset + len(deep)
            offset = end
            inner_shallow = {
                k: v
                for k, v in r.__dict__.items()
                if k not in ("_packed", "_raw_deep", "MSG_TYPE_ID")
            }
            items.append((start, end, inner_shallow))
            deep_parts.append(deep)
        container_shallow = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_packed", "_raw_deep", "MSG_TYPE_ID", "data")
        }
        container_shallow["items"] = items
        shallow_blob = cloudpickle.dumps(container_shallow)
        header = struct.pack("!HI", self.MSG_TYPE_ID, len(shallow_blob))
        return [header, shallow_blob, b"".join(deep_parts)]

    @classmethod
    def _rebuild_results(cls, items: list, deep_mv) -> List[Result]:
        results = []
        for start, end, inner_shallow in items:
            r = Result.__new__(Result)
            r.__dict__.update(inner_shallow)
            r._packed = True
            r._raw_deep = (
                deep_mv[start:end] if deep_mv is not None and start < end else None
            )
            r.MSG_TYPE_ID = 2
            results.append(r)
        return results

    @classmethod
    def from_byte_array(cls, frames: list) -> "ResultBatch":
        header = frames[0]
        (type_id,) = struct.unpack_from("!H", header, 0)
        container_shallow = cloudpickle.loads(frames[1])
        items = container_shallow.pop("items")
        deep_mv = memoryview(frames[2]) if len(frames) > 2 else None
        msg = cls.__new__(cls)
        msg.__dict__.update(container_shallow)
        msg.MSG_TYPE_ID = type_id
        msg.data = cls._rebuild_results(items, deep_mv)
        msg._packed = True
        msg._raw_deep = None
        return msg

    def to_bytes(self) -> bytes:
        return b"".join(self.to_byte_array())

    @classmethod
    def from_bytes(cls, data: bytes) -> "ResultBatch":
        (type_id,) = struct.unpack_from("!H", data, 0)
        (shallow_len,) = struct.unpack_from("!I", data, 2)
        container_shallow = cloudpickle.loads(data[6 : 6 + shallow_len])
        items = container_shallow.pop("items")
        deep_mv = memoryview(data)[6 + shallow_len :]
        msg = cls.__new__(cls)
        msg.__dict__.update(container_shallow)
        msg.MSG_TYPE_ID = type_id
        msg.data = cls._rebuild_results(items, deep_mv)
        msg._packed = True
        msg._raw_deep = None
        return msg


# I got lazy and didn't want to change the work done logic in the orchestrator.
@dataclass
class IResultBatch(Message):
    MSG_TYPE_ID: int = field(default=4, init=False, repr=False)

    data: List[Result] = field(default_factory=list)

    def add_result(self, result: Result):
        self.data.append(result)

    def to_dict(self):
        return {r.task_id: r.to_dict() for r in self.data}

    def __add__(self, other) -> "ResultBatch":
        if not isinstance(other, ResultBatch):
            raise TypeError(
                f"Cannot add ResultBatch and {type(other)}"
            )
        packed = self._packed or other._packed
        if packed:
            if not self._packed:
                self.pack()
            if not other._packed:
                other.pack()
        new = ResultBatch(
            sender=self.sender, receiver=self.receiver, data=self.data + other.data
        )
        new._packed = packed
        return new

    def __radd__(self, other) -> "ResultBatch":
        return self.__add__(other)

    def pack(self):
        if self._packed:
            return self
        for r in self.data:
            r.pack()
        self._packed = True
        return self

    def unpack(self) -> None:
        if not self._packed:
            return
        for r in self.data:
            r.unpack()
        self._packed = False

    def to_byte_array(self) -> list:
        if not self._packed:
            self.pack()
        items = []
        deep_parts = []
        offset = 0
        for r in self.data:
            deep = r._raw_deep or b""
            start = offset
            end = offset + len(deep)
            offset = end
            inner_shallow = {
                k: v
                for k, v in r.__dict__.items()
                if k not in ("_packed", "_raw_deep", "MSG_TYPE_ID")
            }
            items.append((start, end, inner_shallow))
            deep_parts.append(deep)
        container_shallow = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("_packed", "_raw_deep", "MSG_TYPE_ID", "data")
        }
        container_shallow["items"] = items
        shallow_blob = cloudpickle.dumps(container_shallow)
        header = struct.pack("!HI", self.MSG_TYPE_ID, len(shallow_blob))
        return [header, shallow_blob, b"".join(deep_parts)]

    @classmethod
    def from_byte_array(cls, frames: list) -> "IResultBatch":
        header = frames[0]
        (type_id,) = struct.unpack_from("!H", header, 0)
        container_shallow = cloudpickle.loads(frames[1])
        items = container_shallow.pop("items")
        deep_mv = memoryview(frames[2]) if len(frames) > 2 else None
        msg = cls.__new__(cls)
        msg.__dict__.update(container_shallow)
        msg.MSG_TYPE_ID = type_id
        msg.data = ResultBatch._rebuild_results(items, deep_mv)
        msg._packed = True
        msg._raw_deep = None
        return msg

    def to_bytes(self) -> bytes:
        return b"".join(self.to_byte_array())

    @classmethod
    def from_bytes(cls, data: bytes) -> "IResultBatch":
        (type_id,) = struct.unpack_from("!H", data, 0)
        (shallow_len,) = struct.unpack_from("!I", data, 2)
        container_shallow = cloudpickle.loads(data[6 : 6 + shallow_len])
        items = container_shallow.pop("items")
        deep_mv = memoryview(data)[6 + shallow_len :]
        msg = cls.__new__(cls)
        msg.__dict__.update(container_shallow)
        msg.MSG_TYPE_ID = type_id
        msg.data = ResultBatch._rebuild_results(items, deep_mv)
        msg._packed = True
        msg._raw_deep = None
        return msg


@dataclass
class TaskUpdate(Message):
    MSG_TYPE_ID: int = field(default=5, init=False, repr=False)

    added_tasks: List[Task] = field(default_factory=list)
    deleted_tasks: List[Task] = field(default_factory=list)

    def pack(self):
        if self._packed:
            return self
        for t in self.added_tasks:
            t.pack()
        for t in self.deleted_tasks:
            t.pack()
        self._packed = True
        return self

    def unpack(self) -> None:
        if not self._packed:
            return
        for t in self.added_tasks:
            t.unpack()
        for t in self.deleted_tasks:
            t.unpack()
        self._packed = False

    @staticmethod
    def _pack_task_list(tasks, deep_parts, offset):
        items = []
        for t in tasks:
            deep = t._raw_deep or b""
            start = offset
            end = offset + len(deep)
            offset = end
            items.append((start, end, t.__dict__.copy()))
            deep_parts.append(deep)
        return items, offset

    @staticmethod
    def _rebuild_tasks(items, deep_mv) -> List[Task]:
        tasks = []
        for start, end, inner_dict in items:
            t = Task.__new__(Task)
            object.__setattr__(t, "__dict__", inner_dict)
            object.__setattr__(t, "__pydantic_fields_set__", set())
            raw_deep = (
                deep_mv[start:end] if deep_mv is not None and start < end else None
            )
            t.__pydantic_private__ = {"_packed": True, "_raw_deep": raw_deep}
            tasks.append(t)
        return tasks

    def to_byte_array(self) -> list:
        if not self._packed:
            self.pack()
        deep_parts = []
        offset = 0
        added_items, offset = self._pack_task_list(
            self.added_tasks, deep_parts, offset
        )
        deleted_items, offset = self._pack_task_list(
            self.deleted_tasks, deep_parts, offset
        )
        container_shallow = {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in (
                "_packed",
                "_raw_deep",
                "MSG_TYPE_ID",
                "added_tasks",
                "deleted_tasks",
            )
        }
        container_shallow["added"] = added_items
        container_shallow["deleted"] = deleted_items
        shallow_blob = cloudpickle.dumps(container_shallow)
        header = struct.pack("!HI", self.MSG_TYPE_ID, len(shallow_blob))
        return [header, shallow_blob, b"".join(deep_parts)]

    @classmethod
    def from_byte_array(cls, frames: list) -> "TaskUpdate":
        header = frames[0]
        (type_id,) = struct.unpack_from("!H", header, 0)
        container_shallow = cloudpickle.loads(frames[1])
        added_items = container_shallow.pop("added")
        deleted_items = container_shallow.pop("deleted")
        deep_mv = memoryview(frames[2]) if len(frames) > 2 else None
        msg = cls.__new__(cls)
        msg.__dict__.update(container_shallow)
        msg.MSG_TYPE_ID = type_id
        msg.added_tasks = cls._rebuild_tasks(added_items, deep_mv)
        msg.deleted_tasks = cls._rebuild_tasks(deleted_items, deep_mv)
        msg._packed = True
        msg._raw_deep = None
        return msg

    def to_bytes(self) -> bytes:
        return b"".join(self.to_byte_array())

    @classmethod
    def from_bytes(cls, data: bytes) -> "TaskUpdate":
        (type_id,) = struct.unpack_from("!H", data, 0)
        (shallow_len,) = struct.unpack_from("!I", data, 2)
        container_shallow = cloudpickle.loads(data[6 : 6 + shallow_len])
        added_items = container_shallow.pop("added")
        deleted_items = container_shallow.pop("deleted")
        deep_mv = memoryview(data)[6 + shallow_len :]
        msg = cls.__new__(cls)
        msg.__dict__.update(container_shallow)
        msg.MSG_TYPE_ID = type_id
        msg.added_tasks = cls._rebuild_tasks(added_items, deep_mv)
        msg.deleted_tasks = cls._rebuild_tasks(deleted_items, deep_mv)
        msg._packed = True
        msg._raw_deep = None
        return msg


@dataclass
class NodeUpdate(Message):
    MSG_TYPE_ID: int = field(default=6, init=False, repr=False)

    nodes: Optional[JobResource] = None


@dataclass
class ResultAck(Message):
    MSG_TYPE_ID: int = field(default=7, init=False, repr=False)


@dataclass
class Ready(Message):
    MSG_TYPE_ID: int = field(default=8, init=False, repr=False)


@dataclass
class Stop(Message):
    MSG_TYPE_ID: int = field(default=9, init=False, repr=False)

    type: Optional[StopType] = None


@dataclass
class TaskRequest(Message):
    MSG_TYPE_ID: int = field(default=10, init=False, repr=False)

    ntasks: int = 0
    free_resources: Optional[JobResource] = None


@dataclass
class NodeRequest(Message):
    MSG_TYPE_ID: int = field(default=11, init=False, repr=False)


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

_MSG_REGISTRY.update(
    {
        next(
            f.default
            for f in cls.__dataclass_fields__.values()
            if f.name == "MSG_TYPE_ID"
        ): cls
        for cls in all_messages
    }
)
