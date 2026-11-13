import enum
import json
import pickle
import struct
from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union

import cloudpickle
import numpy as np

from ensemble_launcher.comm import blob
from ensemble_launcher.ensemble import Task, TaskMeta
from ensemble_launcher.ensemble.ensemble import (
    META_FIELDS,
    build_task,
    decode_deep,
    META_DTYPE,
    decode_meta_array,
    encode_deep,
    encode_meta_array,
    meta_row_values,
)
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
    nfree_gpus: Union[int, float] = 0
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
    uid: Optional[bytes] = None

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


# --------------------------------------------------------------------------- #
#  Blob-backed messages                                                       #
# --------------------------------------------------------------------------- #
#
# TaskUpdate and the result batches carry their items as columns (comm/blob.py) instead of one
# Python object per item. Frames on the wire:
#
#   [header] [container] [column 0] [column 1] ...
#
# ``header`` is ``!HI`` (type id, container length), as for every other message, and
# ``container`` pickles the message's own dataclass fields (sender, message_id, ...). Receiving
# one only wraps the frames: nothing is decoded per item until someone asks for it.

_SKIP_FIELDS = ("MSG_TYPE_ID", "_packed", "_raw_deep")


class _BlobMessage:
    """Shared frame layout for messages whose payload is a fixed list of columns."""

    _COLUMNS: tuple = ()  # attribute names, in frame order; "*_off" ones are int64 offsets

    def _container(self) -> dict:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in _SKIP_FIELDS
        }

    def _frames(self) -> list:
        out = []
        for name in self._COLUMNS:
            col = getattr(self, name)
            out.append(memoryview(col).cast("B") if isinstance(col, np.ndarray) else bytes(col))
        return out

    def to_byte_array(self) -> list:
        container = pickle.dumps(self._container(), protocol=pickle.HIGHEST_PROTOCOL)
        header = struct.pack("!HI", self.MSG_TYPE_ID, len(container))
        return [header, container] + self._frames()

    @classmethod
    def _from_frames(cls, container_blob, columns: list):
        msg = cls.__new__(cls)
        msg.__dict__.update(pickle.loads(container_blob))
        msg._packed = False
        msg._raw_deep = None
        for name, frame in zip(cls._COLUMNS, columns):
            setattr(msg, name, blob.offsets_from_frame(frame) if name.endswith("_off") else frame)
        msg._post_decode()
        return msg

    def _post_decode(self) -> None:
        pass

    @classmethod
    def from_byte_array(cls, frames: list):
        return cls._from_frames(frames[1], frames[2:])

    def to_bytes(self) -> bytes:
        """One buffer: ``!HI`` (type id, frame count), ``!Q`` per frame length, then the frames."""
        frames = self.to_byte_array()[1:]
        head = struct.pack("!HI", self.MSG_TYPE_ID, len(frames))
        lengths = struct.pack(f"!{len(frames)}Q", *(len(f) for f in frames))
        return b"".join([head, lengths] + frames)

    @classmethod
    def from_bytes(cls, data: bytes):
        (_, nframes) = struct.unpack_from("!HI", data, 0)
        lengths = struct.unpack_from(f"!{nframes}Q", data, 6)
        mv, pos, frames = memoryview(data), 6 + 8 * nframes, []
        for n in lengths:
            frames.append(mv[pos : pos + n])
            pos += n
        return cls._from_frames(frames[0], frames[1:])

    @property
    def nbytes(self) -> int:
        total = 0
        for name in self._COLUMNS:
            col = getattr(self, name)
            total += col.nbytes if isinstance(col, np.ndarray) else len(col)
        return total


@dataclass
class _ResultColumns(_BlobMessage):
    """Results as columns: ``uids`` (16 B each), ``success`` (1 B each), and a payload record
    ``cloudpickle((task_id, data, exception))`` per result.

    Masters route on the two fixed-width columns alone (`rows()`); only the final consumer
    decodes payloads (`to_results()`, or the `data` property kept for user code).
    """

    _COLUMNS = ("_uids", "_success", "_payload", "_payload_off")

    def __post_init__(self):
        self._uids, self._success = b"", b""
        self._payload, self._payload_off = b"", blob.EMPTY_OFFSETS
        self._decoded = None

    def _post_decode(self) -> None:
        self._decoded = None

    @classmethod
    def from_results(cls, results: List[Result], **msg_fields):
        """Encode Result objects (each needs ``uid``; a missing one is sent as zeros)."""
        msg = cls(**msg_fields)
        zero = bytes(blob.UID_BYTES)
        msg._uids = b"".join(r.uid or zero for r in results)
        msg._success = bytes(1 if r.success else 0 for r in results)
        msg._payload, msg._payload_off = blob.encode_column(
            cloudpickle.dumps((r.task_id, r.data, r.exception)) for r in results
        )
        msg._decoded = list(results)
        return msg

    @property
    def nresults(self) -> int:
        return len(self._success)

    def uids(self) -> List[bytes]:
        return blob.split_fixed(self._uids, blob.UID_BYTES)

    def rows(self):
        """``(uid, success)`` per result, read from the fixed-width columns only."""
        return zip(self.uids(), (bool(b) for b in bytes(self._success)))

    def to_results(self) -> List[Result]:
        if self._decoded is None:
            out = []
            for i, (uid, ok) in enumerate(self.rows()):
                task_id, data, exception = cloudpickle.loads(
                    blob.record(self._payload, self._payload_off, i)
                )
                out.append(Result(sender=self.sender, uid=uid, task_id=task_id, data=data,
                                  success=ok, exception=exception))  # fmt: skip
            self._decoded = out
        return self._decoded

    @property
    def data(self) -> List[Result]:
        """Decoded results, for user code (``res.data[i].data``). Masters never touch this."""
        return self.to_results()

    def to_dict(self):
        return {r.task_id: r.to_dict() for r in self.data}

    def select(self, rows, **msg_fields):
        rows = np.asarray(rows, dtype=np.int64)
        msg = type(self)(**msg_fields)
        msg._uids = blob.gather_fixed(self._uids, blob.UID_BYTES, rows)
        msg._success = blob.gather_fixed(self._success, 1, rows)
        msg._payload, msg._payload_off = blob.gather(self._payload, self._payload_off, rows)
        return msg

    @classmethod
    def concat(cls, batches: list, **msg_fields):
        msg = cls(**msg_fields)
        msg._uids = b"".join(bytes(b._uids) for b in batches)
        msg._success = b"".join(bytes(b._success) for b in batches)
        msg._payload, msg._payload_off = blob.concat(
            [(b._payload, b._payload_off) for b in batches]
        )
        return msg


@dataclass
class ResultBatch(_ResultColumns, Message):
    MSG_TYPE_ID: int = field(default=3, init=False, repr=False)


# Results as they travel between nodes. `ResultBatch` (same layout) is what `run()` returns and
# what the parent-dead fallback file holds; it is no longer sent between nodes (see `Done`).
@dataclass
class IResultBatch(_ResultColumns, Message):
    MSG_TYPE_ID: int = field(default=4, init=False, repr=False)


@dataclass
class TaskUpdate(_BlobMessage, Message):
    """Tasks as columns: ``uids`` (16 B each), a fixed-width metadata record per task (every
    non-deep field, one `ensemble.META_DTYPE` row; read whole columns with `meta_columns`) and
    a deep record per task (executable, args, kwargs, env,
    result). Built with `from_tasks`; masters split and join it with `select` / `concat`
    without decoding; only a worker turns rows back into `Task`s (`task`, `to_tasks`).
    """

    MSG_TYPE_ID: int = field(default=5, init=False, repr=False)

    deleted_uids: List[bytes] = field(default_factory=list)

    _COLUMNS = ("_uids", "_meta", "_deep", "_deep_off")

    def __post_init__(self):
        self._uids = b""
        self._meta = np.empty(0, dtype=META_DTYPE)
        self._deep, self._deep_off = b"", blob.EMPTY_OFFSETS

    def meta_columns(self, rows=None) -> np.ndarray:
        """The metadata records (``META_DTYPE``), all or ``rows``; ``meta["nnodes"]`` etc."""
        return self._meta if rows is None else self._meta[rows]

    @classmethod
    def from_tasks(cls, tasks, **msg_fields) -> "TaskUpdate":
        """Encode Task objects. The Tasks themselves are left untouched."""
        tasks = list(tasks)
        msg = cls(**msg_fields)
        msg._uids = b"".join(t.uid for t in tasks)
        msg._meta = encode_meta_array(tasks)
        msg._deep, msg._deep_off = blob.encode_column(encode_deep(t) for t in tasks)
        return msg

    def _post_decode(self):
        self._meta = decode_meta_array(self._meta)

    @property
    def ntasks(self) -> int:
        return len(self._uids) // blob.UID_BYTES

    def uids(self) -> List[bytes]:
        return blob.split_fixed(self._uids, blob.UID_BYTES)

    def uid_at(self, row: int) -> bytes:
        return bytes(memoryview(self._uids)[row * blob.UID_BYTES : (row + 1) * blob.UID_BYTES])

    def meta_values(self, row: int) -> tuple:
        return meta_row_values(self._meta, row)

    def meta(self, row: int) -> TaskMeta:
        return TaskMeta(self.uid_at(row), self.meta_values(row))

    def task(self, row: int) -> Task:
        values = self.meta_values(row)
        by_value = values[_BY_VALUE_INDEX]
        deep = decode_deep(blob.record(self._deep, self._deep_off, row), by_value)
        return build_task(self.uid_at(row), values, deep)

    def to_tasks(self) -> List[Task]:
        return [self.task(i) for i in range(self.ntasks)]

    def select(self, rows, **msg_fields) -> "TaskUpdate":
        rows = np.asarray(rows, dtype=np.int64)
        msg = TaskUpdate(**msg_fields)
        msg._uids = blob.gather_fixed(self._uids, blob.UID_BYTES, rows)
        msg._meta = self._meta[rows]
        msg._deep, msg._deep_off = blob.gather(self._deep, self._deep_off, rows)
        return msg

    @classmethod
    def concat(cls, updates: list, **msg_fields) -> "TaskUpdate":
        msg = cls(**msg_fields)
        msg._uids = b"".join(bytes(u._uids) for u in updates)
        if updates:
            msg._meta = np.concatenate([u._meta for u in updates])
        msg._deep, msg._deep_off = blob.concat([(u._deep, u._deep_off) for u in updates])
        return msg


_BY_VALUE_INDEX = META_FIELDS.index("serialize_executable_by_value")


@dataclass
class NodeUpdate(Message):
    MSG_TYPE_ID: int = field(default=6, init=False, repr=False)

    nodes: Optional[JobResource] = None


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


# A child is finished: sent to its parent after every one of its results was delivered (all
# result sends ACKed). Results are buffered by type on the receiver, so they are all there by the
# time Done is read. (A child's final ResultBatch used to carry this meaning.)
@dataclass
class Done(Message):
    MSG_TYPE_ID: int = field(default=12, init=False, repr=False)


all_messages = [
    Message,
    Status,
    Result,
    ResultBatch,
    TaskUpdate,
    NodeUpdate,
    Ready,
    Stop,
    TaskRequest,
    NodeRequest,
    IResultBatch,
    Done,
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
