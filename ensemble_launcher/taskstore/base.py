"""Where a node keeps its tasks.

Masters and workers hold their tasks in a `TaskStore` instead of a dict of `Task` objects.
Schedulers and policies only see its read side, a ``Mapping[uid, TaskMeta]``; nothing outside the
store knows how the tasks are held, so an implementation backed by a real database can replace
`SegmentTaskStore` without touching them.

Keys are the 16-byte ``Task.uid``. ``store[uid]`` returns a `TaskMeta` decoded on demand -- read
it and drop it; do not keep many alive.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Type

from ensemble_launcher.ensemble import Task, TaskMeta, TaskStatus

if TYPE_CHECKING:  # messages imports the scheduler package, which imports this one
    import numpy as np

    from ensemble_launcher.comm.messages import TaskUpdate


class TaskStore(Mapping):
    @abstractmethod
    def add(self, update: "TaskUpdate") -> List[bytes]:
        """Take ownership of a received TaskUpdate. Returns the uids it added -- tasks already
        in the store (a resend) are skipped, so a caller schedules each task only once."""

    @abstractmethod
    def __getitem__(self, uid: bytes) -> TaskMeta:
        """Non-deep fields of one task."""

    @abstractmethod
    def task(self, uid: bytes) -> Task:
        """The full Task, including executable/args/kwargs/env. Only workers need this."""

    @abstractmethod
    def build_update(self, uids: Iterable[bytes], **msg_fields) -> "TaskUpdate":
        """A TaskUpdate carrying these tasks, in this order."""

    @abstractmethod
    def meta_columns(self, uids: Iterable[bytes]) -> "np.ndarray":
        """The tasks' metadata records as one ``META_DTYPE`` array; row i is ``uids[i]``.
        Policies with ``columnar = True`` read these columns instead of ``store[uid]``."""

    @abstractmethod
    def get_status(self, uid: bytes, default: Optional[TaskStatus] = None) -> Optional[TaskStatus]:
        ...

    @abstractmethod
    def set_status(self, uids: Iterable[bytes], status: TaskStatus) -> None:
        ...

    @abstractmethod
    def mark_delivered(self, uids: Iterable[bytes]) -> None:
        """These tasks have left this node (a child ACKed them / the worker started them).
        Their bytes are now needed only for recovery, so they are the first to be offloaded."""

    @abstractmethod
    def release(self, uids: Iterable[bytes]) -> None:
        """These tasks are finished; the store may drop their bytes. Status is kept."""

    @abstractmethod
    def close(self) -> None:
        """Drop everything and delete any files the store wrote."""


_REGISTRY: Dict[str, Type[TaskStore]] = {}


def register_task_store(name: str) -> Callable[[Type[TaskStore]], Type[TaskStore]]:
    def deco(cls: Type[TaskStore]) -> Type[TaskStore]:
        _REGISTRY[name] = cls
        return cls

    return deco


def create_task_store(name: str, **kwargs) -> TaskStore:
    try:
        cls = _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown task store {name!r}; registered: {sorted(_REGISTRY)}") from None
    return cls(**kwargs)
