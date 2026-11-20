"""Where a node keeps its tasks.

Masters and workers hold their tasks in a `TaskStore` instead of a dict of `Task` objects.
Nothing outside the store knows how the tasks are held, so an implementation backed by a real
database can replace `SegmentTaskStore` without touching its users.

**Handles, not uids.** A uid is how *nodes* name a task to each other (it travels in every
`TaskUpdate` and result). Inside one node, every task is named by its **handle**: an opaque
int64 the store hands out in `add`. The contract:

- unique among the store's live tasks and stable until the task is released;
- the handles one `add` returns increase, in the update's row order;
- callers never decode its bits, and never send or checkpoint it (a new store -- e.g. after a
  restart -- hands out new handles).

Uids become handles in one place, `lookup`; handles become uids only when the store builds a
message (`build_update`, `build_failed_results`) or a `Task`.

**Per task, the store also keeps:**

- an int32 **assignment**: `UNASSIGNED`, or any tag >= 0 the caller picks (a master tags a task
  with the wid of the child it is placed on). A released handle reads as `RELEASED`.
- a **client** code: `NO_CLIENT` for a task from the parent, else a code for the submitting
  client (`client_name(code)`).

The Mapping side is keyed by handle: ``store[h]`` is a `TaskMeta` decoded on demand -- read it
and drop it; do not keep many alive. Methods taking ``idx`` take an int64 array (or sequence)
of handles, distinct within one call.
"""

from abc import abstractmethod
from collections.abc import Mapping
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Sequence, Tuple, Type

from ensemble_launcher.ensemble import Task, TaskMeta

if TYPE_CHECKING:  # messages imports the scheduler package, which imports this one
    import numpy as np

    from ensemble_launcher.comm.messages import IResultBatch, TaskUpdate

UNASSIGNED = -1  # assignment of a live task no one has claimed (a master's pool)
RELEASED = -2  # what `assigned` reports for a handle that is no longer live
NO_CLIENT = -1  # client code of a task that came from the parent


class TaskStore(Mapping):
    # ------------------------------------------------------------------ #
    #  Tasks in and out                                                  #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def add(self, update: "TaskUpdate", client_id: Optional[str] = None) -> "np.ndarray":
        """Take ownership of a received TaskUpdate; returns the handles of the tasks it added,
        in row order. Tasks already in the store (a resend) are skipped, so a caller schedules
        each task only once. New tasks are `UNASSIGNED`, with ``client_id``'s code."""

    @abstractmethod
    def lookup(self, uids: Sequence[bytes]) -> "np.ndarray":
        """The handles of these uids, in order; -1 for a uid the store does not hold."""

    @abstractmethod
    def release(self, idx, uids: Optional[Sequence[bytes]] = None) -> None:
        """These tasks are finished: they leave the store at once (``h in store`` is False
        afterwards, `assigned` reports `RELEASED`) and their bytes may go. No record of how
        they ended is kept; handles already released, and repeats, are ignored.

        ``uids``, when the caller has them (results carry them), are the same tasks' uids
        row for row: the store may use them instead of reading them back, and then releases
        exactly the live tasks among them (``idx`` may then hold -1 for unknown ones)."""

    @abstractmethod
    def mark_delivered(self, idx) -> None:
        """These tasks have left this node (a child ACKed them / the worker started them).
        Their bytes are now needed only for recovery, so they are the first to be offloaded."""

    @abstractmethod
    def close(self) -> None:
        """Drop everything and delete any files the store wrote."""

    # ------------------------------------------------------------------ #
    #  Reading tasks (Mapping: handle -> TaskMeta)                       #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def __getitem__(self, h: int) -> TaskMeta:
        """Non-deep fields of one live task (KeyError if it is not live)."""

    @abstractmethod
    def task(self, h: int) -> Task:
        """The full Task, including executable/args/kwargs/env. Only workers need this."""

    @abstractmethod
    def all(self) -> "np.ndarray":
        """Every live handle, in insertion order."""

    @abstractmethod
    def uids(self, idx) -> List[bytes]:
        """The tasks' uids, in order (tests and logging; messages carry uids themselves)."""

    @abstractmethod
    def meta_columns(self, idx) -> "np.ndarray":
        """The tasks' metadata records as one ``META_DTYPE`` array; row i is ``idx[i]``.
        Policies' column entry points read these instead of ``store[h]``."""

    @abstractmethod
    def build_update(self, idx, **msg_fields) -> "TaskUpdate":
        """A TaskUpdate carrying these tasks (with their uids), in this order."""

    @abstractmethod
    def build_failed_results(self, idx, sender: str, reason: str) -> Tuple["IResultBatch", "np.ndarray"]:
        """A failed result for each of these tasks that is still live, built from the store's
        own columns. Returns ``(batch, handles)``: row i of the batch is ``handles[i]``."""

    # ------------------------------------------------------------------ #
    #  Assignment                                                        #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def assign(self, idx, tag: int) -> "np.ndarray":
        """Tag those of these tasks that are still `UNASSIGNED` with ``tag`` (>= 0); returns
        the handles that moved, in the order given."""

    @abstractmethod
    def reassign(self, from_tag: int, to_tag: int) -> "np.ndarray":
        """Move every live task tagged ``from_tag`` to ``to_tag``; returns their handles, in
        insertion order."""

    @abstractmethod
    def assigned(self, idx) -> "np.ndarray":
        """Each task's tag (int32), `RELEASED` for a handle that is not live."""

    @abstractmethod
    def assigned_to(self, tag: int) -> "np.ndarray":
        """Every live task tagged ``tag``, in insertion order."""

    @abstractmethod
    def count(self, tag: int) -> int:
        """How many live tasks are tagged ``tag``."""

    @abstractmethod
    def groups(self) -> List[Tuple[int, "np.ndarray"]]:
        """The live tasks grouped by tag, ``[(tag, handles)]`` in increasing tag order (so
        `UNASSIGNED` first), each group in insertion order."""

    # ------------------------------------------------------------------ #
    #  Clients                                                           #
    # ------------------------------------------------------------------ #

    @abstractmethod
    def clients(self, idx) -> "np.ndarray":
        """Each task's client code (int32), `NO_CLIENT` for tasks from the parent."""

    @abstractmethod
    def client_name(self, code: int) -> Optional[str]:
        """The client id behind a code (None for `NO_CLIENT`)."""

    @abstractmethod
    def has_clients(self) -> bool:
        """Whether any task was ever added with a client (else every code is `NO_CLIENT`)."""


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
