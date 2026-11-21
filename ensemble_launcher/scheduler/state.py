"""Scheduler state.

`SchedulerState` is what a master's `AsyncChildrenScheduler` knows about its *children*: one
`Child` record each, plus counters. It is also the master's checkpoint (`to_json` /
`from_json`), which is the layout only: no task is named in it.

Where each *task* is -- in the pool, on a child, finished -- is not here: it is task data, so it
lives in the task store (its assignment column; see `ensemble_launcher.taskstore`). Each object's
methods change only that object; `AsyncChildrenScheduler` is the one place a move touches both.

`TaskSchedulerState` is a worker's `AsyncTaskScheduler` state: its pending container, running
tasks and finish counters. Task policies receive it as ``scheduler_state``. It is runtime only:
a restarted worker gets its tasks again from its parent.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .child_state import ChildState
from .heap import PendingTaskHeap, PendingTaskQueue
from .resource import JobResource

if TYPE_CHECKING:
    from ensemble_launcher.comm.messages import Status

_TERMINAL = frozenset({ChildState.SUCCESS, ChildState.FAILED})


def _resource_to_json(r: Optional[JobResource]) -> Optional[Dict[str, Any]]:
    return r.serialize() if r is not None else None


def _resource_from_json(d: Optional[Dict[str, Any]]) -> Optional[JobResource]:
    return JobResource.deserialize(d) if d is not None else None


@dataclass(eq=False)
class Child:
    """One child of a master: where it runs and where it is in its lifecycle.

    Its tasks are the store's rows tagged with its ``wid`` (``store.assigned_to(wid)``,
    ``store.count(wid)``); they keep that tag while a removed child is being relaunched.

    ``outbox`` holds handle arrays placed on the child and not yet sent; the master's sender for
    this child consumes it. A queued handle may go stale (finished or moved meanwhile): the
    sender keeps only those still tagged with the child's wid.
    """

    id: str
    wid: int
    job_resource: Optional[JobResource]
    state: ChildState = ChildState.NOTREADY
    # Per-child overrides, read by AsyncMaster when it instantiates the child.
    task_executor_name: Optional[str] = None
    child_class: Optional[str] = None
    # --- runtime only: not checkpointed ---
    status: Optional["Status"] = None  # last Status message from the child
    allocated: bool = False  # holds its cluster resources
    ready: asyncio.Event = field(default_factory=asyncio.Event)  # set once RUNNING
    done: asyncio.Event = field(default_factory=asyncio.Event)  # set once terminal
    outbox: asyncio.Queue = field(default_factory=asyncio.Queue, repr=False)

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "wid": self.wid,
            "job_resource": _resource_to_json(self.job_resource),
            "state": self.state.name,  # informational: a restored child starts NOTREADY
            "task_executor_name": self.task_executor_name,
            "child_class": self.child_class,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Child":
        return cls(
            id=d["id"],
            wid=d["wid"],
            job_resource=_resource_from_json(d["job_resource"]),
            task_executor_name=d.get("task_executor_name"),
            child_class=d.get("child_class"),
        )


# The old name, kept for type hints in custom policies.
ChildrenAssignment = Child


@dataclass(eq=False)
class SchedulerState:
    """A master scheduler's state: **only its children's state and resource allocations.**

    One `Child` record per child (id, wid, allocated `JobResource`, lifecycle state, last
    Status, outbox) plus the master's own nodes and counters. It holds **no task data**: which
    tasks are in the pool or on which child, and which client submitted them, live in the task
    store (its assignment and client columns), read through ``store.assigned_to(wid)`` /
    ``store.count(wid)``.

    Plain data plus queries; its methods change only itself, and `AsyncChildrenScheduler` is
    the only writer. ``to_json`` writes the layout needed to rebuild the tree after a restart;
    the runtime-only fields are rebuilt by `AsyncChildrenScheduler.set_state`.
    """

    node_id: str
    level: Optional[int] = None
    nodes: Optional[JobResource] = None
    children: Dict[str, Child] = field(default_factory=dict)  # insertion-ordered
    # Wids are never reused: a child removed for recovery still tags its tasks with its wid.
    wid_counter: int = 0
    # --- runtime only: not checkpointed ---
    ntasks_added: int = 0  # tasks ever added (the store forgets finished ones)
    all_done: asyncio.Event = field(default_factory=asyncio.Event)
    _by_wid: Dict[int, Child] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_wid = {c.wid: c for c in self.children.values()}
        self.wid_counter = max([self.wid_counter, *(w + 1 for w in self._by_wid)])

    # --- children ------------------------------------------------------ #

    def add_child(self, child: Child) -> None:
        """Register a child, with a fresh (empty) outbox."""
        child.outbox = asyncio.Queue()
        self.children[child.id] = child
        self._by_wid[child.wid] = child
        self.wid_counter = max(self.wid_counter, child.wid + 1)

    def pop_child(self, child_id: str) -> Optional[Child]:
        """Unregister a child. Its outbox is left with it (its sender goes too)."""
        child = self.children.pop(child_id, None)
        if child is None:
            return None
        if self._by_wid.get(child.wid) is child:
            del self._by_wid[child.wid]
        return child

    def by_wid(self, wid: int) -> Child:
        return self._by_wid[wid]

    def child_of(self, wid: int) -> Optional[Child]:
        return self._by_wid.get(wid)

    def next_wid(self) -> int:
        return self.wid_counter

    def all_terminal(self) -> bool:
        return bool(self.children) and all(c.terminal for c in self.children.values())

    def aggregate_status(self) -> "Status":
        """Sum of the children's latest Status messages."""
        from ensemble_launcher.comm.messages import Status

        return sum((c.status for c in self.children.values() if c.status), Status())

    # --- checkpoint ---------------------------------------------------- #

    def to_json(self) -> str:
        return json.dumps(
            {
                "node_id": self.node_id,
                "level": self.level,
                "nodes": _resource_to_json(self.nodes),
                "children": [c.to_dict() for c in self.children.values()],
                "wid_counter": self.wid_counter,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "SchedulerState":
        d = json.loads(raw)
        children = [Child.from_dict(c) for c in d.get("children", [])]
        return cls(
            node_id=d["node_id"],
            level=d.get("level"),
            nodes=_resource_from_json(d.get("nodes")),
            children={c.id: c for c in children},
            wid_counter=d.get("wid_counter", 0),
        )


@dataclass(eq=False)
class TaskSchedulerState:
    """A worker scheduler's state: its pending and running tasks (by store handle) and how
    many finished.

    `AsyncTaskScheduler` is the only writer. Task policies receive this object itself as
    ``scheduler_state`` in ``get_score`` / ``on_task_complete`` and must only read it. Runtime
    only: never checkpointed.
    """

    node_id: str
    nodes: Optional[JobResource] = None
    # The scheduler's dispatch container (PendingTaskQueue / PendingTaskHeap): supports
    # ``in``, ``len`` and iteration over handles.
    pending_tasks: Union[PendingTaskQueue, PendingTaskHeap] = field(default_factory=PendingTaskQueue)
    running_tasks: Dict[int, JobResource] = field(default_factory=dict)  # handle -> resource
    ncompleted: int = 0
    nfailed: int = 0
    all_done: asyncio.Event = field(default_factory=asyncio.Event)

    def unfinished(self) -> list:
        """Handles still to run: pending or running."""
        return list(self.pending_tasks) + list(self.running_tasks)
