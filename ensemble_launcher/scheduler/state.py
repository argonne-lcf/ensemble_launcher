"""Scheduler state.

`SchedulerState` is everything a master's `AsyncChildrenScheduler` knows, in one object: its
children (one `Child` record each) and the tasks not yet given to any of them. The scheduler
mutates it; orchestrators read it; and it is its own checkpoint (`to_json` / `from_json`).

`TaskSchedulerState` is the same for a worker's `AsyncTaskScheduler`: its pending container,
running and finished tasks. Task policies receive it as ``scheduler_state``; it is also the
worker's checkpoint.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Union

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
    """One child of a master: where it runs, what it still has to do, and where it is in its
    lifecycle.

    ``uids`` is an insertion-ordered set (a dict with ``None`` values) of the child's
    *unfinished* tasks: the scheduler drops a task as soon as its result arrives, so it is
    exactly what a restarted child must be resent. It is the source of truth.

    ``outbox`` holds uid lists placed on the child and not yet sent; the master's sender for
    this child consumes it. A queued uid may go stale (deleted meanwhile): the sender keeps only
    uids still in ``uids``.
    """

    id: str
    wid: int
    job_resource: Optional[JobResource]
    uids: Dict[bytes, None] = field(default_factory=dict)
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

    def add(self, uids: Iterable[bytes]) -> None:
        self.uids.update(dict.fromkeys(uids))

    def discard(self, uid: bytes) -> None:
        self.uids.pop(uid, None)

    def __contains__(self, uid: bytes) -> bool:
        return uid in self.uids

    def __len__(self) -> int:
        return len(self.uids)

    @property
    def terminal(self) -> bool:
        return self.state in _TERMINAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "wid": self.wid,
            "job_resource": _resource_to_json(self.job_resource),
            "uids": [u.hex() for u in self.uids],
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
            uids=dict.fromkeys(bytes.fromhex(u) for u in d["uids"]),
            task_executor_name=d.get("task_executor_name"),
            child_class=d.get("child_class"),
        )


# The old name, kept for type hints in custom policies.
ChildrenAssignment = Child


@dataclass(eq=False)
class SchedulerState:
    """A master scheduler's state: its children and its not-yet-assigned tasks.

    Plain data plus queries -- it never touches the cluster, the task store or the policy.
    `AsyncChildrenScheduler` is the only writer. ``to_json`` writes the fields needed to
    rebuild the tree after a restart; the runtime-only ones are rebuilt by
    `AsyncChildrenScheduler.set_state`.

    The moves below keep one rule: **every unfinished uid is in ``unassigned`` or in exactly
    one child's ``uids``.**
    """

    node_id: str
    level: Optional[int] = None
    nodes: Optional[JobResource] = None
    children: Dict[str, Child] = field(default_factory=dict)  # insertion-ordered
    task_to_client: Dict[bytes, str] = field(default_factory=dict)  # unfinished client tasks
    # --- runtime only: not checkpointed ---
    unassigned: Dict[bytes, None] = field(default_factory=dict)  # ordered set of uids
    ntasks_added: int = 0  # tasks ever added (the store forgets finished ones)
    all_done: asyncio.Event = field(default_factory=asyncio.Event)
    _by_wid: Dict[int, Child] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._by_wid = {c.wid: c for c in self.children.values()}

    # --- tasks --------------------------------------------------------- #

    def queue(self, uids: Iterable[bytes], client_id: Optional[str] = None) -> None:
        """New tasks enter the pool."""
        uids = list(uids)
        self.unassigned.update(dict.fromkeys(uids))
        if client_id is not None:
            self.task_to_client.update(dict.fromkeys(uids, client_id))
        self.ntasks_added += len(uids)

    def assign(self, child: Child, uids: Iterable[bytes]) -> None:
        """Pool -> child. Putting the uids on the child's outbox is the caller's job."""
        pool = self.unassigned
        for uid in uids:
            pool.pop(uid, None)
        child.add(uids)

    def finish(self, uid: bytes, child: Optional[Child] = None) -> Optional[str]:
        """A task has a final result: it leaves its child (or the pool, if it never reached
        one). Returns its submitting client, if any."""
        if child is not None:
            child.discard(uid)
        else:
            self.unassigned.pop(uid, None)
        return self.task_to_client.pop(uid, None)

    def forget(self, uid: bytes) -> None:
        """A task is deleted: drop it from wherever it is."""
        self.unassigned.pop(uid, None)
        self.task_to_client.pop(uid, None)
        for child in self.children.values():  # O(nchildren); deletes are rare
            child.discard(uid)

    def empty(self) -> bool:
        """No unfinished task anywhere: nothing in the pool and nothing held by a child."""
        return not self.unassigned and not any(c.uids for c in self.children.values())

    def unfinished(self) -> List[Tuple[Optional[str], List[bytes]]]:
        """Where the unfinished tasks are: ``(None, pool uids)`` first (if any), then
        ``(child_id, uids)`` per child holding tasks. Each unfinished uid appears once."""
        out: List[Tuple[Optional[str], List[bytes]]] = []
        if self.unassigned:
            out.append((None, list(self.unassigned)))
        out.extend((c.id, list(c.uids)) for c in self.children.values() if c.uids)
        return out

    # --- children ------------------------------------------------------ #

    def add_child(self, child: Child) -> None:
        """Add a child. It claims those of its ``uids`` still in the pool (the rest are
        already elsewhere, or gone) and gets a fresh outbox holding all of them: a new child
        process has nothing yet."""
        pool = self.unassigned
        child.uids = {uid: None for uid in child.uids if uid in pool}
        for uid in child.uids:
            del pool[uid]
        child.outbox = asyncio.Queue()
        if child.uids:
            child.outbox.put_nowait(list(child.uids))
        self.children[child.id] = child
        self._by_wid[child.wid] = child

    def pop_child(self, child_id: str) -> Optional[Child]:
        """Remove a child; its unfinished uids go back to the pool. Its outbox is left with
        it (its sender goes too)."""
        child = self.children.pop(child_id, None)
        if child is None:
            return None
        if self._by_wid.get(child.wid) is child:
            del self._by_wid[child.wid]
        self.unassigned.update(dict.fromkeys(child.uids))
        return child

    def by_wid(self, wid: int) -> Child:
        return self._by_wid[wid]

    def next_wid(self) -> int:
        return max(self._by_wid, default=-1) + 1

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
                "task_to_client": {u.hex(): c for u, c in self.task_to_client.items()},
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
            task_to_client={bytes.fromhex(u): c for u, c in d.get("task_to_client", {}).items()},
        )


@dataclass(eq=False)
class TaskSchedulerState:
    """A worker scheduler's state: its pending, running and finished tasks.

    `AsyncTaskScheduler` is the only writer. Task policies receive this object itself as
    ``scheduler_state`` in ``get_score`` / ``on_task_complete`` and must only read it.
    ``to_json`` writes running tasks as pending (their futures do not survive a restart);
    ``from_json`` gives ``pending_tasks`` as a plain uid list, which
    `AsyncTaskScheduler.set_state` turns back into its pending container.
    """

    node_id: str
    nodes: Optional[JobResource] = None
    # The scheduler's dispatch container (PendingTaskQueue / PendingTaskHeap): supports
    # ``in``, ``len`` and iteration over uids.
    pending_tasks: Union[PendingTaskQueue, PendingTaskHeap, List[bytes]] = field(
        default_factory=PendingTaskQueue
    )
    running_tasks: Dict[bytes, JobResource] = field(default_factory=dict)  # uid -> resource
    completed_tasks: Set[bytes] = field(default_factory=set)
    failed_tasks: Set[bytes] = field(default_factory=set)
    task_to_client: Dict[bytes, str] = field(default_factory=dict)  # unfinished client tasks
    # --- runtime only: not checkpointed ---
    all_done: asyncio.Event = field(default_factory=asyncio.Event)

    def unfinished(self) -> Set[bytes]:
        """Uids still to run: pending or running."""
        return set(self.pending_tasks) | set(self.running_tasks)

    def to_json(self) -> str:
        return json.dumps(
            {
                "node_id": self.node_id,
                "nodes": _resource_to_json(self.nodes),
                "pending_tasks": [u.hex() for u in self.unfinished()],
                "completed_tasks": [u.hex() for u in self.completed_tasks],
                "failed_tasks": [u.hex() for u in self.failed_tasks],
                "task_to_client": {u.hex(): c for u, c in self.task_to_client.items()},
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "TaskSchedulerState":
        d = json.loads(raw)
        uids = lambda key: [bytes.fromhex(u) for u in d.get(key, [])]  # noqa: E731
        return cls(
            node_id=d["node_id"],
            nodes=_resource_from_json(d.get("nodes")),
            pending_tasks=uids("pending_tasks"),
            completed_tasks=set(uids("completed_tasks")),
            failed_tasks=set(uids("failed_tasks")),
            task_to_client={bytes.fromhex(u): c for u, c in d.get("task_to_client", {}).items()},
        )
