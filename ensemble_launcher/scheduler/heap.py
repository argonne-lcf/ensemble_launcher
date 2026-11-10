"""Pending-task containers for the worker's `AsyncTaskScheduler`: a FIFO deque and a
priority heap with the same interface. The container is part of `TaskSchedulerState`."""

import asyncio
import heapq
from collections import deque
from typing import Deque, Dict, Iterator, List, Optional, Set, Tuple

PendingEntry = Tuple[float, int, str]  # (priority, seq, task_id)


class _PendingTasksBase:
    """Membership + async notification shared by the pending-task containers.

    Subclasses own the ordering structure and implement ``push`` / ``pop`` /
    ``requeue`` / ``remove`` / ``remove_many`` / ``rescore`` / ``sorted_items``
    / ``clear``. The scheduler's dispatch pass is ``pop()`` until something does
    not fit, then ``requeue(skipped)`` -- per-cycle cost proportional to the
    tasks examined, never to the queue length.

    ``_tasks_available`` must never be replaced: ``_monitor_resources`` parks on
    that exact Event object, and swapping it out silently deadlocks dispatch.
    """

    def __init__(self) -> None:
        self._task_ids: Set[str] = set()
        self._seq: int = 0
        self._tasks_available: asyncio.Event = asyncio.Event()

    def _after_shrink(self) -> None:
        if not self._task_ids:
            self._tasks_available.clear()

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._task_ids

    def __len__(self) -> int:
        return len(self._task_ids)

    def __iter__(self) -> Iterator[bytes]:
        return iter(self._task_ids)

    def empty(self) -> bool:
        return len(self._task_ids) == 0

    def task_ids(self) -> Set[str]:
        return set(self._task_ids)

    async def wait_for_tasks(self) -> None:
        """Block until at least one task is pending."""
        await self._tasks_available.wait()


class PendingTaskQueue(_PendingTasksBase):
    """FIFO collection of pending tasks, backed by a deque.

    Used for ``fifo_policy``: tasks are never scored, so ``push`` ignores its
    priority argument and every operation on the hot path is O(1). Entries keep
    the ``(priority, seq, task_id)`` shape of :class:`PendingTaskHeap` so the two
    are interchangeable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: Deque[PendingEntry] = deque()

    def push(self, priority: float, task_id: str) -> None:
        self._queue.append((0.0, self._seq, task_id))
        self._seq += 1
        self._task_ids.add(task_id)
        self._tasks_available.set()

    def pop(self) -> Optional[PendingEntry]:
        """Remove and return the oldest entry, or ``None`` if empty."""
        if not self._queue:
            return None
        entry = self._queue.popleft()
        self._task_ids.discard(entry[2])
        self._after_shrink()
        return entry

    def requeue(self, entries: List[PendingEntry]) -> None:
        """Put popped-but-not-dispatched entries back at the front, in order."""
        if not entries:
            return
        self._queue.extendleft(reversed(entries))
        self._task_ids.update(tid for _p, _s, tid in entries)
        self._tasks_available.set()

    def remove(self, task_id: str) -> bool:
        if task_id not in self._task_ids:
            return False
        self._task_ids.discard(task_id)
        self._queue = deque(e for e in self._queue if e[2] != task_id)
        self._after_shrink()
        return True

    def remove_many(self, task_ids: Set[str]) -> None:
        self._task_ids -= task_ids
        self._queue = deque(e for e in self._queue if e[2] in self._task_ids)
        self._after_shrink()

    def rescore(self, scores: Dict[str, float]) -> None:
        """FIFO has no priorities: only drop ids absent from *scores*, keep order."""
        self._queue = deque(e for e in self._queue if e[2] in scores)
        self._task_ids = {tid for _p, _s, tid in self._queue}
        self._after_shrink()

    def sorted_items(self) -> List[PendingEntry]:
        """All entries in dispatch order (a copy)."""
        return list(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self._task_ids.clear()
        self._seq = 0
        self._tasks_available.clear()


class PendingTaskHeap(_PendingTasksBase):
    """Priority-ordered collection of pending tasks with async notification.

    Lower priority value dispatches first; ties break on insertion order.
    """

    def __init__(self) -> None:
        super().__init__()
        self._heap: List[PendingEntry] = []

    def push(self, priority: float, task_id: str) -> None:
        heapq.heappush(self._heap, (priority, self._seq, task_id))
        self._seq += 1
        self._task_ids.add(task_id)
        self._tasks_available.set()

    def pop(self) -> Optional[PendingEntry]:
        """Remove and return the highest-priority entry, or ``None`` if empty."""
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        self._task_ids.discard(entry[2])
        self._after_shrink()
        return entry

    def requeue(self, entries: List[PendingEntry]) -> None:
        """Push popped-but-not-dispatched entries back with their original
        ``(priority, seq)``, so their place -- including the FIFO tiebreak -- is
        unchanged."""
        if not entries:
            return
        for entry in entries:
            heapq.heappush(self._heap, entry)
            self._task_ids.add(entry[2])
        self._tasks_available.set()

    def remove(self, task_id: str) -> bool:
        if task_id not in self._task_ids:
            return False
        self._task_ids.discard(task_id)
        self._heap = [(p, s, tid) for p, s, tid in self._heap if tid != task_id]
        heapq.heapify(self._heap)
        self._after_shrink()
        return True

    def remove_many(self, task_ids: Set[str]) -> None:
        self._task_ids -= task_ids
        self._heap = [(p, s, tid) for p, s, tid in self._heap if tid in self._task_ids]
        heapq.heapify(self._heap)
        self._after_shrink()

    def rescore(self, scores: Dict[str, float]) -> None:
        """Re-prioritize every entry in place. Ids absent from *scores* are dropped.

        Each entry keeps its original sequence number, so the FIFO tiebreak among
        equal priorities survives -- which also means the iteration order here is
        irrelevant and a plain dict is a safe argument. ``clear()`` + ``push()``
        would not have either property: it resets ``_seq`` to 0 and renumbers in
        whatever order it is handed.

        ``_tasks_available`` is deliberately not ``set()``: membership only ever
        shrinks here, so an already-set event must stay set and a heap that empties
        out must clear it.
        """
        self._heap = [
            (scores[task_id], seq, task_id)
            for _p, seq, task_id in self._heap
            if task_id in scores
        ]
        heapq.heapify(self._heap)
        self._task_ids = {task_id for _p, _s, task_id in self._heap}
        self._after_shrink()

    def sorted_items(self) -> List[PendingEntry]:
        """All entries in dispatch order (a sorted copy; the heap is untouched).

        O(n log n) -- for introspection and tests only. The dispatch loop uses
        ``pop`` / ``requeue``.
        """
        return sorted(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._task_ids.clear()
        self._seq = 0
        self._tasks_available.clear()
