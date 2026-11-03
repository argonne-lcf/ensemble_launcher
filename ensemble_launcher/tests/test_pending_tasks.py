"""Unit tests for the pending-task containers and the task scheduler's dispatch loop.

Covers the pop/requeue allocation pass that replaced the per-cycle sort, the
deque-backed FIFO path, and the running minimum resource requirement.
"""

import asyncio
import logging
import random
import socket

import pytest
from utils import echo

from ensemble_launcher.config import PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.scheduler import (
    AsyncTaskScheduler,
    PendingTaskHeap,
    PendingTaskQueue,
)
from ensemble_launcher.scheduler.policy import FIFOPolicy, LargeResourcePolicy
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

pytestmark = pytest.mark.core


def _job_resource(ncpus: int = 4) -> JobResource:
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    return JobResource(resources=[sys_info], nodes=[socket.gethostname()])


def _task(tid: str, ppn: int = 1) -> Task:
    return Task(task_id=tid, nnodes=1, ppn=ppn, executable=echo, args=(tid,))


def _ids(entries):
    return [tid for _p, _s, tid in entries]


# ---------------------------------------------------------------------------
# PendingTaskHeap
# ---------------------------------------------------------------------------


def test_heap_pop_priority_then_fifo():
    heap = PendingTaskHeap()
    for tid, prio in [("a", 2.0), ("b", 1.0), ("c", 2.0), ("d", 1.0)]:
        heap.push(prio, tid)
    popped = []
    while (e := heap.pop()) is not None:
        popped.append(e[2])
    assert popped == ["b", "d", "a", "c"]
    assert heap.empty()
    assert not heap._tasks_available.is_set()


def test_heap_requeue_restores_order_and_membership():
    heap = PendingTaskHeap()
    for i, prio in enumerate([3.0, 1.0, 2.0]):
        heap.push(prio, f"t{i}")
    before = heap.sorted_items()
    e1, e2 = heap.pop(), heap.pop()
    assert len(heap) == 1 and e1[2] not in heap and e2[2] not in heap
    heap.requeue([e1, e2])
    assert heap.sorted_items() == before
    assert heap.task_ids() == {"t0", "t1", "t2"}


def test_heap_sorted_items_does_not_mutate():
    heap = PendingTaskHeap()
    for i, prio in enumerate([3.0, 1.0, 2.0, 0.5]):
        heap.push(prio, f"t{i}")
    raw = list(heap._heap)
    assert _ids(heap.sorted_items()) == ["t3", "t1", "t2", "t0"]
    assert heap._heap == raw


def test_heap_event_across_pop_and_requeue():
    heap = PendingTaskHeap()
    heap.push(1.0, "a")
    event = heap._tasks_available
    assert event.is_set()
    e = heap.pop()
    assert not event.is_set()
    heap.requeue([e])
    assert event.is_set()
    assert heap._tasks_available is event
    heap.requeue([])  # no-op must not arm an empty container
    heap.clear()
    heap.requeue([])
    assert not event.is_set()


# ---------------------------------------------------------------------------
# PendingTaskQueue
# ---------------------------------------------------------------------------


def test_queue_pop_is_fifo_and_ignores_priority():
    q = PendingTaskQueue()
    for tid, prio in [("a", 5.0), ("b", -1.0), ("c", 0.0)]:
        q.push(prio, tid)
    assert [q.pop()[2] for _ in range(3)] == ["a", "b", "c"]
    assert q.pop() is None
    assert not q._tasks_available.is_set()


def test_queue_requeue_puts_skipped_back_at_front():
    q = PendingTaskQueue()
    for i in range(5):
        q.push(0.0, f"t{i}")
    skipped = [q.pop(), q.pop()]  # t0, t1 didn't fit
    allocated = q.pop()  # t2 did
    assert allocated[2] == "t2"
    q.requeue(skipped)
    assert _ids(q.sorted_items()) == ["t0", "t1", "t3", "t4"]
    assert q.task_ids() == {"t0", "t1", "t3", "t4"}


def test_queue_remove_and_rescore():
    q = PendingTaskQueue()
    for i in range(4):
        q.push(0.0, f"t{i}")
    assert q.remove("t1") is True
    assert q.remove("t1") is False
    assert _ids(q.sorted_items()) == ["t0", "t2", "t3"]
    q.remove_many({"t3"})
    assert _ids(q.sorted_items()) == ["t0", "t2"]
    event = q._tasks_available
    # Scores are ignored; only membership matters, and order is kept.
    q.rescore({"t2": -100.0, "t0": 100.0})
    assert _ids(q.sorted_items()) == ["t0", "t2"]
    q.rescore({"t2": 0.0})
    assert _ids(q.sorted_items()) == ["t2"]
    q.rescore({})
    assert q.empty() and not event.is_set()
    assert q._tasks_available is event


# ---------------------------------------------------------------------------
# AsyncTaskScheduler
# ---------------------------------------------------------------------------


def test_scheduler_default_policy_is_fifo():
    sched = AsyncTaskScheduler(logging.getLogger("t"), {}, _job_resource())
    assert isinstance(sched.scheduler_policy, FIFOPolicy)
    assert isinstance(sched._pending_tasks, PendingTaskQueue)


def test_scheduler_picks_container_by_policy_name():
    log = logging.getLogger("t")
    assert isinstance(
        AsyncTaskScheduler(log, {}, _job_resource(), policy="large_resource_policy")._pending_tasks,
        PendingTaskHeap,
    )
    # An instance of the registered class resolves to its registered name...
    assert isinstance(
        AsyncTaskScheduler(log, {}, _job_resource(), policy=FIFOPolicy())._pending_tasks,
        PendingTaskQueue,
    )

    # ...but a subclass may override get_score, so it must be scored.
    class MyFIFO(FIFOPolicy):
        pass

    assert isinstance(
        AsyncTaskScheduler(log, {}, _job_resource(), policy=MyFIFO())._pending_tasks,
        PendingTaskHeap,
    )


def test_fifo_scheduler_never_scores(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("get_score must not be called on the FIFO path")

    monkeypatch.setattr(FIFOPolicy, "get_score", boom)
    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {t: _task(t) for t in ("a", "b")},
        _job_resource(),
        policy="fifo_policy",
    )
    assert sched.add_task(_task("c"))
    assert _ids(sched._pending_tasks.sorted_items()) == ["a", "b", "c"]
    assert sched.reprioritize_pending() == 3
    assert _ids(sched._pending_tasks.sorted_items()) == ["a", "b", "c"]


def test_stateless_policy_gets_no_scheduler_state(monkeypatch):
    seen = []
    orig = LargeResourcePolicy.get_score

    def spy(self, task, scheduler_state=None):
        seen.append(scheduler_state)
        return orig(self, task, scheduler_state)

    monkeypatch.setattr(LargeResourcePolicy, "get_score", spy)
    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {"a": _task("a")},
        _job_resource(),
        policy="large_resource_policy",
    )
    sched.add_task(_task("b"))
    assert seen == [None, None]


async def _drain_in_order(sched: AsyncTaskScheduler, n: int):
    order = []
    for _ in range(n):
        tid, _res = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        order.append(tid)
        sched.free(tid, TaskStatus.SUCCESS)
    return order


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["fifo_policy", "large_resource_policy"])
async def test_single_slot_dispatch_order(policy):
    """One CPU: dispatch order is exactly the container's pop order."""
    ids = [f"t{i}" for i in range(20)]
    sched = AsyncTaskScheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=1), policy=policy
    )
    for t in ids:
        sched.add_task(_task(t))
    sched.start_monitoring()
    try:
        assert await _drain_in_order(sched, len(ids)) == ids
        await asyncio.wait_for(sched.wait_for_completion(), timeout=5.0)
        assert sched._pending_tasks.empty()
    finally:
        await sched.stop_monitoring()


@pytest.mark.asyncio
async def test_priority_dispatch_order():
    """Bigger tasks first under large_resource_policy, FIFO among equals."""
    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {},
        _job_resource(ncpus=2),
        policy="large_resource_policy",
        policy_config=PolicyConfig(strict_priority=True),
    )
    for tid, ppn in [("s0", 1), ("b0", 2), ("s1", 1), ("b1", 2)]:
        sched.add_task(_task(tid, ppn=ppn))
    sched.start_monitoring()
    try:
        assert await _drain_in_order(sched, 4) == ["b0", "b1", "s0", "s1"]
    finally:
        await sched.stop_monitoring()


@pytest.mark.asyncio
async def test_fifo_backfill_skips_head_that_does_not_fit():
    """Non-strict FIFO lets a small task pass a big one, and the big one keeps its place."""
    sched = AsyncTaskScheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=2), policy="fifo_policy"
    )
    sched.add_task(_task("hold", ppn=1))
    sched.start_monitoring()
    try:
        hold, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        assert hold == "hold"
        # One CPU left: "big" can't start, "small" can.
        sched.add_task(_task("big", ppn=2))
        sched.add_task(_task("small", ppn=1))
        small, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        assert small == "small"
        assert _ids(sched._pending_tasks.sorted_items()) == ["big"]
        sched.free("hold", TaskStatus.SUCCESS)
        sched.free("small", TaskStatus.SUCCESS)
        big, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        assert big == "big"
    finally:
        await sched.stop_monitoring()


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["fifo_policy", "large_resource_policy"])
async def test_free_sets_resource_event_synchronously(policy):
    """The resource-available event must track the cluster exactly, with no late set.

    deallocate() used to defer the set with call_soon_threadsafe. It could then fire
    after the freed slot was refilled, leaving the event set on a full cluster. A
    failed allocate never clears the event, so the next pass tested every pending
    task -- O(pending) per completion, which made throughput fall with batch size.
    """
    sched = AsyncTaskScheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=2), policy=policy
    )
    sched.cluster.set_event_loop(asyncio.get_running_loop())
    event = sched.cluster._resource_available
    for i in range(1000):
        sched.add_task(_task(f"t{i}"))
    assert sched._allocate_pending() == 2  # fills the cluster
    assert not event.is_set()

    tid, _ = sched.ready_tasks.get_nowait()
    sched.free(tid, TaskStatus.SUCCESS)
    assert event.is_set()  # set in the same step, without yielding to the loop

    assert sched._allocate_pending() == 1  # refills the freed slot
    assert not event.is_set()
    await asyncio.sleep(0)  # run anything deferred
    assert not event.is_set()  # no late set on a full cluster
    assert len(sched._pending_tasks) == 997


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True], ids=["backfill", "strict"])
@pytest.mark.parametrize("policy", ["fifo_policy", "large_resource_policy"])
async def test_heterogeneous_tasks(policy, strict):
    """Mixed task sizes through the real monitor loop, completing in random order.

    Checks the cluster is never oversubscribed, every task runs exactly once, strict
    mode starts tasks in exact priority order, and everything is returned at the end.
    In strict mode it also bounds the number of allocate() attempts: a pass stops at
    the head, so attempts stay O(1) per task. Backfill is deliberately unbounded --
    to find a task that fits it may scan past big tasks parked at the front, which is
    O(pending) per dispatch with mixed sizes (accepted; per-shape queues were judged
    not worth the complexity given per-task affinity).
    """
    ncpus, ntasks = 4, 300
    rng = random.Random(0)
    sizes = {f"t{i}": rng.choice([1, 1, 1, 2, 3, 4]) for i in range(ntasks)}

    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {},
        _job_resource(ncpus=ncpus),
        policy=policy,
        policy_config=PolicyConfig(strict_priority=strict),
    )
    attempts = 0
    real_allocate = sched.cluster.allocate

    def counting_allocate(req):
        nonlocal attempts
        attempts += 1
        return real_allocate(req)

    sched.cluster.allocate = counting_allocate
    for tid, ppn in sizes.items():
        sched.add_task(_task(tid, ppn=ppn))

    started, running, in_use = [], set(), 0

    def take(tid):
        nonlocal in_use
        assert tid not in running and tid not in started
        started.append(tid)
        running.add(tid)
        in_use += sizes[tid]
        assert in_use <= ncpus, f"oversubscribed: {in_use} > {ncpus} cpus"

    sched.start_monitoring()
    try:
        while len(started) < ntasks:
            if not running:
                # Nothing running: the monitor must make progress on its own.
                tid, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
                take(tid)
            for _ in range(3):  # let the monitor run its pass
                await asyncio.sleep(0)
            while not sched.ready_tasks.empty():
                take(sched.ready_tasks.get_nowait()[0])
            if running:
                victim = rng.choice(sorted(running))
                running.discard(victim)
                in_use -= sizes[victim]
                sched.free(victim, TaskStatus.SUCCESS)
        for tid in sorted(running):
            sched.free(tid, TaskStatus.SUCCESS)
        await asyncio.wait_for(sched.wait_for_completion(), timeout=5.0)
    finally:
        await sched.stop_monitoring()

    assert sorted(started) == sorted(sizes)
    assert sched._pending_tasks.empty() and not sched._running_tasks
    assert sched.cluster.get_status()[0] == ncpus
    if strict:
        if policy == "fifo_policy":
            expected = list(sizes)
        else:  # bigger first, submission order among equals
            expected = sorted(sizes, key=lambda t: (-sizes[t], int(t[1:])))
        assert started == expected
        assert attempts <= 5 * ntasks, f"{attempts} allocate attempts for {ntasks} tasks"


def test_min_req_running_minimum():
    sched = AsyncTaskScheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=8), policy="fifo_policy"
    )
    assert sched._find_min_resource_req() is None
    sched.add_task(_task("big", ppn=4))
    assert sched._find_min_resource_req().resources[0].cpu_count == 4
    sched.add_task(_task("small", ppn=2))
    assert sched._find_min_resource_req().resources[0].cpu_count == 2
    # Removing the small one leaves a stale (too low) minimum -- safe by design.
    assert sched.delete_task(sched.tasks["small"]) is True
    assert sched._find_min_resource_req().resources[0].cpu_count == 2
    assert sched.delete_task(sched.tasks["big"]) is True
    assert sched._pending_tasks.empty()
    assert sched._find_min_resource_req() is None


def test_seed_pending_resets_min_req():
    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {"a": _task("a", ppn=1), "b": _task("b", ppn=3)},
        _job_resource(ncpus=8),
        policy="large_resource_policy",
    )
    assert sched._find_min_resource_req().resources[0].cpu_count == 1
    sched._seed_pending(["b"])
    assert sched._find_min_resource_req().resources[0].cpu_count == 3
