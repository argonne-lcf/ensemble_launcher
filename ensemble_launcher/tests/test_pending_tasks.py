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

from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.scheduler import (
    AsyncTaskScheduler,
    PendingTaskHeap,
    PendingTaskQueue,
    TaskSchedulerState,
)
from ensemble_launcher.scheduler.policy import FIFOPolicy, LargeResourcePolicy
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

pytestmark = pytest.mark.core


def _make_scheduler(logger, tasks, nodes, policy="fifo_policy", policy_config=None, node_id="node"):
    """An AsyncTaskScheduler built the way the worker builds it (policy by name, from the
    config), seeded with ``tasks`` (a dict of Tasks)."""
    config = LauncherConfig(
        task_scheduler_policy=policy, policy_config=policy_config or PolicyConfig()
    )
    sched = AsyncTaskScheduler(logger, nodes, config, node_id=node_id)
    for task in tasks.values():
        sched.add_task(task)
    return sched


def _job_resource(ncpus: int = 4) -> JobResource:
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    return JobResource(resources=[sys_info], nodes=[socket.gethostname()])


def _uid(tid: str) -> bytes:
    """Schedulers key tasks by 16-byte uid; tests derive a readable one from the name."""
    return tid.encode().ljust(16, b"\0")


def _name(tid):
    return tid.rstrip(b"\0").decode() if isinstance(tid, bytes) else tid


def _task(tid: str, ppn: int = 1) -> Task:
    return Task(task_id=tid, uid=_uid(tid), nnodes=1, ppn=ppn, executable=echo, args=(tid,))


def _ids(entries):
    return [_name(tid) for _p, _s, tid in entries]


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
    sched = _make_scheduler(logging.getLogger("t"), {}, _job_resource())
    assert isinstance(sched.scheduler_policy, FIFOPolicy)
    assert isinstance(sched.state.pending_tasks, PendingTaskQueue)


def test_scheduler_picks_container_by_policy_name():
    log = logging.getLogger("t")
    assert isinstance(
        _make_scheduler(log, {}, _job_resource(), policy="large_resource_policy").state.pending_tasks,
        PendingTaskHeap,
    )


def test_fifo_scheduler_never_scores(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("get_score must not be called on the FIFO path")

    monkeypatch.setattr(FIFOPolicy, "get_score", boom)
    sched = _make_scheduler(
        logging.getLogger("t"),
        {t: _task(t) for t in ("a", "b")},
        _job_resource(),
        policy="fifo_policy",
    )
    assert sched.add_task(_task("c"))
    assert _ids(sched.state.pending_tasks.sorted_items()) == ["a", "b", "c"]
    assert sched.reprioritize_pending() == 3
    assert _ids(sched.state.pending_tasks.sorted_items()) == ["a", "b", "c"]


def test_policy_gets_the_live_scheduler_state(monkeypatch):
    """No snapshot is built: get_score receives the scheduler's own state object."""
    seen = []
    orig = LargeResourcePolicy.get_score

    def spy(self, task, scheduler_state=None):
        seen.append((scheduler_state, len(scheduler_state.pending_tasks)))
        return orig(self, task, scheduler_state)

    monkeypatch.setattr(LargeResourcePolicy, "get_score", spy)
    sched = _make_scheduler(
        logging.getLogger("t"),
        {"a": _task("a")},
        _job_resource(),
        policy="large_resource_policy",
    )
    sched.add_task(_task("b"))
    assert all(state is sched.state for state, _ in seen)
    assert [n for _, n in seen] == [0, 1]  # live: "a" is pending when "b" is scored


async def _drain_in_order(sched: AsyncTaskScheduler, n: int):
    order = []
    for _ in range(n):
        tid, _res = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        order.append(_name(tid))
        sched.free(tid, TaskStatus.SUCCESS)
    return order


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["fifo_policy", "large_resource_policy"])
async def test_single_slot_dispatch_order(policy):
    """One CPU: dispatch order is exactly the container's pop order."""
    ids = [f"t{i}" for i in range(20)]
    sched = _make_scheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=1), policy=policy
    )
    for t in ids:
        sched.add_task(_task(t))
    sched.start_monitoring()
    try:
        assert await _drain_in_order(sched, len(ids)) == ids
        await asyncio.wait_for(sched.wait_for_completion(), timeout=5.0)
        assert sched.state.pending_tasks.empty()
    finally:
        await sched.stop_monitoring()


@pytest.mark.asyncio
async def test_priority_dispatch_order():
    """Bigger tasks first under large_resource_policy, FIFO among equals."""
    sched = _make_scheduler(
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
    sched = _make_scheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=2), policy="fifo_policy"
    )
    sched.add_task(_task("hold", ppn=1))
    sched.start_monitoring()
    try:
        hold, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        assert _name(hold) == "hold"
        # One CPU left: "big" can't start, "small" can.
        sched.add_task(_task("big", ppn=2))
        sched.add_task(_task("small", ppn=1))
        small, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        assert _name(small) == "small"
        assert _ids(sched.state.pending_tasks.sorted_items()) == ["big"]
        sched.free(_uid("hold"), TaskStatus.SUCCESS)
        sched.free(_uid("small"), TaskStatus.SUCCESS)
        big, _ = await asyncio.wait_for(sched.ready_tasks.get(), timeout=5.0)
        assert _name(big) == "big"
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
    sched = _make_scheduler(
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
    assert len(sched.state.pending_tasks) == 997


@pytest.mark.asyncio
async def test_checkpoint_round_trip_and_restore():
    """The state is the checkpoint: running tasks are written as pending, and a restored
    scheduler re-queues only what did not finish, with clients and statuses intact."""
    ids = [f"t{i}" for i in range(5)]
    sched = _make_scheduler(
        logging.getLogger("t"), {t: _task(t) for t in ids}, _job_resource(ncpus=2), node_id="w0"
    )
    sched.cluster.set_event_loop(asyncio.get_running_loop())
    sched.state.task_to_client[_uid("t4")] = "client-1"
    assert sched._allocate_pending() == 2  # t0, t1 running
    sched.free(_uid("t0"), TaskStatus.SUCCESS)
    sched.free(_uid("t1"), TaskStatus.FAILED)
    sched._allocate_pending()  # t2, t3 running; t4 pending

    raw = sched.get_state()
    back = TaskSchedulerState.from_json(raw)
    assert back.node_id == "w0" and back.nodes.serialize() == sched.cluster.nodes.serialize()
    assert sorted(_name(u) for u in back.pending_tasks) == ["t2", "t3", "t4"]
    assert back.completed_tasks == {_uid("t0")} and back.failed_tasks == {_uid("t1")}
    assert back.task_to_client == {_uid("t4"): "client-1"} and not back.running_tasks

    # A restarted worker is seeded with every task again.
    fresh = _make_scheduler(
        logging.getLogger("t"), {t: _task(t) for t in ids}, _job_resource(ncpus=2), node_id="w0"
    )
    container = fresh.state.pending_tasks
    fresh.set_state(TaskSchedulerState.from_json(raw))
    assert fresh.state.pending_tasks is container  # the monitor parks on its event
    assert sorted(_name(u) for u in fresh.state.pending_tasks) == ["t2", "t3", "t4"]
    assert _uid("t0") not in fresh.tasks and _uid("t1") not in fresh.tasks
    assert fresh.tasks.get_status(_uid("t0")) is TaskStatus.SUCCESS
    assert fresh.state.task_to_client == {_uid("t4"): "client-1"}


@pytest.mark.asyncio
async def test_fail_unfinished_fails_pending_running_and_ready():
    """A stopping worker fails everything left: allocated-but-unsubmitted (still in
    ready_tasks), running and pending tasks, and frees every resource."""
    sched = _make_scheduler(logging.getLogger("t"), {}, _job_resource(ncpus=2))
    sched.cluster.set_event_loop(asyncio.get_running_loop())
    for i in range(5):
        sched.add_task(_task(f"t{i}"))
    assert sched._allocate_pending() == 2
    sched.ready_tasks.get_nowait()  # t0 submitted; t1 still waits in ready_tasks
    free_before = sched.cluster.free_cpus

    failed = sched.fail_unfinished()

    assert sorted(_name(u) for u in failed) == [f"t{i}" for i in range(5)]
    assert sched.ready_tasks.empty() and sched.state.pending_tasks.empty()
    assert not sched.state.unfinished()
    assert sched.cluster.free_cpus == free_before + 2
    assert {_name(u) for u in sched.state.failed_tasks} == {f"t{i}" for i in range(5)}
    assert all(sched.tasks.get_status(u) is TaskStatus.FAILED for u in failed)
    assert len(sched.tasks) == 0
    await asyncio.wait_for(sched.wait_for_completion(), timeout=1.0)


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

    sched = _make_scheduler(
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
                take(_name(tid))
            for _ in range(3):  # let the monitor run its pass
                await asyncio.sleep(0)
            while not sched.ready_tasks.empty():
                take(_name(sched.ready_tasks.get_nowait()[0]))
            if running:
                victim = rng.choice(sorted(running))
                running.discard(victim)
                in_use -= sizes[victim]
                sched.free(_uid(victim), TaskStatus.SUCCESS)
        for tid in sorted(running):
            sched.free(_uid(tid), TaskStatus.SUCCESS)
        await asyncio.wait_for(sched.wait_for_completion(), timeout=5.0)
    finally:
        await sched.stop_monitoring()

    assert sorted(started) == sorted(sizes)
    assert sched.state.pending_tasks.empty() and not sched.state.running_tasks
    assert sched.cluster.get_status()[0] == ncpus
    if strict:
        if policy == "fifo_policy":
            expected = list(sizes)
        else:  # bigger first, submission order among equals
            expected = sorted(sizes, key=lambda t: (-sizes[t], int(t[1:])))
        assert started == expected
        assert attempts <= 5 * ntasks, f"{attempts} allocate attempts for {ntasks} tasks"


def test_min_req_running_minimum():
    sched = _make_scheduler(
        logging.getLogger("t"), {}, _job_resource(ncpus=8), policy="fifo_policy"
    )
    assert sched._find_min_resource_req() is None
    sched.add_task(_task("big", ppn=4))
    assert sched._find_min_resource_req().resources[0].cpu_count == 4
    sched.add_task(_task("small", ppn=2))
    assert sched._find_min_resource_req().resources[0].cpu_count == 2
    # Removing the small one leaves a stale (too low) minimum -- safe by design.
    assert sched.delete_task(_uid("small")) is True
    assert sched._find_min_resource_req().resources[0].cpu_count == 2
    assert sched.delete_task(_uid("big")) is True
    assert sched.state.pending_tasks.empty()
    assert sched._find_min_resource_req() is None


def test_seed_pending_resets_min_req():
    sched = _make_scheduler(
        logging.getLogger("t"),
        {"a": _task("a", ppn=1), "b": _task("b", ppn=3)},
        _job_resource(ncpus=8),
        policy="large_resource_policy",
    )
    assert sched._find_min_resource_req().resources[0].cpu_count == 1
    sched._seed_pending([_uid("b")])
    assert sched._find_min_resource_req().resources[0].cpu_count == 3
