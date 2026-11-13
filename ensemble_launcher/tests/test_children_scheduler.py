"""AsyncChildrenScheduler and its SchedulerState: one record per child, tasks placed by a
chunked assigner onto per-child outboxes, and the state serialized as its own checkpoint."""

import asyncio
import logging
import socket

import pytest

from ensemble_launcher.comm.messages import Message, Status, TaskUpdate
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler import (
    AsyncChildrenScheduler,
    Child,
    ChildState,
    SchedulerState,
    policy_registry,
)
from ensemble_launcher.scheduler.policy import SimpleSplitChildrenPolicy
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

pytestmark = pytest.mark.core

R, F, S = ChildState.RUNNING, ChildState.FAILED, ChildState.SUCCESS


def _nodes(ncpus: int = 8) -> JobResource:
    node = NodeResourceList.from_config(SystemConfig(name="local", ncpus=ncpus))
    return JobResource(resources=[node], nodes=[socket.gethostname()])


def _update(n: int, prefix: str = "t", nnodes: int = 1):
    tasks = [
        Task(task_id=f"{prefix}{i}", nnodes=nnodes, ppn=1, executable="true") for i in range(n)
    ]
    return tasks, Message.from_byte_array(TaskUpdate.from_tasks(tasks).to_byte_array())


def _scheduler(nchildren: int = 2, policy: str = None, placement="push", chunk: int = 4096):
    config = LauncherConfig(
        policy_config=PolicyConfig(nlevels=1, nchildren=nchildren), assign_chunk_size=chunk
    )
    if policy is not None:
        config = config.model_copy(update={"children_scheduler_policy": policy})
    return AsyncChildrenScheduler(
        logging.getLogger("test"), _nodes(), config, node_id="main", placement=placement
    )


async def _placed(ntasks: int, **kw):
    """A started scheduler with two children and ``ntasks`` placed."""
    sched = _scheduler(**kw)
    tasks, update = _update(ntasks)
    await sched.queue(update)
    sched.create_children(level=0, node_id="main")
    sched.start()
    await sched._input.join()
    return sched, [t.uid for t in tasks]


def _held(sched):
    return {u for c in sched.state.children.values() for u in c.uids}


def _drain(child):
    items = []
    while not child.outbox.empty():
        items.append(child.outbox.get_nowait())
    return items


def _one_place_rule(state: SchedulerState):
    """Every unfinished uid is in the pool or in exactly one child."""
    seen = list(state.unassigned)
    for child in state.children.values():
        seen.extend(child.uids)
    assert len(seen) == len(set(seen))


# --------------------------------------------------------------------------- #
#  SchedulerState on its own                                                  #
# --------------------------------------------------------------------------- #


def test_child_is_an_ordered_uid_set():
    c = Child(id="m.w0", wid=0, job_resource=None)
    c.add([b"b", b"a", b"b"])
    assert list(c.uids) == [b"b", b"a"] and len(c) == 2 and b"a" in c
    c.discard(b"b")
    c.discard(b"missing")
    assert list(c.uids) == [b"a"]


def test_add_and_pop_keep_the_wid_index():
    state = SchedulerState(node_id="m")
    a, b = Child(id="m.w0", wid=0, job_resource=None), Child(id="m.w3", wid=3, job_resource=None)
    state.add_child(a)
    state.add_child(b)
    assert state.by_wid(3) is b and state.next_wid() == 4
    assert state.pop_child("m.w3") is b and state.pop_child("nope") is None
    assert state.next_wid() == 1
    with pytest.raises(KeyError):
        state.by_wid(3)


def test_moves_keep_one_place_rule():
    state = SchedulerState(node_id="m")
    a, b = Child(id="m.w0", wid=0, job_resource=None), Child(id="m.w1", wid=1, job_resource=None)
    state.add_child(a)
    state.add_child(b)
    uids = [bytes([i]) * 16 for i in range(6)]

    state.queue(uids, client_id="c")
    assert list(state.unassigned) == uids and state.ntasks_added == 6
    _one_place_rule(state)
    state.assign(a, uids[:3])
    state.assign(b, uids[3:5])
    assert list(state.unassigned) == uids[5:]
    _one_place_rule(state)

    assert state.finish_many([uids[0]], a) == ["c"] and uids[0] not in a
    assert state.finish_many([uids[5]]) == ["c"] and uids[5] not in state.unassigned  # never placed
    state.forget(uids[3])
    assert uids[3] not in b and uids[3] not in state.task_to_client
    _one_place_rule(state)

    held = list(a.uids)
    assert state.pop_child(a.id) is a and set(held) <= set(state.unassigned)
    _one_place_rule(state)
    state.add_child(a)  # claims its uids back from the pool, outbox holds all of them
    assert list(a.uids) == held and not set(held) & set(state.unassigned)
    assert _drain(a) == [held]
    _one_place_rule(state)
    assert not state.empty()


def test_add_child_claims_only_uids_still_in_the_pool():
    """A removed child's tasks placed elsewhere meanwhile are not taken back."""
    state = SchedulerState(node_id="m")
    a, b = Child(id="m.w0", wid=0, job_resource=None), Child(id="m.w1", wid=1, job_resource=None)
    state.add_child(a)
    state.add_child(b)
    uids = [bytes([i]) * 16 for i in range(4)]
    state.queue(uids)
    state.assign(a, uids)
    state.pop_child(a.id)
    state.assign(b, uids[:2])  # placed elsewhere while `a` was out
    state.add_child(a)
    assert list(a.uids) == uids[2:] and _drain(a) == [uids[2:]]
    _one_place_rule(state)


def test_aggregate_status_sums_children():
    state = SchedulerState(node_id="m")
    for wid, n in enumerate((2, 3)):
        c = Child(id=f"m.w{wid}", wid=wid, job_resource=None)
        c.status = Status(nrunning_tasks=n)
        state.add_child(c)
    state.add_child(Child(id="m.w2", wid=2, job_resource=None))  # no status yet
    assert state.aggregate_status().nrunning_tasks == 5


def test_json_round_trip():
    state = SchedulerState(node_id="main.m0", level=1, nodes=_nodes())
    c = Child(id="main.m0.w4", wid=4, job_resource=_nodes(4), state=R,
              task_executor_name="async_mpi", child_class="AsyncWorker")  # fmt: skip
    state.queue([b"\x01" * 16, b"\x02" * 16])
    state.add_child(c)
    state.assign(c, [b"\x01" * 16, b"\x02" * 16])
    c.status, c.allocated = Status(nrunning_tasks=1), True
    state.task_to_client[b"\x02" * 16] = "client-7"
    state.unassigned[b"\x03" * 16] = None

    back = SchedulerState.from_json(state.to_json())

    assert (back.node_id, back.level) == ("main.m0", 1)
    assert back.nodes.serialize() == state.nodes.serialize()
    assert back.task_to_client == {b"\x02" * 16: "client-7"}
    (r,) = back.children.values()
    assert back.by_wid(4) is r
    assert (r.id, r.wid, list(r.uids)) == ("main.m0.w4", 4, [b"\x01" * 16, b"\x02" * 16])
    assert (r.task_executor_name, r.child_class) == ("async_mpi", "AsyncWorker")
    assert r.job_resource.serialize() == c.job_resource.serialize()
    # Runtime-only fields are not written: a restored child starts over.
    assert r.state is ChildState.NOTREADY and r.status is None and not r.allocated
    assert not r.ready.is_set() and not r.done.is_set() and r.outbox.empty()
    assert not back.unassigned and back.ntasks_added == 0


# --------------------------------------------------------------------------- #
#  The scheduler: queue in, assigner, outboxes                                #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_layout_and_placement():
    sched, uids = await _placed(10)
    children = sched.state.children
    assert list(children) == ["main.w0", "main.w1"]
    assert all(c.state is ChildState.READY and c.allocated for c in children.values())
    assert _held(sched) == set(uids) and not sched.state.unassigned
    assert sched.state.ntasks_added == 10
    _one_place_rule(sched.state)


@pytest.mark.asyncio
async def test_one_outbox_item_per_child_per_update():
    """However many chunks an update takes, each child gets one list for it."""
    sched, uids = await _placed(100, chunk=7)
    items = {cid: _drain(c) for cid, c in sched.state.children.items()}
    assert all(len(v) == 1 for v in items.values())
    assert sorted(u for v in items.values() for u in v[0]) == sorted(uids)
    for cid, (item,) in items.items():
        assert item == list(sched.state.children[cid].uids)


@pytest.mark.asyncio
async def test_assigner_yields_between_chunks():
    sched = _scheduler(chunk=10)
    sched.create_children(level=0, node_id="main")
    sched.start()
    ticks = []

    async def ticker():
        while True:
            ticks.append(len(sched.state.unassigned))
            await asyncio.sleep(0)

    t = asyncio.create_task(ticker())
    _, update = _update(200)
    await sched.queue(update)
    await sched._input.join()
    t.cancel()
    # The ticker ran while placement was part-way through (some, not all, still pooled).
    assert any(0 < n < 200 for n in ticks)


@pytest.mark.asyncio
async def test_unplaced_tasks_are_reported_and_drained():
    sched = _scheduler()
    sched.create_children(level=0, node_id="main")
    sched.start()
    _, ok = _update(2, "ok")
    big, too_big = _update(1, "big", nnodes=5)  # more nodes than any child has
    await sched.queue(ok)
    await sched.queue(too_big)
    await sched._input.join()
    assert sched.unplaced.get_nowait() == [big[0].uid]
    assert big[0].uid in sched.state.unassigned and not sched.drained.is_set()
    sched.finish_tasks([big[0].uid])  # what the master's failure result does
    assert big[0].uid not in sched.state.unassigned


@pytest.mark.asyncio
async def test_drained_tracks_unfinished_tasks():
    sched = _scheduler()
    assert sched.drained.is_set()
    sched.create_children(level=0, node_id="main")
    sched.start()
    tasks, update = _update(4)
    await sched.queue(update)
    assert not sched.drained.is_set()
    await sched._input.join()
    assert not sched.drained.is_set()  # held by children
    for cid, c in sched.state.children.items():
        sched.finish_tasks(list(c.uids), cid)
    assert sched.drained.is_set()
    _, more = _update(1, "more")
    await sched.queue(more)
    assert not sched.drained.is_set()


@pytest.mark.asyncio
async def test_finished_tasks_leave_their_child():
    sched, uids = await _placed(10)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    first = list(a.uids)[:3]
    sched.state.task_to_client.update(dict.fromkeys(first[:2], "client-1"))

    clients = sched.finish_tasks(first, cid_a)

    assert clients == ["client-1", "client-1", None]
    assert not any(u in a for u in first) and len(a) + len(b) == len(uids) - 3
    assert not sched.state.task_to_client
    assert not any(u in sched.tasks for u in first)


@pytest.mark.asyncio
async def test_finish_without_clients_returns_none():
    """A sub-master's tasks all come from its parent: no per-uid client lookup."""
    sched, uids = await _placed(6)
    cid, child = next(iter(sched.state.children.items()))
    mine = list(child.uids)
    assert sched.finish_tasks(mine, cid) is None
    assert not any(u in child for u in mine) and not any(u in sched.tasks for u in mine)


@pytest.mark.asyncio
async def test_transitions_and_side_effects():
    sched, _ = await _placed(4)
    (cid_a, a), (cid_b, b) = sched.state.children.items()

    assert sched.transition(cid_a, R)
    assert a.ready.is_set() and not a.done.is_set()

    with pytest.raises(RuntimeError):
        sched.transition(cid_a, ChildState.READY)

    # SUCCESS while still holding tasks (e.g. placed after the child took Stop): the child is
    # emptied and, without requeue, they go to unplaced (the master fails them).
    first = list(a.uids)[0]
    sched.finish_tasks([first], cid_a)
    held = set(a.uids)
    free_before = sched.cluster.free_cpus
    assert sched.transition(cid_a, S)
    assert a.done.is_set() and not a.allocated and sched.cluster.free_cpus > free_before
    assert not a.uids and set(sched.unplaced.get_nowait()) == held
    assert not sched.state.all_done.is_set()

    sched.transition(cid_b, R)
    sched.transition(cid_b, ChildState.RECOVERING)
    assert sched.transition(cid_b, F) and b.done.is_set()
    assert sched.state.all_done.is_set()


@pytest.mark.asyncio
async def test_delete_task_leaves_its_child():
    sched, uids = await _placed(4)
    holder = next(c for c in sched.state.children.values() if uids[0] in c)
    sched.delete_task(uids[0])
    assert uids[0] not in holder and uids[0] not in sched.tasks


@pytest.mark.asyncio
async def test_remove_and_reinstate_child():
    """Recovery: the removed record keeps its tasks and comes back on its own resources,
    with an outbox holding every unfinished task."""
    sched, _ = await _placed(6)
    child_id, child = next(iter(sched.state.children.items()))
    _drain(child)  # pretend the first process got them
    sched.transition(child_id, R)
    held = list(child.uids)
    free_before = sched.cluster.free_cpus

    removed = sched.remove_child(child_id)
    assert removed is child and child_id not in sched.state.children
    assert set(held) <= set(sched.state.unassigned)
    assert sched.cluster.free_cpus > free_before

    assert sched.reinstate_child(removed)
    assert sched.state.children[child_id] is child and child.state is ChildState.READY
    assert list(child.uids) == held and not set(held) & set(sched.state.unassigned)
    assert _drain(child) == [held]  # all of it is resent to the new process
    assert sched.cluster.free_cpus == free_before and not child.ready.is_set()
    _one_place_rule(sched.state)


@pytest.mark.asyncio
async def test_checkpoint_restore():
    """Restart: initial tasks queued first, then set_state, then start(). Restored children
    claim what they held; the assigner places only the rest."""
    tasks, update = _update(6)
    sched, _ = await _placed(0)
    await sched.queue(update)
    await sched._input.join()
    first = next(iter(sched.state.children.values()))
    sched.state.task_to_client[next(iter(first.uids))] = "client-1"
    held = {cid: list(c.uids) for cid, c in sched.state.children.items()}
    raw = sched.get_state()
    assert isinstance(raw, str)

    # The restarted master is re-seeded with the same tasks plus one that arrived later.
    extra, extra_update = _update(1, "x")
    fresh = _scheduler()
    await fresh.queue(update)
    await fresh.queue(extra_update)
    fresh.set_state(SchedulerState.from_json(raw))
    fresh.start()
    await fresh._input.join()

    restored = fresh.state
    assert list(restored.children) == list(held)
    for cid, c in restored.children.items():
        assert c.state is ChildState.READY and c.allocated
        items = _drain(c)
        assert items[0] == held[cid]  # everything it held is resent first
        assert set(held[cid]) <= set(c.uids)
    assert _held(fresh) == {t.uid for t in tasks} | {extra[0].uid}
    assert not restored.unassigned
    assert restored.task_to_client == sched.state.task_to_client
    _one_place_rule(restored)


@pytest.mark.asyncio
async def test_pull_mode_places_on_request():
    sched = _scheduler(placement="pull")
    sched.create_children(level=0, node_id="main")
    sched.start()
    tasks, update = _update(10)
    await sched.queue(update)
    await sched._input.join()
    assert len(sched.state.unassigned) == 10 and not _held(sched)

    a = next(iter(sched.state.children))
    assert sched.request(a, 3) == 3
    (item,) = _drain(sched.state.children[a])
    assert len(item) == 3 and set(item) == set(sched.state.children[a].uids)
    assert len(sched.state.unassigned) == 7
    _one_place_rule(sched.state)


@pytest.mark.asyncio
async def test_requeue_places_pool_tasks_again():
    """A child that can't come back: its tasks are placed on the others."""
    sched, uids = await _placed(6)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    held = list(a.uids)
    sched.remove_child(cid_a)
    sched.requeue(held)
    await sched._input.join()
    assert set(held) <= set(b.uids) and not sched.state.unassigned
    await sched.stop()


@pytest.mark.asyncio
async def test_failed_with_requeue_moves_tasks_to_the_other_child():
    sched, uids = await _placed(6)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    held = set(a.uids)
    free_before = sched.cluster.free_cpus
    assert sched.transition(cid_a, F, requeue=True)  # from READY: never got running
    await sched._input.join()
    assert not a.uids and held <= set(b.uids) and not sched.state.unassigned
    assert a.done.is_set() and not a.allocated and sched.cluster.free_cpus > free_before
    assert sched.unplaced.empty() and not sched.state.all_done.is_set()
    _one_place_rule(sched.state)
    await sched.stop()


@pytest.mark.asyncio
async def test_failed_without_requeue_sends_tasks_to_unplaced():
    sched, uids = await _placed(6)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    held = set(a.uids)
    b_before = set(b.uids)
    sched.transition(cid_a, R)
    sched.transition(cid_a, F)
    await sched._input.join()
    assert not a.uids and set(b.uids) == b_before
    assert set(sched.unplaced.get_nowait()) == held
    assert held <= set(sched.state.unassigned)  # until the master fails them
    _one_place_rule(sched.state)
    await sched.stop()


@pytest.mark.asyncio
async def test_failed_with_no_live_child_left_goes_to_unplaced_and_sets_all_done():
    sched, uids = await _placed(6)
    for cid in list(sched.state.children):
        sched.transition(cid, F, requeue=True)
    await sched._input.join()
    unplaced = []
    while not sched.unplaced.empty():
        unplaced.extend(sched.unplaced.get_nowait())
    assert set(unplaced) == set(uids)
    assert sched.state.all_done.is_set()
    await sched.stop()


@pytest.mark.asyncio
async def test_failed_with_requeue_in_pull_mode_waits_in_the_pool():
    sched, uids = await _placed(0, placement="pull")
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    tasks, update = _update(4)
    await sched.queue(update)
    sched.transition(cid_a, R)
    assert sched.request(cid_a, None) == 4
    sched.transition(cid_a, F, requeue=True)
    assert not a.uids and {t.uid for t in tasks} <= set(sched.state.unassigned)
    assert sched.unplaced.empty()
    await sched.stop()


@pytest.mark.asyncio
async def test_placement_skips_finished_children():
    sched, _ = await _placed(0)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    a.state = S
    tasks, update = _update(4, prefix="late")
    await sched.queue(update)
    await sched._input.join()
    assert not a.uids and {t.uid for t in tasks} <= set(b.uids)
    await sched.stop()


@pytest.mark.asyncio
async def test_policy_receives_child_records():
    seen = {}

    @policy_registry.register("_records_policy", type="children_policy")
    class _Records(SimpleSplitChildrenPolicy):
        def get_children_tasks(self, uids, children, ntask=None, level=None, **kwargs):
            seen.update(children)
            return super().get_children_tasks(uids, children, ntask=ntask, level=level)

    sched, _ = await _placed(4, policy="_records_policy")
    assert seen and all(isinstance(c, Child) for c in seen.values())
    assert {c.wid: c for c in sched.state.children.values()} == seen


def test_layout_needs_tasks_flag():
    from ensemble_launcher.scheduler.policy import BinPackingChildrenPolicy

    assert BinPackingChildrenPolicy.layout_needs_tasks
    assert not SimpleSplitChildrenPolicy.layout_needs_tasks


# --- results out: queue_result ------------------------------------------------


def _results(uids, ok=True):
    from ensemble_launcher.comm.messages import IResultBatch, Result

    return IResultBatch.from_results(
        [Result(sender="w", uid=u, task_id=f"r{i}", success=ok) for i, u in enumerate(uids)]
    )


def _pieces(outbox):
    items = []
    while not outbox.empty():
        items.append(outbox.get_nowait())
    return items


@pytest.mark.asyncio
async def test_queue_result_finishes_and_routes_by_client():
    sched = _scheduler()
    _, update = _update(2, "p")
    await sched.queue(update)
    _, client_update = _update(2, "c")
    await sched.queue(client_update, client_id="client-1")
    sched.create_children(level=0, node_id="main")
    sched.start()
    await sched._input.join()
    held = {c.id: list(c.uids) for c in sched.state.children.values()}
    child_id = next(cid for cid, uids in held.items() if uids)
    uids = held[child_id]

    sched.queue_result(_results(uids), child_id)

    assert not sched.state.children[child_id].uids  # finished: left its child
    parent = _pieces(sched.parent_outbox)
    client = _pieces(sched.client_outboxes.get("client-1", asyncio.Queue()))
    routed = {u for p in parent + client for u in p.uids()}
    assert routed == set(uids)
    for piece in client:
        assert all(u in client_update.uids() for u in piece.uids())
    for piece in parent:
        assert all(u in update.uids() for u in piece.uids())
    await sched.stop()


@pytest.mark.asyncio
async def test_queue_result_splits_a_mixed_batch_and_announces_new_clients_once():
    sched = _scheduler(nchildren=1)
    _, a = _update(2, "a")
    _, b = _update(2, "b")
    _, c = _update(1, "c")
    await sched.queue(a, client_id="client-1")
    await sched.queue(b, client_id="client-2")
    await sched.queue(c)
    sched.create_children(level=0, node_id="main")
    sched.start()
    await sched._input.join()
    (child,) = sched.state.children.values()
    uids = list(child.uids)

    sched.queue_result(_results(uids[:3]), child.id)
    sched.queue_result(_results(uids[3:]), child.id)

    assert sorted(u for p in _pieces(sched.client_outboxes["client-1"]) for u in p.uids()) == sorted(a.uids())
    assert sorted(u for p in _pieces(sched.client_outboxes["client-2"]) for u in p.uids()) == sorted(b.uids())
    assert [u for p in _pieces(sched.parent_outbox) for u in p.uids()] == c.uids()
    announced = []
    while not sched.new_clients.empty():
        announced.append(sched.new_clients.get_nowait())
    assert sorted(announced) == ["client-1", "client-2"]  # once each; never the parent
    assert sched.drained.is_set()
    await sched.stop()


@pytest.mark.asyncio
async def test_unfinished_lists_pool_then_children_once_each():
    sched = _scheduler()
    _, update = _update(6)
    await sched.queue(update)  # all in the pool: no assigner started
    sched.create_children(level=0, node_id="main")
    first = next(iter(sched.state.children.values()))
    sched.state.assign(first, update.uids()[:2])

    unfinished = sched.state.unfinished()

    assert unfinished[0] == (None, update.uids()[2:])
    assert unfinished[1] == (first.id, update.uids()[:2])
    assert len(unfinished) == 2
    flat = [u for _, uids in unfinished for u in uids]
    assert sorted(flat) == sorted(update.uids())

    # Failing them through queue_result empties the scheduler.
    for child_id, uids in unfinished:
        sched.queue_result(_results(uids, ok=False), child_id)
    assert sched.state.empty()
