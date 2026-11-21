"""TaskStore behaviour, run against every registered implementation through the factory.

A future database-backed store gets these tests for free by being registered.
"""

import asyncio
import gc
import os

import numpy as np
import pytest

from ensemble_launcher.comm.messages import Message, TaskUpdate
from ensemble_launcher.ensemble import Task, TaskMeta
from ensemble_launcher.taskstore import NO_CLIENT, RELEASED, UNASSIGNED, create_task_store
from ensemble_launcher.taskstore.base import _REGISTRY

pytestmark = pytest.mark.core

STORES = sorted(_REGISTRY)


def _double(x):
    return 2 * x


def _received(prefix: str, n: int):
    """Tasks and the TaskUpdate a node would receive for them (encoded, then decoded)."""
    tasks = [
        Task(task_id=f"{prefix}{i}", nnodes=1, ppn=1 + i % 3, executable=_double, args=(i,))
        for i in range(n)
    ]
    return tasks, Message.from_byte_array(TaskUpdate.from_tasks(tasks).to_byte_array())


def _spill_files(tmp_path):
    return sorted(p.name for p in tmp_path.rglob("seg-*.bin"))


@pytest.fixture(params=STORES)
def make_store(request, tmp_path):
    stores = []

    def make(**kwargs):
        kwargs.setdefault("spill_dir", str(tmp_path))
        store = create_task_store(request.param, node_id="test", **kwargs)
        stores.append(store)
        return store

    yield make
    for store in stores:
        store.close()


def _h(store, tasks):
    """The handles of these tasks (via the uid -> handle inverse map)."""
    return store.lookup([t.uid for t in tasks]).tolist()


class TestBasics:
    def test_add_and_lookup(self, make_store):
        store = make_store()
        tasks, update = _received("a", 10)
        idx = store.add(update)
        assert idx.dtype == np.int64 and len(idx) == 10
        assert (np.diff(idx) > 0).all()  # increasing, in row order
        assert store.lookup([t.uid for t in tasks]).tolist() == idx.tolist()
        assert store.uids(idx) == [t.uid for t in tasks]
        assert len(store) == 10 and set(store) == set(idx.tolist())
        assert store.all().tolist() == idx.tolist()
        meta = store[idx[4]]
        assert isinstance(meta, TaskMeta) and meta.task_id == "a4" and meta.ppn == 2
        task = store.task(idx[4])
        assert task.task_id == "a4" and task.executable(task.args[0]) == 8

    def test_resend_is_skipped(self, make_store):
        store = make_store()
        tasks, update = _received("a", 5)
        store.add(update)
        assert len(store.add(update)) == 0
        more, update2 = _received("b", 2)
        mixed = TaskUpdate.concat([update.select([0, 1]), update2])
        assert store.uids(store.add(mixed)) == [t.uid for t in more]
        assert len(store) == 7

    def test_unknown(self, make_store):
        store = make_store()
        assert store.lookup([b"x" * 16]).tolist() == [-1]
        assert 12345 not in store and -1 not in store
        with pytest.raises(KeyError):
            store[12345]

    def test_build_update_keeps_order_across_segments(self, make_store):
        store = make_store()
        a, ua = _received("a", 4)
        b, ub = _received("b", 4)
        store.add(ua)
        store.add(ub)
        order = [b[2], a[0], a[1], b[0], a[3]]
        msg = store.build_update(_h(store, order), sender="m")
        msg = Message.from_byte_array(msg.to_byte_array())
        assert msg.sender == "m"
        assert msg.uids() == [t.uid for t in order]  # uids travel in the message
        assert [t.task_id for t in msg.to_tasks()] == ["b2", "a0", "a1", "b0", "a3"]

    @pytest.mark.parametrize("spill", [False, True])
    def test_release_across_segments(self, make_store, tmp_path, spill):
        """Interleaved, unknown and already released handles: each task leaves the store at
        once, and a segment's bytes go exactly when its last task does."""
        store = make_store(hot_bytes=1 if spill else 1 << 30)
        segs = [store.add(_received(p, 5)[1]).tolist() for p in "abc"]
        held = lambda: len(_spill_files(tmp_path)) if spill else store.hot_bytes  # noqa: E731
        full = held()

        store.release([segs[0][0], segs[1][0], -1, segs[2][4]])
        assert len(store) == 12 and segs[0][0] not in store and segs[1][1] in store
        assert store.assigned([segs[0][0], segs[1][1]]).tolist() == [RELEASED, UNASSIGNED]
        assert held() == full  # no segment is empty yet

        store.release(segs[0][1:] + [segs[1][2]])  # a's last rows
        assert all(h not in store for h in segs[0]) and len(store) == 7
        assert 0 < held() < full  # segment a went
        assert store.assigned(segs[0]).tolist() == [RELEASED] * 5  # its segment is gone
        assert store[segs[1][1]].task_id == "b1" and store.task(segs[2][0]).args == (0,)

        store.release([segs[0][0]])  # already released, segment gone: ignored
        store.release(segs[1][1:2] + segs[1][3:] + segs[2][:4])
        assert len(store) == 0 and held() == 0

    def test_release_forgets_the_uid(self, make_store):
        store = make_store()
        tasks, update = _received("a", 3)
        idx = store.add(update)
        store.release(idx[:1])
        store.release(idx[:1])
        assert store.lookup([tasks[0].uid]).tolist() == [-1] and len(store) == 2
        # The same uid sent again is a new task with a new handle.
        again = store.add(update)
        assert len(again) == 1 and int(again[0]) not in idx.tolist()

    @pytest.mark.parametrize("by_uid", [False, True])
    def test_repeats_in_one_release_count_once(self, make_store, by_uid):
        """The same task twice in one call (e.g. a forwarded duplicate result) releases it
        once: the counts and the segment's remaining rows stay right."""
        store = make_store()
        tasks, update = _received("a", 3)
        idx = store.add(update).tolist()
        store.assign(idx[:2], 4)
        twice = [idx[0], idx[0], idx[2]]
        if by_uid:
            store.release(None, [tasks[0].uid, tasks[0].uid, tasks[2].uid, b"z" * 16])
        else:
            store.release(twice)
        assert len(store) == 1 and store.count(4) == 1 and store.count(UNASSIGNED) == 0
        assert store[idx[1]].task_id == "a1"  # the segment is still there
        store.release([idx[1]])
        assert len(store) == 0 and not store._segments

    def test_no_per_task_objects_tracked(self, make_store):
        """Holding tasks must not mean holding GC-tracked objects per task."""
        store = make_store()
        gc.collect()
        before = len(gc.get_objects())
        tasks, update = _received("a", 2000)
        del tasks  # the user's Task objects; the node never has them
        store.add(update)
        del update
        gc.collect()
        assert len(gc.get_objects()) - before < 200


class TestAssignment:
    def test_assign_claims_only_unassigned(self, make_store):
        store = make_store()
        a = store.add(_received("a", 4)[1])
        b = store.add(_received("b", 4)[1])
        assert store.count(UNASSIGNED) == 8
        moved = store.assign([a[0], b[1], a[2]], 3)
        assert moved.tolist() == [a[0], b[1], a[2]]  # in the order given
        assert store.assign([a[0], a[1]], 5).tolist() == [a[1]]  # a[0] is already on 3
        assert store.assigned([a[0], a[1], a[3]]).tolist() == [3, 5, UNASSIGNED]
        assert (store.count(3), store.count(5), store.count(UNASSIGNED)) == (3, 1, 4)
        assert store.assigned_to(3).tolist() == [a[0], a[2], b[1]]  # insertion order
        assert store.assigned_to(UNASSIGNED).tolist() == [a[3], b[0], b[2], b[3]]

    def test_reassign_and_groups(self, make_store):
        store = make_store()
        a = store.add(_received("a", 6)[1])
        store.assign(a[:2], 0)
        store.assign(a[2:4], 1)
        groups = {tag: idx.tolist() for tag, idx in store.groups()}
        assert list(groups) == [UNASSIGNED, 0, 1]
        assert groups == {UNASSIGNED: a[4:].tolist(), 0: a[:2].tolist(), 1: a[2:4].tolist()}
        assert store.reassign(1, UNASSIGNED).tolist() == a[2:4].tolist()
        assert store.count(1) == 0 and store.count(UNASSIGNED) == 4
        store.release(a[:1])
        assert store.count(0) == 1 and store.assigned_to(0).tolist() == [a[1]]
        assert store.assign(a[:1], 2).tolist() == []  # released: not claimable

    def test_assignment_survives_spill(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a = store.add(_received("a", 5)[1])
        store.assign(a[1:3], 7)
        assert _spill_files(tmp_path)
        store.assign(a[3:], 8)  # writes go to the in-memory column, not the mapping
        assert store.assigned(a).tolist() == [UNASSIGNED, 7, 7, 8, 8]
        assert store[a[4]].task_id == "a4"


class TestClients:
    def test_client_codes(self, make_store):
        store = make_store()
        assert not store.has_clients()
        a = store.add(_received("a", 2)[1])
        b = store.add(_received("b", 2)[1], client_id="client-1")
        c = store.add(_received("c", 1)[1], client_id="client-2")
        assert store.has_clients()
        codes = store.clients(np.concatenate([a, b, c])).tolist()
        assert codes[:2] == [NO_CLIENT, NO_CLIENT] and codes[2] == codes[3] != codes[4]
        assert [store.client_name(x) for x in codes] == [None, None, "client-1", "client-1", "client-2"]

    def test_build_failed_results(self, make_store):
        store = make_store()
        a, ua = _received("a", 3)
        b, ub = _received("b", 3)
        ia, ib = store.add(ua), store.add(ub)
        store.release(ia[1:2])
        batch, done = store.build_failed_results([ib[0], ia[1], ia[2]], "m", "boom")
        assert done.tolist() == [ia[2], ib[0]]  # the released one is skipped
        batch = Message.from_byte_array(batch.to_byte_array())
        results = batch.to_results()
        assert batch.sender == "m" and batch.uids() == store.uids(done)
        assert [(r.task_id, r.success, r.exception) for r in results] == [
            ("a2", False, "boom"),
            ("b0", False, "boom"),
        ]


class TestSpilling:
    def test_spills_over_budget_and_still_reads(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a, ua = _received("a", 20)
        b, ub = _received("b", 20)
        store.add(ua)
        store.add(ub)
        assert len(_spill_files(tmp_path)) == 2
        assert store.hot_bytes == 0
        assert store[_h(store, a)[7]].task_id == "a7"
        assert store.task(_h(store, b)[19]).args == (19,)
        msg = store.build_update(_h(store, [b[1], a[2]]))
        assert [t.task_id for t in msg.to_tasks()] == ["b1", "a2"]

    def test_delivered_segments_go_first(self, make_store, tmp_path):
        _, probe = _received("p", 10)
        store = make_store(hot_bytes=int(probe.nbytes * 1.5))
        a, ua = _received("a", 10)
        b, ub = _received("b", 10)
        store.add(ua)
        store.mark_delivered(_h(store, a))
        store.add(ub)  # over budget: the delivered segment (a) is offloaded, not the new one
        assert _spill_files(tmp_path) == ["seg-0.bin"]

    def test_files_removed_on_release_and_close(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a, ua = _received("a", 5)
        b, ub = _received("b", 5)
        store.add(ua)
        store.add(ub)
        store.release(_h(store, a))
        assert _spill_files(tmp_path) == ["seg-1.bin"]
        store.close()
        assert _spill_files(tmp_path) == []
        assert not any(tmp_path.iterdir())

    def test_unwritable_dir_keeps_tasks_in_memory(self, make_store, tmp_path):
        store = make_store(hot_bytes=1, spill_dir=str(tmp_path / "missing" / "dir"))
        a, ua = _received("a", 5)
        store.add(ua)
        assert store[_h(store, a)[3]].task_id == "a3"
        assert store.hot_bytes > 0

    @pytest.mark.asyncio
    async def test_spill_runs_off_the_event_loop(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a, ua = _received("a", 2000)
        ha = store.add(ua).tolist()  # schedules the write on an executor thread and returns
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        t = asyncio.create_task(ticker())
        # Reads work while the write is in flight and after it lands.
        assert store[ha[1999]].task_id == "a1999"
        for _ in range(200):
            await asyncio.sleep(0.005)
            if _spill_files(tmp_path) and store.hot_bytes == 0:
                break
        t.cancel()
        assert store.hot_bytes == 0 and ticks > 0
        assert store.task(ha[1234]).args == (1234,)
        store.release(ha)
        for _ in range(100):
            await asyncio.sleep(0.005)
            if not _spill_files(tmp_path):
                break
        assert _spill_files(tmp_path) == []
