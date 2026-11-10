"""TaskStore behaviour, run against every registered implementation through the factory.

A future database-backed store gets these tests for free by being registered.
"""

import asyncio
import gc
import os

import pytest

from ensemble_launcher.comm.messages import Message, TaskUpdate
from ensemble_launcher.ensemble import Task, TaskMeta, TaskStatus
from ensemble_launcher.taskstore import create_task_store
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


class TestBasics:
    def test_add_and_lookup(self, make_store):
        store = make_store()
        tasks, update = _received("a", 10)
        uids = store.add(update)
        assert uids == [t.uid for t in tasks]
        assert len(store) == 10 and set(store) == set(uids)
        meta = store[uids[4]]
        assert isinstance(meta, TaskMeta) and meta.task_id == "a4" and meta.ppn == 2
        task = store.task(uids[4])
        assert task.task_id == "a4" and task.executable(task.args[0]) == 8

    def test_resend_is_skipped(self, make_store):
        store = make_store()
        tasks, update = _received("a", 5)
        store.add(update)
        assert store.add(update) == []
        more, update2 = _received("b", 2)
        mixed = TaskUpdate.concat([update.select([0, 1]), update2])
        assert store.add(mixed) == [t.uid for t in more]
        assert len(store) == 7

    def test_unknown_uid(self, make_store):
        store = make_store()
        assert b"x" * 16 not in store
        with pytest.raises(KeyError):
            store[b"x" * 16]

    def test_build_update_keeps_order_across_segments(self, make_store):
        store = make_store()
        a, ua = _received("a", 4)
        b, ub = _received("b", 4)
        store.add(ua)
        store.add(ub)
        order = [b[2].uid, a[0].uid, a[1].uid, b[0].uid, a[3].uid]
        msg = Message.from_byte_array(store.build_update(order, sender="m").to_byte_array())
        assert msg.sender == "m"
        assert msg.uids() == order
        assert [t.task_id for t in msg.to_tasks()] == ["b2", "a0", "a1", "b0", "a3"]

    def test_status_survives_release(self, make_store):
        store = make_store()
        tasks, update = _received("a", 3)
        uids = store.add(update)
        assert store.get_status(uids[0]) is None
        store.set_status(uids[:2], TaskStatus.SUCCESS)
        store.release(uids[:2])
        assert uids[0] not in store and uids[2] in store
        assert store.get_status(uids[0]) is TaskStatus.SUCCESS

    def test_release_is_idempotent(self, make_store):
        store = make_store()
        _, update = _received("a", 3)
        uids = store.add(update)
        store.release(uids[:1])
        store.release(uids[:1])
        store.release([b"y" * 16])  # unknown uids are ignored
        assert len(store) == 2

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


class TestSpilling:
    def test_spills_over_budget_and_still_reads(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a, ua = _received("a", 20)
        b, ub = _received("b", 20)
        store.add(ua)
        store.add(ub)
        assert len(_spill_files(tmp_path)) == 2
        assert store.hot_bytes == 0
        assert store[a[7].uid].task_id == "a7"
        assert store.task(b[19].uid).args == (19,)
        msg = store.build_update([b[1].uid, a[2].uid])
        assert [t.task_id for t in msg.to_tasks()] == ["b1", "a2"]

    def test_delivered_segments_go_first(self, make_store, tmp_path):
        _, probe = _received("p", 10)
        store = make_store(hot_bytes=int(probe.nbytes * 1.5))
        a, ua = _received("a", 10)
        b, ub = _received("b", 10)
        store.add(ua)
        store.mark_delivered([t.uid for t in a])
        store.add(ub)  # over budget: the delivered segment (a) is offloaded, not the new one
        assert _spill_files(tmp_path) == ["seg-0.bin"]

    def test_files_removed_on_release_and_close(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a, ua = _received("a", 5)
        b, ub = _received("b", 5)
        store.add(ua)
        store.add(ub)
        store.release([t.uid for t in a])
        assert _spill_files(tmp_path) == ["seg-1.bin"]
        store.close()
        assert _spill_files(tmp_path) == []
        assert not any(tmp_path.iterdir())

    def test_unwritable_dir_keeps_tasks_in_memory(self, make_store, tmp_path):
        store = make_store(hot_bytes=1, spill_dir=str(tmp_path / "missing" / "dir"))
        a, ua = _received("a", 5)
        store.add(ua)
        assert store[a[3].uid].task_id == "a3"
        assert store.hot_bytes > 0

    @pytest.mark.asyncio
    async def test_spill_runs_off_the_event_loop(self, make_store, tmp_path):
        store = make_store(hot_bytes=1)
        a, ua = _received("a", 2000)
        store.add(ua)  # schedules the write on an executor thread and returns
        ticks = 0

        async def ticker():
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        t = asyncio.create_task(ticker())
        # Reads work while the write is in flight and after it lands.
        assert store[a[1999].uid].task_id == "a1999"
        for _ in range(200):
            await asyncio.sleep(0.005)
            if _spill_files(tmp_path) and store.hot_bytes == 0:
                break
        t.cancel()
        assert store.hot_bytes == 0 and ticks > 0
        assert store.task(a[1234].uid).args == (1234,)
        store.release([x.uid for x in a])
        for _ in range(100):
            await asyncio.sleep(0.005)
            if not _spill_files(tmp_path):
                break
        assert _spill_files(tmp_path) == []
