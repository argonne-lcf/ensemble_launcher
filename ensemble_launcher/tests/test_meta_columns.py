"""Fixed-width task meta (`META_DTYPE`) and the column entry points policies can use.

Covers: Python types on the row path, (de)serialization edge cases, the store's
``meta_columns`` / ``build_update`` with uids in any order (including after a spill), and
that the vectorized built-in policies place tasks as their row versions do.
"""

import random
import warnings

import numpy as np
import pytest

from ensemble_launcher.comm.messages import Message, TaskUpdate
from ensemble_launcher.config import PolicyConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.ensemble.ensemble import META_DTYPE, STR_BYTES, meta_row_values
from ensemble_launcher.scheduler.policy import (
    BinPackingChildrenPolicy,
    ChildrenPolicy,
    SimpleSplitChildrenPolicy,
    prefers_columns,
    wids_from_groups,
)
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceCount
from ensemble_launcher.scheduler.state import Child
from ensemble_launcher.taskstore import create_task_store

pytestmark = pytest.mark.core


def _received(tasks):
    return Message.from_byte_array(TaskUpdate.from_tasks(tasks).to_byte_array())


# --------------------------------------------------------------------------- #
#  Encoding                                                                   #
# --------------------------------------------------------------------------- #


def test_row_path_returns_python_types():
    t = Task(task_id="t", nnodes=2, ppn=3, ngpus_per_process=0.5, executable="echo", tag="x")
    plain = Task(task_id="p", nnodes=1, ppn=1, executable="echo")
    upd = _received([t, plain])
    m = upd.meta(0)
    assert (m.task_id, m.nnodes, m.ppn, m.ngpus_per_process, m.tag) == ("t", 2, 3, 0.5, "x")
    assert type(m.nnodes) is int and type(m.task_id) is str
    assert m.executor_name is None
    worker_task = upd.task(1)
    assert worker_task.task_id == "p" and worker_task.executor_name is None
    assert worker_task.tag is None and type(worker_task.ngpus_per_process) is int


def test_long_task_id_is_truncated_with_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = Task(task_id="a" * (STR_BYTES + 8), nnodes=1, ppn=1, executable="echo")
    assert len(t.task_id) == STR_BYTES
    assert any("truncated" in str(w.message) for w in caught)


def test_empty_and_delete_only_updates_serialize():
    assert _received([]).ntasks == 0
    upd = TaskUpdate(deleted_uids=[b"d" * 16])
    back = Message.from_byte_array(upd.to_byte_array())
    assert back.deleted_uids == [b"d" * 16] and len(back.meta_columns()) == 0


def test_nbytes_counts_meta_bytes():
    upd = _received([Task(task_id=f"t{i}", nnodes=1, ppn=1, executable="echo") for i in range(10)])
    assert upd.nbytes >= 10 * META_DTYPE.itemsize


# --------------------------------------------------------------------------- #
#  Store                                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("spill", [False, True])
def test_store_gathers_handles_in_any_order(tmp_path, spill):
    store = create_task_store(
        "segment", node_id="t", hot_bytes=1 if spill else 1 << 30, spill_dir=str(tmp_path)
    )
    try:
        handles = []
        for seg in range(4):
            tasks = [
                Task(task_id=f"s{seg}t{i}", nnodes=1 + i % 3, ppn=1 + seg, executable="x", args=(i,))
                for i in range(20)
            ]
            handles += store.add(_received(tasks)).tolist()
        rng = random.Random(0)
        for _ in range(30):
            sample = rng.sample(handles, rng.randint(0, len(handles)))  # interleaved segments
            meta = store.meta_columns(sample)
            assert [meta_row_values(meta, i) for i in range(len(sample))] == [
                store[u]._values for u in sample
            ]
            upd = store.build_update(sample, deleted_uids=[b"d" * 16])
            assert upd.uids() == store.uids(sample) and upd.deleted_uids == [b"d" * 16]
            assert [upd.task(i).args for i in range(len(sample))] == [
                store.task(u).args for u in sample
            ]
    finally:
        store.close()


# --------------------------------------------------------------------------- #
#  Column entry points                                                        #
# --------------------------------------------------------------------------- #


def _children(k, cpus):
    return {
        w: Child(
            id=f"c{w}",
            wid=w,
            job_resource=JobResource(
                resources=[NodeResourceCount(ncpus=cpus[w % len(cpus)], ngpus=2)], nodes=[f"n{w}"]
            ),
        )
        for w in range(k)
    }


def _both(policy, tasks, children, ntask):
    """The row method's placement and the column method's, both as one wid per task."""
    store = create_task_store("segment", node_id="t")
    idx = store.add(_received(tasks)) if tasks else np.empty(0, dtype=np.int64)
    policy.bind_tasks(store)
    by_wid, _, unplaced = policy.get_children_tasks(idx, children, ntask=ntask)
    rows = wids_from_groups(idx, by_wid)
    assert set(np.asarray(unplaced, dtype=np.int64).tolist()) == set(idx[rows == -1].tolist())
    cols = policy.get_children_tasks_columns(idx, store.meta_columns(idx), children, ntask=ntask)
    store.close()
    return rows, np.asarray(cols, dtype=np.int64)


@pytest.mark.parametrize("seed", range(40))
def test_simple_split_columns_match_rows(seed):
    rng = random.Random(seed)
    k, ntask = rng.randint(1, 9), rng.choice([None, 1, 3, 50])
    children = _children(k, [rng.choice([1, 2, 4, 8]) for _ in range(3)])
    policy = SimpleSplitChildrenPolicy(PolicyConfig(nchildren=k))

    # One shape: identical assignment.
    same = [Task(task_id=f"t{i}", nnodes=1, ppn=2, executable="x") for i in range(rng.randint(0, 60))]
    rows, cols = _both(policy, same, children, ntask)
    assert cols.tolist() == rows.tolist()

    # Mixed shapes: same unplaced set without a cap, nothing lost, cap respected.
    mixed = [
        Task(task_id=f"t{i}", nnodes=1, ppn=rng.choice([1, 2, 4, 16]),
             ngpus_per_process=rng.choice([0, 0.5]), executable="x")
        for i in range(rng.randint(0, 60))
    ]  # fmt: skip
    rows, cols = _both(policy, mixed, children, ntask)
    assert len(cols) == len(mixed) and set(cols.tolist()) <= set(children) | {-1}
    if ntask is None:
        assert ((cols == -1) == (rows == -1)).all()
    else:
        assert all(int((cols == w).sum()) <= ntask for w in children)


@pytest.mark.parametrize("seed", range(20))
def test_bin_packing_columns_match_rows(seed):
    rng = random.Random(seed)
    k, ntask = rng.randint(1, 9), rng.choice([None, 1, 3, 50])
    children = _children(k, [4])
    policy = BinPackingChildrenPolicy(PolicyConfig(nlevels=1))
    tasks = [Task(task_id=f"t{i}", nnodes=rng.randint(1, 4), ppn=1, executable="x")
             for i in range(rng.randint(0, 60))]  # fmt: skip
    rows, cols = _both(policy, tasks, children, ntask)
    assert cols.tolist() == rows.tolist()


def test_prefers_columns_follows_the_most_derived_override():
    class RowOnly(ChildrenPolicy):
        def get_children_resources(self, idx, nodes, level):
            return {}

        def get_children_tasks(self, idx, children, ntask=None, level=None, **kw):
            return {}, {}, []

    class RowOverrideOfColumnar(SimpleSplitChildrenPolicy):
        def get_children_tasks(self, idx, children, ntask=None, level=None, **kw):
            return {}, {}, []

    assert not prefers_columns(RowOnly(), "get_children_tasks", "get_children_tasks_columns")
    assert prefers_columns(
        SimpleSplitChildrenPolicy(PolicyConfig(nchildren=1)),
        "get_children_tasks", "get_children_tasks_columns",
    )  # fmt: skip
    assert not prefers_columns(
        RowOverrideOfColumnar(PolicyConfig(nchildren=1)),
        "get_children_tasks", "get_children_tasks_columns",
    )  # fmt: skip
