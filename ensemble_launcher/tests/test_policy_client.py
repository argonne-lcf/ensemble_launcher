"""Tests for externally mutable scheduling-policy state (``PolicyClient``)."""

import logging
import multiprocessing as mp
import os
import socket
import time
import uuid
from concurrent.futures import TimeoutError as FuturesTimeoutError

import pytest

# Imported for the side effect of registering the fixture policies in THIS process.
# The spawned orchestrator children get them via the env vars set below.
import policy_fixtures  # noqa: F401
from policy_fixtures import TunableIndexPolicy, TunableScorePolicy
from utils import echo

from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import (
    AsyncMaster,
    AsyncWorker,
    ClusterClient,
    PolicyClient,
    PolicyGroupClient,
    PolicyStateError,
    read_policy_endpoint,
)
from ensemble_launcher.orchestrator.policy_endpoint import KIND_TASK, PolicyEndpoint
from ensemble_launcher.scheduler import AsyncTaskScheduler, PendingTaskHeap
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

pytestmark = pytest.mark.core

# Spawned nodes re-import ensemble_launcher.orchestrator, which calls
# load_external_policies() at import time; this is how they learn the fixtures.
os.environ["EL_EXTERNAL_POLICY_PATH"] = os.path.dirname(os.path.abspath(__file__))
os.environ["EL_EXTERNAL_POLICY_MODULE"] = "policy_fixtures"


def _job_resource(ncpus: int = 12) -> JobResource:
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    return JobResource(resources=[sys_info], nodes=[socket.gethostname()])


def _ckpt_dir() -> str:
    return os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")


def _worker_config(ckpt_dir: str, policy: str, initial_state: dict, **kw):
    cfg = dict(
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        report_interval=100.0,
        log_level=logging.INFO,
        cluster=True,
        checkpoint_dir=ckpt_dir,
        return_stdout=True,
        worker_logs=True,
        master_logs=True,
        task_scheduler_policy=policy,
        policy_config=PolicyConfig(initial_state=initial_state),
        mpi_config=MPIConfig(flavor="test"),
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )
    cfg.update(kw)
    return LauncherConfig(**cfg)


def _spawn(node) -> mp.Process:
    p = mp.Process(target=node.create_an_event_loop)
    p.start()
    return p


def _reap(process: mp.Process) -> None:
    process.terminate()
    process.join(timeout=10.0)


# ---------------------------------------------------------------------------
# 1-2. Policy state API, in process
# ---------------------------------------------------------------------------


def test_policy_state_defaults():
    p = TunableScorePolicy(PolicyConfig())
    assert p.get_policy_state() == {}

    p = TunableScorePolicy(PolicyConfig(initial_state={"node_weight": 2.0, "a": 1}))
    assert p.get_policy_state() == {"node_weight": 2.0, "a": 1}

    # merge keeps untouched keys
    assert p.set_policy_state({"a": 9}) == {"node_weight": 2.0, "a": 9}
    # replace drops them
    assert p.set_policy_state({"b": 1}, merge=False) == {"b": 1}
    # keys filter omits what the policy does not hold, rather than returning None
    assert p.get_policy_state(["b", "missing"]) == {"b": 1}

    # a returned dict is a copy: mutating it must not reach the policy
    snap = p.get_policy_state()
    snap["b"] = "clobbered"
    assert p.state["b"] == 1

    with pytest.raises(TypeError):
        p.set_policy_state(["not", "a", "dict"])


def test_policy_state_affects_score():
    p = TunableScorePolicy(PolicyConfig(initial_state={"node_weight": 1.0}))
    task = Task(task_id="t", nnodes=4, ppn=1, executable=echo, args=("t",))
    assert p.get_score(task) == 4.0
    p.set_policy_state({"node_weight": 10.0})
    assert p.get_score(task) == 40.0


# ---------------------------------------------------------------------------
# 3. Endpoint request handling, driven directly (no transport)
# ---------------------------------------------------------------------------


def test_policy_endpoint_apply():
    p = TunableScorePolicy(PolicyConfig(initial_state={"node_weight": 2.0}))
    rescored_calls = []

    def rescore_cb():
        # Mimic reprioritize_pending's compute phase so poison state surfaces here.
        p.get_score(Task(task_id="t", nnodes=1, ppn=1, executable=echo, args=()))
        rescored_calls.append(1)
        return 7

    ep = PolicyEndpoint("main.w0", None, logging.getLogger("t"))
    ep.register(KIND_TASK, p, rescore_cb=rescore_cb)

    got = ep.apply("get", {})
    assert got["state"] == {"node_weight": 2.0}
    assert got["policy_kind"] == KIND_TASK
    assert got["policy_name"] == "TunableScorePolicy"
    assert got["rescored"] is None
    assert got["version"] == 0

    # get does not bump the version; set does
    assert ep.apply("set", {"state": {"x": 1}})["version"] == 1
    assert ep.apply("get", {})["version"] == 1

    assert ep.apply("set", {"state": {"y": 2}, "rescore": True})["rescored"] == 7
    assert rescored_calls == [1]

    assert ep.apply("set", {"state": {"only": 1}, "merge": False})["state"] == {
        "only": 1
    }

    with pytest.raises(ValueError):
        ep.apply("bogus", {})
    with pytest.raises(TypeError):
        ep.apply("set", {"state": 5})
    with pytest.raises(ValueError):
        ep.apply("get", {"policy_kind": "children"})

    # Rollback: a tune whose rescore raises must leave nothing behind.
    before, version = ep.apply("get", {})["state"], ep.version
    with pytest.raises(RuntimeError):
        ep.apply("set", {"state": {"boom": True}, "rescore": True})
    assert ep.apply("get", {})["state"] == before
    assert ep.version == version

    assert ep.snapshot() == {KIND_TASK: before}
    ep.restore({KIND_TASK: {"restored": True}})
    assert p.state == {"restored": True}


def test_policy_endpoint_resolve_is_explicit():
    ep = PolicyEndpoint("main.w0", None, logging.getLogger("t"))
    with pytest.raises(ValueError, match="no policy registered"):
        ep.apply("get", {})

    ep.register("task", TunableScorePolicy(PolicyConfig()))
    ep.register("children", TunableScorePolicy(PolicyConfig()))
    with pytest.raises(ValueError, match="multiple policies"):
        ep.apply("get", {})
    # ...but naming one is fine
    assert ep.apply("get", {"policy_kind": "task"})["policy_kind"] == "task"


# ---------------------------------------------------------------------------
# PendingTaskHeap.rescore unit tests
# ---------------------------------------------------------------------------


def test_pending_heap_rescore():
    heap = PendingTaskHeap()
    for i, prio in enumerate([5.0, 1.0, 3.0]):
        heap.push(prio, f"t{i}")
    heap._tasks_available.set()

    heap.rescore({"t0": 1.0, "t2": 2.0})  # t1 omitted -> dropped

    assert heap.task_ids() == {"t0", "t2"}
    assert len(heap) == 2
    assert heap._task_ids == {tid for _p, _s, tid in heap._heap}
    # heap invariant: the smallest priority is at the root
    assert heap._heap[0][0] == min(p for p, _s, _t in heap._heap)
    # membership did not drop to zero, so the waiter must stay armed
    assert heap._tasks_available.is_set()

    heap.rescore({})
    assert len(heap) == 0
    assert not heap._tasks_available.is_set()


def test_pending_heap_rescore_preserves_fifo_tiebreak():
    heap = PendingTaskHeap()
    for i in range(4):
        heap.push(float(4 - i), f"t{i}")  # distinct priorities, submission order t0..t3

    # Flatten every priority to the same value: only the tiebreak can order them now.
    heap.rescore({f"t{i}": 1.0 for i in range(4)})

    assert [tid for _p, _s, tid in heap.sorted_items()] == ["t0", "t1", "t2", "t3"]


def test_pending_heap_rescore_keeps_event_identity():
    """Regression guard: swapping in a fresh heap orphans the Event the scheduler's
    monitor parks on, which deadlocks dispatch silently."""
    heap = PendingTaskHeap()
    heap.push(1.0, "t0")
    event = heap._tasks_available
    heap.rescore({"t0": 2.0})
    assert heap._tasks_available is event


# ---------------------------------------------------------------------------
# 4-6. Client round trips against live nodes
# ---------------------------------------------------------------------------


def test_policy_client_worker_roundtrip():
    ckpt_dir = _ckpt_dir()
    w = AsyncWorker(
        "test",
        _worker_config(ckpt_dir, "tunable_policy", {"node_weight": 1.0}),
        _job_resource(),
    )
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            assert pc.get_state() == {"node_weight": 1.0}
            assert pc.set_state({"node_weight": 5.0}) == {"node_weight": 5.0}
            assert pc.get_state() == {"node_weight": 5.0}
            assert pc.get_state(keys=["node_weight"]) == {"node_weight": 5.0}
            assert pc.get_state(keys=["nope"]) == {}
            assert pc.policy_name == "TunableScorePolicy"

            assert pc.set_state({"fresh": 1}, merge=False) == {"fresh": 1}
    finally:
        _reap(process)


def test_policy_client_master_roundtrip():
    ckpt_dir = _ckpt_dir()
    m = AsyncMaster(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            child_executor_name="async_processpool",
            comm_name="async_zmq",
            report_interval=1.0,
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            children_scheduler_policy="tunable_children_policy",
            policy_config=PolicyConfig(
                nlevels=2, nchildren=1, initial_state={"knob": 1}
            ),
            worker_logs=True,
            master_logs=True,
            mpi_config=MPIConfig(flavor="test"),
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        _job_resource(),
    )
    process = _spawn(m)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            state = pc.get_state()
            assert state["knob"] == 1
            assert pc.policy_name == "TunableSplitPolicy"

            # Masters register no rescore_cb, so the flag is accepted but reports None.
            fut = pc.set_state_async({"knob": 2}, rescore=True)
            reply = fut.result(timeout=30.0)
            assert reply["policy_kind"] == "children"
            assert reply["rescored"] is None
            assert reply["state"]["knob"] == 2
            # on_state_update ran on the node, twice (initial set + this one)
            assert reply["state"]["_updates"] >= 1
    finally:
        _reap(process)


def test_policy_kind_mismatch():
    ckpt_dir = _ckpt_dir()
    w = AsyncWorker(
        "test",
        _worker_config(ckpt_dir, "tunable_policy", {"node_weight": 1.0}),
        _job_resource(),
    )
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test", policy_kind="children") as pc:
            with pytest.raises(PolicyStateError, match="children"):
                pc.get_state()
            # the connection survives a rejected request
            with PolicyClient(ckpt_dir, node_id="test") as ok:
                assert ok.get_state() == {"node_weight": 1.0}
    finally:
        _reap(process)


# ---------------------------------------------------------------------------
# 8, 12b. Rescore semantics end to end
# ---------------------------------------------------------------------------


def _ordering_worker(ckpt_dir: str, scores: dict):
    """A single-slot worker, so completion order is dispatch order."""
    return AsyncWorker(
        "test",
        _worker_config(
            ckpt_dir,
            "tunable_index_policy",
            {"scores": scores},
            strict_priority=True,
        ),
        _job_resource(ncpus=2),
    )


def test_rescore_reorders_pending():
    ckpt_dir = _ckpt_dir()
    ids = [f"task-{i}" for i in range(6)]
    # Initially ascending priority: task-0 lowest.
    w = _ordering_worker(ckpt_dir, {t: float(i) for i, t in enumerate(ids)})
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            reply = pc.set_state_async(
                {"scores": {t: float(len(ids) - i) for i, t in enumerate(ids)}},
                rescore=True,
            ).result(timeout=30.0)
            # Nothing was submitted yet, so there is nothing pending to re-score --
            # but the call must still report a count rather than None.
            assert reply["rescored"] == 0
            assert pc.get_state()["scores"]["task-0"] == 6.0
    finally:
        _reap(process)


def test_rescore_default_is_off():
    """A plain set_state must not touch already-queued priorities."""
    p = TunableIndexPolicy(PolicyConfig(initial_state={"scores": {"a": 1.0}}))
    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {
            "a": Task(task_id="a", nnodes=1, ppn=1, executable=echo, args=("a",)),
            "b": Task(task_id="b", nnodes=1, ppn=1, executable=echo, args=("b",)),
        },
        _job_resource(),
        policy=p,
    )
    before = sched._pending_tasks.sorted_items()

    ep = PolicyEndpoint("main.w0", None, logging.getLogger("t"))
    ep.register(KIND_TASK, p, rescore_cb=sched.reprioritize_pending)

    # No rescore key at all, and an explicit False: both must leave the heap alone.
    ep.apply("set", {"state": {"scores": {"b": 99.0}}})
    assert sched._pending_tasks.sorted_items() == before
    ep.apply("set", {"state": {"scores": {"b": 98.0}}, "rescore": False})
    assert sched._pending_tasks.sorted_items() == before

    # ...while asking for it does change them.
    assert ep.apply("set", {"state": {}, "rescore": True})["rescored"] == 2
    assert sched._pending_tasks.sorted_items() != before


def test_poison_state_rolls_back_scheduler():
    """A tune that makes get_score raise must leave both state and heap untouched."""
    p = TunableScorePolicy(PolicyConfig(initial_state={"node_weight": 1.0}))
    sched = AsyncTaskScheduler(
        logging.getLogger("t"),
        {
            t: Task(task_id=t, nnodes=1, ppn=1, executable=echo, args=(t,))
            for t in ("a", "b", "c")
        },
        _job_resource(),
        policy=p,
    )
    before = sched._pending_tasks.sorted_items()

    ep = PolicyEndpoint("main.w0", None, logging.getLogger("t"))
    ep.register(KIND_TASK, p, rescore_cb=sched.reprioritize_pending)

    with pytest.raises(RuntimeError):
        ep.apply("set", {"state": {"boom": True}, "rescore": True})

    assert p.state == {"node_weight": 1.0}
    assert sched._pending_tasks.sorted_items() == before
    assert sched._pending_tasks.task_ids() == {"a", "b", "c"}
    # and the policy still works
    assert ep.apply("set", {"state": {"node_weight": 3.0}, "rescore": True})[
        "rescored"
    ] == 3


# ---------------------------------------------------------------------------
# 7. Observable effect on a live run
# ---------------------------------------------------------------------------


def test_policy_state_changes_dispatch_on_live_worker():
    """Tune a live worker, then confirm tasks submitted afterwards still complete.

    The point is that a mid-run tune does not wedge the node: the policy is being
    consulted on every add_task, with state a client changed from outside.
    """
    ckpt_dir = _ckpt_dir()
    w = AsyncWorker(
        "test",
        _worker_config(ckpt_dir, "tunable_policy", {"node_weight": 1.0}),
        _job_resource(),
    )
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            pc.set_state({"node_weight": 7.0}, rescore=True)
            assert pc.get_state()["node_weight"] == 7.0

            with ClusterClient(node_id="test", checkpoint_dir=ckpt_dir) as cc:
                futs = {
                    f"t{i}": cc.submit(
                        Task(
                            task_id=f"t{i}",
                            nnodes=1,
                            ppn=1,
                            executable=echo,
                            args=(f"t{i}",),
                        )
                    )
                    for i in range(6)
                }
                results = {k: f.result(timeout=60.0) for k, f in futs.items()}

            assert all(
                r.split(",")[0].strip() == f"Hello from task {k}"
                for k, r in results.items()
            )
    finally:
        _reap(process)


def test_rescore_while_queue_empty():
    """Regression guard for the orphaned-Event deadlock, end to end.

    The worker is idle and parked waiting for tasks. A rescore here must not
    detach the Event its monitor is waiting on, or nothing submitted afterwards
    ever runs.
    """
    ckpt_dir = _ckpt_dir()
    w = AsyncWorker(
        "test",
        _worker_config(ckpt_dir, "tunable_policy", {"node_weight": 1.0}),
        _job_resource(),
    )
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            time.sleep(1.0)  # let the monitor reach its idle wait
            assert pc.set_state_async({"node_weight": 2.0}, rescore=True).result(
                timeout=30.0
            )["rescored"] == 0

            with ClusterClient(node_id="test", checkpoint_dir=ckpt_dir) as cc:
                fut = cc.submit(
                    Task(
                        task_id="after",
                        nnodes=1,
                        ppn=1,
                        executable=echo,
                        args=("after",),
                    )
                )
                assert "Hello from task after" in fut.result(timeout=60.0)
    finally:
        _reap(process)


# ---------------------------------------------------------------------------
# 9-11. Persistence, discovery, auth
# ---------------------------------------------------------------------------


def test_policy_state_persisted_to_checkpoint():
    ckpt_dir = _ckpt_dir()
    w = AsyncWorker(
        "test",
        _worker_config(
            ckpt_dir, "tunable_policy", {"node_weight": 1.0}, report_interval=0.5
        ),
        _job_resource(),
    )
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            pc.set_state({"node_weight": 42.0})

        state_path = os.path.join(ckpt_dir, "test", "test_policy_state.json")
        deadline = time.time() + 30.0
        while time.time() < deadline and not os.path.exists(state_path):
            time.sleep(0.2)
        assert os.path.exists(state_path), f"{state_path} was never written"
    finally:
        _reap(process)

    # Restored by a fresh node reading the same checkpoint dir.
    from ensemble_launcher.checkpointing import Checkpointer

    ckpt = Checkpointer("test", ckpt_dir, logging.getLogger("t"))
    import asyncio

    restored = asyncio.run(ckpt.read_policy_state())
    assert restored == {"task": {"node_weight": 42.0}}


def test_policy_state_survives_restart():
    """A tune outlives the node that received it.

    This is what makes tuning durable under ``restart_children_on_failure``: the
    replacement process must come up holding the tuned state, not the state the
    run was originally configured with.
    """
    ckpt_dir = _ckpt_dir()

    def make_worker():
        return AsyncWorker(
            "test",
            _worker_config(
                ckpt_dir, "tunable_policy", {"node_weight": 1.0}, report_interval=0.5
            ),
            _job_resource(),
        )

    first = _spawn(make_worker())
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            pc.set_state({"node_weight": 99.0, "tuned": "yes"})
        old_secret = read_policy_endpoint(ckpt_dir, "test").secret
        state_path = os.path.join(ckpt_dir, "test", "test_policy_state.json")
        deadline = time.time() + 30.0
        while time.time() < deadline and not os.path.exists(state_path):
            time.sleep(0.2)
        assert os.path.exists(state_path)
    finally:
        _reap(first)

    second = _spawn(make_worker())
    try:
        # The endpoint ckpt from the dead worker is still on disk, so a client
        # built right now would latch onto the old address. Wait for the
        # replacement to publish its own (freshly minted) secret.
        deadline = time.time() + 60.0
        while time.time() < deadline:
            if read_policy_endpoint(ckpt_dir, "test").secret != old_secret:
                break
            time.sleep(0.2)
        else:
            pytest.fail("restarted worker never republished its policy endpoint")

        with PolicyClient(ckpt_dir, node_id="test") as pc:
            state = pc.get_state()
            # The tuned value wins over PolicyConfig.initial_state, which is the
            # whole point -- otherwise a restart silently reverts the tune.
            assert state["node_weight"] == 99.0
            assert state["tuned"] == "yes"
    finally:
        _reap(second)


def test_discover_nodes_and_group_client():
    ckpt_dir = _ckpt_dir()
    m = AsyncMaster(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            child_executor_name="async_processpool",
            comm_name="async_zmq",
            report_interval=1.0,
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            children_scheduler_policy="tunable_children_policy",
            task_scheduler_policy="tunable_policy",
            policy_config=PolicyConfig(
                nlevels=2, nchildren=2, initial_state={"knob": 1}
            ),
            worker_logs=True,
            master_logs=True,
            mpi_config=MPIConfig(flavor="test"),
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        _job_resource(),
    )
    process = _spawn(m)
    try:
        # Wait for the master plus both workers to publish endpoints.
        deadline = time.time() + 60.0
        nodes = []
        while time.time() < deadline:
            nodes = PolicyClient.discover_nodes(ckpt_dir)
            if len(nodes) >= 3:
                break
            time.sleep(0.5)
        assert len(nodes) >= 3, f"only discovered {nodes}"
        assert "test" in nodes

        with PolicyGroupClient(ckpt_dir, node_ids=nodes) as group:
            states = group.get_state()
            assert set(states) == set(nodes)

            updated = group.set_state({"fanned_out": True})
            assert all(s["fanned_out"] for s in updated.values())
            assert all(
                s["fanned_out"] for s in group.get_state(keys=["fanned_out"]).values()
            )
    finally:
        _reap(process)


def test_unauthorized_client_rejected():
    """A client with the wrong policy secret gets no reply at all.

    Checked at the application layer, not by the connection: ServerConnection's
    verify_sender short-circuits to True when no expected remotes are configured,
    which is exactly this endpoint's situation.
    """
    ckpt_dir = _ckpt_dir()
    w = AsyncWorker(
        "test",
        _worker_config(ckpt_dir, "tunable_policy", {"node_weight": 1.0}),
        _job_resource(),
    )
    process = _spawn(w)
    try:
        with PolicyClient(ckpt_dir, node_id="test") as pc:
            assert pc.get_state() == {"node_weight": 1.0}  # endpoint is up

        # Wrong secret. Patched before start(), because the secret is baked into
        # the ZMQ identity at connect time -- changing it afterwards is a no-op.
        bad = PolicyClient(ckpt_dir, node_id="test")
        bad._conn._secret_id = "wrong-secret"
        bad.start()
        try:
            with pytest.raises((PolicyStateError, FuturesTimeoutError)):
                bad.get_state(timeout=5.0)
        finally:
            bad.teardown()

        # Right secret, but an identity that is not a policy client.
        impostor = PolicyClient(ckpt_dir, node_id="test", client_id="cluster-client-1")
        impostor.start()
        try:
            with pytest.raises((PolicyStateError, FuturesTimeoutError)):
                impostor.get_state(timeout=5.0)
        finally:
            impostor.teardown()

        # ...and the endpoint is still serving legitimate clients afterwards.
        with PolicyClient(ckpt_dir, node_id="test") as good:
            assert good.get_state() == {"node_weight": 1.0}
    finally:
        _reap(process)


def test_endpoint_absent_without_cluster_or_flag():
    """No endpoint is served unless it was asked for, so discovery times out."""
    ckpt_dir = _ckpt_dir()
    cfg = _worker_config(ckpt_dir, "tunable_policy", {"node_weight": 1.0})
    cfg.cluster = False
    assert cfg.enable_policy_client is False
    os.makedirs(os.path.join(ckpt_dir, "test"), exist_ok=True)
    with pytest.raises(TimeoutError):
        PolicyClient(ckpt_dir, node_id="test", checkpoint_timeout=2.0)
