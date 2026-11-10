import asyncio
import logging
import socket
import time
import uuid
from collections import deque

import pytest
from utils import echo, echo_stdout

from ensemble_launcher.ensemble_launcher import run_batch
from ensemble_launcher.comm.messages import Done, IResultBatch, Ready, Result, TaskUpdate
from ensemble_launcher.helper_functions import get_nodes
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import AsyncMaster
from ensemble_launcher.scheduler.child_state import ChildState
from ensemble_launcher.scheduler.state import Child, SchedulerState
from ensemble_launcher.taskstore import SegmentTaskStore
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)

pytestmark = pytest.mark.core



@pytest.mark.asyncio
async def test_async_master(nlevels=1, ntask_per_core=1):
    ##create tasks
    tasks = {}
    for i in range(12 * ntask_per_core):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo, args=(f"task-{i}",)
        )

    print(tasks.keys())
    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    m = AsyncMaster(
        "test",
        LauncherConfig(
            return_stdout=True,
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=nlevels),
            child_executor_name="async_processpool",
            task_executor_name="async_processpool",
            log_level=logging.DEBUG,
            heartbeat_interval=0.1,
            heartbeat_dead_threshold=2.0,
            master_logs=True,
            worker_logs=True,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    resultbatch = await run_batch(m)
    results = {r.task_id: r.data for r in resultbatch.data}

    assert len(results) > 0 and all(
        [result == f"Hello from task {task_id}" for task_id, result in results.items()]
    ), f"{[result for task_id, result in results.items()]}"



@pytest.mark.asyncio
async def test_async_mpi_master(nlevels=1):
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}",
            nnodes=1,
            ppn=1,
            executable=echo_stdout,
            args=(f"task-{i}",),
        )

    nodes = get_nodes()
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", cpus=list(range(1, 13)), ncpus=12)
    )
    job_resource = JobResource(resources=[sys_info] * len(nodes), nodes=nodes)

    m = AsyncMaster(
        "test",
        LauncherConfig(
            return_stdout=True,
            comm_name="async_zmq",
            child_scheduler_policy="simple_split_children_policy",
            policy_config=PolicyConfig(nlevels=nlevels,nchildren=2),
            child_executor_name="async_mpi",
            task_executor_name="async_mpi",
            log_level=logging.INFO,
            mpi_config=MPIConfig(flavor="test"),
            sequential_child_launch=True,
            heartbeat_interval=0.1,
            heartbeat_dead_threshold=2.0,
            master_logs=True,
            worker_logs=True,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    resultbatch = await run_batch(m)
    results = {r.task_id: r.data for r in resultbatch.data}

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


def _batch_master(tasks, ncpus=4):
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    return AsyncMaster(
        "test",
        LauncherConfig(
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=1, nchildren=2),
            child_executor_name="async_processpool",
            task_executor_name="async_processpool",
            heartbeat_interval=0.1,
            heartbeat_dead_threshold=2.0,
            result_flush_interval=0.5,
        ),
        JobResource(resources=[sys_info], nodes=[socket.gethostname()]),
        tasks,
    )


@pytest.mark.asyncio
async def test_batch_run_with_zero_tasks_returns():
    """Nothing to do: wait_drained resolves at once and the run stops through the one stop path."""
    resultbatch = await asyncio.wait_for(run_batch(_batch_master({})), timeout=120)
    assert resultbatch.nresults == 0


@pytest.mark.asyncio
async def test_batch_run_fails_unplaceable_task():
    """A task no child can take is failed at once, so the run still drains and returns."""
    tasks = {
        "ok": Task(task_id="ok", nnodes=1, ppn=1, executable=echo, args=("ok",)),
        "big": Task(task_id="big", nnodes=3, ppn=1, executable=echo, args=("big",)),
    }
    resultbatch = await asyncio.wait_for(run_batch(_batch_master(tasks)), timeout=120)
    results = {r.task_id: r for r in resultbatch.data}
    assert results["ok"].success and results["ok"].data == "Hello from task ok"
    assert not results["big"].success and "Routing failed" in results["big"].exception


# ---------------------------------------------------------------------------
# Windowed flushes, against a stub comm and scheduler -- no processes launched
# ---------------------------------------------------------------------------


class _StubComm:
    """Records every flush send; `behavior[dest]` decides how each one goes."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.sent = []  # (dest, msg) in send order
        self.active = 0
        self.max_active = 0

    async def _send(self, dest, msg):
        if msg.message_id is None:  # the real comm assigns it once, on first send
            msg.message_id = uuid.uuid4().hex
        self.sent.append((dest, msg))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            return await self.behavior[dest](msg)
        finally:
            self.active -= 1

    async def send_message_to_child(self, child_id, msg, retries=1):
        return await self._send(child_id, msg)

    async def send_message_to_parent(self, msg, retries=1):
        return await self._send("parent", msg)


class _RunningChildren(dict):
    """Every child the flush code asks about exists and is RUNNING unless a test says otherwise."""

    def get(self, child_id, default=None):
        if child_id not in self:
            self[child_id] = Child(id=child_id, wid=len(self), job_resource=None,
                                   state=ChildState.RUNNING)  # fmt: skip
        return dict.__getitem__(self, child_id)

    def __getitem__(self, child_id):
        return self.get(child_id)


class _StubScheduler:
    def __init__(self):
        self.tasks = SegmentTaskStore()
        self.state = SchedulerState(node_id="test", children=_RunningChildren())
        self.delivered_uids = []
        self.queued_results = []  # (batch, child_id) handed to queue_result
        self.parent_outbox = asyncio.Queue()
        self.client_outboxes = {}
        self.new_clients = asyncio.Queue()

    def queue_result(self, batch, child_id=None):
        self.queued_results.append((batch, child_id))

    def fail(self, child_id):
        self.state.children.get(child_id).state = ChildState.FAILED

    def delivered(self, uids):
        self.delivered_uids.extend(uids)


def _ok():
    async def send(msg):
        return True

    return send


def _fail_then_ok(first):
    """Fail the first send (False, or raise `first` if it is an exception), then succeed."""
    calls = []

    async def send(msg):
        calls.append(msg)
        if len(calls) == 1:
            if isinstance(first, Exception):
                raise first
            return False
        return True

    return send


def _gated(gate: asyncio.Event):
    async def send(msg):
        await gate.wait()
        return True

    return send


def _make_flush_master(behavior, **config):
    m = AsyncMaster("test", LauncherConfig(**config))
    m.logger = logging.getLogger("test_flush")
    m._comm = _StubComm(behavior)
    m._scheduler = _StubScheduler()
    return m


def _start_sender(m, child_id, ready=True):
    """What _init_child does for a child's sending: a window and a sender task."""
    child = m._scheduler.state.children.get(child_id)
    m._task_window[child_id] = asyncio.Semaphore(m._config.max_inflight_task_sends)
    if ready:
        child.ready.set()
    m._child_sender_task[child_id] = asyncio.create_task(m._child_sender(child_id))
    return child


def _queue_tasks(m, child_id, *task_ids):
    """Put tasks in the master's store and on the child, as one outbox item -- what the
    scheduler's assigner does for one incoming update."""
    tasks = [Task(task_id=tid, nnodes=1, ppn=1, executable="echo") for tid in task_ids]
    m._scheduler.tasks.add(TaskUpdate.from_tasks(tasks))
    child = m._scheduler.state.children.get(child_id)
    uids = [t.uid for t in tasks]
    child.add(uids)
    child.outbox.put_nowait(uids)
    return uids


async def _stop_senders(m):
    for t in list(m._child_sender_task.values()):
        t.cancel()
    for inflight in m._task_flush_inflight.values():
        for t in inflight:
            t.cancel()
    await asyncio.sleep(0)


def _names(m, uids):
    return [m._scheduler.tasks[u].task_id for u in uids]


async def _wait_until(cond, timeout=5.0):
    async def poll():
        while not cond():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


@pytest.mark.asyncio
async def test_task_flush_slow_child_does_not_delay_fast_child():
    async def slow(msg):
        await asyncio.sleep(3.0)
        return True

    m = _make_flush_master({"A": slow, "B": _ok()})
    _start_sender(m, "A")
    _start_sender(m, "B")
    _queue_tasks(m, "A", "a0")
    _queue_tasks(m, "B", "b0")

    start = time.monotonic()
    await _wait_until(lambda: [d for d, _ in m._comm.sent].count("B") == 1
                      and not m._task_flush_inflight.get("B"))  # fmt: skip
    assert time.monotonic() - start < 1.0
    assert m._task_flush_inflight["A"]  # A's send is still out
    await _stop_senders(m)


@pytest.mark.asyncio
async def test_child_not_ready_does_not_hold_up_others():
    m = _make_flush_master({"A": _ok(), "B": _ok()})
    _start_sender(m, "A", ready=False)
    _start_sender(m, "B")
    _queue_tasks(m, "A", "a0")
    _queue_tasks(m, "B", "b0")
    await _wait_until(lambda: m._comm.sent)
    await asyncio.sleep(0.05)
    assert [d for d, _ in m._comm.sent] == ["B"]
    m._scheduler.state.children["A"].ready.set()  # A's NodeUpdate delivered
    await _wait_until(lambda: len(m._comm.sent) == 2)
    await _stop_senders(m)


@pytest.mark.asyncio
async def test_task_flush_default_window_is_one():
    gate = asyncio.Event()
    m = _make_flush_master({"A": _gated(gate)})
    child = _start_sender(m, "A")
    _queue_tasks(m, "A", "t0")
    _queue_tasks(m, "A", "t1")
    await _wait_until(lambda: m._comm.active == 1)
    await asyncio.sleep(0.05)
    assert len(m._task_flush_inflight["A"]) == 1 and child.outbox.qsize() == 1
    gate.set()
    await _wait_until(lambda: len(m._comm.sent) == 2 and not m._task_flush_inflight["A"])
    assert m._comm.max_active == 1
    assert [_names(m, msg.uids()) for _, msg in m._comm.sent] == [["t0"], ["t1"]]
    await _stop_senders(m)


@pytest.mark.asyncio
async def test_task_flush_window_cap():
    gate = asyncio.Event()
    m = _make_flush_master({"A": _gated(gate)}, max_inflight_task_sends=2)
    child = _start_sender(m, "A")
    for i in range(5):
        _queue_tasks(m, "A", f"t{i}")

    await _wait_until(lambda: m._comm.active == 2)
    await asyncio.sleep(0.05)
    assert len(m._task_flush_inflight["A"]) == 2
    # The three items that arrived while the window was full wait in the outbox.
    assert child.outbox.qsize() == 3

    gate.set()
    await _wait_until(lambda: len(m._comm.sent) == 5 and not m._task_flush_inflight["A"])
    assert m._comm.max_active == 2
    assert [msg.ntasks for _, msg in m._comm.sent] == [1, 1, 1, 1, 1]  # items sent as is
    assert sorted(m._scheduler.delivered_uids) == sorted(child.uids)
    await _stop_senders(m)


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [False, RuntimeError("boom")], ids=["nack", "raise"])
async def test_failed_task_update_resent_as_same_object(first):
    m = _make_flush_master({"A": _fail_then_ok(first)}, send_timeout=0.05)
    _start_sender(m, "A")
    _queue_tasks(m, "A", "t0", "t1")
    _queue_tasks(m, "A", "t2")  # newer work queued behind the failing message

    await _wait_until(lambda: len(m._comm.sent) == 3 and not m._task_flush_inflight["A"])

    (_, first_msg), (_, resent), (_, newer) = m._comm.sent
    assert resent is first_msg  # resent from the done-callback, before anything newer
    assert resent.message_id == first_msg.message_id is not None
    assert _names(m, newer.uids()) == ["t2"]
    await _stop_senders(m)


@pytest.mark.asyncio
async def test_failed_task_update_to_recovering_child_is_dropped():
    async def nack(msg):
        m._scheduler.state.children["A"].state = ChildState.RECOVERING
        return False

    m = _make_flush_master({"A": nack})
    _start_sender(m, "A")
    _queue_tasks(m, "A", "t0")
    await _wait_until(lambda: m._comm.sent and not m._task_flush_inflight["A"])
    await asyncio.sleep(0.05)
    assert len(m._comm.sent) == 1  # not resent: recovery resends through the new outbox
    _queue_tasks(m, "A", "t1")  # the slot was freed, so the next item still goes out
    await _wait_until(lambda: len(m._comm.sent) == 2)
    assert _names(m, m._comm.sent[1][1].uids()) == ["t1"]
    await _stop_senders(m)


@pytest.mark.asyncio
async def test_failed_task_update_to_terminal_child_fails_tasks():
    async def nack(msg):
        return False

    m = _make_flush_master({"A": nack})
    m._scheduler.fail("A")
    _start_sender(m, "A")
    _queue_tasks(m, "A", "t0", "t1")

    await _wait_until(lambda: m._scheduler.queued_results)

    ((batch, child_id),) = m._scheduler.queued_results  # routed directly, no task created
    assert child_id == "A"
    assert [(r.task_id, r.success) for r in batch.to_results()] == [("t0", False), ("t1", False)]
    assert len(m._comm.sent) == 1
    await _stop_senders(m)


@pytest.mark.asyncio
async def test_deleted_task_is_not_sent():
    m = _make_flush_master({"A": _ok()})
    child = m._scheduler.state.children.get("A")
    uids = _queue_tasks(m, "A", "t0", "t1")
    child.discard(uids[0])  # deleted while its item waited
    _start_sender(m, "A")
    await _wait_until(lambda: m._comm.sent)
    (_, msg), = m._comm.sent
    assert _names(m, msg.uids()) == ["t1"]
    await _stop_senders(m)


def _result_piece(*names):
    return IResultBatch.from_results(
        [Result(sender="w", uid=bytes(16), task_id=n, success=True) for n in names]
    )


def _start_result_sender(m, dest):
    outbox = asyncio.Queue()
    m._start_result_sender(dest, outbox)
    return outbox


async def _stop_result_senders(m):
    for t in list(m._result_sender_task.values()):
        t.cancel()
    for inflight in m._result_flush_inflight.values():
        for t in inflight:
            t.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_result_sender_sends_everything_queued_as_one_message():
    gate = asyncio.Event()
    m = _make_flush_master({"parent": _gated(gate)})
    outbox = _start_result_sender(m, "parent")
    outbox.put_nowait(_result_piece("r0"))
    await _wait_until(lambda: len(m._comm.sent) == 1)  # first send out, held at the gate
    for name in ("r1", "r2", "r3"):
        outbox.put_nowait(_result_piece(name))
    await asyncio.sleep(0.05)
    assert len(m._comm.sent) == 1  # window 1: the rest wait for the slot
    gate.set()
    await _wait_until(lambda: len(m._comm.sent) == 2)
    second = m._comm.sent[1][1]
    assert [r.task_id for r in second.to_results()] == ["r1", "r2", "r3"]
    await asyncio.wait_for(outbox.join(), 1.0)  # every item delivered
    await _stop_result_senders(m)


@pytest.mark.asyncio
async def test_result_resent_as_same_object_and_counted_once_delivered():
    m = _make_flush_master({"client-x": _fail_then_ok(False)})
    outbox = _start_result_sender(m, "client-x")
    outbox.put_nowait(_result_piece("t0", "t1", "t2"))
    await asyncio.wait_for(outbox.join(), 2.0)
    assert len(m._comm.sent) == 2
    assert m._comm.sent[0][1] is m._comm.sent[1][1]  # same message_id: receiver can dedupe
    assert m._total_streamed == 3  # only the delivered send counts
    await _stop_result_senders(m)


@pytest.mark.asyncio
async def test_result_send_that_raised_is_resent_after_send_timeout():
    m = _make_flush_master({"parent": _fail_then_ok(RuntimeError("down"))}, send_timeout=0.3)
    outbox = _start_result_sender(m, "parent")
    outbox.put_nowait(_result_piece("t0"))
    await _wait_until(lambda: len(m._comm.sent) == 1)
    start = time.monotonic()
    await _wait_until(lambda: len(m._comm.sent) == 2)
    assert time.monotonic() - start >= 0.25
    await _stop_result_senders(m)


@pytest.mark.asyncio
async def test_flush_results_waits_for_delivery_and_is_bounded():
    gate = asyncio.Event()
    m = _make_flush_master(
        {"client-x": _gated(gate), "client-y": _gated(asyncio.Event())},
        send_retries=1, send_timeout=0.3,
    )
    for dest in ("client-x", "client-y"):
        _start_result_sender(m, dest).put_nowait(_result_piece("t"))
    asyncio.get_running_loop().call_later(0.1, gate.set)
    start = time.monotonic()
    await m._flush_results()  # x is delivered; y never is, so the wait gives up
    assert m._results_all_sent("client-x") and not m._results_all_sent("client-y")
    assert 0.25 <= time.monotonic() - start < 2.0
    await _stop_result_senders(m)


@pytest.mark.asyncio
async def test_task_resend_dropped_after_child_torn_down():
    m = _make_flush_master({"A": _fail_then_ok(RuntimeError("down"))}, send_timeout=0.2)
    _start_sender(m, "A")
    _queue_tasks(m, "A", "t0")
    await _wait_until(lambda: len(m._comm.sent) == 1)
    # The child is torn down and reinstated before the resend timer fires: a new window.
    m._child_sender_task.pop("A").cancel()
    new_window = asyncio.Semaphore(1)
    m._task_window["A"] = new_window
    await asyncio.sleep(0.4)
    assert len(m._comm.sent) == 1  # the old message never reaches the new process
    assert new_window._value == 1  # and the new window is untouched
    await _stop_senders(m)



# ---------------------------------------------------------------------------
# Giving up on a child: it becomes FAILED, its tasks are requeued, and nothing waits on it
# ---------------------------------------------------------------------------


class _ChildInfo:
    secret_id = "s"


class _RecoveryComm:
    """Just enough comm for the liveness monitor, teardown and the Done wait."""

    def __init__(self):
        self._child_dead_events = {}
        self.done_sent = {}  # child_id -> Event: set to make that child's Done arrive
        self.sent = []

    def update_node_info(self, info):
        pass

    async def send_message_to_child(self, child_id, msg, retries=1):
        self.sent.append((child_id, msg))
        return True

    async def recv_message_from_child(self, cls, child_id, block=False, timeout=None):
        assert cls is Done
        await self.done_sent.setdefault(child_id, asyncio.Event()).wait()
        return Done(sender=child_id)


async def _recovery_master(restart: bool):
    """A master with a real scheduler and two RUNNING children holding 6 tasks, each with
    its Done wait running; no processes."""
    from ensemble_launcher.scheduler import AsyncChildrenScheduler

    config = LauncherConfig(
        policy_config=PolicyConfig(nlevels=1, nchildren=2),
        restart_children_on_failure=restart,
    )
    m = AsyncMaster("test", config)
    m.logger = logging.getLogger("test_recovery")
    m._comm = _RecoveryComm()
    node = NodeResourceList.from_config(SystemConfig(name="local", ncpus=4))
    sched = AsyncChildrenScheduler(
        m.logger, JobResource(resources=[node], nodes=[socket.gethostname()]), config,
        node_id="test",
    )  # fmt: skip
    tasks = [Task(task_id=f"t{i}", nnodes=1, ppn=1, executable="echo") for i in range(6)]
    await sched.queue(TaskUpdate.from_tasks(tasks))
    sched.create_children(level=0, node_id="test")
    sched.start()
    await sched._input.join()
    m._scheduler = sched
    for cid in sched.state.children:
        sched.transition(cid, ChildState.RUNNING)
        m.children[cid] = _ChildInfo()  # what _init_child registers
        m._comm._child_dead_events[cid] = asyncio.Event()
        m._child_done_task[cid] = asyncio.create_task(m._wait_for_child_done(cid))
    return m, sched


async def _survivor_finishes_and_stop_path_returns(m, sched, cid):
    """The surviving child finishes and sends Done; the stop path's wait must return."""
    sched.state.children[cid].uids.clear()
    m._comm.done_sent.setdefault(cid, asyncio.Event()).set()
    await asyncio.wait_for(m._wait_for_children_done(killed=False), timeout=5)


@pytest.mark.asyncio
async def test_child_reinstated_but_never_relaunched_is_failed_and_tasks_requeued():
    """Recovery reinstates the child (READY) but launch/sync fails every time: after the
    retries it is FAILED, its tasks are on the other child, and the stop path doesn't wait on
    it."""
    m, sched = await _recovery_master(restart=True)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    held = set(a.uids)
    attempts = []

    async def reinstated_but_launch_failed(child_id):
        child = await m._teardown_child(child_id)
        assert sched.reinstate_child(child)  # back READY on its resources
        m.children[child_id] = _ChildInfo()
        m._child_done_task[child_id] = asyncio.create_task(m._wait_for_child_done(child_id))
        attempts.append(child_id)
        return False

    m._recover_dead_child = reinstated_but_launch_failed
    m._comm._child_dead_events[cid_a].set()
    await asyncio.wait_for(m._child_liveness_monitor(cid_a), timeout=5)
    await sched._input.join()

    a = sched.state.children[cid_a]  # the reinstated record
    assert attempts == [cid_a] * 3
    assert a.state is ChildState.FAILED and not a.uids and not a.allocated
    assert held <= set(b.uids) and not sched.state.unassigned and sched.unplaced.empty()
    assert not any(isinstance(msg, Ready) for _, msg in m._comm.sent)
    await _survivor_finishes_and_stop_path_returns(m, sched, cid_b)
    await sched.stop()


@pytest.mark.asyncio
async def test_dead_child_without_restarts_is_failed_and_stop_does_not_wait_for_its_done():
    m, sched = await _recovery_master(restart=False)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    held = set(a.uids)
    m._comm._child_dead_events[cid_a].set()
    await asyncio.wait_for(m._child_liveness_monitor(cid_a), timeout=5)
    await sched._input.join()

    assert a.state is ChildState.FAILED and not a.uids
    assert held <= set(b.uids) and sched.unplaced.empty()
    await asyncio.wait_for(m._child_done_task[cid_a], timeout=5)  # ended without a Done
    await _survivor_finishes_and_stop_path_returns(m, sched, cid_b)
    await sched.stop()


@pytest.mark.asyncio
async def test_failed_send_to_failed_child_does_not_fail_requeued_tasks():
    """A TaskUpdate to a child that then failed comes back NACKed: its tasks were requeued on
    the other child, so none of them gets a failed result."""
    m, sched = await _recovery_master(restart=False)
    (cid_a, a), (cid_b, b) = sched.state.children.items()
    msg = sched.tasks.build_update(list(a.uids), sender="test")
    m._fail_tasks = lambda uids, reason, child_id=None: failed.extend(uids)
    failed = []
    sched.transition(cid_a, ChildState.FAILED, requeue=True)

    async def nack():
        return False

    t = asyncio.create_task(nack())
    await t
    m._task_flush_done(cid_a, msg, asyncio.Semaphore(0), t)
    assert failed == []
    for task in m._child_done_task.values():
        task.cancel()
    await sched.stop()


@pytest.mark.asyncio
async def test_done_while_holding_tasks_fails_them_and_stop_does_not_wait():
    """Children send Done while still holding tasks (e.g. placed after they took Stop): each
    is marked SUCCESS anyway, what it held goes to unplaced (failed), and the stop path's wait
    returns."""
    m, sched = await _recovery_master(restart=False)
    held = {u for c in sched.state.children.values() for u in c.uids}
    assert held
    for cid in sched.state.children:
        m._comm.done_sent.setdefault(cid, asyncio.Event()).set()
    await asyncio.wait_for(m._wait_for_children_done(killed=False), timeout=5)

    assert all(c.state is ChildState.SUCCESS and not c.uids for c in sched.state.children.values())
    unplaced = set()
    while not sched.unplaced.empty():
        unplaced.update(sched.unplaced.get_nowait())
    assert unplaced == held
    await sched.stop()

if __name__ == "__main__":
    #print("Testing Async Master with ProcessPool Executor for 1 task per core")
    #asyncio.run(test_async_master(nlevels=1, ntask_per_core=1))
    # print("Testing Async Master with ProcessPool Executor for 10 tasks per core")
    # asyncio.run(test_async_master(nlevels=1, ntask_per_core=10))
    print("Testing Async Master with MPI Executor")
    asyncio.run(test_async_mpi_master(nlevels=3))

