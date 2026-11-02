import asyncio
import logging
import socket
import time
import uuid
from collections import deque

import pytest
from utils import echo, echo_stdout

from ensemble_launcher.comm.messages import Result
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

    resultbatch = await m.run()
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

    resultbatch = await m.run()
    results = {r.task_id: r.data for r in resultbatch.data}

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


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


class _StubScheduler:
    def __init__(self):
        self.terminal = set()

    def get_child_state(self, child_id):
        return ChildState.FAILED if child_id in self.terminal else ChildState.RUNNING

    def is_child_terminal(self, child_id):
        return child_id in self.terminal


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


def _queue_tasks(m, child_id, *task_ids):
    q = m._itask_q.setdefault(child_id, deque())
    for tid in task_ids:
        q.append(Task(task_id=tid, nnodes=1, ppn=1, executable="echo"))


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
    _queue_tasks(m, "A", "a0")
    _queue_tasks(m, "B", "b0")

    start = time.monotonic()
    m._schedule_task_flush("A")
    m._schedule_task_flush("B")
    await _wait_until(lambda: not m._task_flush_inflight["B"])
    assert time.monotonic() - start < 1.0
    assert m._task_flush_inflight["A"]  # A's send is still out

    for t in m._task_flush_inflight["A"]:
        t.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_task_flush_window_cap():
    gate = asyncio.Event()
    m = _make_flush_master({"A": _gated(gate)}, max_inflight_flushes=2)

    for i in range(5):
        _queue_tasks(m, "A", f"t{i}")
        m._schedule_task_flush("A")

    assert len(m._task_flush_inflight["A"]) == 2
    # The three tasks that arrived while the window was full wait in the buffer.
    assert [t.task_id for t in m._itask_q["A"]] == ["t2", "t3", "t4"]

    await _wait_until(lambda: m._comm.active == 2)
    gate.set()
    await _wait_until(lambda: not m._task_flush_inflight["A"])
    m._schedule_task_flush("A")
    await _wait_until(lambda: not m._task_flush_inflight["A"])

    assert m._comm.max_active == 2
    assert [len(msg.added_tasks) for _, msg in m._comm.sent] == [1, 1, 3]
    assert not m._itask_q["A"]


@pytest.mark.asyncio
@pytest.mark.parametrize("first", [False, RuntimeError("boom")], ids=["nack", "raise"])
async def test_failed_task_update_resent_as_same_object(first):
    m = _make_flush_master({"A": _fail_then_ok(first)})
    _queue_tasks(m, "A", "t0", "t1")

    m._schedule_task_flush("A")
    await _wait_until(lambda: not m._task_flush_inflight["A"])
    assert len(m._task_retry_q["A"]) == 1  # requeued, not lost

    m._schedule_task_flush("A")  # what the next periodic tick does
    await _wait_until(lambda: not m._task_flush_inflight["A"])

    (_, first_msg), (_, second_msg) = m._comm.sent
    assert second_msg is first_msg
    assert second_msg.message_id == first_msg.message_id is not None
    assert not m._task_retry_q["A"]


@pytest.mark.asyncio
async def test_periodic_task_flush_loop_resends_failed_update():
    m = _make_flush_master({"A": _fail_then_ok(False)}, task_flush_interval=0.05)
    _queue_tasks(m, "A", "t0")

    loop_task = asyncio.create_task(m._periodic_task_flush_loop())
    try:
        # Nothing new is queued after the first failure; the loop resends from the retry queue.
        await _wait_until(lambda: len(m._comm.sent) == 2 and not m._task_flush_inflight["A"])
    finally:
        loop_task.cancel()
    assert m._comm.sent[0][1] is m._comm.sent[1][1]
    assert not m._task_retry_q["A"]


@pytest.mark.asyncio
async def test_failed_task_update_to_terminal_child_fails_tasks():
    async def nack(msg):
        return False

    m = _make_flush_master({"A": nack})
    m._scheduler.terminal.add("A")
    forwarded = []
    m._create_forward_task = forwarded.append
    _queue_tasks(m, "A", "t0", "t1")

    m._schedule_task_flush("A")
    await _wait_until(lambda: not m._task_flush_inflight["A"])

    assert [(r.task_id, r.success) for r in forwarded] == [("t0", False), ("t1", False)]
    assert not m._task_retry_q.get("A")


@pytest.mark.asyncio
async def test_result_flush_counts_streamed_only_on_success():
    m = _make_flush_master({"client-x": _fail_then_ok(False)})
    m._iresult_q["client-x"] = deque(
        Result(sender="w", task_id=f"t{i}", success=True) for i in range(3)
    )

    m._schedule_result_flush("client-x")
    await _wait_until(lambda: not m._result_flush_inflight["client-x"])
    assert m._total_streamed == 0

    m._schedule_result_flush("client-x")
    await _wait_until(lambda: not m._result_flush_inflight["client-x"])
    assert m._total_streamed == 3
    assert m._comm.sent[0][1] is m._comm.sent[1][1]


if __name__ == "__main__":
    #print("Testing Async Master with ProcessPool Executor for 1 task per core")
    #asyncio.run(test_async_master(nlevels=1, ntask_per_core=1))
    # print("Testing Async Master with ProcessPool Executor for 10 tasks per core")
    # asyncio.run(test_async_master(nlevels=1, ntask_per_core=10))
    print("Testing Async Master with MPI Executor")
    asyncio.run(test_async_mpi_master(nlevels=3))
