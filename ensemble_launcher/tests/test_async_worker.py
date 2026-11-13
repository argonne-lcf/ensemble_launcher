import asyncio
import logging
import os
import random
import shutil
import socket
import time
import uuid
from collections import deque

import pytest

from ensemble_launcher.ensemble_launcher import run_batch
from ensemble_launcher.comm.messages import Result
from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import AsyncTask, Task
from ensemble_launcher.orchestrator import AsyncWorker
from ensemble_launcher.scheduler import TaskSchedulerState
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

pytestmark = pytest.mark.core

# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def echo(task_id: str):
    time.sleep(0.01)
    return f"Hello from task {task_id}"


def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")


def echo_cwd(task_id: str):
    print(f"Hello from task {task_id}: {os.getcwd()}")


@pytest.mark.asyncio
async def test_async_worker(task_executor="async_processpool", ntasks_per_core=1):
    ##create tasks
    tasks = {}
    for i in range(12 * ntasks_per_core):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo, args=(f"task-{i}",)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=100.0,
            log_level=logging.INFO,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    res = await run_batch(w)
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all(
        [
            result.strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


def timed_sleep(task_id: str, sleep_time: float):
    start = time.time()
    time.sleep(sleep_time)
    return (task_id, start, time.time())


@pytest.mark.asyncio
@pytest.mark.parametrize("strict", [False, True], ids=["backfill", "strict"])
@pytest.mark.parametrize("policy", ["fifo_policy", "large_resource_policy"])
async def test_async_worker_heterogeneous(policy, strict):
    """Mixed-size tasks end to end through a real worker and process pool.

    Each task reports its wall-clock execution window. A task only runs while its
    cores are allocated, so at every instant the cores of the tasks running
    together must fit on the node -- checked with a sweep over those windows.
    Whole-node tasks are included so starvation behind small tasks would hang.
    """
    ncpus = 12
    rng = random.Random(0)
    sizes = {f"task-{i}": rng.choice([1, 1, 1, 2, 4, 6, 12]) for i in range(60)}
    tasks = {
        tid: Task(
            task_id=tid,
            nnodes=1,
            ppn=ppn,
            executable=timed_sleep,
            args=(tid, 0.05),
        )
        for tid, ppn in sizes.items()
    }

    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=[socket.gethostname()])

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            report_interval=100.0,
            log_level=logging.INFO,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
            task_scheduler_policy=policy,
            policy_config=PolicyConfig(strict_priority=strict),
        ),
        job_resource,
        tasks,
    )

    res = await asyncio.wait_for(run_batch(w), timeout=120)
    results = {r.task_id: r for r in res.data}

    assert sorted(results) == sorted(sizes)
    assert all(r.success for r in results.values()), [
        r.exception for r in results.values() if not r.success
    ]

    # Sweep the execution windows; ends sort before starts at the same instant.
    events = []
    for tid, r in results.items():
        _tid, start, end = r.data
        assert _tid == tid
        events.append((start, 1, sizes[tid]))
        events.append((end, 0, -sizes[tid]))
    in_use = peak = 0
    for _t, _kind, delta in sorted(events):
        in_use += delta
        peak = max(peak, in_use)
        assert in_use <= ncpus, f"{in_use} cores in use on a {ncpus}-core node"
    assert peak > 1  # tasks really ran concurrently


@pytest.mark.asyncio
async def test_async_mpi_worker(task_executor="async_mpi"):
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

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=0.5,
            mpi_config=MPIConfig(flavor="test"),
            log_level=logging.INFO,
            worker_logs=True,
            return_stdout=True,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    res = await run_batch(w)
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


@pytest.mark.asyncio
async def test_async_mpi_worker_stdout_file(tmp_path, task_executor="async_mpi"):
    tasks = {}
    for i in range(2):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}",
            nnodes=1,
            ppn=1,
            executable=f"echo Hello from task task-{i}",
            args=(),
            run_dir=str(tmp_path),
            stdout_file=f"task-{i}.stdout",
            stderr_file=f"task-{i}.stderr",
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=0.5,
            mpi_config=MPIConfig(processes_per_node_flag=None),
            log_level=logging.INFO,
            return_stdout=True,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    res = await run_batch(w)
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) == 2

    for i in range(2):
        stdout_path = tmp_path / f"task-{i}.stdout"
        assert stdout_path.exists(), f"stdout file not found: {stdout_path}"
        content = stdout_path.read_text()
        assert f"Hello from task task-{i}" in content

        stderr_path = tmp_path / f"task-{i}.stderr"
        assert stderr_path.exists(), f"stderr file not found: {stderr_path}"


@pytest.mark.asyncio
async def test_async_mpi_pool_worker(task_executor="async_mpi_processpool"):
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo, args=(f"task-{i}",)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=100.0,
            mpi_config=MPIConfig(
                processes_per_node_flag=None,
                hosts_flag=None,
                cpu_bind_method="none",
                extra_launcher_flags=["--oversubscribe", "--allow-run-as-root"],
            ),
            log_level=logging.INFO,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    res = await run_batch(w)
    results = {r.task_id: r.data for r in res.data}

    assert len(results) > 0 and all(
        result.strip() == f"Hello from task {task_id}"
        for task_id, result in results.items()
    ), f"{[result for task_id, result in results.items()]}"


async def async_echo(task_id: str) -> str:
    await asyncio.sleep(0)
    return f"Hello from task {task_id}"


@pytest.mark.asyncio
async def test_async_task_worker():
    tasks = {
        f"task-{i}": AsyncTask(
            task_id=f"task-{i}",
            nnodes=1,
            ppn=1,
            executable=async_echo,
            args=(f"task-{i}",),
        )
        for i in range(12)
    }

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            report_interval=100.0,
            log_level=logging.INFO,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    res = await run_batch(w)
    results = {r.task_id: r.data for r in res.data}

    assert len(results) == len(tasks)
    for task_id in tasks:
        assert results[task_id] == f"Hello from task {task_id}", (
            f"{results[task_id]} != Hello from task {task_id}"
        )


@pytest.mark.asyncio
async def test_run_dir():
    ##create tasks
    tasks = {}
    for i in range(12):
        dirname = os.path.join(os.getcwd(), f"task-{i}")
        os.makedirs(dirname, exist_ok=True)
        shutil.copy(os.path.join(os.getcwd(), "test_async_worker.py"), dirname)
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}",
            nnodes=1,
            ppn=1,
            executable=echo_cwd,
            args=(f"task-{i}",),
            run_dir=dirname,
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name="async_mpi",
            comm_name="async_zmq",
            report_interval=100.0,
            mpi_config=MPIConfig(flavor="test"),
            log_level=logging.INFO,
            return_stdout=True,
            worker_logs=True,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    res = await run_batch(w)
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip()
            == f"Hello from task {task_id}: {os.getcwd()}/{task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"
    for i in range(12):
        dirname = os.path.join(os.getcwd(), f"task-{i}")
        shutil.rmtree(dirname)


# ---------------------------------------------------------------------------
# Windowed result flushes, against a stub comm -- no processes launched
# ---------------------------------------------------------------------------


class _StubComm:
    """Records every result send; `behavior[dest]` decides how each one goes."""

    def __init__(self, behavior):
        self.behavior = behavior
        self.sent = []  # (dest, msg) in send order

    async def _send(self, dest, msg):
        if msg.message_id is None:  # the real comm assigns it once, on first send
            msg.message_id = uuid.uuid4().hex
        self.sent.append((dest, msg))
        return await self.behavior[dest](msg)

    async def send_message_to_child(self, child_id, msg, retries=1):
        return await self._send(child_id, msg)

    async def send_message_to_parent(self, msg, retries=1):
        return await self._send("parent", msg)


def _make_flush_worker(behavior, **config):
    w = AsyncWorker("test", LauncherConfig(**config))
    w.logger = logging.getLogger("test_flush")
    w._comm = _StubComm(behavior)
    return w


def _uid(tid: str) -> bytes:
    """A readable fixed 16-byte uid for a test task name."""
    return tid.encode().ljust(16, b"\0")


def _queue_results(w, dest_id, *task_ids):
    q = w._iresult_q.setdefault(dest_id, deque())
    for tid in task_ids:
        q.append(Result(sender="test", uid=_uid(tid), task_id=tid, success=True))


async def _wait_until(cond, timeout=5.0):
    async def poll():
        while not cond():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


@pytest.mark.asyncio
async def test_result_flush_slow_dest_does_not_delay_fast_dest():
    async def slow(msg):
        await asyncio.sleep(3.0)
        return True

    async def fast(msg):
        return True

    w = _make_flush_worker({"client-a": slow, "parent": fast})
    _queue_results(w, "client-a", "a0")
    _queue_results(w, "main", "m0")  # any non-client destination goes to the parent

    start = time.monotonic()
    w._schedule_result_flush("client-a")
    w._schedule_result_flush("main")
    await _wait_until(lambda: not w._result_flush_inflight["main"])
    assert time.monotonic() - start < 1.0
    assert w._result_flush_inflight["client-a"]

    for t in w._result_flush_inflight["client-a"]:
        t.cancel()
    await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_failed_result_batch_resent_as_same_object():
    calls = []

    async def fail_then_ok(msg):
        calls.append(msg)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return True

    w = _make_flush_worker({"parent": fail_then_ok})
    _queue_results(w, "main", "t0", "t1")

    w._schedule_result_flush("main")
    await _wait_until(lambda: not w._result_flush_inflight["main"])
    assert len(w._result_retry_q["main"]) == 1

    w._schedule_result_flush("main")
    await _wait_until(lambda: not w._result_flush_inflight["main"])

    (_, first_msg), (_, second_msg) = w._comm.sent
    assert second_msg is first_msg
    assert second_msg.message_id == first_msg.message_id is not None
    assert not w._result_retry_q["main"]


@pytest.mark.asyncio
async def test_drain_results_waits_for_inflight_flush():
    gate = asyncio.Event()

    async def gated(msg):
        await gate.wait()
        return True

    w = _make_flush_worker({"parent": gated})
    _queue_results(w, "main", "t0")
    w._schedule_result_flush("main")
    _queue_results(w, "main", "t1")  # arrives while the first batch is in flight

    drain = asyncio.create_task(w._drain_results("main"))
    await asyncio.sleep(0.1)
    assert not drain.done()  # Done must not go out ahead of the stream

    gate.set()
    await asyncio.wait_for(drain, 5.0)
    assert [[r.task_id for r in msg.data] for _, msg in w._comm.sent] == [["t0"], ["t1"]]
    assert not w._result_flush_inflight["main"]
    assert not w._undelivered


@pytest.mark.asyncio
async def test_drain_failure_keeps_results_as_undelivered():
    async def nack(msg):
        return False

    w = _make_flush_worker({"parent": nack})
    _queue_results(w, "main", "t0", "t1")

    await w._drain_results("main")
    # Not delivered: kept for the results file, and Done is not reported to the parent.
    ((dest, msg),) = w._undelivered
    assert dest == "main" and [r.task_id for r in msg.data] == ["t0", "t1"]



class _StubTaskScheduler:
    """Just what the finishing path reads: no tasks left, nothing to fail."""

    def __init__(self):
        from types import SimpleNamespace

        self.state = TaskSchedulerState(node_id="test")
        self.tasks = {}
        self.cluster = SimpleNamespace(free_cpus=0, free_gpus=0)

    async def stop_monitoring(self):
        pass

    def fail_unfinished(self):
        return []


def _ok():
    async def send(msg):
        return True

    return send


async def _finish_worker(behavior):
    from types import SimpleNamespace

    w = _make_flush_worker(behavior)
    w.parent = SimpleNamespace(node_id="main")
    w._scheduler = _StubTaskScheduler()
    w._parent_ready_event.set()
    _queue_results(w, "main", "t0", "t1")
    await w._send_final_results_and_status()
    return w


@pytest.mark.asyncio
async def test_worker_sends_done_after_its_results_and_no_result_batch():
    w = await _finish_worker({"parent": _ok()})
    kinds = [type(msg).__name__ for _, msg in w._comm.sent]
    assert kinds == ["IResultBatch", "Status", "Done"]  # Done last, after every result
    assert "ResultBatch" not in kinds


@pytest.mark.asyncio
async def test_worker_withholds_done_when_results_were_not_delivered(tmp_path, monkeypatch):
    async def nack(msg):
        return type(msg).__name__ != "IResultBatch"  # results fail; status would succeed

    monkeypatch.chdir(tmp_path)
    w = await _finish_worker({"parent": nack})
    kinds = [type(msg).__name__ for _, msg in w._comm.sent]
    assert "Done" not in kinds
    assert (tmp_path / "test_results.bin").exists()  # the undelivered results are kept


async def async_sleep(task_id: str, seconds: float) -> str:
    await asyncio.sleep(seconds)
    return task_id


SLOW = 8.0  # seconds a "slow" task sleeps


@pytest.mark.asyncio
async def test_stop_fails_unfinished_tasks():
    """A worker stopped mid-run reports every task: finished ones as they were, the
    running and pending ones as failed, without waiting for them."""
    tasks = {"quick": AsyncTask(task_id="quick", nnodes=1, ppn=1, executable=async_echo, args=("quick",))}
    for i in range(5):
        tasks[f"slow-{i}"] = AsyncTask(
            task_id=f"slow-{i}", nnodes=1, ppn=1, executable=async_sleep, args=(f"slow-{i}", SLOW)
        )
    sys_info = NodeResourceList.from_config(SystemConfig(name="local", ncpus=2, cpus=[1, 2]))
    w = AsyncWorker(
        "test",
        LauncherConfig(task_executor_name="async_processpool", report_interval=100.0,
                       task_flush_interval=0.5, result_flush_interval=0.5),
        JobResource(resources=[sys_info], nodes=[socket.gethostname()]),
        tasks,
    )  # fmt: skip
    quick_uid = tasks["quick"].uid
    # Time the final results, not run(): stop() then waits for the pool's processes, and
    # a task already running in one is not killed by cancelling its future.
    send_final, finished_at = w._send_final_results_and_status, []

    async def timed_send_final():
        batch = await send_final()
        finished_at.append(time.monotonic())
        return batch

    w._send_final_results_and_status = timed_send_final

    run = asyncio.create_task(w.run())
    await _wait_until(lambda: quick_uid in w._results, timeout=30.0)
    start = time.monotonic()
    w._stop_signal_received.set()
    res = await asyncio.wait_for(run, timeout=60.0)

    assert finished_at[0] - start < SLOW / 2  # did not wait for the running tasks
    by_name = {r.task_id: r for r in res.data}
    assert set(by_name) == set(tasks)
    assert by_name["quick"].success and by_name["quick"].data == "Hello from task quick"
    for i in range(5):
        r = by_name[f"slow-{i}"]
        assert not r.success and "stopped before the task finished" in r.exception


if __name__ == "__main__":
    # print("Testing Async Worker with ProcessPool Executor for 1 task per core")
    # asyncio.run(test_async_worker(task_executor="async_processpool"))
    # print("Testing Async Worker with ProcessPool Executor for 10 tasks per core")
    # asyncio.run(
    #     test_async_worker(task_executor="async_processpool", ntasks_per_core=10)
    # )
    # print("Testing Async Worker with MPI Executor")
    # asyncio.run(test_async_mpi_worker(task_executor="async_mpi"))
    # print("Testing Async Worker with AsyncTask")
    # asyncio.run(test_async_task_worker())
    # print("Testing Async Worker with MPI Pool Executor")
    # asyncio.run(test_async_mpi_pool_worker())
    print("Testing Async Worker rundir")
    asyncio.run(test_run_dir())
