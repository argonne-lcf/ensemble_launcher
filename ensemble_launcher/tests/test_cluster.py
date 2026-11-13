import asyncio
import logging
import multiprocessing as mp
import os
import socket
import uuid

import pytest
from utils import echo, echo_sleep, echo_stdout

from ensemble_launcher.config import (
    LauncherConfig,
    MPIConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import AsyncMaster, AsyncWorker, ClusterClient
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)

pytestmark = pytest.mark.core


@pytest.mark.asyncio
async def test_async_worker_cluster(
    task_executor="async_processpool", ntasks_per_core=1, exec=echo
):
    ##create tasks
    tasks = {}
    for i in range(12 * ntasks_per_core):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=exec, args=(f"task-{i}",)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())}")
    print(ckpt_dir)
    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=100.0,
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            worker_logs=True,
            master_logs=True,
            mpi_config=MPIConfig(flavor="test"),
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
    )

    process = mp.Process(target=w.create_an_event_loop)
    process.start()
    client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
    client.start()
    futures = {}
    for task_id, task in tasks.items():
        futures[task_id] = client.submit(task)

    results = {}
    for task_id, fut in futures.items():
        results[task_id] = fut.result()
    client.teardown()
    process.terminate()
    process.join(timeout=10.0)

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


@pytest.mark.asyncio
async def test_async_master_cluster(
    task_executor="async_processpool", ntasks_per_core=1, exec=echo
):
    ##create tasks
    tasks = {}
    for i in range(12 * ntasks_per_core):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=exec, args=(f"task-{i}",)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())}")
    print(ckpt_dir)
    w = AsyncMaster(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            child_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=1.0,
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            children_scheduler_policy="simple_split_children_policy",
            policy_config=PolicyConfig(nlevels=2, nchildren=1),
            result_buffer_size=100,
            worker_logs=True,
            master_logs=True,
            mpi_config=MPIConfig(flavor="test"),
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
    )

    process = mp.Process(target=w.create_an_event_loop)
    process.start()
    client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
    client.start()
    futures = {}
    for task_id, task in tasks.items():
        futures[task_id] = client.submit(task)

    results = {}
    for task_id, fut in futures.items():
        results[task_id] = fut.result()
    client.teardown()
    process.terminate()
    process.join(timeout=10.0)

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


@pytest.mark.asyncio
async def test_async_master_cluster_dependencies(
    task_executor="async_processpool", ntasks_per_core=1, exec=echo
):
    ##create tasks
    tasks = {}
    for i in range(12 * ntasks_per_core):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=exec, args=(f"task-{i}",)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())}")
    w = AsyncMaster(
        "test",
        LauncherConfig(
            task_executor_name=task_executor,
            child_executor_name=task_executor,
            comm_name="async_zmq",
            report_interval=1.0,
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            children_scheduler_policy="simple_split_children_policy",
            policy_config=PolicyConfig(nlevels=2, nchildren=1),
            result_buffer_size=100,
            worker_logs=True,
            master_logs=True,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
    )

    process = mp.Process(target=w.create_an_event_loop)
    process.start()
    client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
    client.start()
    futures = {}
    prev_future = None
    for task_id, task in tasks.items():
        prev_future = client.submit(
            task, dependencies=[prev_future] if prev_future else None
        )
        futures[task_id] = prev_future

    results = {}
    for task_id, fut in futures.items():
        results[task_id] = fut.result()
    client.teardown()
    process.terminate()
    process.join(timeout=10.0)

    assert len(results) > 0 and all(
        [
            result.split(",")[0].strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


@pytest.mark.asyncio
async def test_stop_mid_run_fails_unfinished_client_tasks():
    """Two levels (root -> sub-master -> workers). The root is stopped while slow client tasks
    are still unfinished below it: every client future still resolves -- finished tasks with
    their result, the rest with a failure -- and the root's scheduler is left empty."""
    ncpus = 2
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=[socket.gethostname()])
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    m = AsyncMaster(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            child_executor_name="async_processpool",
            comm_name="async_zmq",
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            children_scheduler_policy="simple_split_children_policy",
            policy_config=PolicyConfig(nlevels=2, nchildren=1),
            heartbeat_interval=0.1,
            result_flush_interval=0.2,
        ),
        job_resource,
    )
    run_task = asyncio.create_task(m.run())

    client = await asyncio.to_thread(ClusterClient, node_id="test", checkpoint_dir=ckpt_dir)
    await asyncio.to_thread(client.start)
    try:
        quick = {
            f"quick-{i}": client.submit(Task(task_id=f"quick-{i}", nnodes=1, ppn=1,
                                             executable=echo, args=(f"quick-{i}",)))
            for i in range(2)
        }  # fmt: skip
        # More slow tasks than cores: some run, the rest wait below the root.
        slow = {
            f"slow-{i}": client.submit(Task(task_id=f"slow-{i}", nnodes=1, ppn=1,
                                            executable=echo_sleep, args=(f"slow-{i}", 60.0)))
            for i in range(3 * ncpus)
        }  # fmt: skip

        for task_id, fut in quick.items():
            data = await asyncio.wait_for(asyncio.wrap_future(fut), timeout=60)
            assert data == f"Hello from task {task_id}"

        m.request_stop()
        await asyncio.wait_for(run_task, timeout=120)

        # Every slow future resolves, and with a failure (none of them could finish).
        done, pending = await asyncio.wait(
            [asyncio.wrap_future(f) for f in slow.values()], timeout=30
        )
        assert not pending, f"{len(pending)} client futures never resolved"
        errors = [f.exception() for f in done]
        assert all(e is not None for e in errors), "a slow task reported success"
        assert m._scheduler.state.empty()
    finally:
        if not run_task.done():
            m.request_stop()
            run_task.cancel()
        await asyncio.to_thread(client.teardown)


if __name__ == "__main__":
    print("Testing Async Master with ProcessPool Executor for 1 task per core")
    asyncio.run(test_async_master_cluster(task_executor="async_processpool"))
    print("Testing Async Master with ProcessPool Executor for 10 tasks per core")
    asyncio.run(
        test_async_master_cluster(task_executor="async_processpool", ntasks_per_core=10)
    )
    print("Testing Async Master with MPI Executor")
    asyncio.run(
        test_async_master_cluster(
            task_executor="async_mpi", ntasks_per_core=10, exec=echo_stdout
        )
    )
    ###
    print("Testing Async Worker with ProcessPool Executor for 1 task per core")
    asyncio.run(test_async_worker_cluster(task_executor="async_processpool"))
    print("Testing Async Worker with ProcessPool Executor for 10 tasks per core")
    asyncio.run(
        test_async_worker_cluster(task_executor="async_processpool", ntasks_per_core=10)
    )
    print("Testing Async Worker with MPI Executor")
    asyncio.run(
        test_async_worker_cluster(
            task_executor="async_mpi", ntasks_per_core=10, exec=echo_stdout
        )
    )


def _start_cluster_worker(ckpt_dir):
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=4, cpus=list(range(1, 5)))
    )
    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            report_interval=100.0,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            result_flush_interval=0.1,
        ),
        JobResource(resources=[sys_info], nodes=[socket.gethostname()]),
    )
    process = mp.Process(target=w.create_an_event_loop)
    process.start()
    return process


@pytest.mark.parametrize("io_process", [False, True])
def test_cluster_client_io_process(io_process):
    """One client, its IOProcess in or out of process: submit, map and a dependency all
    resolve, and teardown leaves no IOProcess or ipc files behind."""
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    process = _start_cluster_worker(ckpt_dir)
    try:
        client = ClusterClient(
            node_id="test", checkpoint_dir=ckpt_dir, io_process=io_process,
            task_flush_interval=0.1,
        )  # fmt: skip
        with client:
            conn = client._conn
            assert conn.out_of_process is io_process
            proc, paths = conn._proc, list(conn._io_paths)
            first = client.submit(echo, "a")
            after = client.submit(echo, "b", dependencies=[first])
            mapped = client.map(echo, ["c", "d", "e"])
            assert first.result(timeout=60).startswith("Hello from task a")
            assert after.result(timeout=60).startswith("Hello from task b")
            assert [f.result(timeout=60).split(",")[0] for f in mapped] == [
                f"Hello from task {x}" for x in "cde"
            ]
        assert client._thread is None
        if io_process:
            assert proc is not None and not proc.is_alive()
            assert paths and not any(os.path.exists(p) for p in paths)
    finally:
        process.terminate()
        process.join(timeout=10.0)
