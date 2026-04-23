import asyncio
import logging
import multiprocessing as mp
import os
import socket
import time
import uuid

import pytest
from utils import echo, echo_stdout

from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import AsyncMaster, AsyncWorker, ClusterClient
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)


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
