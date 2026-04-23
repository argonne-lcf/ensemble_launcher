import asyncio
import logging
import multiprocessing as mp
import os
import socket
import time

import pytest

from ensemble_launcher.config import LauncherConfig, MPIConfig, SystemConfig
from ensemble_launcher.ensemble import AsyncTask, Task
from ensemble_launcher.orchestrator import AsyncWorker
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def echo(task_id: str):
    time.sleep(1.0)
    return f"Hello from task {task_id}"


def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")


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
        ),
        job_resource,
        tasks,
    )

    res = await w.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all(
        [
            result.strip() == f"Hello from task {task_id}"
            for task_id, result in results.items()
        ]
    ), f"{[result for task_id, result in results.items()]}"


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
            report_interval=100.0,
            mpi_config=MPIConfig(processes_per_node_flag=None),
            log_level=logging.INFO,
            return_stdout=True,
        ),
        job_resource,
        tasks,
    )

    res = await w.run()
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
                processes_per_node_flag=None, hosts_flag=None, cpu_bind_method="none"
            ),
            log_level=logging.INFO,
        ),
        job_resource,
        tasks,
    )

    res = await w.run()
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
        ),
        job_resource,
        tasks,
    )

    res = await w.run()
    results = {r.task_id: r.data for r in res.data}

    assert len(results) == len(tasks)
    for task_id in tasks:
        assert results[task_id] == f"Hello from task {task_id}", (
            f"{results[task_id]} != Hello from task {task_id}"
        )


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
    print("Testing Async Worker with MPI Pool Executor")
    asyncio.run(test_async_mpi_pool_worker())
