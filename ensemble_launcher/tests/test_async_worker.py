from ensemble_launcher.orchestrator import AsyncWorker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, JobResource, NodeResourceCount
import multiprocessing as mp
import os
import logging
import asyncio
import time
import pytest

# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def echo(task_id: str):
    time.sleep(1.0)
    return f"Hello from task {task_id}"

def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")

@pytest.mark.asyncio
async def test_async_worker(task_executor="async_processpool",ntasks_per_core=1):
    ##create tasks
    tasks = {}
    for i in range(12*ntasks_per_core):
        tasks[f"task-{i}"] = \
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo,
                 args=(f"task-{i}",))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local",ncpus=12,cpus=list(range(1,13))))
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",LauncherConfig(task_executor_name=task_executor, comm_name="async_zmq", worker_logs=True, report_interval=100.0, use_mpi_ppn=False, log_level=logging.DEBUG),job_resource,tasks
    )

    res = await w.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all([result.strip() == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

@pytest.mark.asyncio
async def test_async_mpi_worker(task_executor="async_mpi"):
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = \
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo_stdout,
                 args=(f"task-{i}",))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    w = AsyncWorker(
        "test",LauncherConfig(task_executor_name=task_executor, comm_name="async_zmq", worker_logs=True, report_interval=100.0, use_mpi_ppn=False, log_level=logging.DEBUG, return_stdout=True),job_resource,tasks
    )

    res = await w.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all([result.split(",")[0].strip() == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

if __name__ == "__main__":
    print("Testing Async Worker with ProcessPool Executor for 1 task per core")
    asyncio.run(test_async_worker(task_executor="async_processpool"))
    print("Testing Async Worker with ProcessPool Executor for 10 tasks per core")
    asyncio.run(test_async_worker(task_executor="async_processpool",ntasks_per_core=10))
    print("Testing Async Worker with MPI Executor")
    asyncio.run(test_async_mpi_worker(task_executor="async_mpi"))
