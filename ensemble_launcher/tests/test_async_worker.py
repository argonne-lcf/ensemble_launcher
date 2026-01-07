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

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def echo(task_id: str):
    time.sleep(1.0)
    return f"Hello from task {task_id}"

def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")

async def test_async_worker(task_executor="async_processpool"):
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = \
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo,
                 args=(f"task-{i}",))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local"))

    w = AsyncWorker(
        "test",LauncherConfig(task_executor_name=task_executor, comm_name="async_zmq", worker_logs=True, report_interval=100.0, use_mpi_ppn=False, log_level=logging.DEBUG),sys_info,nodes,tasks
    )

    res = await w.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

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

    w = AsyncWorker(
        "test",LauncherConfig(task_executor_name=task_executor, comm_name="async_zmq", worker_logs=True, report_interval=100.0, use_mpi_ppn=False, log_level=logging.DEBUG, return_stdout=True),sys_info,nodes,tasks
    )

    res = await w.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all([result.strip() == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

if __name__ == "__main__":
    # asyncio.run(test_async_worker(task_executor="async_processpool"))
    asyncio.run(test_async_mpi_worker(task_executor="async_mpi"))
