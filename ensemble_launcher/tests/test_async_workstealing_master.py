import asyncio
import logging
import socket

import pytest
from utils import echo, echo_stdout

from ensemble_launcher.config import LauncherConfig, MPIConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import AsyncWorkStealingMaster as AsyncMaster
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)


# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
@pytest.mark.asyncio
async def test_async_master(nlevels=1, ntask_per_core=1):
    ##create tasks
    tasks = {}
    for i in range(12 * ntask_per_core):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo, args=(f"task-{i}",)
        )

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
            policy_config=PolicyConfig(nlevels=nlevels, nchildren=2),
            child_executor_name="async_processpool",
            task_executor_name="async_processpool",
            log_level=logging.DEBUG,
            children_scheduler_policy="simple_split_children_policy",
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

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", cpus=list(range(1, 13)), ncpus=12)
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    m = AsyncMaster(
        "test",
        LauncherConfig(
            return_stdout=True,
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=nlevels, nchildren=2),
            child_executor_name="async_mpi",
            task_executor_name="async_mpi",
            log_level=logging.INFO,
            mpi_config=MPIConfig(cpu_bind_method="none", processes_per_node_flag=None),
            sequential_child_launch=True,
            children_scheduler_policy="simple_split_children_policy",
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


if __name__ == "__main__":
    print("Testing Async Master with ProcessPool Executor for 1 task per core")
    asyncio.run(test_async_master(nlevels=1, ntask_per_core=1))
    print("Testing Async Master with ProcessPool Executor for 10 tasks per core")
    asyncio.run(test_async_master(nlevels=1, ntask_per_core=10))
    print("Testing Async Master with MPI Executor")
    asyncio.run(test_async_mpi_master(nlevels=1))
