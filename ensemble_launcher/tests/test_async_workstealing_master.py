import asyncio
import logging
import multiprocessing as mp
import os
import socket
import uuid

import pytest
from utils import echo

from ensemble_launcher.config import (
    LauncherConfig,
    PolicyConfig,
    SystemConfig,
)
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import AsyncWorkStealingMaster, ClusterClient
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)

pytestmark = pytest.mark.core


@pytest.mark.asyncio
async def test_workstealing_master(ntask_per_core=1):
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

    m = AsyncWorkStealingMaster(
        "test",
        LauncherConfig(
            return_stdout=True,
            comm_name="async_zmq",
            enable_workstealing=True,
            policy_config=PolicyConfig(nlevels=1, nchildren=2),
            child_executor_name="async_processpool",
            task_executor_name="async_processpool",
            log_level=logging.DEBUG,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
        tasks,
    )

    resultbatch = await m.run()
    results = {r.task_id: r.data for r in resultbatch.data}

    assert len(results) == len(tasks) and all(
        result == f"Hello from task {task_id}" for task_id, result in results.items()
    ), f"{[result for task_id, result in results.items()]}"


@pytest.mark.asyncio
async def test_workstealing_cluster(ntask_per_core=1):
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

    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())}")
    m = AsyncWorkStealingMaster(
        "test",
        LauncherConfig(
            return_stdout=True,
            comm_name="async_zmq",
            enable_workstealing=True,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            policy_config=PolicyConfig(nlevels=1, nchildren=2),
            child_executor_name="async_processpool",
            task_executor_name="async_processpool",
            log_level=logging.DEBUG,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        job_resource,
    )

    process = mp.Process(target=m.create_an_event_loop)
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

    assert len(results) == len(tasks) and all(
        result == f"Hello from task {task_id}" for task_id, result in results.items()
    ), f"{[result for task_id, result in results.items()]}"


if __name__ == "__main__":
    print("Testing WorkStealing Master with 1 task per core")
    asyncio.run(test_workstealing_master(ntask_per_core=1))
    print("Testing WorkStealing Master with 10 tasks per core")
    asyncio.run(test_workstealing_master(ntask_per_core=10))
    print("Testing WorkStealing Cluster mode")
    asyncio.run(test_workstealing_cluster(ntask_per_core=1))
