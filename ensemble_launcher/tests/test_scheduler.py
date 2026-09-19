import logging
import socket
import sys

import pytest
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, NodeResourceList

pytestmark = pytest.mark.core

logging.basicConfig(
    stream=sys.stdout,
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger()


def echo(task_id: int):
    return f"Hello from task {task_id}"


def test_scheduler():
    ##create tasks
    tasks = []
    for i in range(12):
        tasks.append(
            Task(task_id=f"task-{i}", nnodes=1, ppn=12 - i, executable=echo, args=(i,))
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local", ncpus=12))

    from ensemble_launcher.scheduler.resource import JobResource

    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    scheduler = TaskScheduler(
        logger, {task.task_id: task for task in tasks}, nodes=job_resource
    )

    # ready_tasks = scheduler.get_ready_tasks()

    iter = 0
    while True:
        ready_tasks = scheduler.get_ready_tasks()
        if len(ready_tasks) == 0:
            break
        if iter == 0:
            assert len(ready_tasks) == 1, f"{len(ready_tasks)} != 1"
            assert list(ready_tasks.keys())[0] == f"task-0", (
                f"{list(ready_tasks.keys())[0]} != task-0"
            )
        else:
            assert len(ready_tasks) == 2, f"{len(ready_tasks)} != 2"
            assert list(ready_tasks.keys())[0] == f"task-{iter}", (
                f"{list(ready_tasks.keys())[0]} != task-{11 - iter}"
            )


def test_scheduler_fractional_gpu():
    """
    Four tasks each requesting a quarter of a single GPU should all become
    ready in the same round, since together they fit within the node's one
    GPU (see issue #52: ngpus_per_process=0.25 must not truncate to 0).
    """
    tasks = []
    for i in range(4):
        tasks.append(
            Task(
                task_id=f"gpu-task-{i}",
                nnodes=1,
                ppn=1,
                ngpus_per_process=0.25,
                executable=echo,
                args=(i,),
            )
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=4, ngpus=1, gpus=[0])
    )

    from ensemble_launcher.scheduler.resource import JobResource

    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    scheduler = TaskScheduler(
        logger, {task.task_id: task for task in tasks}, nodes=job_resource
    )

    ready_tasks = scheduler.get_ready_tasks()
    assert len(ready_tasks) == 4, f"expected all 4 fractional-GPU tasks ready, got {len(ready_tasks)}"
    for resource in ready_tasks.values():
        assert resource.resources[0].gpu_count == 0.25


if __name__ == "__main__":
    test_scheduler()
    test_scheduler_fractional_gpu()
