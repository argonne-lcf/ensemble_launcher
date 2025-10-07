from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, LocalClusterResource
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.ensemble import TaskStatus

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def echo(task_id: int):
    return f"Hello from task {task_id}"

def test_scheduler():
    ##create tasks
    tasks = []
    for i in range(12):
        tasks.append(
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo,
                 args=(i,))
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local"))

    cluster = LocalClusterResource(nodes,system_info=sys_info)

    scheduler = TaskScheduler({task.task_id: task for task in tasks},cluster=cluster)

    ready_tasks = scheduler.get_ready_tasks()

    for task,req in ready_tasks.items():
        assert int(task.split("-")[1]) == req.resources[0].cpus[0]

    #should get zero
    ready_tasks = scheduler.get_ready_tasks()
    assert len(ready_tasks) == 0

    #
    for i in range(12):
        scheduler.free(task_id=f"task-{i}",status=TaskStatus.SUCCESS)
        assert scheduler.remaining_tasks == set([f"task-{id}" for id in range(i+1,12)]) , f"{scheduler.remaining_tasks} != {set([f'task-{id}' for id in range(i,12)])}"
        assert scheduler.successful_tasks == set([f"task-{id}" for id in range(i+1)])


if __name__ == "__main__":
    test_scheduler()
