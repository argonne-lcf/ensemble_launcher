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
logger = logging.getLogger()

def echo(task_id: int):
    return f"Hello from task {task_id}"

def test_scheduler():
    ##create tasks
    tasks = []
    for i in range(12):
        tasks.append(
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=12-i,
                 executable=echo,
                 args=(i,))
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local",ncpus=12))

    cluster = LocalClusterResource(logger, nodes,system_info=sys_info)

    scheduler = TaskScheduler(logger, {task.task_id: task for task in tasks},cluster=cluster)

    # ready_tasks = scheduler.get_ready_tasks()

    iter = 0
    while True:
        ready_tasks = scheduler.get_ready_tasks()
        if len(ready_tasks) == 0:
            break
        if iter == 0:
            assert len(ready_tasks) == 1, f"{len(ready_tasks)} != 1"
            assert list(ready_tasks.keys())[0] == f"task-0", f"{list(ready_tasks.keys())[0]} != task-0"
        else:
            assert len(ready_tasks) == 2, f"{len(ready_tasks)} != 2"
            assert list(ready_tasks.keys())[0] == f"task-{iter}", f"{list(ready_tasks.keys())[0]} != task-{11-iter}"


if __name__ == "__main__":
    test_scheduler()
