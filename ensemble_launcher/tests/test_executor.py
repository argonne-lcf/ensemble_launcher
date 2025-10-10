from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, LocalClusterResource, NodeResourceCount
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.executors.mp_executor import MultiprocessingExecutor
from ensemble_launcher.executors.mpi_executor import MPIExecutor

import logging
import sys

logging.basicConfig(
    stream=sys.stdout,
    level=logging.ERROR,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

def echo(task_id: str):
    return f"Hello from task {task_id}"

def echo_mpi(task_id: str):
    import sys
    sys.stdout.write(f"Hello from task {task_id}")
    return

def test_mp_executor():
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = Task(task_id=f"task-{i}",
                                    nnodes=1,
                                    ppn=1,
                                    executable=echo,
                                    args=(i,))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local"))

    cluster = LocalClusterResource(logger, nodes,system_info=sys_info)

    scheduler = TaskScheduler(logger, tasks,cluster=cluster)

    ready_tasks = scheduler.get_ready_tasks()

    exec = MultiprocessingExecutor()

    exec_ids = {}

    for task_id,req in ready_tasks.items():
        exec_ids[task_id] = exec.start(req,tasks[task_id].executable,task_args=(task_id,))
    
    results = {}
    for task_id in tasks:
        exec.wait(exec_ids[task_id])
        results[task_id] = exec.result(exec_ids[task_id])
        assert results[task_id] == f"Hello from task {task_id}"
        assert exec.done(exec_ids[task_id])

def test_mpi_executor():
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = Task(task_id=f"task-{i}",
                                    nnodes=1,
                                    ppn=1,
                                    executable=echo_mpi,
                                    args=(i,))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))

    cluster = LocalClusterResource(logger, nodes,system_info=sys_info)

    scheduler = TaskScheduler(logger, tasks,cluster=cluster)

    ready_tasks = scheduler.get_ready_tasks()

    exec = MPIExecutor()

    exec_ids = {}

    for task_id,req in ready_tasks.items():
        exec_ids[task_id] = exec.start(req,tasks[task_id].executable,task_args=(task_id,))
    
    results = {}
    for task_id in tasks:
        exec.wait(exec_ids[task_id])
        results[task_id] = exec.result(exec_ids[task_id])
        assert results[task_id].decode('utf-8') == f"Hello from task {task_id}", f"Got: {results[task_id]}"
        assert exec.done(exec_ids[task_id])

if __name__ == "__main__":
    test_mp_executor()
    test_mpi_executor()
