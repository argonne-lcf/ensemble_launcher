from ensemble_launcher.orchestrator import Worker, Master
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, JobResource
import multiprocessing as mp
import os
import logging
import copy

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def echo(task_id: str):
    return f"Hello from task {task_id}"

def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")




def test_worker():
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

    w = Worker(
        "test",LauncherConfig(executor_name="multiprocessing"),sys_info,nodes,tasks
    )

    res = w.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

def test_worker_in_mp():
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
    sys_info = NodeResourceList.from_config(SystemConfig(name="local"))

    w = Worker(
        "test-worker",LauncherConfig(executor_name="multiprocessing", comm_name="zmq"),sys_info,nodes,tasks
    )
    
    p = mp.Process(target=w.run)
    p.start()
    p.join()

if __name__ == "__main__":
    test_worker()
    test_worker_in_mp()
