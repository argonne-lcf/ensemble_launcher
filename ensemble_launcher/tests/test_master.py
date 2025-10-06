from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, JobResource
from ensemble_launcher.orchestrator import Master
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def echo(task_id: str):
    return f"Hello from task {task_id}"

def test_master():
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

    m = Master(
        "test",LauncherConfig(executor_name="multiprocessing"),sys_info,nodes,tasks
    )
    m.run()
    results = {r.task_id:r.data for r in m.results()["test.w0"]}
    m.stop()
    
    assert all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

if __name__ == "__main__":
    test_master()
