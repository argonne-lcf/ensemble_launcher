from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList

def echo(task_id: str):
    return f"Hello from task {task_id}"

def test_worker():
    ##create tasks
    tasks = []
    for i in range(12):
        tasks.append(
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo,
                 args=(f"task-{i}",))
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local"))

    w = Worker(
        "test",LauncherConfig(executor_name="multiprocessing"),sys_info,nodes,tasks
    )
    w.run_tasks()
    results = w.results()
    # w.stop()

    assert all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

if __name__ == "__main__":
    test_worker()
