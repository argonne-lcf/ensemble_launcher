from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, JobResource
from ensemble_launcher.orchestrator import Master
import logging
from utils import echo


# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

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
        "test",LauncherConfig(return_stdout=True,master_logs=True, worker_logs=True),sys_info,nodes,tasks
    )

    result = m.run()
    results = {r.task_id:r.data for r in result.data}
    
    assert len(results) > 0 and  all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

def test_master_zmq_comm():
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
        "test",LauncherConfig(comm_name="zmq",master_logs=False, worker_logs=False),sys_info,nodes,tasks
    )
    result = m.run()
    results = {r.task_id:r.data for r in result.data}
    
    assert len(results) > 0 and all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

def test_master_multilevel():
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
        "test",LauncherConfig(child_executor_name="mpi",comm_name="zmq",nlevels=2,report_interval=0.1, return_stdout=True,master_logs=True, worker_logs=True),sys_info,nodes,tasks
    )
    ret_result = m.run()
    
    results = {r.task_id:r.data for r in ret_result.data}
    
    assert len(results) > 0 and all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"
    
if __name__ == "__main__":
    test_master()
    # test_master_zmq_comm()
    # test_master_multilevel()