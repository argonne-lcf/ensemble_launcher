from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceList, JobResource
from ensemble_launcher.orchestrator import AsyncMaster
import logging
from utils import echo, echo_stdout
import asyncio


# logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

async def test_async_master(nlevels=1):
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

    m = AsyncMaster(
        "test",LauncherConfig(return_stdout=True,
                              master_logs=True, 
                              comm_name="async_zmq",
                              nlevels=nlevels,
                              child_executor_name="async_processpool",
                              task_executor_name="async_processpool", 
                              log_level=logging.DEBUG,
                              worker_logs=True),sys_info,nodes,tasks
    )

    resultbatch = await m.run()
    results = {r.task_id:r.data for r in resultbatch.data}

    print(results)
    
    assert len(results) > 0 and  all([result == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"

async def test_async_mpi_master(nlevels=1):
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = \
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo_stdout,
                 args=(f"task-{i}",)
                 )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(SystemConfig(name="local"))

    m = AsyncMaster(
        "test",
        LauncherConfig(return_stdout=True,
                        master_logs=True, 
                        comm_name="async_zmq",
                        nlevels=nlevels,
                        child_executor_name="async_mpi",
                        task_executor_name="async_mpi", 
                        log_level=logging.INFO,
                        worker_logs=True,
                        use_mpi_ppn=False,
                        pin_resources=False,
                        sequential_child_launch=True),
        sys_info,nodes,tasks
    )

    resultbatch = await m.run()
    results = {r.task_id:r.data for r in resultbatch.data}
    
    assert len(results) > 0 and  all([result.strip() == f"Hello from task {task_id}" for task_id, result in results.items()]), f"{[result for task_id, result in results.items()]}"
    
if __name__ == "__main__":
    asyncio.run(test_async_mpi_master(nlevels=3))
    # test_master_zmq_comm()
    # test_master_multilevel()