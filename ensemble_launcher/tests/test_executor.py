import logging
import socket
import sys

from ensemble_launcher.config import SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.executors.mp_executor import MultiprocessingExecutor
from ensemble_launcher.executors.mpi_executor import MPIExecutor
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import (
    LocalClusterResource,
    NodeResourceCount,
    NodeResourceList,
)

logging.basicConfig(
    stream=sys.stdout,
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo, args=(i,)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=12, cpus=list(range(1, 13)))
    )

    from ensemble_launcher.scheduler.resource import JobResource

    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    scheduler = TaskScheduler(logger, tasks, nodes=job_resource)

    ready_tasks = scheduler.get_ready_tasks()

    exec = MultiprocessingExecutor()

    exec_ids = {}

    for task_id, req in ready_tasks.items():
        exec_ids[task_id] = exec.start(
            req, tasks[task_id].executable, task_args=(task_id,)
        )

    results = {}
    for task_id in tasks:
        exec.wait(exec_ids[task_id])
        results[task_id] = exec.result(exec_ids[task_id])
        assert results[task_id] == f"Hello from task {task_id}"
        assert exec.done(exec_ids[task_id])
    exec.shutdown()


def test_mpi_executor():
    ##create tasks
    tasks = {}
    for i in range(12):
        tasks[f"task-{i}"] = Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo_mpi, args=(i,)
        )

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))

    from ensemble_launcher.scheduler.resource import JobResource

    job_resource = JobResource(resources=[sys_info], nodes=nodes)

    scheduler = TaskScheduler(logger, tasks, nodes=job_resource)

    ready_tasks = scheduler.get_ready_tasks()

    exec = MPIExecutor(use_ppn=False)

    exec_ids = {}

    for task_id, req in ready_tasks.items():
        exec_ids[task_id] = exec.start(
            req, tasks[task_id].executable, task_args=(task_id,), task_kwargs={}
        )

    results = {}
    for task_id in tasks:
        exec.wait(exec_ids[task_id])
        results[task_id] = exec.result(exec_ids[task_id])
        assert results[task_id][0].decode("utf-8") == f"Hello from task {task_id}", (
            f"Got: {results[task_id]}"
        )
        assert exec.done(exec_ids[task_id])

    exec.shutdown()


if __name__ == "__main__":
    test_mp_executor()
    test_mpi_executor()
