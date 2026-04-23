import concurrent.futures
import os
import socket
import time
import uuid

from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.orchestrator import ClusterClient
from utils import noop

from ensemble_launcher import EnsembleLauncher


def _create_tasks(ntasks):
    tasks = [
        Task(task_id=f"{idx}", nnodes=1, ppn=1, executable=noop)
        for idx in range(ntasks)
    ]
    return tasks


def benchmark_max_throughput(task_executor="async_mpi_processpool"):
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    os.makedirs(ckpt_dir)
    logger = setup_logger("Pool_benchmark", log_dir="logs")
    launcher_config = LauncherConfig(
        task_executor_name=task_executor,
        cluster=True,
        worker_logs=False,
        policy_config=PolicyConfig(nlevels=0),
        result_flush_interval=0.05,
        checkpoint_dir=ckpt_dir,
    )
    cpus = list(range(104))
    cpus.pop(52)  # can't use these cores on Aurora
    cpus.pop(0)  # can't use these cores on Aurora
    gpus = list(range(12))
    sys_config = SystemConfig(name="Aurora", cpus=cpus, gpus=gpus)

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=sys_config,
        launcher_config=launcher_config,
        Nodes=[socket.gethostname()],
    )
    el.start()
    time.sleep(10.0)
    with ClusterClient(checkpoint_dir=ckpt_dir) as client:
        futures = client.submit_batch(tasks=_create_tasks(102))

        done, not_done = concurrent.futures.wait(futures, timeout=60)
        times = []
        for i in range(1, 30):
            futures = client.submit_batch(tasks=_create_tasks(10**3))
            tic = time.perf_counter()
            done, not_done = concurrent.futures.wait(futures, timeout=60)
            toc = time.perf_counter()

            times.append(toc - tic)

        logger.info(
            f"Throughput: {10**3 / (sum(times) / len(times))} tasks/sec, ntasks: {10**3} for {task_executor} executor"
        )
    el.stop()


if __name__ == "__main__":
    benchmark_max_throughput(task_executor="async_mpi_processpool")
