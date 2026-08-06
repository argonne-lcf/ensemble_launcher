"""Cluster mode: start orchestrator with multiple executors, submit tasks
pinned to specific executors, then shut down.

LauncherConfig.task_executor_name accepts a list of executor names.
Each Task can set executor_name to target a specific executor.
"""

import math
import os
import socket
import uuid

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient

CHECKPOINT_DIR = os.path.join("/tmp", f"el_cluster_{uuid.uuid4().hex[:8]}")


def simulate(x: float) -> float:
    return math.sin(x) ** 2


# 1. Start the orchestrator with both process pool and MPI executors
el = EnsembleLauncher(
    ensemble_file={},
    system_config=SystemConfig(name="local"),
    launcher_config=LauncherConfig(
        task_executor_name=["async_processpool", "async_mpi"],
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
        policy_config=PolicyConfig(nlevels=0),
        return_stdout=True,
    ),
    Nodes=[socket.gethostname()],
)
el.start()

# 2. Connect a client and submit tasks pinned to different executors
with ClusterClient(checkpoint_dir=CHECKPOINT_DIR) as client:
    futures = {}

    # Python callables -> process pool
    for i in range(10):
        task = Task(
            task_id=f"serial-{i}",
            nnodes=1,
            ppn=1,
            executable=simulate,
            args=(i * 0.5,),
            executor_name="async_processpool",
        )
        futures[task.task_id] = client.submit(task)

    # Shell commands -> MPI executor
    for i in range(3):
        task = Task(
            task_id=f"mpi-{i}",
            nnodes=1,
            ppn=2,
            executable=f"echo 'MPI task {i} on $HOSTNAME'",
            executor_name="async_mpi",
        )
        futures[task.task_id] = client.submit(task)

    for task_id, future in futures.items():
        result = future.result(timeout=30)
        print(f"{task_id}: {result}")

# 3. Shut down
el.stop()
