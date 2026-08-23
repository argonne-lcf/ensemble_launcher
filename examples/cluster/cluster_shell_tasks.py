"""Cluster mode with shell commands submitted dynamically."""

import os
import socket
import uuid

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient

CHECKPOINT_DIR = os.path.join("/tmp", f"el_cluster_{uuid.uuid4().hex[:8]}")

# 1. Start cluster
el = EnsembleLauncher(
    ensemble_file={},
    system_config=SystemConfig(name="local"),
    launcher_config=LauncherConfig(
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
        policy_config=PolicyConfig(nlevels=0),
        return_stdout=True,
    ),
    Nodes=[socket.gethostname()],
)
el.start()

# 2. Submit shell commands
with ClusterClient(checkpoint_dir=CHECKPOINT_DIR) as client:
    futures = {}
    for i in range(5):
        task = Task(
            task_id=f"echo-{i}",
            nnodes=1,
            ppn=1,
            executable=f"echo 'Hello from task {i}' && sleep 1",
        )
        futures[task.task_id] = client.submit(task)

    for task_id, future in futures.items():
        result = future.result(timeout=30)
        print(f"{task_id}: {result}")

# 3. Shut down
el.stop()
