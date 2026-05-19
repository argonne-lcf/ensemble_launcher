import os
import uuid

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.orchestrator import ClusterClient


def start_el():
    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())[:6]}")
    os.makedirs(ckpt_dir, exist_ok=True)
    system_config = SystemConfig(
        name="local", ncpus=12, cpus=list(range(12)), ngpus=1, gpus=[0]
    )
    launcher_config = LauncherConfig(
        worker_logs=True,
        policy_config=PolicyConfig(nlevels=0),
        cluster=True,
        checkpoint_dir=ckpt_dir,
    )
    el = EnsembleLauncher(
        ensemble_file={}, system_config=system_config, launcher_config=launcher_config
    )
    el.start()
    client = ClusterClient(checkpoint_dir=ckpt_dir)
    client.start()
    return client, el
