import os
import socket
import time

# ---------------------------------------------------------------------------
# Cluster mode — start() + ClusterClient
# ---------------------------------------------------------------------------
import uuid

from test_helpers import nested_task

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.orchestrator import ClusterClient


def test_nested_submission():
    ckpt_dir = os.path.join(os.getcwd(), f"ckpt_{str(uuid.uuid4())}")
    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=0),
            return_stdout=True,
            worker_logs=True,
            cpu_binding_option="",
            use_mpi_ppn=False,
            cluster=True,
            checkpoint_dir=ckpt_dir,
        ),
        Nodes=[socket.gethostname()],
    )

    el.start()
    time.sleep(2.0)

    with ClusterClient(checkpoint_dir=ckpt_dir, node_id="global") as client:
        future = client.submit(nested_task, 0, ckpt_dir)
        assert future.result() == "0,1,2,3"
    el.stop()


if __name__ == "__main__":
    print("testing nested submission")
    test_nested_submission()
    print("All tests passed")
