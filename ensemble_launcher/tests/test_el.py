import logging
import socket

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm.messages import ResultBatch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def test_EL():
    from ensemble_launcher.config import LauncherConfig, SystemConfig

    el = EnsembleLauncher(
        ensemble_file="ensembles.json",
        Nodes=[socket.gethostname()],
        launcher_config=LauncherConfig(
            return_stdout=True, worker_logs=True, master_logs=True
        ),
        async_orchestrator=True,
    )
    res = el.run()

    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0


if __name__ == "__main__":
    test_EL()
