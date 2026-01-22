from ensemble_launcher import EnsembleLauncher
import socket
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def test_EL():
    from ensemble_launcher.config import SystemConfig, LauncherConfig
    el = EnsembleLauncher(
        ensemble_file="ensembles.json",
        Nodes=[socket.gethostname()],
        launcher_config=LauncherConfig(return_stdout=True)
    )
    res = el.run()
    results = {}
    for r in res.data:
        results[r.task_id] = r.data

    assert len(results) > 0 and all([result.strip() == f"Hello from task task-{task_id}" for task_id, result in enumerate(results.values())]), f"{[result.strip() for task_id, result in results.items()]}"


if __name__ == "__main__":
    test_EL()