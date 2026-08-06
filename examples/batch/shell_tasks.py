"""Batch mode: run a parameter sweep of shell commands."""

from ensemble_launcher import EnsembleLauncher, write_results_to_json
from ensemble_launcher.config import LauncherConfig

ensemble = {
    "sleep_sweep": {
        "cmd_template": "echo 'task {task_id} sleeping {duration}s' && sleep {duration}",
        "duration": [1, 2, 3, 4, 5],
        "relation": "one-to-one",
        "nnodes": 1,
        "ppn": 1,
    },
}

el = EnsembleLauncher(
    ensemble_file=ensemble,
    launcher_config=LauncherConfig(return_stdout=True),
)
results = el.run()
write_results_to_json(results)
