"""Batch mode: run Python callables as an ensemble."""

import math

from ensemble_launcher import EnsembleLauncher, write_results_to_json
from ensemble_launcher.ensemble import Task


def simulate(x: float, y: float) -> dict:
    return {"x": x, "y": y, "result": math.sin(x) * math.cos(y)}


tasks = {
    f"sim-{i}": Task(
        task_id=f"sim-{i}",
        nnodes=1,
        ppn=1,
        executable=simulate,
        args=(i * 0.1, i * 0.2),
    )
    for i in range(20)
}

el = EnsembleLauncher(ensemble_file=tasks)
results = el.run()
write_results_to_json(results)
