"""Batch mode: co-schedule tasks pinned to different executors.

LauncherConfig.task_executor_name accepts a list of executor names.
Each Task can set executor_name to target a specific executor.
"""

from ensemble_launcher import EnsembleLauncher, write_results_to_json
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task


def cpu_work(x: float) -> float:
    return sum(i * x for i in range(100_000))


tasks = {}

# Serial tasks pinned to the process pool executor
for i in range(10):
    tasks[f"serial-{i}"] = Task(
        task_id=f"serial-{i}",
        nnodes=1,
        ppn=1,
        executable=cpu_work,
        args=(float(i),),
        executor_name="async_processpool",
    )

# MPI tasks pinned to the MPI executor
for i in range(3):
    tasks[f"mpi-{i}"] = Task(
        task_id=f"mpi-{i}",
        nnodes=1,
        ppn=4,
        executable=f"echo 'MPI task {i} on $HOSTNAME'",
        executor_name="async_mpi",
    )

el = EnsembleLauncher(
    ensemble_file=tasks,
    launcher_config=LauncherConfig(
        task_executor_name=["async_processpool", "async_mpi"],
        return_stdout=True,
    ),
)
results = el.run()
write_results_to_json(results)
