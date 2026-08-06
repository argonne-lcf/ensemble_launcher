# Examples

Complete examples are available in the [`examples/`](https://github.com/argonne-lcf/ensemble_launcher/tree/main/examples) directory.

## Batch Mode

Run an ensemble of tasks and block until all complete. Results are returned as a dictionary.

### Shell Commands

[`examples/batch/shell_tasks.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/batch/shell_tasks.py) -- Parameter sweep using `cmd_template`:

```python
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
```

### Python Callables

[`examples/batch/python_tasks.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/batch/python_tasks.py) -- Submit Python functions as tasks:

```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.ensemble import Task

def simulate(x: float, y: float) -> dict:
    return {"x": x, "y": y, "result": math.sin(x) * math.cos(y)}

tasks = {
    f"sim-{i}": Task(
        task_id=f"sim-{i}", nnodes=1, ppn=1,
        executable=simulate, args=(i * 0.1, i * 0.2),
    )
    for i in range(20)
}

el = EnsembleLauncher(ensemble_file=tasks)
results = el.run()
```

### Mixed Workload with Executor Pinning

[`examples/batch/mixed_workload.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/batch/mixed_workload.py) -- Register multiple executors and pin each task to a specific one via `Task.executor_name`:

```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task

tasks = {}

# Serial tasks -> process pool
for i in range(10):
    tasks[f"serial-{i}"] = Task(
        task_id=f"serial-{i}", nnodes=1, ppn=1,
        executable=cpu_work, args=(float(i),),
        executor_name="async_processpool",
    )

# MPI tasks -> MPI executor
for i in range(3):
    tasks[f"mpi-{i}"] = Task(
        task_id=f"mpi-{i}", nnodes=1, ppn=4,
        executable=f"echo 'MPI task {i}'",
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
```

## Cluster Mode

Start the orchestrator as a background service, submit tasks dynamically via `ClusterClient`, then shut down.

### Mixed Executors

[`examples/cluster/cluster_mode.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/cluster/cluster_mode.py) -- Full lifecycle with multiple executors and executor pinning:

```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient

# 1. Start cluster with both executors
el = EnsembleLauncher(
    ensemble_file={},
    launcher_config=LauncherConfig(
        task_executor_name=["async_processpool", "async_mpi"],
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
        policy_config=PolicyConfig(nlevels=0),
    ),
    Nodes=[socket.gethostname()],
)
el.start()

# 2. Submit tasks pinned to different executors
with ClusterClient(checkpoint_dir=CHECKPOINT_DIR) as client:
    futures = {}
    for i in range(10):
        task = Task(task_id=f"serial-{i}", nnodes=1, ppn=1,
                    executable=simulate, args=(i * 0.5,),
                    executor_name="async_processpool")
        futures[task.task_id] = client.submit(task)

    for i in range(3):
        task = Task(task_id=f"mpi-{i}", nnodes=1, ppn=2,
                    executable=f"echo 'MPI task {i}'",
                    executor_name="async_mpi")
        futures[task.task_id] = client.submit(task)

    for task_id, future in futures.items():
        print(f"{task_id}: {future.result(timeout=30)}")

# 3. Shut down
el.stop()
```

### Shell Commands

[`examples/cluster/cluster_shell_tasks.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/cluster/cluster_shell_tasks.py) -- Same lifecycle with shell commands submitted as tasks.

## Task Scripts

Standalone scripts used as task executables:

| Example | Description |
|---|---|
| [`python/serial_example.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/python/serial_example.py) | Serial task with optional CPU/GPU affinity debug output |
| [`python/mpi_example.py`](https://github.com/argonne-lcf/ensemble_launcher/blob/main/examples/python/mpi_example.py) | MPI task with per-rank CPU/GPU binding output |

## MCP

[`examples/mcp/combustion/`](https://github.com/argonne-lcf/ensemble_launcher/tree/main/examples/mcp/combustion) -- Cantera flame speed computation served as an MCP tool over HTTP.

```bash
cd examples/mcp/combustion
python3 start_mcp_http.py
claude mcp add combustion --transport http http://localhost:8295/mcp
```

## Agents

[`examples/agents/pydantic/`](https://github.com/argonne-lcf/ensemble_launcher/tree/main/examples/agents/pydantic) -- Multi-agent joke generator using pydantic-ai with distributed actors over ZMQ and a local vLLM server.

```bash
cd examples/agents/pydantic
export HF_HOME=/path/to/your/hf/cache
python joke_generator.py
```
