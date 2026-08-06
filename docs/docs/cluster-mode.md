# Cluster Mode

Cluster mode turns the orchestrator into a long-lived background service. Clients can connect at any time to submit tasks and receive results without restarting the orchestrator between runs.

## How It Works

1. The orchestrator starts in the background and writes a **comm checkpoint** to `checkpoint_dir` recording its ZMQ address.
2. Any number of `ClusterClient` instances read that checkpoint to discover the address and connect.
3. Clients submit tasks and receive results via `concurrent.futures.Future`.
4. The orchestrator shuts down gracefully on `SIGTERM` (sent by `el stop` or `EnsembleLauncher.stop()`).

## Via the CLI

**launcher_cluster.json:**
```json
{
    "task_executor_name": "async_processpool",
    "comm_name": "async_zmq",
    "nlevels": 1,
    "cluster": true,
    "checkpoint_dir": "/scratch/my_job/ckpt"
}
```

```bash
# Start the orchestrator in the background
el start my_ensemble.json --launcher-config-file launcher_cluster.json

# Submit tasks from Python (see below)

# Graceful shutdown
el stop
```

## Via the Python API

### Start / Stop

```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig

el = EnsembleLauncher(
    ensemble_file={},
    system_config=SystemConfig(name="local", ncpus=8),
    launcher_config=LauncherConfig(
        cluster=True,
        checkpoint_dir="/scratch/my_job/ckpt",
    ),
    Nodes=["node-001", "node-002"],
)

el.start()   # non-blocking; spawns orchestrator in a separate process
# ...
el.stop()    # sends SIGTERM, waits for graceful exit, force-kills if needed
```

### Context Manager

```python
with EnsembleLauncher(...) as el:
    # orchestrator is running
    ...
# stop() called automatically on exit
```

## Submitting Tasks with ClusterClient

```python
from ensemble_launcher.orchestrator import ClusterClient
from ensemble_launcher.ensemble import Task

with ClusterClient(checkpoint_dir="/scratch/my_job/ckpt") as client:
    futures = {}
    for i in range(10):
        task = Task(task_id=f"task-{i}", nnodes=1, ppn=1,
                    executable=my_fn, args=(i,))
        futures[task.task_id] = client.submit(task)

    results = {tid: fut.result(timeout=60) for tid, fut in futures.items()}
```

### Connecting to a Specific Node

```python
client = ClusterClient(
    checkpoint_dir="/scratch/my_job/ckpt",
    node_id="main.w0",
)
```

## Node ID Naming Convention

| Node ID | Role |
|---|---|
| `main` | Global master (root) |
| `main.w0`, `main.w1` | Workers of the global master |
| `main.m0`, `main.m1` | Sub-masters (nlevels=2) |
| `main.m0.w0` | Worker under sub-master 0 |

`node_id="global"` always resolves to the root master (shortest name in the checkpoint directory).
