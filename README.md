# Ensemble Launcher

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A lightweight, scalable tool for launching and orchestrating task ensembles across HPC clusters with intelligent resource management and hierarchical execution.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Configuration](#configuration)
  - [Basic Configuration](#basic-configuration)
  - [Advanced Configuration](#advanced-configuration)
  - [Resource Pinning](#resource-pinning)
- [Execution Modes](#execution-modes)
- [Cluster Mode](#cluster-mode)
- [Examples](#examples)
- [Performance Tuning](#performance-tuning)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Contributing](#contributing)
- [Support](#support)

---

## Features

- **Flexible Execution**: Support for serial, MPI, and mixed workloads
- **Intelligent Scheduling**: Automatic resource allocation with customizable policies
- **Hierarchical Architecture**: Efficient master-worker patterns for large-scale deployments (1-2048+ nodes)
- **Multiple Communication Backends**: Choose between Python multiprocessing, [ZMQ](https://zeromq.org/), or [DragonHPC](https://dragonhpc.org/portal/index.html) for performance at scale
- **Resource Pinning**: Fine-grained CPU and GPU affinity control
- **Real-time Monitoring**: Track task execution with configurable status updates
- **Fault Tolerance**: Graceful handling of task failures with detailed error reporting
- **Python & Shell Support**: Execute Python callables or shell commands seamlessly
- **Cluster Mode**: Run the orchestrator as a long-lived background service and submit tasks dynamically from any client process

---

## Installation

### Requirements

- Python 3.6+
- numpy
- matplotlib
- scienceplots
- pytest
- cloudpickle
- pydantic
- pyzmq

### Optional Dependencies

- MPI implementation (for distributed execution via `mpirun` or `mpiexec`)
- [DragonHPC](https://github.com/DragonHPC/dragon) (for extreme-scale deployment on HPC systems)
- [mcp](https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file) and [paramiko] (https://www.paramiko.org/) for hosting mcp server on HPC compute nodes

### Quick Install

```bash
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
python3 -m pip install .
```

---

## Quick Start

### 1. Define Your Ensemble

Create a JSON configuration file describing your task ensemble:

```json
{
    "ensembles": {
        "example_ensemble": {
            "nnodes": 1,
            "ppn": 1,
            "cmd_template": "./exe -a {arg1} -b {arg2}",
            "arg1": "linspace(0, 10, 5)",
            "arg2": "linspace(0, 1, 5)",
            "relation": "one-to-one"
        }
    }
}
```

The configuration specifies an ensemble with:

- Tasks running on a single node with a single process per node
- Tasks executed with `./exe -a {arg1} -b {arg2}` taking two input arguments
- The values of the two input arguments are defined as 5 linearly spaced numbers between 0-10 and 0-1 for `arg1` and `arg2`, respectively.
- The raletionship between the values of the two arguments is set to `one-to-one`, meaning the ensemble consists of 5 tasks, one for each pair of values. 

**Supported Relations:**
- `one-to-one`: Pair parameters element-wise (N tasks)
- `many-to-many`: Cartesian product of parameters (N×M tasks)

### 2. Create a Launcher Script

```python
from ensemble_launcher import EnsembleLauncher

if __name__ == '__main__':
    # Auto-configure based on system and workload
    el = EnsembleLauncher("config.json")
    results = el.run()
    
    # Write results to file
    from ensemble_launcher import write_results_to_json
    write_results_to_json(results, "results.json")
```

### 3. Execute

```bash
python3 launcher_script.py
```

---

## Command Line Interface (CLI)

Ensemble Launcher provides a command-line interface for quick execution without writing launcher scripts.

### Commands

The CLI has two subcommands: `start` and `stop`.

```bash
el --help
el start --help
el stop --help
```

#### `el start` — launch an ensemble

Starts the ensemble. In normal mode it blocks until all tasks finish and writes `results.json`. In cluster mode (when `launcher.json` contains `"cluster": true`) it starts the orchestrator in the background and returns immediately.

```bash
el start ENSEMBLE_FILE [OPTIONS]
```

**Options:**

| Option | Description |
|---|---|
| `--system-config-file` | Path to system configuration JSON |
| `--launcher-config-file` | Path to launcher configuration JSON |
| `--nodes-str` | Comma-separated compute nodes, e.g. `"node-001,node-002"` |
| `--pin-resources / --no-pin-resources` | CPU/GPU resource pinning (default: enabled) |
| `--async-orchestrator / --no-async-orchestrator` | Event-driven orchestrator (default: enabled) |

#### `el stop` — stop a running cluster

Sends `SIGTERM` to a cluster-mode orchestrator started with `el start`, triggering graceful shutdown.

```bash
el stop
```

The PID of the background process is stored in `.el_launcher.pid` in the working directory.

### Examples

**Normal blocking execution:**
```bash
el start my_ensemble.json
```

**With custom configurations:**
```bash
el start my_ensemble.json \
    --system-config-file system.json \
    --launcher-config-file launcher.json
```

**Specify compute nodes:**
```bash
el start my_ensemble.json \
    --nodes-str "node-001,node-002,node-003,node-004"
```

**Start a cluster-mode orchestrator in the background:**
```bash
el start my_ensemble.json --launcher-config-file cluster_launcher.json
# Returns immediately; orchestrator is running in the background

# ... submit tasks from Python (see Cluster Mode section) ...

el stop   # gracefully shut down
```

### Configuration Files

**System Configuration (system.json):**
```json
{
    "name": "my_cluster",
    "ncpus": 104,
    "ngpus": 12,
    "cpus": [0, 1, 2, 3, 4],
    "gpus": [0, 1, 2, 3]
}
```

**Launcher Configuration (launcher.json):**
```json
{
    "child_executor_name": "mpi",
    "task_executor_name": "mpi",
    "comm_name": "zmq",
    "nlevels": 2,
    "report_interval": 10.0,
    "return_stdout": true,
    "worker_logs": true,
    "master_logs": true
}
```

---

## Architecture

![Ensemble Launcher Architecture](assets/el.png)

### Key Components

- **EnsembleLauncher**: Main API entry point with auto-configuration
- **Global/Local Master**: Orchestrates workers, handles task distribution and aggregation
- **Worker**: Executes tasks using configured executor
- **Scheduler**: Allocates resources across cluster nodes with intelligent policies
- **Executors**: Backend task launching engines (Python multiprocessing, MPI, DragonHPC)
- **Communication Layer**: ZMQ or Python multiprocessing pipes

### Hierarchical Execution Model

The master-worker architecture scales from single nodes to thousands of nodes:
- **Single Node** (nlevels=0): Direct execution without master overhead
- **Small Scale** (nlevels=1): Global master coordinates workers directly
- **Large Scale** (nlevels=2): Global master → Local masters → Workers for thousands of tasks
- **Extreme Scale** (nlevels=3): Deep hierarchy for supercomputer-scale deployments

---

## Configuration

### Basic Configuration

The launcher automatically configures itself based on your workload and system:

```python
from ensemble_launcher import EnsembleLauncher

el = EnsembleLauncher(
    ensemble_file="config.json",
    Nodes=["node-001", "node-002"],  # Optional: auto-detects from PBS_NODEFILE, works only on PBS
    pin_resources=True,              # Enable CPU/GPU pinning
)
```

### Advanced Configuration

For fine-grained control, explicitly configure system and launcher settings:

```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import SystemConfig, LauncherConfig

# Define system resources
system_config = SystemConfig(
    name="my_cluster",
    ncpus=104,                      # CPUs per node
    ngpus=12,                       # GPUs per node
    cpus=list(range(104)),          # Specific CPU IDs (optional)
    gpus=list(range(12))            # Specific GPU IDs (optional)
)

# Configure launcher behavior
launcher_config = LauncherConfig(
    child_executor_name="mpi",      # multiprocessing, mpi, dragon
    task_executor_name="mpi",       # Executor for tasks
    comm_name="zmq",                # multiprocessing, zmq, dragon
    nlevels=2,                      # Hierarchy depth (auto-computed if None)
    report_interval=10.0,           # Status update frequency (seconds)
    return_stdout=True,             # Capture stdout
    worker_logs=True,               # Enable worker logging
    master_logs=True                # Enable master logging
)

el = EnsembleLauncher(
    ensemble_file="config.json",
    system_config=system_config,
    launcher_config=launcher_config,
    pin_resources=True,
    async_orchestrator=False #use event driven orchestrator (only for zmq communication backend)
)

results = el.run()
```

### Resource Pinning

Pin tasks to specific CPUs and GPUs for optimal performance:

```json
{
    "ensembles": {
        "pinned_ensemble": {
            "nnodes": 1,
            "ppn": 4,
            "cmd_template": "./gpu_code",
            "cpu_affinity": "0,1,2,3",
            "gpu_affinity": "0,1,2,3",
            "ngpus_per_process": 1
        }
    }
}
```
Resources are pinned using the `gpu_selector` option in the LauncherConfig (defaults to "ZE_AFFINITY_MASK" for Intel GPUs). The specific string the `gpu_selector` is set to depends on the SystemConfig. For example, setting:

```python
system_config = SystemConfig(
    name="my_cluster",
    cpus=list(range(104)),          # Specific CPU IDs (optional)
    gpus=['0','0','1','1','2','3']  # Specific GPU IDs (optional)
)
```
will overload the GPU 0 and 1 and the Scheduler assumes that node has a 6 GPUs instead of 4 GPUs.

---

## Execution Modes

### Python Callables

Execute Python functions directly:

```python
def my_simulation(param_a, param_b):
    # Your simulation code
    return result

from ensemble_launcher.ensemble import Task

tasks = {
    "task-1": Task(
        task_id="task-1",
        nnodes=1,
        ppn=1,
        executable=my_simulation,
        args=(10, 0.5)
    )
}

el = EnsembleLauncher(
    ensemble_file=tasks,  # Pass dict directly
)
results = el.run()
```
Note that, internally, the dictionary definition of the ensemble is converted to a collection of Task()s. 
### Shell Commands

Execute binaries and shell commands with files as inputs:

```json
{
    "ensembles": {
        "shell_ensemble": {
            "nnodes": 1,
            "ppn": 1,
            "cmd_template": "./simulation --config {config_file}",
            "config_file": ["config1.json", "config2.json", "config3.json"],
            "relation": "one-to-one"
        }
    }
}
```
which is launched using the following script. 


```python
from ensemble_launcher import EnsembleLauncher

if __name__ == '__main__':
    # Auto-configure based on system and workload
    el = EnsembleLauncher("config.json")
    results = el.run()
    
    # Write results to file
    from ensemble_launcher import write_results_to_json
    write_results_to_json(results, "results.json")
```

## MCP

`ensemble_launcher.mcp.ELFastMCP` is a subclass of [FastMCP](https://github.com/modelcontextprotocol/python-sdk) and exposes two decorators:

- **`@mcp.tool`** — submits a single task to the EnsembleLauncher cluster per MCP call.
- **`@mcp.ensemble_tool`** — accepts lists of arguments and runs one task per element (ensemble in a single call).

Both decorators automatically detect whether the registered function is an `async def` and create an `AsyncTask` instead of a plain `Task`, with no extra configuration required.

The cluster lifecycle is decoupled from the MCP server: start `EnsembleLauncher` separately, then point `ELFastMCP` at its checkpoint directory.

### Minimal example (`start_mcp.py`)

```python
import socket
import time
import uuid
import os

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.mcp import ELFastMCP
from my_module import my_sim   # your simulation function


CHECKPOINT_DIR = f"{os.getcwd()}/mcp_{uuid.uuid4()}"

# 1. Start the EnsembleLauncher cluster (non-blocking)
el = EnsembleLauncher(
    ensemble_file={},
    system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
    launcher_config=LauncherConfig(
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        nlevels=0,
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
    ),
    Nodes=[socket.gethostname()],
)
el.start()
time.sleep(2.0)   # wait for cluster to be ready

# 2. Create the MCP interface, pointing at the running cluster
mcp = ELFastMCP(checkpoint_dir=CHECKPOINT_DIR)

# 3. Register tools — works with both def and async def
mcp.tool(my_sim, nnodes=1, ppn=1)           # single-call tool
mcp.ensemble_tool(my_sim, nnodes=1, ppn=1)  # batch ensemble tool

# 4. Serve (stdio by default; also supports "sse" and "streamable-http")
mcp.run()
```

Decorator style is also supported:

```python
@mcp.tool(nnodes=1, ppn=4)
def my_sim(a: float, b: float) -> str:
    ...

@mcp.ensemble_tool(nnodes=1, ppn=4)
def my_sim(a: float, b: float) -> str:
    ...
```

### Running via stdio (default)

Configure your MCP client (e.g. Claude Desktop) to launch the server:

```json
{
    "mcpServers": {
        "my_sim": {
            "command": "python3",
            "args": ["start_mcp.py"]
        }
    }
}
```

### Port-forwarding helper (HPC login → compute node)

When the MCP server runs on a compute node and the client runs on a login node, use the built-in SSH tunnel helpers:

```python
from ensemble_launcher.mcp import start_tunnel, stop_tunnel

ret = start_tunnel("<username>", "<head-node-hostname>", local_port=9276, remote_port=9276)
# ... run your async client ...
stop_tunnel(*ret)
```

---

## Cluster Mode

Cluster mode turns the orchestrator into a long-lived background service. Clients can connect at any time to submit tasks and receive results — without restarting the orchestrator between runs.

### How it works

1. The orchestrator starts in the background and writes a **comm checkpoint** to `checkpoint_dir` recording its ZMQ address.
2. Any number of `ClusterClient` instances read that checkpoint to discover the address and connect.
3. Clients submit tasks and receive results via `concurrent.futures.Future`.
4. The orchestrator shuts down gracefully on `SIGTERM` (sent by `el stop` or `EnsembleLauncher.stop()`).

### Via the CLI

**`launcher_cluster.json`:**
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

### Via the Python API

**Start / stop:**
```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig

el = EnsembleLauncher(
    ensemble_file={},          # tasks will be submitted by clients
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

**Context manager:**
```python
with EnsembleLauncher(...) as el:
    # orchestrator is running
    ...
# stop() called automatically on exit
```

### Submitting tasks with `ClusterClient`

```python
import time
from ensemble_launcher.orchestrator import ClusterClient
from ensemble_launcher.ensemble import Task

# Wait for the orchestrator to write its checkpoint, then connect.
# node_id="global" (default) resolves to the root master automatically.
with ClusterClient(checkpoint_dir="/scratch/my_job/ckpt") as client:
    futures = {}
    for i in range(10):
        task = Task(task_id=f"task-{i}", nnodes=1, ppn=1,
                    executable=my_fn, args=(i,))
        futures[task.task_id] = client.submit(task)

    results = {tid: fut.result(timeout=60) for tid, fut in futures.items()}
```

**Connecting to a specific node:**
```python
# Connect to a specific worker (useful for targeted task routing)
client = ClusterClient(
    checkpoint_dir="/scratch/my_job/ckpt",
    node_id="main.w0",   # scheduler naming: main, main.w0, main.m0.w1, ...
)
```

### Node ID naming convention

Orchestrator nodes follow the scheduler naming scheme:

| Node ID | Role |
|---|---|
| `main` | Global master (root) |
| `main.w0`, `main.w1` | Workers of the global master |
| `main.m0`, `main.m1` | Sub-masters (nlevels=2) |
| `main.m0.w0` | Worker under sub-master 0 |

`node_id="global"` always resolves to the root master (shortest name in the checkpoint directory).

---

## Examples

See the [`examples`](examples/) directory for complete workflow samples:

### C++ Examples
- [`examples/c++/workflow_pattern1.py`](examples/c++/workflow_pattern1.py) - Basic parallel execution
- [`examples/c++/workflow_pattern2.py`](examples/c++/workflow_pattern2.py) - Parameter sweeps
- [`examples/c++/workflow_pattern3.py`](examples/c++/workflow_pattern3.py) - Complex dependencies

<!-- ### Python Examples
- [`examples/python/mpi_example.py`](examples/python/mpi_example.py) - MPI-based execution
- [`examples/python/serial_example.py`](examples/python/serial_example.py) - Serial task execution -->

### MCP examples
- [`examples/mcp/combustion_agent`](examples/mcp/combustion_agent/) - A simple combustion agent

---

## Performance Tuning

### Communication Backend Selection

| Backend          | Best For                    | Nodes    |
|------------------|-----------------------------|----------|
| `multiprocessing`| Single node, small ensembles| 1        |
| `zmq`            | Multi-node, large scale     | 2-2048+  |

### Hierarchy Levels

The launcher automatically determines hierarchy depth based on node count, but you can override it with:

```python
launcher_config = LauncherConfig(
    nlevels=0   # Direct worker execution (single node)
    nlevels=1   # Master + Workers (up to ~64 nodes)
    nlevels=2   # Master + Sub-masters + Workers (64-2048 nodes)
    nlevels=3   # Deep hierarchy (2048+ nodes)
)
```

**Auto-computed hierarchy:**
- 1 node: `nlevels=0` (worker only)
- 2-64 nodes: `nlevels=1` (master + workers)
- 65-2048 nodes: `nlevels=2` (master + sub-masters + workers)
- 2048+ nodes: `nlevels=3` (deep hierarchy)

### Monitoring and Debugging

Enable logging for detailed execution traces:

```python
# import logging
# logging.basicConfig(level=logging.INFO)

launcher_config = LauncherConfig(
    worker_logs=True,
    master_logs=True,
    report_interval=5.0,  # Report status every 5 seconds
    profile = "basic" or "timeline" #basic ouputs the communication latencies and task runtime. timeline outputs the mean, std, sum, and counts of various events in the orchestrator
)
```

Logs are written to `logs/master-*.log` and `logs/worker-*.log`. Profiles are written to `profiles/*`

---

## API Reference

### EnsembleLauncher

```python
EnsembleLauncher(
    ensemble_file: Union[str, Dict[str, Dict]],
    system_config: SystemConfig = SystemConfig(name="local"),
    launcher_config: Optional[LauncherConfig] = None,
    Nodes: Optional[List[str]] = None,
    pin_resources: bool = True,
    async_orchestrator: bool = True
)
```

**Parameters:**
- `ensemble_file`: Path to JSON config or dict of task definitions
- `system_config`: System resource configuration
- `launcher_config`: Launcher behavior configuration (auto-configured if None)
- `Nodes`: List of compute nodes (auto-detected if None)
- `pin_resources`: Enable CPU/GPU affinity
- `async_orchestrator`: Use event-driven orchestrator (only for ZMQ backend)

**Methods:**
- `run()`: Execute ensemble synchronously and return results (raises `RuntimeError` in cluster mode)
- `start()`: Start the orchestrator in a background process (cluster mode)
- `stop()`: Send SIGTERM to the background process; force-kill after 30 s if needed
- `__enter__` / `__exit__`: Context manager — calls `start()` on entry, `stop()` on exit

### ClusterClient

```python
ClusterClient(
    checkpoint_dir: str,
    node_id: str = "global",
    client_id: Optional[str] = None,
)
```

**Parameters:**
- `checkpoint_dir`: Directory containing orchestrator checkpoint files
- `node_id`: Node to connect to. `"global"` (default) resolves to the root master; pass a specific scheduler name such as `"main.w0"` to connect to a particular node
- `client_id`: Optional identity string; auto-generated if omitted

**Methods:**
- `start()`: Connect transport and start receive thread
- `teardown()`: Disconnect and stop receive thread
- `submit(task)`: Send a `Task` and return a `concurrent.futures.Future`
- `__enter__` / `__exit__`: Context manager — calls `start()` on entry, `teardown()` on exit

### SystemConfig

```python
SystemConfig(
    name: str,
    ncpus: int = mp.cpu_count(),
    ngpus: int = 0,
    cpus: List[int] = [],
    gpus: List[Union[str, int]] = []
)
```

### LauncherConfig

```python
LauncherConfig(
    child_executor_name: Literal["multiprocessing","dragon","mpi"] = "multiprocessing",
    task_executor_name: Literal["multiprocessing","dragon","mpi"] = "multiprocessing",
    comm_name: Literal["multiprocessing","zmq","dragon"] = "multiprocessing",
    report_interval: float = 10.0,
    nlevels: int = 1,
    return_stdout: bool = False,
    worker_logs: bool = False,
    master_logs: bool = False,
    nchildren: Optional[int] = None #Forces number of children at every level
    profile: Optional[Literal["basic","timeline"]] = None
    gpu_selector: str = "ZE_AFFINITY_MASK"
)
```

---

## Testing

Run the test suite:

```bash
pytest tests/
```

Run specific tests:

```bash
pytest tests/test_el.py          # End-to-end tests
pytest tests/test_executor.py    # Executor tests
pytest tests/test_master.py      # Master/Worker tests
```

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
python3 -m pip install -e ".[dev]"
pytest tests/
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/argonne-lcf/ensemble_launcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/argonne-lcf/ensemble_launcher/discussions)
- **Documentation**: See [`examples`](examples/) directory

---

## Acknowledgments

This work was supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357.

---

## Citation

If you use Ensemble Launcher in your research, please cite:

```bibtex
@software{ensemble_launcher,
  title = {Ensemble Launcher: Scalable Task Orchestration for HPC},
  author = {Argonne National Laboratory},
  year = {2025},
  url = {https://github.com/argonne-lcf/ensemble_launcher}
}
```



