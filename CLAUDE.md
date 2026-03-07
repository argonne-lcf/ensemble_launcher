# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Install & Setup

```bash
# Editable install with dev dependencies
python3 -m pip install -e ".[dev]"

# Optional extras
python3 -m pip install -e ".[mcp]"       # mcp + paramiko
python3 -m pip install -e ".[dragonhpc]" # DragonHPC backend
```

The CLI entry point is `el` (`ensemble_launcher/cli.py`).

## Running Tests

Tests live in `ensemble_launcher/tests/`. They must be run from that directory because several tests import from `utils.py` via a relative import:

```bash
cd ensemble_launcher/tests
pytest                                   # all tests
pytest test_ensemble_launcher.py         # end-to-end launcher tests
pytest test_async_master.py              # async master/worker tests
pytest test_cluster.py                   # cluster-mode tests
pytest test_mcp.py                       # MCP interface tests
pytest test_ensemble_launcher.py::test_el_run  # single test
```

Tests require `pytest-asyncio` (already in `install_requires`). Async tests are decorated with `@pytest.mark.asyncio`.

## Architecture Overview

### Core abstraction layers (bottom-up)

1. **Executors** (`ensemble_launcher/executors/`) — launch subprocesses or coroutines. Registry pattern: `executor_registry.create_executor(name)`. Key executors:
   - `async_processpool` — `AsyncProcessPoolExecutor` (default for serial tasks)
   - `async_mpi` — `AsyncMPIExecutor` (MPI tasks)
   - `multiprocessing`, `mpi`, `dragon` — legacy sync executors

2. **Communication** (`ensemble_launcher/comm/`) — message passing between nodes. Two async backends:
   - `async_zmq` — `AsyncZMQComm` (preferred for multi-node)
   - `multiprocessing` — `MPComm` (single-node only)
   Messages are typed dataclasses in `comm/messages.py` (`Task`, `Result`, `ResultBatch`, `Status`, `Action`, `NodeUpdate`, `TaskUpdate`, `HeartBeat`, `Stop`).

3. **Scheduler** (`ensemble_launcher/scheduler/`) — assigns tasks to worker nodes. `WorkerScheduler` wraps a `LocalClusterResource` and a pluggable `ChildrenPolicy`. The default policy is `greedy_children_policy`. Custom policies can be loaded at runtime via env vars `EL_EXTERNAL_POLICY_MODULE` / `EL_EXTERNAL_POLICY_PATH`.

4. **Orchestrator** (`ensemble_launcher/orchestrator/`) — the master/worker tree:
   - `AsyncMaster` — manages a layer of children (sub-masters or workers); the root is always named `"main"`.
   - `AsyncWorker` — leaf node that executes tasks using a task executor.
   - `AsyncWorkStealingMaster` / `AsyncWorkStealingWorker` — work-stealing variant (enabled via `LauncherConfig.enable_workstealing`).
   - `ClusterClient` — connects to a running cluster via checkpoint directory to submit tasks and retrieve `concurrent.futures.Future`s.

5. **EnsembleLauncher** (`ensemble_launcher/ensemble_launcher.py`) — top-level entry point. Reads JSON config or a dict of `Task` objects, auto-configures `LauncherConfig` if not provided, builds the orchestrator tree, and exposes `run()` (blocking) / `start()` + `stop()` (non-blocking cluster mode).

### Node naming convention

Orchestrator nodes follow a hierarchical naming scheme:
- `main` — root master
- `main.w0`, `main.w1` — workers directly under root (nlevels=1)
- `main.m0`, `main.m1` — sub-masters (nlevels=2)
- `main.m0.w0` — worker under sub-master 0

`ClusterClient(node_id="global")` auto-resolves to the root master by reading checkpoints.

### Key configuration types (`ensemble_launcher/config.py`)

`SystemConfig` — describes one node: `ncpus`, `ngpus`, `cpus` (list of IDs), `gpus` (list of IDs or strings for overloading).

`LauncherConfig` — controls the entire orchestration:
- `comm_name`: `"async_zmq"` only (sync backends removed)
- `task_executor_name`: `"async_processpool"` | `"async_mpi"` | list for mixed workloads
- `child_executor_name`: executor used to launch sub-master/worker processes
- `nlevels`: hierarchy depth (0=worker only, 1=master+workers, 2=master+sub-masters+workers)
- `cluster`: enables long-lived cluster mode + `ClusterClient` API
- `checkpoint_dir`: where the cluster writes its ZMQ address for clients to discover
- `enable_workstealing`: switches to `AsyncWorkStealingMaster`

### Cluster / MCP mode

When `cluster=True`, the orchestrator runs as a background service. `ClusterClient` reads the checkpoint directory to find the ZMQ address and submit tasks dynamically.

`ensemble_launcher/mcp/Interface` wraps FastMCP and connects to a running cluster:
- `@interface.tool` — single-task MCP tool
- `@interface.ensemble_tool` — batch ensemble MCP tool
SSH tunnel helpers for HPC login→compute node are in `ensemble_launcher/mcp/utils.py`.

### Checkpointing & Profiling

- Checkpoints written to `checkpoint_dir/` (ZMQ addresses for cluster discovery)
- Logs written to `logs/master-*.log` and `logs/worker-*.log`
- Profiling: set `LauncherConfig(profile="perfetto")` → outputs `profiles/*_perfetto.json` and `profiles/*_stats.json`

### Auto-configuration logic

When `launcher_config=None`, `EnsembleLauncher.__init__` auto-selects:
- If all tasks have `nnodes*ppn == 1` → `async_processpool`; otherwise `async_mpi`
- 1 node → `nlevels=0`, `async_zmq`
- 2–64 nodes → `nlevels=1`; 65–2048 → `nlevels=2`; 2048+ → `nlevels=3`
