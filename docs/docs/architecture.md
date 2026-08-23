# Architecture

## Overview

Ensemble Launcher is built as a layered system, with each layer providing a distinct abstraction.

## Core Layers (Bottom-Up)

### 1. Executors

**Location:** `ensemble_launcher/executors/`

Executors launch subprocesses or coroutines. They are created via a registry pattern:

```python
from ensemble_launcher.executors import executor_registry
executor = executor_registry.create_executor("async_processpool")
```

| Executor | Use Case |
|---|---|
| `async_processpool` | Default for serial tasks |
| `async_mpi` | MPI-parallel tasks |

### 2. Communication

**Location:** `ensemble_launcher/comm/`

A layered, transport-agnostic messaging system for node-to-node communication:

| Layer | Class | Role |
|---|---|---|
| Coordination | `AsyncComm` | High-level message routing, per-node caches, heartbeat management |
| Transport | `AsyncTransport` | Pluggable backend (`AsyncZMQTransport` for distributed, `AsyncMPTransport` for local) |
| Connection | `AsyncConnection` | Low-level send/recv per link (`AsyncZMQRouterConnection`, `AsyncZMQDealerConnection`, `AsyncMPConnection`) |

Parent-child links are created via `transport.create_child_pipe()`, which returns a paired server and client connection. Each node runs a separate `HeartBeatProcess` for dead-connection detection. Transports are registered via a `TransportRegistry` for extensibility.

Messages are typed dataclasses defined in `comm/messages.py`: `Task`, `Result`, `ResultBatch`, `Status`, `Action`, `NodeUpdate`, `TaskUpdate`, `HeartBeat`, `Stop`.

### 3. Scheduler

**Location:** `ensemble_launcher/scheduler/`

Assigns tasks to worker nodes. `WorkerScheduler` wraps a `LocalClusterResource` and a pluggable `ChildrenPolicy`. The default policy is `greedy_children_policy`.

Custom policies can be loaded at runtime via environment variables -- see [Custom Scheduling](custom-scheduling.md).

### 4. Orchestrator

**Location:** `ensemble_launcher/orchestrator/`

The master/worker tree that coordinates execution:

| Component | Role |
|---|---|
| `AsyncMaster` | Manages a layer of children (sub-masters or workers). The root is always named `"main"`. |
| `AsyncWorker` | Leaf node that executes tasks using a task executor. |
| `AsyncWorkStealingMaster` / `AsyncWorkStealingWorker` | Work-stealing variant enabled via `LauncherConfig.enable_workstealing`. |
| `ClusterClient` | Connects to a running cluster to submit tasks and retrieve `concurrent.futures.Future`s. |

### 5. EnsembleLauncher

**Location:** `ensemble_launcher/ensemble_launcher.py`

Top-level entry point. Reads a JSON config or a dict of `Task` objects, auto-configures `LauncherConfig` if not provided, builds the orchestrator tree, and exposes `run()` (blocking) or `start()` / `stop()` (non-blocking cluster mode).

## Node Naming Convention

Orchestrator nodes follow a hierarchical naming scheme:

| Node ID | Role |
|---|---|
| `main` | Root master |
| `main.w0`, `main.w1` | Workers directly under root |
| `main.m0`, `main.m1` | Sub-masters (nlevels >= 2) |
| `main.m0.w0` | Worker under sub-master 0 |

`ClusterClient(node_id="global")` auto-resolves to the root master by reading checkpoints.

## Auto-Configuration

When `launcher_config=None`, `EnsembleLauncher` auto-selects settings:

- **Executor:** If all tasks have `nnodes * ppn == 1`, uses `async_processpool`; otherwise `async_mpi`
- **Hierarchy:** Based on node count (see table above)
- **Communication:** ZMQ transport for data and heartbeat channels

## Checkpointing & Profiling

- **Checkpoints** are written to `checkpoint_dir/` (ZMQ addresses for cluster discovery)
- **Logs** are written to `logs/master-*.log` and `logs/worker-*.log`
- **Profiling:** Set `LauncherConfig(profile="perfetto")` to output `profiles/*_perfetto.json` and `profiles/*_stats.json`
