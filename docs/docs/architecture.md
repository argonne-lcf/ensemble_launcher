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

Messages are typed dataclasses defined in `comm/messages.py`: `Message`, `Status`, `Result`, `ResultBatch`, `IResultBatch`, `TaskUpdate`, `NodeUpdate`, `Ready`, `Stop`, `Done`, `TaskRequest`, `NodeRequest`. Heartbeats are not a message type; they are handled by `HeartBeatProcess` in `comm/hb.py`.

**Tasks and results travel as blobs.** `TaskUpdate`, `ResultBatch` and `IResultBatch` carry their items as columns (`comm/blob.py`) rather than one Python object each:

| Message | Frames after header + container |
|---|---|
| `TaskUpdate` | `uids` (16 B per task), metadata records + `int64` offsets, deep records (executable, args, kwargs, env, result) + offsets |
| `ResultBatch` / `IResultBatch` | `uids`, `success` (1 B per result), payload records `(task_id, data, exception)` + offsets |

Receiving one only wraps the frames. Masters route on the fixed-width columns and split / join messages with `select` / `concat`, copying byte ranges; only a worker builds `Task`s (`TaskUpdate.task`), only the root master or a `ClusterClient` builds `Result`s (`to_results`, or the `.data` property returned by `run()`).

Why: CPython's cyclic GC walks every live container object on each full collection. Decoding a message into ~7 objects per task, kept alive by the master, made GC 38% of `main`'s busy time at 256 nodes with ~1 s pauses (#65). With blobs, `benchmarks/benchmark_master_routing.py` routes 1M tasks in ~5 s with 0.3% GC and 3 ms pauses, versus 30.8 s, 47% and 1.5 s before.

Every task has an internal 16-byte `Task.uid`, generated when the `Task` is created. EL keys everything by it; `task_id` is only the user's name, returned on `Result.task_id`, and need not be unique.

### 3. Scheduler

**Location:** `ensemble_launcher/scheduler/`

Assigns tasks to worker nodes. The live async path is `AsyncChildrenScheduler` / `AsyncTaskScheduler` in `scheduler/async_scheduler.py`; `WorkerScheduler` is the legacy sync class, used only by `orchestrator/master.py`.

There are two pluggable policy families, each with its own registry namespace:

| Family | Base class | Key method | Default |
|---|---|---|---|
| Task scoring | `Policy` | `get_score` | `fifo_policy` |
| Child layout / distribution | `ChildrenPolicy` | `get_children_resources`, `get_children_tasks` | `simple_split_children_policy` |

Every policy carries a mutable `self.state` dict, seeded from `PolicyConfig.initial_state` and tunable at runtime through `PolicyClient`.

Policies receive task uids and read fields through `self.tasks[uid]` (a `TaskMeta` with every non-deep field), never `Task` objects.

**Scheduler state.** A master's `AsyncChildrenScheduler` keeps everything it knows in one `SchedulerState` (`scheduler/state.py`): a `Child` record per child (wid, resources, lifecycle state, last status, the uids of its unfinished tasks, and an outbox of tasks waiting to be sent) plus the pool of tasks not yet placed. Its moves keep one rule: every unfinished task is in the pool or held by exactly one child. The orchestrator reads it directly and reports events; the scheduler changes it -- `queue`, `create_children`, `finish_tasks(rows, child_id)`, `transition(child_id, state)` (a table of legal moves with their side effects), `remove_child` / `reinstate_child` for recovery. The same object is the checkpoint: `to_json()` on the event loop, `SchedulerState.from_json()` on restart, with runtime-only fields (events, statuses, outboxes, the pool) rebuilt on restore.

**Task placement.** Placement is event driven and never blocks the event loop for long:

```
await scheduler.queue(update) --> input queue --> assigner (chunks of assign_chunk_size, yields between)
                                                    |-- one uid list per child per update --> child.outbox
                                                    '-- tasks no child can take          --> unplaced (failed)
child.outbox --> master's sender for that child: take a window slot, then an item --> one TaskUpdate
                 (a failed send is resent as the same message in the same slot)
```

In work-stealing (pull) mode, queued tasks wait in the pool until a child's `TaskRequest` places them on its outbox. A relaunched child gets an outbox holding every task it still owed.

**Result routing.** Results go back up the same way, as they arrive:

```
child's IResultBatch / failed tasks --> scheduler.queue_result(batch, child_id)
                                          finish the tasks, split by destination
                                          |-- client tasks --> that client's outbox
                                          '-- the rest     --> parent outbox (at the root: the run's results)
each outbox --> its sender: take a window slot, then everything queued --> one IResultBatch
                (a failed send is resent as the same message in the same slot)
```

Masters forward without a timer: whatever arrives during a send leaves together in the next one. Workers keep a periodic flush (`result_flush_interval`), because their results finish one at a time and would otherwise each become a message.

**Lifecycle.** Every node starts and stops the same way, in every mode. A child starts with a heartbeat sync; its parent pushes the `NodeUpdate`, waits for it to be acknowledged, then marks the child running and starts sending it tasks as they are placed. No node decides by itself that it is finished: each stops on its parent's `Stop`, on SIGTERM (or `request_stop()`), or when its parent dies. It then stops its children, waits for each one's `Done`, fails whatever is still unfinished, delivers every result, and sends its parent a final `Status` and `Done`. `Done` is sent only once every result has been acknowledged, so a parent that reads it has all of that child's results. Batch mode is decided only by `EnsembleLauncher`: it waits until the root reports that every task has a result (`wait_drained()`), then stops it.

**Task store.** Each node -- master or worker -- keeps its tasks in a `TaskStore` (`ensemble_launcher/taskstore/`), the scheduler's `self.tasks`. The interface (`add`, `store[uid]`, `task`, `build_update`, status, `mark_delivered`, `release`, `close`) hides how tasks are held, so a database-backed store can replace the default `SegmentTaskStore` by registering under a name and setting `LauncherConfig.task_store`. `SegmentTaskStore` keeps received `TaskUpdate`s as they arrived (segments; `add` neither copies nor decodes), indexes `uid -> segment, row`, keeps up to `task_store_hot_bytes` in memory and offloads the rest -- delivered segments first -- to files in `task_store_dir`, read back through `mmap`. Writes run on a thread, so the event loop never waits on disk. A finished task is released at once; its segment's bytes go when all of its tasks have.

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
- **Profiling:** attach py-spy to running nodes with `el profile`; see [CLI Reference](cli.md#el-profile)
