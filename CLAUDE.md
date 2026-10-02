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

`conftest.py` forces the `spawn` start method, so a spawned orchestrator node does **not**
import the test module that launched it. A policy registered with `@policy_registry.register`
inside a test file is therefore invisible to the node. Put fixture policies in a separate
module (see `tests/policy_fixtures.py`) and point the child at it with
`EL_EXTERNAL_POLICY_PATH` / `EL_EXTERNAL_POLICY_MODULE`.

Tests leave scratch directories behind in `tests/` and nothing cleans them up:
`.actor_ckpt_*` (hundreds accumulate) and `task-N/` from `test_async_worker.py`'s
working-directory test. The `task-N/` ones contain a *copy of the test file*, so an
interrupted run breaks the next collection with `import file mismatch`. Clear them with
`rm -rf task-[0-9]*` before re-running.

## Architecture Overview

### Core abstraction layers (bottom-up)

1. **Executors** (`ensemble_launcher/executors/`) — launch subprocesses or coroutines. Registry pattern: `executor_registry.create_executor(name)`. Key executors:
   - `async_processpool` — `AsyncProcessPoolExecutor` (default for serial tasks)
   - `async_mpi` — `AsyncMPIExecutor` (MPI tasks)
   - `multiprocessing`, `mpi`, `dragon` — legacy sync executors

2. **Communication** (`ensemble_launcher/comm/`) — message passing between nodes.
   `AsyncComm` (`comm/async_base.py`) is the base; `AsyncZMQComm` (`comm/async_zmq.py`) is
   the ZMQ implementation. Below it, `comm/pipe/` holds the connection layer
   (`AsyncZMQRouterConnection` / `AsyncZMQDealerConnection`) that both the comm layer and
   the standalone endpoints (actors, policy) build on.
   Messages are typed dataclasses in `comm/messages.py`: `Message`, `Status`, `Result`,
   `ResultBatch`, `IResultBatch`, `TaskUpdate`, `NodeUpdate`, `Ready`, `Stop`, `TaskRequest`,
   `NodeRequest`. (Heartbeats are not a message type — they live in `comm/hb.py` as
   `HeartBeatProcess`.)

3. **Scheduler** (`ensemble_launcher/scheduler/`) — assigns tasks to worker nodes.
   The live async path is `AsyncChildrenScheduler` / `AsyncTaskScheduler`
   (`scheduler/async_scheduler.py`); `WorkerScheduler` (`scheduler/scheduler.py`) is the
   legacy sync class, used only by `orchestrator/master.py`.
   Two policy families, registered in separate namespaces by
   `policy_registry.register(name, type="policy"|"children_policy")`:
   - task policies (`Policy.get_score`) — `large_resource_policy` (default), `fifo_policy`
   - children policies (`ChildrenPolicy`) — `simple_split_children_policy` (default),
     `bin_packing_children_policy`, `fixed_leafs_children_policy`

   Custom policies can be loaded at runtime via env vars `EL_EXTERNAL_POLICY_MODULE` /
   `EL_EXTERNAL_POLICY_PATH` (honoured by `orchestrator/__init__.py` at import time, so
   spawned children pick them up too).

   **Policy state.** Both families mix in `PolicyStateMixin`, giving every policy a
   `self.state` dict seeded from `PolicyConfig.initial_state` and readable/writable from
   outside the run via `PolicyClient` (see below). `on_state_update(changed)` is the hook
   for recomputing derived values. State updates run on the orchestrator's event loop, so
   `set_policy_state` and `on_state_update` **must be synchronous and non-blocking**.

4. **Orchestrator** (`ensemble_launcher/orchestrator/`) — the master/worker tree:
   - `AsyncMaster` — manages a layer of children (sub-masters or workers); the root is always named `"main"`.
   - `AsyncWorker` — leaf node that executes tasks using a task executor.
   - `AsyncWorkStealingMaster` / `AsyncWorkStealingWorker` — work-stealing variant (enabled via `LauncherConfig.enable_workstealing`).
   - `ClusterClient` — connects to a running cluster via checkpoint directory to submit tasks and retrieve `concurrent.futures.Future`s.
   - `PolicyEndpoint` / `PolicyClient` — read and retune a node's policy state while the run is in flight (see below).
   - `discovery.py` — shared checkpoint-directory lookup (`_resolve_node_id`, `read_policy_endpoint`, `discover_policy_nodes`), used by both clients.

5. **EnsembleLauncher** (`ensemble_launcher/ensemble_launcher.py`) — top-level entry point. Reads JSON config or a dict of `Task` objects, auto-configures `LauncherConfig` if not provided, builds the orchestrator tree, and exposes `run()` (blocking) / `start()` + `stop()` (non-blocking cluster mode).

### Node naming convention

Orchestrator nodes follow a hierarchical naming scheme:
- `main` — root master
- `main.w0`, `main.w1` — workers directly under root (nlevels=1)
- `main.m0`, `main.m1` — sub-masters (nlevels=2)
- `main.m0.w0` — worker under sub-master 0

`ClusterClient(node_id="global")` auto-resolves to the root master by reading checkpoints.

### Actors (`ensemble_launcher/ensemble/actor.py`)

Long-lived stateful objects addressable from outside the run, independent of the task tree.

- `@action` marks a method as remotely callable; `@actor` turns a function into a `PublicActor`.
- `PublicActor` runs its own process **and its own event loop**, binds a dedicated ZMQ
  ROUTER, and publishes the address to `<ckpt_dir>/<name>.ckpt`. `PrivateActor` is scoped
  to the run instead of being externally discoverable.
- `create_handle()` returns an `ActorHandle` (DEALER) that discovers the actor from that
  ckpt file. `ActorHandle._recv_loop` demuxes replies into per-action-name FIFO queues.
- `ActorPool` fans calls out across replicas.

`PolicyEndpoint` deliberately mirrors this architecture — see the next section for where
it differs.

### Policy state and `PolicyClient`

Lets an external controller read and retune scheduling-policy state mid-run, so policies
can react to *workflow* state and not just the `SchedulerState` they are handed per call.

- **`PolicyEndpoint`** (`orchestrator/policy_endpoint.py`) — one per node. Same shape as
  `PublicActor` (dedicated ROUTER, address + secret written to
  `{node}_policy.ckpt` / `{node}_policy_secret`) with one deliberate difference: **it has no
  event loop of its own**. Its serve coroutine is an `asyncio.create_task` on the
  orchestrator's existing loop, and `apply()` is fully synchronous — which is what makes a
  tune atomic with respect to every policy decision without any locking.
- Masters register their children policy under kind `"children"`; workers register their
  task policy under kind `"task"`.
- **`PolicyClient(checkpoint_dir, node_id=...)`** (`orchestrator/policy_client.py`) — sync,
  background-thread client (mirrors `ClusterClient`, not `ActorHandle`) with `get_state` /
  `set_state`. `PolicyGroupClient` fans out across nodes; the group call is *not* atomic
  across nodes, though each node is individually all-or-nothing.
- Requires `checkpoint_dir`, plus `cluster=True` or `enable_policy_client=True`. Tuned
  state is checkpointed to `{node}_policy_state.json` and restored on restart, so it
  survives `restart_children_on_failure`.
- **`rescore` defaults to `False`.** A task's priority is cached in the pending heap at
  enqueue time, so a plain `set_state` only affects work submitted *afterwards*.
  `set_state(..., rescore=True)` calls `AsyncTaskScheduler.reprioritize_pending()` to
  re-score the pending heap in place — worker-only (masters report `rescored: None`,
  since the children-policy analogue would tear down live child processes).
- If `set_state` raises — typically a poison state whose `get_score` throws, surfacing
  inside the rescore — the endpoint rolls the state back, so a failed tune is a no-op.

Gotchas worth knowing before touching this code:
- `PendingTaskHeap.rescore` mutates `self._heap` in place rather than swapping in a fresh
  heap, because the heap owns the `asyncio.Event` that `_monitor_resources` parks on.
  Replacing the object orphans that Event and silently deadlocks dispatch.
- `AsyncConnection._recv_loop` only runs when `req_res=True`, and in that mode it auto-ACKs
  and strips the msg_id frame. So the policy wire format carries no msg_id, correlating
  replies by an explicit `request_id` instead.
- `ServerConnection.verify_sender` short-circuits to `True` when no expected remotes are
  configured, which is exactly this endpoint's situation — so the shared-secret check is
  enforced at the application layer in `PolicyEndpoint._handle`, not by the connection.
- `AsyncConnection.send()` does **not** retry internally; it returns `False` on ACK timeout.
  Retrying is the caller's job.

### Key configuration types (`ensemble_launcher/config/config.py`)

`SystemConfig` — describes one node: `ncpus`, `ngpus`, `cpus` (list of IDs), `gpus` (list of IDs or strings for overloading).

`LauncherConfig` — controls the entire orchestration:
- `comm_name`: `"async_zmq"` only (sync backends removed)
- `task_executor_name`: `"async_processpool"` | `"async_mpi"` | list for mixed workloads
- `child_executor_name`: executor used to launch sub-master/worker processes
- `nlevels`: hierarchy depth (0=worker only, 1=master+workers, 2=master+sub-masters+workers)
- `cluster`: enables long-lived cluster mode + `ClusterClient` API (also implies a policy endpoint)
- `checkpoint_dir`: where the cluster writes its ZMQ address for clients to discover
- `enable_workstealing`: switches to `AsyncWorkStealingMaster`
- `enable_policy_client`: serves the `PolicyEndpoint` without requiring `cluster=True`
- `children_scheduler_policy` / `task_scheduler_policy`: policy names (defaults
  `simple_split_children_policy` / `large_resource_policy`)
- `policy_config`: `PolicyConfig(nlevels, nchildren, leaf_nodes, strict_priority, initial_state)`
- `req_res`: enables ACK-based guaranteed delivery at the comm layer (default `True`)
- `send_retries`: number of retry attempts on ACK timeout, only when `req_res=True` (default `10`)
- `send_timeout`: per-attempt ACK timeout in seconds, only when `req_res=True` (default `1.0`)

### Cluster / MCP mode

When `cluster=True`, the orchestrator runs as a background service. `ClusterClient` reads the checkpoint directory to find the ZMQ address and submit tasks dynamically.

`ensemble_launcher/mcp/Interface` wraps FastMCP and connects to a running cluster:
- `@interface.tool` — single-task MCP tool
- `@interface.ensemble_tool` — batch ensemble MCP tool
SSH tunnel helpers for HPC login→compute node are in `ensemble_launcher/mcp/utils.py`.

### Checkpointing & Profiling

- Checkpoints written to `checkpoint_dir/`, one directory per node-id segment
  (`main.m0.w0` → `<ckpt>/main/m0/w0/`). Per node: the cluster ZMQ address, plus
  `{node}_policy.ckpt` / `{node}_policy_secret` / `{node}_policy_state.json` for the
  policy endpoint. Secret files are written `chmod 0600`.
- Logs written to `logs/master-*.log` and `logs/worker-*.log`
- Profiling: set `LauncherConfig(profile="perfetto")` → outputs `profiles/*_perfetto.json` and `profiles/*_stats.json`

### Auto-configuration logic

When `launcher_config=None`, `EnsembleLauncher.__init__` auto-selects:
- If all tasks have `nnodes*ppn == 1` → `async_processpool`; otherwise `async_mpi`
- 1 node → `nlevels=0`, `async_zmq`
- 2–64 nodes → `nlevels=1`; 65–2048 → `nlevels=2`; 2048+ → `nlevels=3`
