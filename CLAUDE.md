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
   the ZMQ implementation. Below it, `comm/pipe/` holds the connection layer that both the
   comm layer and the standalone endpoints (actors, policy) build on: the shared bases at the
   top (`async_connection.py` front ends, `io_process.py` IOProcess + queues,
   `async_transport.py`, `registry.py`), and one subpackage per transport, each with
   `connection.py` / `io_process.py` / `transport.py` -- `comm/pipe/zmq/`
   (`AsyncZMQRouterConnection` / `AsyncZMQDealerConnection`, `ZMQRouterIOProcess` /
   `ZMQDealerIOProcess`, `AsyncZMQTransport`) and `comm/pipe/mp/` (multiprocessing.Pipe).
   **Connections are a front end plus an `IOProcess`, joined by two queues.**
   `AsyncConnection` (and `ServerConnection` / `ClientConnection`, the ZMQ router/dealer and MP
   classes) is only the API: `send` puts `("send", req_id, target, timeout, frames)` on the
   input queue and awaits `("sent", req_id, ok)`; `recv` reads `("data", frames)` the IOProcess
   put on the output queue; `add/remove_expected_remote` and `set_unknown_sender_validator`
   become `("control", ...)` items. **All ACK handling lives in the `IOProcess`**
   (`comm/pipe/io_process.py`; the per-transport subclasses in `zmq/io_process.py` /
   `mp/io_process.py` hold the socket code): its receive loop verifies the sender, ACKs at once and forwards; its
   sends assign msg ids and wait for the peer's ACK with the timeout. The IOProcess runs on the
   caller's loop (`LocalIOQueue` = `asyncio.Queue`), or with `io_process=True` + `req_res=True`
   on a ZMQ connection, in its own spawned process (`ZMQIOQueue` = PUSH/PULL over `ipc://`), so
   a busy event loop can't delay ACKs -- the 2048-node failure in #69. `LauncherConfig.comm_io_process`
   turns that on for every node's data transport (heartbeats, `PolicyEndpoint`, clients and
   actors stay in process); it is carried in the transport and connection states, so a child's
   parent connection runs out of process in the child too. Out of process the unknown-sender
   validator must be a `SenderRule` (prefix + secret), not a closure. The IOProcess exits when
   its owner dies. Tests: `EL_TEST_COMM_IO_PROCESS=1 pytest -m core` runs the suite with it on.
   Messages are typed dataclasses in `comm/messages.py`: `Message`, `Status`, `Result`,
   `ResultBatch`, `IResultBatch`, `TaskUpdate`, `NodeUpdate`, `Ready`, `Stop`, `Done`, `TaskRequest`,
   `NodeRequest`. (Heartbeats are not a message type — they live in `comm/hb.py` as
   `HeartBeatProcess`.)

3. **Scheduler** (`ensemble_launcher/scheduler/`) — assigns tasks to worker nodes.
   The live async path is `AsyncChildrenScheduler` / `AsyncTaskScheduler`
   (`scheduler/async_scheduler.py`); `WorkerScheduler` (`scheduler/scheduler.py`) is the
   legacy sync class, used only by `orchestrator/master.py`.
   Two policy families, registered in separate namespaces by
   `policy_registry.register(name, type="policy"|"children_policy")`:
   - task policies (`Policy.get_score`) — `fifo_policy` (default), `large_resource_policy`
   - children policies (`ChildrenPolicy`) — `simple_split_children_policy` (default),
     `bin_packing_children_policy`, `fixed_leafs_children_policy`

   **Tasks are uids and blobs.** Every `Task` has an internal 16-byte `uid` (set at creation;
   `task_id` is only the user's name and need not be unique). Schedulers, policies, result
   routing and `ClusterClient` futures all key by uid. `TaskUpdate` / `ResultBatch` /
   `IResultBatch` carry items as columns (`comm/blob.py`: fixed-width `uids` plus
   blob + `int64` offsets) and are built with `TaskUpdate.from_tasks` /
   `ResultBatch.from_results`; masters split and join them with `select` / `concat` and never
   decode them. Only workers build `Task`s (`store.task(uid)`, right before submitting) and only
   the root / `ClusterClient` build `Result`s (`.to_results()`, `.data`). Do not reintroduce
   per-task Python objects on masters: that was #65 (GC at 38% of `main`'s busy time at 256
   nodes). Measure with `benchmarks/benchmark_master_routing.py` (`--hold` keeps tasks held).

   **Task store.** `self.tasks` on both schedulers is a `TaskStore` (`taskstore/`), chosen by
   `LauncherConfig.task_store`; policies get it via `bind_tasks` and read `self.tasks[uid]`
   (a `TaskMeta`, decoded on demand -- don't keep them). `SegmentTaskStore` keeps received
   updates as segments, offloads past `task_store_hot_bytes` to `task_store_dir` (mmap), and
   `release(uids)` drops finished tasks from the index at once -- so `uid in store` means
   "still to run", which the master's recovery resend relies on. The store forgets finished
   tasks, so `len(scheduler.tasks)` shrinks; use `scheduler.state.ntasks_added` for totals.
   `taskstore` must not import `comm.messages` at module level (circular import via
   `scheduler.resource`).

   **Master scheduler state.** Everything `AsyncChildrenScheduler` knows is in
   `scheduler.state`, a `SchedulerState` (`scheduler/state.py`): `children` (one `Child`
   record each -- `wid`, `job_resource`, `state`, `status`, `allocated`, `ready`/`done` events,
   `uids`, the ordered set of its *unfinished* tasks, and `outbox`), `unassigned` (the pool),
   `task_to_client`, `ntasks_added`, `all_done`. Its moves (`queue`, `assign`, `finish`,
   `forget`, `add_child`, `pop_child`) keep one rule: **every unfinished uid is in the pool or
   in exactly one child's `uids`**. `add_child` claims the child's uids still in the pool and
   gives it a fresh outbox holding all of them; `pop_child` returns its uids to the pool.
   Orchestrators **read** the state; only the scheduler writes it. Don't add parallel
   per-child dicts -- put fields on `Child`. `finish_tasks(rows, child_id)` gets `child_id`
   from the loop the results arrived on, so there is no uid -> child map. `SchedulerState` is
   also the checkpoint (`get_state()` = `to_json()`, on the loop); `set_state` adds each
   restored child with `add_child`, so it must run **after** the initial tasks are queued and
   **before** `start()`. Children policies receive `{wid: Child}`.

   **Task placement is event driven.** `await scheduler.queue(update, client_id)` stores the
   tasks and puts them in the pool; in push mode it also puts the uid list on the private
   `_input` queue. The assigner task (`start()` / `stop()`) places each item in chunks of
   `assign_chunk_size`, yielding between chunks (so a huge update never holds the loop), and
   puts **one uid list per child per input item** on `child.outbox`. Uids no child can take go
   on `scheduler.unplaced` (the master fails them); `scheduler.drained` is set while nothing
   unfinished is left. Pull mode (work stealing) only pools, and `request(child_id, n)` places
   on demand. `child.uids` is the source of truth: queues may hold stale uids, dropped where
   they are consumed (assigner: not in the pool; sender: not in the child). On the master,
   `_child_sender` per child takes a window slot (`max_inflight_task_sends`, default 1) *then*
   an outbox item and hands it straight to `_schedule_task_flush(child_id, msg)`;
   `_task_flush_done` frees the slot, or resends the same message object in the same slot (so
   the child can drop a duplicate). No task-flush timer: `task_flush_interval` and
   `task_buffer_size` no longer affect masters. `reinstate_child` refills the outbox for a
   relaunched child; if its resources can't be re-allocated, `requeue(child.uids)` places its
   tasks on the other children (the retry loop and `_give_up_on_child` find no record to act
   on, so nothing else would). A child given up on (restarts off, retries used up -- including
   a reinstated child that never relaunched, so READY -- or still failing at startup) stays as
   a FAILED record: `_give_up_on_child` is `transition(child_id, FAILED, requeue=True)`.
   Any terminal move (SUCCESS or FAILED) empties the child -- its tasks go back to the pool,
   then `requeue=True` places them on the other children and `requeue=False` puts them on
   `unplaced` (failed). SUCCESS is never refused: on `Done` the master moves the child to
   SUCCESS (no requeue), so tasks it still held (e.g. placed after it took `Stop`) are failed
   instead of blocking the stop. Placement
   skips terminal children, and `_wait_for_child_done` ends on `Done` *or* the child turning
   terminal, so the stop path never waits on a failed child.

   **Result routing is event driven on masters.** Every result batch -- streamed from a child,
   or made by `_fail_tasks` -- goes through `scheduler.queue_result(batch, child_id)`: it
   finishes the tasks (`finish_tasks`) and puts the batch, split by row per destination, on the
   submitting client's outbox (`client_outboxes`, made on first use and announced on the
   `new_clients` queue) or on the one `parent_outbox` (a node has one parent). Never call
   `finish_tasks` from the orchestrator. Per destination, `_result_sender` takes a window slot
   (`max_inflight_flushes`, default 1), then an item **and everything else already queued**,
   and sends them as one `IResultBatch`; `_result_flush_done` marks the items done on delivery
   (so `outbox.join()` means "delivered") or resends the same message in the same slot. The
   root has no parent sender: its `parent_outbox` is the run's results, returned by `run()`.
   **Workers keep the timer** (`result_flush_interval`, `result_buffer_size`): results finish
   one at a time there, and without it each would be its own message, landing on the masters.
   Masters no longer read either setting.

   **`Done`, not a final `ResultBatch`.** A node streams every result, and when it finishes it
   delivers everything still queued (`_flush_results` / the worker's `_drain_results`), sends
   its final `Status`, then `Done`. Results are buffered by type on the receiver, so when the
   parent reads `Done` every result from that child is there; the parent then marks the child
   SUCCESS (failing anything it still held). `ResultBatch` is only what `run()` returns and what the `{node}_results.bin`
   fallback holds (results that could not be delivered; no `Done` is sent then).

   **One start path, one stop path.** Only the root is built with tasks (its first
   `queue()`). A child is started by heartbeat sync, then the parent pushes its `NodeUpdate`
   and waits for the ACK, then `transition(RUNNING)` sets `ready` and its sender starts --
   no `NodeRequest` / initial `TaskRequest` round trips, and every child always runs its
   task-update monitor. A sub-master whose children policy has `layout_needs_tasks` (bin
   packing) takes its parent's first `TaskUpdate` before laying out children. No node decides
   by itself that it is finished: every node stops on its parent's `Stop`, its stop signal
   (SIGTERM or `request_stop()`), or its parent dying. A master's `_wait_for_finish` is the
   whole stop path, in order: stop condition -> propagate `Stop` -> every child's `Done` ->
   drain the forwarders -> `_fail_unfinished` (everything left in the pool or a child's `uids`,
   from `state.unfinished()`, via `_fail_tasks`) -> `_flush_results` -> final status + `Done`.
   Batch mode lives only in
   `ensemble_launcher.py`: `run_batch(node)` awaits `node.wait_drained()` then calls
   `request_stop()`. Tests that run a node directly use `run_batch` too.

   **Worker scheduler state.** Same pattern on `AsyncTaskScheduler`: `scheduler.state` is a
   `TaskSchedulerState` dataclass -- `pending_tasks` (the pending container itself),
   `running_tasks` (uid -> resource), `completed_tasks`, `failed_tasks`, `task_to_client`,
   `all_done`. The worker reads it directly; task policies get the object itself as
   `scheduler_state` (read-only, no snapshot is built). `get_state()` writes running tasks as
   pending; `set_state` keeps the current pending container (`_monitor_resources` parks on
   its event) and re-queues what the store still holds.

   **Pending-task containers.** `AsyncTaskScheduler` picks its pending container by policy
   *name*: `"fifo_policy"` gets `PendingTaskQueue` (deque, tasks never scored); everything
   else gets `PendingTaskHeap` (scored). An instance resolves to its name by exact-class
   lookup in the registry, so a `FIFOPolicy` subclass is scored like any custom policy.
   `_monitor_resources` dispatches with a synchronous `pop()` / `requeue(skipped)` pass --
   never sort or scan the whole pending set per cycle; that was O(n log n) per cycle and
   made throughput fall as the queue grew. (`Policy.needs_scheduler_state` is no longer
   read: the live state is passed, so there is nothing to skip.)

   The pass trusts `AsyncLocalClusterResource._resource_available`: a failed allocate never
   clears it, so if it is ever set on a full cluster, backfill tests *every* pending task.
   `deallocate()` therefore sets it **synchronously** (it always runs on the loop thread).
   Deferring the set with `call_soon_threadsafe` let it fire after the freed slot was
   refilled, and that O(pending)-per-completion scan was the main cause of throughput
   falling with batch size. Known, accepted cost: with *mixed* task sizes, backfill still
   scans past big tasks parked at the head to find one that fits (O(pending) per dispatch);
   strict priority does not. Per-shape sub-queues were rejected as too complex given
   per-task affinity.

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
- The IOProcess receive loop (`IOProcess._recv_loop`) only verifies and ACKs when
  `req_res=True`, and in that mode it auto-ACKs and strips the msg_id frame. So the policy wire format carries no msg_id, correlating
  replies by an explicit `request_id` instead.
- `ServerConnection.verify_sender` short-circuits to `True` when no expected remotes are
  configured, which is exactly this endpoint's situation — so the shared-secret check is
  enforced at the application layer in `PolicyEndpoint._handle`, not by the connection.
- `AsyncConnection.send()` (really `IOProcess._send`) does **not** retry internally; it
  returns `False` on ACK timeout.
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
  `simple_split_children_policy` / `fifo_policy`)
- `policy_config`: `PolicyConfig(nlevels, nchildren, leaf_nodes, strict_priority, initial_state)`
- `req_res`: enables ACK-based guaranteed delivery at the comm layer (default `True`)
- `send_retries`: number of retry attempts on ACK timeout, only when `req_res=True` (default `10`)
- `send_timeout`: per-attempt ACK timeout in seconds, only when `req_res=True` (default `1.0`)
- `comm_io_process`: run each node's data connections (socket, ACKs, sends) in a process of
  their own, so a busy event loop can't delay ACKs (default `False`; `req_res=True` only)

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
- Profiling is external, via py-spy: `el profile --checkpoint-dir X --node main --node 'main.m*'`
  (`profiling/pyspy.py`) reads each node's host/pid from `{node}_meta.json` and runs
  `py-spy record` on that host, over ssh when remote. Traces land in
  `profiles/{node}_pyspy.json`; `python -m ensemble_launcher.profiling.merge_traces` shifts
  py-spy's relative timestamps by each sidecar's `start_us`, gives every node its own pid, and
  remaps py-spy's 64-bit pthread ids to small tids named by role (`event-loop`, `thread-pool N`)
  -- Perfetto's JSON importer folds raw ones onto a single track.
  There is no in-process profiler (`EventRegistry` / `LauncherConfig.profile` were removed).

### Auto-configuration logic

When `launcher_config=None`, `EnsembleLauncher.__init__` auto-selects:
- If all tasks have `nnodes*ppn == 1` → `async_processpool`; otherwise `async_mpi`
- 1 node → `nlevels=0`, `async_zmq`
- 2–64 nodes → `nlevels=1`; 65–2048 → `nlevels=2`; 2048+ → `nlevels=3`
