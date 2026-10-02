# Custom Scheduling

Ensemble Launcher uses a pluggable policy system for scheduling tasks and partitioning resources across the orchestrator hierarchy.

## Built-in Policies

### Children Policies

Children policies decide how to partition cluster resources among child workers and distribute tasks to them.

| Policy Name | Description |
|---|---|
| `bin_packing_children_policy` | Greedily bin-packs tasks onto workers based on resource requirements. Configurable via `nlevels`. |
| `simple_split_children_policy` | Splits nodes evenly across a fixed number of children. Set `nchildren` to control the split. |
| `fixed_leafs_children_policy` | Extends `SimpleSplitChildrenPolicy` to target a specific number of leaf (worker) nodes. Set `leaf_nodes` in `PolicyConfig`. |

### Task Scoring Policies

Task scoring policies determine the priority order in which tasks are assigned to a worker.

| Policy Name | Description |
|---|---|
| `large_resource_policy` | Prioritizes tasks that require more resources (nodes, GPUs). |
| `fifo_policy` | First-in, first-out ordering. |

## Writing a Custom Children Policy

Subclass `ChildrenPolicy` and implement two methods:

```python
from ensemble_launcher.scheduler import ChildrenPolicy, policy_registry
from ensemble_launcher.scheduler.resource import JobResource
from ensemble_launcher.ensemble import Task

@policy_registry.register("my_custom_policy", type="children_policy")
class MyCustomPolicy(ChildrenPolicy):
    def get_children_resources(self, tasks, nodes, level):
        """
        Decide how many child workers to create and what resources each gets.

        Args:
            tasks: Dict mapping task IDs to Task objects.
            nodes: JobResource containing available nodes and per-node resources.
            level: Current hierarchy level (0 = root master).

        Returns:
            Dict mapping integer worker IDs to JobResource allocated to each.
        """
        # Example: one worker per node
        result = {}
        for i, node in enumerate(nodes):
            result[i] = JobResource([node])
        return result

    def get_children_tasks(self, tasks, children_resources, ntask=None,
                           child_assignments=None, child_status=None,
                           level=None, **kwargs):
        """
        Distribute tasks across workers given their pre-allocated resources.

        Args:
            tasks: Dict mapping task IDs to Task objects.
            children_resources: Dict mapping worker ID to JobResource.
            ntask: Max tasks per worker (None = no limit).
            child_assignments: Current assignment state per worker.
            child_status: Most recent Status message per worker.
            level: Current hierarchy level.

        Returns:
            Tuple of:
            - Dict mapping worker ID to list of task IDs assigned.
            - Dict mapping task ID to worker ID.
            - List of task IDs that could not be assigned.
        """
        assignments = {}
        task_to_worker = {}
        unassigned = []
        # Your assignment logic here
        return assignments, task_to_worker, unassigned
```

## Writing a Custom Task Scoring Policy

Subclass `Policy` and implement the `get_score` method:

```python
from ensemble_launcher.scheduler import Policy, policy_registry

@policy_registry.register("my_scoring_policy")
class MyScoringPolicy(Policy):
    def get_score(self, task, scheduler_state=None):
        """Return a numeric score; higher = higher priority."""
        return task.nnodes * task.ppn

    def on_task_complete(self, task, status, scheduler_state):
        """Optional: react to task completions."""
        pass
```

## Stateful / Auto-tunable Policies

Every policy -- both families -- carries a mutable `self.state` dict. Policies can
therefore react to **workflow** state, not just to the `SchedulerState` snapshot they are
handed on each call. An external process reads and updates that state while the run is in
flight, via `PolicyClient`.

### Reading state inside a policy

Seed the state from `PolicyConfig.initial_state` and read it wherever you make a decision:

```python
@policy_registry.register("gpu_weighted_policy")
class GPUWeightedPolicy(Policy):
    def get_score(self, task, scheduler_state=None):
        # self.state is live -- it reflects the most recent set_state.
        gpus = task.ngpus_per_process * task.ppn * task.nnodes
        return gpus * self.state.get("gpu_weight", 1.0)

    def on_state_update(self, changed):
        """Optional hook, called after every state update.

        Use it to recompute anything derived from the state. MUST be synchronous
        and non-blocking -- see the contract below.
        """
        self._threshold = self.state.get("gpu_weight", 1.0) * 2
```

```python
launcher_config = LauncherConfig(
    task_scheduler_policy="gpu_weighted_policy",
    policy_config=PolicyConfig(initial_state={"gpu_weight": 1.0}),
    checkpoint_dir="/scratch/my_run",
    enable_policy_client=True,
)
```

### Tuning it from outside

```python
from ensemble_launcher.orchestrator import ClusterClient, PolicyClient

with ClusterClient(checkpoint_dir=ckpt) as cc, \
     PolicyClient(ckpt, node_id="main.w0") as pc:

    print(pc.get_state())                       # {'gpu_weight': 1.0}
    pc.set_state({"gpu_weight": 8.0})           # merge by default
    futs = [cc.submit(t) for t in next_round]
```

`PolicyClient` requires `checkpoint_dir` to be set, plus either `cluster=True` or
`enable_policy_client=True`. Discovery is file-based: each node publishes its endpoint
address to `{node_id}_policy.ckpt` in its checkpoint directory.

| Method | Notes |
|---|---|
| `get_state(keys=None)` | Returns a deep copy. `keys` filters; absent keys are omitted, not `None`. |
| `set_state(state, merge=True, rescore=False)` | `merge=False` replaces the whole state. Returns the resulting state. |
| `get_state_async` / `set_state_async` | Same, returning a `concurrent.futures.Future`. |
| `PolicyClient.discover_nodes(ckpt)` | Lists every node serving an endpoint. |
| `PolicyGroupClient(ckpt, node_ids=None)` | Fans out; returns `{node_id: state}`. |

A `PolicyGroupClient` call is **not** atomic across nodes, though each individual node's
update is all-or-nothing.

### Which policy gets tuned

Each node hosts exactly one policy, so `policy_kind` defaults to `"auto"`:

- a **worker** hosts its task policy, kind `"task"`
- a **master** hosts its children policy, kind `"children"`

Passing a `policy_kind` the node does not host is an error, never a silent write to the
wrong policy.

### `rescore` -- and why it defaults to off

A task's priority is computed **once**, when it enters the worker's pending heap. The heap
is a cache of scores, so changing policy state does nothing to work already queued:
by default `set_state` only affects tasks submitted *afterwards*.

`set_state(..., rescore=True)` invalidates that cache, re-scoring every pending task
against the new state. It is off by default because reordering a live queue is a visible
scheduling change that costs one `get_score` per pending task, run synchronously on the
orchestrator's event loop -- it should never happen as a side effect of a tune the caller
thought was passive.

Scope: the pending heap only. Running tasks are already dispatched and are not reordered;
completed and failed tasks are untouched. Rescoring is worker-only -- on a master the
response reports `rescored: None`, and the new children-policy state takes effect at the
next natural assignment point instead.

### Safety guarantees

- **Atomic with respect to policy decisions.** Everything runs on one event loop and the
  endpoint's apply path is fully synchronous, so a `set_state` can never land in the middle
  of a scoring or assignment decision.
- **A failed tune is a no-op.** If the new state makes `get_score` raise -- which surfaces
  during the rescore -- both the state and the pending heap are rolled back and the client
  gets a `PolicyStateError`. Without this, one bad `set_state` would wedge the worker,
  since every subsequent submission calls `get_score`.
- **Tuned state is durable.** It is checkpointed and restored, so it survives a node
  restart rather than silently reverting to `initial_state`.

### Contract for policy authors

`set_policy_state` and `on_state_update` run on the orchestrator's event loop. They **must
be synchronous and non-blocking**: no file I/O, no network calls, no `time.sleep`. The
atomicity guarantee above depends on it.

## Loading External Policies

Custom policies can be loaded at runtime without modifying the Ensemble Launcher source. Set these environment variables before launching:

| Variable | Description |
|---|---|
| `EL_EXTERNAL_POLICY_MODULE` | Python module to import (e.g. `my_custom_policies`) |
| `EL_EXTERNAL_POLICY_PATH` | Directory to add to `sys.path` before importing |

```bash
export EL_EXTERNAL_POLICY_PATH=/path/to/my/policies
export EL_EXTERNAL_POLICY_MODULE=my_policies

el start my_ensemble.json --launcher-config-file launcher.json
```

The module is imported at orchestrator startup. Any classes decorated with `@policy_registry.register(...)` in that module become available for use in your `LauncherConfig`.

## Using a Custom Policy

Reference the registered policy name in your `LauncherConfig`:

```python
from ensemble_launcher.config import LauncherConfig, PolicyConfig

launcher_config = LauncherConfig(
    children_scheduler_policy="my_custom_policy",
    policy_config=PolicyConfig(
        nlevels=2,
        leaf_nodes=64,
    ),
)
```

`PolicyConfig` is passed to the policy constructor, so you can add fields there to parameterize your policy.

## Programmatic Registration

You can also register policies programmatically without using the decorator:

```python
from ensemble_launcher.scheduler import policy_registry

policy_registry.register_policy(
    "my_policy",
    MyPolicyClass,
    type="children_policy"
)
```
