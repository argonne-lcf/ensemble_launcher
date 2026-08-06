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

Subclass `Policy` and implement the `score` method:

```python
from ensemble_launcher.scheduler import Policy, policy_registry

@policy_registry.register("my_scoring_policy")
class MyScoringPolicy(Policy):
    def score(self, task, scheduler_state):
        """Return a numeric score; higher = higher priority."""
        return task.nnodes * task.ppn

    def on_task_complete(self, task, status, scheduler_state):
        """Optional: react to task completions."""
        pass
```

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
