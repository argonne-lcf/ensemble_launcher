# API Reference

## EnsembleLauncher

```python
from ensemble_launcher import EnsembleLauncher

EnsembleLauncher(
    ensemble_file: Union[str, Dict[str, Dict]],
    system_config: SystemConfig = SystemConfig(name="local"),
    launcher_config: Optional[LauncherConfig] = None,
    Nodes: Optional[List[str]] = None,
)
```

| Parameter | Description |
|---|---|
| `ensemble_file` | Path to JSON config or dict of task definitions |
| `system_config` | System resource configuration |
| `launcher_config` | Launcher behavior configuration (auto-configured if `None`) |
| `Nodes` | List of compute nodes (auto-detected from `PBS_NODEFILE` if `None`) |

### Methods

| Method | Description |
|---|---|
| `run()` | Execute ensemble synchronously and return results. Raises `RuntimeError` in cluster mode. |
| `start()` | Start the orchestrator in a background process (cluster mode). |
| `stop()` | Send SIGTERM to background process; force-kill after 30s if needed. |
| `__enter__` / `__exit__` | Context manager -- calls `start()` on entry, `stop()` on exit. |

## ClusterClient

```python
from ensemble_launcher.orchestrator import ClusterClient

ClusterClient(
    checkpoint_dir: str,
    node_id: str = "global",
    client_id: Optional[str] = None,
)
```

| Parameter | Description |
|---|---|
| `checkpoint_dir` | Directory containing orchestrator checkpoint files |
| `node_id` | Node to connect to. `"global"` resolves to root master. |
| `client_id` | Optional identity string; auto-generated if omitted |

### Methods

| Method | Description |
|---|---|
| `start()` | Connect transport and start receive thread |
| `teardown()` | Disconnect and stop receive thread |
| `submit(task)` | Send a `Task` and return a `concurrent.futures.Future` |
| `__enter__` / `__exit__` | Context manager |

## PolicyClient

Reads and updates a node's scheduling-policy state while the run is in flight. Requires
`checkpoint_dir`, plus `cluster=True` or `enable_policy_client=True`.

```python
from ensemble_launcher.orchestrator import PolicyClient

PolicyClient(
    checkpoint_dir: str,
    node_id: str = "global",
    policy_kind: Literal["auto", "children", "task"] = "auto",
    client_id: Optional[str] = None,
    log_level: int = logging.INFO,
    checkpoint_timeout: float = 60.0,
    request_timeout: float = 30.0,
)
```

| Parameter | Description |
|---|---|
| `checkpoint_dir` | Directory containing orchestrator checkpoint files |
| `node_id` | Node to connect to. `"global"` resolves to root master. |
| `policy_kind` | Which policy to target. `"auto"` picks the node's only policy: `"task"` on a worker, `"children"` on a master. A mismatch is an error. |
| `client_id` | Optional identity string; auto-generated if omitted |
| `checkpoint_timeout` | How long to wait for the node to publish its endpoint |
| `request_timeout` | Default per-request timeout |

### Methods

| Method | Description |
|---|---|
| `start()` / `teardown()` | Connect / disconnect; also available as a context manager |
| `get_state(keys=None, timeout=None)` | Deep copy of the state. `keys` filters; absent keys are omitted. |
| `set_state(state, merge=True, rescore=False, timeout=None)` | Update and return the resulting state. `merge=False` replaces it wholesale. |
| `get_state_async` / `set_state_async` | Same, returning a `concurrent.futures.Future` |
| `node_id` / `policy_name` | Resolved node and the policy class name serving it |
| `PolicyClient.discover_nodes(ckpt)` | *(static)* List every node serving a policy endpoint |

`rescore=True` re-scores the worker's already-pending tasks against the new state; it is
off by default, so a plain `set_state` only affects tasks submitted afterwards. Failures
raise `PolicyStateError` and leave both the state and the pending queue unchanged.

## PolicyGroupClient

Fan-out wrapper over several `PolicyClient`s. Not atomic across nodes.

```python
from ensemble_launcher.orchestrator import PolicyGroupClient

PolicyGroupClient(
    checkpoint_dir: str,
    node_ids: Optional[List[str]] = None,   # None -> every discovered node
    **policy_client_kwargs,
)
```

`get_state` and `set_state` take the same arguments as `PolicyClient` and return
`{node_id: state}`.

## Task

```python
from ensemble_launcher.ensemble import Task

Task(
    task_id: str,
    nnodes: int = 1,
    ppn: int = 1,
    executable: Union[Callable, str] = None,
    args: tuple = (),
    kwargs: dict = {},
)
```

## AsyncTask

```python
from ensemble_launcher.ensemble import AsyncTask

AsyncTask(
    task_id: str,
    nnodes: int = 1,
    ppn: int = 1,
    executable: Union[Callable, str] = None,
    args: tuple = (),
    kwargs: dict = {},
)
```

For use with async callables. Automatically used when registering `async def` functions via MCP tools.

## SystemConfig

```python
from ensemble_launcher.config import SystemConfig

SystemConfig(
    name: str,
    ncpus: int = cpu_count(),
    ngpus: int = 0,
    cpus: List[int] = [],
    gpus: List[Union[str, int]] = [],
)
```

## LauncherConfig

```python
from ensemble_launcher.config import LauncherConfig

LauncherConfig(
    child_executor_name: str = "async_processpool",
    task_executor_name: Union[str, List[str]] = "async_processpool",
    comm_name: Literal["async_zmq"] = "async_zmq",
    report_interval: float = 10.0,
    return_stdout: bool = False,
    worker_logs: bool = False,
    master_logs: bool = False,
    profile: Optional[Literal["perfetto"]] = None,
    gpu_selector: str = "ZE_AFFINITY_MASK",
    cluster: bool = False,
    checkpoint_dir: Optional[str] = None,
    enable_workstealing: bool = False,
    enable_policy_client: bool = False,
    children_scheduler_policy: str = "simple_split_children_policy",
    task_scheduler_policy: str = "large_resource_policy",
    policy_config: PolicyConfig = PolicyConfig(),
)
```

`enable_policy_client` serves each node's policy state for `PolicyClient` without requiring
full cluster mode; `cluster=True` implies it. Either way `checkpoint_dir` must be set,
since clients discover the endpoint from disk.

## PolicyConfig

```python
from ensemble_launcher.config import PolicyConfig

PolicyConfig(
    nlevels: int = 1,
    nchildren: int = 1,
    leaf_nodes: int = 1,
    strict_priority: bool = False,
    initial_state: Dict[str, Any] = {},
)
```

`initial_state` seeds the policy's `self.state` dict -- the state `PolicyClient` reads and
updates at runtime. See [Custom Scheduling](custom-scheduling.md#stateful--auto-tunable-policies).

## Preset System Configs

```python
from ensemble_launcher.config import aurora_config, polaris_config, get_system_config

# ALCF Aurora (Intel XPU)
sys_config = aurora_config

# ALCF Polaris (NVIDIA A100)
sys_config = polaris_config

# Auto-detect from hostname
sys_config = get_system_config()
```
