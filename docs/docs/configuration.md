# Configuration

## Auto-Configuration

The launcher automatically configures itself based on your workload and system:

```python
from ensemble_launcher import EnsembleLauncher

el = EnsembleLauncher(
    ensemble_file="config.json",
    Nodes=["node-001", "node-002"],  # Optional: auto-detects from PBS_NODEFILE
)
```

## SystemConfig

Describes the resources available on each node:

```python
from ensemble_launcher.config import SystemConfig

system_config = SystemConfig(
    name="my_cluster",
    ncpus=104,
    ngpus=12,
    cpus=list(range(104)),
    gpus=list(range(12))
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | Required | Cluster name |
| `ncpus` | `int` | `cpu_count()` | CPUs per node |
| `ngpus` | `int` | `0` | GPUs per node |
| `cpus` | `List[int]` | `[]` | Specific CPU IDs |
| `gpus` | `List[Union[str, int]]` | `[]` | Specific GPU IDs |

### GPU Overloading

You can overload GPUs by repeating IDs in the `gpus` list. The scheduler treats each entry as a separate GPU slot:

```python
system_config = SystemConfig(
    name="my_cluster",
    cpus=list(range(104)),
    gpus=['0', '0', '1', '1', '2', '3']  # GPU 0 and 1 are overloaded
)
```

The scheduler sees 6 GPU slots instead of 4, allowing more tasks to share GPUs 0 and 1.

## LauncherConfig

Controls orchestration behavior:

```python
from ensemble_launcher.config import LauncherConfig

launcher_config = LauncherConfig(
    child_executor_name="async_mpi",
    task_executor_name="async_processpool",
    comm_name="async_zmq",
    policy_config=PolicyConfig(nlevels=2),
    report_interval=10.0,
    return_stdout=True,
    worker_logs=True,
    master_logs=True,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `child_executor_name` | `str` | `"async_processpool"` | Executor for launching sub-master/worker processes |
| `task_executor_name` | `str` or `list` | `"async_processpool"` | Executor for running tasks |
| `comm_name` | `Literal["async_zmq"]` | `"async_zmq"` | Communication backend |
| `report_interval` | `float` | `10.0` | Status update frequency (seconds) |
| `return_stdout` | `bool` | `False` | Capture task stdout |
| `worker_logs` | `bool` | `False` | Enable worker logging |
| `master_logs` | `bool` | `False` | Enable master logging |
| `profile` | `Literal["perfetto"]` or `None` | `None` | Perfetto profiling for timeline visualization |
| `gpu_selector` | `str` | `"ZE_AFFINITY_MASK"` | Environment variable for GPU affinity |
| `cluster` | `bool` | `False` | Enable cluster mode |
| `checkpoint_dir` | `str` or `None` | `None` | Checkpoint directory for cluster mode |
| `enable_workstealing` | `bool` | `False` | Enable work-stealing scheduler |
| `children_scheduler_policy` | `str` | `"simple_split_children_policy"` | Policy for partitioning resources across children |
| `task_scheduler_policy` | `str` | `"large_resource_policy"` | Policy for task scoring/priority |
| `policy_config` | `PolicyConfig` | `PolicyConfig()` | Configuration passed to scheduling policies |
| `req_res` | `bool` | `True` | Enable ACK-based guaranteed delivery for task messages |
| `send_retries` | `int` | `10` | Retry count on ACK timeout (`req_res=True` only) |
| `send_timeout` | `float` | `1.0` | Per-attempt ACK timeout in seconds (`req_res=True` only) |


### PolicyConfig

Hierarchy and scheduling parameters passed to children policies:

```python
from ensemble_launcher.config import PolicyConfig

policy_config = PolicyConfig(
    nlevels=2,
    nchildren=4,
    leaf_nodes=64,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `nlevels` | `int` | `1` | Hierarchy depth |
| `nchildren` | `int` | `1` | Number of children per master |
| `leaf_nodes` | `int` | `1` | Target number of leaf (worker) nodes |
| `strict_priority` | `bool` | `False` | If `True`, tasks are scheduled in strict priority order |

`PolicyConfig` accepts extra fields (`extra="allow"`) so custom policies can define their own parameters.

## Resource Pinning

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

The `gpu_selector` option in `LauncherConfig` controls which environment variable is set for GPU affinity. It defaults to `"ZE_AFFINITY_MASK"` (Intel GPUs). For NVIDIA GPUs, set it to `"CUDA_VISIBLE_DEVICES"`.

## JSON Configuration Files

Both system and launcher configs can be provided as JSON files via the CLI:

**system.json:**
```json
{
    "name": "my_cluster",
    "ncpus": 104,
    "ngpus": 12,
    "cpus": [0, 1, 2, 3, 4],
    "gpus": [0, 1, 2, 3]
}
```

**launcher.json:**
```json
{
    "child_executor_name": "async_mpi",
    "task_executor_name": "async_mpi",
    "comm_name": "async_zmq",
    "nlevels": 2,
    "report_interval": 10.0,
    "return_stdout": true,
    "worker_logs": true,
    "master_logs": true
}
```

## Performance Tuning

### Communication

The communication layer uses a transport-agnostic design with pluggable backends. ZMQ is the default transport for both data and heartbeat channels, supporting 1 to 2048+ nodes.

### Monitoring and Debugging

```python
launcher_config = LauncherConfig(
    worker_logs=True,
    master_logs=True,
    report_interval=5.0,
    profile="basic",       # Communication latencies and task runtime
    # profile="timeline",  # Mean, std, sum, counts of orchestrator events
)
```

Logs are written to `logs/master-*.log` and `logs/worker-*.log`. Profiles go to `profiles/`.
