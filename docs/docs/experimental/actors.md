# Actors

!!! warning
    This feature is experimental and its API may change.

The Actor model provides distributed async/await communication between independent processes over ZMQ. Actors are long-lived objects that expose methods (called **actions**) which can be invoked remotely.

## Core Concepts

### PublicActor vs PrivateActor

| Type | Discovery | Use Case |
|---|---|---|
| `PublicActor` | Writes checkpoint to disk; any process can connect via `ckpt_dir` | Standalone services, inference servers |
| `PrivateActor` | Connected via an existing `ClientConnection` (e.g. from `ActorPool`) | Managed workers in a pool |

### The `@action` Decorator

Mark methods as remotely callable:

```python
from ensemble_launcher.ensemble import Actor, action

class MyActor(PublicActor):
    @action
    def compute(self, x: float, y: float) -> float:
        return x + y

    @action
    async def async_compute(self, x: float) -> float:
        await asyncio.sleep(0.1)
        return x * 2

    async def on_start(self):
        """Called once when the actor starts. Use for initialization."""
        pass

    async def on_stop(self):
        """Called when the actor shuts down. Use for cleanup."""
        pass
```

Both sync and async actions are supported. Sync actions run in a `ThreadPoolExecutor` (configurable via `max_workers`).

## Using ActorHandle

An `ActorHandle` is a proxy that lets you invoke actions on a remote actor:

```python
# Create actor and get a handle
actor = MyActor(name="my-actor", transport="zmq", ckpt_dir="./ckpt")

# In another process:
handle = MyActor.create_handle(ckpt_dir="./ckpt")
await handle.open()

result = await handle.compute(1.0, 2.0)  # returns 3.0
await handle.stop()
await handle.close()
```

## ActorPool

`ActorPool` manages a pool of `PrivateActor` instances for batch workloads:

```python
from ensemble_launcher.ensemble import ActorPool

class MyWorker(PrivateActor):
    @action
    def process(self, data):
        return data * 2

pool = ActorPool(
    name="my-pool",
    actor_class=MyWorker,
    n_actors=4,
    transport="zmq",
)
```

### Key Methods

| Method | Description |
|---|---|
| `invoke_children(actor_index, action_name, args)` | Invoke action on a specific actor |
| `invoke_all_children(action_name, args_list)` | Invoke action on all actors in parallel |
| `get_n_actors()` | Return number of actors in the pool |
| `get_actor_ids()` | Return list of actor identifiers |

## Actor Parameters

### _ActorBase

| Parameter | Type | Default | Description |
|---|---|---|---|
| `name` | `str` | Required | Actor identifier |
| `max_workers` | `int` | `1` | Max concurrent action invocations |
| `run_in_executor` | `bool` | `True` | Run sync actions in ThreadPoolExecutor |
| `send_timeout` | `float` | `5.0` | Send timeout in seconds |
| `send_retries` | `int` | `3` | Number of send retries |

### PublicActor (additional)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `transport` | `str` | `"zmq"` | Transport backend |
| `ckpt_dir` | `str` | Auto-generated | Checkpoint directory for discovery |

### PrivateActor (additional)

| Parameter | Type | Default | Description |
|---|---|---|---|
| `client_conn` | `ClientConnection` | Required | Pre-established connection |

## Lifecycle

1. Actor is constructed with parameters
2. `on_start()` is called when the actor begins its main loop
3. Actions are invoked via handles (remote calls over ZMQ)
4. `stop()` action sets the stop flag
5. `on_stop()` is called during shutdown
