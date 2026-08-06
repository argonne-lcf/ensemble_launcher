# Inference (vLLM)

!!! warning
    This feature is experimental and its API may change.

Ensemble Launcher provides actor-based wrappers around [vLLM](https://docs.vllm.ai/) for serving LLMs on HPC clusters. All inference classes are built on the [Actor model](actors.md), enabling distributed LLM serving with ZMQ-based communication.

## Installation

```bash
python3 -m pip install -e ".[inference]"
```

## Inference Modes

### 1. Offline Inference (In-Process)

Loads the model in-process using vLLM's `LLM` class. Best throughput for batch workloads.

| Class | Discovery | Description |
|---|---|---|
| `VLLMInference` | Public (checkpoint) | Standalone offline inference actor |
| `PrivateVLLMInference` | Private (connection) | Managed by an ActorPool |

```python
from ensemble_launcher.inference import VLLMInference

actor = VLLMInference(
    name="llm-offline",
    model="meta-llama/Llama-3-8B",
    cache_dir="/path/to/hf/cache",
    llm_kwargs={"max_model_len": 2048, "tensor_parallel_size": 1},
    async_engine=False,  # sync LLM (best batch throughput)
)
```

**Engine modes:**
- `async_engine=False` (default) -- Sync `LLM` with `InprocClient`. All prompts batch in one GPU kernel for best throughput.
- `async_engine=True` -- `AsyncLLMEngine` with `AsyncMPClient` (separate process + ZMQ). Supports `max_workers > 1` for concurrent `generate()` calls.

### 2. Online Inference (OpenAI-Compatible Server)

Starts an in-process uvicorn + FastAPI server that serves the vLLM model via an OpenAI-compatible API.

| Class | Discovery | Description |
|---|---|---|
| `OnlineVLLMInference` | Public (checkpoint) | Standalone server actor |
| `PrivateOnlineVLLMInference` | Private (connection) | Managed by an ActorPool |

```python
from ensemble_launcher.inference import OnlineVLLMInference

actor = OnlineVLLMInference(
    name="llm-server",
    model="meta-llama/Llama-3-8B",
    cache_dir="/path/to/hf/cache",
    server_args={"host": "0.0.0.0", "port": 8001},
)
```

### 3. Multi-Node Offline Inference

Distributes a single model across multiple nodes using MPI for tensor parallelism.

| Class | Discovery | Description |
|---|---|---|
| `MultiNodeVLLMInference` | Public (checkpoint) | Standalone multi-node actor |
| `PrivateMultiNodeVLLMInference` | Private (connection) | Managed by an ActorPool |

```python
from ensemble_launcher.inference import MultiNodeVLLMInference

actor = MultiNodeVLLMInference(
    name="llm-multinode",
    model="meta-llama/Llama-3-70B",
    cache_dir="/path/to/hf/cache",
    llm_kwargs={"tensor_parallel_size": 12},
)
```

### 4. Multi-Node Online Inference

Combines multi-node tensor parallelism with an OpenAI-compatible serving endpoint.

| Class | Discovery | Description |
|---|---|---|
| `MultiNodeOnlineVLLMInference` | Public (checkpoint) | Standalone multi-node server |
| `PrivateMultiNodeOnlineVLLMInference` | Private (connection) | Managed by an ActorPool |

## OpenAI-Compatible Client

For online inference actors, use the `OpenAIInference` actor or any OpenAI client:

```python
from ensemble_launcher.inference import OpenAIInference

client = OpenAIInference(
    name="openai-client",
    model="meta-llama/Llama-3-8B",
    base_url="http://hostname:8001/v1",
    api_key="not-needed",
)
```

## Model Distribution

The `copy_model` module provides helpers for distributing HuggingFace model weights across nodes:

```python
from ensemble_launcher.inference import copy_model

# Distribute model from rank 0 to all other ranks
copy_model.distribute_model(model_path, cache_dir)
```

## Launcher Config for Inference

Use `default_inference_launcher_config` for a pre-tuned cluster configuration:

```python
from ensemble_launcher.inference import default_inference_launcher_config

launcher_config = default_inference_launcher_config(
    nnodes=4,
    checkpoint_dir="/scratch/my_job/ckpt",
)
```

This sets up:
- Mixed executors: `async_processpool` for serial tasks, `async_mpi` for MPI tasks
- `fixed_leafs_children_policy` for stable node assignments
- Cluster mode enabled
- Optimized flush intervals (0.5s)

## Class Hierarchy

All inference classes inherit from `PublicActor` or `PrivateActor`:

```
PublicActor
├── VLLMInference          (offline, single-node)
├── OnlineVLLMInference    (online server, single-node)
├── MultiNodeVLLMInference (offline, multi-node MPI)
└── MultiNodeOnlineVLLMInference (online server, multi-node)

PrivateActor
├── PrivateVLLMInference
├── PrivateOnlineVLLMInference
├── PrivateMultiNodeVLLMInference
└── PrivateMultiNodeOnlineVLLMInference
```

Public variants discover each other via checkpoint files. Private variants are managed within an `ActorPool` via direct connections.
