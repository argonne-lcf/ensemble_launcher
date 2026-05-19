# Multi-Agent Joke Generator

A multi-agent example using [pydantic-ai](https://ai.pydantic.dev/) and Ensemble Launcher. A **Selector** agent asks a **Generator** agent to produce jokes, then picks the best one — all coordinated as distributed actors over ZMQ.

## Architecture

```
┌────────────┐       ┌─────────────────────┐   ZMQ  ┌─────────────────────┐
│   Client    │──────▶│  JokeSelectionAgent │──────▶│  JokeGenerationAgent│
│ (submitter) │       │  (pydantic-ai +     │  tool │  (pydantic-ai +     │
│             │◀──────│   tool calling)     │◀──────│   text generation)  │
└────────────┘       └────────┬────────────┘       └────────┬────────────┘
                              │                             │
                              │  API call.                  │  API call
                              ▼                             ▼
                        ┌───────────┐                 ┌───────────┐
                        │   Remote  │                 │  Local    │
                        │    LLM    │                 │   LLM     │
                        └───────────┘                 └───────────┘
                                                 OnlineVLLMInference Actor
```

1. **OnlineVLLMInference** — launches a vLLM server as a `PublicActor`; exposes `get_address()` and `model()` actions.
2. **JokeGenerationAgent** — wraps a pydantic-ai `Agent` that generates jokes via the vLLM server.
3. **JokeSelectionAgent** — wraps a pydantic-ai `Agent` with `generate_jokes` registered as a tool. The LLM decides when to call the tool, the tool call is routed over ZMQ to the Generator, and the result is returned to the Selector for final selection.

## Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
pip install pydantic-ai openai requests huggingface_hub
pip install vllm
```
On macos, you may to install [vllm-metal-plugin](https://github.com/vllm-project/vllm-metal). Make sure to modify the install script to change the environment to `venv`

A tool-calling-capable model must be downloaded locally. The example uses `Qwen/Qwen3-0.6B` but any instruct model with a chat template works:

```bash
hf download Qwen/Qwen3-0.6B
```

## Configuration

| Environment Variable | Description | Default |
|---|---|---|
| `HF_HOME` | HuggingFace cache directory containing the model | Required |

The vLLM server is started with `--enable-auto-tool-choice --tool-call-parser hermes` to support pydantic-ai tool calling.

## Running

```bash
cd examples/agents/pydantic
export HF_TOKEN=your_hf_token
export HF_HOME=/path/to/your/hf/cache
export HF_HUB_CACHE=/path/to/your/hf/cache/hub
python joke_generator.py
```

## Files

| File | Description |
|---|---|
| `joke_generator.py` | Entry point — wires up the actors, submits tasks, and invokes the selector |
| `agents.py` | Defines `JokeGenerationAgent` and `JokeSelectionAgent` (Actor subclasses) |
| `utils.py` | Helper to start Ensemble Launcher in cluster mode |

## How it works

1. `joke_generator.py` starts Ensemble Launcher in cluster mode and submits three actors as tasks: the vLLM server, the generator, and the selector.
2. Each actor gets an `AgentHandle` to communicate with its dependencies over ZMQ.
3. The client calls `selector_handle.invoke("Generate 10 jokes and tell me the best one among those to me.")`.
4. The selector's pydantic-ai agent decides to call `generate_jokes` (registered as a tool via `tool_plain`).
5. The tool call is serialized and sent over ZMQ to the generator actor, which queries local vLLM and returns the jokes.
6. The selector's pydantic-ai agent picks the best joke and returns it to the client.
7. The client sends `stop()` to shut down the actors.
