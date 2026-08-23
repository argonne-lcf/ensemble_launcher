# MCP Integration

!!! warning
    This feature is experimental and its API may change.

`ensemble_launcher.mcp.ELFastMCP` is a subclass of [FastMCP](https://github.com/modelcontextprotocol/python-sdk) that exposes two decorators for registering MCP tools backed by the EnsembleLauncher cluster:

- **`@mcp.tool`** -- submits a single task per MCP call
- **`@mcp.ensemble_tool`** -- accepts lists of arguments and runs one task per element (batch ensemble in a single call)

Both decorators automatically detect `async def` functions and create `AsyncTask` instead of `Task`.

The cluster lifecycle is decoupled from the MCP server: start `EnsembleLauncher` separately, then point `ELFastMCP` at its checkpoint directory.

## Minimal Example

```python
import socket
import time
import uuid
import os

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.mcp import ELFastMCP
from my_module import my_sim

CHECKPOINT_DIR = f"{os.getcwd()}/mcp_{uuid.uuid4()}"

# 1. Start the EnsembleLauncher cluster (non-blocking)
el = EnsembleLauncher(
    ensemble_file={},
    system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
    launcher_config=LauncherConfig(
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        nlevels=0,
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
    ),
    Nodes=[socket.gethostname()],
)
el.start()
time.sleep(2.0)

# 2. Create the MCP interface
mcp = ELFastMCP(checkpoint_dir=CHECKPOINT_DIR)

# 3. Register tools
mcp.tool(my_sim, nnodes=1, ppn=1)           # single-call
mcp.ensemble_tool(my_sim, nnodes=1, ppn=1)  # batch ensemble

# 4. Serve (stdio by default; also "sse" and "streamable-http")
mcp.run()
```

## Decorator Style

```python
@mcp.tool(nnodes=1, ppn=4)
def my_sim(a: float, b: float) -> str:
    ...

@mcp.ensemble_tool(nnodes=1, ppn=4)
def my_sim(a: float, b: float) -> str:
    ...
```

## Running via stdio

Configure your MCP client (e.g. Claude Desktop) to launch the server:

```json
{
    "mcpServers": {
        "my_sim": {
            "command": "python3",
            "args": ["start_mcp.py"]
        }
    }
}
```

## Port-Forwarding (HPC Login to Compute Node)

When the MCP server runs on a compute node and the client on a login node, use the SSH tunnel helpers:

```python
from ensemble_launcher.mcp import start_tunnel, stop_tunnel

ret = start_tunnel("<username>", "<head-node-hostname>",
                   local_port=9276, remote_port=9276)
# ... run your async client ...
stop_tunnel(*ret)
```

## Installation

```bash
python3 -m pip install -e ".[mcp]"
```

This installs the `mcp` and `paramiko` packages.
