import argparse
import asyncio
import atexit
import multiprocessing as mp
import os
import signal
import socket
import uuid

from utils import async_compute_density, compute_density

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.mcp import ELFastMCP


def start_mcp():
    CHECKPOINT_DIR = f"/tmp/mcp_{str(uuid.uuid4())}"

    logger = setup_logger("start_mcp", log_dir=f"{os.getcwd()}/logs")
    # --- Start the EnsembleLauncher cluster before creating the interface ---
    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=0),
            return_stdout=True,
            cluster=True,
            checkpoint_dir=CHECKPOINT_DIR,
        ),
        Nodes=[socket.gethostname()],
    )
    el.start()

    def _stop_el():
        el.stop()

    atexit.register(_stop_el)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: (_stop_el(), exit(0)))

    logger.info("Done starting el")

    # --- Create the MCP interface, pointing it at the running cluster ---
    mcp = ELFastMCP(checkpoint_dir=CHECKPOINT_DIR)

    # Single-call cluster tool — one task per MCP invocation
    mcp.tool(compute_density, nnodes=1, ppn=1)

    # Batch ensemble tool — accepts lists, one task per list element
    mcp.ensemble_tool(compute_density, nnodes=1, ppn=1)

    # Async variants — interface auto-detects async def and uses AsyncTask
    mcp.tool(async_compute_density, nnodes=1, ppn=1)
    mcp.ensemble_tool(async_compute_density, nnodes=1, ppn=1)

    # String command tools — return value is stdout; cluster must have return_stdout=True
    mcp.tool(
        "python3 py_echo.py {task_id}",
        name="py_echo",
        description="Echo a task ID via shell command. Prints 'Hello from task <task_id>' to stdout.",
        nnodes=1,
        ppn=1,
    )
    mcp.ensemble_tool(
        "python3 py_echo.py {task_id}",
        name="py_echo",
        description="Echo multiple task IDs via shell command. Prints 'Hello from task <task_id>' to stdout.",
        nnodes=1,
        ppn=1,
    )

    logger.info("Done registering tools")

    mcp.run()


if __name__ == "__main__":
    start_mcp()
