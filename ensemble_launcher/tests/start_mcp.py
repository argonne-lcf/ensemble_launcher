import argparse
import asyncio
import multiprocessing as mp
import os
import socket
import time
import uuid

from utils import compute_density

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.mcp import Interface


def start_mcp():
    CHECKPOINT_DIR = f"{os.getcwd()}/mcp_{str(uuid.uuid4())}"

    logger = setup_logger("start_mcp", log_dir=f"{os.getcwd()}/logs")
    # --- Start the EnsembleLauncher cluster before creating the interface ---
    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            nlevels=0,
            return_stdout=True,
            worker_logs=True,
            cpu_binding_option="",
            use_mpi_ppn=False,
            cluster=True,
            checkpoint_dir=CHECKPOINT_DIR,
        ),
        Nodes=[socket.gethostname()],
    )
    el.start()
    time.sleep(2.0)  # wait for cluster to be ready
    logger.info("Done starting el")

    # --- Create the MCP interface, pointing it at the running cluster ---
    mcp = Interface(checkpoint_dir=CHECKPOINT_DIR)

    # Single-call cluster tool — one task per MCP invocation
    mcp.tool(compute_density, nnodes=1, ppn=1)

    # Batch ensemble tool — accepts lists, one task per list element
    mcp.ensemble_tool(compute_density, nnodes=1, ppn=1)

    logger.info("Done registering tools")

    mcp.run()


if __name__ == "__main__":
    start_mcp()
