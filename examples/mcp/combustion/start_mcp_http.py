import os
import socket
import time
import uuid

from flamespeed import compute_flame_speed
from mcp.server.fastmcp import FastMCP

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.mcp import ELFastMCP


def start_mcp():
    CHECKPOINT_DIR = os.path.join(os.getcwd(), f"ckpt_{str(uuid.uuid4())}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=12, cpus=list(range(12))),
        launcher_config=LauncherConfig(
            worker_logs=True,
            master_logs=True,
            cluster=True,
            cpu_binding_option="",
            checkpoint_dir=CHECKPOINT_DIR,
            nlevels=1,
            nchildren=1,
        ),
        Nodes=[socket.gethostname()],
    )

    el.start()
    time.sleep(10.0)

    mcp = ELFastMCP(name="combustion_mcp", checkpoint_dir=CHECKPOINT_DIR, port=8295)

    mcp.ensemble_tool(compute_flame_speed)

    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    start_mcp()
