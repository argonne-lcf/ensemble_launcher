import socket
import time

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.mcp import Interface
from sim_script import sim

CHECKPOINT_DIR = "/tmp/mcp_el_ckpt"

# --- Start the EnsembleLauncher cluster before creating the interface ---
el = EnsembleLauncher(
    ensemble_file={},
    system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
    launcher_config=LauncherConfig(
        task_executor_name="async_processpool",
        comm_name="async_zmq",
        nlevels=0,
        return_stdout=True,
        worker_logs=False,
        cpu_binding_option="",
        use_mpi_ppn=False,
        cluster=True,
        checkpoint_dir=CHECKPOINT_DIR,
    ),
    Nodes=[socket.gethostname()],
)
el.start()
time.sleep(2.0)  # wait for cluster to be ready

# --- Create the MCP interface, pointing it at the running cluster ---
mcp = Interface(checkpoint_dir=CHECKPOINT_DIR)


# Single-call cluster tool — one task per MCP invocation
@mcp.tool(nnodes=1, ppn=1)
def run_sim(Temperature: float, Pressure: float) -> str:
    """Runs a CFD simulation at a given temperature and pressure."""
    return sim(Temperature, Pressure)


# Batch ensemble tool — accepts lists, one task per list element
@mcp.ensemble_tool(nnodes=1, ppn=1)
def run_sim_ensemble(Temperature: float, Pressure: float) -> str:
    """Runs a CFD simulation at a given temperature and pressure."""
    return sim(Temperature, Pressure)


if __name__ == "__main__":
    try:
        mcp.run()
    finally:
        el.stop()