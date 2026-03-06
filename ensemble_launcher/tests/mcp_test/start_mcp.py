import socket

from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.mcp import Server
from sim_script import sim

mcp = Server(
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
    ),
    Nodes=[socket.gethostname()],
)


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
    mcp.run()