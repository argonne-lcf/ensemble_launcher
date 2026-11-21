import json
import os
import signal
from typing import List, Optional

import typer

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, SystemConfig

_PID_FILE = ".el_launcher.pid"

el = typer.Typer()


def _load_system_config(system_config_file: Optional[str]) -> SystemConfig:
    try:
        with open(system_config_file, "r") as f:
            config_dict = json.load(f)
        return SystemConfig.model_validate(config_dict)
    except Exception as e:
        print(f"loading system configuration failed with exception {e}")
        print("Falling back to default local system configuration")
        return SystemConfig(name="local")


def _load_launcher_config(launcher_config_file: Optional[str]) -> Optional[LauncherConfig]:
    try:
        with open(launcher_config_file, "r") as f:
            config_dict = json.load(f)
        return LauncherConfig.model_validate(config_dict)
    except Exception as e:
        print(f"loading launcher configuration failed with exception {e}")
        print("Falling back to default launcher configuration")
        return None


@el.command()
def start(
    ensemble_file: str,
    system_config_file: Optional[str] = None,
    launcher_config_file: Optional[str] = None,
    nodes_str: Optional[str] = None,
    pin_resources: bool = True,
    async_orchestrator: bool = True,
):
    """Launch an ensemble of tasks. In cluster mode, starts the launcher in the background and returns immediately."""
    system_config = _load_system_config(system_config_file)
    launcher_config = _load_launcher_config(launcher_config_file)
    nodes = nodes_str.split(",") if nodes_str else None

    launcher = EnsembleLauncher(
        ensemble_file=ensemble_file,
        system_config=system_config,
        launcher_config=launcher_config,
        Nodes=nodes,
        pin_resources=pin_resources,
        async_orchestrator=async_orchestrator,
    )

    if launcher_config is not None and launcher_config.cluster:
        launcher.start()
        pid = launcher._launcher_process.pid
        with open(_PID_FILE, "w") as f:
            f.write(str(pid))
        print(f"Cluster launcher started (PID {pid}). Run 'el stop' to stop it.")
        return

    results = launcher.run()

    with open("results.json", "w") as f:
        json.dump(results.to_dict(), f, indent=2)


@el.command()
def stop():
    """Stop a running cluster launcher."""
    if not os.path.exists(_PID_FILE):
        print("No running cluster launcher found (no PID file).")
        raise typer.Exit(1)

    with open(_PID_FILE, "r") as f:
        pid = int(f.read().strip())

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to launcher process (PID {pid}).")
    except ProcessLookupError:
        print(f"Process {pid} not found. It may have already exited.")
    finally:
        os.remove(_PID_FILE)


@el.command()
def profile(
    checkpoint_dir: str = typer.Option(..., help="The run's LauncherConfig.checkpoint_dir"),
    node: List[str] = typer.Option(
        ["main"], help="Node-id glob to profile; repeat for several, e.g. --node 'main.m2*'"
    ),
    output_dir: str = typer.Option("profiles", help="Shared directory for the traces"),
    rate: int = typer.Option(100, help="Samples per second"),
    native: bool = typer.Option(False, help="Include C frames"),
    gil: bool = typer.Option(False, help="Keep only samples of threads holding the GIL"),
    idle: bool = typer.Option(True, help="Keep samples of idle threads"),
    duration: Optional[int] = typer.Option(None, help="Seconds to record; default until nodes exit"),
    wait_timeout: float = typer.Option(120.0, help="Seconds to wait for the nodes to checkpoint"),
    pyspy: Optional[str] = typer.Option(None, help="py-spy executable (default: from PATH)"),
):
    """Attach py-spy to running orchestrator nodes, found through their checkpoints.

    Run it inside the job once the launcher is up. Each matched node's trace is written when
    the node exits, or when this command is interrupted. Merge them with
    `python -m ensemble_launcher.profiling.merge_traces`.
    """
    from ensemble_launcher.profiling import PySpyProfiler

    profiler = PySpyProfiler(
        checkpoint_dir,
        nodes=node,
        output_dir=output_dir,
        rate=rate,
        idle=idle,
        gil=gil,
        native=native,
        duration=duration,
        pyspy=pyspy,
        wait_timeout=wait_timeout,
    )
    started = profiler.start()
    for session in profiler.sessions:
        status = "attached" if session.error is None else f"FAILED: {session.error}"
        print(f"{session.node.node_id} ({session.node.hostname}, pid {session.node.pid}): {status}")
    if not started:
        raise typer.Exit(1)

    # A batch scheduler stops a job step with SIGTERM; treat it like Ctrl-C so traces flush.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    try:
        profiler.wait()
    except KeyboardInterrupt:
        print("Stopping py-spy sessions...")
        profiler.stop()
    print(f"Traces in {profiler.output_dir}")


if __name__ == "__main__":
    el()
