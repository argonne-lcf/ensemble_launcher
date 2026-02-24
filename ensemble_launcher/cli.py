import json
import os
import signal
from typing import Optional

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


if __name__ == "__main__":
    el()
