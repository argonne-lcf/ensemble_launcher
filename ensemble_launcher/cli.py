import typer
from ensemble_launcher.config import LauncherConfig, SystemConfig
from typing import Optional
from ensemble_launcher import EnsembleLauncher
import json

el = typer.Typer()

@el.command()
def main(ensemble_file: str,
         system_config_file: Optional[str] = None,
         launcher_config_file: Optional[str] = None,
         nodes_str: Optional[str] = None,
         pin_resources: bool = True,
         async_orchestrator: bool = False):
    """
        Launch an ensemble of tasks based on the provided configuration files.

        Args:

            ensemble_file (str): Path to the ensemble configuration file.

            system_config_file (Optional[str]): Path to the system configuration JSON file.

            launcher_config_file (Optional[str]): Path to the launcher configuration JSON file.

            nodes_str (Optional[str]): Comma seperated string of list of compute nodes to use.

            pin_resources (bool): Whether to pin resources for tasks.

            async_orchestrator (bool): Whether to use an asynchronous orchestrator.
    """
    with open(system_config_file, "r") as f:
        config_dict = json.load(f)  
    system_config = SystemConfig.model_validate(config_dict) if system_config_file else SystemConfig(name="local")

    with open(launcher_config_file, "r") as f:
        config_dict = json.load(f)
    launcher_config = LauncherConfig.model_validate(config_dict) if launcher_config_file else None
    nodes = nodes_str.split(",") if nodes_str else None

    launcher = EnsembleLauncher(
        ensemble_file=ensemble_file,
        system_config=system_config,
        launcher_config=launcher_config,
        Nodes=nodes,
        pin_resources=pin_resources,
        async_orchestrator=async_orchestrator
    )
    results = launcher.run()
    return results

if __name__ == "__main__":
    el()
