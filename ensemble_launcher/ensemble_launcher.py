import asyncio
import copy
import json
import logging
import multiprocessing
from typing import Dict, List, Optional, Union

from ensemble_launcher.orchestrator import (
    AsyncMaster,
    AsyncWorker,
    AsyncWorkStealingMaster,
)
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from .config import LauncherConfig, PolicyConfig, SystemConfig
from .ensemble import Task, TaskFactory
from .helper_functions import get_nodes

logger = logging.getLogger(__name__)


class EnsembleLauncher:
    def __init__(
        self,
        ensemble_file: Union[str, Dict[str, Union[Dict, Task]]],
        system_config: SystemConfig = SystemConfig(name="local"),
        launcher_config: Optional[LauncherConfig] = None,
        Nodes: Optional[List[str]] = None,
        pin_resources: bool = True,
        async_orchestrator: bool = True,
    ) -> None:
        self.ensemble_file = ensemble_file
        self.system_config = system_config
        self.launcher_config = launcher_config
        self.pin_resources = pin_resources
        self.async_orchestrator = async_orchestrator
        if isinstance(self.ensemble_file, dict) and all(
            [isinstance(t, Task) for t in self.ensemble_file.values()]
        ):
            self._tasks = self.ensemble_file
        else:
            self._tasks = self._generate_tasks()

        logger.info(f"Created {len(self._tasks)} tasks")

        if Nodes:
            self.nodes = Nodes
        else:
            self.nodes = get_nodes()
            logger.info(f"Found {len(self.nodes)} nodes for execution.")

        if len(self.nodes) == 0:
            raise ValueError(f"No compute nodes to execute tasks")
        # analyze the tasks to get launcher parameters like
        # - task_executor_name
        # - number of levels
        # - comm_name
        # - children_executor_name
        if self.launcher_config is None:
            task_np = [task.nnodes * task.ppn for task in self._tasks.values()]
            nnodes = len(self.nodes)
            if all([np == 1 for np in task_np]):
                # all serial tasks
                task_executor_name = (
                    "multiprocessing"
                    if async_orchestrator == False
                    else "async_processpool"
                )
            else:
                # some serial and some mpi
                task_executor_name = (
                    "mpi" if async_orchestrator == False else "async_mpi"
                )

            if nnodes == 1:
                comm_name = (
                    "multiprocessing" if async_orchestrator == False else "async_zmq"
                )
                nlevels = 0  ##Just the worker would be good enough
            else:
                # nnodes > 1
                comm_name = "zmq" if async_orchestrator == False else "async_zmq"
                if nnodes <= 64:
                    nlevels = 1
                elif nnodes > 64 and nnodes <= 256:
                    nlevels = 2
                elif nnodes > 256 and nnodes <= 2048:
                    nlevels = 2
                else:
                    nlevels = 3

            if nlevels == 0:
                child_executor_name = (
                    "multiprocessing"
                    if async_orchestrator == False
                    else "async_processpool"
                )
            else:
                child_executor_name = (
                    "mpi" if async_orchestrator == False else "async_mpi"
                )

            self.launcher_config = LauncherConfig(
                child_executor_name=child_executor_name,
                task_executor_name=task_executor_name,
                comm_name=comm_name,
                policy_config=PolicyConfig(nlevels=nlevels),
                return_stdout=True,
                master_logs=True,
                worker_logs=True,
            )

        logger.info(f"LauncherConfig: {self.launcher_config}")

        self._launcher = self._create_launcher()
        self._launcher_process: Optional[multiprocessing.Process] = None

    def _generate_tasks(self) -> Dict[str, Task]:
        if isinstance(self.ensemble_file, str):
            with open(self.ensemble_file, "r") as file:
                data = json.load(file)
                ensemble_infos = data["ensembles"]
        else:
            ensemble_infos = self.ensemble_file

        tasks = {}
        for name, info in ensemble_infos.items():
            tasks.update(TaskFactory.get_tasks(name, info))
        return tasks

    def _get_resource_config(self):
        """Get the appropriate resource configuration based on pin_resources setting."""
        if self.pin_resources:
            return NodeResourceList.from_config(self.system_config)
        else:
            return NodeResourceCount.from_config(self.system_config)

    def _create_launcher(self):
        """Create and return the appropriate launcher (Master or Worker) based on configuration."""
        resource_config = self._get_resource_config()
        nodes = JobResource(
            resources=[copy.deepcopy(resource_config) for _ in self.nodes],
            nodes=self.nodes,
        )
        launcher_args = ("main", self.launcher_config, nodes, self._tasks)

        if self.launcher_config.policy_config.nlevels == 0:
            if self.async_orchestrator:
                return AsyncWorker(*launcher_args)
            else:
                raise ValueError("Sync worker is no longer supported")
        elif (
            self.launcher_config.policy_config.nlevels == 1
            and self.launcher_config.enable_workstealing
        ):
            logger.info("!!!!!!Using workstealing!!!!!!")
            return AsyncWorkStealingMaster(*launcher_args)
        else:
            if self.async_orchestrator:
                return AsyncMaster(*launcher_args)
            else:
                raise ValueError("Sync Master version is no longer supported")

    def run(self):
        """Simply blocks until all the tasks are done."""
        if not self.async_orchestrator:
            raise RuntimeError("Sync orchestrator is deprecated")
        try:
            results = asyncio.run(self._launcher.run())
        except BaseException:
            logger.exception("EnsembleLauncher.run() failed, stopping launcher")
            raise
        return results

    def start(self):
        """Start the launcher in a separate process."""
        self._launcher_process = multiprocessing.Process(target=self.run)
        self._launcher_process.start()

    def stop(self):
        """Stop the launcher, terminating child processes if needed."""
        if self._launcher_process is not None:
            self._launcher_process.terminate()
            self._launcher_process.join(timeout=30)
            if self._launcher_process.is_alive():
                logger.warning("Launcher process did not exit cleanly; killing.")
                self._launcher_process.kill()
                self._launcher_process.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()
        return False
