from .worker import *
from .Node import *
from ensemble_launcher.executors import executor_registry, MPIExecutor
from ensemble_launcher.scheduler import WorkerScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource
from ensemble_launcher.config import SystemConfig

class Master(Node):
    def __init__(self,
                 id:str,
                 Nodes:List[str],
                 system_info: SystemConfig):
        super().__init__(id)
        self._executor: MPIExecutor = executor_registry.create_executor("mpi")
        ##init a clusterresource
        self._scheduler = WorkerScheduler(cluster_resource=LocalClusterResource(Nodes,system_info))
        self._comm = None ## will be intialized once the workers are launched

    def _assign_children(self):
        pass

    def _build_launch_cmd(self):
        cmd = ""
        return cmd
    
    def launch_children(self):
        cmd = self._build_launch_cmd()
        self._executor.start()