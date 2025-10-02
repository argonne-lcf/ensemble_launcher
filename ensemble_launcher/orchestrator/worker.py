from .worker import *
from .Node import *
from ensemble_launcher.executors import executor_registry, Executor, MPIExecutor
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.comm import ZMQComm, MPComm

class Worker(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: SystemConfig,
                Nodes:List[str],
                tasks: List[Task] = [],
                parent = None,
                children: Dict = {}
                ):
        super().__init__(id, parent=parent, children=children)
        self._tasks = tasks
        self._executor: Executor = executor_registry.create_executor(self._config.executor_name)
        ##init a clusterresource
        cluster = LocalClusterResource(Nodes,system_info)
        self._scheduler = TaskScheduler(self._tasks,cluster=cluster)
        if config.executor_name == "multiprocessing":
            self._comm = MPComm(self.info(),self.parent.comm)
        elif config.executor_name == "mpi":
            self._comm = ZMQComm(self.info,self.parent.comm)
        else:
            raise ValueError(f"Unsupported executor for comm {config.executor_name}")
    
    def get_status(self):
        pass

    def run_tasks(self):
        while True:
            ready_tasks = self._scheduler.get_ready_tasks()
            for task in ready_tasks:
                self.