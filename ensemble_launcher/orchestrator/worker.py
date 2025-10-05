from .worker import *
from .node import *
import time
from typing import Any, TYPE_CHECKING, Tuple
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.comm import ZMQComm, MPComm
from ensemble_launcher.comm import Status, Result, HeartBeat, Message, Action, ActionType, TaskUpdate

if TYPE_CHECKING:
    from ensemble_launcher.executors import Executor

import logging

logger = logging.getLogger(__name__)

class Worker(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: SystemConfig,
                Nodes:List[str],
                tasks: List[Task] = [],
                parent: Node = None,
                children: Dict = {}
                ):
        super().__init__(id, parent=parent, children=children)
        self._config = config
        self._tasks: Dict[str, Task] = {task.task_id: task for task in tasks}
        # Import executor_registry lazily to avoid circular import
        from ensemble_launcher.executors import executor_registry, Executor
        self._executor: Executor = executor_registry.create_executor(self._config.executor_name)
        ##init a clusterresource
        cluster = LocalClusterResource(Nodes,system_info)
        self._scheduler = TaskScheduler(self._tasks,cluster=cluster)
        if config.executor_name == "multiprocessing":
            self._comm = MPComm(self.info(),self.parent.comm if self.parent else None)
        elif config.executor_name == "mpi":
            self._comm = ZMQComm(self.info,self.parent.comm)
        else:
            raise ValueError(f"Unsupported executor for comm {config.executor_name}")
        ##map from executor ids to task ids
        self._executor_task_ids: Dict[str, List[str]] = {}
    
    def get_status(self):
        """Gets the status of all the tasks and resources in terms of counts"""
        return Status(
            nrunning_tasks=len(self._scheduler.running_tasks),
            nfailed_tasks=len(self._scheduler.failed_tasks),
            nsuccessful_tasks=len(self._scheduler.successful_tasks),
            nfree_cores=self._scheduler.cluster.free_cpus,
            nfree_gpus=self._scheduler.cluster.free_gpus
        )

    def _update_tasks(self, taskupdate: TaskUpdate) -> Tuple[Dict[str, bool],Dict[str,bool]]:
        ##Add the tasks to scheduler
        add_status = {}
        del_status = {}
        for new_task in taskupdate.added_tasks:
            add_status[new_task.task_id] = self._scheduler.add_task(new_task)
        
        ##delete tasks if needed
        for task in taskupdate.deleted_tasks:
            del_status[task.task_id] = self._scheduler.delete_task(task)
            if task.task_id in self._scheduler.running_tasks:
                self._executor.stop(task_id=self._executor_task_ids[task.task_id])
        
        return (add_status, del_status)

    def _poll_tasks(self):
        """Poll the tasks and set its status"""
        for task_id, exec_id in self._executor_task_ids.items():
            task = self._tasks[task_id]
            if self._executor.done(exec_id):
                exception = self._executor.exception(exec_id) ##excepiton will be None if the task is succesful
                logger.debug(f"Task {task_id} completed with executor ID {exec_id}")
                task.status = TaskStatus.SUCCESS
                if exception is not None:
                    task.status = TaskStatus.FAILED
                    task.exception = str(exception)
                    logger.error(f"Task {task_id} failed with exception: {task.exception}")
                else:
                    task.result = self._executor.result(exec_id)
                    logger.debug(f"Task {task_id} completed successfully")
            ##free the resources
            self._scheduler.free(task_id, task.status)
            logger.debug(f"Resources freed for task {task_id} with status {task.status}")

    def run_tasks(self):
        next_report_time = time.time() + self._config.report_interval
        while True:
            ready_tasks = self._scheduler.get_ready_tasks()
            for task_id,req in ready_tasks.items():
                task = self._tasks[task_id]
                task.status = TaskStatus.READY
                exec_task_id = self._executor.start(req, task.executable,
                                                    task_args=task.args,
                                                    task_kwargs=task.kwargs,
                                                    env=task.env)
                self._executor_task_ids[task_id] = exec_task_id
                task.status = TaskStatus.RUNNING

            ##poll tasks and free resources
            self._poll_tasks()
            if time.time() > next_report_time:
                status = self.get_status()
                if self.parent:
                    self._comm.send_message_to_parent(status)
                else:
                    logger.info(status)
                next_report_time = time.time() + self._config.report_interval

            if self.parent:
                #check for actions
                action = self._comm.recv_message_from_parent(Action, timeout=1.0)
                if action:
                    if action == ActionType.STOP:
                        break
                
                #check for task updates
                taskupdate = self._comm.recv_message_from_parent(TaskUpdate,timeout=1.0)
                if taskupdate:
                    add_status, del_status = self._update_tasks(taskupdate)
                
            
            if len(self._scheduler.remaining_tasks) == 0:
                break
            time.sleep(1.0)
            
        self._stop()

    def results(self) -> Dict[str, Any]:
        ret = {}
        for task_id,task in self._tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                ret[task_id] = self._tasks[task_id].result
        return ret
    
    def exceptions(self) -> Dict[str, str]:
        ret = {}
        for task_id,task in self._tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                ret[task_id] = self._tasks[task_id].exception
        return ret

    def report_results(self) -> Dict[str, Any]:
        results = self.results()
        exceptions = self.exceptions()
        if self.parent:
            for task_id in results:
                self._comm.send_message_to_parent(Result(sender=self.node_id,
                                                         task_id=task_id,
                                                         data=results[task_id],
                                                         error_message=exceptions[task_id]))
                time.sleep(0.1)
        else:
            logger.info(f"{self.node_id} results: {results}")
            logger.info(f"{self.node_id} exceptions: {exceptions}")
        
    def _stop(self):
        self._executor.shutdown()
        self._comm.close()