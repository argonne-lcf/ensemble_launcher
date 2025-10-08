from .worker import *
from .node import *
import time
from typing import Any, TYPE_CHECKING, Tuple, Optional
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, NodeResource
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.comm import ZMQComm, MPComm, Comm
from ensemble_launcher.comm import Status, Result, HeartBeat, Message, Action, ActionType, TaskUpdate
from ensemble_launcher.executors import executor_registry, Executor

import logging

logger = logging.getLogger(__name__)

# # Configure file handler for this specific logger
# file_handler = logging.FileHandler('worker.log')
# file_handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(formatter)

# # Add handler to this logger only
# logger.addHandler(file_handler)
# logger.setLevel(logging.INFO)

class Worker(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: NodeResource,
                Nodes:List[str],
                tasks: Dict[str, Task],
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[Comm] = None
                ):
        super().__init__(id, parent=parent, children=children)
        self._config = config
        self._tasks: Dict[str, Task] = tasks
        
        self._parent_comm = parent_comm

        ##lazy init in run function
        self._comm = None
        ##lazy init in run function
        self._executor = None
        ##init scheduler
        self._scheduler = TaskScheduler(self._tasks,cluster=LocalClusterResource(Nodes,system_info))
        
        ##map from executor ids to task ids
        self._executor_task_ids: Dict[str, List[str]] = {}
        logger.info(f"{self.node_id} Done with init")
    
    @property
    def nodes(self):
        return self._scheduler.cluster.nodes
    
    @property
    def parent_comm(self):
        return self._parent_comm
    
    @parent_comm.setter
    def parent_comm(self, value: Comm):
        self._parent_comm = value
    
    @property
    def comm(self):
        return self._comm
    
    def _create_comm(self):
        if self._config.comm_name == "multiprocessing":
            self._comm = MPComm(self.info(),self.parent_comm if self.parent_comm else None)
        elif self._config.comm_name == "zmq":
            logger.info(f"{self.node_id}: Starting comm init")
            ##sending parent address here because all zmq objects are not picklable
            self._comm = ZMQComm(self.info(),parent_address=self.parent_comm.my_address if self.parent_comm else None)
            logger.info(f"{self.node_id}: Done with comm init")
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")

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

    def run(self) -> Result:
        ##lazy executor creation
        self._executor: Executor = executor_registry.create_executor(self._config.task_executor_name)
        ##Lazy comm creation
        self._create_comm()
        if self._config.comm_name == "zmq":
            self._comm.setup_zmq_sockets()
        
        if not self._comm.sync_heartbeat_with_parent(timeout=30.0):
            raise TimeoutError(f"{self.node_id}: Can't connect to parent")

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
            logger.info(f"{self.node_id}: Submitted {len(ready_tasks)} for execution")
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
        
        logger.info(f"{self.node_id}: Done executing all the tasks")

        all_results = self._results()
        while True and self.parent is not None:
            msg = self._comm.recv_message_from_parent(Action,timeout=1.0)
            if msg is not None:
                if msg.type == ActionType.STOP:
                    logger.info(f"{self.node_id}: Received stop from parent")
                    break
            time.sleep(1.0)
        self._stop()
        return all_results

    def _results(self) -> Result:
        results = []
        for task_id,task in self._tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                task_result = Result(task_id=task_id,
                                    data=self._tasks[task_id].result,
                                    error_message=self._tasks[task_id].exception)
                results.append(task_result)

        new_result = Result(
            sender=self.node_id,
            data=results
        )
        
        status = self._comm.send_message_to_parent(new_result)
        if self.parent:
            if status:
                logger.info(f"{self.node_id}: Successfully sent the results to parent")
            else:
                logger.warning(f"{self.node_id}: Failed to send results to parent")
        return new_result
        
    def _stop(self):
        self._comm.close()
        self._executor.shutdown()