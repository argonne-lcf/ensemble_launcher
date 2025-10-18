from .node import *
import time
import os
from typing import Any, TYPE_CHECKING, Tuple, Optional
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, NodeResource
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.comm import ZMQComm, MPComm, Comm
from ensemble_launcher.comm import Status, Result, HeartBeat, Message, Action, ActionType, TaskUpdate
from ensemble_launcher.executors import executor_registry, Executor
import logging
import cloudpickle
import socket
import json
from contextlib import contextmanager
from collections import defaultdict


class Worker(Node):
    """Synchronous worker implementation - all operations in main loop"""
    
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
        self._nodes = Nodes
        self._sys_info = system_info

        ##lazy init in run function
        self._comm = None
        ##lazy init in run function
        self._executor = None

        self._scheduler = None
        
        ##map from executor ids to task ids
        self._executor_task_ids: Dict[str, str] = {}

        self.logger = None


        self._event_timings: Dict[str, List[float]] = defaultdict(list)  # Store all timing measurements
        if self._config.profile == "timeline":
            self._timer = self._profile_timer
        else:
            self._timer = self._noop_timer
    

    @contextmanager
    def _profile_timer(self,event_name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._event_timings[event_name].append(time.perf_counter() - start_time)


    @contextmanager
    def _noop_timer(self, event_name: str):
        yield

    @property
    def nodes(self):
        return self._nodes
    
    @property
    def parent_comm(self):
        return self._parent_comm
    
    @parent_comm.setter
    def parent_comm(self, value: Comm):
        self._parent_comm = value
    
    @property
    def comm(self):
        return self._comm
    
    def _setup_logger(self):
        if self._config.worker_logs:
            os.makedirs(os.path.join(os.getcwd(),"logs"),exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(os.getcwd(),f'logs/worker-{self.node_id}.log'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger = logging.getLogger(f"{__name__}.{self.node_id}")
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logging.getLogger(__name__)
    
    def _create_comm(self):
        if self._config.comm_name == "multiprocessing":
            self._comm = MPComm(self.logger, self.info(),self.parent_comm if self.parent_comm else None, profile=self._config.profile)
        elif self._config.comm_name == "zmq":
            self.logger.info(f"{self.node_id}: Starting comm init")
            self._comm = ZMQComm(self.logger, self.info(),parent_address=self.parent_comm.my_address if self.parent_comm else None, profile=self._config.profile)
            self.logger.info(f"{self.node_id}: Done with comm init")
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

    def _free_task(self, task_id: str):
        exec_id = self._executor_task_ids[task_id]
        task = self._tasks[task_id]
        if self._executor.done(exec_id):
            task.end_time = time.time()
            exception = self._executor.exception(exec_id)
            self.logger.debug(f"Task {task_id} completed with executor ID {exec_id}")
            task.status = TaskStatus.SUCCESS
            if exception is not None:
                task.status = TaskStatus.FAILED
                task.exception = str(exception)
                self.logger.error(f"Task {task_id} failed with exception: {task.exception}")
            else:
                task.result = self._executor.result(exec_id)
                self.logger.debug(f"Task {task_id} completed successfully")
            ##free the resources
            self._scheduler.free(task_id, task.status)
            self.logger.debug(f"Resources freed for task {task_id} with status {task.status}")
            ##remove from tracking
            del self._executor_task_ids[task_id]

    def _poll_tasks(self):
        """Poll the tasks and set its status"""
        running_tasks = list(self._scheduler.running_tasks)
        for task_id in running_tasks:
            self._free_task(task_id)

    def _lazy_init(self):
        #lazy logger creation
        self._setup_logger()

        ##init scheduler
        self._scheduler = TaskScheduler(self.logger, self._tasks,cluster=LocalClusterResource(self.logger, self._nodes, self._sys_info))

        ##lazy executor creation
        self._executor: Executor = executor_registry.create_executor(self._config.task_executor_name,kwargs={"return_stdout": self._config.return_stdout,
                                                                                                             "profile":self._config.profile,
                                                                                                             "gpu_selector":self._config.gpu_selector,
                                                                                                             "logger":self.logger})

        ##Lazy comm creation
        self._create_comm()
        self._comm.init_cache()

        if self._config.comm_name == "zmq":
            self._comm.setup_zmq_sockets()

    def _submit_ready_tasks(self):
        ready_tasks = self._scheduler.get_ready_tasks()
        for task_id,req in ready_tasks.items():
            task = self._tasks[task_id]
            task.status = TaskStatus.READY
            task.start_time = time.time()
            exec_task_id = self._executor.start(req, task.executable,
                                                task_args=task.args,
                                                task_kwargs=task.kwargs,
                                                env=task.env)
            self._executor_task_ids[task_id] = exec_task_id
            task.status = TaskStatus.RUNNING
            
        if len(ready_tasks) > 0: 
            self.logger.info(f"{self.node_id}: Submitted {len(ready_tasks)} for execution")
        return

    def run(self) -> Result:
        with self._timer("init"):
            ##lazy init
            self._lazy_init()
        
        self.logger.info(f"Running {self._tasks.keys()} tasks")
        with self._timer("heartbeat_sync"):
            self._comm.async_recv()
            #sync with parent
            if self.parent and not self._comm.sync_heartbeat_with_parent(timeout=30.0):
                self.logger.error(f"{self.node_id}: Failed to connect to parent")
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")

        next_report_time = time.time() + self._config.report_interval
        
        while True:
            with self._timer("submit"):
                # Submit ready tasks synchronously
                self._submit_ready_tasks()
            with self._timer("poll"):
                # Poll and free completed tasks synchronously
                self._poll_tasks()
            # Report status periodically
            if time.time() > next_report_time:
                with self._timer("report_status"):
                    status = self.get_status()
                    if self.parent:
                        self._comm.send_message_to_parent(status)
                        self.logger.info(status)
                    else:
                        self.logger.info(status)
                    next_report_time = time.time() + self._config.report_interval
            # Check if all tasks are done
            if len(self._scheduler.remaining_tasks) == 0:
                break
            
            with self._timer("sleep"):
                time.sleep(0.01)
        
        self.logger.info(f"{self.node_id}: Done executing all the tasks")

        with self._timer("result_collection"):
            all_results = self._results()
        
        with self._timer("final_status"):
            ##also send the final status
            final_status = self.get_status()
            success = self._comm.send_message_to_parent(final_status)
            if success:
                self.logger.info(f"{self.node_id}: Sent final status to parent")
            else:
                self.logger.warning(f"{self.node_id}: Failed to send final status to parent")
                fname = os.path.join(os.getcwd(),f"{self.node_id}_status.json")
                final_status.to_file(fname)

        with self._timer("wait_for_stop"):
            self.logger.info(f"{self.node_id}: Started waiting for STOP from parent")
            while True and self.parent is not None:
                msg = self._comm.recv_message_from_parent(Action,timeout=1.0)
                if msg is not None:
                    if msg.type == ActionType.STOP:
                        self.logger.info(f"{self.node_id}: Received stop from parent")
                        break
                time.sleep(1.0)
        
        self._stop()
        return all_results
    
    @classmethod
    def load(cls, fname: str):
        """
            This method loads the master object from a file. 
            The file is pickled as Dict[hostname, Master]
        """
        with open(fname, "rb") as f:
            obj_dict: Dict[str, 'Worker'] = cloudpickle.load(f)
        hostname = socket.gethostname()
        worker_obj = None
        try:
            worker_obj = obj_dict[hostname]
        except KeyError:
            for key in obj_dict.keys():
                if hostname in key:
                    worker_obj = obj_dict[key]
                    break
        if worker_obj is None:
            return
        
        worker_obj.run()

    def _results(self) -> Result:
        results = []
        for task_id,task in self._tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                task_result = Result(task_id=task_id,
                                    data=self._tasks[task_id].result,
                                    exception=str(self._tasks[task_id].exception))
                results.append(task_result)

        new_result = Result(
            sender=self.node_id,
            data=results
        )
        
        status = self._comm.send_message_to_parent(new_result)
        if self.parent:
            if status:
                self.logger.info(f"{self.node_id}: Successfully sent the results to parent")
            else:
                self.logger.warning(f"{self.node_id}: Failed to send results to parent")
        return new_result
        
    def _stop(self):
        if self._config.profile:
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            fname = os.path.join(os.getcwd(),"profiles",f"{self.node_id}_comm_profile.json")
            with open(fname,"w") as f:
                json.dump(self._comm._profile_info, f, indent=2)
            
            fname = os.path.join(os.getcwd(),"profiles",f"{self.node_id}_executor_profile.json")
            with open(fname,"w") as f:
                json.dump(self._executor._profile_info, f, indent=2)
    
        if self._config.profile == "timeline":
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            # Compute statistics for all timed events
            stats = {}
            for event_name, timings in self._event_timings.items():
                if timings:  # Check if list is not empty
                    stats[event_name] = {
                        'mean': sum(timings) / len(timings),
                        'sum': sum(timings),
                        'std': (sum((x - sum(timings) / len(timings)) ** 2 for x in timings) / len(timings)) ** 0.5 if len(timings) > 1 else 0.0,
                        'count': len(timings)
                    }

            # Write statistics to file
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_timeline_stats.json")
            with open(fname, "w") as f:
                json.dump(stats, f, indent=2)
        
        self._comm.stop_async_recv()
        self._comm.clear_cache()
        self._comm.close()
        self._executor.shutdown()


