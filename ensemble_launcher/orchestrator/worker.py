from .node import *
import time
import os
from typing import Any, TYPE_CHECKING, Tuple, Optional
from ensemble_launcher.scheduler import TaskScheduler
from ensemble_launcher.scheduler.resource import JobResource
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.comm import ZMQComm, MPComm, Comm
from ensemble_launcher.comm import Status, Result, ResultBatch, TaskUpdate, NodeUpdate
from ensemble_launcher.executors import executor_registry, Executor
from ensemble_launcher.profiling import get_registry, EventRegistry
import logging
from ensemble_launcher.logging import setup_logger
import cloudpickle
import socket
import json
from contextlib import contextmanager
from dataclasses import asdict


class Worker(Node):
    """Synchronous worker implementation - all operations in main loop"""
    
    def __init__(self,
                id:str,
                config:LauncherConfig,
                Nodes: Optional[JobResource] = None,
                tasks: Optional[Dict[str, Task]] = None,
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[Comm] = None
                ):
        super().__init__(id, parent=parent, children=children)
        self._config = config
        self._tasks: Dict[str, Task] = tasks
        self._parent_comm = parent_comm
        self._nodes = Nodes

        ##lazy init in run function
        self._comm = None
        ##lazy init in run function
        self._executor = None

        self._scheduler = None
        
        ##map from executor ids to task ids
        self._executor_task_ids: Dict[str, str] = {}

        self.logger = None

        # Initialize event registry for perfetto profiling
        self._event_registry: Optional[EventRegistry] = None
    
    @contextmanager
    def _timer(self, event_name: str):
        """Timer that records to event registry for Perfetto export."""
        if self._event_registry is not None:
            with self._event_registry.measure(event_name, "worker", node_id=self.node_id, pid=os.getpid()):
                yield
        else:
            yield

    @property
    def nodes(self):
        try:
            return self._scheduler.cluster.nodes
        except Exception as e:
            return self._nodes
    
    @nodes.setter
    def nodes(self, value: JobResource):
        self._nodes = value
        if self._scheduler is not None and getattr(self._scheduler, "cluster", None) is not None:
            self._scheduler.cluster.update_nodes(value)
    
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
        log_dir = os.path.join(os.getcwd(), self._config.log_dir) if self._config.worker_logs else None
        self.logger = setup_logger(__name__, self.node_id, log_dir=log_dir, level=self._config.log_level)
    
    def _create_comm(self):
        if self._config.comm_name == "multiprocessing":
            self._comm = MPComm(self.logger.getChild('comm'), self.info(),self.parent_comm if self.parent_comm else None)
        elif self._config.comm_name == "zmq":
            self.logger.info(f"{self.node_id}: Starting comm init")
            self._comm = ZMQComm(self.logger.getChild('comm'), self.info(),parent_address=self.parent_comm.my_address if self.parent_comm else None)
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
        if self._config.profile == "perfetto":
            self._event_registry = get_registry()
            self._event_registry.enable()
        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")

        ##init scheduler
        self._scheduler = TaskScheduler(self.logger.getChild('scheduler'), self._tasks, self.nodes)

        ##lazy executor creation
        self._executor: Executor = executor_registry.create_executor(self._config.task_executor_name,kwargs={"return_stdout": self._config.return_stdout,
                                                                                                             "gpu_selector":self._config.gpu_selector,
                                                                                                             "logger":self.logger.getChild('executor')})

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
        
        with self._timer("heartbeat_sync"):
            self._comm.async_recv_parent()
            #sync with parent
            if self.parent and not self._comm.sync_heartbeat_with_parent(timeout=30.0):
                self.logger.error(f"{self.node_id}: Failed to connect to parent")
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
        
        # Receive node update from parent
        node_update: NodeUpdate = self._comm.recv_message_from_parent(NodeUpdate, timeout=10.0)
        if node_update is not None:
            self.logger.info(f"{self.node_id}: Received node update from parent")
            self.nodes = node_update.nodes
        else:
            self.logger.warning(f"{self.node_id}: No node update received from parent at start")
        
        # Validate that nodes are initialized
        if not self.nodes:
            self.logger.error(f"{self.node_id}: Nodes not initialized!")
            raise RuntimeError(f"{self.node_id}: Nodes must be initialized before execution")
        
        task_update: TaskUpdate = self._comm.recv_message_from_parent(TaskUpdate, timeout=10.0)
        if task_update is not None:
            self.logger.info(f"{self.node_id}: Received task update from parent")
            self._update_tasks(task_update)
        else:
            self.logger.warning(f"{self.node_id}: No task update received from parent at start")
        
        self.logger.info(f"Running {list(self._tasks.keys())} tasks")

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

        # with self._timer("wait_for_stop"):
        #     self.logger.info(f"{self.node_id}: Started waiting for STOP from parent")
        #     while True and self.parent is not None:
        #         msg = self._comm.recv_message_from_parent(Action,timeout=1.0)
        #         if msg is not None:
        #             if msg.type == ActionType.STOP:
        #                 self.logger.info(f"{self.node_id}: Received stop from parent")
        #                 break
        #         time.sleep(1.0)
        
        self._stop()
        return all_results
    
    @classmethod
    def load(cls, dirname: str):
        """
            This method loads the master object from a file. 
            The file is pickled as Dict[hostname, Master]
        """
        hostname = socket.gethostname()
        fname = os.path.join(dirname,f"{hostname}_child_obj.pkl")
    
        worker_obj = None
        try:
            with open(fname, "rb") as f:
                worker_obj: 'Worker' = cloudpickle.load(f)
        except:
            pass
        if worker_obj is None:
            print(f"failed loading child from {fname}")
            return
        worker_obj.run()

    def _results(self) -> ResultBatch:
        result_batch = ResultBatch(sender=self.node_id)
        for task_id,task in self._tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                task_result = Result(task_id=task_id,
                                    data=self._tasks[task_id].result,
                                    exception=str(self._tasks[task_id].exception))
                result_batch.add_result(task_result)
        
        status = self._comm.send_message_to_parent(result_batch)
        if self.parent:
            if status:
                self.logger.info(f"{self.node_id}: Successfully sent the results to parent")
            else:
                self.logger.warning(f"{self.node_id}: Failed to send results to parent")
        return result_batch
        
    def _stop(self):
        if self._config.profile == "perfetto" and self._event_registry is not None:
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            # Export to Perfetto format
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_perfetto.json")
            self._event_registry.export_perfetto(fname)
            
            # Also export statistics
            stats = self._event_registry.get_statistics()
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_stats.json")
            with open(fname, "w") as f:
                json.dump(stats, f, indent=2)
        
        self._comm.stop_async_recv()
        self._comm.clear_cache()
        self._comm.close()
        self._executor.shutdown()
    
    def asdict(self,include_tasks:bool = False) -> dict:
        obj_dict = {
            "type": "Worker",
            "node_id": self.node_id,
            "config": self._config.model_dump_json(),
            "parent": asdict(self.parent) if self.parent else None,
            "children": {child_id: asdict(child) for child_id, child in self.children.items()},
            "parent_comm": self.parent_comm.asdict() if self.parent_comm else None
        }

        if include_tasks:
            raise NotImplementedError("Including tasks in serialization is not implemented yet.")
        
        return obj_dict
    
    @classmethod
    def fromdict(cls, data: dict) -> 'Worker':
        config = LauncherConfig.model_validate_json(data["config"])
        parent = NodeInfo(**data["parent"]) if data["parent"] else None
        children = {child_id: NodeInfo(**child_dict) for child_id, child_dict in data["children"].items()}

        if config.comm_name == "zmq":
            # ZMQComm might need special handling due to non-picklable attributes
            parent_comm = ZMQComm.fromdict(data["parent_comm"]) if data["parent_comm"] else None
        elif config.comm_name == "multiprocessing":
            parent_comm = MPComm.fromdict(data["parent_comm"]) if data["parent_comm"] else None
        else:
            raise ValueError(f"Unsupported comm type {config.comm_name}")

        worker = cls(
            id=data["node_id"],
            config=config,
            Nodes=None,  # Nodes will be sent via NodeUpdate
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm
        )
        return worker


