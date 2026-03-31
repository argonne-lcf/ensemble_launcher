from .worker import Worker
from .node import Node
from ensemble_launcher.executors import executor_registry, MPIExecutor, Executor
from ensemble_launcher.scheduler import WorkerScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, JobResource, NodeResourceList, NodeResource, NodeResourceCount
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.comm import ZMQComm, MPComm, NodeInfo, Comm
from ensemble_launcher.comm.messages import Status, Result, ResultBatch, Action, ActionType, TaskUpdate, NodeUpdate
from ensemble_launcher.profiling import get_registry, EventRegistry
import copy
import logging
from ensemble_launcher.logging import setup_logger
from itertools import accumulate
from typing import Optional, List, Dict, Any
import os
import time
import numpy as np
import cloudpickle
import socket
import json
import base64
from contextlib import contextmanager
from collections import defaultdict
from .utils import load_str, simple_load_str
from dataclasses import asdict

# self.logger = logging.getself.logger(__name__)

class Master(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                Nodes: Optional[JobResource] = None,
                tasks: Optional[Dict[str, Task]] = None,
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[Comm] = None):
        super().__init__(id, parent=parent, children=children)
        self._tasks = tasks
        self._config = config
        self._parent_comm = parent_comm
        self._nodes = Nodes

        ##lazily created in run
        self._executor = None
        self._comm = None

        self._scheduler = None

        ##maps
        self._children_exec_ids: Dict[str, str] = {}
        self._child_assignment: Dict[str, Dict] = {}

        ##most recent Status
        self._status: Status = None

        self.logger = None
        
        # Initialize event registry for perfetto profiling
        self._event_registry: Optional[EventRegistry] = None
    
    @contextmanager
    def _timer(self, event_name: str):
        """Timer that records to event registry for Perfetto export."""
        if self._event_registry is not None:
            with self._event_registry.measure(event_name, "master", node_id=self.node_id, pid=os.getpid()):
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
        log_dir = os.path.join(os.getcwd(), self._config.log_dir) if self._config.master_logs else None
        self.logger = setup_logger(__name__, self.node_id, log_dir=log_dir, level=self._config.log_level)

    def _create_comm(self):
        if self._config.comm_name == "multiprocessing":
            self._comm = MPComm(self.logger.getChild('comm'), 
                                self.info(),
                                self.parent_comm if self.parent_comm else None)
        elif self._config.comm_name == "zmq":
            ##sending parent address here because all zmq objects are not picklable
            self._comm = ZMQComm(self.logger.getChild('comm'), 
                                 self.info(),
                                 parent_address=self.parent_comm.my_address if self.parent_comm else None)
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")

    def _create_children(self, include_tasks: bool = False) -> Dict[str, Node]:
        assignments, remove_tasks = self._scheduler.assign(self._tasks, self.level)
        if len(remove_tasks) > 0:
            self.logger.warning(f"Removed tasks due to resource constraints: {remove_tasks}")
        self._child_assignment = {}
        self.logger.info(f"Children assignment: {self._child_assignment}")

        children = {}
        if self.level + 1 == self._config.policy_config.nlevels:
            for wid, alloc in assignments.items():
                child_id = self.node_id+f".w{wid}"
                self._child_assignment[child_id] = alloc
                #create a worker
                children[child_id] = \
                    Worker(
                        child_id,
                        config=self._config,
                        Nodes=alloc["job_resource"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]} if include_tasks else {},
                        parent=None
                    )
        else:
            #create a master again
            for wid, alloc in assignments.items():
                child_id = self.node_id+f".m{wid}"
                self._child_assignment[child_id] = alloc
                #create a worker
                children[child_id] = \
                    Master(
                        child_id,
                        config=self._config,
                        Nodes=alloc["job_resource"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]} if include_tasks else {},
                        parent=None
                    )
        return children

    def _lazy_init(self) -> Dict[str, Node]:
        if self._config.profile == "perfetto":
            self._event_registry = get_registry()
            self._event_registry.enable()

        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")
        
        ##create a scheduler. maybe this can be removed??
        self._scheduler = WorkerScheduler(self.logger.getChild('scheduler'), self.nodes, self._config)

        #create executor
        self._executor: Executor = executor_registry.create_executor(self._config.child_executor_name, kwargs={"logger": self.logger.getChild('executor'),
                                                                                                               "use_ppn": self._config.use_mpi_ppn})

        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm() ###This will only create picklable objects
        ##lazy creation of non-pickable objects
        if self._config.comm_name == "zmq":
            self._comm.setup_zmq_sockets()

        with self._timer("heartbeat_sync"):
            self._comm.async_recv_parent() ###start the recv thread
            ##heart beat sync with parent
            if not self._comm.sync_heartbeat_with_parent(timeout=30.0):
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            self.logger.info(f"{self.node_id}: Synced heartbeat with parent")

        # Receive node update from parent if it has a parent
        if self.parent:
            node_update: NodeUpdate = self._comm.recv_message_from_parent(NodeUpdate, timeout=10.0)
            if node_update is not None:
                self.logger.info(f"{self.node_id}: Received node update from parent")
                if node_update.nodes:
                    self.nodes = node_update.nodes
                    self.logger.info(f"{self.node_id}: Updated nodes list with {len(self.nodes.nodes)} nodes")
                else:
                    self.logger.warning(f"{self.node_id}: Received empty node update from parent")
            else:
                self.logger.warning(f"{self.node_id}: No node update received from parent at start")
        
        # Validate that nodes are initialized
        if not self.nodes:
            self.logger.error(f"{self.node_id}: Nodes not initialized!")
            raise RuntimeError(f"{self.node_id}: Nodes must be initialized before execution")

        task_update: TaskUpdate = self._comm.recv_message_from_parent(TaskUpdate,timeout=5.0)
        if task_update is not None:
            self.logger.info(f"{self.node_id}: Received task update from parent")
            for task in task_update.added_tasks:
                self._tasks[task.task_id] = task
        
        self.logger.info(f"{self.node_id}: Have {len(self._tasks)} tasks after update from parent")

        ##create children
        children = self._create_children()
        
        self.logger.info(f"{self.node_id} Created {len(children)} children: {children.keys()}")

        #add children
        for child_id, child in children.items():
            self.add_child(child_id, child.info())
            child.set_parent(self.info())
            child.parent_comm = self.comm.pickable_copy()
        
        self._comm.update_node_info(self.info())  ##update the node info with children ids
        self._comm.async_recv_children() ###start the recv thread for children

        return children

    def run(self):
        with self._timer("init"):
            children = self._lazy_init()
        
        with self._timer("launch_children"):
            if self._config.child_executor_name == "mpi":
                if not self._config.sequential_child_launch:
                    ##launch all children in a single shot
                    child_head_nodes = []
                    child_resources = []
                    child_obj_dict = {}
                    
                    for child_name, child_obj in children.items():
                        head_node = child_obj.nodes.nodes[0]
                        child_head_nodes.append(head_node)
                        child_resources.append(NodeResourceCount(ncpus=1))
                        child_obj_dict[head_node] = child_obj
                    
                    # Build combined dictionary structure
                    common_keys = ["type", "config", "parent", "parent_comm"]
                    first_child = next(iter(child_obj_dict.values()))
                    first_dict = first_child.asdict()
                    
                    # Initialize with common keys from first child
                    final_dict = {key: first_dict[key] for key in common_keys}
                    
                    # Initialize per-host keys as empty dicts
                    for key in first_dict.keys():
                        if key not in common_keys:
                            final_dict[key] = {}
                    
                    # Populate per-host values
                    for hostname, child_obj in child_obj_dict.items():
                        child_dict = child_obj.asdict()
                        for key, value in child_dict.items():
                            if key not in common_keys:
                                final_dict[key][hostname] = value
                    
                    # Create embedded command string
                    json_str = json.dumps(final_dict, default=str)
                    json_str_b64 = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
                    common_keys_str = ','.join(common_keys)
                    load_str_embed = load_str.replace("json_str_b64", f"b'{json_str_b64}'")
                    load_str_embed = load_str_embed.replace("common_keys_str", f"'{common_keys_str}'")
                    
                    req = JobResource(resources=child_resources, nodes=child_head_nodes)
                    env = os.environ.copy()
                    
                    self.logger.info(f"Launching worker using one shot mpiexec")
                    self._children_exec_ids["all"] = self._executor.start(req, ["python", "-c", load_str_embed], env=env)
                else:
                    ##launch children sequentially one by one
                    for child_idx, (child_name, child_obj) in enumerate(children.items()):
                        child_nodes = child_obj.nodes
                        head_node = child_nodes[0]
                        
                        # Serialize child object
                        child_dict = child_obj.asdict()
                        json_str = json.dumps(child_dict, default=str)
                        json_str_b64 = base64.b64encode(json_str.encode('utf-8')).decode('ascii')
                        
                        # Create embedded command string for this child (simple version, no per-host logic)
                        load_str_embed = simple_load_str.replace("json_str_b64", f"b'{json_str_b64}'")
                        
                        req = JobResource(
                                resources=[NodeResourceCount(ncpus=1)], nodes=[head_node]
                            )
                        env = os.environ.copy()
                        env["EL_CHILDID"] = str(child_idx)
                        
                        self.logger.info(f"Launching child {child_name} using MPI executor (sequential)")
                        self._children_exec_ids[child_name] = self._executor.start(req, ["python", "-c", load_str_embed], env=env)
            else:
                for child_idx, (child_name,child_obj) in enumerate(children.items()):
                    child_nodes = child_obj.nodes.nodes
                    req = JobResource(
                            resources=[NodeResourceCount(ncpus=1)], nodes=child_nodes[:1]
                        )
                    env = os.environ.copy()
                    
                    env["EL_CHILDID"] = str(child_idx)

                    self._children_exec_ids[child_name] = self._executor.start(req, child_obj.run, env = env)

        with self._timer("sync_with_children"):
            if not self._comm.sync_heartbeat_with_children(timeout=30.0):
                self.logger.error(f"{self.node_id}: Can't connect to children.")
                return self._get_child_exceptions() # Should return and report
            else:
                self.logger.info(f"{self.node_id}: Synced heartbeat with children")
            
            # Send node allocation to each child
            for child_id in self.children:
                child_nodes = self._child_assignment[child_id]["job_resource"]
                node_update = NodeUpdate(sender=self.node_id, nodes=child_nodes)
                success = self._comm.send_message_to_child(child_id, node_update)
                if success:
                    self.logger.info(f"{self.node_id}: Sent node update to {child_id} containing {len(child_nodes.nodes)} nodes")
                else:
                    self.logger.error(f"{self.node_id}: Failed to send node update to {child_id}")
            
            # Send task updates to each child
            for child_id in self.children:
                new_tasks = [self._tasks[task_id] for task_id in self._child_assignment[child_id]["task_ids"]]
                task_update = TaskUpdate(sender=self.node_id, added_tasks=new_tasks)
                success = self._comm.send_message_to_child(child_id, task_update)
                if success:
                    self.logger.info(f"{self.node_id}: Sent task update to {child_id} containing {len(new_tasks)} tasks")
                else:
                    self.logger.error(f"{self.node_id}: Failed to send task update to {child_id}")
            
            return self._results() #should return and report
    
    @classmethod
    def load(cls, dirname: str):
        """
            This method loads the master object from a file. 
            The file is pickled as Dict[hostname, Master]
        """
        hostname = socket.gethostname()
        fname = os.path.join(dirname,f"{hostname}_child_obj.pkl")
    
        master_obj = None
        try:
            with open(fname, "rb") as f:
                master_obj: 'Master' = cloudpickle.load(f)
        except:
            pass
        if master_obj is None:
            print(f"failed loading child from {fname}")
            return
        master_obj.run()


    def _get_child_exceptions(self) -> Result:
        """
        Collect and handle exceptions from child processes.
        This method stops all running child processes and collects any exceptions
        that occurred during their execution. It creates Result objects for each
        exception found and optionally sends them to the parent node.
        Returns:
            Result: A Result object containing exception results from failed child processes.
                    The data field contains a list of Result objects, one for each child
                    that failed with an exception. Each child Result has the exception
                    stored as a string in its exception attribute.
        Notes:
            - All running children are stopped before collecting exceptions
            - Only processes that are done and have exceptions are included
            - Exception results are automatically sent to parent node if one exists
            - Logs information about stopped children and found exceptions
        """
        
        # First, stop all children
        for child_id, exec_id in self._children_exec_ids.items():
            if self._executor.running(exec_id):
                self.logger.info(f"Stopping child {child_id}")
                self._executor.stop(exec_id)
    
        # Collect exceptions without waiting
        exceptions = {}
        for child_id, exec_id in self._children_exec_ids.items():
            if self._executor.done(exec_id):
                exception = self._executor.exception(exec_id)
                if exception is not None:
                    exceptions[child_id] = exception
                    self.logger.error(f"Child {child_id} failed with exception: {exception}")

        self.logger.info(f"{self.node_id}: Stopped children. Found {len(exceptions)} exceptions")

        # Create result objects for each exception
        exception_results = []
        for child_id, exception in exceptions.items():
            exception_result = Result(sender=child_id, data=[])
            exception_result.exception = str(exception)
            exception_results.append(exception_result)
        
        # Create a result with the exception results
        result = Result(sender=self.node_id, data=exception_results)

        # Send to parent if exists
        if self.parent:
            success = self._comm.send_message_to_parent(result)
            if not success:
                self.logger.warning(f"{self.node_id}: Failed to send exception results to parent")

        self.stop()
        return result
    
    def _results(self) -> ResultBatch:
        next_report_time = time.time() + self._config.report_interval
        children_status = {}
        results: Dict[str, ResultBatch] = {}
        
        done = set()
        while True:
            with self._timer("check_children"):
                # Special handling for MPI executor - check once before child loop
                if self._config.child_executor_name == "mpi":
                    if self._executor.done(self._children_exec_ids["all"]):
                        for child_id in self.children:
                            done.add(child_id)
            
                for child_id in self.children:
                    if child_id in done:
                        continue

                    ##look for results
                    result = self._comm.recv_message_from_child(ResultBatch, child_id)
                    if result is not None:
                        self.logger.info(f"{self.node_id}: Received result from {child_id}.")
                        ##final status of the child
                        final_status = self._comm.recv_message_from_child(Status, child_id=child_id, timeout=5.0)
                        if final_status is not None:
                            children_status[child_id] = final_status
                            self.logger.info(f"{self.node_id}: Received final status from {child_id}")
                            self.logger.info(f"{self.node_id}: Final status {final_status}")
                        ##
                        results[child_id] = result
                        done.add(child_id)
                        self._comm.send_message_to_child(child_id, Action(sender=self.node_id, type=ActionType.STOP))
                    
                    # For non-MPI executors, check individual exec_ids
                    if self._config.child_executor_name != "mpi":
                        if child_id not in self._children_exec_ids:
                            self.logger.error(f"{child_id} not in exec_id map!!")
                            raise RuntimeError
                        else:
                            exec_id = self._children_exec_ids[child_id]
                            if self._executor.done(exec_id):
                                done.add(child_id)
            
            ##send status to parent
            if time.time() > next_report_time:
                with self._timer("report_status"):
                    ##receive status updates from ALL children
                    for child_id in self.children:
                        status = self._comm.recv_message_from_child(Status, child_id=child_id, timeout=0.1)
                        if status is not None:
                            children_status[child_id] = status

                    self._status = sum(children_status.values(), Status())
                    if self.parent:
                        self._comm.send_message_to_parent(self._status)
                    else:
                        if isinstance(self._status, Status):
                            self.logger.info(f"{self.node_id}: Status: {self._status}")
                    next_report_time = time.time() + self._config.report_interval

            time.sleep(0.1)
            if len(done) == len(self.children):
                self.logger.info(f"{self.node_id}: All children are done")
                break
        with self._timer("collect_results"):
            ##Create a new result batch from all the results
            result_batch = ResultBatch(sender=self.node_id)
            for child_id, child_result in results.items():
                if isinstance(child_result, ResultBatch):
                    result_batch.data.extend(child_result.data)
                else:
                    raise ValueError(f"{self.node_id}: Received unknown type result from child {child_id}")

        with self._timer("report_to_parent"):
            #report it to parent
            if self.parent:
                success = self._comm.send_message_to_parent(result_batch)

                if not success:
                    self.logger.warning(f"{self.node_id}: Failed to send results to parent")
                else:
                    self.logger.info(f"{self.node_id}: Succesfully sent results to parent")
                
                ##also send the final_status
                self._status = sum(children_status.values(), Status())
                success = self._comm.send_message_to_parent(self._status)
                if not success:
                    self.logger.warning(f"{self.node_id}: Failed to send status to parent")
                else:
                    self.logger.info(f"{self.node_id}: Succesfully sent status to parent")
                    fname = os.path.join(os.getcwd(),f"{self.node_id}_status.json")
                    self._status.to_file(fname)
            else:
                try:
                    self._status = sum(children_status.values(), Status())
                    #write to a json file
                    fname = os.path.join(os.getcwd(),f"{self.node_id}_status.json")
                    self._status.to_file(fname)
                    self.logger.info(f"{self.node_id}: Successfully reported final status")
                except Exception as e:
                    self.logger.warning(f"{self.node_id}: Reporting final status failed with excepiton {e}")
        
        with self._timer("wait_for_stop"):
            #wait for my parent to instruct me
            while True and self.parent is not None:
                msg = self._comm.recv_message_from_parent(Action,timeout=1.0)
                if msg is not None:
                    if msg.type == ActionType.STOP:
                        self.logger.info(f"{self.node_id}: Received stop from parent")
                        break
        
                time.sleep(1.0)
        
        for child_id, exec_id in self._children_exec_ids.items():
            result = self._executor.result(exec_id)
            self.logger.info(f"{self.node_id}: Child {child_id} final (stdout,stderr): {result}")

        self.stop()
        return result_batch

    def stop(self):
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
        self._executor.shutdown(force=True)
    
    def asdict(self,include_tasks:bool = False) -> dict:
        obj_dict = {
            "type": "Master",
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
    def fromdict(cls, data: dict) -> 'Master':
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

        master = cls(
            id=data["node_id"],
            config=config,
            Nodes=None,  # Nodes will be sent via NodeUpdate
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm
        )
        return master
