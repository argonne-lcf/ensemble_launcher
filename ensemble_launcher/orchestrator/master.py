from .worker import Worker
from .node import Node
from ensemble_launcher.executors import executor_registry, MPIExecutor, Executor
from ensemble_launcher.scheduler import WorkerScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, JobResource, NodeResourceList, NodeResource, NodeResourceCount
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.comm import ZMQComm, MPComm, NodeInfo, Comm
from ensemble_launcher.comm.messages import Status, Result, HeartBeat, Action, ActionType, TaskUpdate
import copy
import logging
from itertools import accumulate
from typing import Optional, List, Dict, Any
import os
import time
import numpy as np
import cloudpickle
import socket
import json
from contextlib import contextmanager
from collections import defaultdict
from .utils import load_str
from dataclasses import asdict

# self.logger = logging.getself.logger(__name__)

class Master(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: NodeResource,
                Nodes:List[str],
                tasks: Dict[str, Task],
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[Comm] = None):
        super().__init__(id, parent=parent, children=children)
        self._tasks = tasks
        self._config = config
        self._parent_comm = parent_comm
        self._nodes = Nodes
        self._sys_info = system_info

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

        if self._config.master_logs:
            os.makedirs(os.path.join(os.getcwd(),"logs"),exist_ok=True)
            # Configure file handler for this specific self.self.logger
            file_handler = logging.FileHandler(os.path.join(os.getcwd(),f'logs/master-{self.node_id}.log'))
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            # Create instance self.self.logger and add handler
            self.logger = logging.getLogger(f"{__name__}.{self.node_id}")
            self.logger.addHandler(file_handler)
            self.logger.setLevel(logging.INFO)
        else:
            self.logger = logging.getLogger(__name__)

    def _create_comm(self):
        if self._config.comm_name == "multiprocessing":
            self._comm = MPComm(self.logger, 
                                self.info(),
                                self.parent_comm if self.parent_comm else None, 
                                profile=self._config.profile)
        elif self._config.comm_name == "zmq":
            ##sending parent address here because all zmq objects are not picklable
            self._comm = ZMQComm(self.logger, 
                                 self.info(),
                                 parent_address=self.parent_comm.my_address if self.parent_comm else None,
                                 profile=self._config.profile)
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")

    def _assign_children(self,
                        tasks:Dict[str, Task],
                        nodes:list):
        """
        Assign tasks to workers based on resource requirements.
        """
        
        # Step 1: Sort tasks by decreasing number of nodes required
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].nnodes, reverse=True)
        
        ###remove tasks that have num nodes > total nodes of this master
        removed_tasks = []
        while sorted_tasks and sorted_tasks[0][1].nnodes > len(nodes):
            removed_tasks.append((sorted_tasks.pop(0))[0])
        
        if len(removed_tasks) > 0:
            self.logger.warning(f"Can't schedule {','.join(removed_tasks)}!")

        # Step 1: Calculate cumulative sum of nodes required for tasks
        cum_sum_nnodes = list(accumulate(task.nnodes for _, task in sorted_tasks))
        
        if self._config.nchildren is None:
            # Step 2: Determine the max number of workers
            nworkers_max = 0
            while nworkers_max < len(cum_sum_nnodes) and cum_sum_nnodes[nworkers_max] <= len(nodes):
                nworkers_max += 1

            # Step 3: Do a simple interpolation in the log2 space to determine number of workers at the current level
            # Calculate nworkers using log2 interpolation
            if self._config.nlevels > 1:
                # Create log2 space arrays for interpolation
                x_vals = np.array([0, self._config.nlevels],dtype = float)  # level range
                y_vals = np.array([0, np.log2(max(nworkers_max, 1))],dtype = float)  # log2 space range

                # Interpolate in log2 space
                log2_nworkers = int(np.interp(self.level + 1, x_vals, y_vals))

                # Convert back from log2 space
                nworkers = max(1, min(int(2 ** log2_nworkers), nworkers_max))
            else:
                nworkers = nworkers_max
        else:
            nworkers = self._config.nchildren
        
        if len(nodes) == 1:
            children_assignments = {
                wid: {
                "nodes": nodes,
                "task_ids": [task_id]        # List of task IDs assigned to the worker
                }
                for wid, (task_id, task) in enumerate(sorted_tasks[:nworkers])
            }
        else:
            if len(nodes) < nworkers:
                self.logger.error(f"number of nodes < number of children")
                raise RuntimeError
            
            # Step 4: Initialize worker assignments for the first set of tasks
            nnodes = [task.nnodes for task_id,task in sorted_tasks[:nworkers]]
            if sum(nnodes) < len(nodes):
                nremaining = len(nodes) - sum(nnodes)
                for i in range(nremaining):
                    nnodes[i%nworkers] += 1
            nnodes = list(accumulate(nnodes))
            children_assignments = {
                wid: {
                "nodes": nodes[:nnodes[wid]] if wid == 0 else nodes[nnodes[wid-1]:nnodes[wid]],
                "task_ids": [task_id]        # List of task IDs assigned to the worker
                }
                for wid, (task_id, task) in enumerate(sorted_tasks[:nworkers])
            }
        
        

        # Step 5: Assign remaining tasks to workers in a round-robin fashion
        for i, (task_id, task) in enumerate(sorted_tasks[nworkers:]):
            # Determine the worker ID in a round-robin manner
            worker_id = i % nworkers
            # Add the task ID to the worker's task list
            children_assignments[worker_id]["task_ids"].append(task_id)

        return children_assignments, removed_tasks

    def _create_children(self) -> Dict[str, Node]:
        assignments,remove_tasks = self._assign_children(self._tasks, self._scheduler.cluster.nodes)
        self._child_assignment = {}
        self.logger.info(f"Children assignment: {self._child_assignment}")

        children = {}
        if self.level + 1 == self._config.nlevels:
            for wid,alloc in assignments.items():
                child_id = self.node_id+f".w{wid}"
                self._child_assignment[child_id] = alloc
                #create a worker
                children[child_id] = \
                    Worker(
                        child_id,
                        config=self._config,
                        system_info=self._scheduler.cluster.system_info,
                        Nodes=alloc["nodes"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]},
                        parent=None
                    )
        else:
            #create a master again
            for wid,alloc in assignments.items():
                child_id = self.node_id+f".m{wid}"
                self._child_assignment[child_id] = alloc
                #create a worker
                children[child_id] = \
                    Master(
                        child_id,
                        config=self._config,
                        system_info=self._scheduler.cluster.system_info,
                        Nodes=alloc["nodes"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]},
                        parent=None
                    )
        return children

    # def _build_launch_cmd(self) -> str:
    #     ##save the state of the child obj        
    #     for child_name,child_obj in self.children.items():
    #             with open(os.path.join(os.getcwd(),".obj",f"{child_name}.pkl"),"wb") as f:
    #                 cloudpickle.dump(child_obj,f)

    #     if self.level + 1 == self._config.nlevels:
    #         cmd = f"python -m ensemble_launcher.orchestrator.worker"+\
    #                     f" {os.path.join(os.getcwd(),'.obj',f'{child_name}.pkl')}"
    #     else:   
    #         cmd = f"python -m ensemble_launcher.orchestrator.master"+\
    #                     f" {os.path.join(os.getcwd(),'.obj',f'{child_name}.pkl')}"
    #     return cmd
    

    def _lazy_init(self) -> Dict[str, Node]:
        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")
        
        ##create a scheduler. maybe this can be removed??
        self._scheduler = WorkerScheduler(self.logger, cluster=LocalClusterResource(self.logger, self._nodes,self._sys_info))

        #create executor
        self._executor: Executor = executor_registry.create_executor(self._config.child_executor_name, kwargs={"profile": self._config.profile,
                                                                                                               "logger": self.logger})

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
                ##launch all children in a single shot
                child_obj_dict = {}
                child_head_nodes = []
                child_resources = []
                for child_name,child_obj in children.items():
                    child_head_nodes.append(child_obj.nodes[0])
                    child_resources.append(NodeResourceCount(ncpus=1))
                    child_obj_dict[child_head_nodes[-1]] = child_obj
                req = JobResource(
                            resources=child_resources, nodes=child_head_nodes
                    )
                env = os.environ.copy()
                dirname = os.path.join(os.getcwd(),f".tmp_{self.node_id}")
                os.makedirs(dirname,exist_ok=True)
                for k,v in child_obj_dict.items():
                    fname = os.path.join(dirname,f"{k}_child_obj.json")
                    with open(fname,"w") as f:
                        self.logger.info(f"Saving child object dict {v.asdict()}, {k} to {fname}")
                        json.dump(v.asdict(),f)
                self.logger.info(f"Launching worker using one shot mpiexec")
                load_str_embed = load_str.replace("dirname",f"'{dirname}'")
                self._children_exec_ids["all"] = self._executor.start(req, ["python", "-c" ,load_str_embed] , env = env)
            else:
                for child_idx, (child_name,child_obj) in enumerate(children.items()):
                    child_nodes = child_obj.nodes
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
                for child_id in self.children:
                    new_tasks = [self._tasks[task_id] for task_id in self._child_assignment[child_id]["task_ids"]]
                    task_update = TaskUpdate(sender=self.node_id, added_tasks=new_tasks)
                    self._comm.send_message_to_child(child_id, task_update)
                    self.logger.info(f"{self.node_id}: Sent task update to {child_id} containing {len(new_tasks)}")
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
    
    def _results(self):
        next_report_time = time.time() + self._config.report_interval
        children_status = {}
        results: Dict[str, Result] = {}
        
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
                    result = self._comm.recv_message_from_child(Result, child_id)
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
            ##Create a new result from all the results
            data: List[Result] = []
            for child_id, result in results.items():
                if isinstance(result.data,list):
                    data.extend(result.data)
                elif isinstance(result.data, Result):
                    data.append(result.data)
                else:
                    raise ValueError(f"{self.node_id}: Received unknown type result from child {child_id}")
                    
            new_result = Result(sender = self.node_id, data = data)

        with self._timer("report_to_parent"):
            #report it to parent
            if self.parent:
                success = self._comm.send_message_to_parent(new_result)

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
        return new_result

    def stop(self):
        if self._config.profile:
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            fname = os.path.join(os.getcwd(),"profiles",f"{self.node_id}_comm_profile.json")
            with open(fname,"w") as f:
                json.dump(self._comm._profile_info, f, indent=2)
        
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
        self._executor.shutdown(force=True)
    
    def asdict(self,include_tasks:bool = False) -> dict:
        obj_dict = {
            "type": "Master",
            "node_id": self.node_id,
            "nodes": self._nodes,
            "config": self._config.model_dump_json(),
            "system_info": asdict(self._sys_info),
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
        system_info = NodeResourceList(**data["system_info"])
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
            system_info=system_info,
            Nodes=data["nodes"],
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm
        )
        return master
