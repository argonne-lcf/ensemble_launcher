from .async_worker import AsyncWorker
from .node import Node
from ensemble_launcher.executors import executor_registry, Executor
from ensemble_launcher.scheduler import WorkerScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, JobResource, NodeResourceList, NodeResource, NodeResourceCount
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.comm import AsyncComm, AsyncZMQComm, NodeInfo
from ensemble_launcher.comm.messages import Status, Result, TaskUpdate, ResultBatch
import logging
from itertools import accumulate
from typing import Optional, List, Dict, Union
import os
import time
import numpy as np
import socket
import json
import base64
import asyncio
AsyncFuture = asyncio.Future
from concurrent.futures import Future as ConcurrentFuture
from contextlib import contextmanager
from collections import defaultdict
from .utils import async_load_str, async_simple_load_str
from dataclasses import asdict
import threading

# self.logger = logging.getself.logger(__name__)

class AsyncMaster(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: NodeResource,
                Nodes:List[str],
                tasks: Dict[str, Task],
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[AsyncComm] = None):
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
        self._children_futures: Dict[str, Union[AsyncFuture,ConcurrentFuture]] = {}
        self._child_assignment: Dict[str, Dict] = {}
        self._children_status: Dict[str, Status] = {}
        self._children_results: Dict[str, Result] = {}

        self.logger = None
        self._event_timings: Dict[str, List[float]] = defaultdict(list)  # Store all timing measurements
        if self._config.profile == "timeline":
            self._timer = self._profile_timer
        else:
            self._timer = self._noop_timer
        
        #asyncio event
        self._all_children_done_event = asyncio.Event()
        self._stop_reporting_event = asyncio.Event()
        self._done_children = set()
        self._event_loop = None  # Will be set in run()
        self._lock = None  # Protect _done_children

    

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
    def parent_comm(self, value: AsyncComm):
        self._parent_comm = value
    
    @property
    def comm(self):
        return self._comm
    
    def _setup_logger(self):

        if self._config.master_logs:
            os.makedirs(os.path.join(os.getcwd(),"logs"),exist_ok=True)
            # Configure file handler for this specific self.self.logger
            file_handler = logging.FileHandler(os.path.join(os.getcwd(),f'logs/master-{self.node_id}.log'))
            file_handler.setLevel(self._config.log_level)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            # Create instance self.self.logger and add handler
            self.logger = logging.getLogger(f"{__name__}.{self.node_id}")
            self.logger.addHandler(file_handler)
            self.logger.setLevel(self._config.log_level)
        else:
            self.logger = logging.getLogger(__name__)

    def _create_comm(self):
        if self._config.comm_name == "async_zmq":
            self.logger.info(f"{self.node_id}: Starting comm init")
            self._comm = AsyncZMQComm(self.logger.getChild('comm'), 
                                 self.info(),
                                 parent_address=self.parent_comm.my_address if self.parent_comm else None,
                                 profile=self._config.profile)
            self.logger.info(f"{self.node_id}: Done with comm init")
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

    def _create_children(self, include_tasks: bool = False) -> Dict[str, Node]:
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
                    AsyncWorker(
                        child_id,
                        config=self._config,
                        system_info=self._scheduler.cluster.system_info,
                        Nodes=alloc["nodes"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]} if include_tasks else {},
                        parent=None
                    )
        else:
            #create a master again
            for wid,alloc in assignments.items():
                child_id = self.node_id+f".m{wid}"
                self._child_assignment[child_id] = alloc
                #create a worker
                children[child_id] = \
                    AsyncMaster(
                        child_id,
                        config=self._config,
                        system_info=self._scheduler.cluster.system_info,
                        Nodes=alloc["nodes"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]} if include_tasks else {},
                        parent=None
                    )
        return children

    async def _lazy_init(self) -> Dict[str, Node]:

        self._lock = threading.RLock()  # Protect _done_children

        # Store event loop for thread-safe event signaling from callbacks
        self._event_loop = asyncio.get_event_loop()

        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")
        
        ##create a scheduler. maybe this can be removed??
        self._scheduler = WorkerScheduler(self.logger.getChild('scheduler'), cluster=LocalClusterResource(self.logger.getChild('cluster'), self._nodes,self._sys_info))

        assert self._config.child_executor_name in executor_registry.async_executors, f"Executor {self._config.child_executor_name} not found in async executors {executor_registry.async_executors}"

        kwargs = {}
        kwargs["logger"] = self.logger.getChild('executor')
        if self._config.child_executor_name == "async_mpi":
            kwargs["use_ppn"] = self._config.use_mpi_ppn
        #create executor
        self._executor: Executor = executor_registry.create_executor(self._config.child_executor_name, 
                                                                     kwargs=kwargs)

        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm() ###This will only create picklable objects
        ##lazy creation of non-pickable objects
        await self._comm.start_monitors(parent_only = True)
        
        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

        with self._timer("heartbeat_sync"):
            ##heart beat sync with parent
            if self.parent and not await self._comm.sync_heartbeat_with_parent(timeout=30.0):
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            self.logger.info(f"{self.node_id}: Synced heartbeat with parent")

        task_update: TaskUpdate = await self._comm.recv_message_from_parent(TaskUpdate,timeout=5.0)
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
        
        await self._comm.update_node_info(self.info())  ##update the node info with children ids

        await self._comm.start_monitors(children_only = True)

        return children

    async def run(self):
        with self._timer("init"):
            children = await self._lazy_init()
        
        with self._timer("launch_children"):
            if self._config.child_executor_name == "async_mpi":
                if not self._config.sequential_child_launch:
                    ##launch all children in a single shot
                    child_head_nodes = []
                    child_resources = []
                    child_obj_dict = {}
                    
                    for child_name, child_obj in children.items():
                        head_node = child_obj.nodes[0]
                        child_head_nodes.append(head_node)
                        child_resources.append(NodeResourceCount(ncpus=1))
                        child_obj_dict[head_node] = child_obj
                    
                    # Build combined dictionary structure
                    common_keys = ["type", "config", "system_info", "parent", "parent_comm"]
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
                    load_str_embed = async_load_str.replace("json_str_b64", f"b'{json_str_b64}'")
                    load_str_embed = load_str_embed.replace("common_keys_str", f"'{common_keys_str}'")
                    
                    req = JobResource(resources=child_resources, nodes=child_head_nodes)
                    env = os.environ.copy()
                    
                    self.logger.info(f"Launching worker using one shot mpiexec")
                    future = self._executor.submit(req, ["python", "-c", load_str_embed], env=env)
                    future.add_done_callback(self.create_done_callback("all"))
                    self._children_futures["all"] = future
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
                        load_str_embed = async_simple_load_str.replace("json_str_b64", f"b'{json_str_b64}'")
                        
                        req = JobResource(
                                resources=[NodeResourceCount(ncpus=1)], nodes=[head_node]
                            )
                        env = os.environ.copy()
                        env["EL_CHILDID"] = str(child_idx)
                        
                        self.logger.info(f"Launching child {child_name} using MPI executor (sequential)")
                        future = self._executor.submit(req, ["python", "-c", load_str_embed], env=env)
                        future.add_done_callback(self.create_done_callback(child_name))
                        self._children_futures[child_name] = future
            else:
                for child_idx, (child_name,child_obj) in enumerate(children.items()):
                    child_nodes = child_obj.nodes
                    req = JobResource(
                            resources=[NodeResourceCount(ncpus=1)], nodes=child_nodes[:1]
                        )
                    env = os.environ.copy()
                    
                    env["EL_CHILDID"] = str(child_idx)

                    future = self._executor.submit(req, child_obj.create_an_event_loop, env = env)
                    future.add_done_callback(self.create_done_callback(child_name))
                    self._children_futures[child_name] = future

        with self._timer("sync_with_children"):
            for child_id in self.children:
                if not await self._comm.sync_heartbeat_with_child(child_id=child_id, timeout=30.0):
                    self.logger.error(f"Failed to sync heartbeat with child {child_id}")
                    return await self._get_child_exceptions()
                new_tasks = [self._tasks[task_id] for task_id in self._child_assignment[child_id]["task_ids"]]
                task_update = TaskUpdate(sender=self.node_id, added_tasks=new_tasks)
                await self._comm.send_message_to_child(child_id, task_update)
                self.logger.info(f"{self.node_id}: Sent task update to {child_id} containing {len(new_tasks)}")
            
            asyncio.create_task(self.report_status())
            return await self._results() #should return and report
    
    def create_an_event_loop(self):
        """This function is an entry point for the new process"""
        asyncio.run(self.run())

    def create_done_callback(self, child_id: str):
        if child_id == "all":
            def _done_callback(future: ConcurrentFuture):
                with self._lock:
                    self._done_children = set(self.children.keys())
                if self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(self._all_children_done_event.set)
                else:
                    self.logger.warning("No event loop stored, setting event directly (may not work!)")
                    self._all_children_done_event.set()
            return _done_callback
        else:
            def _done_callback(future: AsyncFuture):
                with self._lock:
                    self._done_children.add(child_id)
                    all_done = len(self._done_children) == len(self.children)
                
                if all_done:
                    if self._event_loop is not None:
                        self._event_loop.call_soon_threadsafe(self._all_children_done_event.set)
                    else:
                        self.logger.warning("No event loop stored, setting event directly (may not work!)")
                        self._all_children_done_event.set()
            return _done_callback

    async def _get_child_exceptions(self) -> Result:
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
        for child_id, future in self._children_futures.items():
            if not future.done():
                self.logger.info(f"Stopping child {child_id}")
                future.cancel()
    
        # Collect exceptions without waiting
        exceptions = {}
        for child_id, future in self._children_futures.items():
            if future.done():
                try:
                    exception = future.exception()
                    if exception is not None:
                        exceptions[child_id] = exception
                        self.logger.error(f"Child {child_id} failed with exception: {exception}")
                except asyncio.CancelledError:
                    pass

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
            success = await self._comm.send_message_to_parent(result)
            if not success:
                self.logger.warning(f"{self.node_id}: Failed to send exception results to parent")

        await self.stop()
        return result
    
    async def report_status(self):
        while not self._stop_reporting_event.is_set():
            try:
                for child_id in self.children:
                    status = await self._comm.recv_message_from_child(Status, child_id=child_id)
                    if status is not None:
                        self._children_status[child_id] = status
                status = sum(self._children_status.values(), Status())
                if self.parent:
                    await self._comm.send_message_to_parent(status)
                    self.logger.info(status)
                else:
                    self.logger.info(status)
                # Use wait with timeout so we can exit quickly when stopped
                try:
                    await asyncio.wait_for(self._stop_reporting_event.wait(), timeout=self._config.report_interval)
                    break  # Exit if stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue loop after interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.info(f"Reporting loop failed with error {e}")
                await asyncio.sleep(0.1)
    
    async def _results(self) -> ResultBatch:
        await self._all_children_done_event.wait()
        self.logger.info(f"{self.node_id}: All children have completed execution")

        # Stop the reporting loop
        self._stop_reporting_event.set()
        self.logger.info(f"{self.node_id}: Stopped reporting loop")

        with self._timer("collect_results"):
            ##Create a new result from all the results
            data: List[Result] = []
            result_batch = ResultBatch(sender=self.node_id)
            for child_id in self.children:
                result_batch += await self._comm.recv_message_from_child(ResultBatch, child_id=child_id, timeout=5.0)

        for child_id in self.children:
            status = await self._comm.recv_message_from_child(Status, child_id=child_id)
            if status is not None:
                if child_id not in self._children_status:
                    self._children_status[child_id] = status
                elif status.timestamp > self._children_status[child_id].timestamp:
                    self._children_status[child_id] = status
        
        self.logger.info(f"Status from children: {self._children_status}")

        #send final results to parent
        if self.parent:
            success = await self._comm.send_message_to_parent(result_batch)

            if not success:
                self.logger.warning(f"{self.node_id}: Failed to send results to parent")
            else:
                self.logger.info(f"{self.node_id}: Succesfully sent results to parent")

        with self._timer("report_to_parent"):
            #report it to parent
            if self.parent:
                success = await self._comm.send_message_to_parent(sum(self._children_status.values(), Status()))
                if not success:
                    self.logger.warning(f"{self.node_id}: Failed to send final status to parent")
                else:
                    self.logger.info(f"{self.node_id}: Successfully reported final status to parent")
            else:
                try:
                    status = sum(self._children_status.values(), Status())
                    #write to a json file
                    fname = os.path.join(os.getcwd(),f"{self.node_id}_status.json")
                    status.to_file(fname)
                    self.logger.info(f"{self.node_id}: Successfully reported final status")
                except Exception as e:
                    self.logger.warning(f"{self.node_id}: Reporting final status failed with excepiton {e}")

        await self.stop()
        return result_batch

    async def stop(self):
        if self._config.profile:
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            fname = os.path.join(os.getcwd(),"profiles",f"{self.node_id}_comm_profile.json")
            with open(fname,"w") as f:
                json.dump(self._comm.profile, f, indent=4)
        
        if self._config.profile == "timeline":
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            # Compute statistics for all timed events
            stats = {}
            for event_name, timings in self._event_timings.items():
                stats[event_name] = {
                    "count": len(timings),
                    "total_time": sum(timings),
                    "mean_time": sum(timings) / len(timings) if timings else 0,
                    "min_time": min(timings) if timings else 0,
                    "max_time": max(timings) if timings else 0,
                }

            # Write statistics to file
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_timeline_stats.json")
            with open(fname, "w") as f:
                json.dump(stats, f, indent=4)
            
        await self._comm.close()        
        self._executor.shutdown()
    
    def asdict(self,include_tasks:bool = False) -> dict:
        obj_dict = {
            "type": "AsyncMaster",
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
    def fromdict(cls, data: dict) -> 'AsyncMaster':
        config = LauncherConfig.model_validate_json(data["config"])
        system_info = NodeResourceList(**data["system_info"]) if "cpus" in data["system_info"] else NodeResourceCount(**data["system_info"])
        parent = NodeInfo(**data["parent"]) if data["parent"] else None
        children = {child_id: NodeInfo(**child_dict) for child_id, child_dict in data["children"].items()}

        if config.comm_name == "async_zmq":
            # AsyncZMQComm might need special handling due to non-picklable attributes
            parent_comm = AsyncZMQComm.fromdict(data["parent_comm"]) if data["parent_comm"] else None
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
