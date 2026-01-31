from .async_worker import AsyncWorker
from .node import Node
from ensemble_launcher.executors import executor_registry, Executor
from ensemble_launcher.scheduler import AsyncWorkerScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, JobResource, NodeResourceList, NodeResource, NodeResourceCount
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.comm import AsyncComm, AsyncZMQComm, NodeInfo
from ensemble_launcher.comm.messages import Status, Result, TaskUpdate, NodeUpdate, ResultBatch
from ensemble_launcher.profiling import get_registry, EventRegistry
import logging
from itertools import accumulate
from typing import Optional, List, Dict, Union
import os
import time
import numpy as np
import json
import base64
import asyncio
AsyncFuture = asyncio.Future
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from .utils import async_load_str, async_simple_load_str
from dataclasses import asdict

# self.logger = logging.getself.logger(__name__)

class AsyncMaster(Node):
    
    type = "AsyncMaster"
    
    def __init__(self,
                id:str,
                config:LauncherConfig,
                Nodes: Optional[JobResource] = None,
                tasks: Optional[Dict[str, Task]] = None,
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[AsyncComm] = None):
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
        self._children_futures: Dict[str, Union[AsyncFuture,ConcurrentFuture]] = {}
        self._child_assignment: Dict[str, Dict] = {}
        self._children_status: Dict[str, Status] = {}
        self._children_results: Dict[str, Result] = {}

        self.logger = None
        
        # Initialize event registry for perfetto profiling
        self._event_registry: Optional[EventRegistry] = None
        
        #asyncio event
        self._all_children_done_event = asyncio.Event()
        self._stop_reporting_event = asyncio.Event()
        self._done_children = set()
        self._event_loop = None  # Will be set in run()
        self._lock = None  # Protect _done_children

    @asynccontextmanager
    async def _timer(self, event_name: str):
        """Timer that records to event registry for Perfetto export."""
        if self._event_registry is not None:
            with self._event_registry.measure(event_name, "async_master", node_id=self.node_id, pid=os.getpid()):
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
                                 parent_address=self.parent_comm.my_address if self.parent_comm else None)
            self.logger.info(f"{self.node_id}: Done with comm init")
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")

    def _create_children(self, include_tasks: bool = False) -> Dict[str, Node]:
        assignments,remove_tasks = self._scheduler.assign(self._tasks, self.level)
        if len(remove_tasks) > 0:
            self.logger.warning(f"Removed tasks due to resource constraints: {remove_tasks}")
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
                        Nodes=alloc["job_resource"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]} if include_tasks else {},
                        parent=None
                    )
        else:
            #create a master again - use work-stealing master if dynamic task updates enabled
            from .async_workstealing_master import AsyncWorkStealingMaster
            MasterClass = AsyncWorkStealingMaster if self._config.enable_workstealing else AsyncMaster
            
            for wid,alloc in assignments.items():
                child_id = self.node_id+f".m{wid}"
                self._child_assignment[child_id] = alloc
                #create a worker
                children[child_id] = \
                    MasterClass(
                        child_id,
                        config=self._config,
                        Nodes=alloc["job_resource"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]} if include_tasks else {},
                        parent=None
                    )
        return children

    async def _lazy_init(self) -> Dict[str, Node]:
        if self._config.profile == "perfetto":
            self._event_registry = get_registry()
            self._event_registry.enable()
            os.environ["EL_ENABLE_PROFILING"] = "1"

        # Store event loop for thread-safe event signaling from callbacks
        self._event_loop = asyncio.get_event_loop()

        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")

        self.logger.info(f"My cpu affinity: {os.sched_getaffinity(0)}")
        self._scheduler = AsyncWorkerScheduler(self.logger.getChild('scheduler'), 
                                                self.nodes, 
                                                self._config)
        
        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm() ###This will only create picklable objects
        ##lazy creation of non-pickable objects
        await self._comm.start_monitors(parent_only = True)
        
        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

        async with self._timer("heartbeat_sync"):
            ##heart beat sync with parent
            if self.parent and not await self._comm.sync_heartbeat_with_parent(timeout=30.0):
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            self.logger.info(f"{self.node_id}: Synced heartbeat with parent")

        # Receive node update from parent if it has a parent
        if self.parent:
            node_update: NodeUpdate = await self._comm.recv_message_from_parent(NodeUpdate, timeout=10.0)
            if node_update is not None:
                self.logger.info(f"{self.node_id}: Received node update from parent")
                if node_update.nodes:
                    self.nodes = node_update.nodes
                    self.logger.info(f"{self.node_id}: Updated nodes list with {len(self.nodes.nodes)} nodes")
                    self.logger.debug(f"{self.node_id}: Nodes details: {self.nodes}")
                else:
                    self.logger.warning(f"{self.node_id}: Received empty node update from parent")
            else:
                self.logger.warning(f"{self.node_id}: No node update received from parent at start")
        
        # Validate that nodes are initialized
        if not self.nodes:
            self.logger.error(f"{self.node_id}: Nodes not initialized!")
            raise RuntimeError(f"{self.node_id}: Nodes must be initialized before execution")

        task_update: TaskUpdate = await self._comm.recv_message_from_parent(TaskUpdate,timeout=5.0)
        if task_update is not None:
            self.logger.info(f"{self.node_id}: Received task update from parent containing {len(task_update.added_tasks)}")
            for task in task_update.added_tasks:
                self._tasks[task.task_id] = task
        
        self.logger.info(f"{self.node_id}: Have {len(self._tasks)} tasks after update from parent")

        assert self._config.child_executor_name in executor_registry.async_executors, f"Executor {self._config.child_executor_name} not found in async executors {executor_registry.async_executors}"

        kwargs = {}
        kwargs["logger"] = self.logger.getChild('executor')
        kwargs["max_workers"] = self.nodes.resources[0].cpu_count
        if self._config.child_executor_name == "async_mpi":
            kwargs["cpu_binding_option"] = self._config.cpu_binding_option
            kwargs["use_ppn"] = self._config.use_mpi_ppn

        #create executor
        self._executor: Executor = executor_registry.create_executor(self._config.child_executor_name, 
                                                                     kwargs=kwargs)

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

    async def _launch_children(self, children: Dict[str, Node]):
        """Launch all children processes."""
        if self._config.child_executor_name == "async_mpi":
            first_headnode = next(iter(children.values())).nodes.resources[0]
            worker_equality = all([child.nodes.resources[0] == first_headnode for child in children.values()])
            if not self._config.sequential_child_launch and worker_equality:
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
                load_str_embed = async_load_str.replace("json_str_b64", f"b'{json_str_b64}'")
                load_str_embed = load_str_embed.replace("common_keys_str", f"'{common_keys_str}'")
                
                req = JobResource(resources=child_resources, nodes=child_head_nodes)
                env = os.environ.copy()
                
                self.logger.info(f"Launching worker using one shot mpiexec")
                ##get mpi kwargs
                if isinstance(first_child.nodes.resources[0],NodeResourceList):
                    cpus = ":".join(map(str,first_child.nodes.resources[0].cpus))
                else:
                    cpus = ":".join(map(str,list(range(first_child.nodes.resources[0].cpu_count))))
                if self._config.cpu_binding_option == "--cpu-bind":
                    mpi_kwargs={self._config.cpu_binding_option: f"list:{cpus}"}
                else:
                    mpi_kwargs = {}
                    self.logger.warning(f"Unknown cpu binding option {self._config.cpu_binding_option}. Ignoring child pinning.")

                future = self._executor.submit(req, ["python", "-c", load_str_embed], env=env, mpi_kwargs=mpi_kwargs)
                future.add_done_callback(self.create_done_callback("all"))
                self._children_futures["all"] = future
            else:
                ##launch children sequentially one by one
                for child_idx, (child_name, child_obj) in enumerate(children.items()):
                    child_nodes = child_obj.nodes
                    head_node = child_nodes.nodes[0]
                    
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
                    
                    ##get mpi kwargs
                    if isinstance(child_obj.nodes.resources[0],NodeResourceList):
                        cpus = ":".join(map(str,child_obj.nodes.resources[0].cpus))
                    else:
                        cpus = ":".join(map(str,list(range(child_obj.nodes.resources[0].cpu_count))))
                    if self._config.cpu_binding_option == "--cpu-bind":
                        mpi_kwargs={self._config.cpu_binding_option: f"list:{cpus}"}
                        self.logger.info(f"Setting cpu affinity to child:{mpi_kwargs}")
                    else:
                        mpi_kwargs = {}
                        self.logger.warning(f"Unknown cpu binding option {self._config.cpu_binding_option}. Ignoring child pinning.")


                    self.logger.info(f"Launching child {child_name} using MPI executor (sequential)")
                    future = self._executor.submit(req, ["python", "-c", load_str_embed], env=env, mpi_kwargs=mpi_kwargs)
                    future.add_done_callback(self.create_done_callback(child_name))
                    self._children_futures[child_name] = future
        else:
            for child_idx, (child_name,child_obj) in enumerate(children.items()):
                child_nodes = child_obj.nodes.nodes
                req = JobResource(
                        resources=[NodeResourceList(cpus=child_obj.nodes.resources[0].cpus)], nodes=child_nodes[:1]
                    )
                env = os.environ.copy()
                
                env["EL_CHILDID"] = str(child_idx)

                future = self._executor.submit(req, child_obj.create_an_event_loop, env = env)
                future.add_done_callback(self.create_done_callback(child_name))
                self._children_futures[child_name] = future
    
    async def _sync_with_children(self):
        """Sync with children and send initial node/task updates."""
        for child_id in self.children:
            if not await self._comm.sync_heartbeat_with_child(child_id=child_id, timeout=30.0):
                self.logger.error(f"Failed to sync heartbeat with child {child_id}")
                return await self._get_child_exceptions()
            
            # Send node update first
            child_nodes = self._child_assignment[child_id]["job_resource"]
            node_update = NodeUpdate(sender=self.node_id, nodes=child_nodes)
            await self._comm.send_message_to_child(child_id, node_update)
            self.logger.info(f"{self.node_id}: Sent node update to {child_id} containing {len(child_nodes.nodes)} nodes")
            
            # Then send task update
            new_tasks = [self._tasks[task_id] for task_id in self._child_assignment[child_id]["task_ids"]]
            task_update = TaskUpdate(sender=self.node_id, added_tasks=new_tasks)
            await self._comm.send_message_to_child(child_id, task_update)
            self.logger.info(f"{self.node_id}: Sent task update to {child_id} containing {len(new_tasks)} tasks")
    
    async def run(self):
        async with self._timer("init"):
            children = await self._lazy_init()
        
        async with self._timer("launch_children"):
            await self._launch_children(children)

        async with self._timer("sync_with_children"):
            await self._sync_with_children()
            
            asyncio.create_task(self.report_status())
            return await self._results() #should return and report
    
    def create_an_event_loop(self):
        """This function is an entry point for the new process"""
        asyncio.run(self.run())
    
    def _mark_all_children_done(self):
        """Mark all children as done (runs in event loop)."""
        self._done_children = set(self.children.keys())
        self._all_children_done_event.set()
    
    def _mark_child_done(self, child_id: str):
        """Mark a single child as done (runs in event loop)."""
        self._done_children.add(child_id)
        if len(self._done_children) == len(self.children):
            self._all_children_done_event.set()

    def create_done_callback(self, child_id: str):
        if child_id == "all":
            def _done_callback(future: ConcurrentFuture):
                if self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(self._mark_all_children_done)
                else:
                    self.logger.warning("No event loop stored, can't mark children done!")
            return _done_callback
        else:
            def _done_callback(future: AsyncFuture):
                if self._event_loop is not None:
                    self._event_loop.call_soon_threadsafe(self._mark_child_done, child_id)
                else:
                    self.logger.warning("No event loop stored, can't mark child done!")
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
                    else:
                        result = future.result()
                        self.logger.error(f"Child {child_id}: No child exception found! Got {result}")
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

        async with self._timer("collect_results"):
            retry_children = set(self.children.keys())
            result_batch = ResultBatch(sender=self.node_id)
            max_retries = 10
            for retry in range(max_retries):
                for child_id in retry_children.copy():
                    temp_result_batch: ResultBatch = await self._comm.recv_message_from_child(ResultBatch, child_id=child_id, timeout=1.0)
                    if temp_result_batch is not None:
                        result_batch += temp_result_batch
                        retry_children.remove(child_id)
                if len(retry_children) == 0:
                    break
            if len(retry_children) > 0:
                self.logger.warning(f"{self.node_id}: Failed to receive results from children {retry_children} after {max_retries} retries")

        #collect final status from children
        async with self._timer("collect_status"):
            remaining_children = set()
            
            # Check which children already have final status
            for child_id in self.children.keys():
                if child_id in self._children_status and self._children_status[child_id].tag == "final":
                    self.logger.debug(f"{self.node_id}: Child {child_id} already has final status")
                else:
                    remaining_children.add(child_id)
            
            # For remaining children, drain their status queues
            for child_id in remaining_children.copy():
                empty_count = 0
                while empty_count < 2:
                    status = await self._comm.recv_message_from_child(Status, child_id=child_id, timeout=0.01)
                    if status is not None:
                        empty_count = 0  # Reset counter on successful recv
                        self._children_status[child_id] = status
                        if status.tag == "final":
                            remaining_children.remove(child_id)
                            break
                    else:
                        empty_count += 1  # Increment when queue is empty
                
            if len(remaining_children) > 0:
                self.logger.warning(f"{self.node_id}: Failed to receive final status from children {remaining_children}")
        
        self.logger.debug(f"Status from children: {self._children_status}")

        #send final results to parent
        if self.parent:
            success = await self._comm.send_message_to_parent(result_batch)

            if not success:
                self.logger.warning(f"{self.node_id}: Failed to send results to parent")
            else:
                self.logger.info(f"{self.node_id}: Succesfully sent results to parent")

        async with self._timer("report_to_parent"):
            #report it to parent
            if self.parent:
                final_status = sum(self._children_status.values(), Status())
                final_status.tag = "final"
                success = await self._comm.send_message_to_parent(final_status)
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
        if self._config.profile == "perfetto" and self._event_registry is not None:
            os.makedirs(os.path.join(os.getcwd(),"profiles"),exist_ok=True)
            # Export to Perfetto format
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_perfetto.json")
            self.logger.info(f"Exporting Perfetto trace to {fname}")
            self._event_registry.export_perfetto(fname)
            
            # Also export statistics
            stats = self._event_registry.get_statistics()
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_stats.json")
            self.logger.info(f"Exporting event statistics to {fname}")
            with open(fname, "w") as f:
                json.dump(stats, f, indent=2)
            
        await self._comm.close()        
        self._executor.shutdown()
    
    def asdict(self,include_tasks:bool = False) -> dict:
        obj_dict = {
            "type": self.type,
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
    def fromdict(cls, data: dict) -> 'AsyncMaster':
        config = LauncherConfig.model_validate_json(data["config"])
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
            Nodes=None,  # Nodes will be received via NodeUpdate message
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm
        )
        return master
