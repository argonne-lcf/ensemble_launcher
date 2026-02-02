from .node import *
import time
import os
from typing import Tuple, Optional
from ensemble_launcher.scheduler import AsyncTaskScheduler
from ensemble_launcher.scheduler.resource import JobResource
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.comm import AsyncComm, AsyncZMQComm
from ensemble_launcher.comm import Status, Result, ResultBatch, TaskUpdate, NodeUpdate
from ensemble_launcher.executors import executor_registry, AsyncThreadPoolExecutor, AsyncProcessPoolExecutor, AsyncMPIExecutor
from ensemble_launcher.profiling import get_registry, EventRegistry
import logging
import json
from contextlib import asynccontextmanager
from dataclasses import asdict
import threading
import asyncio
AsyncFuture = asyncio.Future
from concurrent.futures import Future as ConcurrentFuture
import uuid
import socket


class AsyncWorker(Node):
    """Synchronous worker implementation - all operations in main loop"""
    
    type = "AsyncWorker"
    
    def __init__(self,
                id:str,
                config:LauncherConfig,
                Nodes: Optional[JobResource] = None,
                tasks: Optional[Dict[str, Task]] = None,
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[AsyncComm] = None
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

        self.logger = None

        # Initialize event registry for perfetto profiling
        self._event_registry: Optional[EventRegistry] = None

        self._stop_submission = asyncio.Event()
        self._stop_reporting = asyncio.Event()

        self._submission_task = None

        self._futures: Dict[str, Union[AsyncFuture,ConcurrentFuture]] = {}
        self._event_loop = None
    
    @asynccontextmanager
    async def _timer(self, event_name: str):
        """Timer that records to event registry for Perfetto export."""
        if self._event_registry is not None:
            with self._event_registry.measure(event_name, "async_worker", node_id=self.node_id, pid=os.getpid()):
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
        if self._scheduler is not None:
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
        if self._config.worker_logs:
            os.makedirs(os.path.join(os.getcwd(),"logs"),exist_ok=True)
            file_handler = logging.FileHandler(os.path.join(os.getcwd(),f'logs/worker-{self.node_id}.log'))
            file_handler.setLevel(self._config.log_level)
            
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)

            self.logger = logging.getLogger(f"{__name__}.{self.node_id}")
            self.logger.addHandler(file_handler)
            self.logger.setLevel(self._config.log_level)
        else:
            self.logger = logging.getLogger(__name__)

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
            self.logger.debug(f"Adding new task {new_task}")
            add_status[new_task.task_id] = self._scheduler.add_task(new_task)
            if not add_status[new_task.task_id]:
                self.logger.error(f"Failed to add new task {new_task.task_id}")
            else:
                self.logger.debug(f"Added new task {new_task.task_id}")
        
        ##delete tasks if needed
        for task in taskupdate.deleted_tasks:
            if task.task_id in self._scheduler._running_tasks:
                self._executor.stop(task_id=self._executor_task_ids[task.task_id])
                self._futures[task.task_id].cancel()
            del_status[task.task_id] = self._scheduler.delete_task(task)
            
        return (add_status, del_status)

    def _create_comm(self):
        if self._config.comm_name == "async_zmq":
            self.logger.info(f"{self.node_id}: Starting comm init")
            self._comm = AsyncZMQComm(self.logger.getChild('comm'), self.info(), parent_address=self.parent_comm.my_address if self.parent_comm else None)
            self.logger.info(f"{self.node_id}: Done with comm init")
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")
        
    async def _lazy_init(self):
        if self._config.profile == "perfetto":
            self._event_registry = get_registry()
            self._event_registry.enable()

        self._event_loop = asyncio.get_running_loop()
        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")
        
        self.logger.info(f"My cpu affinity: {os.sched_getaffinity(0)}")
        ##init scheduler
        self._scheduler = AsyncTaskScheduler(self.logger.getChild('scheduler'), self._tasks, self.nodes)
        
        ##Lazy comm creation
        self._create_comm()
        await self._comm.start_monitors()

        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

    def create_done_callback(self, task: Task):
        def done_callback(future):
            self._event_loop.call_soon_threadsafe(self._task_callback, task, future)
        return done_callback
    
    def _task_callback(self, task: Task, future):
        task_id = task.task_id
        if self._config.profile == "perfetto" and self._event_registry is not None:
            self._event_registry.record_async_end(
                name=task.task_id,
                category="task_execution",
                node_id=self.node_id,
                pid=os.getpid(),
                async_id=task.task_id
            )
        if task_id in self._tasks:
            exception = future.exception()
            task.end_time = time.time()
            if exception is None:
                task.status = TaskStatus.SUCCESS
                task.result = future.result()
            else:
                task.status = TaskStatus.FAILED
                task.exception = str(exception)
            
            self._scheduler.free(task_id, task.status)

    
    async def _submit_ready_tasks(self):
        self.logger.info("Starting task submission loop")

        while not self._stop_submission.is_set():
            try:
                task_id, req = await self._scheduler.ready_tasks.get()
        
                task = self._tasks[task_id]
                self.logger.debug(f"Submitting task {task_id}: {task.executable} with resources {req.resources} {task.env}")
                task.status = TaskStatus.READY
                task.start_time = time.time()
                if self._config.profile == "perfetto" and self._event_registry is not None:
                    self._event_registry.record_async_begin(
                        name=task.task_id,
                        category="task_execution",
                        node_id=self.node_id,
                        pid=os.getpid(),
                        async_id=task.task_id
                    )
                    self._event_registry.record_counter(
                        name="tasks_submitted",
                        category="task_execution",
                        value=1,
                        pid=os.getpid(),
                        node_id=self.node_id
                    )
                future = self._executor.submit(req, task.executable,
                                                    task_args=task.args,
                                                    task_kwargs=task.kwargs,
                                                    env=task.env)
                future.add_done_callback(self.create_done_callback(task))
                self._futures[task_id] = future
                task.status = TaskStatus.RUNNING
            except asyncio.CancelledError:
                self.logger.info("Submission loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in task submission loop: {e}", exc_info=True)
                raise e
    
    async def _receive_initial_tasks(self):
        """Receive initial task assignment. Can be overridden by subclasses."""
        task_update: TaskUpdate = await self._comm.recv_message_from_parent(TaskUpdate, timeout=10.0)
        if task_update is not None:
            self.logger.info(f"{self.node_id}: Received task update from parent")
            self._update_tasks(task_update)
        else:
            self.logger.warning(f"{self.node_id}: No task update received from parent at start")

    async def report_status(self):
        while not self._stop_reporting.is_set():
            try:
                status = self.get_status()
                if self.parent:
                    await self._comm.send_message_to_parent(status)
                    self.logger.info(status)
                else:
                    self.logger.info(status)
                # Use wait with timeout so we can exit quickly when stopped
                try:
                    await asyncio.wait_for(self._stop_reporting.wait(), timeout=self._config.report_interval)
                    break  # Exit if stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue loop after interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.info(f"Reporting loop failed with error {e}")
                await asyncio.sleep(0.1)
    
    async def _wait_for_stop_condition(self):
        """Wait for completion condition. Can be overridden by subclasses."""
        await self._scheduler.wait_for_completion()

    async def run(self) -> Result:
        async with self._timer("init"):
            ##lazy init
            await self._lazy_init()
        
        async with self._timer("heartbeat_sync"):
            #sync with parent
            if self.parent and not await self._comm.sync_heartbeat_with_parent(timeout=30.0):
                self.logger.error(f"{self.node_id}: Failed to connect to parent")
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            else:
                self.logger.info(f"{self.node_id}: Connected to parent")
        
        # Receive node update from parent
        node_update: NodeUpdate = await self._comm.recv_message_from_parent(NodeUpdate, timeout=10.0)
        if node_update is not None:
            self.logger.info(f"{self.node_id}: Received node update from parent")
            self.nodes = node_update.nodes
            self.logger.debug(f"{self.node_id}: Nodes details: {self.nodes}")
        else:
            self.logger.warning(f"{self.node_id}: No node update received from parent at start")
        
        # Validate that nodes are initialized
        if not self.nodes:
            self.logger.error(f"{self.node_id}: Nodes not initialized!")
            raise RuntimeError(f"{self.node_id}: Nodes must be initialized before execution")
        
        ##lazy executor creation
        assert self._config.task_executor_name in executor_registry.async_executors, f"Executor {self._config.task_executor_name} not found in async executors {executor_registry.async_executors}"
        kwargs = {}
        kwargs["logger"] = self.logger.getChild('executor')
        kwargs["gpu_selector"] = self._config.gpu_selector
        if self._config.task_executor_name == "async_mpi":
            kwargs["cpu_binding_option"] = self._config.cpu_binding_option
            kwargs["use_ppn"] = self._config.use_mpi_ppn
            kwargs["return_stdout"] = self._config.return_stdout
        elif self._config.task_executor_name == "async_processpool":
            kwargs["max_workers"] = self._nodes.resources[0].cpu_count if self._nodes else None
            
        self._executor: Union[AsyncProcessPoolExecutor, AsyncThreadPoolExecutor, AsyncMPIExecutor] = \
            executor_registry.create_executor(self._config.task_executor_name, kwargs=kwargs)

        await self._receive_initial_tasks()
        
        self.logger.info(f"Running {list(self._tasks.keys())} tasks")
        self.logger.debug(f"Sorted tasks sizeL {self._scheduler._sorted_tasks.qsize()}")

        ##lazy executor creation
        assert self._config.task_executor_name in executor_registry.async_executors, f"Executor {self._config.task_executor_name} not found in async executors {executor_registry.async_executors}"

        kwargs = {}
        kwargs["logger"] = self.logger.getChild('executor')
        kwargs["gpu_selector"] = self._config.gpu_selector
        kwargs["max_workers"] = self.nodes.resources[0].cpu_count
        if self._config.task_executor_name == "async_mpi":
            kwargs["use_ppn"] = self._config.use_mpi_ppn
            kwargs["return_stdout"] = self._config.return_stdout
        self._executor: Union[AsyncProcessPoolExecutor, AsyncThreadPoolExecutor, AsyncMPIExecutor] = \
            executor_registry.create_executor(self._config.task_executor_name, kwargs=kwargs)

        self._scheduler.start_monitoring() #start the schduler monitoring

        ##start submission loop
        self._submission_task = asyncio.create_task(self._submit_ready_tasks())

        ##start reporting loop
        self._reporting_task = asyncio.create_task(self.report_status())

        self.logger.info("Started waiting for stop condition")
        await self._wait_for_stop_condition()

        ##stop scheduler monitoring first
        await self._scheduler.stop_monitoring()

        ##stop submission and reporting tasks
        self._stop_submission.set()
        self._stop_reporting.set()
        
        if self._submission_task:
            self._submission_task.cancel()
            try:
                await self._submission_task
            except asyncio.CancelledError:
                self.logger.debug("Submission task cancelled")
        self.logger.info("Stopped submission loop!")

        if self._reporting_task:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                self.logger.debug("Reporting task cancelled")
        
        self.logger.info("Stopped reporting loop!")

        async with self._timer("result_collection"):
            all_results = await self._results()
        
        async with self._timer("final_status"):
            ##also send the final status
            final_status = self.get_status()
            final_status.tag = "final"
            success = await self._comm.send_message_to_parent(final_status)
            if success:
                self.logger.info(f"{self.node_id}: Sent final status to parent")
            else:
                self.logger.warning(f"{self.node_id}: Failed to send final status to parent")
                fname = os.path.join(os.getcwd(),f"{self.node_id}_status.json")
                self.logger.info(f"{final_status}")
                final_status.to_file(fname)
        
        await self.stop()

        self.logger.info(f"{self.node_id} stopped")
        return all_results

    async def _results(self) -> ResultBatch:
        result_batch = ResultBatch(sender=self.node_id)
        for task_id,task in self._tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                task_result = Result(task_id=task_id,
                                    data=self._tasks[task_id].result,
                                    exception=str(self._tasks[task_id].exception))
                result_batch.add_result(task_result)
            else:
                self.logger.warning(f"Task {task_id} status {task.status}")
        
        status = await self._comm.send_message_to_parent(result_batch)
        if self.parent:
            if status:
                self.logger.info(f"{self.node_id}: Successfully sent the results to parent")
            else:
                self.logger.warning(f"{self.node_id}: Failed to send results to parent")
        return result_batch
    
    def create_an_event_loop(self):
        """This fuction is the entry point for a new process"""
        asyncio.run(self.run())

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
    def fromdict(cls, data: dict) -> 'AsyncWorker':
        config = LauncherConfig.model_validate_json(data["config"])
        parent = NodeInfo(**data["parent"]) if data["parent"] else None
        print(socket.gethostname(),data["children"])
        children = {child_id: NodeInfo(**child_dict) for child_id, child_dict in data["children"].items()}
        

        if config.comm_name == "async_zmq":
            # ZMQComm might need special handling due to non-picklable attributes
            parent_comm = AsyncZMQComm.fromdict(data["parent_comm"]) if data["parent_comm"] else None
        else:
            raise ValueError(f"Unsupported comm type {config.comm_name}")

        worker = cls(
            id=data["node_id"],
            config=config,
            Nodes=None,  # Nodes will be received via NodeUpdate message
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm
        )
        return worker