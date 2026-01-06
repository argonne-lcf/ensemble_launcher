from .node import *
import time
import os
from typing import Tuple, Optional
from ensemble_launcher.scheduler import AsyncTaskScheduler
from ensemble_launcher.scheduler.resource import AsyncLocalClusterResource, NodeResource, NodeResourceList
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.comm import AsyncComm, AsyncZMQComm, ZMQComm, MPComm
from ensemble_launcher.comm import Status, Result, ResultBatch, TaskUpdate
from ensemble_launcher.executors import executor_registry, AsyncThreadPoolExecutor, AsyncProcessPoolExecutor, AsyncMPIExecutor
import logging
import cloudpickle
import socket
import json
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import asdict
import threading
import asyncio
from asyncio import Future as AsyncFuture
from concurrent.futures import Future as ConcurrentFuture


class AsyncWorker(Node):
    """Synchronous worker implementation - all operations in main loop"""
    
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: NodeResource,
                Nodes:List[str],
                tasks: Dict[str, Task],
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[AsyncComm] = None
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

        self.logger = None


        self._event_timings: Dict[str, List[float]] = defaultdict(list)  # Store all timing measurements

        if self._config.profile == "timeline":
            self._timer = self._profile_timer
        else:
            self._timer = self._noop_timer

        self._stop_submission = asyncio.Event()
        self._stop_reporting = asyncio.Event()

        self._submission_task = None

        self._futures: Dict[str, Union[AsyncFuture,ConcurrentFuture]] = {}
        self._event_loop = None
    

    @contextmanager
    def _profile_timer(self,event_name: str):
        start_time = time.perf_counter()
        try:
            yield
        finally:
            self._event_timings[event_name].append(time.perf_counter() - start_time)


    @contextmanager
    def _noop_timer(self, _event_name: str):
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
            add_status[new_task.task_id] = self._scheduler.add_task(new_task)
        
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
            self._comm = AsyncZMQComm(self.logger.getChild('comm'), self.info(), parent_address=self.parent_comm.my_address if self.parent_comm else None, profile=self._config.profile)
            self.logger.info(f"{self.node_id}: Done with comm init")
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")
        
    async def _lazy_init(self):
        self._event_loop = asyncio.get_running_loop()
        #lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds")

        ##init scheduler
        self._scheduler = AsyncTaskScheduler(self.logger.getChild('scheduler'), self._tasks,cluster=AsyncLocalClusterResource(self.logger.getChild('cluster'), self._nodes, self._sys_info))

        self._scheduler.start_monitoring()

        ##lazy executor creation
        assert self._config.task_executor_name == "async_processpool" or \
            self._config.task_executor_name == "async_threadpool" or \
            self._config.task_executor_name == "async_mpi", f"Unsupported executor {self._config.task_executor_name} in AsyncWorker"

        kwargs = {}
        kwargs["logger"] = self.logger.getChild('executor')
        kwargs["gpu_selector"] = self._config.gpu_selector
        if self._config.task_executor_name == "async_mpi":
            kwargs["use_ppn"] = self._config.use_mpi_ppn
            kwargs["return_stdout"] = self._config.return_stdout
            kwargs["pin_resources"] = self._config.pin_resources
        self._executor: Union[AsyncProcessPoolExecutor, AsyncThreadPoolExecutor, AsyncMPIExecutor] = \
            executor_registry.create_executor(self._config.task_executor_name, kwargs=kwargs)
        
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
        while not self._stop_submission.is_set():
            self.logger.debug("In submit ready tasks loop")
            try:
                task_id, req = await self._scheduler.ready_tasks.get()
        
                task = self._tasks[task_id]
                task.status = TaskStatus.READY
                task.start_time = time.time()
                future = self._executor.submit(req, task.executable,
                                                    task_args=task.args,
                                                    task_kwargs=task.kwargs,
                                                    env=task.env)
                future.add_done_callback(self.create_done_callback(task))
                self._futures[task_id] = future
                task.status = TaskStatus.RUNNING
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in task submission loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

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

    async def run(self) -> Result:
        with self._timer("init"):
            ##lazy init
            await self._lazy_init()
        
        with self._timer("heartbeat_sync"):
            #sync with parent
            if self.parent and not await self._comm.sync_heartbeat_with_parent(timeout=30.0):
                self.logger.error(f"{self.node_id}: Failed to connect to parent")
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            else:
                self.logger.info(f"{self.node_id}: Connected to parent")
        
        task_update: TaskUpdate = await self._comm.recv_message_from_parent(TaskUpdate, timeout=10.0)
        if task_update is not None:
            self.logger.info(f"{self.node_id}: Received task update from parent")
            self._update_tasks(task_update)
        else:
            self.logger.warning(f"{self.node_id}: No task update received from parent at start")
        
        self.logger.info(f"Running {list(self._tasks.keys())} tasks")

        ##start submission loop
        self._submission_task = asyncio.create_task(self._submit_ready_tasks())

        ##start reporting loop
        self._reporting_task = asyncio.create_task(self.report_status())

        self.logger.info("Started waiting")
        await self._scheduler.wait_for_completion()

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
                pass
        self.logger.info("Stopped submission loop!")
        if self._reporting_task:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Stopped reporting loop!")

        with self._timer("result_collection"):
            all_results = await self._results()
        
        with self._timer("final_status"):
            ##also send the final status
            final_status = self.get_status()
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
        
        await self._comm.close()
        # await self._scheduler.stop_monitoring()
        self._executor.shutdown()
    
    def asdict(self,include_tasks:bool = False) -> dict:
        obj_dict = {
            "type": "AsyncWorker",
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
    def fromdict(cls, data: dict) -> 'AsyncWorker':
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

        worker = cls(
            id=data["node_id"],
            config=config,
            system_info=system_info,
            Nodes=data["nodes"],
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm
        )
        return worker