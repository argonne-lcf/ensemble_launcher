from .resource import NodeResourceList, JobResource, AsyncLocalClusterResource, LocalClusterResource, NodeResource
from .resource import NodeResourceCount
from ensemble_launcher.ensemble import Task, TaskStatus
from typing import List, Dict, Union, Set, Tuple
from .policy import policy_registry, Policy
from  logging import Logger
import copy
import asyncio
from asyncio import Queue, PriorityQueue
from .scheduler import Scheduler
from collections import Counter

# self.logger = logging.getself.logger(__name__)


class AsyncScheduler(Scheduler):
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """
    def __init__(self, logger: Logger, cluster_resource: AsyncLocalClusterResource):
        super().__init__(logger=logger, cluster_resource=cluster_resource)


class AsyncWorkerScheduler(AsyncScheduler):
    def __init__(self, logger: Logger, nodes: List[str], system_info: NodeResource):
        cluster = LocalClusterResource(logger.getChild('cluster'), nodes, system_info)
        super().__init__(logger, cluster)
        self.workers: Dict[str, JobResource] = {}
    
    def assign(self, worker_id: str,  worker_resource: JobResource):
        allocated, resource = self.cluster.allocate(worker_resource)
        if allocated:
            self.workers[worker_id] = resource
        return allocated, resource
    
    def free(self, worker_id: str):
        if worker_id in self.workers:
            return self.cluster.deallocate(self.workers[worker_id])
        return None
    
    def get_worker_assignment(self):
        return copy.deepcopy(self.workers)

class AsyncTaskScheduler(AsyncScheduler):
    def __init__(self, 
                 logger: Logger,
                 tasks: Dict[str, Task], 
                 nodes: List[str],
                 system_info: NodeResource,
                 policy: Union[str,Policy]= "large_resource_policy"):
        cluster = AsyncLocalClusterResource(logger.getChild('cluster'), nodes, system_info)
        super().__init__(logger, cluster)
        self.tasks: Dict[str, Task] = tasks
        if isinstance(policy, str):
            self.scheduler_policy: Policy = policy_registry.create_policy(policy)
        else:
            self.scheduler_policy: Policy = policy
        self._sorted_tasks: PriorityQueue[Tuple[float,str]] = PriorityQueue()
        for task_id, task in self.tasks.items():
            try:
                self._sorted_tasks.put_nowait((1.0/self.scheduler_policy.get_score(self.tasks[task_id]),task_id))
            except asyncio.QueueFull:
                self.logger.error("Sorted task queue is full!")
                raise RuntimeError("Sorted task is full!")
        self.logger.debug(f"Sorted tasks {self._sorted_tasks}")
        ##
        self.ready_tasks: Queue[Tuple[str,JobResource]] = Queue()
        self._running_tasks: Dict[str,JobResource] = {}
        self._done_tasks: Counter[str] = Counter()
        self._failed_tasks: Set[str] = set()
        self._successful_tasks: Set[str] = set()
        
        self._stop_monitoring = asyncio.Event()
        self._all_tasks_done = asyncio.Event()
        self._consecutive_failed_allocations = 0
        self._monitoring_task = None
        self._event_loop = None  # Will be set when monitoring starts
    
    def _buld_task_resource_req(self, task: Task) -> JobResource:
        req = JobResource(
                resources=[NodeResourceCount(ncpus=task.ppn,ngpus=task.ngpus_per_process*task.ppn) for i in range(task.nnodes)]
            )
        if len(task.cpu_affinity) > 0 or len(task.gpu_affinity) > 0:
            if task.cpu_affinity and (task.ngpus_per_process > 0 and not task.gpu_affinity):
                self.logger.warning(f"Task {task.task_id}: Ignoring cpu_affinity as gpu_affinity is not set")
                return req
            
            if task.gpu_affinity and not task.cpu_affinity:
                self.logger.warning(f"Task {task.task_id}: Ignoring gpu_affinitiy as cpu_affinity is not set")
                return req
            
            req = JobResource(
                resources=[NodeResourceList(cpus=task.cpu_affinity, gpus=task.gpu_affinity) for node in range(task.nnodes)]
            )
        return req

    def _find_min_resource_req(self) -> JobResource:
        """
        Find the minimum resource requirement among PENDING tasks only.
        """
        if self._sorted_tasks.empty():
            return None
        
        # Get pending task IDs from sorted_tasks queue
        pending_task_ids = [task_id for _, task_id in list(self._sorted_tasks._queue)]
        
        if not pending_task_ids:
            return None
        
        pending_tasks = [self.tasks[tid] for tid in pending_task_ids if tid in self.tasks]
        
        if not pending_tasks:
            return None
            
        min_nnodes = min(task.nnodes for task in pending_tasks)
        min_ppn = min(task.ppn for task in pending_tasks)
        min_ngpus = min(task.ngpus_per_process*task.ppn for task in pending_tasks)
        
        return JobResource(
            resources=[NodeResourceCount(ncpus=min_ppn, ngpus=min_ngpus) for _ in range(min_nnodes)]
        )
    
    async def _monitor_resources(self) -> None:
        """
        Monitors free resources and checks if any tasks can be allocated, and moves them to the ready queue.
        """
        self.logger.info("Starting resource monitor")
        
        while not self._stop_monitoring.is_set():
            try:
                # Wait for resources - will be cancelled when monitor is stopped
                min_req = self._find_min_resource_req()
                self.logger.debug(f"Waiting for free resources with min requirement: {min_req}. Current free resources: {self.cluster.get_status()}")
                await self._cluster_resource.wait_for_free(min_resources=min_req)
                self.logger.debug(f"Resource monitor woke up, checking for task allocation. Current free resources: {self.cluster.get_status()}")
                
                # Check stop immediately
                if self._stop_monitoring.is_set():
                    break

                #interrupt sleep to allow other coroutines to when no tasks are available
                if self._sorted_tasks.empty():
                    self.logger.info("No tasks to schedule, resource monitor quitting")
                    self._stop_monitoring.set()
                
                unallocated_tasks = []
                allocated_count = 0
                
                # Try to allocate all pending tasks
                for _ in range(self._sorted_tasks.qsize()):
                    if self._stop_monitoring.is_set():
                        break

                    try:
                        priority, task_id = await self._sorted_tasks.get()
                    except asyncio.QueueEmpty:
                        break
                    
                    # Thread-safe access to tasks
                    if task_id not in self.tasks:
                        self.logger.warning(f"Task {task_id} no longer exists, skipping")
                        continue
                    task = self.tasks[task_id]
                    
                    req = self._buld_task_resource_req(task)
                    
                    allocated, resource = self.cluster.allocate(req)
                    
                    if allocated:
                        # Put in ready queue - awaitable operation yields control
                        await self.ready_tasks.put((task_id, resource))
                        
                        # Track as running
                        self._running_tasks[task_id] = resource
                        
                        allocated_count += 1
                        self.logger.debug(f"Task {task_id} ready for execution")
                    else:
                        unallocated_tasks.append((priority, task_id))
                        # Only break if cluster has no free resources at all
                        if not self.cluster._resource_available.is_set():
                            self.logger.debug("No more free resources available")
                            break
                        else:
                            self.logger.debug(f"Insufficient resources for task {task_id}. Resources requested: {req}. Free resources: {self.cluster.get_status()}")
                
                if allocated_count == 0:
                    self.logger.warning("No tasks allocated in this cycle. Clearing resource available event to wait for new resources.")
                    self._cluster_resource.clear_resource_available()
                
                # Put back unallocated tasks
                for item in unallocated_tasks:
                    self._sorted_tasks.put_nowait(item)
                        
            except asyncio.CancelledError:
                self.logger.info("Scheduler Monitor task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)
    
    def start_monitoring(self) -> asyncio.Task:
        """Start the monitoring task. Must be called from async context."""
        if self._monitoring_task is not None and not self._monitoring_task.done():
            self.logger.warning("Monitor task already running")
            return
        
        # Store the event loop for thread-safe event signaling
        self._event_loop = asyncio.get_event_loop()
        self.cluster.set_event_loop(self._event_loop)
        self._stop_monitoring.clear()
        self._all_tasks_done.clear()
        self._consecutive_failed_allocations = 0
        self._monitor_task = asyncio.create_task(self._monitor_resources())
    
    async def stop_monitoring(self):
        """Stop the monitoring task gracefully."""
        self.logger.info("Stopping resource monitoring")
        self._stop_monitoring.set()
        
        # Wake up the monitor loop if it's blocked waiting for resources
        await self._cluster_resource.signal_resource_available()
        
        if self._monitor_task and not self._monitor_task.done():
            # Cancel immediately instead of waiting
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Resource monitoring stopped")

    def _check_all_tasks_done(self):
        """
        Check if all tasks are complete and signal completion event.
        Thread-safe - called from executor callbacks.
        """
        remaining = set(self.tasks.keys()) - (self._successful_tasks | self._failed_tasks)
        self.logger.debug(f"Checking completion: {len(remaining)} tasks remaining")
        if not remaining:
            self.logger.info("All tasks completed")
            if self._event_loop is not None:
                self.logger.debug(f"Setting _all_tasks_done event via stored loop {self._event_loop}")
                self._event_loop.call_soon_threadsafe(self._all_tasks_done.set)
            else:
                self.logger.warning("No event loop stored, setting event directly (may not work!)")
                self._all_tasks_done.set()

    async def wait_for_completion(self):
        """
        Wait for all tasks to complete.
        This replaces the while loop in the worker's run() method.
        """
        self.logger.debug("Waiting for all tasks to complete")
        await self._all_tasks_done.wait()
        self.logger.debug("Done waiting for task completion!")

    def add_task(self, task: Task) -> bool:
        try:
            if task.nnodes > len(self.cluster.nodes):
                raise ValueError(f"Task {task.task_id} requires {task.nnodes} nodes, but only {len(self.cluster.nodes)} are available")
            self.tasks[task.task_id] = task
            self._sorted_tasks.put_nowait((self.scheduler_policy.get_score(task),task.task_id))
            return True
        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False
    
    def delete_task(self, task: Task) -> bool:
        if task.task_id not in self.tasks:
            self.logger.warning(f"Unknown task: {task.task_id}")
            return False
        
        try:
            # Remove from tasks dict
            del self.tasks[task.task_id]

            # Remove from sorted tasks queue
            temp_items = []
            while not self._sorted_tasks.empty():
                try:
                    priority, tid = self._sorted_tasks.get_nowait()
                    if tid != task.task_id:
                        temp_items.append((priority, tid))
                except asyncio.QueueEmpty:
                    break
                
            # Put back all items except the deleted task
            for item in temp_items:
                self._sorted_tasks.put_nowait(item)


            # Remove from ready tasks queue if present
            temp_ready = []
            while not self.ready_tasks.empty():
                try:
                    tid, resource = self.ready_tasks.get_nowait()
                    if tid != task.task_id:
                        temp_ready.append((tid, resource))
                    else:
                        # Deallocate resource if task is in ready queue
                        self.cluster.deallocate(resource)
                except asyncio.QueueEmpty:
                    break

            # Put back all items except the deleted task
            for item in temp_ready:
                self.ready_tasks.put_nowait(item)

            # If running, free the resources
            if task.task_id in self._running_tasks:
                self.cluster.deallocate(self._running_tasks[task.task_id])

            # Remove from running and status sets
            del self._running_tasks[task.task_id]

            # Remove all occurrences from done_tasks
            if task.task_id in self._done_tasks:
                del self._done_tasks[task.task_id]

            #remove from failed and succesful tasks
            self._failed_tasks.discard(task.task_id)
            self._successful_tasks.discard(task.task_id)

            return True
        except Exception as e:
            self.logger.warning(f"Failed to delete task {task.task_id}: {e}")
            return False
    
    def free(self, task_id: str, status: TaskStatus):
        if task_id in self.tasks:
            if task_id not in self._running_tasks:
                self.logger.error(f"{task_id} is not running")
                raise RuntimeError

            # deallocate
            self.cluster.deallocate(self._running_tasks[task_id])

            # delete from running tasks
            del self._running_tasks[task_id]

            # Add to done tasks
            self._done_tasks[task_id] += 1

            # add to failed/successful tasks
            if status == TaskStatus.FAILED:
                self._failed_tasks.add(task_id)
            elif status == TaskStatus.SUCCESS:
                self._successful_tasks.add(task_id)
                self._failed_tasks.discard(task_id)

            self.logger.debug(f"Freed {task_id}")
        self._check_all_tasks_done()
        return None
    
    def get_task_assignment(self):
        return copy.deepcopy(self._running_tasks)
    
    @property
    def running_tasks(self) -> Set[str]:
        """Return IDs of currently running tasks."""
        return copy.deepcopy(self._running_tasks)

    @property
    def failed_tasks(self) -> Set[str]:
        """Return IDs of tasks that have failed."""
        return copy.deepcopy(self._failed_tasks)

    @property
    def done_tasks(self) -> List[str]:
        """Return IDs of tasks that are done. Can have duplicates"""
        return copy.deepcopy(self._done_tasks)

    @property
    def successful_tasks(self) -> Set[str]:
        """Return IDs of tasks that have completed successfully."""
        return copy.deepcopy(self._successful_tasks)
    
    @property
    def remaining_tasks(self) -> Set[str]:
        return set(self.tasks.keys()) - (self._successful_tasks | self._failed_tasks)