from .async_worker import AsyncWorker
from .node import Node
from ensemble_launcher.comm.messages import TaskUpdate, Action, ActionType, TaskRequest, ResultBatch
from ensemble_launcher.ensemble import Task
from typing import Optional, List, Dict
import asyncio
from .async_master import AsyncMaster
from ensemble_launcher.scheduler.resource import NodeResourceList, NodeResourceCount

class AsyncWorkStealingMaster(AsyncMaster):
    """
    Work-stealing variant of AsyncMaster that enables dynamic task distribution.
    
    This master keeps tasks unassigned initially and distributes them on-demand
    when workers request more work, enabling better load balancing.
    """
    
    type = "AsyncWorkStealingMaster"
    
    def __init__(self,
                id:str,
                config,
                Nodes=None,
                tasks=None,
                parent=None,
                children=None,
                parent_comm=None):
        super().__init__(id, config, Nodes, tasks, parent, children, parent_comm)
        
        # Track unassigned tasks for dynamic task updates
        self._unassigned_tasks: Dict[str, Task] = {}
        self._task_monitor_tasks: List[asyncio.Task] = []
        self._stop_task_monitor_event = asyncio.Event()

    def _create_children(self, include_tasks: bool = False) -> Dict[str, Node]:
        """
        Override parent method to enable work-stealing mode.
        
        All tasks are kept unassigned initially for dynamic distribution to workers.
        """
        children_are_workers = (self.level + 1 == self._config.nlevels)
        
        if children_are_workers:
            # Import here to avoid circular dependency
            from .async_workstealing_worker import AsyncWorkStealingWorker
            
            # Keep all tasks in the unassigned pool
            self._unassigned_tasks = dict(self._tasks)
            self.logger.info(f"{self.node_id}: Work-stealing mode - all {len(self._unassigned_tasks)} tasks kept unassigned")
            
            # Create children without task assignments (resources only)
            assignments, remove_tasks = self._scheduler.assign({}, self.level)  # Pass empty task dict

            ##since this master does a lot of work, overlaoding cpus can cause much higher task request latency
            ##remove the first cpu from the first node of first child
            first_child_job_resource = assignments[0]["job_resource"]
            first_node = first_child_job_resource.resources[0]
            if isinstance(first_node, NodeResourceList):
                new_first_node = NodeResourceList(cpus=first_node.cpus[1:],gpus=first_node.gpus)
            else:
                new_first_node = NodeResourceCount(ncpus=first_node.cpu_count - 1, ngpus=first_node.gpu_count)
            first_child_job_resource.resources[0] = new_first_node

            if len(remove_tasks) > 0:
                self.logger.warning(f"Removed tasks due to resource constraints: {remove_tasks}")
            
            self._child_assignment = {}
            children = {}
            
            for wid, alloc in assignments.items():
                child_id = self.node_id + f".w{wid}"
                self._child_assignment[child_id] = alloc
                # Create a work-stealing worker with no tasks
                children[child_id] = AsyncWorkStealingWorker(
                        child_id,
                        config=self._config,
                        Nodes=alloc["job_resource"],
                        tasks={},
                        parent=None
                    )
            
            return children
        else:
            # Non-worker children: use parent's logic (which will choose the right master class)
            return super()._create_children(include_tasks)

    async def _monitor_single_child_task_requests(self, child_id: str):
        """Dedicated monitor for task requests from a single child."""
        self.logger.debug(f"{self.node_id}: Started monitoring task requests from child {child_id}")
        failures = 0
        
        while not self._stop_task_monitor_event.is_set():
            try:
                # Blocking recv - instantly returns when message arrives
                task_request: TaskRequest = await self._comm.recv_message_from_child(TaskRequest, child_id=child_id, block = True)
                
                if task_request is not None:
                    failures = 0  # Reset failure counter on success
                    self.logger.info(f"{self.node_id}: Received task request from {child_id} for {task_request.ntasks} tasks")
                    
                    # Check if there are no unassigned tasks left
                    if len(self._unassigned_tasks) == 0:
                        self.logger.info(f"{self.node_id}: No unassigned tasks remaining, sending stop message to {child_id}")
                        stop_action = Action(sender=self.node_id, type=ActionType.STOP)
                        await self._comm.send_message_to_child(child_id, stop_action)
                        self._stop_task_monitor_event.set()
                        break
                    
                    # Use policy to assign tasks to this child
                    worker_assignment = {0: self._child_assignment[child_id].copy()}
                    worker_assignment[0]["task_ids"] = []  # Start with empty task list
                    
                    # Let policy decide which tasks to assign
                    updated_assignment, removed_tasks = self._scheduler.policy.get_task_assignment(
                        tasks=self._unassigned_tasks,
                        worker_assignments=worker_assignment,
                        ntask=task_request.ntasks
                    )
                    
                    assigned_task_ids = updated_assignment[0]["task_ids"]
                    
                    if assigned_task_ids:
                        # Get task objects and remove from unassigned pool
                        available_tasks = [self._unassigned_tasks[task_id] for task_id in assigned_task_ids]
                        for task_id in assigned_task_ids:
                            del self._unassigned_tasks[task_id]
                        
                        # Send task update to child
                        task_update = TaskUpdate(sender=self.node_id, added_tasks=available_tasks)
                        await self._comm.send_message_to_child(child_id, task_update)
                        self.logger.info(f"{self.node_id}: Sent {len(available_tasks)} tasks to {child_id} (requested {task_request.ntasks})")
                    else:
                        # No tasks available that fit, send empty update
                        task_update = TaskUpdate(sender=self.node_id, added_tasks=[])
                        await self._comm.send_message_to_child(child_id, task_update)
                        self.logger.info(f"{self.node_id}: No tasks available that fit for {child_id}")
                        
            except asyncio.CancelledError:
                self.logger.info(f"{self.node_id}: Task monitor for child {child_id} cancelled")
                break
            except Exception as e:
                failures += 1
                self.logger.error(f"{self.node_id}: Error monitoring task requests from child {child_id} (failure {failures}): {e}")
                if failures >= 10:
                    await asyncio.sleep(0.1)  # Backoff after repeated failures
        
        self.logger.debug(f"{self.node_id}: Stopped monitoring task requests from child {child_id}")
    
    async def monitor_task_requests(self):
        """Monitor for task requests from all children - one dedicated task per child."""
        if len(self.children) == 0:
            self.logger.debug(f"{self.node_id}: No children to monitor for task requests")
            return
        
        self.logger.info(f"{self.node_id}: Starting task request monitor for {len(self.children)} children")
        
        # Create one monitoring task per child
        self._task_monitor_tasks = [
            asyncio.create_task(
                self._monitor_single_child_task_requests(child_id), 
                name=f"task_monitor_{child_id}"
            )
            for child_id in self.children
        ]
        
        # Wait for all tasks to complete (they run until stop_event is set)
        await asyncio.gather(*self._task_monitor_tasks, return_exceptions=True)
        
        self.logger.info(f"{self.node_id}: Task request monitor stopped")
    
    async def _lazy_init(self):
        children = await super()._lazy_init()

        self.logger.info(f"I am workstealing master")

        # Start task request monitor if children are workers
        if self.level + 1 == self._config.nlevels:
            asyncio.create_task(self.monitor_task_requests())
            self.logger.info(f"{self.node_id}: Started task request monitor")
        
        return children
    
    def _build_init_task_update(self, child_id: str):
        return None
    
    async def stop(self):
        # Stop the task monitor loop if it was started
        if self._task_monitor_tasks:
            self._stop_task_monitor_event.set()
            self.logger.info(f"{self.node_id}: Signaled task monitor tasks to stop")
            # Cancel all task monitor tasks
            for task in self._task_monitor_tasks:
                task.cancel()
            # Wait for them to finish
            await asyncio.gather(*self._task_monitor_tasks, return_exceptions=True)
            self.logger.info(f"{self.node_id}: All task monitor tasks stopped")

        return await super().stop()