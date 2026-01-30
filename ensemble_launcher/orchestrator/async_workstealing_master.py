from .async_worker import AsyncWorker
from .node import Node
from ensemble_launcher.comm.messages import TaskUpdate, Action, ActionType, TaskRequest, ResultBatch
from ensemble_launcher.ensemble import Task
from typing import Optional, List, Dict
import asyncio
from .async_master import AsyncMaster

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
                    
                    # Get child's available resources
                    child_resources = self._child_assignment[child_id]["job_resource"]
                    
                    # Filter tasks that can fit in child's resources
                    available_tasks = []
                    for task in self._unassigned_tasks.values():
                        if len(available_tasks) >= task_request.ntasks:
                            break  # Got enough tasks
                        
                        # Build task resource requirements
                        task_resource = task.get_resource_requirements()
                        
                        # Check if task fits in child's resources
                        if task_resource in child_resources:
                            available_tasks.append(task)
                    
                    if available_tasks:
                        # Remove assigned tasks from unassigned pool
                        for task in available_tasks:
                            del self._unassigned_tasks[task.task_id]
                        
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
    
    async def _sync_with_children(self):
        """Override to send empty task updates for work-stealing mode."""
        from ensemble_launcher.comm.messages import NodeUpdate
        
        for child_id in self.children:
            if not await self._comm.sync_heartbeat_with_child(child_id=child_id, timeout=30.0):
                self.logger.error(f"Failed to sync heartbeat with child {child_id}")
                return await self._get_child_exceptions()
            
            # Send node update only - worker will request tasks
            child_nodes = self._child_assignment[child_id]["job_resource"]
            node_update = NodeUpdate(sender=self.node_id, nodes=child_nodes)
            await self._comm.send_message_to_child(child_id, node_update)
            self.logger.info(f"{self.node_id}: Sent node update to {child_id} containing {len(child_nodes.nodes)} nodes (waiting for task request)")
    
    async def run(self):
        """Override to add task request monitoring for work-stealing."""
        async with self._timer("init"):
            children = await self._lazy_init()
        
        async with self._timer("launch_children"):
            await self._launch_children(children)

        async with self._timer("sync_with_children"):
            await self._sync_with_children()
            
            asyncio.create_task(self.report_status())
            
            # Start task request monitor if children are workers
            if self.level + 1 == self._config.nlevels:
                asyncio.create_task(self.monitor_task_requests())
                self.logger.info(f"{self.node_id}: Started task request monitor")
            
            return await self._results()
    
    async def _results(self) -> ResultBatch:
        """Wrap parent's _results to handle task monitor cleanup."""
        await self._all_children_done_event.wait()
        self.logger.info(f"{self.node_id}: All children have completed execution")

        # Stop the reporting loop
        self._stop_reporting_event.set()
        self.logger.info(f"{self.node_id}: Stopped reporting loop")
        
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

        # Call parent's result collection logic
        return await super()._results()
