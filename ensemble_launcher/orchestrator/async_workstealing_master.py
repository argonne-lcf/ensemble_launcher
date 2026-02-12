from .async_worker import AsyncWorker
from .node import Node
from ensemble_launcher.comm.messages import TaskUpdate, Action, ActionType, TaskRequest, ResultBatch
from ensemble_launcher.ensemble import Task
from typing import Optional, List, Dict
import asyncio
from .async_master import AsyncMaster
from ensemble_launcher.scheduler.resource import NodeResourceList, NodeResourceCount
from asyncio import Future as AsyncFuture
import copy

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

    def _create_children(self, tasks: Dict[str, Task], include_tasks: bool = False) -> Dict[str, Node]:
        """
        Override parent method to enable work-stealing mode.
        
        All tasks are kept unassigned initially for dynamic distribution to workers.
        """
        children_are_workers = (self.level + 1 == self._config.nlevels)
        
        if children_are_workers:
            # Import here to avoid circular dependency
            from .async_workstealing_worker import AsyncWorkStealingWorker
            
            # Keep all tasks in the unassigned pool
            self._unassigned_tasks = dict(tasks)
            self.logger.info(f"{self.node_id}: Work-stealing mode - all {len(self._unassigned_tasks)} tasks kept unassigned")
            
            # Create children with task assignments from policy
            assignments, remove_tasks = self._scheduler.assign(tasks, self.level)

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
            
            start_wid = len(self._child_assignment)
            for wid, alloc in assignments.items():
                child_id = self.node_id + f".w{wid + start_wid}"
                self._child_assignment[child_id] = alloc
                self._child_assignment[child_id]["wid"] = wid
                # Store pre-assigned task_ids as priority tasks for this child
                # These will be sent first when child requests tasks
                if "task_ids" in alloc and alloc["task_ids"]:
                    self._child_assignment[child_id]["priority_task_ids"] = list(alloc["task_ids"])
                else:
                    self._child_assignment[child_id]["priority_task_ids"] = []
                # Check if allocation specifies a custom task_executor_name
                child_config = self._config
                if "task_executor_name" in alloc:
                    child_config = self._config.model_copy(update={"task_executor_name": alloc["task_executor_name"]})
                    self.logger.info(f"{self.node_id}: Child {child_id} using task_executor_name: {alloc['task_executor_name']}")
                # Create a work-stealing worker with no tasks
                children[child_id] = AsyncWorkStealingWorker(
                        child_id,
                        config=child_config,
                        Nodes=alloc["job_resource"],
                        tasks={},
                        parent=None
                    )
            
            return children
        else:
            # Non-worker children: use parent's logic (which will choose the right master class)
            return super()._create_children(tasks, include_tasks)

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
                    
                    # Check if this child has pre-assigned priority tasks from policy
                    priority_task_ids = self._child_assignment[child_id].get("priority_task_ids", [])
                    assigned_task_ids = []
                    
                    # First, send priority tasks if available
                    if priority_task_ids:
                        # Get up to ntasks from priority list
                        num_to_send = min(len(priority_task_ids), task_request.ntasks)
                        assigned_task_ids = priority_task_ids[:num_to_send]
                        # Remove assigned tasks from priority list
                        self._child_assignment[child_id]["priority_task_ids"] = priority_task_ids[num_to_send:]
                        self.logger.info(f"{self.node_id}: Sending {len(assigned_task_ids)} priority tasks to {child_id}")
                    else:
                        # No priority tasks, use policy to assign from unassigned pool
                        worker_assignment = {0: self._child_assignment[child_id].copy()}
                        worker_assignment[0]["task_ids"] = []  # Start with empty task list
                        
                        # Let policy decide which tasks to assign
                        updated_assignment, removed_tasks = self._scheduler.policy.get_task_assignment(
                            tasks=copy.deepcopy(self._unassigned_tasks),
                            worker_assignments=worker_assignment,
                            ntask=task_request.ntasks,
                            free_resources = task_request.free_resources
                        )
                        
                        assigned_task_ids = updated_assignment[0]["task_ids"]
                        for task_id in assigned_task_ids:
                            del self._unassigned_tasks[task_id]
                        self.logger.info(f"{self.node_id}: Policy assigned {len(assigned_task_ids)} tasks to {child_id}")
                    
                    # Check if we have any tasks to assign
                    if len(assigned_task_ids) == 0:
                        # No tasks available, send stop message
                        self.logger.info(f"{self.node_id}: No tasks to assign to {child_id}, sending stop message")
                        stop_action = Action(sender=self.node_id, type=ActionType.STOP)
                        await self._comm.send_message_to_child(child_id, stop_action)
                    else:
                        # Get task objects and remove from unassigned pool
                        available_tasks = [self._tasks[task_id] for task_id in assigned_task_ids]
                        
                        # Send task update to child
                        task_update = TaskUpdate(sender=self.node_id, added_tasks=available_tasks)
                        await self._comm.send_message_to_child(child_id, task_update)
                        self.logger.info(f"{self.node_id}: Sent {len(available_tasks)} tasks to {child_id} (requested {task_request.ntasks})")
                        
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
    
    def _mark_and_launch(self, child_ids: List[str]):
        """Mark children done and potentially launch new ones if tasks remain.
        
        This method marks the completed children as done and checks if there are 
        unassigned tasks remaining. If so, it schedules the launch of new children.
        """
        # Mark the children as done
        super()._mark_children_done(child_ids)
        
        # # Check if there are still tasks left to assign
        # if len(self._unassigned_tasks) > 0:
        #     self.logger.info(f"{self.node_id}: {len(self._unassigned_tasks)} tasks remain, scheduling relaunch")
        #     # Schedule async relaunch
        #     asyncio.create_task(self._relaunch_children())
    
    async def _relaunch_children(self):
        """Asynchronously create, launch, and sync new children for remaining tasks."""
        try:
            # Create new children for remaining tasks
            children = self._create_children(self._unassigned_tasks)
            self.logger.info(f"{self.node_id}: Created {len(children)} new children for relaunching")
            
            # Add children and setup parent relationships
            for child_id, child in children.items():
                self.add_child(child_id, child.info())
                child.set_parent(self.info())
                child.parent_comm = self.comm.pickable_copy()
            
            # Update comm with new children (existing monitors will handle them)
            await self._comm.update_node_info(self.info())
            
            # Launch the new children
            await self._launch_children(children)
            
            # Sync with each new child individually
            for child_id in children.keys():
                # Build node and task updates
                node_update = self._build_init_node_update(child_id)
                task_update = self._build_init_task_update(child_id)
                
                # Sync with this child
                result = await self._sync_with_child(child_id, node_update, task_update)
                if result is not None:
                    self.logger.error(f"{self.node_id}: Failed to sync with relaunched child {child_id}: {result.exception}")
                else:
                    # Only start monitor if sync succeeded
                    task = asyncio.create_task(
                        self._monitor_single_child_task_requests(child_id),
                        name=f"task_monitor_{child_id}"
                    )
                    self._task_monitor_tasks.append(task)
                    self.logger.info(f"{self.node_id}: Started task monitor for relaunched child {child_id}")
                
        except Exception as e:
            self.logger.error(f"{self.node_id}: Error during relaunch: {e}")
        
    def create_done_callback(self, child_ids: List[str]):
        def _done_callback(future: AsyncFuture):
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._mark_and_launch, child_ids)
            else:
                self.logger.warning("No event loop stored, can't mark child done!")
        return _done_callback
    
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
