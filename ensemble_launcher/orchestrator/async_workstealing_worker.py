
from typing import Optional, Dict
import asyncio

from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import JobResource
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.comm import NodeInfo, AsyncComm
from ensemble_launcher.comm.messages import TaskRequest, TaskUpdate, Action, ActionType

from .async_worker import AsyncWorker

class AsyncWorkStealingWorker(AsyncWorker):
    """
    Work-stealing variant of AsyncWorker that dynamically requests tasks from master.
    
    When this worker's task queue is empty, it requests more tasks from the master.
    Instead of completing when all tasks are done, it waits for a STOP signal from master.
    """
    
    type = "AsyncWorkStealingWorker"
    
    def __init__(self,
                id:str,
                config:LauncherConfig,
                Nodes: Optional[JobResource] = None,
                tasks: Optional[Dict[str, Task]] = None,
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[AsyncComm] = None
                ):
        
        super().__init__(id,config,Nodes,tasks,parent,children,parent_comm)
        self._stop_signal_received = asyncio.Event()
    
    async def _receive_initial_tasks(self):
        """Override to send initial task request instead of waiting for TaskUpdate."""
        self.logger.info(f"{self.node_id}: Sending initial task request")
        await self._request_tasks_from_master()
    
    async def _wait_for_stop_condition(self):
        """Override to wait for STOP signal from master instead of task completion."""
        self.logger.info("Waiting for STOP signal from master")
        await self._stop_signal_received.wait()
        self.logger.info("Received STOP signal")
    
    def _task_callback(self, task: Task, future):
        """Override to check if we need more tasks after each task completion."""
        # Call parent callback to handle normal task completion
        super()._task_callback(task, future)
        
        # Check if we need to request more tasks
        if self._scheduler._sorted_tasks.empty() and not self._stop_signal_received.is_set():
            self.logger.debug(f"{self.node_id}: Task queue empty after {task.task_id} completion, triggering task request")
            # Schedule task request in event loop
            self._event_loop.call_soon_threadsafe(
                asyncio.create_task,
                self._request_tasks_from_master()
            )
    
    async def _request_tasks_from_master(self):
        """
        Request tasks from master when local queue is empty.
        Called from task completion callback.
        """
        try:
            # Calculate how many tasks to request based on available resources
            ntasks = self.nodes.resources[0].cpu_count * len(self.nodes.nodes)
            
            self.logger.info(f"{self.node_id}: Requesting {ntasks} tasks from master")
            
            # Send task request
            task_request = TaskRequest(sender=self.node_id, ntasks=ntasks)
            await self._comm.send_message_to_parent(task_request)
            
            # Wait for response - either TaskUpdate or Action(STOP), whichever comes first
            task_update_task = asyncio.create_task(
                self._comm.recv_message_from_parent(TaskUpdate, timeout=None, block = True)
            )
            action_task = asyncio.create_task(
                self._comm.recv_message_from_parent(Action, timeout=None, block = True)
            )
            
            done, pending = await asyncio.wait(
                [task_update_task, action_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            
            # Check which task completed
            for task in done:
                result = await task
                if isinstance(result, TaskUpdate):
                    self.logger.info(f"{self.node_id}: Received {len(result.added_tasks)} tasks from master")
                    if len(result.added_tasks) > 0:
                        self._update_tasks(result)
                    else:
                        self.logger.debug(f"{self.node_id}: Received empty task update")
                elif isinstance(result, Action) and result.type == ActionType.STOP:
                    self.logger.info(f"{self.node_id}: Received STOP signal from master")
                    self._stop_signal_received.set()
        
        except Exception as e:
            self.logger.error(f"Error requesting tasks from master: {e}", exc_info=True)
    
