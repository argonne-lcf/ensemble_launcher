import asyncio
from typing import Dict, Optional

from ensemble_launcher.comm import AsyncComm, NodeInfo
from ensemble_launcher.comm.messages import Action, ActionType, TaskRequest, TaskUpdate
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import JobResource

from .async_worker import AsyncWorker


class AsyncWorkStealingWorker(AsyncWorker):
    """Work-stealing variant of AsyncWorker that dynamically requests tasks from master.

    Tasks are not assigned upfront; instead the worker requests a batch whenever
    its local queue drains. The worker stays alive until it receives a STOP action
    from the master rather than stopping when its initial task list is empty.
    """

    type = "AsyncWorkStealingWorker"

    def __init__(
        self,
        id: str,
        config: LauncherConfig,
        Nodes: Optional[JobResource] = None,
        tasks: Optional[Dict[str, Task]] = None,
        parent: Optional[NodeInfo] = None,
        children: Optional[Dict[str, NodeInfo]] = None,
        parent_comm: Optional[AsyncComm] = None,
    ) -> None:
        """Initialise work-stealing specific state on top of AsyncWorker."""
        super().__init__(id, config, Nodes, tasks, parent, children, parent_comm)
        self._task_request_in_progress = asyncio.Event()

    # ------------------------------------------------------------------
    # Initialisation overrides
    # ------------------------------------------------------------------

    async def _receive_initial_tasks(self) -> None:
        """No-op: tasks are requested dynamically after full init, not during parent sync."""
        pass

    async def _lazy_init(self) -> None:
        """Extend base init: fire the initial task request after the scheduler is ready."""
        await super()._lazy_init()
        self.logger.info(f"{self.node_id}: Sending initial task request")
        asyncio.create_task(self._request_tasks_from_master())

    # ------------------------------------------------------------------
    # Stop condition
    # ------------------------------------------------------------------

    async def _wait_for_stop_condition(self) -> None:
        """Wait for the condition that ends this work-stealing worker's execution loop.

        Unlike the base AsyncWorker, this worker has no upfront task pool — tasks
        arrive on-demand via TaskRequests.  There is therefore no task-completion
        condition; the worker stays alive until externally stopped.

        Races two conditions with asyncio.FIRST_COMPLETED:
        1. STOP action received from parent (non-root) / external signal (root).
        2. Parent dead locally — consecutive send failures exceeded threshold.
        """
        stop_tasks: Dict[str, asyncio.Task] = {}

        # Condition 1: Stop event set either explicitly or received from parent
        if self.parent is None:
            # Root worker: external stop signal
            stop_tasks["stop_signal"] = asyncio.create_task(
                self._stop_signal_received.wait()
            )
        else:

            async def _recv_stop_from_parent():
                while True:
                    msg = await self._comm.recv_message_from_parent(Action, block=True)
                    if msg is not None and msg.type == ActionType.STOP:
                        return

            stop_tasks["parent_stop"] = asyncio.create_task(_recv_stop_from_parent())

        # Condition 2: parent detected dead locally
        stop_tasks["parent_dead"] = asyncio.create_task(self._parent_dead_event.wait())

        if not stop_tasks:
            # Should not happen, but guard against empty wait
            return

        done, pending = await asyncio.wait(
            set(stop_tasks.values()), return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _task_callback(self, task: Task, future) -> None:
        """After each task completes, request more tasks if the local queue is now empty."""
        super()._task_callback(task, future)

        if (
            self._scheduler._sorted_tasks.empty()
            and not self._stop_signal_received.is_set()
        ):
            self.logger.debug(
                f"{self.node_id}: Task queue empty after {task.task_id} completion, requesting more"
            )
            asyncio.create_task(self._request_tasks_from_master())

    async def _request_tasks_from_master(self) -> None:
        """Send a TaskRequest to the master and process the response (TaskUpdate or STOP).

        A guard flag prevents concurrent requests. On receiving a TaskUpdate the
        tasks are forwarded to the scheduler; on STOP the stop event is set.
        """
        # Early check without setting the flag
        if self._task_request_in_progress.is_set():
            self.logger.debug(
                "Task request is already in progress. Not sending a new one"
            )
            return

        try:
            self._task_request_in_progress.set()

            # Calculate how many tasks to request based on available resources
            ntasks = self.nodes.resources[0].cpu_count * len(self.nodes.nodes)
            free_resources = self.nodes

            self.logger.info(f"{self.node_id}: Requesting {ntasks} tasks from master")

            # Send task request
            task_request = TaskRequest(
                sender=self.node_id, ntasks=ntasks, free_resources=free_resources
            )
            await self._comm.send_message_to_parent(task_request)

            # Wait for response - either TaskUpdate or Action(STOP), whichever comes first
            task_update_task = asyncio.create_task(
                self._comm.recv_message_from_parent(
                    TaskUpdate, timeout=None, block=True
                )
            )
            action_task = asyncio.create_task(
                self._comm.recv_message_from_parent(Action, timeout=None, block=True)
            )

            done, pending = await asyncio.wait(
                [task_update_task, action_task], return_when=asyncio.FIRST_COMPLETED
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
                    self.logger.info(
                        f"{self.node_id}: Received {len(result.added_tasks)} tasks from master"
                    )
                    if len(result.added_tasks) > 0:
                        self._update_tasks(result)
                    else:
                        self.logger.debug(f"{self.node_id}: Received empty task update")

        except Exception as e:
            self.logger.error(f"Error requesting tasks from master: {e}", exc_info=True)
        finally:
            # Always clear the flag to prevent deadlock
            self._task_request_in_progress.clear()
