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
        """Wait for an explicit STOP signal rather than local task exhaustion.

        Unlike the base ``AsyncWorker``, this worker never exits simply because
        its local task queue is empty.  The master may steal tasks from other
        workers and push them here at any time, so the worker must stay alive
        until told otherwise.

        Behaviour depends on deployment mode:

        Non-cluster:
            Awaits ``_stop_signal_received``, which is set when the master sends
            a STOP Action over the comm channel (handled in the parent-message
            monitor).  Local task exhaustion alone does not trigger shutdown.

        Cluster mode:
            Loops receiving Action messages from the parent master.  On receipt
            of a STOP action, sets ``_stop_signal_received`` and exits.
            Behaviour is the same regardless of whether this is a root or
            non-root worker since work-stealing workers always have a master.
        """
        if self._config.cluster:
            self.logger.info("Cluster mode enabled - listening for stop message")
            while not self._stop_signal_received.is_set():
                msg = await self._comm.recv_message_from_parent(Action, block=True)
                if msg.type == ActionType.STOP:
                    self._stop_signal_received.set()
        else:
            self.logger.info("Waiting for stop event to be set")
            await self._stop_signal_received.wait()
            self.logger.info("Received STOP signal")

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
                elif isinstance(result, Action) and result.type == ActionType.STOP:
                    self.logger.info(
                        f"{self.node_id}: Received STOP signal from master"
                    )
                    self._stop_signal_received.set()

        except Exception as e:
            self.logger.error(f"Error requesting tasks from master: {e}", exc_info=True)
        finally:
            # Always clear the flag to prevent deadlock
            self._task_request_in_progress.clear()
