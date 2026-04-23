import asyncio
import random
from typing import Dict, List, Optional

from ensemble_launcher.comm import AsyncComm, NodeInfo
from ensemble_launcher.comm.messages import Stop, StopType, TaskRequest
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

    # ------------------------------------------------------------------
    # Initialisation overrides
    # ------------------------------------------------------------------

    async def _receive_initial_tasks(self) -> None:
        """No-op: tasks are requested dynamically after full init, not during parent sync."""
        pass

    async def _lazy_init(self) -> None:
        """Extend base init: start periodic task requester and task update monitor."""
        await super()._lazy_init()
        asyncio.create_task(self._periodic_task_requester())
        # Base only starts _task_update_monitor in cluster mode; start it always here.
        if self.parent and not self._config.cluster:
            self._task_update_task = asyncio.create_task(self._task_update_monitor())

    # ------------------------------------------------------------------
    # Stop condition
    # ------------------------------------------------------------------

    async def _wait_for_stop_condition(self) -> None:
        """Wait for the condition that ends this work-stealing worker's execution loop.

        Unlike the base AsyncWorker, this worker has no upfront task pool — tasks
        arrive on-demand via TaskRequests.  There is therefore no task-completion
        condition; the worker stays alive until externally stopped.

        Races conditions with asyncio.FIRST_COMPLETED:
        1. SIGTERM on this process.
        2. Parent dead locally — consecutive send failures exceeded threshold.
        3. Stop(TERMINATE/KILL) received from parent (non-root only).
        """
        import sys

        stop_tasks: Dict[str, asyncio.Task] = {}
        received_stop_type: List[Optional[StopType]] = [None]

        stop_tasks["stop_signal"] = asyncio.create_task(
            self._stop_signal_received.wait()
        )
        if self._comm.parent_dead_event is not None:
            stop_tasks["parent_dead"] = asyncio.create_task(
                self._comm.parent_dead_event.wait()
            )

        if self.parent is not None:

            async def _recv_stop_from_parent():
                msg = await self._comm.recv_message_from_parent(Stop, block=True)
                if msg is not None:
                    received_stop_type[0] = msg.type

            stop_tasks["parent_stop"] = asyncio.create_task(_recv_stop_from_parent())

        done, pending = await asyncio.wait(
            set(stop_tasks.values()), return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if (
            self._comm.parent_dead_event is not None
            and self._comm.parent_dead_event.is_set()
        ):
            await self.stop()
            sys.exit(1)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _task_callback(self, task: Task, future) -> None:
        """After each task completes, delegate to base callback."""
        super()._task_callback(task, future)
        # Periodic requester loop handles task requests; callback-triggered requests removed.
        # if (
        #     self._scheduler._sorted_tasks.empty()
        #     and not self._stop_signal_received.is_set()
        # ):
        #     self.logger.debug(
        #         f"{self.node_id}: Task queue empty after {task.task_id} completion, requesting more"
        #     )
        #     asyncio.create_task(self._request_tasks_from_master())

    async def _request_tasks_from_master(self) -> None:
        """Send a TaskRequest to the master (fire-and-forget).

        The response (TaskUpdate or Stop) is handled separately: TaskUpdate by
        _task_update_monitor, Stop by _wait_for_stop_condition.
        """
        if (
            self._scheduler.cluster.free_cpus
            and self._scheduler.not_ready_tasks.empty()
        ):
            try:
                ntasks = (
                    self._scheduler.cluster.free_cpus
                    if self._config.task_request_size is None
                    else self._config.task_request_size
                )
                task_request = TaskRequest(sender=self.node_id, ntasks=ntasks)
                self.logger.debug(
                    f"{self.node_id}: Requesting {ntasks} tasks from master"
                )
                await self._comm.send_message_to_parent(task_request)

            except Exception as e:
                self.logger.error(f"Error sending task request: {e}", exc_info=True)
        else:
            self.logger.debug(
                "Skipping task request. Either no free resourcses or still have tasks in queue"
            )

    async def _periodic_task_requester(self) -> None:
        """Periodically send TaskRequests to master at a fixed interval."""
        while not self._stop_signal_received.is_set():
            await self._request_tasks_from_master()
            ## Avoiding thundering herd issue
            jitter = random.uniform(0.9, 1.1)
            await asyncio.sleep(jitter * self._config.task_request_interval)
