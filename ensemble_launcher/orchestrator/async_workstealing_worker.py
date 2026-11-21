import asyncio
import random
from typing import Dict, Optional

from ensemble_launcher.comm import NodeInfo
from ensemble_launcher.comm.messages import TaskRequest
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import JobResource

from .async_worker import AsyncWorker


class AsyncWorkStealingWorker(AsyncWorker):
    """Work-stealing variant of AsyncWorker that asks its master for tasks.

    It sends a TaskRequest for its free cores whenever its local queue is empty; the master
    places that many tasks on its outbox. Otherwise it is the base worker.
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
        parent_data_conn=None,
        parent_hb_conn=None,
    ) -> None:
        """Initialise work-stealing specific state on top of AsyncWorker."""
        super().__init__(id, config, Nodes, tasks, parent, children, parent_data_conn=parent_data_conn, parent_hb_conn=parent_hb_conn)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    async def _lazy_init(self) -> None:
        """Base init, plus the periodic task requester. Starting and stopping are the base
        worker's: tasks arrive through _task_update_monitor, and the worker runs until its
        parent sends Stop."""
        await super()._lazy_init()
        self.logger.info("I am workstealing worker!")
        self._requester_task = asyncio.create_task(self._periodic_task_requester())

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------
    
    async def _request_tasks_from_master(self) -> None:
        """Send a TaskRequest to the master (fire-and-forget).

        The TaskUpdate that answers it is read by _task_update_monitor.
        """
        if (
            self._scheduler.cluster.free_cpus
            and self._scheduler.state.pending_tasks.empty()
        ):
            try:
                ntasks = (
                    self._scheduler.cluster.free_cpus
                    if self._config.task_request_size is None
                    else self._config.task_request_size
                )
                task_request = TaskRequest(sender=self.node_id, ntasks=ntasks)
                self.logger.info(
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
