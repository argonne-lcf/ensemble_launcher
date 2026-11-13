import asyncio

from ensemble_launcher.comm.messages import TaskRequest
from ensemble_launcher.scheduler import AsyncChildrenScheduler

from .async_master import AsyncMaster
from .node import Node


class AsyncWorkStealingMaster(AsyncMaster):
    """Work-stealing variant of AsyncMaster: children pull their tasks.

    The scheduler runs in pull mode, so incoming tasks wait in its pool. Each child sends
    TaskRequests; ``scheduler.request(child, n)`` places up to ``n`` tasks on that child's
    outbox, and the shared per-child sender sends them. Everything else -- starting, routing
    input, stopping -- is the base AsyncMaster.
    """

    type = "AsyncWorkStealingMaster"
    _placement = AsyncChildrenScheduler.PULL

    # ------------------------------------------------------------------
    # Child class selection
    # ------------------------------------------------------------------

    def _get_child_class(self) -> type:
        """Return AsyncWorkStealingWorker for leaf level, otherwise delegate to base."""
        if self.level + 1 == self._config.policy_config.nlevels:
            from .async_workstealing_worker import AsyncWorkStealingWorker

            return AsyncWorkStealingWorker
        return super()._get_child_class()

    # ------------------------------------------------------------------
    # Task requests
    # ------------------------------------------------------------------

    async def _init_child(self, child_id: str, child: Node) -> None:
        """Base per-child monitors, plus one serving the child's TaskRequests."""
        await super()._init_child(child_id, child)
        self._child_task_request_task[child_id] = asyncio.create_task(
            self._monitor_single_child_task_requests(child_id)
        )

    async def _monitor_single_child_task_requests(self, child_id: str) -> None:
        """Serve a child's TaskRequests until cancelled: place up to the requested number of
        pool tasks on its outbox. Only the latest of several queued requests counts."""
        self.logger.info(f"{self.node_id}: Started monitoring task requests from {child_id}")
        failures = 0
        while True:
            try:
                request: TaskRequest = await self._comm.recv_message_from_child(
                    TaskRequest, child_id=child_id, block=True
                )
                if request is None:
                    continue
                failures = 0
                while True:
                    newer = await self._comm.recv_message_from_child(
                        TaskRequest, child_id=child_id, block=False
                    )
                    if newer is None:
                        break
                    request = newer
                n = self._scheduler.request(child_id, request.ntasks)
                self.logger.debug(
                    f"{self.node_id}: {child_id} asked for {request.ntasks} tasks, placed {n}"
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                failures += 1
                self.logger.error(
                    f"{self.node_id}: Error serving task requests from {child_id} "
                    f"(failure {failures}): {e}"
                )
                if failures >= 10:
                    await asyncio.sleep(0.1)
        self.logger.debug(f"{self.node_id}: Stopped monitoring task requests from {child_id}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _lazy_init(self) -> None:
        """Base lazy init, plus a log line with the pool size."""
        await super()._lazy_init()
        self.logger.info(
            f"{self.node_id}: Work-stealing mode — {len(self._scheduler.state.unassigned)} "
            f"tasks in the pool"
        )

    async def _relaunch_children(self) -> None:
        """Create, launch, and sync a fresh set of children."""
        try:
            children = self._create_children()
            self.logger.info(
                f"{self.node_id}: Created {len(children)} new children for relaunching"
            )
            for child_id, child in children.items():
                await self._init_child(child_id, child)
            child_names = list(children.keys())
            results = await self._launch_and_sync_children(child_names)
            for child_id, result in zip(child_names, results):
                if result is not None:
                    self.logger.error(
                        f"{self.node_id}: Failed to sync with relaunched child {child_id}: {result.exception}"
                    )
                    await self._teardown_child(child_id)
        except Exception as e:
            self.logger.error(f"{self.node_id}: Error during relaunch: {e}")
