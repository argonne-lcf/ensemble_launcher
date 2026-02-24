import asyncio
from asyncio import Future as AsyncFuture
from typing import Callable, Dict, List, Optional

from ensemble_launcher.comm.messages import (
    Action,
    ActionType,
    TaskRequest,
    TaskUpdate,
)
from ensemble_launcher.scheduler.resource import JobResource

from .async_master import AsyncMaster
from .node import Node


class AsyncWorkStealingMaster(AsyncMaster):
    """Work-stealing variant of AsyncMaster that enables dynamic task distribution.

    Tasks are kept in the scheduler's unassigned pool at startup and dispatched
    on-demand when workers send TaskRequests, enabling better load balancing
    across heterogeneous or slow-starting workers.

    The full task dict and unassigned task pool are owned by the scheduler
    (scheduler.tasks, scheduler.unassigned_task_ids), seeded at creation time.
    """

    type = "AsyncWorkStealingMaster"

    def __init__(
        self,
        id: str,
        config,
        Nodes=None,
        tasks=None,
        parent=None,
        children=None,
        parent_comm=None,
    ) -> None:
        """Initialise work-stealing specific state on top of AsyncMaster."""
        super().__init__(id, config, Nodes, tasks, parent, children, parent_comm)

        self._task_monitor_tasks: List[asyncio.Task] = []
        self._stop_task_monitor_event = asyncio.Event()
        self._monitor_task_requests_task: Optional[asyncio.Task] = None
        self._all_work_done_event = asyncio.Event()
        self._relaunch_attempts = 0
        self._max_relaunch_attempts = 2

    # ------------------------------------------------------------------
    # Child class selection
    # ------------------------------------------------------------------

    def _get_child_class(self) -> type:
        """Return AsyncWorkStealingWorker for leaf level, otherwise delegate to base."""
        if self.level + 1 == self._config.nlevels:
            from .async_workstealing_worker import AsyncWorkStealingWorker

            return AsyncWorkStealingWorker
        return super()._get_child_class()

    # ------------------------------------------------------------------
    # Children creation — resources only, no upfront task assignment
    # ------------------------------------------------------------------

    def _create_children(
        self,
        include_tasks: bool = False,
        partial: bool = False,
        nodes: Optional[JobResource] = None,
    ) -> Dict[str, Node]:
        """Allocate resources for children without assigning tasks upfront.

        Tasks remain in the scheduler's unassigned pool and are dispatched
        on-demand when workers issue TaskRequests. include_tasks is accepted
        for API compatibility but is intentionally ignored.
        """
        existing_ids = set(self._scheduler.children_names) if partial else set()

        self._scheduler.assign_resources(
            self.level, self.node_id, reset=not partial, nodes=nodes
        )

        if not partial:
            self._apply_resource_headroom()

        target_ids = set(self._scheduler.child_assignments.keys()) - existing_ids
        return self._instantiate_children(include_tasks, target_ids)

    def _build_init_task_update(self, child_id: str) -> None:
        """No initial task update is sent in work-stealing mode; tasks arrive on demand."""
        return None

    # ------------------------------------------------------------------
    # Dynamic task monitoring
    # ------------------------------------------------------------------

    async def _monitor_single_child_task_requests(self, child_id: str) -> None:
        """Serve task requests from a single child until the stop event is set.

        Blocks on each TaskRequest, assigns tasks from the unassigned pool via
        the scheduler, and replies with a TaskUpdate or a STOP Action if the
        pool is empty.
        """
        self.logger.debug(
            f"{self.node_id}: Started monitoring task requests from child {child_id}"
        )
        failures = 0

        while not self._stop_task_monitor_event.is_set():
            try:
                task_request: TaskRequest = await self._comm.recv_message_from_child(
                    TaskRequest, child_id=child_id, block=True
                )

                if task_request is not None:
                    failures = 0
                    self.logger.info(
                        f"{self.node_id}: Received task request from {child_id} for {task_request.ntasks} tasks"
                    )

                    child_assignments = self._scheduler.assign_task_ids(
                        self._scheduler.unassigned_task_ids,
                        ntask=task_request.ntasks,
                        child_ids=[child_id],
                    )

                    assigned_task_ids = child_assignments.get(child_id, [])
                    if not assigned_task_ids:
                        self.logger.info(
                            f"{self.node_id}: No tasks to assign, sending stop to {child_id}"
                        )
                        stop_action = Action(sender=self.node_id, type=ActionType.STOP)
                        await self._comm.send_message_to_child(child_id, stop_action)
                    else:
                        available_tasks = [
                            self._scheduler.tasks[tid] for tid in assigned_task_ids
                        ]
                        task_update = TaskUpdate(
                            sender=self.node_id, added_tasks=available_tasks
                        )
                        await self._comm.send_message_to_child(child_id, task_update)
                        self.logger.info(
                            f"{self.node_id}: Sent {len(available_tasks)} tasks to {child_id} (requested {task_request.ntasks})"
                        )

            except asyncio.CancelledError:
                self.logger.info(
                    f"{self.node_id}: Task monitor for child {child_id} cancelled"
                )
                break
            except Exception as e:
                failures += 1
                self.logger.error(
                    f"{self.node_id}: Error monitoring task requests from child {child_id} (failure {failures}): {e}"
                )
                if failures >= 10:
                    await asyncio.sleep(0.1)

        self.logger.debug(
            f"{self.node_id}: Stopped monitoring task requests from child {child_id}"
        )

    async def monitor_task_requests(self) -> None:
        """Spawn one _monitor_single_child_task_requests coroutine per child and await all."""
        if len(self.children) == 0:
            self.logger.debug(
                f"{self.node_id}: No children to monitor for task requests"
            )
            return

        self.logger.info(
            f"{self.node_id}: Starting task request monitor for {len(self.children)} children"
        )

        self._task_monitor_tasks = [
            asyncio.create_task(
                self._monitor_single_child_task_requests(child_id),
                name=f"task_monitor_{child_id}",
            )
            for child_id in self.children
        ]

        await asyncio.gather(*self._task_monitor_tasks, return_exceptions=True)
        self.logger.info(f"{self.node_id}: Task request monitor stopped")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _lazy_init(self) -> None:
        """Extend base lazy init: start the task-request monitor at the leaf level."""
        await super()._lazy_init()

        self.logger.info(f"I am workstealing master")

        if self.level + 1 == self._config.nlevels:
            self.logger.info(
                f"{self.node_id}: Work-stealing mode — {len(self._scheduler.unassigned_task_ids)} tasks in unassigned pool"
            )
            self._monitor_task_requests_task = asyncio.create_task(
                self.monitor_task_requests()
            )
            self.logger.info(f"{self.node_id}: Started task request monitor")

    def _mark_and_launch(self, child_ids: List[str]) -> None:
        """Mark children done; relaunch with remaining tasks or signal all-work-done.

        Called via call_soon_threadsafe from done callbacks. If all children
        are done but unassigned tasks remain, relaunches up to
        _max_relaunch_attempts times before giving up and setting the done event.
        """
        super()._mark_children_done(child_ids)
        if not self._scheduler.all_children_done:
            return
        if self._scheduler.unassigned_task_ids:
            if self._relaunch_attempts < self._max_relaunch_attempts:
                self._relaunch_attempts += 1
                self.logger.warning(
                    f"{self.node_id}: Relaunching children for remaining "
                    f"{len(self._scheduler.unassigned_task_ids)} tasks "
                    f"(attempt {self._relaunch_attempts}/{self._max_relaunch_attempts})"
                )
                asyncio.create_task(self._relaunch_children())
            else:
                self.logger.error(
                    f"{self.node_id}: Max relaunch attempts reached, "
                    f"{len(self._scheduler.unassigned_task_ids)} tasks will not be executed"
                )
                self._all_work_done_event.set()
        else:
            self._all_work_done_event.set()

    async def _relaunch_children(self) -> None:
        """Create, launch, and sync a fresh set of children for the remaining unassigned tasks."""
        try:
            children = self._create_children()
            self.logger.info(
                f"{self.node_id}: Created {len(children)} new children for relaunching"
            )

            for child_id, child in children.items():
                await self._init_child(child_id, child)

            # Rebuild the aggregate task so it includes the new collect tasks.
            if self._aggregate_task and not self._aggregate_task.done():
                self._aggregate_task.cancel()
                try:
                    await self._aggregate_task
                except asyncio.CancelledError:
                    pass
            self._aggregate_task = asyncio.create_task(
                self._aggregate_and_send_results(self._result_tasks)
            )

            child_names = list(children.keys())
            results = await self._launch_and_sync_children(child_names)

            for child_id, result in zip(child_names, results):
                if result is not None:
                    self.logger.error(
                        f"{self.node_id}: Failed to sync with relaunched child {child_id}: {result.exception}"
                    )
                    await self._teardown_child(child_id)
                else:
                    task = asyncio.create_task(
                        self._monitor_single_child_task_requests(child_id),
                        name=f"task_monitor_{child_id}",
                    )
                    self._task_monitor_tasks.append(task)
                    self.logger.info(
                        f"{self.node_id}: Started task monitor for relaunched child {child_id}"
                    )

        except Exception as e:
            self.logger.error(f"{self.node_id}: Error during relaunch: {e}")

    def _create_done_callback(
        self, child_ids: List[str]
    ) -> Callable[[AsyncFuture], None]:
        def _done_callback(future: AsyncFuture):
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(self._mark_and_launch, child_ids)
            else:
                self.logger.warning("No event loop stored, can't mark child done!")

        return _done_callback

    async def _wait_for_finish(self) -> None:
        """Done when all children are done and no unassigned tasks remain (with retries)."""
        if self._config.cluster:
            self.logger.info("Cluster mode - listening for stop message from parent")
            if self.parent is not None:
                while not self._stop_signal_received.is_set():
                    msg = await self._comm.recv_message_from_parent(Action, block=True)
                    if msg.type == ActionType.STOP:
                        for child_id in self.children:
                            await self._comm.send_message_to_child(
                                child_id, Action(type=ActionType.STOP)
                            )
                        await self._aggregate_task
                        self._stop_signal_received.set()
            else:
                await self._stop_signal_received.wait()
                for child_id in self.children:
                    await self._comm.send_message_to_child(
                        child_id, Action(type=ActionType.STOP)
                    )
                    await self._aggregate_task
        else:
            await self._all_work_done_event.wait()
            await self._aggregate_task

    async def stop(self) -> None:
        """Cancel all task-monitor coroutines then delegate to the base stop."""
        if self._task_monitor_tasks:
            self._stop_task_monitor_event.set()
            self.logger.info(f"{self.node_id}: Signaled task monitor tasks to stop")
            for task in self._task_monitor_tasks:
                task.cancel()
            await asyncio.gather(*self._task_monitor_tasks, return_exceptions=True)
            self.logger.info(f"{self.node_id}: All task monitor tasks stopped")

        if (
            self._monitor_task_requests_task
            and not self._monitor_task_requests_task.done()
        ):
            self._monitor_task_requests_task.cancel()
            try:
                await self._monitor_task_requests_task
            except asyncio.CancelledError:
                pass

        return await super().stop()
