import asyncio
from asyncio import Future as AsyncFuture
from typing import Callable, Dict, List, Optional, Set

from ensemble_launcher.comm.messages import (
    Stop,
    StopType,
    TaskRequest,
    TaskUpdate,
)
from ensemble_launcher.scheduler.child_state import ChildState
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

        self._all_work_done_event = asyncio.Event()
        self._relaunch_attempts = 0
        self._max_relaunch_attempts = 2

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
        """Serve task requests from a single child until cancelled.

        Blocks on each TaskRequest, assigns tasks from the unassigned pool via
        the scheduler, and replies with a TaskUpdate or a Stop(TERMINATE) if the
        pool is empty.
        """
        self.logger.info(
            f"{self.node_id}: Started monitoring task requests from child {child_id}"
        )
        failures = 0

        while True:
            try:
                task_request: TaskRequest = await self._comm.recv_message_from_child(
                    TaskRequest, child_id=child_id, block=True
                )

                if task_request is not None:
                    failures = 0
                    self.logger.info(
                        f"{self.node_id}: Received task request from {child_id} for {task_request.ntasks} tasks"
                    )

                    child_assignments, _, _ = self._scheduler.assign_task_ids(
                        self._scheduler.unassigned_task_ids,
                        ntask=task_request.ntasks,
                        child_ids=[child_id],
                    )

                    assigned_task_ids = child_assignments.get(child_id, [])
                    if not assigned_task_ids:
                        self.logger.info(
                            f"{self.node_id}: No tasks to assign, sending stop to {child_id}"
                        )
                        stop_msg = Stop(sender=self.node_id, type=StopType.TERMINATE)
                        await self._comm.send_message_to_child(child_id, stop_msg)
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

    # ------------------------------------------------------------------
    # Fault tolerance overrides
    # ------------------------------------------------------------------

    async def _recover_dead_child(self, child_id: str) -> List[str]:
        """Recover a dead child: delegate to base (handles teardown and re-init of monitors)."""
        return await super()._recover_dead_child(child_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _lazy_init(self) -> None:
        """Extend base lazy init: log work-stealing pool size at the leaf level."""
        await super()._lazy_init()

        self.logger.info(f"I am workstealing master")

        if self.level + 1 == self._config.policy_config.nlevels:
            self.logger.info(
                f"{self.node_id}: Work-stealing mode — {len(self._scheduler.unassigned_task_ids)} tasks in unassigned pool"
            )

    def _mark_and_launch(self, child_ids: List[str], future) -> None:
        """Apply state transitions and relaunch children for remaining tasks.

        Called via call_soon_threadsafe from done callbacks. Determines
        SUCCESS/FAILED from the future exit code for RUNNING children.
        RECOVERING children are handled by _recover_dead_child which awaits
        the future directly — skip them here to avoid double-transition.
        If all children reach terminal state but unassigned tasks remain,
        relaunches up to _max_relaunch_attempts times.
        """
        for child_id in child_ids:
            state = self._scheduler.get_child_state(child_id)
            if state == ChildState.RUNNING:
                try:
                    crashed = future.cancelled() or future.exception() is not None
                except Exception:
                    crashed = True
                if crashed:
                    self._scheduler.mark_child_failed(child_id)
                else:
                    self._scheduler.mark_child_success(child_id)
            else:
                # RECOVERING (teardown handles via wrap_future), terminal, or
                # unregistered — no-op.
                self.logger.debug(
                    f"_mark_and_launch: no-op for {child_id} in state {state}"
                )
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

    def _create_done_callback(
        self, child_ids: List[str]
    ) -> Callable[[AsyncFuture], None]:
        def _done_callback(future: AsyncFuture):
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(
                    self._mark_and_launch, child_ids, future
                )
            else:
                self.logger.warning("No event loop stored, can't mark child done!")

        return _done_callback

    async def _wait_for_finish(self) -> None:
        """Wait for all work to complete, accounting for dynamically stolen tasks.

        Races conditions with asyncio.FIRST_COMPLETED:
        1. Work done — _all_work_done_event (set once all tasks assigned and collected).
        2. Parent dead locally — consecutive send failures exceeded threshold.
        3. SIGTERM on this process.
        4. Stop(TERMINATE/KILL) received from parent (non-root only).

        On parent-dead exit, forwards Stop(KILL) to all children. On any other
        non-work-done exit, forwards Stop(TERMINATE) cascading the shutdown down
        the subtree. In both cases skips sending results/status to the dead parent.
        """
        import sys

        stop_tasks: Dict[str, asyncio.Task] = {}
        received_stop_type: List[Optional[StopType]] = [None]

        stop_tasks["work_done"] = asyncio.create_task(self._all_work_done_event.wait())
        if self._comm.parent_dead_event is not None:
            stop_tasks["parent_dead"] = asyncio.create_task(self._comm.parent_dead_event.wait())
        stop_tasks["stop_signal"] = asyncio.create_task(
            self._stop_signal_received.wait()
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

        # Propagate stop to children if work didn't finish naturally.
        # Parent dead → KILL children. Received stop from parent → mirror its type.
        # Any other reason (SIGTERM etc.) → TERMINATE cleanly.
        if not self._all_work_done_event.is_set():
            if self._parent_dead_event.is_set():
                stop_type = StopType.KILL
            elif received_stop_type[0] is not None:
                stop_type = received_stop_type[0]
            else:
                stop_type = StopType.TERMINATE
            self.logger.info(
                f"{self.node_id}: Propagating {stop_type.value} to children"
            )
            for child_id in list(self.children.keys()):
                try:
                    await self._comm.send_message_to_child(
                        child_id, Stop(sender=self.node_id, type=stop_type)
                    )
                except Exception:
                    pass

        # If we KILLed children they won't send results — cancel all result tasks.
        if not self._all_work_done_event.is_set() and stop_type == StopType.KILL:
            for t in list(self._child_result_batch_task.values()):
                t.cancel()
            await asyncio.gather(
                *self._child_result_batch_task.values(),
                return_exceptions=True,
            )
            self._aggregate_task.cancel()
            try:
                await self._aggregate_task
            except (asyncio.CancelledError, Exception):
                pass
        else:
            await self._aggregate_task

        # Force-exit after propagating if we received a KILL from our parent.
        if received_stop_type[0] == StopType.KILL:
            sys.exit(1)

    async def stop(self) -> None:
        """Delegate to AsyncMaster.stop() — per-child monitors are cancelled by _teardown_child."""
        return await super().stop()
