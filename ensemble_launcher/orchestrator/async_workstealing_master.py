import asyncio
from typing import Dict, List, Optional, Set, Union

from ensemble_launcher.comm.messages import (
    IResultBatch,
    Result,
    Stop,
    StopType,
    TaskRequest,
    TaskUpdate,
)
from ensemble_launcher.ensemble import Task
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

        self._completed_task_ids: Set[str] = set()

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

        if not partial and not self._config.overload_orchestrator_core:
            self._apply_resource_headroom()

        target_ids = set(self._scheduler.child_assignments.keys()) - existing_ids
        return self._instantiate_children(include_tasks, target_ids)

    def _build_init_task_update(self, child_id: str) -> None:
        """No initial task update is sent in work-stealing mode; tasks arrive on demand."""
        return None

    # --------------------------------------------------------------------------
    #                               Task Routing
    # --------------------------------------------------------------------------

    def _route_tasks(self, tasks: List[Task]) -> List[Optional[str]]:
        """In workstealing mode tasks are only added to the scheduler queue."""
        for task in tasks:
            self._scheduler.add_task(task)

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
                    # Drain any additional queued requests and keep only the latest.
                    while True:
                        next_request = await self._comm.recv_message_from_child(
                            TaskRequest, child_id=child_id, block=False
                        )
                        if next_request is None:
                            break
                        task_request = next_request
                    self.logger.debug(
                        f"{self.node_id}: Received task request from {child_id} for {task_request.ntasks} tasks"
                    )

                    child_assignments, _, _ = self._scheduler.assign_task_ids(
                        self._scheduler.unassigned_task_ids,
                        ntask=task_request.ntasks,
                        child_ids=[child_id],
                    )

                    assigned_task_ids = child_assignments.get(child_id, [])
                    if not assigned_task_ids:
                        if not self._config.cluster:
                            self.logger.warning(
                                f"{self.node_id}: No tasks to assign, sending stop to {child_id}"
                            )
                            stop_msg = Stop(
                                sender=self.node_id, type=StopType.TERMINATE
                            )
                            await self._comm.send_message_to_child(child_id, stop_msg)
                        else:
                            self.logger.debug(f"{self.node_id}: No tasks to assign.")
                    else:
                        available_tasks = [
                            self._scheduler.tasks[tid] for tid in assigned_task_ids
                        ]
                        task_update = TaskUpdate(
                            sender=self.node_id, added_tasks=available_tasks
                        )
                        await self._comm.send_message_to_child(child_id, task_update)
                        self.logger.debug(
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

    async def _forward_result(self, result: Union[Result, IResultBatch]) -> None:
        """Forward result upstream and track completed task IDs.

        In non-cluster mode, once the count of unique completed task IDs reaches
        the total task count, sends Stop(TERMINATE) to all children.  The base
        _wait_for_finish then observes _all_children_done_event (set when every
        child reaches SUCCESS/FAILED after processing Stop) and handles result
        aggregation normally.
        """
        await super()._forward_result(result)
        if self._config.cluster:
            return
        items = result.data if isinstance(result, IResultBatch) else [result]
        for item in items:
            self._completed_task_ids.add(item.task_id)
        await self._check_all_tasks_done()

    async def _check_all_tasks_done(self) -> None:
        """Send Stop(TERMINATE) to all children once every task has a result."""
        if len(self._completed_task_ids) < len(self._scheduler.tasks):
            return
        self.logger.info(
            f"{self.node_id}: All {len(self._scheduler.tasks)} tasks completed, "
            f"sending Stop to children"
        )
        for child_id in list(self.children.keys()):
            try:
                await self._comm.send_message_to_child(
                    child_id, Stop(sender=self.node_id, type=StopType.TERMINATE)
                )
            except Exception as e:
                self.logger.warning(f"Failed to send Stop to {child_id}: {e}")

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
