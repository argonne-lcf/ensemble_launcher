import asyncio
import copy
import heapq
import os
from asyncio import Queue
from collections import Counter, deque
from logging import Logger
from typing import (
    TYPE_CHECKING,
    Deque,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

from ensemble_launcher.config import LauncherConfig, PolicyConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.profiling import EventRegistry, get_registry

from .child_state import ChildState
from .policy import ChildrenPolicy, Policy, policy_registry
from .resource import (
    AsyncLocalClusterResource,
    JobResource,
    NodeResourceCount,
)
from .scheduler import Scheduler
from .state import ChildrenAssignment, SchedulerState

if TYPE_CHECKING:
    from ensemble_launcher.comm.messages import Result, Status

# self.logger = logging.getself.logger(__name__)

# Task policy that is special-cased by name: unscored, deque-backed pending queue.
FIFO_POLICY_NAME = "fifo_policy"


class AsyncScheduler(Scheduler):
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """

    def __init__(self, logger: Logger, cluster_resource: AsyncLocalClusterResource):
        super().__init__(logger=logger, cluster_resource=cluster_resource)


class AsyncChildrenScheduler(AsyncScheduler):
    """Scheduler that manages a pool of child workers: resource allocation, task
    distribution, lifecycle tracking, and status aggregation."""

    def __init__(
        self,
        logger: Logger,
        nodes: JobResource,
        config: LauncherConfig,
        tasks: Optional[Dict[str, Task]] = None,
        node_id: str = None,
        level: Optional[int] = None,
    ) -> None:
        """Initialise the worker scheduler.

        Args:
            logger: Logger instance for this scheduler.
            nodes: Available cluster resources.
            config: Launcher configuration (policy name, nchildren, etc.).
            tasks: Initial task dict; all tasks start in the unassigned pool.
            node_id: ID of the owning master node.
            level: Optional hierarchy level (0 = root master).
        """
        cluster = AsyncLocalClusterResource(logger.getChild("cluster"), nodes)
        super().__init__(logger, cluster)

        self._config = config
        # Initialize policy - uses the registered state instance
        self.policy: ChildrenPolicy = policy_registry.create_policy(
            self._config.children_scheduler_policy,
            policy_kwargs={
                "policy_config": self._config.policy_config,
                "node_id": node_id,
                "logger": logger.getChild("policy"),
            },
        )

        # Full task dict owned by the scheduler.
        self.tasks: Dict[str, Task] = tasks if tasks is not None else {}

        # Track worker assignments (resource allocation, keyed by child_id)
        self.workers: Dict[str, JobResource] = {}
        self._event_loop = asyncio.get_event_loop()
        self.cluster.set_event_loop(self._event_loop)

        # Child bookkeeping
        self._child_assignments: Dict[str, ChildrenAssignment] = {}
        self._children_status: Dict[str, "Status"] = {}  # child_id -> Status
        self._child_done_events: Dict[str, asyncio.Event] = {}  # child_id -> done event
        self._child_ready_events: Dict[
            str, asyncio.Event
        ] = {}  # child_id -> ready (running) event
        self._child_states: Dict[str, ChildState] = {}  # child_id -> ChildState
        self._all_children_done_event: asyncio.Event = asyncio.Event()
        self._child_to_tasks: Dict[
            str, List[str]
        ] = {}  # child_id -> dynamically assigned task_ids
        self._task_to_child: Dict[str, str] = {}  # task_id -> child_id
        self._task_to_client: Dict[str, str] = {}  # task_id -> client_id
        # Pool of task IDs not yet assigned to any child.
        # Seeded at init from tasks; assign_task_ids() drains it as tasks are placed.
        self._unassigned_tasks: Dict[str, None] = dict.fromkeys(self.tasks.keys())
        self._child_id_to_wid: Dict[str, int] = {}
        self._wid_to_child_id: Dict[int, str] = {}
        self._level: Optional[int] = level

    # ------------------------------------------------------------------
    # Child registration / reset
    # ------------------------------------------------------------------

    def register_child(self, child_id: str, assignment: Dict) -> None:
        """Register a child and initialize all per-child tracking state."""
        self._child_assignments[child_id] = assignment
        self._child_to_tasks.setdefault(child_id, [])
        self._child_done_events[child_id] = asyncio.Event()
        self._child_ready_events[child_id] = asyncio.Event()
        self._child_states[child_id] = ChildState.NOTREADY

    def reset_child_assignments(self) -> None:
        """Clear child bookkeeping. The unassigned task pool is preserved."""
        self._child_assignments = {}
        self._children_status = {}
        self._child_done_events = {}
        self._child_ready_events = {}
        self._child_states = {}
        self._all_children_done_event.clear()
        self._child_to_tasks = {}
        self._task_to_child = {}
        self._child_id_to_wid = {}
        self._wid_to_child_id = {}

    # ------------------------------------------------------------------
    # Assignment accessors
    # ------------------------------------------------------------------

    @property
    def child_assignments(self) -> Dict[str, ChildrenAssignment]:
        """Mapping of child_id to its ChildrenAssignment (resources + task_ids + wid)."""
        return self._child_assignments

    @property
    def children_names(self) -> List[str]:
        """Ordered list of all registered child IDs."""
        return list(self._child_assignments.keys())

    def get_child_assignment(self, child_id: str) -> ChildrenAssignment:
        """Return the ChildrenAssignment for the given child_id."""
        return self._child_assignments[child_id]

    # ------------------------------------------------------------------
    # Child lifecycle state machine
    # ------------------------------------------------------------------

    def _assert_transition(
        self, child_id: str, from_states: set, to_state: ChildState
    ) -> None:
        """Raise RuntimeError if the current state is not in from_states."""
        current = self._child_states.get(child_id)
        if current not in from_states:
            self.logger.error(f"Invalid transition {child_id}: {current} -> {to_state}")
            raise RuntimeError(
                f"Invalid child state transition for {child_id}: {current} -> {to_state}"
            )

    def _check_all_children_terminal(self) -> None:
        """Set _all_children_done_event when every registered child is terminal."""
        terminal = {ChildState.FAILED, ChildState.SUCCESS}
        if self._child_states and all(
            s in terminal for s in self._child_states.values()
        ):
            self._all_children_done_event.set()

    def get_child_state(self, child_id: str) -> Optional[ChildState]:
        """Return the current ChildState for child_id, or None if not registered."""
        return self._child_states.get(child_id)

    def mark_child_ready(self, child_id: str) -> None:
        """NOTREADY → READY: resources allocated, child ready to launch."""
        self._assert_transition(child_id, {ChildState.NOTREADY}, ChildState.READY)
        self._child_states[child_id] = ChildState.READY

    def mark_child_running(self, child_id: str) -> None:
        """READY/RECOVERING → RUNNING: subprocess submitted to executor."""
        self._assert_transition(
            child_id, {ChildState.READY, ChildState.RECOVERING}, ChildState.RUNNING
        )
        self._child_states[child_id] = ChildState.RUNNING
        self._child_done_events[child_id].clear()
        self._child_ready_events[child_id].set()
        self.set_child_tasks_status(child_id, TaskStatus.RUNNING)

    def mark_child_recovering(self, child_id: str) -> None:
        """RUNNING → RECOVERING: child timed out, recovery in progress.

        Resources are NOT freed here — they will be freed when the terminal
        state (FAILED/SUCCESS) is set by the result_batch path or when
        _teardown_child calls remove_child().
        The done event is not used for teardown (which awaits the future
        directly); it is only set by mark_child_success/failed.
        """
        self._assert_transition(child_id, {ChildState.RUNNING}, ChildState.RECOVERING)
        self._child_states[child_id] = ChildState.RECOVERING

    def mark_child_failed(self, child_id: str) -> None:
        """RUNNING/RECOVERING → FAILED: terminal failure; free resources."""
        self._assert_transition(
            child_id,
            {ChildState.RUNNING, ChildState.RECOVERING},
            ChildState.FAILED,
        )
        self._child_states[child_id] = ChildState.FAILED
        self.free(child_id)
        if child_id in self._child_done_events:
            self._child_done_events[child_id].set()
        self._check_all_children_terminal()

    def mark_child_success(self, child_id: str) -> None:
        """RUNNING/RECOVERING → SUCCESS: terminal success; free resources.

        If not all tasks associated with this child have reached a terminal
        status (SUCCESS or FAILED), the transition is skipped. This keeps
        the child in its current state so that ``wait_for_child`` does not
        resolve and the liveness monitor can restart the child.
        """
        if not self.are_child_tasks_terminal(child_id):
            self.logger.warning(
                f"Skipping SUCCESS transition for {child_id}: "
                "not all tasks are terminal"
            )
            return
        self._assert_transition(
            child_id,
            {ChildState.RUNNING, ChildState.RECOVERING},
            ChildState.SUCCESS,
        )
        self._child_states[child_id] = ChildState.SUCCESS
        self.free(child_id)
        if child_id in self._child_done_events:
            self._child_done_events[child_id].set()
        self._check_all_children_terminal()

    def is_child_recovering(self, child_id: str) -> bool:
        """Return True if child is in RECOVERING state."""
        return self._child_states.get(child_id) == ChildState.RECOVERING

    def is_child_failed(self, child_id: str) -> bool:
        """Return True if child is in terminal FAILED state."""
        return self._child_states.get(child_id) == ChildState.FAILED

    def is_child_terminal(self, child_id: str) -> bool:
        """Return True if child is in a terminal state (SUCCESS or FAILED)."""
        return self._child_states.get(child_id) in {
            ChildState.SUCCESS,
            ChildState.FAILED,
        }

    # Deprecated shims — keep until all callers are migrated
    def mark_child_dead(self, child_id: str) -> None:
        self.mark_child_failed(child_id)

    def mark_child_done(self, child_id: str) -> None:
        self.mark_child_success(child_id)

    def is_child_dead(self, child_id: str) -> bool:
        return self.is_child_failed(child_id)

    def is_child_done(self, child_id: str) -> bool:
        return self.is_child_terminal(child_id)

    async def wait_for_child_ready(self, child_id: str) -> None:
        """Await the ready event for the given child_id (set when marked running)."""
        await self._child_ready_events[child_id].wait()

    @property
    def all_children_done(self) -> bool:
        """True when every registered child is in a terminal state."""
        terminal = {ChildState.SUCCESS, ChildState.FAILED}
        return bool(self._child_states) and all(
            s in terminal for s in self._child_states.values()
        )

    # ------------------------------------------------------------------
    # Status bookkeeping
    # ------------------------------------------------------------------

    def set_child_status(self, child_id: str, status: "Status") -> None:
        """Store the most recent Status message received from a child."""
        self._children_status[child_id] = status

    def has_final_status(self, child_id: str) -> bool:
        """Return True if the child's last recorded status has tag='final'."""
        status = self._children_status.get(child_id)
        return status is not None and status.tag == "final"

    def aggregate_status(self) -> "Status":
        """Sum all children statuses into a single aggregated Status object."""
        from ensemble_launcher.comm.messages import Status as _Status

        return sum(self._children_status.values(), _Status())

    # ------------------------------------------------------------------
    # Done-event accessors
    # ------------------------------------------------------------------

    def get_done_event(self, child_id: str) -> asyncio.Event:
        """Return the asyncio.Event that is set when the given child finishes."""
        return self._child_done_events[child_id]

    def get_ready_event(self, child_id: str) -> asyncio.Event:
        """Return the asyncio.Event that is set when the child is marked running."""
        return self._child_ready_events[child_id]

    # ------------------------------------------------------------------
    # Dynamic task routing
    # ------------------------------------------------------------------

    def get_worker_task_assignments(self) -> Dict[str, Dict]:
        """Return child_id-keyed assignment dict for use by the routing policy."""
        return {
            child_id: {
                "job_resource": assignment["job_resource"],
                "task_ids": list(assignment["task_ids"])
                + self._child_to_tasks.get(child_id, []),
            }
            for child_id, assignment in self._child_assignments.items()
        }

    def get_task_to_child(self, task_id: str) -> Optional[str]:
        """Return the child id to which task is assigned"""
        return self._task_to_child.get(task_id, None)

    def get_child_task_ids(self, child_id: str) -> List[str]:
        """Return the task IDs assigned to a child."""
        return list(self._child_assignments.get(child_id, {}).get("task_ids", []))

    def get_all_child_task_ids(self, child_id: str) -> List[str]:
        """Return all task IDs for a child (both static assignments and dynamically routed)."""
        static = list(self._child_assignments.get(child_id, {}).get("task_ids", []))
        dynamic = list(self._child_to_tasks.get(child_id, []))
        return static + dynamic

    # ------------------------------------------------------------------
    # Task status tracking
    # ------------------------------------------------------------------

    def set_task_status(self, task_id: str, status: TaskStatus) -> None:
        """Set the status of a single task."""
        if task_id in self.tasks:
            self.tasks[task_id].status = status

    def set_tasks_status(self, task_ids: List[str], status: TaskStatus) -> None:
        """Set the status of multiple tasks."""
        for task_id in task_ids:
            self.set_task_status(task_id, status)

    def set_child_tasks_status(self, child_id: str, status: TaskStatus) -> None:
        """Set the status of all tasks associated with a child."""
        self.set_tasks_status(self.get_all_child_task_ids(child_id), status)

    def are_child_tasks_terminal(self, child_id: str) -> bool:
        """Return True if all tasks for a child are in a terminal state (SUCCESS or FAILED)."""
        terminal = {TaskStatus.SUCCESS, TaskStatus.FAILED}
        task_ids = self.get_all_child_task_ids(child_id)
        if not task_ids:
            return True
        return all(
            self.tasks[tid].status in terminal for tid in task_ids if tid in self.tasks
        )

    @property
    def unassigned_task_ids(self) -> Dict[str, None]:
        """Read-only view of the unassigned task pool (insertion-ordered)."""
        return dict(self._unassigned_tasks)

    def has_unassigned_tasks(self) -> bool:
        """Whether the unassigned pool holds anything, without copying it.

        :attr:`unassigned_task_ids` copies the pool to hand out a read-only view, which is
        the wrong thing to pay for merely to ask whether it is empty -- and a work-stealing
        master is asked exactly that on every task request from every child.
        """
        return bool(self._unassigned_tasks)

    def discard_unassigned(self, task_id: str) -> None:
        """Remove a task from the unassigned pool (e.g. after work-stealing dispatch)."""
        self._unassigned_tasks.pop(task_id, None)

    def add_task(self, task: Task, client_id: Optional[str] = None) -> None:
        """Add a task to the scheduler and mark it as unassigned."""
        self.tasks[task.task_id] = task
        self._unassigned_tasks[task.task_id] = None
        if client_id is not None:
            self._task_to_client[task.task_id] = client_id

    def get_task_client(self, task_id: str) -> Optional[str]:
        """Return the client_id that submitted this task, or None."""
        return self._task_to_client.get(task_id)

    def unassign_tasks(self, child_id: str, task_ids: List[str]) -> None:
        """Move tasks from a child's assignment back to the unassigned pool."""
        if child_id not in self._child_assignments:
            return
        child_task_list = self._child_assignments[child_id]["task_ids"]
        for tid in task_ids:
            if tid in child_task_list:
                child_task_list.remove(tid)
            self._unassigned_tasks[tid] = None

    def delete_task(self, task_id: str) -> None:
        """Remove a task from the scheduler entirely.

        Clears the task from self.tasks, the unassigned pool, and any child
        assignment that references it.
        """
        self.tasks.pop(task_id, None)
        self._unassigned_tasks.pop(task_id, None)
        self._task_to_client.pop(task_id, None)
        for assignment in self._child_assignments.values():
            task_ids: List[str] = assignment["task_ids"]
            if task_id in task_ids:
                task_ids.remove(task_id)

    def remove_child(self, child_id: str) -> None:
        """Remove a child from all bookkeeping and return its resources to the cluster.

        Any task_ids still assigned to this child are returned to the unassigned pool
        so they can be redistributed (e.g. on failure recovery).
        """
        if child_id not in self._child_assignments:
            return
        for task_id in self._child_assignments[child_id].get("task_ids", []):
            self._unassigned_tasks[task_id] = None
        del self._child_assignments[child_id]
        self._child_done_events.pop(child_id, None)
        self._child_ready_events.pop(child_id, None)
        self._child_states.pop(child_id, None)
        self._children_status.pop(child_id, None)
        self._child_to_tasks.pop(child_id, None)
        self.free(child_id)  # no-op if already freed by mark_child_success/failed

    def get_state(self, node_id: str) -> SchedulerState:
        """Snapshot current state for checkpointing.

        Captures child assignment bookkeeping (resources + task IDs) so the
        master can rebuild its tree after a restart.  Tasks that were
        in-flight (assigned to running children) are preserved in
        ``children_task_ids`` so they can be redistributed on recovery.
        """
        children_task_ids: Dict[str, List[str]] = {
            cid: list(asgn["task_ids"]) for cid, asgn in self._child_assignments.items()
        }
        children_resources: Dict[str, JobResource] = {
            cid: asgn["job_resource"] for cid, asgn in self._child_assignments.items()
        }
        return SchedulerState(
            node_id=node_id,
            level=self._level,
            nodes=self.cluster.nodes,
            children_task_ids=children_task_ids,
            children_resources=children_resources,
            child_id_to_wid=dict(self._child_id_to_wid),
            wid_to_child_id={wid: cid for wid, cid in self._wid_to_child_id.items()},
            task_to_child=self._task_to_child,
            task_to_client=dict(self._task_to_client),
            child_states={cid: s.name for cid, s in self._child_states.items()},
        )

    def set_state(self, state: SchedulerState) -> None:
        """Restore scheduler from a checkpointed SchedulerState.

        Repopulates child assignments, resource allocations, and wid mappings
        so that the master can re-instantiate child Node objects with the same
        task layout as before the crash, without re-running the scheduling policy.

        Must be called after scheduler creation but before _create_children().
        """
        self._level = state.level
        for child_id, task_ids in state.children_task_ids.items():
            resource = state.children_resources.get(child_id)
            wid = state.child_id_to_wid.get(child_id, len(self._child_assignments))
            assignment: ChildrenAssignment = {
                "job_resource": resource,
                "task_ids": list(task_ids),
                "wid": wid,
            }
            self.register_child(child_id, assignment)  # → NOTREADY
            if resource is not None:
                allocated, res = self.cluster.allocate(resource)
                if allocated:
                    self.workers[child_id] = res
                    self.mark_child_ready(child_id)  # NOTREADY → READY
            self._child_id_to_wid[child_id] = wid
            self._wid_to_child_id[wid] = child_id

        self._task_to_child.update(state.task_to_child)
        self._task_to_client.update(state.task_to_client)

        # All tasks that were assigned to children leave the unassigned pool
        assigned = {
            task_id
            for task_ids in state.children_task_ids.values()
            for task_id in task_ids
        }
        for task_id in assigned:
            self._unassigned_tasks.pop(task_id, None)

        self.logger.info(
            f"set_state: restored {len(state.children_task_ids)} children, "
            f"{len(assigned)} tasks assigned, {len(self._unassigned_tasks)} unassigned"
        )

    def assign(
        self,
        level: int,
        node_id: str,
        reset: bool = True,
        nodes: Optional[JobResource] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str]]:
        """Convenience wrapper: assign_resources then assign_task_ids.

        Returns
        - dict mapping child_id -> list of task_ids assigned in this call
        - dict mapping of task_id -> child id (str)
        - list of unassigned tasks in this call.
        """
        self.assign_resources(level, node_id, reset=reset, nodes=nodes)
        return self.assign_task_ids(self._unassigned_tasks)

    def assign_resources(
        self,
        level: int,
        node_id: str,
        reset: bool = True,
        nodes: Optional[JobResource] = None,
    ) -> None:
        """
        Call the policy to decide the worker layout and allocate cluster resources.

        Uses self.tasks to inform the policy. Registers each child with empty
        task_ids; call assign_task_ids() afterwards to distribute tasks from the
        unassigned pool.

        reset=True  — clears all child bookkeeping first (full re-assignment).
        reset=False — additive; preserves existing children and offsets new wids.
        nodes       — restrict allocation to these nodes (e.g. recovered nodes on retry).
        """
        if reset:
            self.reset_child_assignments()

        self._level = level
        child_suffix = ".w" if level + 1 == self._config.policy_config.nlevels else ".m"
        wid_offset = (
            max((a["wid"] for a in self._child_assignments.values()), default=-1) + 1
            if not reset
            else 0
        )

        available_nodes = nodes if nodes is not None else self.cluster.nodes
        children_resources = self.policy.get_children_resources(
            tasks=self.tasks, nodes=available_nodes, level=level
        )

        for orig_wid, job_resource in children_resources.items():
            wid = orig_wid + wid_offset
            allocated, resource = self.cluster.allocate(job_resource)
            if allocated:
                child_id = node_id + f"{child_suffix}{wid}"
                self.workers[child_id] = resource
                alloc: ChildrenAssignment = {
                    "job_resource": resource,
                    "task_ids": [],
                    "wid": wid,
                }
                self.register_child(child_id, alloc)  # → NOTREADY
                self.mark_child_ready(child_id)  # NOTREADY → READY
                self._child_id_to_wid[child_id] = wid
                self._wid_to_child_id[wid] = child_id
            else:
                self.logger.warning(f"Failed to allocate resources for worker {wid}")

    def assign_task_ids(
        self,
        task_ids: Union[Set[str], Dict[str, None]],
        ntask: Optional[int] = None,
        child_ids: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, List[str]], Dict[str, str], List[str]]:
        """
        Distribute the given task_ids to registered children via the policy.

        Looks up each task from self.tasks, runs get_children_tasks, updates each
        child's task_ids list, and removes successfully placed tasks from the
        unassigned pool.

        Returns a Tuple or
        - dict mapping child_id -> list of task_ids assigned in this call
        - dict mapping of task_id -> child id (str)
        - list of unassigned tasks in this call.
        """
        if not self._child_assignments or not task_ids:
            return {}, {}, []

        # Restrict to requested child_ids if provided.
        target_assignments = (
            {
                cid: self._child_assignments[cid]
                for cid in child_ids
                if cid in self._child_assignments
            }
            if child_ids is not None
            else self._child_assignments
        )
        # Build wid-keyed children_resources using the pre-built maps from assign_resources.
        children_resources: Dict[int, JobResource] = {
            self._child_id_to_wid[cid]: assignment["job_resource"]
            for cid, assignment in target_assignments.items()
            if cid in self._child_id_to_wid
        }

        # Convert child_assignments and child_status to wid-keyed for the policy.
        wid_assignments = {
            self._child_id_to_wid[cid]: assignment
            for cid, assignment in target_assignments.items()
            if cid in self._child_id_to_wid
        }
        wid_status = {
            self._child_id_to_wid[cid]: status
            for cid, status in self._children_status.items()
            if cid in self._child_id_to_wid
        }

        task_objs = {tid: self.tasks[tid] for tid in task_ids if tid in self.tasks}
        wid_to_task_id_map, task_id_to_wid_map, removed_tasks = (
            self.policy.get_children_tasks(
                tasks=task_objs,
                children_resources=children_resources,
                ntask=ntask,
                child_assignments=wid_assignments,
                child_status=wid_status,
                level=self._level,
            )
        )

        if removed_tasks:
            self.logger.debug(f"Policy could not place {len(removed_tasks)} tasks")

        child_assignments: Dict[str, List[str]] = {}
        for wid, assigned_ids in wid_to_task_id_map.items():
            child_id = self._wid_to_child_id[wid]
            self._child_assignments[child_id]["task_ids"].extend(assigned_ids)
            for tid in assigned_ids:
                self._unassigned_tasks.pop(tid, None)
            child_assignments[child_id] = assigned_ids

        task_to_child: Dict[str, str] = {}
        for task_id, wid in task_id_to_wid_map.items():
            task_to_child[task_id] = self._wid_to_child_id[wid]

        self._task_to_child.update(task_to_child)

        return child_assignments, task_to_child, removed_tasks

    def free(self, child_id: str) -> bool:
        """Deallocate cluster resources for a child. No-op if already freed."""
        if child_id in self.workers:
            result = self.cluster.deallocate(self.workers[child_id])
            if result:
                del self.workers[child_id]
            return result
        return False


PendingEntry = Tuple[float, int, str]  # (priority, seq, task_id)


class _PendingTasksBase:
    """Membership + async notification shared by the pending-task containers.

    Subclasses own the ordering structure and implement ``push`` / ``pop`` /
    ``requeue`` / ``remove`` / ``remove_many`` / ``rescore`` / ``sorted_items``
    / ``clear``. The scheduler's dispatch pass is ``pop()`` until something does
    not fit, then ``requeue(skipped)`` -- per-cycle cost proportional to the
    tasks examined, never to the queue length.

    ``_tasks_available`` must never be replaced: ``_monitor_resources`` parks on
    that exact Event object, and swapping it out silently deadlocks dispatch.
    """

    def __init__(self) -> None:
        self._task_ids: Set[str] = set()
        self._seq: int = 0
        self._tasks_available: asyncio.Event = asyncio.Event()

    def _after_shrink(self) -> None:
        if not self._task_ids:
            self._tasks_available.clear()

    def __contains__(self, task_id: str) -> bool:
        return task_id in self._task_ids

    def __len__(self) -> int:
        return len(self._task_ids)

    def empty(self) -> bool:
        return len(self._task_ids) == 0

    def task_ids(self) -> Set[str]:
        return set(self._task_ids)

    async def wait_for_tasks(self) -> None:
        """Block until at least one task is pending."""
        await self._tasks_available.wait()


class PendingTaskQueue(_PendingTasksBase):
    """FIFO collection of pending tasks, backed by a deque.

    Used for ``fifo_policy``: tasks are never scored, so ``push`` ignores its
    priority argument and every operation on the hot path is O(1). Entries keep
    the ``(priority, seq, task_id)`` shape of :class:`PendingTaskHeap` so the two
    are interchangeable.
    """

    def __init__(self) -> None:
        super().__init__()
        self._queue: Deque[PendingEntry] = deque()

    def push(self, priority: float, task_id: str) -> None:
        self._queue.append((0.0, self._seq, task_id))
        self._seq += 1
        self._task_ids.add(task_id)
        self._tasks_available.set()

    def pop(self) -> Optional[PendingEntry]:
        """Remove and return the oldest entry, or ``None`` if empty."""
        if not self._queue:
            return None
        entry = self._queue.popleft()
        self._task_ids.discard(entry[2])
        self._after_shrink()
        return entry

    def requeue(self, entries: List[PendingEntry]) -> None:
        """Put popped-but-not-dispatched entries back at the front, in order."""
        if not entries:
            return
        self._queue.extendleft(reversed(entries))
        self._task_ids.update(tid for _p, _s, tid in entries)
        self._tasks_available.set()

    def remove(self, task_id: str) -> bool:
        if task_id not in self._task_ids:
            return False
        self._task_ids.discard(task_id)
        self._queue = deque(e for e in self._queue if e[2] != task_id)
        self._after_shrink()
        return True

    def remove_many(self, task_ids: Set[str]) -> None:
        self._task_ids -= task_ids
        self._queue = deque(e for e in self._queue if e[2] in self._task_ids)
        self._after_shrink()

    def rescore(self, scores: Dict[str, float]) -> None:
        """FIFO has no priorities: only drop ids absent from *scores*, keep order."""
        self._queue = deque(e for e in self._queue if e[2] in scores)
        self._task_ids = {tid for _p, _s, tid in self._queue}
        self._after_shrink()

    def sorted_items(self) -> List[PendingEntry]:
        """All entries in dispatch order (a copy)."""
        return list(self._queue)

    def clear(self) -> None:
        self._queue.clear()
        self._task_ids.clear()
        self._seq = 0
        self._tasks_available.clear()


class PendingTaskHeap(_PendingTasksBase):
    """Priority-ordered collection of pending tasks with async notification.

    Lower priority value dispatches first; ties break on insertion order.
    """

    def __init__(self) -> None:
        super().__init__()
        self._heap: List[PendingEntry] = []

    def push(self, priority: float, task_id: str) -> None:
        heapq.heappush(self._heap, (priority, self._seq, task_id))
        self._seq += 1
        self._task_ids.add(task_id)
        self._tasks_available.set()

    def pop(self) -> Optional[PendingEntry]:
        """Remove and return the highest-priority entry, or ``None`` if empty."""
        if not self._heap:
            return None
        entry = heapq.heappop(self._heap)
        self._task_ids.discard(entry[2])
        self._after_shrink()
        return entry

    def requeue(self, entries: List[PendingEntry]) -> None:
        """Push popped-but-not-dispatched entries back with their original
        ``(priority, seq)``, so their place -- including the FIFO tiebreak -- is
        unchanged."""
        if not entries:
            return
        for entry in entries:
            heapq.heappush(self._heap, entry)
            self._task_ids.add(entry[2])
        self._tasks_available.set()

    def remove(self, task_id: str) -> bool:
        if task_id not in self._task_ids:
            return False
        self._task_ids.discard(task_id)
        self._heap = [(p, s, tid) for p, s, tid in self._heap if tid != task_id]
        heapq.heapify(self._heap)
        self._after_shrink()
        return True

    def remove_many(self, task_ids: Set[str]) -> None:
        self._task_ids -= task_ids
        self._heap = [(p, s, tid) for p, s, tid in self._heap if tid in self._task_ids]
        heapq.heapify(self._heap)
        self._after_shrink()

    def rescore(self, scores: Dict[str, float]) -> None:
        """Re-prioritize every entry in place. Ids absent from *scores* are dropped.

        Each entry keeps its original sequence number, so the FIFO tiebreak among
        equal priorities survives -- which also means the iteration order here is
        irrelevant and a plain dict is a safe argument. ``clear()`` + ``push()``
        would not have either property: it resets ``_seq`` to 0 and renumbers in
        whatever order it is handed.

        ``_tasks_available`` is deliberately not ``set()``: membership only ever
        shrinks here, so an already-set event must stay set and a heap that empties
        out must clear it.
        """
        self._heap = [
            (scores[task_id], seq, task_id)
            for _p, seq, task_id in self._heap
            if task_id in scores
        ]
        heapq.heapify(self._heap)
        self._task_ids = {task_id for _p, _s, task_id in self._heap}
        self._after_shrink()

    def sorted_items(self) -> List[PendingEntry]:
        """All entries in dispatch order (a sorted copy; the heap is untouched).

        O(n log n) -- for introspection and tests only. The dispatch loop uses
        ``pop`` / ``requeue``.
        """
        return sorted(self._heap)

    def clear(self) -> None:
        self._heap.clear()
        self._task_ids.clear()
        self._seq = 0
        self._tasks_available.clear()


class AsyncTaskScheduler(AsyncScheduler):
    """Task-level scheduler used by workers: allocates cluster resources per task
    and exposes a ready_tasks queue consumed by the worker's execution loop."""

    def __init__(
        self,
        logger: Logger,
        tasks: Dict[str, Task],
        nodes: JobResource,
        policy: Union[str, Policy] = FIFO_POLICY_NAME,
        policy_config: PolicyConfig = PolicyConfig(),
    ) -> None:
        """Initialise the task scheduler.

        Args:
            logger: Logger instance.
            tasks: Initial task dict to schedule.
            nodes: Available cluster resources for this worker.
            policy: Policy name or instance used to score/prioritise tasks.
                ``"fifo_policy"`` (the default) is special-cased: tasks go into
                a deque and are never scored. Any other policy scores each task
                into a priority heap.
            policy_config: Configuration passed to the policy constructor
                when *policy* is given as a string.
        """
        cluster = AsyncLocalClusterResource(logger.getChild("cluster"), nodes)
        super().__init__(logger, cluster)
        self.tasks: Dict[str, Task] = tasks
        if isinstance(policy, str):
            self.scheduler_policy: Policy = policy_registry.create_policy(
                policy,
                policy_kwargs={
                    "policy_config": policy_config,
                    "logger": logger.getChild("policy"),
                },
            )
        else:
            self.scheduler_policy: Policy = policy
        self._strict_priority: bool = policy_config.strict_priority
        self.ready_tasks: Queue[Tuple[str, JobResource]] = Queue()
        self._running_tasks: Dict[str, JobResource] = {}
        self._done_tasks: Counter[str] = Counter()
        self._failed_tasks: Set[str] = set()
        self._successful_tasks: Set[str] = set()
        self._fifo: bool = self._policy_name(policy) == FIFO_POLICY_NAME
        self._pending_tasks: Union[PendingTaskQueue, PendingTaskHeap] = (
            PendingTaskQueue() if self._fifo else PendingTaskHeap()
        )
        # Running per-dimension minimum (nnodes, ppn, ngpus) over enqueued tasks.
        # Only lowered on enqueue, reset when the queue empties -- so it may be
        # stale-low (a spurious wakeup) but never too high (which would stall).
        self._min_req: Optional[Tuple[int, int, float]] = None
        self._seed_pending(self.tasks.keys())
        self.logger.debug(f"Pending tasks: {len(self._pending_tasks)}")

        self._task_to_client: Dict[str, str] = {}  # task_id -> client_id

        self._stop_monitoring = asyncio.Event()
        self._all_tasks_done = asyncio.Event()
        self._consecutive_failed_allocations = 0
        self._monitoring_task = None
        self._event_loop = None  # Will be set when monitoring starts

        self._event_registry: Optional[EventRegistry] = None
        if os.environ.get("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry = get_registry()

    @property
    def policy(self) -> Policy:
        """Alias for ``scheduler_policy``, matching ``AsyncChildrenScheduler.policy``.

        Lets callers that hold either scheduler (notably ``PolicyEndpoint``) reach the
        policy without branching on the scheduler type.
        """
        return self.scheduler_policy

    @staticmethod
    def _policy_name(policy: Union[str, Policy]) -> Optional[str]:
        """Registered name of *policy*.

        Instances resolve by exact class, so a subclass of a built-in policy
        (which may override ``get_score``) never inherits its fast path.
        """
        if isinstance(policy, str):
            return policy
        for name, cls in policy_registry.available_policies.items():
            if type(policy) is cls:
                return name
        return None

    @staticmethod
    def _priority_from_score(score: float) -> float:
        """Convert a policy score (higher=better) to a heap priority (lower=better)."""
        return -score

    def _update_min_req(self, task: Task) -> None:
        shape = (task.nnodes, task.ppn, task.ngpus_per_process * task.ppn)
        if self._min_req is None:
            self._min_req = shape
        else:
            self._min_req = tuple(map(min, self._min_req, shape))

    def _enqueue(
        self, task: Task, sched_state: Optional[SchedulerState] = None
    ) -> None:
        """The single place a task enters the pending container.

        FIFO pushes without scoring. Any other policy is scored, and the
        O(total tasks) ``SchedulerState`` is only built when the policy declares
        ``needs_scheduler_state`` (callers enqueuing a batch pass one in).
        """
        if self._fifo:
            self._pending_tasks.push(0.0, task.task_id)
        else:
            if sched_state is None and self.scheduler_policy.needs_scheduler_state:
                sched_state = self._build_scheduler_state()
            score = self.scheduler_policy.get_score(task, scheduler_state=sched_state)
            self._pending_tasks.push(self._priority_from_score(score), task.task_id)
        self._update_min_req(task)

    def _seed_pending(self, task_ids: Iterable[str]) -> int:
        """Replace the pending container with fresh entries for *task_ids*.

        Used when the set of pending tasks itself is being established (construction,
        checkpoint restore). To re-score the tasks already queued without changing
        membership, use :meth:`reprioritize_pending` instead.
        """
        sched_state = None
        if not self._fifo and self.scheduler_policy.needs_scheduler_state:
            sched_state = self._build_scheduler_state()
        self._pending_tasks.clear()
        self._min_req = None
        for task_id in task_ids:
            if task_id not in self.tasks:
                continue
            self._enqueue(self.tasks[task_id], sched_state)
        return len(self._pending_tasks)

    def reprioritize_pending(self) -> int:
        """Re-score every pending task against the policy's current state.

        A task's priority is computed once, when it enters the pending heap, so the
        heap is a cache of scores: mutating policy state does nothing to work already
        queued. This invalidates that cache. Scope is the pending heap only -- running
        tasks are already dispatched and completed ones are untouched.

        Two phases on purpose. Every new priority is computed first, where a raising
        ``get_score`` leaves the heap untouched; only then is the result published in a
        single call that cannot fail. Callers (``PolicyEndpoint.apply``) rely on this to
        roll a rejected state change back cleanly.

        Returns:
            Number of tasks re-scored.
        """
        if self._fifo:
            # Nothing to score: FIFO order is the submission order.
            return len(self._pending_tasks)
        sched_state = (
            self._build_scheduler_state()
            if self.scheduler_policy.needs_scheduler_state
            else None
        )
        scores = {
            task_id: self._priority_from_score(
                self.scheduler_policy.get_score(
                    self.tasks[task_id], scheduler_state=sched_state
                )
            )
            for task_id in self._pending_tasks.task_ids()
            if task_id in self.tasks
        }
        self._pending_tasks.rescore(scores)
        return len(scores)

    def _build_scheduler_state(self) -> SchedulerState:
        """Build a lightweight state snapshot for passing to policy methods."""
        successful = set(self._successful_tasks)
        failed = set(self._failed_tasks)
        running = set(self._running_tasks.keys())
        all_ids = set(self.tasks.keys())
        pending = all_ids - successful - failed - running

        return SchedulerState(
            node_id="",
            nodes=self.cluster.nodes,
            pending_tasks=pending,
            running_tasks=running,
            completed_tasks=successful,
            failed_tasks=failed,
        )

    def get_state(self, node_id: str) -> SchedulerState:
        """Snapshot current state for checkpointing.

        Running tasks are folded back into ``pending_tasks`` because their
        executor futures will not survive a restart; they must be re-queued.
        """
        successful = self.successful_tasks  # Set[str]
        failed = self.failed_tasks  # Set[str]
        running = set(self._running_tasks.keys())
        all_ids = set(self.tasks.keys())

        # Running tasks must be retried on recovery, so include them in pending.
        pending = (all_ids - successful - failed) | running

        return SchedulerState(
            node_id=node_id,
            nodes=self.cluster.nodes,
            pending_tasks=pending,
            running_tasks=set(),  # nothing is running post-recovery
            completed_tasks=successful,
            failed_tasks=failed,
        )

    def set_state(
        self,
        state: "SchedulerState",
        results: Optional[Dict[str, "Result"]] = None,
    ) -> None:
        """Restore scheduler from a checkpointed SchedulerState.

        Marks completed/failed tasks, rebuilds the priority queue with only
        truly-pending tasks, and optionally restores result data on Task objects
        so the final ResultBatch includes pre-completed work.

        Args:
            state: SchedulerState snapshot from a previous run.
            results: Optional dict of checkpointed Result objects keyed by task_id.
                     If provided, task.result and task.exception are restored for
                     completed/failed tasks (needed for final ResultBatch).
        """
        # 1. Populate internal status sets
        self._successful_tasks = set(state.completed_tasks)
        self._failed_tasks = set(state.failed_tasks)

        # 2. Update Task object statuses and optionally restore result data
        for task_id in state.completed_tasks:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.SUCCESS
                if results and task_id in results:
                    self.tasks[task_id].result = results[task_id].data
                    self.tasks[task_id].exception = results[task_id].exception
        for task_id in state.failed_tasks:
            if task_id in self.tasks:
                self.tasks[task_id].status = TaskStatus.FAILED
                if results and task_id in results:
                    self.tasks[task_id].result = results[task_id].data
                    self.tasks[task_id].exception = results[task_id].exception

        # 3. Rebuild pending heap with only pending tasks
        pending = set(self.tasks.keys()) - self._successful_tasks - self._failed_tasks
        self._seed_pending(pending)

        restored = len(state.completed_tasks) + len(state.failed_tasks)
        self.logger.info(
            f"set_state: {restored} tasks restored from checkpoint, "
            f"{len(pending)} tasks pending execution"
        )

        # Signal completion immediately if no tasks remain
        self._check_all_tasks_done()

    def _find_min_resource_req(self) -> Optional[JobResource]:
        """Per-dimension minimum resource requirement among pending tasks, in O(1).

        Read from the running minimum kept by :meth:`_enqueue`. After small tasks
        have been dispatched it can be lower than the true minimum of what is
        left; that only causes an early wakeup, never a missed one.
        """
        if self._pending_tasks.empty():
            self._min_req = None
        if self._min_req is None:
            return None

        min_nnodes, min_ppn, min_ngpus = self._min_req
        return JobResource(
            resources=[
                NodeResourceCount(ncpus=min_ppn, ngpus=min_ngpus)
                for _ in range(min_nnodes)
            ]
        )

    def _allocate_pending(self) -> int:
        """One dispatch pass: pop pending tasks in order and allocate what fits.

        Tasks that don't fit are set aside and requeued at the end with their
        original ``(priority, seq)``, so they keep their place. Strict priority
        stops at the first task that doesn't fit; otherwise smaller tasks behind
        it may backfill until the cluster reports no free resources.

        Cost is O(k log n) for k tasks examined (O(k) for FIFO) -- never a sort
        or scan of the whole queue. Deliberately synchronous (``put_nowait`` on
        the unbounded ready queue), so no ``add_task`` / ``delete_task`` / policy
        rescore can interleave with a half-finished pass.

        Returns:
            Number of tasks moved to ``ready_tasks``.
        """
        allocated = 0
        skipped: List[PendingEntry] = []
        # Exact minimum over the skipped tasks; becomes the new running minimum
        # when the pass saw everything that is still pending (see below).
        skipped_min: Optional[Tuple[int, int, float]] = None
        exact = False
        try:
            while not self._stop_monitoring.is_set():
                entry = self._pending_tasks.pop()
                if entry is None:
                    exact = True  # every remaining task was examined
                    break
                task_id = entry[2]
                task = self.tasks.get(task_id)
                if task is None:
                    continue  # stale entry -> dropped

                req = task.get_resource_requirements()
                ok, resource = self.cluster.allocate(req)
                if ok:
                    self._running_tasks[task_id] = resource
                    self.ready_tasks.put_nowait((task_id, resource))
                    allocated += 1
                    continue

                skipped.append(entry)
                shape = (task.nnodes, task.ppn, task.ngpus_per_process * task.ppn)
                skipped_min = (
                    shape if skipped_min is None else tuple(map(min, skipped_min, shape))
                )
                if self._strict_priority:
                    # Nothing can run until this head task fits, so its own
                    # requirement is the exact wake-up threshold.
                    exact = True
                    self.logger.debug(
                        f"Strict priority: blocking on task {task_id} until resources available. "
                        f"Resources requested: {req}. Free resources: {self.cluster.get_status()}"
                    )
                    break
                # Backfill relies on this event being accurate: a failed
                # allocate never clears it, so if it were set on a full cluster
                # the pass would test every pending task. deallocate() sets it
                # synchronously to keep it accurate (see its docstring).
                if not self.cluster._resource_available.is_set():
                    self.logger.debug("No more free resources available")
                    break
        finally:
            self._pending_tasks.requeue(skipped)
        if exact:
            # Tighten the running minimum (it only ever drifts low otherwise),
            # which stops deallocations from waking the loop for tasks that
            # still cannot fit.
            self._min_req = skipped_min
        if allocated:
            self.logger.debug(f"Dispatched {allocated} tasks")
        return allocated

    async def _monitor_resources(self) -> None:
        """Monitors free resources, allocates tasks from the pending heap, and
        moves them to the ready queue for execution."""
        self.logger.info("Starting resource monitor")

        while not self._stop_monitoring.is_set():
            try:
                # Wait until the cluster has enough free resources for at least one task
                min_req = self._find_min_resource_req()
                self.logger.debug(
                    f"Waiting for free resources with min requirement: {min_req}. "
                    f"Current free resources: {self.cluster.get_status()}"
                )
                await self._cluster_resource.wait_for_free(min_resources=min_req)

                if self._stop_monitoring.is_set():
                    break

                # Wait for tasks if none are pending
                if self._pending_tasks.empty():
                    self.logger.info("No tasks available, waiting for new tasks")
                    stop_task = asyncio.create_task(self._stop_monitoring.wait())
                    tasks_task = asyncio.create_task(
                        self._pending_tasks.wait_for_tasks()
                    )

                    done, pending_aws = await asyncio.wait(
                        [stop_task, tasks_task], return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in pending_aws:
                        t.cancel()
                        try:
                            await t
                        except asyncio.CancelledError:
                            pass

                    if stop_task in done:
                        self.logger.info("Stop signal received while waiting for tasks")
                        break

                allocated = self._allocate_pending()

                if not allocated:
                    self.logger.debug(
                        "No tasks allocated in this cycle. Clearing resource available event to wait for new resources."
                    )
                    self._cluster_resource.clear_resource_available()

            except asyncio.CancelledError:
                self.logger.info("Scheduler Monitor task cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in monitor loop: {e}", exc_info=True)
                await asyncio.sleep(0.1)

    def start_monitoring(self) -> asyncio.Task:
        """Start the monitoring task. Must be called from async context."""
        if self._monitoring_task is not None and not self._monitoring_task.done():
            self.logger.warning("Monitor task already running")
            return

        # Store the event loop for thread-safe event signaling
        self._event_loop = asyncio.get_event_loop()
        self.cluster.set_event_loop(self._event_loop)
        self._stop_monitoring.clear()
        self._all_tasks_done.clear()
        self._consecutive_failed_allocations = 0
        self._monitoring_task = asyncio.create_task(self._monitor_resources())

    async def stop_monitoring(self):
        """Stop the monitoring task gracefully."""
        self.logger.info("Stopping resource monitoring")
        self._stop_monitoring.set()

        # Wake up the monitor loop if it's blocked waiting for resources
        await self._cluster_resource.signal_resource_available()

        if self._monitoring_task and not self._monitoring_task.done():
            # Cancel immediately instead of waiting
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Resource monitoring stopped")

    def _check_all_tasks_done(self) -> None:
        """Check if all tasks are complete and signal the completion event.

        Thread-safe — called from executor callbacks via call_soon_threadsafe.
        """
        # Fast path: called on every task completion, and the exact check below is
        # O(total tasks). Anything still pending or running means not done.
        if self._running_tasks or not self._pending_tasks.empty():
            return
        remaining = set(self.tasks.keys()) - (
            self._successful_tasks | self._failed_tasks
        )
        self.logger.debug(f"Checking completion: {len(remaining)} tasks remaining")
        if not remaining:
            self.logger.info("All tasks completed")
            if self._event_loop is not None:
                self.logger.debug(
                    f"Setting _all_tasks_done event via stored loop {self._event_loop}"
                )
                self._event_loop.call_soon_threadsafe(self._all_tasks_done.set)
            else:
                self.logger.warning(
                    "No event loop stored, setting event directly (may not work!)"
                )
                self._all_tasks_done.set()

    async def wait_for_completion(self):
        """
        Wait for all tasks to complete.
        This replaces the while loop in the worker's run() method.
        """
        self.logger.debug("Waiting for all tasks to complete")
        await self._all_tasks_done.wait()
        self.logger.debug("Done waiting for task completion!")

    def add_task(self, task: Task, client_id: Optional[str] = None) -> bool:
        """Add a task to the priority queue for scheduling. Returns True on success."""
        try:
            if task.nnodes > len(self.cluster.nodes.nodes):
                raise ValueError(
                    f"Task {task.task_id} requires {task.nnodes} nodes, but only {len(self.cluster.nodes.nodes)} are available"
                )
            self.tasks[task.task_id] = task
            if client_id is not None:
                self._task_to_client[task.task_id] = client_id
            self._enqueue(task)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False

    def get_task_client(self, task_id: str) -> Optional[str]:
        """Return the client_id that submitted this task, or None."""
        return self._task_to_client.get(task_id)

    def delete_task(self, task: Task) -> bool:
        """Remove a task from all queues and free its resources. Returns True on success."""
        if task.task_id not in self.tasks:
            self.logger.warning(f"Unknown task: {task.task_id}")
            return False

        try:
            # Remove from tasks dict
            del self.tasks[task.task_id]

            # Remove from pending tasks
            self._pending_tasks.remove(task.task_id)
            if self._pending_tasks.empty():
                self._min_req = None

            # Remove from ready tasks queue if present
            temp_ready = []
            while not self.ready_tasks.empty():
                try:
                    tid, resource = self.ready_tasks.get_nowait()
                    if tid != task.task_id:
                        temp_ready.append((tid, resource))
                    else:
                        # Deallocate resource if task is in ready queue
                        self.cluster.deallocate(resource)
                except asyncio.QueueEmpty:
                    break

            # Put back all items except the deleted task
            for item in temp_ready:
                self.ready_tasks.put_nowait(item)

            # If running, free the resources
            if task.task_id in self._running_tasks:
                self.cluster.deallocate(self._running_tasks[task.task_id])

            # Remove from running and status sets (a pending task was never running)
            self._running_tasks.pop(task.task_id, None)

            # Remove all occurrences from done_tasks
            if task.task_id in self._done_tasks:
                del self._done_tasks[task.task_id]

            # remove from failed and succesful tasks
            self._failed_tasks.discard(task.task_id)
            self._successful_tasks.discard(task.task_id)

            return True
        except Exception as e:
            self.logger.warning(f"Failed to delete task {task.task_id}: {e}")
            return False

    def free(self, task_id: str, status: TaskStatus) -> None:
        """Deallocate resources for a completed task and record its final status."""
        if task_id in self.tasks:
            if task_id not in self._running_tasks:
                self.logger.error(f"{task_id} is not running")
                raise RuntimeError

            # deallocate
            self.cluster.deallocate(self._running_tasks[task_id])

            # delete from running tasks
            del self._running_tasks[task_id]

            # Add to done tasks
            self._done_tasks[task_id] += 1

            # add to failed/successful tasks
            if status == TaskStatus.FAILED:
                self._failed_tasks.add(task_id)
            elif status == TaskStatus.SUCCESS:
                self._successful_tasks.add(task_id)
                self._failed_tasks.discard(task_id)

            self.logger.debug(f"Freed {task_id}")

            # Notify the policy about the completed task. Building the state is
            # O(total tasks), so skip it entirely for policies without the hook.
            if (
                type(self.scheduler_policy).on_task_complete
                is not Policy.on_task_complete
            ):
                self.scheduler_policy.on_task_complete(
                    task=self.tasks[task_id],
                    status=status,
                    scheduler_state=self._build_scheduler_state(),
                )

        self._check_all_tasks_done()
        return None

    def get_task_assignment(self) -> Dict[str, JobResource]:
        """Return a snapshot of the currently running task_id → resource mapping."""
        return copy.deepcopy(self._running_tasks)

    @property
    def running_tasks(self) -> Set[str]:
        """Return IDs of currently running tasks."""
        return copy.deepcopy(self._running_tasks)

    @property
    def failed_tasks(self) -> Set[str]:
        """Return IDs of tasks that have failed."""
        return copy.deepcopy(self._failed_tasks)

    @property
    def done_tasks(self) -> List[str]:
        """Return IDs of tasks that are done. Can have duplicates"""
        return copy.deepcopy(self._done_tasks)

    @property
    def successful_tasks(self) -> Set[str]:
        """Return IDs of tasks that have completed successfully."""
        return copy.deepcopy(self._successful_tasks)

    @property
    def remaining_tasks(self) -> Set[str]:
        """Task IDs that have not yet succeeded or failed."""
        return set(self.tasks.keys()) - (self._successful_tasks | self._failed_tasks)

    @property
    def not_ready_tasks(self) -> PendingTaskHeap:
        return self._pending_tasks
