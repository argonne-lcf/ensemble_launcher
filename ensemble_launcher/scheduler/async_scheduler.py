import asyncio
import copy
import os
from asyncio import PriorityQueue, Queue
from collections import Counter
from logging import Logger
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple, Union

from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.profiling import EventRegistry, get_registry

from .policy import ChildrenPolicy, Policy, policy_registry
from .resource import (
    AsyncLocalClusterResource,
    JobResource,
    NodeResourceCount,
)
from .scheduler import Scheduler
from .state import SchedulerState, WorkerAssignment

if TYPE_CHECKING:
    from ensemble_launcher.comm.messages import Status

# self.logger = logging.getself.logger(__name__)


class AsyncScheduler(Scheduler):
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """

    def __init__(self, logger: Logger, cluster_resource: AsyncLocalClusterResource):
        super().__init__(logger=logger, cluster_resource=cluster_resource)


class AsyncWorkerScheduler(AsyncScheduler):
    """Scheduler that manages a pool of child workers: resource allocation, task
    distribution, lifecycle tracking, and status aggregation."""

    def __init__(
        self,
        logger: Logger,
        nodes: JobResource,
        config: LauncherConfig,
        tasks: Optional[Dict[str, Task]] = None,
    ) -> None:
        """Initialise the worker scheduler.

        Args:
            logger: Logger instance for this scheduler.
            nodes: Available cluster resources.
            config: Launcher configuration (policy name, nchildren, etc.).
            tasks: Initial task dict; all tasks start in the unassigned pool.
        """
        cluster = AsyncLocalClusterResource(logger.getChild("cluster"), nodes)
        super().__init__(logger, cluster)

        self._config = config
        # Initialize policy - uses the registered state instance
        self.policy: ChildrenPolicy = policy_registry.create_policy(
            self._config.children_scheduler_policy,
            policy_kwargs={
                "nchildren": self._config.nchildren,
                "nlevels": self._config.nlevels,
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
        self._child_assignments: Dict[str, WorkerAssignment] = {}
        self._children_status: Dict[str, "Status"] = {}  # child_id -> Status
        self._child_done_events: Dict[str, asyncio.Event] = {}  # child_id -> done event
        self._running_children: Set[str] = set()  # child_ids currently running
        self._child_to_tasks: Dict[
            str, List[str]
        ] = {}  # child_id -> dynamically assigned task_ids
        self._task_to_child: Dict[str, str] = {}  # task_id -> child_id
        # Pool of task IDs not yet assigned to any child.
        # Seeded at init from tasks; assign_task_ids() drains it as tasks are placed.
        self._unassigned_tasks: Set[str] = set(self.tasks.keys())
        self._child_id_to_wid: Dict[str, int] = {}
        self._wid_to_child_id: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Child registration / reset
    # ------------------------------------------------------------------

    def register_child(self, child_id: str, assignment: Dict) -> None:
        """Register a child and initialize all per-child tracking state."""
        self._child_assignments[child_id] = assignment
        self._child_to_tasks.setdefault(child_id, [])
        self._child_done_events[child_id] = asyncio.Event()

    def reset_child_assignments(self) -> None:
        """Clear child bookkeeping. The unassigned task pool is preserved."""
        self._child_assignments = {}
        self._children_status = {}
        self._child_done_events = {}
        self._running_children = set()
        self._child_to_tasks = {}
        self._task_to_child = {}
        self._child_id_to_wid = {}
        self._wid_to_child_id = {}

    # ------------------------------------------------------------------
    # Assignment accessors
    # ------------------------------------------------------------------

    @property
    def child_assignments(self) -> Dict[str, WorkerAssignment]:
        """Mapping of child_id to its WorkerAssignment (resources + task_ids + wid)."""
        return self._child_assignments

    @property
    def children_names(self) -> List[str]:
        """Ordered list of all registered child IDs."""
        return list(self._child_assignments.keys())

    def get_child_assignment(self, child_id: str) -> WorkerAssignment:
        """Return the WorkerAssignment for the given child_id."""
        return self._child_assignments[child_id]

    # ------------------------------------------------------------------
    # Running-children lifecycle
    # ------------------------------------------------------------------

    def mark_child_running(self, child_id: str) -> None:
        """Record that a child process has been submitted to the executor."""
        self._running_children.add(child_id)

    def mark_child_done(self, child_id: str) -> None:
        """Discard from running set, free cluster resources, and set done event."""
        self._running_children.discard(child_id)
        self.free(child_id)
        if child_id in self._child_done_events:
            self._child_done_events[child_id].set()

    async def wait_for_child(self, child_id: str) -> None:
        """Await the done event for the given child_id."""
        await self._child_done_events[child_id].wait()

    @property
    def all_children_done(self) -> bool:
        """True when every registered child has set its done event."""
        return all([event.is_set() for event in self._child_done_events.values()])

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

    def record_dynamic_assignment(self, task_id: str, child_id: str) -> None:
        """Record that task_id was dynamically routed to child_id."""
        self._child_to_tasks[child_id].append(task_id)
        self._task_to_child[task_id] = child_id

    def get_child_task_ids(self, child_id: str) -> List[str]:
        """Return the task IDs assigned to a child."""
        return list(self._child_assignments.get(child_id, {}).get("task_ids", []))

    @property
    def unassigned_task_ids(self) -> Set[str]:
        """Read-only view of the unassigned task pool."""
        return self._unassigned_tasks

    def discard_unassigned(self, task_id: str) -> None:
        """Remove a task from the unassigned pool (e.g. after work-stealing dispatch)."""
        self._unassigned_tasks.discard(task_id)

    def add_task(self, task: Task) -> None:
        """Add a task to the scheduler and mark it as unassigned."""
        self.tasks[task.task_id] = task
        self._unassigned_tasks.add(task.task_id)

    def delete_task(self, task_id: str) -> None:
        """Remove a task from the scheduler entirely.

        Clears the task from self.tasks, the unassigned pool, and any child
        assignment that references it.
        """
        self.tasks.pop(task_id, None)
        self._unassigned_tasks.discard(task_id)
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
            self._unassigned_tasks.add(task_id)
        del self._child_assignments[child_id]
        self._child_done_events.pop(child_id, None)
        self._running_children.discard(child_id)
        self._children_status.pop(child_id, None)
        self._child_to_tasks.pop(child_id, None)
        self.free(child_id)  # no-op if already freed by mark_child_done

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
            nodes=self.cluster.nodes,
            children_task_ids=children_task_ids,
            children_resources=children_resources,
        )

    def assign(
        self,
        level: int,
        node_id: str,
        reset: bool = True,
        nodes: Optional[JobResource] = None,
    ) -> Dict[str, List[str]]:
        """Convenience wrapper: assign_resources then assign_task_ids.

        Returns child_id -> assigned task_ids mapping.
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

        child_suffix = ".w" if level + 1 == self._config.nlevels else ".m"
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
                alloc: WorkerAssignment = {
                    "job_resource": resource,
                    "task_ids": [],
                    "wid": wid,
                }
                self.register_child(child_id, alloc)
                self._child_id_to_wid[child_id] = wid
                self._wid_to_child_id[wid] = child_id
            else:
                self.logger.warning(f"Failed to allocate resources for worker {wid}")

    def assign_task_ids(
        self,
        task_ids: Set[str],
        ntask: Optional[int] = None,
        child_ids: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Distribute the given task_ids to registered children via the policy.

        Looks up each task from self.tasks, runs get_children_tasks, updates each
        child's task_ids list, and removes successfully placed tasks from the
        unassigned pool.

        Returns a dict mapping child_id -> list of task_ids assigned in this call.
        """
        if not self._child_assignments or not task_ids:
            return {}

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
        task_ids_map, removed_tasks = self.policy.get_children_tasks(
            tasks=task_objs,
            children_resources=children_resources,
            ntask=ntask,
            child_assignments=wid_assignments,
            child_status=wid_status,
        )

        if removed_tasks:
            self.logger.warning(
                f"Policy could not place {len(removed_tasks)} tasks: {removed_tasks}"
            )

        child_assignments: Dict[str, List[str]] = {}
        for wid, assigned_ids in task_ids_map.items():
            child_id = self._wid_to_child_id[wid]
            self._child_assignments[child_id]["task_ids"].extend(assigned_ids)
            for tid in assigned_ids:
                self._unassigned_tasks.discard(tid)
            child_assignments[child_id] = assigned_ids

        return child_assignments

    def free(self, child_id: str) -> bool:
        """Deallocate cluster resources for a child. No-op if already freed."""
        if child_id in self.workers:
            result = self.cluster.deallocate(self.workers[child_id])
            if result:
                del self.workers[child_id]
            return result
        return False


class AsyncTaskScheduler(AsyncScheduler):
    """Task-level scheduler used by workers: allocates cluster resources per task
    and exposes a ready_tasks queue consumed by the worker's execution loop."""

    def __init__(
        self,
        logger: Logger,
        tasks: Dict[str, Task],
        nodes: JobResource,
        policy: Union[str, Policy] = "large_resource_policy",
    ) -> None:
        """Initialise the task scheduler.

        Args:
            logger: Logger instance.
            tasks: Initial task dict to schedule.
            nodes: Available cluster resources for this worker.
            policy: Policy name or instance used to score/prioritise tasks.
        """
        cluster = AsyncLocalClusterResource(logger.getChild("cluster"), nodes)
        super().__init__(logger, cluster)
        self.tasks: Dict[str, Task] = tasks
        if isinstance(policy, str):
            self.scheduler_policy: Policy = policy_registry.create_policy(policy)
        else:
            self.scheduler_policy: Policy = policy
        self._sorted_tasks: PriorityQueue[Tuple[float, str]] = PriorityQueue()
        for task_id, task in self.tasks.items():
            try:
                self._sorted_tasks.put_nowait(
                    (
                        1.0 / self.scheduler_policy.get_score(self.tasks[task_id]),
                        task_id,
                    )
                )
            except asyncio.QueueFull:
                self.logger.error("Sorted task queue is full!")
                raise RuntimeError("Sorted task is full!")
        self.logger.debug(f"Sorted tasks {self._sorted_tasks}")
        ##
        self.ready_tasks: Queue[Tuple[str, JobResource]] = Queue()
        self._running_tasks: Dict[str, JobResource] = {}
        self._done_tasks: Counter[str] = Counter()
        self._failed_tasks: Set[str] = set()
        self._successful_tasks: Set[str] = set()

        self._stop_monitoring = asyncio.Event()
        self._all_tasks_done = asyncio.Event()
        self._consecutive_failed_allocations = 0
        self._monitoring_task = None
        self._event_loop = None  # Will be set when monitoring starts

        self._event_registry: Optional[EventRegistry] = None
        if os.environ.get("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry = get_registry()

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

    def _find_min_resource_req(self) -> JobResource:
        """
        Find the minimum resource requirement among PENDING tasks only.
        """
        if self._sorted_tasks.empty():
            return None

        # Get pending task IDs from sorted_tasks queue
        pending_task_ids = [task_id for _, task_id in list(self._sorted_tasks._queue)]

        if not pending_task_ids:
            return None

        pending_tasks = [
            self.tasks[tid] for tid in pending_task_ids if tid in self.tasks
        ]

        if not pending_tasks:
            return None

        min_nnodes = min(task.nnodes for task in pending_tasks)
        min_ppn = min(task.ppn for task in pending_tasks)
        min_ngpus = min(task.ngpus_per_process * task.ppn for task in pending_tasks)

        return JobResource(
            resources=[
                NodeResourceCount(ncpus=min_ppn, ngpus=min_ngpus)
                for _ in range(min_nnodes)
            ]
        )

    async def _monitor_resources(self) -> None:
        """
        Monitors free resources and checks if any tasks can be allocated, and moves them to the ready queue.
        """
        self.logger.info("Starting resource monitor")

        while not self._stop_monitoring.is_set():
            try:
                # Wait for resources - will be cancelled when monitor is stopped
                min_req = self._find_min_resource_req()
                self.logger.debug(
                    f"Waiting for free resources with min requirement: {min_req}. Current free resources: {self.cluster.get_status()}"
                )
                await self._cluster_resource.wait_for_free(min_resources=min_req)
                self.logger.debug(
                    f"Resource monitor woke up, checking for task allocation. Current free resources: {self.cluster.get_status()}"
                )

                # Check stop immediately
                if self._stop_monitoring.is_set():
                    break

                # Wait for tasks if queue is empty
                if self._sorted_tasks.empty():
                    self.logger.info("No tasks available, waiting for new tasks")
                    # Wait for either a task to be added or stop signal
                    wait_task = asyncio.create_task(self._sorted_tasks.get())
                    stop_task = asyncio.create_task(self._stop_monitoring.wait())

                    done, pending = await asyncio.wait(
                        [wait_task, stop_task], return_when=asyncio.FIRST_COMPLETED
                    )

                    # Cancel pending tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                    # Check which completed
                    if stop_task in done:
                        self.logger.info("Stop signal received while waiting for tasks")
                        break

                    # Put the task back
                    if wait_task in done:
                        priority, task_id = await wait_task
                        self._sorted_tasks.put_nowait((priority, task_id))
                        self.logger.info("New tasks available, resuming scheduling")

                unallocated_tasks = []
                allocated_count = 0

                # Try to allocate all pending tasks
                for _ in range(self._sorted_tasks.qsize()):
                    if self._stop_monitoring.is_set():
                        break

                    try:
                        priority, task_id = await self._sorted_tasks.get()
                    except asyncio.QueueEmpty:
                        break

                    # Thread-safe access to tasks
                    if task_id not in self.tasks:
                        self.logger.warning(
                            f"Task {task_id} no longer exists, skipping"
                        )
                        continue
                    task = self.tasks[task_id]

                    req = task.get_resource_requirements()

                    allocated, resource = self.cluster.allocate(req)

                    if allocated:
                        # Put in ready queue - awaitable operation yields control
                        await self.ready_tasks.put((task_id, resource))

                        # Track as running
                        self._running_tasks[task_id] = resource

                        allocated_count += 1
                        self.logger.debug(f"Task {task_id} ready for execution")
                    else:
                        unallocated_tasks.append((priority, task_id))
                        # Only break if cluster has no free resources at all
                        if not self.cluster._resource_available.is_set():
                            self.logger.debug("No more free resources available")
                            break
                        else:
                            self.logger.debug(
                                f"Insufficient resources for task {task_id}. Resources requested: {req}. Free resources: {self.cluster.get_status()}"
                            )

                if allocated_count == 0:
                    self.logger.debug(
                        "No tasks allocated in this cycle. Clearing resource available event to wait for new resources."
                    )
                    self._cluster_resource.clear_resource_available()

                # Put back unallocated tasks
                for item in unallocated_tasks:
                    self._sorted_tasks.put_nowait(item)

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

    def add_task(self, task: Task) -> bool:
        """Add a task to the priority queue for scheduling. Returns True on success."""
        try:
            if task.nnodes > len(self.cluster.nodes.nodes):
                raise ValueError(
                    f"Task {task.task_id} requires {task.nnodes} nodes, but only {len(self.cluster.nodes.nodes)} are available"
                )
            self.tasks[task.task_id] = task
            self._sorted_tasks.put_nowait(
                (self.scheduler_policy.get_score(task), task.task_id)
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False

    def delete_task(self, task: Task) -> bool:
        """Remove a task from all queues and free its resources. Returns True on success."""
        if task.task_id not in self.tasks:
            self.logger.warning(f"Unknown task: {task.task_id}")
            return False

        try:
            # Remove from tasks dict
            del self.tasks[task.task_id]

            # Remove from sorted tasks queue
            temp_items = []
            while not self._sorted_tasks.empty():
                try:
                    priority, tid = self._sorted_tasks.get_nowait()
                    if tid != task.task_id:
                        temp_items.append((priority, tid))
                except asyncio.QueueEmpty:
                    break

            # Put back all items except the deleted task
            for item in temp_items:
                self._sorted_tasks.put_nowait(item)

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

            # Remove from running and status sets
            del self._running_tasks[task.task_id]

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
