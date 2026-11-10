import asyncio
from asyncio import Queue
from logging import Logger
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

from ensemble_launcher.config import LauncherConfig, PolicyConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.taskstore import TaskStore, create_task_store

from .child_state import ChildState
from .heap import PendingEntry, PendingTaskHeap, PendingTaskQueue
from .policy import ChildrenPolicy, Policy, policy_registry
from .resource import (
    AsyncLocalClusterResource,
    JobResource,
    NodeResourceCount,
)
from .scheduler import Scheduler
from .state import Child, SchedulerState, TaskSchedulerState

if TYPE_CHECKING:
    from ensemble_launcher.comm.messages import Result, Status, TaskUpdate

# self.logger = logging.getself.logger(__name__)

# Task policy that is special-cased by name: unscored, deque-backed pending queue.
FIFO_POLICY_NAME = "fifo_policy"


def make_task_store(config: LauncherConfig, node_id: Optional[str], logger: Logger) -> TaskStore:
    """The node's task store, as configured. Task ids throughout the schedulers are its uids."""
    return create_task_store(
        config.task_store,
        node_id=node_id or "node",
        hot_bytes=config.task_store_hot_bytes,
        spill_dir=config.task_store_dir,
        logger=logger,
    )


class AsyncScheduler(Scheduler):
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """

    def __init__(self, logger: Logger, cluster_resource: AsyncLocalClusterResource):
        super().__init__(logger=logger, cluster_resource=cluster_resource)


class AsyncChildrenScheduler(AsyncScheduler):
    """A master's scheduler: lays out children on its nodes, places tasks on them, and tracks
    their lifecycle.

    Everything it knows lives in ``self.state`` (a `SchedulerState`: one `Child` record per
    child plus the unassigned pool). Callers read ``self.state`` directly and report events;
    placement happens here:

    - ``await queue(update)`` stores new tasks and, in push mode, puts them on a private input
      queue. The assigner task places them in chunks of ``assign_chunk_size`` -- yielding to the
      loop between chunks -- and puts **one uid list per child per input item** on that child's
      ``outbox``. The master's per-child sender consumes the outbox.
    - In pull mode (work stealing) tasks wait in the pool until ``request(child_id, n)``.
    - Uids no child can take go on ``unplaced``; the master fails them.
    - ``drained`` is set while no unfinished task is left anywhere.
    """

    # Legal child-state moves: target -> allowed sources.
    _ALLOWED = {
        ChildState.READY: frozenset({ChildState.NOTREADY}),
        ChildState.RUNNING: frozenset({ChildState.READY, ChildState.RECOVERING}),
        ChildState.RECOVERING: frozenset({ChildState.RUNNING}),
        ChildState.FAILED: frozenset({ChildState.READY, ChildState.RUNNING, ChildState.RECOVERING}),
        ChildState.SUCCESS: frozenset({ChildState.RUNNING, ChildState.RECOVERING}),
    }

    PUSH = "push"
    PULL = "pull"

    def __init__(
        self,
        logger: Logger,
        nodes: JobResource,
        config: LauncherConfig,
        node_id: Optional[str] = None,
        level: Optional[int] = None,
        placement: str = PUSH,
    ) -> None:
        """
        Args:
            logger: Logger instance for this scheduler.
            nodes: Available cluster resources.
            config: Launcher configuration (policy name, nchildren, chunk size, ...).
            node_id: ID of the owning master node.
            level: Optional hierarchy level (0 = root master).
            placement: ``"push"`` (place tasks as they arrive) or ``"pull"`` (wait for
                ``request``; work stealing).
        """
        if placement not in (self.PUSH, self.PULL):
            raise ValueError(f"placement must be 'push' or 'pull', not {placement!r}")
        cluster = AsyncLocalClusterResource(logger.getChild("cluster"), nodes)
        super().__init__(logger, cluster)

        self._config = config
        self.placement = placement
        self.policy: ChildrenPolicy = policy_registry.create_policy(
            self._config.children_scheduler_policy,
            policy_kwargs={
                "policy_config": self._config.policy_config,
                "node_id": node_id,
                "logger": logger.getChild("policy"),
            },
        )
        # Every task this master holds, keyed by uid.
        self.tasks: TaskStore = make_task_store(config, node_id, logger.getChild("taskstore"))
        self.policy.bind_tasks(self.tasks)
        self._event_loop = asyncio.get_event_loop()
        self.cluster.set_event_loop(self._event_loop)

        self.state = SchedulerState(node_id=node_id or "node", level=level)
        # Work for the assigner: uid lists. Private -- the way in is queue().
        self._input: asyncio.Queue = asyncio.Queue()
        self._assigner: Optional[asyncio.Task] = None
        self._placing = False  # an input item is being placed
        # Uid lists no child could take; the master fails them.
        self.unplaced: asyncio.Queue = asyncio.Queue()
        # Set while no unfinished task is left anywhere (input, pool, children).
        self.drained = asyncio.Event()
        self.drained.set()

        # Results out (queue_result). Items are IResultBatch pieces. A node has at most one
        # parent, so results with no client share one fixed queue; at the root nothing sends it
        # and it simply holds the run's results. Client outboxes are made on first use, and
        # each new client is announced on ``new_clients`` so the master can start its sender.
        self.parent_outbox: asyncio.Queue = asyncio.Queue()
        self.client_outboxes: Dict[str, asyncio.Queue] = {}
        self.new_clients: asyncio.Queue = asyncio.Queue()

    # ------------------------------------------------------------------
    # Tasks in
    # ------------------------------------------------------------------

    async def queue(self, update: "TaskUpdate", client_id: Optional[str] = None) -> List[bytes]:
        """Store a received TaskUpdate and queue its tasks for placement.

        Storing and pooling happen now, so from here on the state accounts for every task
        (checkpoints, ``delete_task``, totals); only placement is deferred to the assigner.
        Returns the uids added -- tasks the store already holds (a resend) are left out.
        Async so that a bounded input queue could apply backpressure later.
        """
        uids = self.tasks.add(update)
        if not uids:
            return uids
        self.state.queue(uids, client_id)
        self.drained.clear()
        if self.placement == self.PUSH:
            self._input.put_nowait(uids)
        return uids

    def finish_tasks(
        self, rows: Iterable[Tuple[bytes, bool]], child_id: Optional[str] = None
    ) -> List[Optional[str]]:
        """Record the final status of ``(uid, success)`` rows and forget the tasks.

        ``child_id`` is the child the results came from -- the one the tasks were assigned
        to, since results travel back up the same branch; None for tasks failed before they
        reached a child (they leave the pool). Sets SUCCESS/FAILED in the store and releases
        the tasks' bytes. Returns each row's submitting client (None if it came from the
        parent), in row order.
        """
        state = self.state
        ok, failed, clients = [], [], []
        child = state.children.get(child_id) if child_id is not None else None
        for uid, success in rows:
            (ok if success else failed).append(uid)
            clients.append(state.finish(uid, child))
        self.tasks.set_status(ok, TaskStatus.SUCCESS)
        self.tasks.set_status(failed, TaskStatus.FAILED)
        self.tasks.release(ok + failed)
        self._check_drained()
        return clients

    def requeue(self, uids: Iterable[bytes]) -> None:
        """Place tasks back in the pool again -- e.g. a removed child's, when it can't come
        back. Uids no longer in the pool (finished, or claimed by a child) are skipped. In
        pull mode they just wait in the pool for the next request."""
        pool = self.state.unassigned
        uids = [uid for uid in uids if uid in pool]
        if uids and self.placement == self.PUSH:
            self.drained.clear()
            self._input.put_nowait(uids)

    def queue_result(self, batch, child_id: Optional[str] = None) -> None:
        """Results arrived (from ``child_id``, or made here for tasks no child holds).

        Finishes the tasks, then puts the batch -- split per destination by row, as bytes --
        on the submitting client's outbox, or on ``parent_outbox`` for tasks with no client.
        """
        rows = list(batch.rows())
        if not rows:
            return
        clients = self.finish_tasks(rows, child_id)
        groups: Dict[Optional[str], List[int]] = {}
        for i, client in enumerate(clients):
            groups.setdefault(client, []).append(i)
        for client, picked in groups.items():
            piece = batch if len(picked) == len(rows) else batch.select(picked)
            if client is None:
                self.parent_outbox.put_nowait(piece)
                continue
            outbox = self.client_outboxes.get(client)
            if outbox is None:
                outbox = self.client_outboxes[client] = asyncio.Queue()
                self.new_clients.put_nowait(client)
            outbox.put_nowait(piece)

    def delete_task(self, uid: bytes) -> None:
        """Remove a task from the scheduler entirely."""
        self.tasks.release((uid,))
        self.state.forget(uid)
        self._check_drained()

    def delivered(self, uids: Iterable[bytes]) -> None:
        """A TaskUpdate carrying these tasks reached its child: their bytes are now needed
        only for recovery, so the store may offload them first."""
        self.tasks.mark_delivered(uids)

    # ------------------------------------------------------------------
    # Placement
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the assigner. Call once the children exist."""
        if self._assigner is None or self._assigner.done():
            self._assigner = asyncio.create_task(self._assign_loop())

    async def stop(self) -> None:
        """Cancel the assigner."""
        if self._assigner is not None and not self._assigner.done():
            self._assigner.cancel()
            try:
                await self._assigner
            except asyncio.CancelledError:
                pass
        self._assigner = None

    async def _assign_loop(self) -> None:
        while True:
            uids = await self._input.get()
            self._placing = True
            try:
                await self._place_item(uids)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Placing {len(uids)} tasks failed: {e}", exc_info=True)
            finally:
                self._placing = False
                self._input.task_done()
                self._check_drained()

    async def _place_item(self, uids: List[bytes]) -> None:
        """Place one input item in chunks, yielding between them; then give each child one
        list for the whole item, so one incoming update is one TaskUpdate per child."""
        size = max(1, self._config.assign_chunk_size)
        buffers: Dict[str, List[bytes]] = {}
        unplaced: List[bytes] = []
        for start in range(0, len(uids), size):
            chunk = [uid for uid in uids[start : start + size] if uid in self.state.unassigned]
            if chunk:
                placed, left = self._place(chunk)
                for child_id, got in placed.items():
                    buffers.setdefault(child_id, []).extend(got)
                unplaced.extend(left)
            await asyncio.sleep(0)
        for child_id, got in buffers.items():
            if child_id in self.state.children:
                self.state.children[child_id].outbox.put_nowait(got)
        if unplaced:
            self.unplaced.put_nowait(unplaced)

    def _place(
        self,
        uids: List[bytes],
        child_ids: Optional[Iterable[str]] = None,
        ntask: Optional[int] = None,
    ) -> Tuple[Dict[str, List[bytes]], List[bytes]]:
        """Ask the policy where ``uids`` go (all children, or ``child_ids``) and record it.

        Returns ``(child_id -> placed uids, unplaced uids)``. Does not touch outboxes.
        """
        state = self.state
        if child_ids is None:
            targets = state.children.values()
        else:
            targets = [state.children[c] for c in child_ids if c in state.children]
        # A finished child takes nothing more (e.g. a FAILED child's own requeued tasks).
        children = {child.wid: child for child in targets if not child.terminal}
        if not children:
            return {}, list(uids)
        by_wid, _, unplaced = self.policy.get_children_tasks(
            uids=uids, children=children, ntask=ntask, level=state.level
        )
        placed: Dict[str, List[bytes]] = {}
        for wid, got in by_wid.items():
            if got:
                child = children[wid]
                state.assign(child, got)
                placed[child.id] = got
        return placed, list(unplaced)

    def request(self, child_id: str, ntask: Optional[int]) -> int:
        """Pull mode: a child asked for up to ``ntask`` tasks. Places them from the pool and
        puts them on its outbox as one list. Returns how many were placed."""
        child = self.state.children.get(child_id)
        if child is None or not self.state.unassigned:
            return 0
        placed, _ = self._place(list(self.state.unassigned), child_ids=[child_id], ntask=ntask)
        got = placed.get(child_id, [])
        if got:
            child.outbox.put_nowait(got)
        return len(got)

    def _check_drained(self) -> None:
        if self._input.empty() and not self._placing and self.state.empty():
            self.drained.set()
        else:
            self.drained.clear()

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------

    def create_children(
        self,
        level: int,
        node_id: str,
        reset: bool = True,
        nodes: Optional[JobResource] = None,
    ) -> None:
        """Ask the policy for a child layout, allocate it, and add the children (READY).

        reset=True  -- start from no children (full re-assignment).
        reset=False -- additive; existing children are kept and new wids follow theirs.
        nodes       -- restrict allocation to these nodes (e.g. recovered nodes on retry).
        """
        state = self.state
        if reset:
            for child_id in list(state.children):
                state.pop_child(child_id)
            state.all_done.clear()
        state.level = level
        suffix = ".w" if level + 1 == self._config.policy_config.nlevels else ".m"
        wid_offset = state.next_wid()

        layout = self.policy.get_children_resources(
            uids=list(self.tasks),
            nodes=nodes if nodes is not None else self.cluster.nodes,
            level=level,
        )
        for orig_wid, job_resource in layout.items():
            wid = orig_wid + wid_offset
            allocated, resource = self.cluster.allocate(job_resource)
            if not allocated:
                self.logger.warning(f"Failed to allocate resources for worker {wid}")
                continue
            child = Child(id=f"{node_id}{suffix}{wid}", wid=wid, job_resource=resource)
            child.allocated = True
            state.add_child(child)
            self.transition(child.id, ChildState.READY)

    def transition(self, child_id: str, to: ChildState, requeue: bool = False) -> bool:
        """Move a child to ``to`` and apply that state's side effects.

        Raises RuntimeError on an illegal move; otherwise returns True.

        A terminal move (SUCCESS or FAILED) always empties the child -- a finished child runs
        nothing more: its tasks go back to the pool, then ``requeue=True`` places them on the
        other children, and ``requeue=False`` puts them on ``unplaced`` (the master fails them).
        """
        child = self.state.children.get(child_id)
        current = child.state if child is not None else None
        if current not in self._ALLOWED[to]:
            self.logger.error(f"Invalid transition {child_id}: {current} -> {to}")
            raise RuntimeError(
                f"Invalid child state transition for {child_id}: {current} -> {to}"
            )
        child.state = to
        if to is ChildState.RUNNING:
            child.done.clear()
            child.ready.set()
            self.tasks.set_status(child.uids, TaskStatus.RUNNING)
        elif child.terminal:
            self._free(child)
            child.done.set()
            if child.uids:
                # Back to the pool; the child is terminal already, so placement skips it.
                uids = list(child.uids)
                self.state.unassigned.update(child.uids)
                child.uids.clear()
                if requeue:
                    self.requeue(uids)
                else:
                    self.unplaced.put_nowait(uids)
            if self.state.all_terminal():
                self.state.all_done.set()
        return True

    def remove_child(self, child_id: str) -> Optional[Child]:
        """Drop a child: its unfinished tasks go back to the pool and its resources back to
        the cluster. Returns the removed record, for `reinstate_child` (or `requeue` of its
        tasks if it can't come back)."""
        child = self.state.pop_child(child_id)
        if child is None:
            return None
        self._free(child)
        return child

    def reinstate_child(self, child: Child) -> bool:
        """Put a removed child back on its own resources (READY). It claims its unfinished
        tasks back from the pool, and its fresh outbox holds all of them, to be sent once the
        new process is ready. Returns False if its resources can't be re-allocated."""
        assert child.id not in self.state.children, "Remove the child before reinstating"
        allocated, resource = self.cluster.allocate(child.job_resource)
        if not allocated:
            return False
        child.job_resource = resource
        child.allocated = True
        child.state = ChildState.NOTREADY
        child.status = None
        child.ready = asyncio.Event()
        child.done = asyncio.Event()
        self.state.add_child(child)
        self.transition(child.id, ChildState.READY)
        return True

    def _free(self, child: Child) -> None:
        """Return a child's resources to the cluster. No-op if already freed."""
        if child.allocated and self.cluster.deallocate(child.job_resource):
            child.allocated = False

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def get_state(self) -> str:
        """The state as checkpoint JSON. Runs on the loop, so it never sees a half-made
        change; the checkpointer only writes the string."""
        self.state.nodes = self.cluster.nodes
        return self.state.to_json()

    def set_state(self, restored: SchedulerState) -> None:
        """Adopt the children of a checkpointed state: same ids, resources and tasks, without
        re-running the policy.

        Call after the initial tasks were queued and before ``start()``: each restored child
        claims its tasks from the pool (``add_child``), so the assigner then only places
        tasks no child held. A child whose resources can't be re-allocated is skipped and its
        tasks stay in the pool.
        """
        state = self.state
        state.level = restored.level
        for child_id in list(state.children):
            state.pop_child(child_id)
        state.task_to_client.update(restored.task_to_client)
        nheld = 0
        for child in restored.children.values():
            if child.job_resource is None:
                continue
            allocated, resource = self.cluster.allocate(child.job_resource)
            if not allocated:
                self.logger.warning(f"set_state: can't re-allocate {child.id}; skipping it")
                continue
            child.job_resource = resource
            child.allocated = True
            state.add_child(child)
            nheld += len(child)
            self.transition(child.id, ChildState.READY)
        self.logger.info(
            f"set_state: restored {len(state.children)} children holding {nheld} tasks, "
            f"{len(state.unassigned)} in the pool"
        )


class AsyncTaskScheduler(AsyncScheduler):
    """Task-level scheduler used by workers: allocates cluster resources per task
    and exposes a ready_tasks queue consumed by the worker's execution loop."""

    def __init__(
        self,
        logger: Logger,
        nodes: JobResource,
        config: LauncherConfig,
        node_id: str = "node",
    ) -> None:
        """Initialise the task scheduler.

        Args:
            logger: Logger instance.
            tasks: The worker's TaskStore (task ids below are its uids). A TaskUpdate, an
                iterable of Tasks or a dict of Tasks is put into a fresh in-memory store.
            nodes: Available cluster resources for this worker.
            policy: Policy name or instance used to score/prioritise tasks.
                ``"fifo_policy"`` (the default) is special-cased: tasks go into
                a deque and are never scored. Any other policy scores each task
                into a priority heap.
            policy_config: Configuration passed to the policy constructor
                when *policy* is given as a string.
            node_id: The owning node's id, written into the checkpoint.
        """
        cluster = AsyncLocalClusterResource(logger.getChild("cluster"), nodes)
        super().__init__(logger, cluster)
        self._config = config
        self.tasks: TaskStore = make_task_store(config, node_id, logger.getChild("TaskScheduler"))
        policy = self._config.task_scheduler_policy
        self.scheduler_policy: Policy = policy_registry.create_policy(
            policy,
            policy_kwargs={
                "policy_config": self._config.policy_config,
                "logger": logger.getChild("policy"),
            },
        )
        self.scheduler_policy.bind_tasks(self.tasks)
        self._strict_priority: bool = self._config.policy_config.strict_priority
        self.ready_tasks: Queue[Tuple[str, JobResource]] = Queue()
        self._fifo: bool = policy == FIFO_POLICY_NAME
        # Everything this scheduler knows about its tasks; see TaskSchedulerState.
        self.state = TaskSchedulerState(
            node_id=node_id,
            nodes=self.cluster.nodes,
            pending_tasks=PendingTaskQueue() if self._fifo else PendingTaskHeap(),
        )
        # Running per-dimension minimum (nnodes, ppn, ngpus) over enqueued tasks.
        # Only lowered on enqueue, reset when the queue empties -- so it may be
        # stale-low (a spurious wakeup) but never too high (which would stall).
        self._min_req: Optional[Tuple[int, int, float]] = None
        self._seed_pending(list(self.tasks))
        self.logger.debug(f"Pending tasks: {len(self.state.pending_tasks)}")

        self._stop_monitoring = asyncio.Event()
        self._consecutive_failed_allocations = 0
        self._monitoring_task = None
        self._event_loop = None  # Will be set when monitoring starts

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

    def _update_min_req(self, task) -> None:
        shape = (task.nnodes, task.ppn, task.ngpus_per_process * task.ppn)
        if self._min_req is None:
            self._min_req = shape
        else:
            self._min_req = tuple(map(min, self._min_req, shape))

    def _enqueue(self, task_id: bytes) -> None:
        """The single place a task enters the pending container.

        FIFO pushes without scoring. Any other policy is scored, and gets the live
        ``self.state`` as its ``scheduler_state``.
        """
        if self._fifo:
            self.state.pending_tasks.push(0.0, task_id)
        else:
            score = self.scheduler_policy.get_score(task_id, scheduler_state=self.state)
            self.state.pending_tasks.push(self._priority_from_score(score), task_id)
        self._update_min_req(self.tasks[task_id])

    def _seed_pending(self, task_ids: Iterable[str]) -> int:
        """Replace the pending container with fresh entries for *task_ids*.

        Used when the set of pending tasks itself is being established (construction,
        checkpoint restore). To re-score the tasks already queued without changing
        membership, use :meth:`reprioritize_pending` instead.
        """
        self.state.pending_tasks.clear()
        self._min_req = None
        for task_id in task_ids:
            if task_id not in self.tasks:
                continue
            self._enqueue(task_id)
        return len(self.state.pending_tasks)

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
            return len(self.state.pending_tasks)
        scores = {
            task_id: self._priority_from_score(
                self.scheduler_policy.get_score(task_id, scheduler_state=self.state)
            )
            for task_id in self.state.pending_tasks.task_ids()
            if task_id in self.tasks
        }
        self.state.pending_tasks.rescore(scores)
        return len(scores)

    def get_state(self) -> str:
        """The checkpoint: ``self.state`` as JSON, built here on the loop because the
        checkpointer writes on a thread. Running tasks are written as pending."""
        self.state.nodes = self.cluster.nodes
        return self.state.to_json()

    def set_state(self, restored: TaskSchedulerState) -> None:
        """Adopt a checkpointed state: record its finished tasks and re-queue everything
        else the store still holds. Results of completed tasks are restored by the worker.

        The pending container and ``all_done`` are kept from the current state; the
        restored ``pending_tasks`` list is only informational -- the store decides.
        """
        finished = restored.completed_tasks | restored.failed_tasks
        self.tasks.set_status(restored.completed_tasks, TaskStatus.SUCCESS)
        self.tasks.set_status(restored.failed_tasks, TaskStatus.FAILED)
        self.tasks.release(finished)

        restored.pending_tasks = self.state.pending_tasks
        restored.running_tasks = self.state.running_tasks
        restored.all_done = self.state.all_done
        restored.task_to_client = {**self.state.task_to_client, **restored.task_to_client}
        self.state = restored

        pending = [uid for uid in self.tasks if uid not in finished]
        self._seed_pending(pending)
        self.logger.info(
            f"set_state: {len(finished)} tasks restored from checkpoint, "
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
        if self.state.pending_tasks.empty():
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
                entry = self.state.pending_tasks.pop()
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
                    self.state.running_tasks[task_id] = resource
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
            self.state.pending_tasks.requeue(skipped)
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
                if self.state.pending_tasks.empty():
                    self.logger.info("No tasks available, waiting for new tasks")
                    stop_task = asyncio.create_task(self._stop_monitoring.wait())
                    tasks_task = asyncio.create_task(
                        self.state.pending_tasks.wait_for_tasks()
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
        self.state.all_done.clear()
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
        # Every task is either pending, running or finished (finished ones leave the
        # store), so nothing pending and nothing running means done.
        if self.state.running_tasks or not self.state.pending_tasks.empty():
            return
        remaining = [
            tid for tid in self.tasks
            if tid not in self.state.completed_tasks and tid not in self.state.failed_tasks
        ]
        self.logger.debug(f"Checking completion: {len(remaining)} tasks remaining")
        if not remaining:
            self.logger.info("All tasks completed")
            if self._event_loop is not None:
                self.logger.debug(
                    f"Setting _all_tasks_done event via stored loop {self._event_loop}"
                )
                self._event_loop.call_soon_threadsafe(self.state.all_done.set)
            else:
                self.logger.warning(
                    "No event loop stored, setting event directly (may not work!)"
                )
                self.state.all_done.set()

    async def wait_for_completion(self):
        """
        Wait for all tasks to complete.
        This replaces the while loop in the worker's run() method.
        """
        self.logger.debug("Waiting for all tasks to complete")
        await self.state.all_done.wait()
        self.logger.debug("Done waiting for task completion!")

    def add_task(self, task: Task, client_id: Optional[str] = None) -> bool:
        """Convenience for one Task object: encode it and add it. True if it was queued."""
        from ensemble_launcher.comm.messages import TaskUpdate

        accepted, _ = self.add_tasks(TaskUpdate.from_tasks([task]), client_id=client_id)
        return bool(accepted)

    def add_tasks(
        self, update: "TaskUpdate", client_id: Optional[str] = None
    ) -> Tuple[List[bytes], List[Tuple[bytes, str]]]:
        """Store a received TaskUpdate and queue its tasks.

        Returns ``(accepted uids, rejected (uid, task_id) pairs)``. A task needing more
        nodes than this worker has is rejected (and released); tasks the store already holds
        are skipped.
        """
        nnodes = len(self.cluster.nodes.nodes)
        accepted, rejected = [], []
        for task_id in self.tasks.add(update):
            task = self.tasks[task_id]
            if task.nnodes > nnodes:
                self.logger.error(
                    f"Task {task.task_id} requires {task.nnodes} nodes, but only {nnodes} are available"
                )
                rejected.append((task_id, task.task_id))
                continue
            if client_id is not None:
                self.state.task_to_client[task_id] = client_id
            self._enqueue(task_id)
            accepted.append(task_id)
        if accepted:
            self.state.all_done.clear()  # new work: not drained any more
        if rejected:
            uids = [uid for uid, _ in rejected]
            self.tasks.set_status(uids, TaskStatus.FAILED)
            self.tasks.release(uids)
        return accepted, rejected

    def delete_task(self, task_id: bytes) -> bool:
        """Remove a task from all queues and free its resources. Returns True on success."""
        if task_id not in self.tasks:
            self.logger.warning(f"Unknown task: {task_id.hex()}")
            return False

        task = _TaskRef(task_id)
        try:
            # Drop the task's bytes
            self.tasks.release((task_id,))
            self.state.task_to_client.pop(task_id, None)

            # Remove from pending tasks
            self.state.pending_tasks.remove(task.task_id)
            if self.state.pending_tasks.empty():
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
            if task.task_id in self.state.running_tasks:
                self.cluster.deallocate(self.state.running_tasks[task.task_id])

            # Remove from running and status sets (a pending task was never running)
            self.state.running_tasks.pop(task.task_id, None)

            # remove from failed and succesful tasks
            self.state.failed_tasks.discard(task.task_id)
            self.state.completed_tasks.discard(task.task_id)

            return True
        except Exception as e:
            self.logger.warning(f"Failed to delete task {task.task_id}: {e}")
            return False

    def free(self, task_id: str, status: TaskStatus) -> None:
        """Deallocate resources for a completed task and record its final status."""
        if task_id in self.tasks:
            if task_id not in self.state.running_tasks:
                self.logger.error(f"{task_id} is not running")
                raise RuntimeError

            # deallocate
            self.cluster.deallocate(self.state.running_tasks[task_id])

            # delete from running tasks
            del self.state.running_tasks[task_id]

            # add to failed/successful tasks
            if status == TaskStatus.FAILED:
                self.state.failed_tasks.add(task_id)
            elif status == TaskStatus.SUCCESS:
                self.state.completed_tasks.add(task_id)
                self.state.failed_tasks.discard(task_id)

            self.logger.debug(f"Freed {task_id}")

            # Notify the policy about the completed task (only if it has the hook).
            if (
                type(self.scheduler_policy).on_task_complete
                is not Policy.on_task_complete
            ):
                self.scheduler_policy.on_task_complete(
                    uid=task_id,
                    status=status,
                    scheduler_state=self.state,
                )

            self.tasks.set_status((task_id,), status)
            self.tasks.release((task_id,))

        self._check_all_tasks_done()
        return None

    def fail_unfinished(self) -> List[bytes]:
        """Fail every pending and running task, for a worker that is stopping; returns
        their uids. Call after `stop_monitoring`, so nothing is dispatched meanwhile.

        Running tasks (including ones allocated but still in ``ready_tasks``) go through
        `free`; pending ones never ran, so they are only recorded as failed and released.
        """
        while not self.ready_tasks.empty():  # their uids are in _running_tasks
            self.ready_tasks.get_nowait()
        running = list(self.state.running_tasks)
        for uid in running:
            self.free(uid, TaskStatus.FAILED)
        pending = [uid for uid in self.state.pending_tasks.task_ids() if uid in self.tasks]
        self.state.pending_tasks.clear()
        self.state.failed_tasks.update(pending)
        self.tasks.set_status(pending, TaskStatus.FAILED)
        self.tasks.release(pending)
        self._check_all_tasks_done()
        return running + pending


class _TaskRef:
    """Just the uid of a task, for code paths that used to take a Task only for its id."""

    __slots__ = ("task_id",)

    def __init__(self, task_id: bytes):
        self.task_id = task_id


def _as_task_store(tasks, logger: Logger) -> TaskStore:
    """A TaskStore for the task scheduler, wrapping plain Tasks or a TaskUpdate if given."""
    if isinstance(tasks, TaskStore):
        return tasks
    from ensemble_launcher.comm.messages import TaskUpdate
    from ensemble_launcher.taskstore import SegmentTaskStore

    store = SegmentTaskStore(logger=logger.getChild("taskstore"))
    if tasks is None:
        return store
    if isinstance(tasks, dict):
        tasks = list(tasks.values())
    if not isinstance(tasks, TaskUpdate):
        tasks = TaskUpdate.from_tasks(tasks)
    store.add(tasks)
    return store
