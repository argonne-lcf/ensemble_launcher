import asyncio
from asyncio import Queue
from logging import Logger
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from ensemble_launcher.config import LauncherConfig, PolicyConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.taskstore import NO_CLIENT, UNASSIGNED, TaskStore, create_task_store

from .child_state import ChildState
from .heap import PendingEntry, PendingTaskHeap, PendingTaskQueue
from .policy import ChildrenPolicy, Policy, policy_registry, prefers_columns, wids_from_groups
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

def _as_handles(idx) -> np.ndarray:
    return np.asarray(idx, dtype=np.int64).reshape(-1)


# Task policy that is special-cased by name: unscored, deque-backed pending queue.
FIFO_POLICY_NAME = "fifo_policy"


def make_task_store(config: LauncherConfig, node_id: Optional[str], logger: Logger) -> TaskStore:
    """The node's task store, as configured. Task ids throughout the schedulers are its handles."""
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

    It owns two things and is the only code that changes both: ``self.state`` (a
    `SchedulerState`: the children and their resource allocations) and ``self.tasks`` (the
    `TaskStore`, which also records where each task is: `UNASSIGNED` = the pool, or the wid of
    the child it is placed on). Everything is by store handle; uids appear only in messages.
    Callers read both directly and report events; placement happens here:

    - ``await queue(update)`` stores new tasks and, in push mode, puts their handles on a
      private input queue. The assigner task places them in chunks of ``assign_chunk_size`` --
      yielding to the loop between chunks -- and puts **one handle array per child per input
      item** on that child's ``outbox``. The master's per-child sender consumes the outbox.
    - In pull mode (work stealing) tasks wait in the pool until ``request(child_id, n)``.
    - Handles no child can take go on ``unplaced``; the master fails them.
    - ``drained`` is set while the store holds no unfinished task and nothing is being placed.
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
        # Every task this master holds (by handle), and where each one is.
        self.tasks: TaskStore = make_task_store(config, node_id, logger.getChild("taskstore"))
        self.policy.bind_tasks(self.tasks)
        # Hand the policy whole-chunk columns when it implements the column entry points.
        self._columns_place = prefers_columns(
            self.policy, "get_children_tasks", "get_children_tasks_columns"
        )
        self._columns_layout = prefers_columns(
            self.policy, "get_children_resources", "get_children_resources_columns"
        )
        self._event_loop = asyncio.get_event_loop()
        self.cluster.set_event_loop(self._event_loop)

        self.state = SchedulerState(node_id=node_id or "node", level=level)
        # Work for the assigner: handle arrays. Private -- the way in is queue().
        self._input: asyncio.Queue = asyncio.Queue()
        self._assigner: Optional[asyncio.Task] = None
        self._placing = False  # an input item is being placed
        # Handle arrays no child could take; the master fails them.
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

    async def queue(self, update: "TaskUpdate", client_id: Optional[str] = None) -> np.ndarray:
        """Store a received TaskUpdate and queue its tasks for placement.

        Storing happens now, so from here on the store accounts for every task (it is in the
        pool: `UNASSIGNED`); only placement is deferred to the assigner. Returns the handles
        added -- tasks the store already holds (a resend) are left out. Async so that a bounded
        input queue could apply backpressure later.
        """
        idx = self.tasks.add(update, client_id)
        if not len(idx):
            return idx
        self.state.ntasks_added += len(idx)
        self.drained.clear()
        if self.placement == self.PUSH:
            self._input.put_nowait(idx)
        return idx

    def finish_tasks(self, idx, uids: Optional[List[bytes]] = None) -> None:
        """These tasks have a final result: they leave the store (no status is kept: how a
        task ended travels in its result). Unknown, finished and repeated ones are ignored.
        ``uids`` (row for row with ``idx``) saves the store reading them back."""
        self.tasks.release(idx, uids)
        self._check_drained()

    def requeue(self, idx) -> None:
        """Place pool tasks again -- e.g. a removed child's, when it can't come back. Handles
        no longer in the pool (finished, or claimed by a child) are skipped. In pull mode they
        just wait in the pool for the next request."""
        idx = _as_handles(idx)
        idx = idx[self.tasks.assigned(idx) == UNASSIGNED]
        if len(idx) and self.placement == self.PUSH:
            self.drained.clear()
            self._input.put_nowait(idx)

    def queue_result(self, batch, idx: Optional[np.ndarray] = None) -> None:
        """Results arrived (from a child, or made here for tasks no child holds).

        ``idx`` are the batch's handles, row for row, when the caller has them (the batch came
        from `build_failed_results`); otherwise its uids are looked up. Puts the batch -- split
        per destination by row, as bytes -- on the submitting client's outbox, or on
        ``parent_outbox`` for tasks with no client, then finishes the tasks.
        """
        n = batch.nresults
        if not n:
            return
        uids = None
        if idx is None:
            uids = batch.uids()
            idx = self.tasks.lookup(uids) if self.tasks.has_clients() else None
        codes = self.tasks.clients(idx) if self.tasks.has_clients() else None
        self.finish_tasks(idx, uids)
        if codes is None or (codes == NO_CLIENT).all():  # no client tasks: all of it goes up
            self.parent_outbox.put_nowait(batch)
            return
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        bounds = np.flatnonzero(np.diff(sorted_codes)) + 1
        for rows in np.split(order, bounds):
            piece = batch if len(rows) == n else batch.select(rows)
            client = self.tasks.client_name(int(codes[rows[0]]))
            if client is None:
                self.parent_outbox.put_nowait(piece)
                continue
            outbox = self.client_outboxes.get(client)
            if outbox is None:
                outbox = self.client_outboxes[client] = asyncio.Queue()
                self.new_clients.put_nowait(client)
            outbox.put_nowait(piece)

    def delete_tasks(self, uids: List[bytes]) -> None:
        """Remove tasks from the scheduler entirely (they get no result)."""
        self.tasks.release(self.tasks.lookup(uids))
        self._check_drained()

    def delivered(self, idx) -> None:
        """A TaskUpdate carrying these tasks reached its child: their bytes are now needed
        only for recovery, so the store may offload them first."""
        self.tasks.mark_delivered(idx)

    def unfinished(self) -> List[Tuple[Optional[str], np.ndarray]]:
        """Where the unfinished tasks are: ``(child_id, handles)`` per child holding tasks, and
        ``(None, handles)`` for the pool (first) and for tasks of a removed child. Each
        unfinished task appears once."""
        out: List[Tuple[Optional[str], np.ndarray]] = []
        for tag, idx in self.tasks.groups():
            child = self.state.child_of(tag) if tag >= 0 else None
            out.append((child.id if child is not None else None, idx))
        return out

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
            idx = await self._input.get()
            self._placing = True
            try:
                await self._place_item(idx)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.error(f"Placing {len(idx)} tasks failed: {e}", exc_info=True)
            finally:
                self._placing = False
                self._input.task_done()
                self._check_drained()

    async def _place_item(self, idx: np.ndarray) -> None:
        """Place one input item in chunks, yielding between them; then give each child one
        array for the whole item, so one incoming update is one TaskUpdate per child."""
        size = max(1, self._config.assign_chunk_size)
        buffers: Dict[str, List[np.ndarray]] = {}
        unplaced: List[np.ndarray] = []
        assigned = self.tasks.assigned
        for start in range(0, len(idx), size):
            chunk = idx[start : start + size]
            chunk = chunk[assigned(chunk) == UNASSIGNED]
            if len(chunk):
                placed, left = self._place(chunk)
                for child_id, got in placed.items():
                    buffers.setdefault(child_id, []).append(got)
                if len(left):
                    unplaced.append(left)
            await asyncio.sleep(0)
        for child_id, parts in buffers.items():
            child = self.state.children.get(child_id)
            if child is not None:
                child.outbox.put_nowait(parts[0] if len(parts) == 1 else np.concatenate(parts))
        if unplaced:
            self.unplaced.put_nowait(np.concatenate(unplaced))

    def _place(
        self,
        idx: np.ndarray,
        child_ids: Optional[Iterable[str]] = None,
        ntask: Optional[int] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Ask the policy where pool tasks ``idx`` go (all children, or ``child_ids``) and tag
        them in the store.

        Returns ``(child_id -> placed handles, unplaced handles)``. Does not touch outboxes.
        """
        state = self.state
        if child_ids is None:
            targets = state.children.values()
        else:
            targets = [state.children[c] for c in child_ids if c in state.children]
        # A finished child takes nothing more (e.g. a FAILED child's own requeued tasks).
        children = {child.wid: child for child in targets if not child.terminal}
        if not children:
            return {}, idx
        if self._columns_place:
            wids = self.policy.get_children_tasks_columns(
                idx=idx,
                meta=self.tasks.meta_columns(idx),
                children=children,
                ntask=ntask,
                level=state.level,
            )
        else:
            by_wid, _, _ = self.policy.get_children_tasks(
                idx=idx, children=children, ntask=ntask, level=state.level
            )
            wids = wids_from_groups(idx, by_wid)
        wids = np.asarray(wids, dtype=np.int64)
        placed: Dict[str, np.ndarray] = {}
        unplaced: List[np.ndarray] = []
        order = np.argsort(wids, kind="stable")
        bounds = np.flatnonzero(np.diff(wids[order])) + 1
        for rows in np.split(order, bounds):
            if not len(rows):
                continue
            wid = int(wids[rows[0]])
            got = idx[np.sort(rows)]  # in the order given
            child = children.get(wid)
            if child is None:  # -1, or a wid the policy made up
                unplaced.append(got)
                continue
            placed[child.id] = self.tasks.assign(got, wid)
        left = np.concatenate(unplaced) if unplaced else idx[:0]
        return placed, left

    def request(self, child_id: str, ntask: Optional[int]) -> int:
        """Pull mode: a child asked for up to ``ntask`` tasks. Places them from the pool and
        puts them on its outbox as one array. Returns how many were placed."""
        child = self.state.children.get(child_id)
        if child is None or not self.tasks.count(UNASSIGNED):
            return 0
        placed, _ = self._place(self.tasks.assigned_to(UNASSIGNED), child_ids=[child_id], ntask=ntask)
        got = placed.get(child_id)
        if got is None or not len(got):
            return 0
        child.outbox.put_nowait(got)
        return len(got)

    def _check_drained(self) -> None:
        if self._input.empty() and not self._placing and len(self.tasks) == 0:
            self.drained.set()
        else:
            self.drained.clear()

    # ------------------------------------------------------------------
    # Children
    # ------------------------------------------------------------------

    def _reset_child(self, child_id: str) -> None:
        """Forget a child for good: its tasks go back to the pool (not re-placed)."""
        child = self.state.pop_child(child_id)
        if child is not None:
            self.tasks.reassign(child.wid, UNASSIGNED)

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
                self._reset_child(child_id)
            state.all_done.clear()
        state.level = level
        suffix = ".w" if level + 1 == self._config.policy_config.nlevels else ".m"
        wid_offset = state.next_wid()

        idx = self.tasks.all()
        nodes = nodes if nodes is not None else self.cluster.nodes
        if self._columns_layout:
            layout = self.policy.get_children_resources_columns(
                idx=idx, meta=self.tasks.meta_columns(idx), nodes=nodes, level=level
            )
        else:
            layout = self.policy.get_children_resources(idx=idx, nodes=nodes, level=level)
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
        elif child.terminal:
            self._free(child)
            child.done.set()
            # Back to the pool; the child is terminal already, so placement skips it.
            idx = self.tasks.reassign(child.wid, UNASSIGNED)
            if len(idx):
                if requeue:
                    self.requeue(idx)
                else:
                    self.unplaced.put_nowait(idx)
            if self.state.all_terminal():
                self.state.all_done.set()
        return True

    def remove_child(self, child_id: str) -> Optional[Child]:
        """Drop a child for recovery: its resources go back to the cluster, while its
        unfinished tasks keep its wid in the store, for `reinstate_child`. Returns the removed
        record."""
        child = self.state.pop_child(child_id)
        if child is None:
            return None
        self._free(child)
        return child

    def reinstate_child(self, child: Child) -> bool:
        """Put a removed child back on its own resources (READY); its fresh outbox holds all
        its unfinished tasks, to be sent once the new process is ready. If its resources can't
        be re-allocated, its tasks go back to the pool and are placed on the other children,
        and this returns False."""
        assert child.id not in self.state.children, "Remove the child before reinstating"
        allocated, resource = self.cluster.allocate(child.job_resource)
        if not allocated:
            self.requeue(self.tasks.reassign(child.wid, UNASSIGNED))
            return False
        child.job_resource = resource
        child.allocated = True
        child.state = ChildState.NOTREADY
        child.status = None
        child.ready = asyncio.Event()
        child.done = asyncio.Event()
        self.state.add_child(child)
        held = self.tasks.assigned_to(child.wid)
        if len(held):
            child.outbox.put_nowait(held)
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
        """The state as checkpoint JSON (the children layout). Runs on the loop, so it never
        sees a half-made change; the checkpointer only writes the string."""
        self.state.nodes = self.cluster.nodes
        return self.state.to_json()

    def set_state(self, restored: SchedulerState) -> None:
        """Adopt the children of a checkpointed state: same ids, wids and resources, without
        re-running the policy. The checkpoint names no tasks (they come back from where they
        came from: the initial tasks, or the parent's resend), so the assigner places them on
        the restored children like any others. Call before ``start()``. A child whose
        resources can't be re-allocated is skipped.
        """
        state = self.state
        state.level = restored.level
        for child_id in list(state.children):
            self._reset_child(child_id)
        state.wid_counter = max(state.wid_counter, restored.wid_counter)
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
            self.transition(child.id, ChildState.READY)
        self.logger.info(
            f"set_state: restored {len(state.children)} children; "
            f"{len(self.tasks)} tasks to place"
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
            tasks: The worker's TaskStore (task ids below are its handles). A TaskUpdate, an
                iterable of Tasks or a dict of Tasks is put into a fresh in-memory store.
            nodes: Available cluster resources for this worker.
            policy: Policy name or instance used to score/prioritise tasks.
                ``"fifo_policy"`` (the default) is special-cased: tasks go into
                a deque and are never scored. Any other policy scores each task
                into a priority heap.
            policy_config: Configuration passed to the policy constructor
                when *policy* is given as a string.
            node_id: The owning node's id (logs, task store spill directory).
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
        self._columns_score = prefers_columns(
            self.scheduler_policy, "get_score", "get_scores_columns"
        )
        # A task's JobResource depends only on its shape: built once per shape.
        self._req_by_shape: Dict[tuple, JobResource] = {}
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

    def _update_min_req(self, meta) -> None:
        """Fold the column minima of ``meta`` (a ``META_DTYPE`` array) into ``_min_req``."""
        if len(meta) == 0:
            return
        nn, ppn = meta["nnodes"], meta["ppn"]
        shape = (int(nn.min()), int(ppn.min()), float((meta["ngpus_per_process"] * ppn).min()))
        if self._min_req is None:
            self._min_req = shape
        else:
            self._min_req = tuple(map(min, self._min_req, shape))

    def _scores(self, task_ids: List[bytes], meta) -> List[float]:
        """Policy scores for ``task_ids`` (``meta`` = their records), whole batch or per task."""
        policy = self.scheduler_policy
        if self._columns_score:
            return list(policy.get_scores_columns(task_ids, meta, scheduler_state=self.state))
        return [policy.get_score(t, scheduler_state=self.state) for t in task_ids]

    def _enqueue(self, task_ids: List[bytes], meta=None) -> None:
        """The single place tasks enter the pending container.

        FIFO pushes without scoring. Any other policy is scored (`_scores`), and gets the
        live ``self.state`` as its ``scheduler_state``.
        """
        if not task_ids:
            return
        if meta is None:
            meta = self.tasks.meta_columns(task_ids)
        push = self.state.pending_tasks.push
        if self._fifo:
            for task_id in task_ids:
                push(0.0, task_id)
        elif self._columns_score:
            to_priority = self._priority_from_score
            for task_id, score in zip(task_ids, self._scores(task_ids, meta)):
                push(to_priority(score), task_id)
        else:  # score each task right before pushing it, so it sees the ones before it
            get_score, to_priority = self.scheduler_policy.get_score, self._priority_from_score
            for task_id in task_ids:
                push(to_priority(get_score(task_id, scheduler_state=self.state)), task_id)
        self._update_min_req(meta)

    def _requirement(self, task) -> JobResource:
        shape = (task.nnodes, task.ppn, task.ngpus_per_process)
        req = self._req_by_shape.get(shape)
        if req is None:
            req = self._req_by_shape[shape] = task.get_resource_requirements()
        return req

    def _seed_pending(self, task_ids: Iterable[str]) -> int:
        """Replace the pending container with fresh entries for *task_ids*.

        Used when the set of pending tasks itself is being established (construction,
        checkpoint restore). To re-score the tasks already queued without changing
        membership, use :meth:`reprioritize_pending` instead.
        """
        self.state.pending_tasks.clear()
        self._min_req = None
        self._enqueue([task_id for task_id in task_ids if task_id in self.tasks])
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
        task_ids = [t for t in self.state.pending_tasks.task_ids() if t in self.tasks]
        meta = self.tasks.meta_columns(task_ids) if self._columns_score else None
        scores = {
            task_id: self._priority_from_score(score)
            for task_id, score in zip(task_ids, self._scores(task_ids, meta))
        }
        self.state.pending_tasks.rescore(scores)
        return len(scores)

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

                req = self._requirement(task)
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
        # Anything still pending or running means not done.
        if self.state.running_tasks or not self.state.pending_tasks.empty():
            return
        # Finished, rejected and deleted tasks all leave the store, so what it still holds is
        # what is left to do.
        remaining = len(self.tasks)
        self.logger.debug(f"Checking completion: {remaining} tasks remaining")
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
    ) -> Tuple[List[int], List[Tuple[bytes, str]]]:
        """Store a received TaskUpdate (from ``client_id``, or the parent) and queue its tasks.

        Returns ``(accepted handles, rejected (uid, task_id) pairs)``. A task needing more
        nodes than this worker has is rejected (and released) -- the caller reports it by uid,
        as it will be gone from the store. Tasks the store already holds are skipped.
        """
        nnodes = len(self.cluster.nodes.nodes)
        new = self.tasks.add(update, client_id)
        meta = self.tasks.meta_columns(new)
        too_big = meta["nnodes"] > nnodes
        rejected = []
        if too_big.any():
            bad = new[too_big]
            for h, uid in zip(bad.tolist(), self.tasks.uids(bad)):
                task = self.tasks[h]
                self.logger.error(
                    f"Task {task.task_id} requires {task.nnodes} nodes, but only {nnodes} are available"
                )
                rejected.append((uid, task.task_id))
            self.tasks.release(bad)
            new, meta = new[~too_big], meta[~too_big]
        accepted = new.tolist()
        self._enqueue(accepted, meta)
        if accepted:
            self.state.all_done.clear()  # new work: not drained any more
        return accepted, rejected

    def delete_task(self, task_id: int) -> bool:
        """Remove a task (by handle) from all queues and free its resources. Returns True on
        success."""
        if task_id not in self.tasks:
            self.logger.warning(f"Unknown task: {task_id}")
            return False

        task = _TaskRef(task_id)
        try:
            # Drop the task's bytes
            self.tasks.release((task_id,))

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

            # Remove from running tasks (a pending task was never running)
            self.state.running_tasks.pop(task.task_id, None)

            return True
        except Exception as e:
            self.logger.warning(f"Failed to delete task {task.task_id}: {e}")
            return False

    def free(self, task_id: int, status: TaskStatus) -> None:
        """Deallocate resources for a completed task (by handle), count its final status and
        release it from the store."""
        if task_id in self.tasks:
            if task_id not in self.state.running_tasks:
                self.logger.error(f"{task_id} is not running")
                raise RuntimeError

            # deallocate
            self.cluster.deallocate(self.state.running_tasks[task_id])

            # delete from running tasks
            del self.state.running_tasks[task_id]

            # count it as failed/successful
            if status == TaskStatus.FAILED:
                self.state.nfailed += 1
            elif status == TaskStatus.SUCCESS:
                self.state.ncompleted += 1

            self.logger.debug(f"Freed {task_id}")

            # Notify the policy about the completed task (only if it has the hook).
            if (
                type(self.scheduler_policy).on_task_complete
                is not Policy.on_task_complete
            ):
                self.scheduler_policy.on_task_complete(
                    handle=task_id,
                    status=status,
                    scheduler_state=self.state,
                )

            self.tasks.release((task_id,))

        self._check_all_tasks_done()
        return None

    def fail_unfinished(self) -> List[int]:
        """Fail every pending and running task, for a worker that is stopping; returns
        their handles. All of them are released, so read anything needed from the store
        (uids, task ids) before calling. Call after `stop_monitoring`, so nothing is dispatched meanwhile.

        Running tasks (including ones allocated but still in ``ready_tasks``) go through
        `free`; pending ones never ran, so they are only recorded as failed and released.
        """
        while not self.ready_tasks.empty():  # their handles are in running_tasks
            self.ready_tasks.get_nowait()
        running = list(self.state.running_tasks)
        for h in running:
            self.free(h, TaskStatus.FAILED)
        pending = [h for h in self.state.pending_tasks.task_ids() if h in self.tasks]
        self.state.pending_tasks.clear()
        self.state.nfailed += len(pending)
        self.tasks.release(pending)
        self._check_all_tasks_done()
        return running + pending


class _TaskRef:
    """Just the handle of a task, for code paths that used to take a Task only for its id."""

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
