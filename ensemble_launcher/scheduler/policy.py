import copy
import logging
from abc import ABC, abstractmethod
from itertools import accumulate
from logging import Logger
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
)

import numpy as np

from ensemble_launcher.config import PolicyConfig
from ensemble_launcher.ensemble import TaskMeta, TaskStatus
from ensemble_launcher.ensemble.ensemble import resource_requirements
from ensemble_launcher.scheduler.resource import JobResource

if TYPE_CHECKING:
    from ensemble_launcher.scheduler.state import Child, TaskSchedulerState

logger = logging.getLogger(__name__)


def prefers_columns(policy: Any, row_method: str, columns_method: str) -> bool:
    """Whether to call ``policy.<columns_method>`` (whole-chunk columns) instead of
    ``policy.<row_method>`` (uids + ``self.tasks[uid]``): whichever of the two is defined
    furthest down the class's MRO wins. So a plugged-in policy that only implements the row
    method keeps getting it, and a subclass that overrides the row method of a columnar
    built-in is not bypassed by the inherited columns method."""
    for cls in type(policy).__mro__:
        if columns_method in cls.__dict__:
            return True
        if row_method in cls.__dict__:
            return False
    return False


def _shape_index(meta: np.ndarray) -> Tuple[List[tuple], np.ndarray]:
    """Distinct ``(nnodes, ppn, ngpus_per_process)`` shapes as Python values, and each row's
    index into them."""
    nn, ppn, g = meta["nnodes"], meta["ppn"], meta["ngpus_per_process"]
    n = len(meta)
    if n == 0:
        return [], np.empty(0, dtype=np.int64)
    if (nn == nn[0]).all() and (ppn == ppn[0]).all() and (g == g[0]).all():
        shapes = np.array([[nn[0], ppn[0], g[0]]], dtype=np.float64)
        inv = np.zeros(n, dtype=np.int64)
    else:
        stacked = np.stack([nn.astype(np.float64), ppn.astype(np.float64), g], axis=1)
        shapes, inv = np.unique(stacked, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
    out = []
    for a, b, c in shapes.tolist():
        out.append((int(a), int(b), int(c) if float(c).is_integer() else c))
    return out, inv


def _round_robin(n: int, order: List[int], counts: Dict[int, int], ntask: Optional[int]) -> np.ndarray:
    """Children for ``n`` tasks dealt round robin over ``order`` (wids, in rotation order).
    With ``ntask``, a child takes at most ``ntask - counts[wid]`` more; tasks past the total
    capacity get -1. Updates ``counts``. Vectorized: no Python work per task."""
    k = len(order)
    wids = np.asarray(order, dtype=np.int64)
    if k == 0 or n == 0:
        return np.full(n, -1, dtype=np.int64)
    if ntask is None:
        out = wids[np.arange(n) % k]
        per = np.bincount(np.arange(n) % k, minlength=k)
    else:
        caps = np.maximum(0, ntask - np.array([counts[w] for w in order], dtype=np.int64))
        total = int(caps.sum())
        m = min(n, total)
        # Smallest number of full rounds R with sum(min(caps, R)) >= m.
        lo, hi = 0, int(caps.max()) if m else 0
        while lo < hi:
            mid = (lo + hi) // 2
            if int(np.minimum(caps, mid).sum()) >= m:
                hi = mid
            else:
                lo = mid + 1
        R = lo
        per = np.minimum(caps, max(R - 1, 0))
        extra = m - int(per.sum())
        if extra:
            last_round = np.flatnonzero(caps >= R)[:extra]  # in rotation order
            per[last_round] += 1
        # Slot (round r, position j) is dealt in order r * k + j.
        j = np.repeat(np.arange(k), per)
        starts = np.repeat(np.cumsum(per) - per, per)
        r = np.arange(len(j)) - starts
        out = np.full(n, -1, dtype=np.int64)
        out[:m] = wids[j[np.argsort(r * k + j, kind="stable")]]
    for w, c in zip(order, per.tolist()):
        counts[w] += c
    return out


def _group_by_child(
    uids: List[bytes], assign: np.ndarray, worker_ids: List[int]
) -> Tuple[Dict[int, List[bytes]], List[bytes]]:
    """``assign[i]`` is uids[i]'s child (-1: unplaced) -> per-child uid lists (in uid order)
    and the unplaced list."""
    by_wid: Dict[int, List[bytes]] = {wid: [] for wid in worker_ids}
    order = np.argsort(assign, kind="stable")
    sorted_assign = assign[order]
    bounds = np.flatnonzero(np.diff(sorted_assign)) + 1
    unplaced: List[bytes] = []
    for rows in np.split(order, bounds):
        if len(rows) == 0:
            continue
        wid = int(assign[rows[0]])
        got = [uids[i] for i in rows.tolist()]
        if wid < 0:
            unplaced = got
        else:
            by_wid[wid] = got
    return by_wid, unplaced


class PolicyStateMixin:
    """Externally readable/writable key-value state, shared by both policy ABCs.

    The state is a plain ``dict`` that policies read from inside their decision
    methods (``get_score``, ``get_children_tasks``, ...). It is seeded from
    :attr:`PolicyConfig.initial_state` and can be updated at runtime from outside
    the orchestrator through ``PolicyClient``, which lets a policy react to
    *workflow* state and not just to the scheduler state it is handed per call.

    Contract for policy authors: :meth:`set_policy_state` and
    :meth:`on_state_update` run on the orchestrator's event loop and must be
    synchronous and non-blocking. No file I/O, no network, no ``time.sleep``.
    """

    def _init_policy_state(self, initial: Optional[Dict[str, Any]] = None) -> None:
        self._policy_state: Dict[str, Any] = dict(initial) if initial else {}

    @property
    def state(self) -> Dict[str, Any]:
        """The live state dict. Read this from inside decision methods."""
        # Subclasses may forget to call _init_policy_state(); don't make that fatal.
        if not hasattr(self, "_policy_state"):
            self._policy_state = {}
        return self._policy_state

    def get_policy_state(self, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """Return a deep copy of the state, optionally restricted to ``keys``.

        A copy, so a caller (including the endpoint's rollback snapshot) cannot
        mutate policy state by holding on to the returned dict.
        """
        state = self.state
        if keys is None:
            return copy.deepcopy(state)
        return copy.deepcopy({k: state[k] for k in keys if k in state})

    def set_policy_state(
        self, state: Dict[str, Any], merge: bool = True
    ) -> Dict[str, Any]:
        """Update the state and return a deep copy of the result.

        Args:
            state: New keys/values.
            merge: ``True`` updates only the supplied keys; ``False`` replaces
                the whole state.
        """
        if not isinstance(state, dict):
            raise TypeError(f"policy state must be a dict, got {type(state).__name__}")
        current = self.state
        if merge:
            current.update(copy.deepcopy(state))
        else:
            current.clear()
            current.update(copy.deepcopy(state))
        self.on_state_update(copy.deepcopy(state))
        return copy.deepcopy(current)

    def on_state_update(self, changed: Dict[str, Any]) -> None:
        """Hook for recomputing values derived from the state.

        Called after every :meth:`set_policy_state`. MUST be synchronous and
        non-blocking -- see the class docstring.
        """
        pass


class Policy(PolicyStateMixin, ABC):
    """Task scoring policy.

    ``scheduler_state`` (in :meth:`get_score` / :meth:`on_task_complete`) is the task
    scheduler's live `TaskSchedulerState`, passed as is: read it, never modify it.

    Class attributes:
        needs_scheduler_state: No longer read. The state used to be rebuilt per call
            (O(total tasks)) and this let a policy opt out; passing the live state costs
            nothing, so it is kept only so policies that set it keep working.
    """

    needs_scheduler_state: ClassVar[bool] = True

    # The scheduler's tasks, keyed by uid: ``self.tasks[uid]`` is a TaskMeta with every
    # non-deep field. Bound by the scheduler (`bind_tasks`) before any call below.
    tasks: Mapping[bytes, TaskMeta] = {}

    def bind_tasks(self, tasks: Mapping[bytes, TaskMeta]) -> None:
        self.tasks = tasks

    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        logger: Optional[Logger] = None,
    ):
        self.policy_config = policy_config
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._init_policy_state(getattr(policy_config, "initial_state", None))

    @abstractmethod
    def get_score(
        self,
        uid: bytes,
        scheduler_state: Optional["TaskSchedulerState"] = None,
    ) -> float:
        """
        Returns a score for scheduling the given task on the given node.
        Higher score means higher priority.

        Args:
            uid: The task to score; read its fields with ``self.tasks[uid]``.
            scheduler_state: The scheduler's live state (``pending_tasks``,
                ``running_tasks``, ``completed_tasks``, ``failed_tasks``) for
                state-aware prioritisation. Read-only.
        """
        pass

    def get_scores_columns(
        self,
        uids: List[bytes],
        meta: np.ndarray,
        scheduler_state: Optional["TaskSchedulerState"] = None,
    ):
        """Scores for many tasks at once (higher = higher priority), aligned with ``uids``.

        Override this instead of `get_score` to score a whole update with numpy: ``meta`` is
        a ``META_DTYPE`` array aligned with ``uids`` (``meta["nnodes"]``, ...). The scheduler
        calls whichever of the two is defined furthest down the class hierarchy. All tasks of
        one call see the same ``scheduler_state`` (taken before any of them is queued).
        """
        return [self.get_score(uid, scheduler_state=scheduler_state) for uid in uids]

    def on_task_complete(
        self,
        uid: bytes,
        status: TaskStatus,
        scheduler_state: "TaskSchedulerState",
    ) -> None:
        """
        Callback invoked by the task scheduler whenever a task finishes.

        Override this in subclasses to implement custom logic that reacts to
        task completions (e.g. updating internal weights, logging, triggering
        re-prioritisation).

        Args:
            uid: The task that just completed (``self.tasks[uid]``).
            status: Final status of the task (SUCCESS or FAILED).
            scheduler_state: The scheduler's live state, after the task has been
                marked complete and its resources freed. Read-only.
        """


class ChildrenPolicy(PolicyStateMixin, ABC):
    """Lays out a master's children (`get_children_resources`) and places tasks on them
    (`get_children_tasks`).

    Class attributes:
        layout_needs_tasks: ``get_children_resources`` sizes children from the tasks. A
            sub-master using such a policy waits for its parent's first TaskUpdate before
            laying out its children; with the default ``False`` it lays them out at once,
            and tasks arrive afterwards.
    """

    layout_needs_tasks: ClassVar[bool] = False

    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        node_id: Optional[str] = None,
        logger: Optional[Logger] = None,
    ):
        self.policy_config = policy_config
        self.node_id = node_id
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self._init_policy_state(getattr(policy_config, "initial_state", None))

    # The scheduler's tasks, keyed by uid (see `Policy.tasks`).
    tasks: Mapping[bytes, TaskMeta] = {}

    def bind_tasks(self, tasks: Mapping[bytes, TaskMeta]) -> None:
        self.tasks = tasks

    @abstractmethod
    def get_children_resources(
        self, uids: List[bytes], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        """
        Decide how many child workers to create and what cluster resources each gets.

        Implementations may use the tasks to size the worker layout (e.g. number of
        nodes per worker based on task requirements) but must NOT assign tasks here.

        Args:
            uids: The tasks to consider; read fields with ``self.tasks[uid]``.
            nodes: JobResource containing the available nodes and per-node resources.
            level: Current hierarchy level (0 = root master).

        Returns:
            Dict mapping integer worker IDs to the JobResource allocated to each worker.
            An empty dict is valid when no workers can be created.
        """
        pass

    @abstractmethod
    def get_children_tasks(
        self,
        uids: List[bytes],
        children: Dict[int, "Child"],
        ntask: Optional[int] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[bytes]], Dict[bytes, int], List[bytes]]:
        """
        Distribute tasks across workers given their pre-allocated resources.

        Args:
            uids: The tasks to assign; read fields with ``self.tasks[uid]``.
            children: The children to choose from, by wid. Each `Child` has
                ``.job_resource`` (its allocated resources), ``.uids`` / ``len(child)``
                (its unfinished tasks), ``.status`` (its last Status message) and ``.state``.
            ntask: Maximum tasks to assign per worker.  ``None`` means no limit.
            level: Optional current hierarchy level (0 = root master).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Tuple of:
            - Dict mapping wid (int) to the list of uids assigned to them.
            - Dict mapping uid to the wid (int) it is assigned to
            - List of uids that could not be assigned to any worker.
        """
        pass

    # ---- Column entry points ------------------------------------------------------------
    # Override these to get a whole chunk at once: ``meta`` is a ``META_DTYPE`` structured
    # array aligned with ``uids`` (``meta["nnodes"]``, ``meta["ppn"]``, ...), so a policy can
    # work with numpy instead of reading ``self.tasks[uid]`` field by field (which is slow at
    # scale). The scheduler calls whichever of the pair is defined furthest down the class
    # hierarchy (`prefers_columns`). Same arguments and return values as the row methods.

    def get_children_resources_columns(
        self, uids: List[bytes], meta: np.ndarray, nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        return self.get_children_resources(uids, nodes, level)

    def get_children_tasks_columns(
        self,
        uids: List[bytes],
        meta: np.ndarray,
        children: Dict[int, "Child"],
        ntask: Optional[int] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[bytes]], Dict[bytes, int], List[bytes]]:
        return self.get_children_tasks(uids, children, ntask=ntask, level=level, **kwargs)


class PolicyRegistry:
    def __init__(self):
        self.available_policies: Dict[str, Type[Policy]] = {}
        self.available_children_policies: Dict[str, Type[ChildrenPolicy]] = {}

    def register(self, policy_name: str, type: str = "policy"):
        """Register a policy class by name.

        Args:
            policy_name: Name to register the policy under
            type: Either "policy" or "children_policy"
        """

        def decorator(cls: Type[Policy]):
            if type == "children_policy":
                self.available_children_policies[policy_name] = cls
            else:
                self.available_policies[policy_name] = cls
            return cls

        return decorator

    def register_policy(
        self, policy_name: str, policy_class: Type[Policy], type: str = "policy"
    ):
        """Programmatically register a policy class.

        Args:
            policy_name: Name to register the policy under
            policy_class: The policy class to register
            type: Either "policy" for task scoring policies or "children_policy" for children assignment policies
        """
        if type == "children_policy":
            self.available_children_policies[policy_name] = policy_class
        else:
            self.available_policies[policy_name] = policy_class

    def create_policy(
        self, policy_name: str, policy_args: Tuple = (), policy_kwargs: Dict = {}
    ) -> Policy:
        """Create a policy instance.

        Args:
            policy_name: Name of the registered policy
            policy_args: Positional arguments to pass to the policy constructor
            policy_kwargs: Keyword arguments to pass to the policy constructor

        Returns:
            Policy instance with default configuration
        """
        if policy_name in self.available_policies:
            return self.available_policies[policy_name](*policy_args, **policy_kwargs)
        elif policy_name in self.available_children_policies:
            return self.available_children_policies[policy_name](
                *policy_args, **policy_kwargs
            )
        else:
            logger.error(
                f"{policy_name} not available. Available policy names {list(self.available_policies.keys()) + list(self.available_children_policies.keys())}"
            )
            raise ValueError(f"Unknown policy: {policy_name}")


policy_registry = PolicyRegistry()


@policy_registry.register("large_resource_policy")
class LargeResourcePolicy(Policy):
    """
    A simple policy that always prioritizes a larger task.
    The task that uses gpus is given more priority.

    Configuration (class variables):
        cpu_weight: Weight for CPU resources (default: 1.0)
        gpu_weight: Weight for GPU resources (default: 2.0)

    To customize, subclass and override:
        class MyPolicy(LargeResourcePolicy):
            cpu_weight = 2.0
            gpu_weight = 5.0
    """

    cpu_weight: float = 1.0
    gpu_weight: float = 2.0
    needs_scheduler_state: ClassVar[bool] = False

    def __init__(
        self, policy_config: PolicyConfig = PolicyConfig(), logger: Logger = None
    ):
        super().__init__(policy_config, logger)

    def get_score(
        self, uid: bytes, scheduler_state: Optional["TaskSchedulerState"] = None
    ) -> float:
        task = self.tasks[uid]
        return (
            task.nnodes
            * task.ppn
            * (task.ngpus_per_process * self.gpu_weight + self.cpu_weight)
        )

    def get_scores_columns(
        self,
        uids: List[bytes],
        meta: np.ndarray,
        scheduler_state: Optional["TaskSchedulerState"] = None,
    ):
        scores = (
            meta["nnodes"].astype(np.float64)
            * meta["ppn"]
            * (meta["ngpus_per_process"] * self.gpu_weight + self.cpu_weight)
        )
        return scores.tolist()


@policy_registry.register("fifo_policy")
class FIFOPolicy(Policy):
    """Schedules tasks in submission order (FIFO). The default task policy.

    ``AsyncTaskScheduler`` special-cases this policy by its registered name
    ``"fifo_policy"``: pending tasks go into a plain deque and ``get_score`` is
    never called. With ``strict_priority=False`` (the default) a head task that
    does not fit is skipped and keeps its place, letting smaller tasks backfill.

    ``get_score`` is kept for the legacy sync ``WorkerScheduler``: the negative
    pending-task count, so earlier-added tasks score higher.
    """

    # Only scored on the heap path (e.g. a subclass): each new task scores lower than
    # the ones already pending, and ties break on sequence number, so order stays FIFO.
    needs_scheduler_state: ClassVar[bool] = False

    def __init__(
        self, policy_config: PolicyConfig = PolicyConfig(), logger: Logger = None
    ):
        super().__init__(policy_config, logger)

    def get_score(
        self, uid: bytes, scheduler_state: Optional["TaskSchedulerState"] = None
    ) -> float:
        pending_size = (
            len(scheduler_state.pending_tasks)
            if scheduler_state and scheduler_state.pending_tasks
            else 0
        )
        return -pending_size


@policy_registry.register("bin_packing_children_policy", type="children_policy")
class BinPackingChildrenPolicy(ChildrenPolicy):
    """
    A children policy that assigns workers to fit all tasks using bin-packing.
    Tasks are sorted by decreasing node requirements and distributed across workers.

    Configuration (class variables):
        nlevels: Total number of hierarchy levels (default: 1, must be >= 1)

    To customize, subclass and override:
        class MyGreedyPolicy(GreedyBinPackingChildrenPolicy):
            nlevels = 3
    """

    layout_needs_tasks: ClassVar[bool] = True  # sizes children from task node counts

    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        node_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(policy_config, node_id, logger)
        nlevels = policy_config.nlevels
        if nlevels is None:
            raise ValueError("nlevels must be specified for BinPackingChildrenPolicy")
        self.nlevels = nlevels
        self.logger.info(
            f"Initialized BinPackingChildrenPolicy with nlevels={self.nlevels}"
        )

    def get_children_resources(
        self, uids: List[bytes], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        """
        Determine the number of workers and distribute nodes among them.

        Uses task node-requirements (sorted descending) to size each worker's
        node allocation.  Tasks that require more nodes than are available are
        simply skipped for sizing purposes — they will be caught as unassignable
        in ``get_children_tasks``.
        """
        return self._layout([self.tasks[uid].nnodes for uid in uids], nodes, level)

    def get_children_resources_columns(
        self, uids: List[bytes], meta: np.ndarray, nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        return self._layout(meta["nnodes"], nodes, level)

    def _layout(self, task_nnodes, nodes: JobResource, level: int) -> Dict[int, JobResource]:
        if len(task_nnodes) == 0:
            self.logger.error("Bin packing policy needs tasks")
            raise ValueError("Needs Tasks for creating workers")

        node_names = nodes.nodes
        node_resources = {name: res for name, res in zip(nodes.nodes, nodes.resources)}

        # Task node requirements, decreasing; drop tasks that cannot fit.
        sizes = np.sort(np.asarray(task_nnodes, dtype=np.int64))[::-1]
        fitting = sizes[sizes <= len(node_names)]

        # Determine the maximum number of workers that fit given cumulative node use.
        nworkers_max = int((np.cumsum(fitting) <= len(node_names)).sum())

        # Interpolate in log2 space to pick a number of workers for this level.
        if self.nlevels > 1:
            x_vals = np.array([0, self.nlevels], dtype=float)
            y_vals = np.array([0, np.log2(max(nworkers_max, 1))], dtype=float)
            log2_nworkers = int(np.interp(level + 1, x_vals, y_vals))
            nworkers = max(1, min(int(2**log2_nworkers), nworkers_max))
        else:
            nworkers = nworkers_max

        if nworkers == 0:
            return {}

        # Single-node case: all workers share the one available node.
        if len(node_names) == 1:
            return {
                wid: JobResource(
                    resources=[node_resources[node_names[0]]], nodes=node_names
                )
                for wid in range(nworkers)
            }

        if len(node_names) < nworkers:
            self.logger.error("number of nodes < number of children")
            raise RuntimeError

        # Distribute nodes based on the first nworkers tasks' node requirements.
        nnodes_per_worker = fitting[:nworkers].tolist()
        # Spread any leftover nodes round-robin.
        remaining = len(node_names) - sum(nnodes_per_worker)
        for i in range(remaining):
            nnodes_per_worker[i % nworkers] += 1
        cum_nodes = list(accumulate(nnodes_per_worker))

        result: Dict[int, JobResource] = {}
        for wid in range(nworkers):
            start = cum_nodes[wid - 1] if wid > 0 else 0
            end = cum_nodes[wid]
            worker_nodes = node_names[start:end]
            worker_resources = [node_resources[n] for n in worker_nodes]
            result[wid] = JobResource(resources=worker_resources, nodes=worker_nodes)

        return result

    def get_children_tasks(
        self,
        uids: List[bytes],
        children: Dict[int, "Child"],
        ntask: int = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[bytes]], Dict[bytes, int], List[bytes]]:
        """
        Assign tasks to workers in round-robin order (largest tasks first).

        Tasks that exceed every worker's node count are added to the removed list.
        """
        worker_ids = list(children.keys())
        nworkers = len(worker_ids)
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []
        worker_task_counts = {wid: 0 for wid in worker_ids}

        sorted_task_items = sorted(
            ((uid, self.tasks[uid].nnodes) for uid in uids), key=lambda x: x[1], reverse=True
        )

        for i, (task_id, _) in enumerate(sorted_task_items):
            preferred = worker_ids[i % nworkers]

            # Check ntask limit on preferred worker, try others in round-robin.
            if ntask is not None and worker_task_counts[preferred] >= ntask:
                assigned = False
                for attempt in range(nworkers):
                    candidate = worker_ids[(i + attempt) % nworkers]
                    if worker_task_counts[candidate] < ntask:
                        wid_to_task_id_map[candidate].append(task_id)
                        task_id_to_wid_map[task_id] = candidate
                        worker_task_counts[candidate] += 1
                        assigned = True
                        break
                if not assigned:
                    removed_tasks.append(task_id)
            else:
                wid_to_task_id_map[preferred].append(task_id)
                task_id_to_wid_map[task_id] = preferred
                worker_task_counts[preferred] += 1

        return wid_to_task_id_map, task_id_to_wid_map, removed_tasks

    def get_children_tasks_columns(
        self,
        uids: List[bytes],
        meta: np.ndarray,
        children: Dict[int, "Child"],
        ntask: Optional[int] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[bytes]], Dict[bytes, int], List[bytes]]:
        """`get_children_tasks` on columns: same assignment. Largest first; the i-th task
        goes to child ``i % k``; with ``ntask`` (every child starts the call at 0) the first
        ``ntask * k`` are dealt that way and the rest are not placed."""
        worker_ids = list(children.keys())
        k = len(worker_ids)
        n = len(uids)
        by_wid: Dict[int, List[bytes]] = {wid: [] for wid in worker_ids}
        if n == 0:
            return by_wid, {}, []
        if k == 0:
            return by_wid, {}, list(uids)
        order = np.argsort(-meta["nnodes"].astype(np.int64), kind="stable").tolist()
        m = n if ntask is None else min(n, ntask * k)
        for j, wid in enumerate(worker_ids):
            by_wid[wid] = [uids[i] for i in order[j:m:k]]
        return by_wid, {}, [uids[i] for i in order[m:]]


@policy_registry.register("simple_split_children_policy", type="children_policy")
class SimpleSplitChildrenPolicy(ChildrenPolicy):
    """
    A children policy that splits nodes evenly among a specified number of children,
    then assigns tasks in round-robin fashion.

    Tasks are assigned to workers in round-robin order, checking if each task fits
    within the worker's resources. Tasks with CPU or GPU affinity set will be skipped
    and added to the removed tasks list.

    Configuration (class variables):
        nchildren: Number of child workers (must be > 0)

    Note: Not registered by default. To use:
        1. Subclass and set nchildren:
            class Split8Policy(SimpleSplitChildrenPolicy):
                nchildren = 8
        2. Register it:
            policy_registry.register_policy("split_8", Split8Policy, type="children_policy")
        3. Use it:
            scheduler = AsyncChildrenScheduler(..., policy="split_8")
    """

    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        node_id: str = None,
        logger: logging.Logger = None,
        **kwargs,
    ):
        super().__init__(policy_config, node_id, logger)
        self.nchildren = policy_config.nchildren
        if self.nchildren is None or self.nchildren <= 0:
            raise ValueError(f"nchildren must be positive, got {self.nchildren}")
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.logger.info(
            f"Initialized SimpleSplitChildrenPolicy with nchildren={self.nchildren}"
        )

    def get_children_resources(
        self, uids: List[bytes], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        """
        Split nodes evenly among ``nchildren`` workers.

        When ``nchildren > nnodes`` each node's resources are subdivided
        (``nchildren`` must be an exact multiple of the node count).
        """
        nchildren = self.nchildren
        nnodes = len(nodes.nodes)
        result: Dict[int, JobResource] = {}

        if nchildren <= nnodes:
            base = nnodes // nchildren
            remainder = nnodes % nchildren
            start = 0
            for wid in range(nchildren):
                count = base + (1 if wid < remainder else 0)
                end = start + count
                result[wid] = JobResource(
                    resources=nodes.resources[start:end],
                    nodes=nodes.nodes[start:end],
                )
                start = end
        else:
            if nchildren % nnodes != 0:
                raise ValueError(
                    f"nchildren ({nchildren}) must be a multiple of nnodes ({nnodes})"
                )
            splits = nchildren // nnodes
            wid = 0
            for node_name, node_resource in zip(nodes.nodes, nodes.resources):
                for part in node_resource.divide(splits):
                    result[wid] = JobResource(resources=[part], nodes=[node_name])
                    wid += 1

        return result

    def get_children_tasks(
        self,
        uids: List[bytes],
        children: Dict[int, "Child"],
        ntask: int = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[bytes]], Dict[bytes, int], List[bytes]]:
        """
        Assign tasks to workers in round-robin order, checking resource fit.

        Tasks that do not fit in any worker are added to the removed list.
        """
        worker_ids = list(children.keys())
        nchildren = len(worker_ids)
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []
        worker_task_counts = {wid: 0 for wid in worker_ids}
        current = worker_ids.index(
            min(
                worker_ids,
                key=lambda worker_id: len(children[worker_id]),
            )
        )

        # Tasks of the same shape need the same resources: build each shape's JobResource once
        # rather than once per task.
        # Tasks of the same shape need the same resources, and a child's capacity is fixed,
        # so both the requirement and "does it fit child wid" are computed once per shape.
        requirements: Dict[tuple, JobResource] = {}
        fits: Dict[tuple, bool] = {}
        for task_id in uids:
            task = self.tasks[task_id]
            shape = (task.nnodes, task.ppn, task.ngpus_per_process)  # fmt: skip
            task_resource = requirements.get(shape)
            if task_resource is None:
                task_resource = requirements[shape] = task.get_resource_requirements()
            assigned = False

            for attempt in range(nchildren):
                wid = worker_ids[(current + attempt) % nchildren]

                if ntask is not None and worker_task_counts[wid] >= ntask:
                    continue

                fit = fits.get((shape, wid))
                if fit is None:
                    fit = fits[(shape, wid)] = task_resource in children[wid].job_resource
                if fit:
                    wid_to_task_id_map[wid].append(task_id)
                    task_id_to_wid_map[task_id] = wid
                    worker_task_counts[wid] += 1
                    current = (worker_ids.index(wid) + 1) % nchildren
                    assigned = True
                    break

            if not assigned:
                removed_tasks.append(task_id)
                # self.logger.warning(f"Task {task_id} does not fit in any worker")

        return wid_to_task_id_map, task_id_to_wid_map, removed_tasks

    def get_children_tasks_columns(
        self,
        uids: List[bytes],
        meta: np.ndarray,
        children: Dict[int, "Child"],
        ntask: Optional[int] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[bytes]], Dict[bytes, int], List[bytes]]:
        """`get_children_tasks` on columns. Round robin per task shape, starting at the
        least-loaded child, over the children that shape fits; the pointer carries over
        from one shape to the next. A single-shape chunk is placed exactly as the row
        version places it; a mixed one is spread as evenly, but not interleaved the same way.
        """
        worker_ids = list(children.keys())
        nchildren = len(worker_ids)
        n = len(uids)
        if n == 0 or nchildren == 0:
            return {wid: [] for wid in worker_ids}, {}, list(uids)
        pointer = worker_ids.index(min(worker_ids, key=lambda wid: len(children[wid])))
        counts = {wid: 0 for wid in worker_ids}
        assign = np.full(n, -1, dtype=np.int64)
        shapes, inv = _shape_index(meta)
        for s, (nnodes, ppn, ngpus) in enumerate(shapes):
            rows = np.flatnonzero(inv == s) if len(shapes) > 1 else np.arange(n)
            req = resource_requirements(nnodes, ppn, ngpus)
            rotation = worker_ids[pointer:] + worker_ids[:pointer]
            fit = [wid for wid in rotation if req in children[wid].job_resource]
            if not fit:
                continue
            got = _round_robin(len(rows), fit, counts, ntask)
            assign[rows] = got
            placed = got[got >= 0]
            if len(placed):
                pointer = (worker_ids.index(int(placed[-1])) + 1) % nchildren
        by_wid, unplaced = _group_by_child(uids, assign, worker_ids)
        return by_wid, {}, unplaced


@policy_registry.register("fixed_leafs_children_policy", type="children_policy")
class FixedLeafNodePolicy(SimpleSplitChildrenPolicy):
    def __init__(
        self,
        policy_config: PolicyConfig | None = None,
        node_id: str = None,
        logger: Logger = None,
    ):
        policy_config = policy_config or PolicyConfig()
        super().__init__(policy_config, node_id, logger)
        self.logger.info(f"Using FixedLeadNodePolicy")

    def get_children_resources(
        self, uids: List[bytes], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        # Interpolate in log2 space to pick a number of workers for this level.
        nlevels = self.policy_config.nlevels
        leaf_nodes = self.policy_config.leaf_nodes

        x_vals = [0.0, float(nlevels)]
        y_vals = [0.0, max(np.log2(leaf_nodes), 0)]
        
        if level == 0:
            nchildren_parent_level = 0
        else:
            nchildren_parent_level = 2 ** (int(np.interp([level-1], x_vals, y_vals)[0]))

        if level == 0:
            nchildren_current_level = 1
        else:
            nchildren_current_level = 2 ** (int(np.interp([level], x_vals, y_vals)[0]))

        if self.policy_config.nlevels == level + 1:
            nchildren_next_level = self.policy_config.leaf_nodes
        else:
            nchildren_next_level = 2 ** (int(np.interp([level + 1], x_vals, y_vals)[0]))


        if level > 0:
            my_local_id = int(self.node_id.split(".")[-1].replace("m", ""))
            if level > 1:
                my_parent_id = int(self.node_id.split(".")[-2].replace("m", ""))
                nchildren_per_parent = nchildren_current_level // nchildren_parent_level
                my_global_id = nchildren_per_parent * my_parent_id + my_local_id
            else:
                my_global_id = my_local_id
        else:
            my_global_id = 0

        n_children = int(nchildren_next_level // nchildren_current_level)
        remainder = int(nchildren_next_level % nchildren_current_level)
        if my_global_id < remainder:
            n_children += 1

        nnodes = len(nodes.nodes)
        result = {}
        if n_children <= nnodes:
            base = nnodes // n_children
            remainder = nnodes % n_children
            start = 0
            for wid in range(n_children):
                count = base + (1 if wid < remainder else 0)
                end = start + count
                result[wid] = JobResource(
                    resources=nodes.resources[start:end],
                    nodes=nodes.nodes[start:end],
                )
                start = end
        else:
            if n_children % nnodes != 0:
                raise ValueError(
                    f"nchildren ({n_children}) must be a multiple of nnodes ({nnodes})"
                )
            splits = n_children // nnodes
            wid = 0
            for node_name, node_resource in zip(nodes.nodes, nodes.resources):
                for part in node_resource.divide(splits):
                    result[wid] = JobResource(resources=[part], nodes=[node_name])
                    wid += 1

        return result
