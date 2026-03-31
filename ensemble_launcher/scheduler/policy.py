import logging
from abc import ABC, abstractmethod
from itertools import accumulate
from logging import Logger
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Type

import numpy as np

from ensemble_launcher.config import PolicyConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResource,
    NodeResourceCount,
)

if TYPE_CHECKING:
    from ensemble_launcher.comm.messages import Status
    from ensemble_launcher.scheduler.state import ChildrenAssignment, SchedulerState

logger = logging.getLogger(__name__)


class Policy(ABC):
    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        logger: Logger = None,
    ):
        self.policy_config = policy_config
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    @abstractmethod
    def get_score(
        self,
        task: Task,
        scheduler_state: Optional["SchedulerState"] = None,
    ) -> float:
        """
        Returns a score for scheduling the given task on the given node.
        Higher score means higher priority.

        Args:
            task: The task to score.
            scheduler_state: Optional snapshot of the current scheduler state
                (pending, running, completed, and failed task sets) that
                implementations can use for state-aware prioritisation.
        """
        pass

    def on_task_complete(
        self,
        task: Task,
        status: TaskStatus,
        scheduler_state: "SchedulerState",
    ) -> None:
        """
        Callback invoked by the task scheduler whenever a task finishes.

        Override this in subclasses to implement custom logic that reacts to
        task completions (e.g. updating internal weights, logging, triggering
        re-prioritisation).

        Args:
            task: The task that just completed.
            status: Final status of the task (SUCCESS or FAILED).
            scheduler_state: Snapshot of the scheduler state after the task
                has been marked complete and its resources freed.
        """
        pass


class ChildrenPolicy(ABC):
    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        node_id: str = None,
        logger: Logger = None,
    ):
        self.policy_config = policy_config
        self.node_id = node_id
        self.logger = logger if logger is not None else logging.getLogger(__name__)

    @abstractmethod
    def get_children_resources(
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        """
        Decide how many child workers to create and what cluster resources each gets.

        Implementations may use ``tasks`` to size the worker layout (e.g. number of
        nodes per worker based on task requirements) but must NOT assign tasks here.

        Args:
            tasks: Dictionary mapping task IDs to Task objects.
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
        tasks: Dict[str, Task],
        children_resources: Dict[int, JobResource],
        ntask: int = None,
        child_assignments: Optional[Dict[int, "ChildrenAssignment"]] = None,
        child_status: Optional[Dict[int, "Status"]] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[str]], Dict[str, int], List[str]]:
        """
        Distribute tasks across workers given their pre-allocated resources.

        Args:
            tasks: Dictionary mapping task IDs to Task objects.
            children_resources: Dict mapping wid (int) to their JobResource
                (as returned by ``get_children_resources``).
            ntask: Maximum tasks to assign per worker.  ``None`` means no limit.
            child_assignments: Full assignment state per wid, including
                already-assigned task_ids and resource info.
            child_status: Most recent Status message per wid.
            level: Optional current hierarchy level (0 = root master).
            **kwargs: Additional implementation-specific arguments.

        Returns:
            Tuple of:
            - Dict mapping wid (int) to the list of task IDs assigned to them.
            - Dict mapping task id to the wid (int) it is assigned to
            - List of task IDs that could not be assigned to any worker.
        """
        pass


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

    def __init__(
        self, policy_config: PolicyConfig = PolicyConfig(), logger: Logger = None
    ):
        super().__init__(policy_config, logger)

    def get_score(
        self, task: Task, scheduler_state: Optional["SchedulerState"] = None
    ) -> float:
        return (
            task.nnodes
            * task.ppn
            * (task.ngpus_per_process * self.gpu_weight + self.cpu_weight)
        )


@policy_registry.register("fifo_policy")
class FIFOPolicy(Policy):
    """Schedules tasks in submission order (FIFO).

    Score is the negative of the current pending-task count so that
    earlier-added tasks always have a higher score than later ones.
    Best used with ``strict_priority=True`` in PolicyConfig.
    """

    def __init__(
        self, policy_config: PolicyConfig = PolicyConfig(), logger: Logger = None
    ):
        super().__init__(policy_config, logger)

    def get_score(
        self, task: Task, scheduler_state: Optional["SchedulerState"] = None
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

    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        node_id: str = None,
        logger: logging.Logger = None,
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
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        """
        Determine the number of workers and distribute nodes among them.

        Uses task node-requirements (sorted descending) to size each worker's
        node allocation.  Tasks that require more nodes than are available are
        simply skipped for sizing purposes — they will be caught as unassignable
        in ``get_children_tasks``.
        """
        if len(tasks) == 0:
            self.logger.error("Bin packing policy needs tasks")
            raise ValueError("Needs Tasks for creating workers")

        node_names = nodes.nodes
        node_resources = {name: res for name, res in zip(nodes.nodes, nodes.resources)}

        # Sort tasks by decreasing node requirement; drop tasks that cannot fit.
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].nnodes, reverse=True)
        fitting_tasks = [
            (tid, t) for tid, t in sorted_tasks if t.nnodes <= len(node_names)
        ]

        # Determine the maximum number of workers that fit given cumulative node use.
        cum_sum = list(accumulate(t.nnodes for _, t in fitting_tasks))
        nworkers_max = sum(1 for c in cum_sum if c <= len(node_names))

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
        nnodes_per_worker = [t.nnodes for _, t in fitting_tasks[:nworkers]]
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
        tasks: Dict[str, Task],
        children_resources: Dict[int, JobResource],
        ntask: int = None,
        child_assignments: Optional[Dict[int, "ChildrenAssignment"]] = None,
        child_status: Optional[Dict[int, "Status"]] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[str]], Dict[str, int], List[str]]:
        """
        Assign tasks to workers in round-robin order (largest tasks first).

        Tasks that exceed every worker's node count are added to the removed list.
        """
        worker_ids = list(children_resources.keys())
        nworkers = len(worker_ids)
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []
        worker_task_counts = {wid: 0 for wid in worker_ids}

        sorted_task_items = sorted(
            tasks.items(), key=lambda x: x[1].nnodes, reverse=True
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
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
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
        tasks: Dict[str, Task],
        children_resources: Dict[int, JobResource],
        ntask: int = None,
        child_assignments: Optional[Dict[int, "ChildrenAssignment"]] = None,
        child_status: Optional[Dict[int, "Status"]] = None,
        level: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Dict[int, List[str]], Dict[str, int], List[str]]:
        """
        Assign tasks to workers in round-robin order, checking resource fit.

        Tasks that do not fit in any worker are added to the removed list.
        """
        worker_ids = list(children_resources.keys())
        nchildren = len(worker_ids)
        wid_to_task_id_map: Dict[int, List[str]] = {wid: [] for wid in worker_ids}
        task_id_to_wid_map: Dict[str, int] = {}
        removed_tasks: List[str] = []
        worker_task_counts = {wid: 0 for wid in worker_ids}
        current = worker_ids.index(
            min(
                worker_ids,
                key=lambda worker_id: len(child_assignments[worker_id]["task_ids"]),
            )
        )

        for task_id, task in tasks.items():
            task_resource = task.get_resource_requirements()
            assigned = False

            for attempt in range(nchildren):
                wid = worker_ids[(current + attempt) % nchildren]

                if ntask is not None and worker_task_counts[wid] >= ntask:
                    continue

                if task_resource in children_resources[wid]:
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


@policy_registry.register("fixed_leafs_children_policy", type="children_policy")
class FixedLeafNodePolicy(SimpleSplitChildrenPolicy):
    def __init__(
        self,
        policy_config: PolicyConfig = PolicyConfig(),
        node_id: str = None,
        logger: Logger = None,
    ):
        super().__init__(policy_config, node_id, logger)
        self.logger.info(f"Using FixedLeadNodePolicy")

    def get_children_resources(
        self, tasks: Dict[str, Task], nodes: JobResource, level: int
    ) -> Dict[int, JobResource]:
        # Interpolate in log2 space to pick a number of workers for this level.
        nlevels = self.policy_config.nlevels
        leaf_nodes = self.policy_config.leaf_nodes

        x_vals = [0.0, float(nlevels)]
        y_vals = [0.0, max(np.log2(leaf_nodes), 0)]
        nchildren_current_level = 2 ** (np.ceil(np.interp([level], x_vals, y_vals)[0]))
        nchildren_next_level = 2 ** (np.ceil(np.interp([level + 1], x_vals, y_vals)[0]))

        if level > 0:
            my_id = int(self.node_id.split(".")[-1].replace("m", ""))
        else:
            my_id = 0

        n_children = int(nchildren_next_level // nchildren_current_level)
        remainder = int(nchildren_next_level % nchildren_current_level)
        if my_id < remainder:
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
