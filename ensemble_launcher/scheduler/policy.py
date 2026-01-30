from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import NodeResource, JobResource, NodeResourceCount
from abc import ABC, abstractmethod
from typing import Dict, Type, Tuple, List
import logging
from itertools import accumulate
import numpy as np

logger = logging.getLogger(__name__)

class Policy(ABC):
    @abstractmethod
    def get_score(self, task: Task) -> float:
        """
        Returns a score for scheduling the given task on the given node.
        Higher score means higher priority.
        """
        pass

class WorkerPolicy(ABC):
    @abstractmethod
    def get_worker_assignment(self,
                            tasks: Dict[str, Task],
                            nodes: JobResource,
                            level: int) -> Tuple[Dict, List]:
        """
        Returns a tuple of a dictionary and a list:
        - The first dictionary maps worker IDs to their assigned resources:
          {worker_id: {"job_resource": JobResource, "task_ids": [...]}}
        - The second list contains unassigned task IDs.
        
        Args:
            tasks: Dictionary mapping task IDs to Task objects
            nodes: JobResource containing available nodes and their resources
            level: Current level in the hierarchy (runtime parameter)
        """
        pass

class PolicyRegistry:

    def __init__(self):
        self.available_policies: Dict[str, Type[Policy]] = {}
        self.available_worker_policies: Dict[str, Type[WorkerPolicy]] = {}
    
    def register(self, policy_name: str, type: str = "policy"):
        """Register a policy class by name.
        
        Args:
            policy_name: Name to register the policy under
            type: Either "policy" or "worker_policy"
        """
        def decorator(cls: Type[Policy]):
            if type == "worker_policy":
                self.available_worker_policies[policy_name] = cls
            else:
                self.available_policies[policy_name] = cls
            return cls
        return decorator
    
    def register_policy(self, policy_name: str, policy_class: Type[Policy], 
                       type: str = "policy"):
        """Programmatically register a policy class.
        
        Args:
            policy_name: Name to register the policy under
            policy_class: The policy class to register
            type: Either "policy" for task scoring policies or "worker_policy" for worker assignment policies
        """
        if type == "worker_policy":
            self.available_worker_policies[policy_name] = policy_class
        else:
            self.available_policies[policy_name] = policy_class
    
    def create_policy(self, policy_name: str, policy_args: Tuple = (), policy_kwargs: Dict = {}) -> Policy:
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
        elif policy_name in self.available_worker_policies:
            return self.available_worker_policies[policy_name](*policy_args, **policy_kwargs)
        else:
            logger.error(f"{policy_name} not available. Available policy names {list(self.available_policies.keys()) + list(self.available_worker_policies.keys())}")
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

    def get_score(self, task: Task) -> float:
        return task.nnodes * task.ppn * (task.ngpus_per_process * self.gpu_weight +
                                         self.cpu_weight)


@policy_registry.register("greedy_worker_policy", type="worker_policy")
class GreedyBinPackingWorkerPolicy(WorkerPolicy):
    """
    A worker policy that greedily assigns workers to fit all tasks using bin-packing.
    Tasks are sorted by decreasing node requirements and distributed across workers.
    
    Configuration (class variables):
        nlevels: Total number of hierarchy levels (default: 1, must be >= 1)
    
    To customize, subclass and override:
        class MyGreedyPolicy(GreedyBinPackingWorkerPolicy):
            nlevels = 3
    """
    def __init__(self, nlevels: int = None, logger: logging.Logger = None, **kwargs):
        self.logger = logging.getLogger(__name__) if logger is None else logger
        if nlevels is None:
            raise ValueError("nlevels must be specified for GreedyBinPackingWorkerPolicy")
        self.nlevels = nlevels
        self.logger.info(f"Initialized GreedyBinPackingWorkerPolicy with nlevels={self.nlevels}")
    
    def get_worker_assignment(self,
                        tasks: Dict[str, Task],
                        nodes: JobResource,
                        level: int) -> Tuple[Dict, List]:
        """
        Assign tasks to workers based on resource requirements.
        
        Args:
            tasks: Dictionary mapping task IDs to Task objects
            nodes: JobResource containing available nodes and their resources
            level: Current level in the hierarchy
        """
        if len(tasks) == 0:
            self.logger.error("Greedy worker policy needs tasks")
            raise ValueError("Needs Tasks for creating workers") 
        
        nlevels = self.nlevels
        
        # Extract node names and resources from JobResource
        node_names = nodes.nodes
        node_resources = {node_name: resource for node_name, resource in zip(nodes.nodes, nodes.resources)}
        
        # Step 1: Sort tasks by decreasing number of nodes required
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].nnodes, reverse=True)
        
        # Remove tasks that have num nodes > total nodes of this master
        removed_tasks = []
        while sorted_tasks and sorted_tasks[0][1].nnodes > len(node_names):
            removed_tasks.append((sorted_tasks.pop(0))[0])
        
        if len(removed_tasks) > 0:
            self.logger.warning(f"Can't schedule {','.join(removed_tasks)}!")

        # Step 2: Calculate cumulative sum of nodes required for tasks
        cum_sum_nnodes = list(accumulate(task.nnodes for _, task in sorted_tasks))
        
        # Step 3: Determine the max number of workers
        nworkers_max = 0
        while nworkers_max < len(cum_sum_nnodes) and cum_sum_nnodes[nworkers_max] <= len(node_names):
            nworkers_max += 1
        
        # Step 4: Do a simple interpolation in the log2 space to determine number of workers at the current level
        if nlevels > 1:
            # Create log2 space arrays for interpolation
            x_vals = np.array([0, nlevels], dtype=float)
            y_vals = np.array([0, np.log2(max(nworkers_max, 1))], dtype=float)
            # Interpolate in log2 space
            log2_nworkers = int(np.interp(level + 1, x_vals, y_vals))
            # Convert back from log2 space
            nworkers = max(1, min(int(2 ** log2_nworkers), nworkers_max))
        else:
            nworkers = nworkers_max
        
        # Step 5: Distribute nodes among workers
        if len(node_names) == 1:
            children_assignments = {
                wid: {
                "job_resource": JobResource(resources=[node_resources[node_names[0]]], nodes=node_names),
                "task_ids": [task_id]
                }
                for wid, (task_id, _) in enumerate(sorted_tasks[:nworkers])
            }
        else:
            if len(node_names) < nworkers:
                self.logger.error(f"number of nodes < number of children")
                raise RuntimeError
            
            # Initialize worker node assignments for the first set of tasks
            nnodes = [task.nnodes for _, task in sorted_tasks[:nworkers]]
            if sum(nnodes) < len(node_names):
                nremaining = len(node_names) - sum(nnodes)
                for i in range(nremaining):
                    nnodes[i % nworkers] += 1
            nnodes = list(accumulate(nnodes))
            
            children_assignments = {}
            for wid, (task_id, _) in enumerate(sorted_tasks[:nworkers]):
                node_indices = range(nnodes[wid]) if wid == 0 else range(nnodes[wid-1], nnodes[wid])
                worker_nodes = [node_names[i] for i in node_indices]
                worker_resources = [node_resources[node_names[i]] for i in node_indices]
                
                children_assignments[wid] = {
                    "job_resource": JobResource(resources=worker_resources, nodes=worker_nodes),
                    "task_ids": [task_id]
                }
        
        # Step 6: Assign remaining tasks to workers in a round-robin fashion
        for i, (task_id, _) in enumerate(sorted_tasks[nworkers:]):
            worker_id = i % nworkers
            children_assignments[worker_id]["task_ids"].append(task_id)

        return children_assignments, removed_tasks

@policy_registry.register("simple_split_worker_policy", type="worker_policy")
class SimpleSplitWorkerPolicy(WorkerPolicy):
    """
    A worker policy that splits nodes evenly among a specified number of children,
    then assigns tasks in round-robin fashion.
    
    Tasks are assigned to workers in round-robin order, checking if each task fits
    within the worker's resources. Tasks with CPU or GPU affinity set will be skipped
    and added to the removed tasks list.
    
    Configuration (class variables):
        nchildren: Number of child workers (must be > 0)
    
    Note: Not registered by default. To use:
        1. Subclass and set nchildren:
            class Split8Policy(SimpleSplitWorkerPolicy):
                nchildren = 8
        2. Register it:
            policy_registry.register_policy("split_8", Split8Policy, type="worker_policy")
        3. Use it:
            scheduler = AsyncWorkerScheduler(..., policy="split_8")
    """
    
    def __init__(self, nchildren: int = None, logger: logging.Logger = None, **kwargs):
        self.nchildren = nchildren
        if self.nchildren is None or self.nchildren <= 0:
            raise ValueError(f"nchildren must be positive, got {self.nchildren}")
        self.logger = logging.getLogger(__name__) if logger is None else logger
        self.logger.info(f"Initialized SimpleSplitWorkerPolicy with nchildren={self.nchildren}")
    
    def get_worker_assignment(self,
                        tasks: Dict[str, Task],
                        nodes: JobResource,
                        level: int) -> Tuple[Dict, List]:
        """
        Split nodes and their resources evenly among the specified number of children,
        then assign tasks in round-robin fashion.
        
        Args:
            tasks: Dictionary mapping task IDs to Task objects
            nodes: JobResource containing available nodes and their resources
            level: Current level in the hierarchy (not used by this policy)
        
        Returns:
            Tuple of (children_assignments dict, list of removed task IDs)
        """
        nchildren = self.nchildren
        nnodes = len(nodes.nodes)
        
        if nchildren <= nnodes:
            # Simple case: distribute whole nodes to children
            base_nodes_per_child = nnodes // nchildren
            remainder = nnodes % nchildren
            
            children_assignments = {}
            start_idx = 0
            
            for wid in range(nchildren):
                count = base_nodes_per_child + (1 if wid < remainder else 0)
                end_idx = start_idx + count
                
                worker_nodes = nodes.nodes[start_idx:end_idx]
                worker_resources = nodes.resources[start_idx:end_idx]
                
                children_assignments[wid] = {
                    "job_resource": JobResource(resources=worker_resources, nodes=worker_nodes),
                    "task_ids": []
                }
                
                start_idx = end_idx
        else:
            # nchildren > nnodes: need to split nodes at resource level
            if nchildren % nnodes != 0:
                self.logger.error(f"nchildren ({nchildren}) must be a multiple of number of nodes ({nnodes}) when nchildren > nnodes")
                raise ValueError(f"nchildren ({nchildren}) must be a multiple of number of nodes ({nnodes})")
            
            splits_per_node = nchildren // nnodes
            
            # Split each node's resources
            children_assignments = {}
            wid = 0
            
            for node_name, node_resource in zip(nodes.nodes, nodes.resources):
                divided_resources = node_resource.divide(splits_per_node)
                
                for resource_part in divided_resources:
                    children_assignments[wid] = {
                        "job_resource": JobResource(resources=[resource_part], nodes=[node_name]),
                        "task_ids": []
                    }
                    wid += 1
        
        # Now assign tasks in round-robin fashion
        removed_tasks = []
        current_worker = 0
        
        for task_id, task in tasks.items():
            # Check if task fits in current worker
            assigned = False
            attempts = 0
            
            # Try each worker once in round-robin order
            while attempts < nchildren:
                worker_job_resource = children_assignments[current_worker]["job_resource"]
                
                # Check if this worker can accommodate the task
                if self._can_task_fit(task, worker_job_resource):
                    children_assignments[current_worker]["task_ids"].append(task_id)
                    assigned = True
                    current_worker = (current_worker + 1) % nchildren
                    break
                
                # Try next worker
                current_worker = (current_worker + 1) % nchildren
                attempts += 1
            
            if not assigned:
                removed_tasks.append(task_id)
                self.logger.warning(f"Task {task_id} does not fit in any worker")
        
        return children_assignments, removed_tasks
    
    def _can_task_fit(self, task: Task, worker_job_resource: JobResource) -> bool:
        """
        Check if a task can fit in the given worker's job resource.
        
        Args:
            task: Task to check
            worker_job_resource: Worker's available resources
        
        Returns:
            True if task can fit, False otherwise
        """
        # Skip tasks with CPU or GPU affinity
        if task.cpu_affinity or task.gpu_affinity:
            self.logger.warning(f"Task {task.task_id} has CPU/GPU affinity set, skipping assignment")
            return False
        
        # Check if worker has enough nodes
        if len(worker_job_resource.nodes) < task.nnodes:
            return False
        
        # Calculate resource requirements per node for this task
        cpus_needed = task.ppn
        gpus_needed = task.ppn * task.ngpus_per_process
        
        # Create a NodeResourceCount requirement to check against each node
        required_resource = NodeResourceCount(ncpus=cpus_needed, ngpus=gpus_needed)
        
        # Check if the first nnodes of the worker can satisfy the task
        for i in range(task.nnodes):
            worker_node_resource = worker_job_resource.resources[i]
            
            # Use the same containment check as in ClusterResource._can_allocate
            if required_resource not in worker_node_resource:
                return False
        
        return True
    
