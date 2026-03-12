from .resource import NodeResourceList, JobResource, LocalClusterResource, ClusterResource, NodeResource
from .resource import NodeResourceCount
from ensemble_launcher.ensemble import Task, TaskStatus
from typing import List, Dict, Union, Set, Tuple
from .policy import policy_registry, Policy
from  logging import Logger
import copy
import threading

# self.logger = logging.getself.logger(__name__)


class Scheduler:
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """
    def __init__(self, logger: Logger, cluster_resource: LocalClusterResource):
        self.logger = logger
        self._cluster_resource = cluster_resource
        
    def assign(self, *args, **kwargs):
        raise NotImplementedError
    
    @property
    def cluster(self):
        return self._cluster_resource

    @cluster.setter
    def cluster(self, value: ClusterResource):
        self._cluster_resource = value
    
    def get_cluster_status(self):
        """Returns the current status of the cluster"""
        return self._cluster_resource.get_status()


class WorkerScheduler(Scheduler):
    def __init__(self, logger: Logger, nodes: JobResource, config):
        cluster = LocalClusterResource(logger.getChild('cluster'), nodes)
        super().__init__(logger, cluster)
        self.workers: Dict[str, JobResource] = {}
        self._lock = threading.RLock()
        self._config = config
        # Initialize policy - uses the registered state instance
        from .policy import ChildrenPolicy
        self.policy: ChildrenPolicy = policy_registry.create_policy(self._config.children_scheduler_policy,
                                                                  policy_kwargs={"nchildren": self._config.nchildren, 
                                                                                "nlevels":self._config.policy_config.nlevels,
                                                                                "logger": logger.getChild('policy')})
    
    def assign(self, tasks: Dict[str, Task], level: int) -> Tuple[Dict[str, Dict], List[str]]:
        """Use policy to assign workers and allocate resources from cluster.
        
        Returns:
            Tuple of (worker_assignments, removed_tasks)
            where worker_assignments maps worker_id to {"job_resource": JobResource, "task_ids": [...]}
        """
        with self._lock:
            # Get worker assignments from policy - pass only runtime parameters
            worker_assignments, removed_tasks = self.policy.get_worker_assignment(
                tasks=tasks,
                nodes=self.cluster.nodes,
                level=level
            )
            
            # Allocate resources for each worker
            allocated_workers = {}
            for worker_id, assignment in worker_assignments.items():
                job_resource = assignment["job_resource"]
                allocated, resource = self.cluster.allocate(job_resource)
                if allocated:
                    self.workers[worker_id] = resource
                    allocated_workers[worker_id] = {
                        "job_resource": resource,
                        "task_ids": assignment["task_ids"]
                    }
                else:
                    self.logger.error(f"Failed to allocate resources for worker {worker_id}")
            
        return allocated_workers, removed_tasks
    
    def free(self, worker_id: str) -> bool:
        """Deallocate resources for a worker.
        
        Args:
            worker_id: ID of the worker to free
            
        Returns:
            True if deallocation was successful, False otherwise
        """
        with self._lock:
            if worker_id in self.workers:
                result = self.cluster.deallocate(self.workers[worker_id])
                if result:
                    del self.workers[worker_id]
                return result
            return False
    
    def get_worker_assignment(self):
        with self._lock:
            return copy.deepcopy(self.workers)

class TaskScheduler(Scheduler):
    def __init__(self, 
                 logger: Logger,
                 tasks: Dict[str, Task], 
                 nodes: JobResource,
                 policy: Union[str,Policy]= "large_resource_policy"):
        cluster = LocalClusterResource(logger.getChild('cluster'), nodes)
        super().__init__(logger, cluster)
        self._lock = threading.RLock()
        self.tasks: Dict[str, Task] = tasks
        self.task_assignment: Dict[str, JobResource] = {}
        self._running_tasks: Set[str] = set()
        self._done_tasks: List[str] = []
        self._failed_tasks: Set[str] = set()
        self._successful_tasks: Set[str] = set()
        if isinstance(policy, str):
            self.scheduler_policy: Policy = policy_registry.create_policy(policy)
        else:
            self.scheduler_policy: Policy = policy
        self.sorted_tasks: List[str] = sorted(list(self.tasks.keys()), key=lambda task_id: self.scheduler_policy.get_score(self.tasks[task_id]), reverse=True)
        self.logger.debug(f"Sorted tasks {self.sorted_tasks}")

    def get_ready_tasks(self) -> Dict[str, JobResource]:
        with self._lock:
            ready_tasks = {}
            for task_id in self.sorted_tasks:
                task = self.tasks[task_id]
                req = task.get_resource_requirements()
                allocated, resource = self.cluster.allocate(req)
                if allocated:
                    if task.task_id in self._running_tasks:
                        self.logger.error(f"Task {task.task_id} is already running")
                        raise RuntimeError
                    # add to running tasks
                    self._running_tasks.add(task.task_id)
                    # save the assignment
                    self.task_assignment[task_id] = resource
                    # save the req
                    ready_tasks[task.task_id] = resource

            # remove from the queue
            for task_id in ready_tasks.keys():
                self.sorted_tasks.remove(task_id)

            self.logger.debug(f"Allocated {list(ready_tasks.keys())}")
            return ready_tasks
    
    def add_task(self, task: Task) -> bool:
        with self._lock:
            try:
                if task.nnodes > len(self.cluster.nodes.nodes):
                    raise ValueError(f"Task {task.task_id} requires {task.nnodes} nodes, but only {len(self.cluster.nodes.nodes)} are available")
                self.tasks[task.task_id] = task
                self.sorted_tasks = sorted(self.tasks.keys(), key=lambda task_id: self.scheduler_policy.get_score(self.tasks[task_id]), reverse=True)
                return True
            except Exception as e:
                self.logger.error(f"Failed to add task {task.task_id}: {e}")
                return False
    
    def delete_task(self, task: Task) -> bool:
        with self._lock:
            if task.task_id not in self.tasks:
                self.logger.warning(f"Unknown task: {task.task_id}")
                return False
            
            try:
                # Remove from tasks dict
                del self.tasks[task.task_id]

                # If running, free the resources
                if task.task_id in self._running_tasks:
                    self.cluster.deallocate(self.task_assignment[task.task_id])

                if task.task_id in self.task_assignment:
                    del self.task_assignment[task.task_id]

                # Remove from running and status sets
                self._running_tasks.discard(task.task_id)
                self._done_tasks = [t for t in self._done_tasks if t != task.task_id]
                self._failed_tasks.discard(task.task_id)
                self._successful_tasks.discard(task.task_id)

                # Remove from sorted tasks
                if task.task_id in self.sorted_tasks:
                    self.sorted_tasks.remove(task.task_id)

                return True
            except Exception as e:
                self.logger.warning(f"Failed to delete task {task.task_id}: {e}")
                return False
    
    def free(self, task_id: str, status: TaskStatus):
        with self._lock:
            if task_id in self.tasks:
                if task_id not in self._running_tasks:
                    self.logger.error(f"{task_id} is not running")
                    raise RuntimeError
                # delete from running tasks
                self._running_tasks.discard(task_id)
                # deallocate
                self.cluster.deallocate(self.task_assignment[task_id])
                # delete the assignment
                del self.task_assignment[task_id]
                # Add to done tasks
                self._done_tasks.append(task_id)
                # add to failed/successful tasks
                if status == TaskStatus.FAILED:
                    self._failed_tasks.add(task_id)
                elif status == TaskStatus.SUCCESS:
                    self._successful_tasks.add(task_id)
                    self._failed_tasks.discard(task_id)
                
                self.logger.debug(f"Freed {task_id}")

            return None
    
    def get_task_assignment(self):
        with self._lock:
            return copy.deepcopy(self.task_assignment)
    
    @property
    def running_tasks(self) -> Set[str]:
        """Return IDs of currently running tasks."""
        with self._lock:
            return copy.deepcopy(self._running_tasks)

    @property
    def failed_tasks(self) -> Set[str]:
        """Return IDs of tasks that have failed."""
        with self._lock:
            return copy.deepcopy(self._failed_tasks)

    @property
    def done_tasks(self) -> List[str]:
        """Return IDs of tasks that are done. Can have duplicates"""
        with self._lock:
            return copy.deepcopy(self._done_tasks)

    @property
    def successful_tasks(self) -> Set[str]:
        """Return IDs of tasks that have completed successfully."""
        with self._lock:
            return copy.deepcopy(self._successful_tasks)
    
    @property
    def remaining_tasks(self) -> Set[str]:
        with self._lock:
            return set(self.tasks.keys()) - (self._successful_tasks | self._failed_tasks)

    def run_count(self, task_id: str):
        with self._lock:
            return self._done_tasks.count(task_id)