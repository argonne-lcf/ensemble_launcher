from .resource import NodeResourceList, JobResource, LocalClusterResource, ClusterResource
from .resource import NodeResourceCount
from ensemble_launcher.ensemble import Task, TaskStatus
from typing import List, Dict, Any, Union, Set
from .policy import policy_registry, Policy
from  logging import Logger
import copy

# self.logger = logging.getself.logger(__name__)


class Scheduler:
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """
    def __init__(self, logger: Logger, cluster_resource: LocalClusterResource):
        self.logger = logger
        self._cluster_resource = cluster_resource
        
    def assign(self):
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
    def __init__(self, logger: Logger, cluster: ClusterResource):
        super().__init__(logger, cluster)
        self.workers: Dict[str, JobResource] = {}
    
    def assign(self, worker_id: str,  worker_resource: JobResource):
        allocated,resource = self.cluster.allocate(worker_resource)
        if allocated:
            self.workers[worker_id] = resource
        return allocated,resource
    
    def free(self, worker_id: str):
        if worker_id in self.workers:
            return self.cluster.deallocate(self.workers[worker_id])
        return None
    
    def get_worker_assignment(self):
        return self.workers

class TaskScheduler(Scheduler):
    def __init__(self, 
                 logger: Logger,
                 tasks: Dict[str, Task], 
                 cluster: ClusterResource, 
                 policy: Union[str,Policy]= "large_resource_policy"):
        super().__init__(logger, cluster)
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
    
    def _buld_task_resource_req(self, task: Task) -> JobResource:
        req = JobResource(
                resources=[NodeResourceCount(ncpus=task.ppn,ngpus=task.ngpus_per_process*task.ppn) for i in range(task.nnodes)]
            )
        if len(task.cpu_affinity) > 0 or len(task.gpu_affinity) > 0:
            if task.cpu_affinity and (task.ngpus_per_process > 0 and not task.gpu_affinity):
                self.logger.warning(f"Task {task.task_id}: Ignoring cpu_affinity as gpu_affinity is not set")
                return req
            
            if task.gpu_affinity and not task.cpu_affinity:
                self.logger.warning(f"Task {task.task_id}: Ignoring gpu_affinitiy as cpu_affinity is not set")
                return req
            
            req = JobResource(
                resources=[NodeResourceList(cpus=task.cpu_affinity, gpus=task.gpu_affinity) for node in range(task.nnodes)]
            )
        return req

    def get_ready_tasks(self) -> Dict[str, JobResource]:
        ready_tasks = {}
        for task_id in self.sorted_tasks:
            task = self.tasks[task_id]
            req = self._buld_task_resource_req(task)
            allocated,resource = self.cluster.allocate(req)
            if allocated:
                if task.task_id in self._running_tasks:
                    self.logger.error(f"Task {task.task_id} is already running")
                    raise RuntimeError
                #add to running tasks
                self._running_tasks.add(task.task_id)
                ##save the assignment
                self.task_assignment[task_id] = resource
                #save the req
                ready_tasks[task.task_id] = resource

        #remove from the queue
        for task_id in ready_tasks.keys():
            self.sorted_tasks.remove(task_id)

        self.logger.debug(f"Allocated {list(ready_tasks.keys())}")
        return ready_tasks
    
    def add_task(self,task: Task) -> bool:
        try:
            if task.nnodes > len(self.cluster.nodes):
                raise ValueError(f"Task {task.task_id} requires {task.nnodes} nodes, but only {len(self.cluster.nodes)} are available")
            self.tasks[task.task_id] = task
            self.sorted_tasks = sorted(self.tasks.keys(), key=lambda task_id: self.scheduler_policy.get_score(self.tasks[task_id]), reverse=True)
            return True
        except Exception as e:
            self.logger.error(f"Failed to add task {task.task_id}: {e}")
            return False
    
    def delete_task(self, task: Task) -> bool:
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
        if task_id in self.tasks:
            if task_id not in self._running_tasks:
                self.logger.error(f"{task_id} is not running")
                raise RuntimeError
            ##delete from running tasks
            self._running_tasks.discard(task_id)
            #deallocate
            self.cluster.deallocate(self.task_assignment[task_id])
            #delete the assignment
            del self.task_assignment[task_id]
            #Add to done tasks
            self._done_tasks.append(task_id)
            #add to failed/successful tasks
            if status == TaskStatus.FAILED:
                self._failed_tasks.add(task_id)
            elif status == TaskStatus.SUCCESS:
                self._successful_tasks.add(task_id)
                self._failed_tasks.discard(task_id)
            
            self.logger.debug(f"Freed {task_id}")
            


        return None
    
    def get_task_assignment(self):
        return self.task_assignment
    
    @property
    def running_tasks(self) -> Set[str]:
        """Return IDs of currently running tasks."""
        return copy.deepcopy(self._running_tasks)

    @property
    def failed_tasks(self) -> Set[str]:
        """Return IDs of tasks that have failed."""
        return self._failed_tasks

    @property
    def done_tasks(self) -> List[str]:
        """Return IDs of tasks that are done. Can have duplicates"""
        return self._done_tasks

    @property
    def successful_tasks(self) -> Set[str]:
        """Return IDs of tasks that have completed successfully."""
        return self._successful_tasks
    
    @property
    def remaining_tasks(self) -> Set[str]:
        return set(self.tasks.keys()) - (self.successful_tasks | self.failed_tasks)

    def run_count(self, task_id: str):
        return self._done_tasks.count(task_id)