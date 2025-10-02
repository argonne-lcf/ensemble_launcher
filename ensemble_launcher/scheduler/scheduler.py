from .resource import NodeResourceList, JobResource, LocalClusterResource, ClusterResource
from .resource import NodeResourceCount
from ensemble_launcher.ensemble import Task, TaskStatus
from typing import List, Dict, Any, Union, Set
from .policy import policy_registry, Policy
from collections import deque
import logging
from collections import deque

logger = logging.getLogger(__name__)


class Scheduler:
    """
    Class responsible for assigning a certain task onto resource.
    The resources of the scheduler could be updated
    """
    def __init__(self,cluster_resource: LocalClusterResource):
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
    def __init__(self, cluster: ClusterResource):
        super().__init__(cluster)
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
                 tasks: Dict[str, Task], 
                 cluster: ClusterResource, 
                 policy: Union[str,Policy]= "large_resource_policy"):
        super().__init__(cluster)
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
        self.sorted_tasks: List[Task] = sorted(self.tasks.values(),key=self.scheduler_policy.get_score,reverse=True)
    
    def assign(self, task_id: str,  task_resource: JobResource):
        allocated,resource = self.cluster.allocate(task_resource)
        if allocated:
            self.task_assignment[task_id] = resource
        return allocated,resource
    
    def _buld_task_resource_req(self, task: Task) -> JobResource:
        req = JobResource(
                resources=[NodeResourceCount(ncpus=task.ppn,ngpus=task.ngpus_per_process*task.ppn) for i in range(task.nnodes)]
            )
        if len(task.cpu_affinity) > 0 or len(task.gpu_affinity) > 0:
            if task.cpu_affinity and (task.ngpus_per_process > 0 and not task.gpu_affinity):
                logger.warning(f"Task {task.task_id}: Ignoring cpu_affinity as gpu_affinity is not set")
                return req
            
            if task.gpu_affinity and not task.cpu_affinity:
                logger.warning(f"Task {task.task_id}: Ignoring gpu_affinitiy as cpu_affinity is not set")
                return req
            
            req = JobResource(
                resources=[NodeResourceList(cpus=task.cpu_affinity, gpus=task.gpu_affinity) for node in range(task.nnodes)]
            )
        return req

    def get_ready_tasks(self) -> Dict[str, JobResource]:
        ready_tasks = {}
        allocated_tasks = []
        for task in self.sorted_tasks:
            req = self._buld_task_resource_req(task)
            allocated, resource = self.assign(task.task_id,req)
            if allocated:
                ready_tasks[task.task_id] = resource
                if task.task_id in self._running_tasks:
                    logger.warning(f"Task {task.task_id} is already running")
                self._running_tasks.add(task.task_id)
                allocated_tasks.append(task)

        ##remove them from the queue
        for task in allocated_tasks:
            self.sorted_tasks.remove(task)
            
        return ready_tasks
    
    def add_task(self,task: Task):
        self.tasks[task.task_id] = task
        self.sorted_tasks = sorted(self.tasks.values(), key=self.scheduler_policy.get_score, reverse=True)
    
    def free(self, task_id: str, status: TaskStatus):
        if task_id in self.tasks:
            self._done_tasks.append(task_id)
            self._running_tasks.discard(task_id)
            if status == TaskStatus.FAILED:
                self._failed_tasks.add(task_id)
            elif status == TaskStatus.SUCCESS:
                self._successful_tasks.add(task_id)
                self._failed_tasks.discard(task_id)
            return self.cluster.deallocate(self.task_assignment[task_id])
        return None
    
    def get_task_assignment(self):
        return self.task_assignment
    
    @property
    def running_tasks(self) -> Set[str]:
        """Return IDs of currently running tasks."""
        return self._running_tasks

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