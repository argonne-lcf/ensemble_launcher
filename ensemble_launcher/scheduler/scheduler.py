
from .resource import NodeResourceList, JobResource, LocalClusterResource, ClusterResource
from .resource import NodeResourceCount
from ensemble_launcher.orchestrator import Node
from ensemble_launcher.ensemble import Task
from typing import List, Dict, Any, Union
from .policy import *
from collections import deque


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
        allocated,resource = self.cluster_resource.allocate(worker_resource)
        if allocated:
            self.workers[worker_id] = resource
        return allocated,resource
    
    def free(self, worker_id: str):
        if worker_id in self.workers:
            return self.cluster_resource.deallocate(self.workers[worker_id])
        return None
    
    def get_worker_assignment(self):
        return self.workers

class TaskScheduler(Scheduler):
    def __init__(self, 
                 tasks: List[Task], 
                 cluster: ClusterResource, 
                 policy: Union[str,Policy]= "large_resource_policy"):
        super().__init__(cluster)
        self.tasks = tasks
        self.task_assignment: Dict[str, JobResource] = {}
        self._running_tasks: List[str] = []
        self._done_tasks: List[str] = []
        if isinstance(policy, str):
            self.scheduler_policy: Policy = policy_registry.create_policy(policy)
        else:
            self.scheduler_policy: Policy = policy
        self.sorted_tasks: List[Task] = sorted(self.tasks,key=self.scheduler_policy.get_score,reverse=True)
    
    def assign(self, task_id: str,  task_resource: JobResource):
        allocated,resource = self.cluster_resource.allocate(task_resource)
        if allocated:
            self.task_assignment[task_id] = resource
        return allocated,resource
    
    def get_ready_tasks(self):
        ready_tasks = {}
        for task in self.sorted_tasks:
            req = JobResource(
                resources=[NodeResourceCount(ncpus=task.ppn,ngpus=task.ngpus_per_process*task.ppn) for i in task.nnodes]
            )
            allocated, resource = self.assign(task.task_id,req)
            if allocated:
                ready_tasks[task.task_id] = resource
                self._running_tasks.append(task.task_id)
        return ready_tasks
    
    def add_task(self,task: Task):
        self.tasks.append(task)
        self.sorted_tasks = sorted(self.tasks, key=self.scheduler_policy.get_score, reverse=True)
    
    def free(self, task_id: str):
        if task_id in self.tasks:
            self._done_tasks.append(task_id)
            self._running_tasks.remove(task_id)
            return self.cluster_resource.deallocate(self.tasks[task_id])
        return None
    
    def get_task_assignment(self):
        return self.task_assignment
    
    @property
    def running_tasks(self):
        """Return currently running tasks"""
        return self._running_tasks
    
    @property
    def done_tasks(self):
        """
            Return done tasks. Either failed or success.
            A task id can repeat.
        """
        return self._done_tasks
    
    def run_count(self, task_id: str):
        return self._done_tasks.count(task_id)