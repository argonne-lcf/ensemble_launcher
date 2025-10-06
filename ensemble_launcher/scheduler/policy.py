from ensemble_launcher.ensemble import Task
from abc import ABC, abstractmethod
from typing import Dict, Type
import logging

logger = logging.getLogger(__name__)

class Policy(ABC):
    @abstractmethod
    def get_score(self, *args, **kwargs) -> float:
        """
        Returns a score for scheduling the given task on the given node.
        Higher score means higher priority.
        """
        pass

class PolicyRegistry:

    def __init__(self):
        self.available_policies: Dict[str, Type[Policy]] = {}
    
    def register(self, policy_name: str):
        def decorator(cls: Type[Policy]):
            self.available_policies[policy_name] = cls
            return cls
        return decorator
    
    def create_policy(self,policy_name:str, *args, **kwargs):
        if policy_name in self.available_policies:
            return self.available_policies[policy_name](*args,**kwargs)
        else:
            logger.error(f"{policy_name} not available. Available policy names {self.available_policies.keys()}")
            raise

policy_registry = PolicyRegistry()

@policy_registry.register("large_resource_policy")
class LargeResourcePolicy(Policy):
    """
    A simple policy that always prioritizes a larger task.
    The task that uses gpus is give more priority.
    """
    def __init__(self,cpu_weight:float=1.0, gpu_weight:float=2.0):
        self.cpu_weight = cpu_weight
        self.gpu_weight = gpu_weight

    def get_score(self, task: Task) -> float:
        return task.nnodes*task.ppn*(task.ngpus_per_process*self.gpu_weight +
                                    self.cpu_weight)