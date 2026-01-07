from typing import Any, Dict, Callable, Tuple, Union
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList, NodeResourceCount
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .utils import run_callable_with_affinity, run_cmd, executor_registry
import uuid
from datetime import datetime

@executor_registry.register("async_processpool", type="async")
class AsyncProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self,logger,
                 gpu_selector: str = "ZE_AFFINITY_MASK",
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._gpu_selector = gpu_selector

    def submit(self,job_resource: JobResource, 
               fn: Union[Callable,str], 
              task_args: Tuple = (),
              task_kwargs: Dict = {}, 
              env: Dict[str, Any] = {}):
        
        if len(job_resource.nodes) > 1 or job_resource.resources[0].cpu_count > 1:
            raise ValueError("MultiProcessingExecutor can only execute serial tasks")
        
        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus[0]
        
        if req.gpu_count > 0:
            if isinstance(req, NodeResourceCount):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpu_count])
                self.logger.warning(f"Received non-zero gpu request using NodeResourceCount. Oversubscribing")
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env.update({self._gpu_selector: gpu_ids})

        if callable(fn):
            future = super().submit(run_callable_with_affinity,*(fn, task_args, task_kwargs, cpu_id, env))
        elif isinstance(fn, str):
            future = super().submit(run_cmd,*(fn, task_args, task_kwargs, cpu_id, env))
        else:
            self.logger.warning(f"Can only excute either a str or a callable")
            return None

        return future


@executor_registry.register("async_threadpool", type="async")
class AsyncThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, logger,
                 gpu_selector: str = "ZE_AFFINITY_MASK",
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._gpu_selector = gpu_selector

    def submit(self, job_resource: JobResource,
               fn: Union[Callable, str],
               task_args: Tuple = (),
               task_kwargs: Dict = {},
               env: Dict[str, Any] = {}):
        
        task_id = str(uuid.uuid4())
        if len(job_resource.nodes) > 1 or job_resource.resources[0].cpu_count > 1:
            raise ValueError("MultiProcessingExecutor can only execute serial tasks")
        
        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus[0]
        
        if req.gpu_count > 0:
            if isinstance(req, NodeResourceCount):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpu_count])
                self.logger.warning(f"Received non-zero gpu request using NodeResourceCount. Oversubscribing")
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env.update({self._gpu_selector: gpu_ids})
        if callable(fn):
            future = super().submit(run_callable_with_affinity, *(fn, task_args, task_kwargs, cpu_id, env))
        elif isinstance(fn, str):
            future = super().submit(run_cmd, *(fn, task_args, task_kwargs, cpu_id, env, self._return_stdout))
        else:
            self.logger.warning(f"Can only excute either a str or a callable")
            return None
        
        return future