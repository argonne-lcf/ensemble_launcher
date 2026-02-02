from typing import Any, Dict, Callable, Tuple, Union, Optional
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList, NodeResourceCount
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from .utils import run_callable_with_affinity, run_cmd, executor_registry
import uuid
import os
from logging import Logger
from ensemble_launcher.profiling import get_registry, EventRegistry

def dummy_task():
    return


def dummy_task():
    return

@executor_registry.register("async_processpool", type="async")
class AsyncProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(self,logger:Logger,
                 gpu_selector: str = "ZE_AFFINITY_MASK",
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._gpu_selector = gpu_selector
        
        super().submit(dummy_task)

        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry: EventRegistry = get_registry()
        self.logger.info("Initialized AsyncProcessPool Executor!")

    def submit(self,job_resource: JobResource, 
               fn: Union[Callable,str], 
              task_args: Tuple = (),
              task_kwargs: Dict = {}, 
              env: Dict[str, Any] = {}):
        
        if len(job_resource.nodes) > 1:
            raise ValueError("MultiProcessingExecutor can only execute single node tasks")
        
        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus
        
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
    def __init__(self, logger: Logger,
                 gpu_selector: str = "ZE_AFFINITY_MASK",
                 **kwargs):
        super().__init__(**kwargs)
        self.logger = logger
        self._gpu_selector = gpu_selector
        self.logger.info("Initialized threadpool executor")

    def submit(self, job_resource: JobResource,
               fn: Union[Callable, str],
               task_args: Tuple = (),
               task_kwargs: Dict = {},
               env: Dict[str, Any] = None):
        
        if env is None:
            env = {}

        if len(job_resource.nodes) > 1 or job_resource.resources[0].cpu_count > 1:
            raise ValueError("AsyncThreadPool can only execute serial tasks. Use MPI/ProcessPool for parallel tasks.")
        
        req = job_resource.resources[0]
        if isinstance(req, NodeResourceList):
            cpu_req = req.cpus
            gpu_req = req.gpus
        else:
            cpu_req = None
            gpu_req = None

        if callable(fn):
            self.logger.info
            # STRICT RULE: Threads cannot have private envs or affinity
            if env or (gpu_req is not None and len(gpu_req) > 0):
                self.logger.error("Safety Violation: Cannot set 'env' or 'gpu' constraints for a Python function"
                    "running in a ThreadPool. \n"
                    "Reason: Threads share the same process environment and affinity.\n"
                    "Solution: Use 'ProcessPool'.")
                raise ValueError(
                    "Safety Violation: Cannot set 'env' or 'gpu' constraints for a Python function"
                    "running in a ThreadPool. \n"
                    "Reason: Threads share the same process environment and affinity.\n"
                    "Solution: Use 'ProcessPool'."
                )
            
            # Warn about CPU affinity if they asked for a specific core
            if cpu_req is not None:
                self.logger.warning(
                    "Ignoring CPU pinning for threaded Python task. "
                    "Threads cannot be pinned individually without affecting the whole process."
                )

            future = super().submit(fn, *task_args, **task_kwargs)

        elif isinstance(fn, str):

            # Prepare the environment (COPY it to avoid race conditions)            
            task_env = os.environ.copy()
            task_env.update(env)
            
            # Inject GPU IDs if present (Safe here because it's a new process)
            if gpu_req is not None:
                gpu_ids = ",".join([str(g) for g in gpu_req])
                task_env[self._gpu_selector] = gpu_ids
            
            # We can also respect CPU affinity using `taskset` inside run_cmd if you implemented that support
            future = super().submit(run_cmd, fn, task_args, task_kwargs, cpu_req, task_env, self._return_stdout)
            
        else:
            raise TypeError(f"Task must be str or callable, got {type(fn)}")
            
        return future