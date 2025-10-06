from typing import Any, Dict, Callable, Tuple, Union
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList, NodeResourceCount
from concurrent.futures import ProcessPoolExecutor
from .utils import run_callable_with_affinity, run_cmd, executor_registry
import multiprocessing as mp
import uuid
import logging

from .base import Executor

logger = logging.getLogger(__name__)

@executor_registry.register("multiprocessing")
class MultiprocessingExecutor(Executor):
    def __init__(self,gpu_selector: str = "ZE_AFFINITY_MASK"):
        self._executor = ProcessPoolExecutor()
        self._futures: Dict[str, mp.Process] = {}
        self._gpu_selector = gpu_selector
        self._results: Dict[str, Any] = {}

    def start(self,job_resource: JobResource, 
               fn: Union[Callable,str], 
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
                logger.warning(f"Received non-zero gpu request using NodeResourceCount. Oversubscribing")
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env.update({self._gpu_selector: gpu_ids})

        if callable(fn):
            future = self._executor.submit(run_callable_with_affinity,*(fn, task_args, task_kwargs, cpu_id, env))
        elif isinstance(fn, str):
            future = self._executor.submit(run_cmd,*(fn, task_args, task_kwargs, cpu_id, env))
        else:
            logger.warning(f"Can only excute either a str or a callable")
            return None
        
        self._futures[task_id] = future
        return task_id
    
    def stop(self, task_id: str) -> bool:
        return self._futures[task_id].cancel()

    def wait(self, task_id: str, timeout: float = None):
        try:
            result = self._futures[task_id].result(timeout=timeout)
            self._results[task_id] = result
            return True
        except TimeoutError:
            logger.warning(f"Task {task_id} did not complete within the timeout.")
            return False

    def result(self, task_id: str, timeout = None) -> Any:
        try:
            return self._results[task_id]
        except KeyError:
            if self.wait(task_id, timeout=timeout):
                return self._results[task_id]
            else:
                return None
    
    def exception(self, task_id: str) -> Exception:
        """This is a blocking call"""
        return self._futures[task_id].exception()
    
    def running(self, task_id: str) -> bool:
        return self._futures[task_id].running()
    
    def done(self, task_id: str) -> bool:
        return self._futures[task_id].done()
    
    def shutdown(self, force: bool = False):
        """
        Shutdown the underlying ProcessPoolExecutor.

        Args:
            wait (bool): If True, waits for all running futures to finish.
        """
        self._executor.shutdown(wait=True, cancel_futures=force)
        self._futures.clear()
        self._results.clear()