from typing import Any, Dict, Callable, Tuple, Union
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList, NodeResourceCount
from ensemble_launcher.comm.queue import QueueProtocol, queue_registry
import uuid
import logging
from .utils import run_callable_with_affinity, run_cmd, return_wrapper, executor_registry
import os
from datetime import datetime
from .base import Executor

try:
    import dragon
    from dragon.native.process import MSG_PIPE, MSG_DEVNULL, Process, ProcessTemplate
    from dragon.infrastructure.connection import Pipe
    from dragon.native.process_group import ProcessGroup
    from dragon.infrastructure.policy import Policy
    from dragon.globalservices.process import ProcessError
    DRAGON_AVAILABLE = True
except ImportError:
    DRAGON_AVAILABLE = False


logger = logging.getLogger(__name__)

@executor_registry.register("dragon")
class DragonExecutor(Executor):
    def __init__(self,logger=logger,
                 gpu_selector: str = "ZE_AFFINITY_MASK",return_stdout: bool = True, profile: bool = False):
        if not DRAGON_AVAILABLE:
            raise ModuleNotFoundError("Dragon is not available")
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._processes: Dict[str, Union[Process, ProcessGroup]] = {}
        self._results: Dict[str, Any] = {}
        self._queues: Dict[str, QueueProtocol] = {}
        self._return_stdout = return_stdout
        self._profile = profile
        self._profile_info: Dict[str, Dict] = {}
    
    def _record_profile_stop(self, task_id: str):
        """Helper method to record profiling stop time and duration"""
        if self._profile and task_id in self._profile_info and "stop" not in self._profile_info[task_id]:
            self._profile_info[task_id]["stop"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self._profile_info[task_id]["duration"] = (datetime.now() - self._profile_info[task_id]["start"]).total_seconds()
            self._profile_info[task_id]["start"] = self._profile_info[task_id]["start"].strftime("%Y-%m-%d %H:%M:%S")
    
    def start(self,job_resource: JobResource, 
                fn: Union[Callable,str], 
                task_args: Tuple = (),
                task_kwargs: Dict = {}, 
                env: Dict[str, Any] = {}):
        
        task_id = str(uuid.uuid4())
        
        nprocs = len(job_resource.nodes)*job_resource.resources[0].cpu_count
        if callable(fn) or isinstance(fn, str):
            merged_env = os.environ.copy().update(env)
            if nprocs > 1:
                req = job_resource.resources[0]
                policy = Policy(
                            placement=Policy.Placement.HOST_NAME,
                            host_name=job_resource.nodes[0],
                            cpu_affinity = req.cpus if isinstance(req, NodeResourceList) else [],
                            gpu_env_str = self._gpu_selector,
                            gpu_affinity = req.gpus if isinstance(req, NodeResourceList) else [0]
                            )
                if callable(fn):
                    q = queue_registry.create_queue("dragon")
                    self._queues[task_id] = q
                    p = Process(target=return_wrapper, args=(q, fn, task_args, task_kwargs), env=merged_env, policy=policy)
                else:
                    p = Process(target=fn, args=task_args, kwargs=task_kwargs, env=merged_env, policy=policy)
                p.start()
            else:
                p = ProcessGroup(restart=False)
                req = job_resource.resources[0]
                if callable(fn):
                    q = queue_registry.create_queue("dragon")
                    self._queues[task_id] = q
                if isinstance(req, NodeResourceCount):
                    ##Set only node level policy
                    for node,req in zip(job_resource.nodes,job_resource.resources):
                        local_policy = Policy(
                                                placement=Policy.Placement.HOST_NAME, 
                                                host_name=node, 
                                                gpu_env_str = self._gpu_selector,
                                                gpu_affinity = list(range(req.gpu_count))
                                            )
                        p.add_process(
                            nproc = req.cpu_count,
                            template=ProcessTemplate(
                                        target=fn if isinstance(fn,str) else return_wrapper, 
                                        args=task_args if isinstance(fn,str) else (q, fn, task_args, task_kwargs), 
                                        kwargs=task_kwargs if isinstance(fn,str) else {}, 
                                        env = merged_env,
                                        policy=local_policy)
                        )
                else:
                    ##Set process level policy
                    for node,req in zip(job_resource.nodes,job_resource.resources):
                        ngpus_per_process = req.gpu_count//req.cpu_count
                        for id,cpu_id in enumerate(req.cpus):
                            gpu_ids = req.gpus[id*ngpus_per_process:(id+1)*ngpus_per_process]
                            local_policy = Policy(
                                                placement=Policy.Placement.HOST_NAME, 
                                                host_name=node, 
                                                cpu_affinity = [cpu_id],
                                                gpu_env_str = self._gpu_selector,
                                                gpu_affinity = gpu_ids
                                            )
                            p.add_process(
                                nproc = 1,
                                template=ProcessTemplate(
                                                        target=fn if isinstance(fn,str) else return_wrapper, 
                                                        args=task_args if isinstance(fn,str) else (q, fn, task_args, task_kwargs), 
                                                        kwargs=task_kwargs if isinstance(fn,str) else {}, 
                                                        env = merged_env, 
                                                        policy=local_policy
                                        )
                            )
                p.start()
        else:
            self.logger.warning(f"Can only excute either a str or a callable")
            return None
        
        if self._profile:
            self._profile_info[task_id] = {"start": datetime.now()}
        
        self._processes[task_id] = p

        return task_id

    def stop(self, task_id:str, force: bool=False):
        try:
            p = self._processes[task_id]
            if force:
                p.kill()
            else:
                p.terminate()
            self._record_profile_stop(task_id)
        except Exception as e:
            self.logger.warning(f"stopping task {task_id} failed with an exception {e}")
        

    def wait(self, task_id: str, timeout:float = None):
        p = self._processes(task_id)
        if isinstance(p,Process):
            return_code = p.join(timeout)
            if return_code is None:
                self.logger.warning(f"Process {task_id} timed out after {timeout} seconds.")
                return False
            else:
                try:
                    if task_id in self._queues:
                        result = []
                        q = self._queues[task_id]
                        while not q.empty():
                            result.append(q.get())
                        self._results[task_id] = result
                        self._record_profile_stop(task_id)
                        return True
                    else:
                        self._results[task_id] = None
                        self._record_profile_stop(task_id)
                        return True
                except Exception as e:
                    self.logger.warning(f"Getting results from {task_id} failed with {e}")
                    self._results[task_id] = None
                    self._record_profile_stop(task_id)
                    return True

        else:
            try:
                p.join(timeout)
                try:
                    if task_id in self._queues:
                        result = []
                        q = self._queues[task_id]
                        while not q.empty():
                            result.append(q.get())
                        self._results[task_id] = result
                        self._record_profile_stop(task_id)
                        return True
                    else:
                        self._results[task_id] = None
                        self._record_profile_stop(task_id)
                        return True
                except Exception as e:
                    self.logger.warning(f"Getting results from {task_id} failed with {e}")
                    self._results[task_id] = None
                    self._record_profile_stop(task_id)
                    return True
            except TimeoutError:
                self.logger.warning(f"Process {task_id} timed out after {timeout} seconds.")
                return False
    
    def result(self, task_id: str, timeout:float = None):
        try:
            return self._results[task_id]
        except KeyError:
            if self.wait(task_id, timeout=timeout):
                return self._results[task_id]
            else:
                return None

    def exception(self, task_id: str):
        self.wait(task_id)
        self._record_profile_stop(task_id)
        p = self._processes[task_id]
        if isinstance(p, Process):
            returncode = p.returncode
            if returncode == 0:
                return None
            else:
                return ProcessError(f"returncode={returncode}")
        else: ##processgroup
            returncodes = [s[1] for s in p.exit_status]
            if any(code != 0 for code in returncodes):
                return ProcessError(f"returncodes={returncodes}")
            else:
                return None

    def done(self, task_id: str):
        if self._profile and task_id in self._profile_info:
            self._profile_info[task_id]["done"] += 1
        p = self._processes[task_id]
        if isinstance(p,Process):
            return not p.is_alive()
        else:
            return len(p.puids) == 0
    
    def shutdown(self, force: bool = False):
        for task_id, p in self._processes.items():
            try:
                if not self.done(task_id):
                    if force:
                        self.stop(task_id,force=True)
                    else:
                        self.wait(task_id)
            except Exception as e:
                self.logger.warning(f"Failed to kill process {task_id}: {e}")
        self._processes.clear()
        self._results.clear()

        for task_id,q in self._queues.items():
            try:
                if not q.closed and not force:
                    q.join()
                q.destroy()
            except Exception as e:
                self.logger.warning(f"Failed to close/destroy q {task_id}: {e}")
        self._queues.clear()