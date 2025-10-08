from typing import Any, Dict, Callable, List, Union, Tuple
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList
import subprocess
import logging
from .utils import gen_affinity_bash_script_1, gen_affinity_bash_script_2, generate_python_exec_command, executor_registry
import os
import stat
import uuid
import shlex
import socket

from .base import Executor

logger = logging.getLogger(__name__)

@executor_registry.register("mpi")
class MPIExecutor(Executor):
    def __init__(self,gpu_selector: str = "ZE_AFFINITY_MASK",
                 tmp_dir:str = ".mpiexec_tmp",
                 mpiexec:str = "mpirun"):
        self.gpu_selector = gpu_selector
        self.tmp_dir = os.path.join(os.getcwd(), tmp_dir)
        self.mpiexec = mpiexec
        self._processes: Dict[str,subprocess.Popen] = {}
        self._results: Dict[str, Any] = {}
        os.makedirs(self.tmp_dir,exist_ok=True)

    def _build_resource_cmd(self, task_id:str, job_resource: JobResource):
        """Function to build the mpi cmd from the job resources"""

        ppn = job_resource.resources[0].cpu_count
        nnodes = len(job_resource.nodes)
        ngpus_per_process = job_resource.resources[0].gpu_count//job_resource.resources[0].cpu_count

        env = {}
        launcher_cmd = []

        launcher_cmd.append("-np")
        launcher_cmd.append(f"{ppn*nnodes}")
        if not(len(job_resource.nodes) == 1 and job_resource.nodes[0] == socket.gethostname()):
            launcher_cmd.append("--hosts")
            launcher_cmd.append(f"{','.join(job_resource.nodes)}")

        ##resource pinning
        if isinstance(job_resource.resources[0],NodeResourceList):
            common_cpus = set.intersection(*[set(node_resource.cpus) for node_resource in job_resource.resources])

            use_common_cpus = common_cpus == set(job_resource.resources[0].cpus)
            if use_common_cpus:
                cores = ":".join(map(str, job_resource.resources[0].cpus))
            else:
                ##TODO: implement host file option
                logger.warning(f"Can't use same CPUs on all the nodes. Over subscribing cores")
                cores = ":".join(map(str, job_resource.resources[0].cpus))
            launcher_cmd.append("--cpu-bind")
            launcher_cmd.append(f"list:{cores}")
        
            if ngpus_per_process > 0:
                ##defaults to Aurora (Level zero)
                logger.info(f"Using {self.gpu_selector} for pinning GPUs")
                common_gpus = set.intersection(*[set(node_resource.gpus) for node_resource in job_resource.resources])
                use_common_gpus = common_gpus == set(job_resource.resources[0].gpus)
                if use_common_gpus:
                    if nnodes == 1 and ppn == 1:
                        env.update({"ZE_AFFINITY_MASK": ",".join([str(i) for i in job_resource.resources[0].gpus])})
                    else:
                        bash_script = gen_affinity_bash_script_1(ngpus_per_process,self.gpu_selector)
                        fname = os.path.join(self.tmp_dir,f"gpu_affinity_file_{task_id}.sh")
                        if not os.path.exists(fname):
                            with open(fname, "w") as f:
                                f.write(bash_script)
                            st = os.stat(fname)
                            os.chmod(fname,st.st_mode | stat.S_IEXEC)
                        launcher_cmd.append(f"{fname}")
                        ##set environment variables
                        env.update({"AVAILABLE_GPUS": ",".join([str(i) for i in job_resource.resources[0].gpus])})
                else:
                    bash_script = gen_affinity_bash_script_2(ngpus_per_process,self.gpu_selector)
                    fname = os.path.join(self.tmp_dir,f"gpu_affinity_file_{task_id}.sh")
                    if not os.path.exists(fname):
                        with open(fname, "w") as f:
                            f.write(bash_script)
                        st = os.stat(fname)
                        os.chmod(fname,st.st_mode | stat.S_IEXEC)
                    launcher_cmd.append(f"{fname}")
                    ##Here you need to set the environment variables for each node
                    for nid,node in enumerate(job_resource.nodes):
                        env.update({f"AVAILABLE_GPUS_{node}": ",".join([str(i) for i in job_resource.resources[nid].gpus])})

        return launcher_cmd, env

    def start(self,job_resource: JobResource, 
                task: Union[str, Callable], 
                task_args: Tuple = (), 
                task_kwargs: Dict[str,Any] = {}, 
                env: Dict[str, Any] = {},
                mpi_args: Tuple = (),
                mpi_kwargs:Dict[str, Any] = {}):
        # task is a str command
        task_id = str(uuid.uuid4())

        resource_pinning_cmd, resource_pinning_env = self._build_resource_cmd(task_id,job_resource)

        additional_mpi_opts = []
        additional_mpi_opts.extend(list(mpi_args))
        for k,v in mpi_kwargs.items():
            additional_mpi_opts.extend([str(k),str(v)])

        if callable(task):
            task_cmd = ["python", "-c", generate_python_exec_command(task,task_args,task_kwargs)]
        elif isinstance(task,str):
            task_cmd = [s.strip() for s in task.split()]
        else:
            logger.warning("Can only execute either a callable or a string")
            return None

        cmd = [self.mpiexec] + resource_pinning_cmd + additional_mpi_opts + task_cmd

        merged_env = os.environ.copy()
        merged_env.update(resource_pinning_env)
        merged_env.update(env)

        logger.debug(f"executing: {' '.join(cmd)}")
        p = subprocess.Popen(cmd, env=merged_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        self._processes[task_id] = p
        return task_id
    
    def stop(self, task_id: str, force: bool = False):
        try:
            if force:
                self._processes[task_id].kill()
            else:
                self._processes[task_id].terminate()
            return True
        except Exception as e:
            logger.warning(f"Failed to kill task {task_id} with an exception {e}")
        return False

    def wait(self, task_id: str, timeout: float = None):
        process = self._processes[task_id]
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Process {task_id} timed out after {timeout} seconds.")
            return False
        stdout, stderr = process.communicate()
        self._results[task_id] = stdout
        return True
    
    def result(self, task_id: str, timeout:float = None):
        try:
            return self._results[task_id]
        except KeyError:
            if self.wait(timeout=timeout):
                return self._results[task_id]
            else:
                return None
    
    def exception(self, task_id: str):
        self.wait(task_id)
        return_code = self._processes[task_id].poll()
        if return_code == 0:
            return None
        return subprocess.CalledProcessError(
            returncode=return_code,
            cmd=self._processes[task_id].args,
            output=self._results.get(task_id, None)
        )

    def done(self, task_id: str):
        process = self._processes[task_id]
        return process.poll() is not None
    
    def shutdown(self, force:bool = False):
        for task_id, process in self._processes.items():
            try:
                if process.poll() is None:
                    if force:
                        process.kill()
                    else:
                        self.wait(task_id)
            except Exception as e:
                logger.warning(f"Failed to kill process {task_id}: {e}")
        self._processes.clear()
        self._results.clear()