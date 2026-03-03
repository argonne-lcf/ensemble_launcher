import asyncio
import logging
import os
import socket
import stat
import subprocess
import uuid
from asyncio import Future as AsyncFuture
from asyncio import Task
from concurrent.futures import Executor
from typing import Any, Callable, Dict, List, Tuple, Union

from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

from .utils import (
    executor_registry,
    gen_affinity_bash_script_1,
    gen_affinity_bash_script_2,
    generate_python_exec_command,
)

logger = logging.getLogger(__name__)


@executor_registry.register("async_mpi", type="async")
class AsyncMPIExecutor(Executor):
    def __init__(
        self,
        logger=logger,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        tmp_dir: str = ".mpiexec_tmp",
        mpiexec: str = "mpirun",
        return_stdout: bool = True,
        use_ppn: bool = True,
        **kwargs,
    ):
        self.logger = logger
        self.gpu_selector = gpu_selector
        self.tmp_dir = os.path.join(os.getcwd(), tmp_dir)
        self.mpiexec = mpiexec
        self._processes: Dict[str, subprocess.Popen] = {}
        self._results: Dict[str, Any] = {}
        self._return_stdout = return_stdout
        self.use_ppn = use_ppn
        self._cpu_binding_option = kwargs.get("cpu_binding_option", "--cpu-bind")
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.logger.info("Initialized AsyncMPI Executor!")

    def _build_resource_cmd(self, task_id: str, job_resource: JobResource):
        """Function to build the mpi cmd from the job resources"""

        ppn = job_resource.resources[0].cpu_count
        nnodes = len(job_resource.nodes)
        ngpus_per_process = (
            job_resource.resources[0].gpu_count // job_resource.resources[0].cpu_count
        )

        env = {}
        launcher_cmd = []

        launcher_cmd.append("-np")
        launcher_cmd.append(f"{ppn * nnodes}")
        if self.use_ppn:
            launcher_cmd.append("-ppn")
            launcher_cmd.append(f"{ppn}")

            if nnodes > 512:
                # Use hostfile for large node counts to avoid command line length limits
                hostfile_id = str(uuid.uuid4())
                hostfile_path = os.path.join(
                    self.tmp_dir, f"hostfile_{hostfile_id}.txt"
                )
                with open(hostfile_path, "w") as f:
                    for node in job_resource.nodes:
                        f.write(f"{node}\n")
                launcher_cmd.append("--hostfile")
                launcher_cmd.append(hostfile_path)
                self.logger.info(
                    f"Created hostfile with {nnodes} nodes at {hostfile_path}"
                )
            else:
                launcher_cmd.append("--hosts")
                launcher_cmd.append(f"{','.join(job_resource.nodes)}")

        ##resource pinning
        if isinstance(job_resource.resources[0], NodeResourceList):
            common_cpus = set.intersection(
                *[set(node_resource.cpus) for node_resource in job_resource.resources]
            )

            use_common_cpus = common_cpus == set(job_resource.resources[0].cpus)
            if use_common_cpus:
                cores = ":".join(map(str, job_resource.resources[0].cpus))
            else:
                ##TODO: implement host file option
                self.logger.warning(
                    f"Can't use same CPUs on all the nodes. Over subscribing cores"
                )
                cores = ":".join(map(str, job_resource.resources[0].cpus))
            # TODO: Add bind to for openmpi
            if self._cpu_binding_option == "--cpu-bind":
                launcher_cmd.append("--cpu-bind")
                launcher_cmd.append(f"list:{cores}")
            else:
                self.logger.warning(
                    f"Unknown bind to core option {self._cpu_binding_option}. Not setting affinity"
                )

            if ngpus_per_process > 0:
                ##defaults to Aurora (Level zero)
                self.logger.info(f"Using {self.gpu_selector} for pinning GPUs")
                common_gpus = set.intersection(
                    *[
                        set(node_resource.gpus)
                        for node_resource in job_resource.resources
                    ]
                )
                use_common_gpus = common_gpus == set(job_resource.resources[0].gpus)
                if use_common_gpus:
                    if nnodes == 1 and ppn == 1:
                        env.update(
                            {
                                "ZE_AFFINITY_MASK": ",".join(
                                    [str(i) for i in job_resource.resources[0].gpus]
                                )
                            }
                        )
                    else:
                        bash_script = gen_affinity_bash_script_1(
                            ngpus_per_process, self.gpu_selector
                        )
                        fname = os.path.join(
                            self.tmp_dir, f"gpu_affinity_file_{task_id}.sh"
                        )
                        if not os.path.exists(fname):
                            with open(fname, "w") as f:
                                f.write(bash_script)
                            st = os.stat(fname)
                            os.chmod(fname, st.st_mode | stat.S_IEXEC)
                        launcher_cmd.append(f"{fname}")
                        ##set environment variables
                        env.update(
                            {
                                "AVAILABLE_GPUS": ",".join(
                                    [str(i) for i in job_resource.resources[0].gpus]
                                )
                            }
                        )
                else:
                    bash_script = gen_affinity_bash_script_2(
                        ngpus_per_process, self.gpu_selector
                    )
                    fname = os.path.join(
                        self.tmp_dir, f"gpu_affinity_file_{task_id}.sh"
                    )
                    if not os.path.exists(fname):
                        with open(fname, "w") as f:
                            f.write(bash_script)
                        st = os.stat(fname)
                        os.chmod(fname, st.st_mode | stat.S_IEXEC)
                    launcher_cmd.append(f"{fname}")
                    ##Here you need to set the environment variables for each node
                    for nid, node in enumerate(job_resource.nodes):
                        env.update(
                            {
                                f"AVAILABLE_GPUS_{node}": ",".join(
                                    [str(i) for i in job_resource.resources[nid].gpus]
                                )
                            }
                        )

        return launcher_cmd, env

    def submit(
        self,
        job_resource: JobResource,
        task: Union[str, Callable, List],
        task_args: Tuple = (),
        task_kwargs: Dict[str, Any] = {},
        env: Dict[str, Any] = {},
        mpi_args: Tuple = (),
        mpi_kwargs: Dict[str, Any] = {},
        serial_launch: bool = False,
    ) -> AsyncFuture:
        # task is a str command
        task_id = str(uuid.uuid4())

        resource_pinning_cmd, resource_pinning_env = self._build_resource_cmd(
            task_id, job_resource
        )

        additional_mpi_opts = []
        additional_mpi_opts.extend(list(mpi_args))
        for k, v in mpi_kwargs.items():
            additional_mpi_opts.extend([str(k), str(v)])

        if callable(task):
            tmp_fname = os.path.join(self.tmp_dir, f"callable_{task_id}.pkl")
            task_cmd = [
                "python",
                "-c",
                generate_python_exec_command(task, task_args, task_kwargs, tmp_fname),
            ]
        elif isinstance(task, str):
            task_cmd = [s.strip() for s in task.split()]
        elif isinstance(task, List):
            task_cmd = task
        else:
            self.logger.warning("Can only execute either a callable or a string")
            return None

        if (
            " ".join(resource_pinning_cmd).strip() == "-np 1"
            and len(additional_mpi_opts) == 0
        ):
            cmd = task_cmd
        else:
            cmd = [self.mpiexec] + resource_pinning_cmd + additional_mpi_opts + task_cmd

        merged_env = os.environ.copy()
        merged_env.update(resource_pinning_env)
        merged_env.update(env)

        asyncio_task = asyncio.create_task(
            self._subprocess_task(task_id, cmd, merged_env)
        )

        return asyncio_task

    async def _subprocess_task(
        self, task_id: str, cmd: List[str], merged_env: Dict[str, Any]
    ):
        self.logger.info(f"executing: {' '.join(cmd)}")

        # We separate the executable from the arguments
        program = cmd[0]
        args = cmd[1:]

        if self._return_stdout:
            p = await asyncio.create_subprocess_exec(
                program,
                *args,
                env=merged_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        else:
            p = await asyncio.create_subprocess_exec(
                program,
                *args,
                env=merged_env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )

        self._processes[task_id] = p

        try:
            std_out, std_err = await p.communicate()

            out_str = std_out.decode() if std_out else ""
            err_str = std_err.decode() if std_err else ""

            if p.returncode != 0:
                self.logger.error(
                    f"Task {task_id} failed with return code {p.returncode}"
                )
                self.logger.error(f"stderr: {err_str}")
                self.logger.error(f"stdout: {out_str}")
                raise RuntimeError(
                    f"Task {task_id} failed with return code {p.returncode}"
                )
            return out_str + "," + err_str

        finally:
            if task_id in self._processes:
                del self._processes[task_id]

    def shutdown(self, force: bool = False):
        for task_id, process in self._processes.items():
            if process.returncode is None:
                if force:
                    process.kill()
                else:
                    process.terminate()
        self._processes.clear()
