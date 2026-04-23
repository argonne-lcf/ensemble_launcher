import asyncio
import logging
import os
import signal
import socket
import stat
import subprocess
import uuid
from asyncio import Future as AsyncFuture
from asyncio import Task
from concurrent.futures import Executor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ensemble_launcher.config import MPIConfig
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
        tmp_dir: str = "/tmp/.mpiexec_tmp",
        return_stdout: bool = True,
        mpi_config: Optional[MPIConfig] = None,
        **kwargs,
    ):
        self.logger = logger
        self.gpu_selector = gpu_selector
        if os.path.isabs(tmp_dir):
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = os.path.join(os.getcwd(), tmp_dir)
        self._use_local_tmp = self.tmp_dir.startswith("/tmp")
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._results: Dict[str, Any] = {}
        self._return_stdout = return_stdout
        self._mpi_config = mpi_config if mpi_config is not None else MPIConfig()
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.logger.info("Initialized AsyncMPI Executor!")

    def _build_resource_cmd(self, task_id: str, job_resource: JobResource):
        """Build the mpirun resource flags from the job resource and MPIConfig.

        Returns (launcher_cmd, env, setup_files) where setup_files maps
        "gpu_affinity" -> script_content when self._use_local_tmp is True
        (the caller must distribute the script via a setup MPI job).
        """

        cfg = self._mpi_config
        ppn = job_resource.resources[0].cpu_count
        nnodes = len(job_resource.nodes)
        ngpus_per_process = (
            job_resource.resources[0].gpu_count // job_resource.resources[0].cpu_count
            if job_resource.resources[0].cpu_count > 0
            else 0
        )

        env = {}
        launcher_cmd = []
        setup_files = {}

        # Total process count
        launcher_cmd += [cfg.nprocesses_flag, str(ppn * nnodes)]

        # Processes per node
        if cfg.processes_per_node_flag:
            launcher_cmd += [cfg.processes_per_node_flag, str(ppn)]

        # Host / node list
        if cfg.hosts_flag is not None:
            if nnodes >= cfg.hostfile_threshold and cfg.hostfile_flag:
                hostfile_path = os.path.join(self.tmp_dir, f"hostfile_{task_id}.txt")
                with open(hostfile_path, "w") as f:
                    for node in job_resource.nodes:
                        f.write(f"{node}\n")
                launcher_cmd += [cfg.hostfile_flag, hostfile_path]
                self.logger.info(
                    f"Created hostfile with {nnodes} nodes at {hostfile_path}"
                )
            else:
                launcher_cmd += [cfg.hosts_flag, ",".join(job_resource.nodes)]

        # CPU affinity
        if isinstance(job_resource.resources[0], NodeResourceList):
            common_cpus = set.intersection(
                *[set(nr.cpus) for nr in job_resource.resources]
            )
            if common_cpus != set(job_resource.resources[0].cpus):
                self.logger.warning(
                    "Can't use same CPUs on all nodes. Skipping cpu binding."
                )
            else:
                cores = ":".join(map(str, job_resource.resources[0].cpus))
                if cfg.cpu_bind_method == "list" and cfg.cpu_bind_flag:
                    launcher_cmd += [cfg.cpu_bind_flag, f"list:{cores}"]
                elif cfg.cpu_bind_method == "bind-to" and cfg.cpu_bind_flag:
                    launcher_cmd += [
                        cfg.cpu_bind_flag,
                        "core",
                        "--map-by",
                        cfg.openmpi_map_by,
                    ]
                elif cfg.cpu_bind_method == "none":
                    pass
                else:
                    self.logger.warning(
                        f"Unknown cpu_bind_method '{cfg.cpu_bind_method}'. Not setting affinity."
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
                fname = (
                    f"/tmp/gpu_affinity_file_{task_id}.sh"
                    if self._use_local_tmp
                    else os.path.join(self.tmp_dir, f"gpu_affinity_file_{task_id}.sh")
                )
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
                        if self._use_local_tmp:
                            setup_files["gpu_affinity"] = bash_script
                        else:
                            if not os.path.exists(fname):
                                with open(fname, "w") as f:
                                    f.write(bash_script)
                                st = os.stat(fname)
                                os.chmod(fname, st.st_mode | stat.S_IEXEC)
                        launcher_cmd.append(fname)
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
                    if self._use_local_tmp:
                        setup_files["gpu_affinity"] = bash_script
                    else:
                        if not os.path.exists(fname):
                            with open(fname, "w") as f:
                                f.write(bash_script)
                            st = os.stat(fname)
                            os.chmod(fname, st.st_mode | stat.S_IEXEC)
                    launcher_cmd.append(fname)
                    ##Here you need to set the environment variables for each node
                    for nid, node in enumerate(job_resource.nodes):
                        env.update(
                            {
                                f"AVAILABLE_GPUS_{node}": ",".join(
                                    [str(i) for i in job_resource.resources[nid].gpus]
                                )
                            }
                        )

        return launcher_cmd, env, setup_files

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

        resource_pinning_cmd, resource_pinning_env, setup_files = (
            self._build_resource_cmd(task_id, job_resource)
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

        total_ranks = sum(res.cpu_count for res in job_resource.resources)
        local_host = socket.gethostname()
        is_local = all(node == local_host for node in job_resource.nodes)
        if total_ranks == 1 and is_local:
            cmd = task_cmd
        else:
            cfg = self._mpi_config
            cmd = (
                [cfg.launcher, *cfg.extra_launcher_flags]
                + resource_pinning_cmd
                + additional_mpi_opts
                + task_cmd
            )

        merged_env = os.environ.copy()
        merged_env.update(resource_pinning_env)
        merged_env.update(env)

        asyncio_task = asyncio.create_task(
            self._subprocess_task(
                task_id,
                cmd,
                merged_env,
                nodes=job_resource.nodes,
                setup_files=setup_files,
            )
        )
        self._tasks[task_id] = asyncio_task
        asyncio_task.add_done_callback(lambda _: self._tasks.pop(task_id, None))

        return asyncio_task

    async def write_file_to_nodes(
        self, path: str, content: str, nodes: List[str], executable: bool = False
    ) -> None:
        """Write a text file to `path` on each node in `nodes` via a 1-rank-per-node MPI job."""
        cfg = self._mpi_config
        setup_code = (
            "import os, stat\n"
            f"content = {repr(content)}\n"
            f"path = {repr(path)}\n"
            "os.makedirs(os.path.dirname(path) or '.', exist_ok=True)\n"
            "open(path, 'w').write(content)\n"
        )
        if executable:
            setup_code += "os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)\n"
        cmd = (
            [cfg.launcher]
            + [cfg.nprocesses_flag, str(len(nodes))]
            + ([cfg.processes_per_node_flag, "1"] if cfg.processes_per_node_flag else [])
            + ([cfg.hosts_flag, ",".join(nodes)] if cfg.hosts_flag else [])
            + ["python", "-c", setup_code]
        )
        self.logger.info(f"[write_file_to_nodes] writing {path} to {len(nodes)} node(s)")
        proc = await asyncio.create_subprocess_exec(*cmd)
        await proc.wait()

    async def _subprocess_task(
        self,
        task_id: str,
        cmd: List[str],
        merged_env: Dict[str, Any],
        nodes: Optional[List[str]] = None,
        setup_files: Optional[Dict] = None,
    ):
        if setup_files and setup_files.get("gpu_affinity") and nodes:
            fname = f"/tmp/gpu_affinity_file_{task_id}.sh"
            await self.write_file_to_nodes(fname, setup_files["gpu_affinity"], nodes, executable=True)
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
                start_new_session=True,
            )
        else:
            p = await asyncio.create_subprocess_exec(
                program,
                *args,
                env=merged_env,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
                start_new_session=True,
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

    def _kill_process_group(
        self, process: asyncio.subprocess.Process, force: bool
    ) -> None:
        """Send signal to the entire process group so MPI worker children are also killed."""
        if process.returncode is not None:
            return
        try:
            pgid = os.getpgid(process.pid)
            sig = signal.SIGKILL if force else signal.SIGTERM
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def shutdown(self, wait: bool = False):
        force = not wait
        for process in list(self._processes.values()):
            self._kill_process_group(process, force)
        self._processes.clear()
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()

    async def ashutdown(self, wait: bool = True) -> None:
        """Async shutdown: kill process groups and await all asyncio tasks."""
        force = not wait
        for process in list(self._processes.values()):
            self._kill_process_group(process, force)
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._processes.clear()
        self._tasks.clear()
