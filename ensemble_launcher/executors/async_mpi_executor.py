import asyncio
import base64
import logging
import os
import shutil
import signal
import socket
import stat
import uuid
from asyncio import Future as AsyncFuture
from concurrent.futures import Executor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle

from ensemble_launcher.config import MPIConfig
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

from .utils import (
    executor_registry,
    gen_affinity_bash_script_1,
    gen_affinity_bash_script_2,
)

logger = logging.getLogger(__name__)


def _is_node_local(path: str) -> bool:
    """True if `path` is on node-local storage (under /tmp), so each node needs its own copy."""
    path = os.path.abspath(path)
    return path == "/tmp" or path.startswith("/tmp/")


def _generate_python_exec_command(tmp_fname: str) -> str:
    """
    Generates a Python command string that reads a base64 cloudpickled (fn, args, kwargs)
    from `tmp_fname` and calls it. The returned string can be run using `python -c "<command>"`.
    """
    script = [
        "import cloudpickle",
        "import base64",
        f"f = open('{tmp_fname}', 'r')",
        "task_str = f.read()",
        "f.close()",
        "task_bytes=base64.b64decode(task_str)",
        "fn, args, kwargs = cloudpickle.loads(task_bytes)",
        "fn(*args, **kwargs)",
    ]
    return ";".join(script)


@executor_registry.register("async_mpi", type="async")
class AsyncMPIExecutor(Executor):
    def __init__(
        self,
        logger=logger,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        tmp_dir: Optional[str] = None,
        return_stdout: bool = True,
        mpi_config: Optional[MPIConfig] = None,
        **kwargs,
    ):
        self.logger = logger
        self.gpu_selector = gpu_selector
        # A tmp_dir this executor made is removed on shutdown; one passed in is kept.
        self._owns_tmp_dir = tmp_dir is None
        if tmp_dir is None:
            # A fresh directory under the working directory: on HPC systems that is a
            # shared filesystem, so each file is written once, not copied to every node.
            self.tmp_dir = os.path.join(os.getcwd(), f".mpi_tmp_{uuid.uuid4().hex}")
        elif os.path.isabs(tmp_dir):
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = os.path.join(os.getcwd(), tmp_dir)
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
        path -> script content; the caller writes each one (executable) with
        write_file_to_nodes before launching.
        """

        cfg = self._mpi_config
        ppn = job_resource.resources[0].cpu_count
        nnodes = len(job_resource.nodes)
        # Rational gpus-per-rank as (num_gpu_ids, ppn) rather than a floored
        # integer division, so fractional requests (e.g. 1 GPU id shared by
        # 4 ranks) are not silently rounded down to 0. num_gpu_ids is the
        # count of *distinct* granted ids (always integral -- a fractional
        # amount still reserves its id once), which is exactly what the
        # affinity scripts index into via AVAILABLE_GPUS.
        num_gpu_ids = len(job_resource.resources[0].gpus)

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

            if num_gpu_ids > 0:
                ##defaults to Aurora (Level zero)
                self.logger.info(f"Using {self.gpu_selector} for pinning GPUs")
                common_gpus = set.intersection(
                    *[
                        set(node_resource.gpus)
                        for node_resource in job_resource.resources
                    ]
                )
                use_common_gpus = common_gpus == set(job_resource.resources[0].gpus)
                fname = os.path.join(self.tmp_dir, f"gpu_affinity_file_{task_id}.sh")
                if use_common_gpus:
                    if nnodes == 1 and ppn == 1:
                        env.update(
                            {
                                self.gpu_selector: ",".join(
                                    [str(i) for i in job_resource.resources[0].gpus]
                                )
                            }
                        )
                    else:
                        bash_script = gen_affinity_bash_script_1(
                            num_gpu_ids, ppn, self.gpu_selector
                        )
                        setup_files[fname] = bash_script
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
                        num_gpu_ids, ppn, self.gpu_selector
                    )
                    setup_files[fname] = bash_script
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
        run_dir: str = os.getcwd(),
        stdout_file: Optional[str] = None,
        stderr_file: Optional[str] = None,
        **kwargs,
    ):
        task_id = str(uuid.uuid4())
        asyncio_task = asyncio.create_task(
            self.asubmit(
                job_resource=job_resource,
                task=task,
                task_args=task_args,
                task_kwargs=task_kwargs,
                env=env,
                mpi_args=mpi_args,
                mpi_kwargs=mpi_kwargs,
                serial_launch=serial_launch,
                run_dir=run_dir,
                stdout_file=stdout_file,
                stderr_file=stderr_file,
                task_id=task_id,
            )
        )
        self._tasks[task_id] = asyncio_task
        asyncio_task.add_done_callback(lambda _: self._tasks.pop(task_id, None))
        return asyncio_task

    async def asubmit(
        self,
        job_resource: JobResource,
        task: Union[str, Callable, List],
        task_args: Tuple = (),
        task_kwargs: Dict[str, Any] = {},
        env: Dict[str, Any] = {},
        mpi_args: Tuple = (),
        mpi_kwargs: Dict[str, Any] = {},
        serial_launch: bool = False,
        run_dir: str = os.getcwd(),
        stdout_file: Optional[str] = None,
        stderr_file: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> AsyncFuture:
        if task_id is None:
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
            task_str = base64.b64encode(
                cloudpickle.dumps((task, task_args, task_kwargs))
            ).decode("utf-8")
            if not await self.write_file_to_nodes(
                tmp_fname, content=task_str, nodes=job_resource.nodes, encoded=True
            ):
                self.logger.warning(f"Failed to write {tmp_fname} to the task's nodes")
                return None
            task_cmd = [
                "python",
                "-c",
                _generate_python_exec_command(tmp_fname),
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

        result = await self._subprocess_task(
            task_id,
            cmd,
            merged_env,
            nodes=job_resource.nodes,
            setup_files=setup_files,
            run_dir=run_dir,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )

        return result

    async def write_file_to_nodes(
        self,
        path: str,
        content: str,
        nodes: List[str],
        executable: bool = False,
        encoded: bool = False,
    ) -> bool:
        """Write a text file to `path` on each node in `nodes`.

        A path outside /tmp is on a shared filesystem: it is written once, here, through
        a temporary file and an atomic rename, so no node ever sees a partial file.
        A node-local path (under /tmp) is written on each node by 1-rank-per-node MPI jobs.

        Each chunk is written at its own byte offset (pwrite + ftruncate), so running a
        chunk again is a no-op on nodes that already have it. A launch that failed on some
        nodes is retried on all of them, and the copy resumes from that chunk instead of
        appending the chunk twice. Each rank prints ``<hostname> ok`` on success, so a
        failure names the nodes that did not report. The last chunk also checks the file
        size and applies the chmod, so neither costs an extra launch.
        """
        if not _is_node_local(path):
            tmp_path = f"{path}.{uuid.uuid4().hex}.tmp"
            try:
                os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                with open(tmp_path, "w", encoding="utf-8") as f:
                    f.write(content)
                if executable:
                    os.chmod(tmp_path, os.stat(tmp_path).st_mode | stat.S_IEXEC)
                os.replace(tmp_path, path)
            except OSError as e:
                self.logger.warning(f"[write_file_to_nodes] writing {path} failed: {e}")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                return False
            return True

        cfg = self._mpi_config

        ## remove any duplicates
        nodes = list(set(nodes))

        base_cmd = (
            [cfg.launcher]
            + [cfg.nprocesses_flag, str(len(nodes))]
            + (
                [cfg.processes_per_node_flag, "1"]
                if cfg.processes_per_node_flag
                else []
            )
        )

        if len(nodes) >= cfg.hostfile_threshold and cfg.hostfile_flag:
            hostfile_path = os.path.join(
                self.tmp_dir, f"hostfile_{str(uuid.uuid4())}.txt"
            )
            with open(hostfile_path, "w") as f:
                for node in nodes:
                    f.write(f"{node}\n")
            base_cmd += [cfg.hostfile_flag, hostfile_path]
        else:
            base_cmd += [cfg.hosts_flag, ",".join(nodes)] if cfg.hosts_flag else []

        data = content.encode("utf-8")
        # Keep each chunk's literal in the command at ~3000 bytes, under the 4096-byte
        # MPI/OS command limits. Raw chunks are base64-encoded (4/3 larger) for transport;
        # already-encoded content is embedded as is.
        chunk_size = 3000 if encoded else 2250
        chunks = [
            (offset, data[offset : offset + chunk_size])
            for offset in range(0, len(data), chunk_size)
        ] or [(0, b"")]

        # Base64 encode the path to prevent shell issues if the path contains spaces or special characters
        encoded_path = base64.b64encode(path.encode("utf-8")).decode("utf-8")

        self.logger.info(
            f"[write_file_to_nodes] writing {path} to {len(nodes)} node(s) in {len(chunks)} chunk(s)"
        )

        for i, (offset, chunk) in enumerate(chunks):
            if encoded:
                payload = f"'{chunk.decode('ascii')}'.encode('ascii')"
            else:
                # Base64 payload sidesteps all remote Bash parsing, quoting, and newline mangling.
                payload = f"base64.b64decode('{base64.b64encode(chunk).decode('ascii')}')"
            code = (
                "import base64, os, socket, stat, sys\n"
                f"p = base64.b64decode('{encoded_path}').decode('utf-8')\n"
                "os.makedirs(os.path.dirname(p) or '.', exist_ok=True)\n"
                f"d = {payload}\n"
                "fd = os.open(p, os.O_WRONLY | os.O_CREAT, 0o644)\n"
                "try:\n"
                "    n = 0\n"
                "    while n < len(d):\n"
                f"        n += os.pwrite(fd, d[n:], {offset} + n)\n"
                f"    os.ftruncate(fd, {offset} + len(d))\n"
                "finally:\n"
                "    os.close(fd)\n"
            )
            if i == len(chunks) - 1:
                code += (
                    f"if os.stat(p).st_size != {len(data)}:\n"
                    f"    sys.exit(f'{{socket.gethostname()}}: size {{os.stat(p).st_size}} != {len(data)}')\n"
                )
                if executable:
                    code += "os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)\n"
            code += "print(socket.gethostname(), 'ok', flush=True)\n"

            self.logger.debug(f"Executing chunk {i + 1}/{len(chunks)} for {path}")
            if not await self._run_on_nodes(
                base_cmd + ["python", "-c", code],
                nodes,
                f"chunk {i + 1}/{len(chunks)} of {path}",
            ):
                return False

        return True

    async def _run_on_nodes(
        self, cmd: List[str], nodes: List[str], what: str, retries: int = 3
    ) -> bool:
        """Run a 1-rank-per-node job whose ranks print ``<hostname> ok``; retry on failure."""
        for attempt in range(1, retries + 1):
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            out, err = await proc.communicate()
            proc._transport.close()
            if proc.returncode == 0:
                return True
            reported = {
                line.split()[0].split(".")[0]
                for line in out.decode(errors="replace").splitlines()
                if line.endswith(" ok")
            }
            missing = [n for n in nodes if n.split(".")[0] not in reported]
            self.logger.warning(
                f"[write_file_to_nodes] {what} failed (rc={proc.returncode}, "
                f"attempt {attempt}/{retries}); {len(missing)}/{len(nodes)} node(s) "
                f"did not report ok: {missing[:8]}{' ...' if len(missing) > 8 else ''}"
            )
            self.logger.warning(f"stderr: {err.decode(errors='replace')[-2000:]}")
        self.logger.warning(f"[write_file_to_nodes] {what} failed after {retries} attempts")
        return False

    async def _subprocess_task(
        self,
        task_id: str,
        cmd: List[str],
        merged_env: Dict[str, Any],
        nodes: Optional[List[str]] = None,
        setup_files: Optional[Dict] = None,
        run_dir: Optional[str] = None,
        stdout_file: Optional[str] = None,
        stderr_file: Optional[str] = None,
    ):
        for fname, content in (setup_files or {}).items():
            if not await self.write_file_to_nodes(
                fname, content, nodes or [], executable=True
            ):
                self.logger.warning(f"Failed to write {fname} to the task's nodes")
        self.logger.info(f"executing: {' '.join(cmd)}")

        program = cmd[0]
        args = cmd[1:]

        base_dir = run_dir if run_dir else os.getcwd()
        stdout_fh = None
        stderr_fh = None

        try:
            if stdout_file:
                stdout_path = os.path.join(base_dir, stdout_file)
                os.makedirs(os.path.dirname(stdout_path) or ".", exist_ok=True)
                stdout_fh = open(stdout_path, "w")
                stdout_target = stdout_fh
            elif self._return_stdout:
                stdout_target = asyncio.subprocess.PIPE
            else:
                stdout_target = asyncio.subprocess.DEVNULL

            if stderr_file:
                stderr_path = os.path.join(base_dir, stderr_file)
                os.makedirs(os.path.dirname(stderr_path) or ".", exist_ok=True)
                stderr_fh = open(stderr_path, "w")
                stderr_target = stderr_fh
            elif self._return_stdout:
                stderr_target = asyncio.subprocess.PIPE
            else:
                stderr_target = asyncio.subprocess.DEVNULL

            try:
                p = await asyncio.create_subprocess_exec(
                    program,
                    *args,
                    env=merged_env,
                    stdout=stdout_target,
                    stderr=stderr_target,
                    start_new_session=True,
                    cwd=run_dir,
                )
            except Exception as e:
                self.logger.error(f"Submitting task {task_id} failed with error: {e}")
                raise e

            self._processes[task_id] = p

            std_out, std_err = await p.communicate()

            if stdout_file and self._return_stdout:
                stdout_fh.flush()
                with open(stdout_path, "r") as f:
                    out_str = f.read()
            else:
                out_str = std_out.decode() if std_out else ""

            if stderr_file and self._return_stdout:
                stderr_fh.flush()
                with open(stderr_path, "r") as f:
                    err_str = f.read()
            else:
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
            if stdout_fh:
                stdout_fh.close()
            if stderr_fh:
                stderr_fh.close()
            if task_id in self._processes:
                proc = self._processes.pop(task_id)
                if proc._transport is not None:
                    proc._transport.close()

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

    def shutdown(self, wait: bool = False, **kwargs):
        force = not wait
        for process in list(self._processes.values()):
            self._kill_process_group(process, force)
            if process._transport is not None:
                process._transport.close()
        self._processes.clear()
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        self._remove_tmp_dir()

    def _remove_tmp_dir(self) -> None:
        """Remove the default ``.mpi_tmp_<uuid>`` directory, if this executor made it."""
        if self._owns_tmp_dir:
            shutil.rmtree(self.tmp_dir, ignore_errors=True)
            self._owns_tmp_dir = False

    async def ashutdown(self, wait: bool = False) -> None:
        """Async shutdown: kill process groups and await all asyncio tasks."""
        force = not wait
        for process in list(self._processes.values()):
            self._kill_process_group(process, force)
            if process._transport is not None:
                process._transport.close()
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._processes.clear()
        self._tasks.clear()
        self._remove_tmp_dir()
