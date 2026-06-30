import asyncio
import base64
import cloudpickle
import logging
import os
import shlex
import socket
import uuid
from asyncio import Future as AsyncFuture
from concurrent.futures import Executor
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from ensemble_launcher.scheduler.resource import JobResource

from .utils import executor_registry

logger = logging.getLogger(__name__)

_DEFAULT_SSH_ARGS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=no",
]

def _generate_python_exec_command(tmp_fname: str) -> str:
    script = [
        "import cloudpickle",
        f"f = open('{tmp_fname}', 'rb')",
        "fn, args, kwargs = cloudpickle.load(f)",
        "f.close()",
        "fn(*args, **kwargs)",
    ]
    return ";".join(script)


@executor_registry.register("async_ssh", type="async")
class AsyncSSHExecutor(Executor):
    """Launches child processes on remote nodes via SSH (fire-and-forget).

    The remote command is backgrounded with nohup so SSH exits immediately.
    Child lifecycle is tracked via ZMQ heartbeats, not the SSH process.
    """

    def __init__(
        self,
        logger=logger,
        tmp_dir: str = "/tmp/.ssh_exec_tmp",
        return_stdout: bool = True,
        ssh_args: Optional[List[str]] = None,
        **kwargs,
    ):
        self.logger = logger
        if os.path.isabs(tmp_dir):
            self.tmp_dir = tmp_dir
        else:
            self.tmp_dir = os.path.join(os.getcwd(), tmp_dir)
        self._use_local_tmp = self.tmp_dir.startswith("/tmp")
        self._tasks: Dict[str, asyncio.Task] = {}
        self._remote_pids: Dict[str, Tuple[str, int]] = {}
        self._return_stdout = return_stdout
        self._ssh_args = ssh_args if ssh_args is not None else list(_DEFAULT_SSH_ARGS)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.logger.info("Initialized AsyncSSH Executor!")

    async def _write_env_file(
        self, env: Dict[str, Any], nodes: List[str]
    ) -> str:
        env_file = os.path.join(self.tmp_dir, f"env_{uuid.uuid4().hex[:8]}.sh")
        content = ""
        for k, v in env.items():
            content += f"export {k}={shlex.quote(str(v))}\n"

        if self._use_local_tmp:
            os.makedirs(os.path.dirname(env_file) or ".", exist_ok=True)
            with open(env_file, "w") as f:
                f.write(content)
            success = await self.write_file_to_nodes(
                env_file, content, nodes
            )
            if not success:
                self.logger.error("Failed to distribute env file to nodes!")
                raise RuntimeError("Failed to distribute env file to nodes")
        else:
            with open(env_file, "w") as f:
                f.write(content)

        return env_file

    async def _build_ssh_cmd(
        self,
        node: str,
        remote_cmd: List[str],
        env: Dict[str, Any],
        log_file: Optional[str] = None,
    ) -> List[str]:
        env_file = await self._write_env_file(env, [node])
        inner_cmd = f". {shlex.quote(env_file)} && {shlex.join(remote_cmd)}"

        if log_file:
            bg_cmd = (
                f"nohup bash -c {shlex.quote(inner_cmd)} "
                f"> {shlex.quote(log_file)} 2>&1 & echo $!"
            )
        else:
            bg_cmd = (
                f"nohup bash -c {shlex.quote(inner_cmd)} "
                f"</dev/null >/dev/null 2>&1 & echo $!"
            )

        return ["ssh"] + self._ssh_args + [node, bg_cmd]

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
    ):
        task_id = str(uuid.uuid4())
        asyncio_task = asyncio.create_task(
            self.asubmit(
                job_resource=job_resource,
                task=task,
                task_args=task_args,
                task_kwargs=task_kwargs,
                env=env,
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
        run_dir: str = os.getcwd(),
        stdout_file: Optional[str] = None,
        stderr_file: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> AsyncFuture:
        if task_id is None:
            task_id = str(uuid.uuid4())

        node = job_resource.nodes[0]
        local_host = socket.gethostname()
        is_local = node == local_host

        if callable(task):
            tmp_fname = os.path.join(self.tmp_dir, f"callable_{task_id}.pkl")
            with open(tmp_fname, "wb") as f:
                cloudpickle.dump((task, task_args, task_kwargs), f)
            task_cmd = ["python", "-c", _generate_python_exec_command(tmp_fname)]
        elif isinstance(task, str):
            task_cmd = [s.strip() for s in task.split()]
        elif isinstance(task, list):
            task_cmd = task
        else:
            self.logger.warning("Can only execute either a callable, string, or list")
            return None

        if is_local:
            cmd = task_cmd
            merged_env = os.environ.copy()
            merged_env.update(env)
        else:
            log_file = stdout_file if stdout_file else None
            cmd = await self._build_ssh_cmd(node, task_cmd, env, log_file=log_file)
            merged_env = os.environ.copy()

        result = await self._subprocess_task(
            task_id,
            cmd,
            merged_env,
            run_dir=run_dir if is_local else None,
            is_local=is_local,
            stdout_file=stdout_file if is_local else None,
            stderr_file=stderr_file if is_local else None,
        )

        if not is_local and result:
            stdout_part = result.split(",")[0].strip()
            try:
                remote_pid = int(stdout_part)
                self._remote_pids[task_id] = (node, remote_pid)
                self.logger.info(
                    f"Remote process started on {node} with PID {remote_pid}"
                )
            except ValueError:
                self.logger.warning(
                    f"Could not parse remote PID from SSH output: {stdout_part!r}"
                )

        return result

    async def _subprocess_task(
        self,
        task_id: str,
        cmd: List[str],
        merged_env: Dict[str, Any],
        run_dir: Optional[str] = None,
        is_local: bool = False,
        stdout_file: Optional[str] = None,
        stderr_file: Optional[str] = None,
    ):
        self.logger.info(f"executing: {' '.join(cmd)}")

        program = cmd[0]
        args = cmd[1:]

        base_dir = run_dir if run_dir else os.getcwd()
        stdout_fh = None
        stderr_fh = None

        try:
            if is_local and stdout_file:
                stdout_path = os.path.join(base_dir, stdout_file)
                os.makedirs(os.path.dirname(stdout_path) or ".", exist_ok=True)
                stdout_fh = open(stdout_path, "w")
                stdout_target = stdout_fh
            elif self._return_stdout:
                stdout_target = asyncio.subprocess.PIPE
            else:
                stdout_target = asyncio.subprocess.DEVNULL

            if is_local and stderr_file:
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
                self.logger.error(
                    f"Submitting task {task_id} failed with error: {e}"
                )
                raise e

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
                    f"Task {task_id} failed with return code {p.returncode}. "
                    f"stderr: {err_str}"
                )
            return out_str + "," + err_str

        finally:
            if stdout_fh:
                stdout_fh.close()
            if stderr_fh:
                stderr_fh.close()

    async def write_file_to_nodes(
        self,
        path: str,
        content: str,
        nodes: List[str],
        executable: bool = False,
        encoded: bool = False,
    ) -> bool:
        if not self._use_local_tmp:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(content)
            if executable:
                import stat

                st = os.stat(path)
                os.chmod(path, st.st_mode | stat.S_IEXEC)
            return True

        local_host = socket.gethostname()
        encoded_path = base64.b64encode(path.encode("utf-8")).decode("utf-8")

        CHUNK_SIZE = 3000
        chunks = (
            [content[i : i + CHUNK_SIZE] for i in range(0, len(content), CHUNK_SIZE)]
            if content
            else [""]
        )

        self.logger.info(
            f"[write_file_to_nodes] writing {path} to {len(nodes)} node(s) "
            f"in {len(chunks)} chunk(s)"
        )

        async def _write_to_node(node: str) -> bool:
            is_local = node == local_host

            for i, chunk in enumerate(chunks):
                mode = "w" if i == 0 else "a"
                if encoded:
                    setup_code = (
                        "import base64, os\n"
                        f"p = base64.b64decode('{encoded_path}').decode('utf-8')\n"
                        "os.makedirs(os.path.dirname(p) or '.', exist_ok=True)\n"
                        f"with open(p, '{mode}') as f:\n"
                        f"    f.write('{chunk}')\n"
                    )
                else:
                    encoded_chunk = base64.b64encode(
                        chunk.encode("utf-8")
                    ).decode("utf-8")
                    setup_code = (
                        "import base64, os\n"
                        f"p = base64.b64decode('{encoded_path}').decode('utf-8')\n"
                        "os.makedirs(os.path.dirname(p) or '.', exist_ok=True)\n"
                        f"with open(p, '{mode}') as f:\n"
                        f"    f.write(base64.b64decode('{encoded_chunk}')"
                        f".decode('utf-8'))\n"
                    )

                if is_local:
                    cmd = ["python", "-c", setup_code]
                else:
                    remote_str = shlex.join(["python", "-c", setup_code])
                    cmd = ["ssh"] + self._ssh_args + [node, remote_str]

                for retry in range(3):
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, std_err = await proc.communicate()
                    if proc.returncode != 0:
                        self.logger.warning(
                            f"Copying chunk {i} to {node} failed, "
                            f"retrying {retry + 1}/3"
                        )
                        self.logger.warning(f"stderr: {std_err.decode()}")
                    else:
                        break
                if proc.returncode != 0:
                    self.logger.warning(
                        f"Copying file to {node} failed after 3 retries!"
                    )
                    return False

            if executable:
                chmod_code = (
                    "import base64, os, stat\n"
                    f"p = base64.b64decode('{encoded_path}').decode('utf-8')\n"
                    "os.chmod(p, os.stat(p).st_mode | stat.S_IEXEC)\n"
                )
                if is_local:
                    cmd = ["python", "-c", chmod_code]
                else:
                    remote_str = shlex.join(["python", "-c", chmod_code])
                    cmd = ["ssh"] + self._ssh_args + [node, remote_str]

                for retry in range(3):
                    proc = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    _, std_err = await proc.communicate()
                    if proc.returncode != 0:
                        self.logger.warning(
                            f"chmod on {node} failed, retrying {retry + 1}/3"
                        )
                        self.logger.warning(f"stderr: {std_err.decode()}")
                    else:
                        break
                if proc.returncode != 0:
                    self.logger.warning(
                        f"chmod on {node} failed after 3 retries!"
                    )
                    return False

            return True

        results = await asyncio.gather(*[_write_to_node(n) for n in nodes])
        return all(results)

    async def _kill_remote_processes(self, force: bool = False) -> None:
        sig = "KILL" if force else "TERM"
        kill_tasks = []
        for task_id, (node, pid) in list(self._remote_pids.items()):
            kill_cmd = ["ssh"] + self._ssh_args + [node, f"kill -{sig} {pid} 2>/dev/null; true"]
            self.logger.info(f"Sending SIG{sig} to PID {pid} on {node}")
            kill_tasks.append(
                asyncio.create_subprocess_exec(
                    *kill_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
            )
        if kill_tasks:
            procs = await asyncio.gather(*kill_tasks, return_exceptions=True)
            for p in procs:
                if isinstance(p, asyncio.subprocess.Process):
                    await p.communicate()
        self._remote_pids.clear()

    def shutdown(self, wait: bool = False):
        for task in list(self._tasks.values()):
            task.cancel()
        self._tasks.clear()
        self._remote_pids.clear()

    async def ashutdown(self, wait: bool = False) -> None:
        await self._kill_remote_processes(force=not wait)
        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        self._tasks.clear()
