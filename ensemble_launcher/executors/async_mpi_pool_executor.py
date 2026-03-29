import asyncio
import itertools
import os
import pathlib
import subprocess
import sys
import uuid
from asyncio import Future as AsyncFuture
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cloudpickle
import zmq
import zmq.asyncio

from ensemble_launcher.config import MPIConfig
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from .utils import executor_registry

_MPI_POOL_SCRIPT = str(pathlib.Path(__file__).parent / "mpi_pool.py")


def _write_hostfile(hosts: List[str]) -> str:
    path = f"/tmp/mpi_hostfile_{uuid.uuid4().hex}.txt"
    with open(path, "w") as f:
        f.write("\n".join(hosts) + "\n")
    return path


def _build_mpirun_cmd(
    mpi_info: Dict[str, Any],
    mpi_config: MPIConfig,
    socket_path: str,
) -> list:
    """Build the mpirun command from structured mpi_info and MPIConfig.

    mpi_info keys (all optional except "np"):
      "np"            – total MPI rank count
      "ppn"           – ranks per node
      "hosts"         – comma-separated host list
      "cpu_binding"   – colon-separated core IDs, e.g. "0:1:2:3"
      "rankfile_path" – path to a pre-written rankfile (when use_rankfile=True)
    """
    cfg = mpi_config
    cmd = [cfg.launcher, *cfg.extra_launcher_flags]

    # Total process count
    cmd += [cfg.nprocesses_flag, str(mpi_info["np"])]

    # Processes per node
    if cfg.processes_per_node_flag and "ppn" in mpi_info:
        cmd += [cfg.processes_per_node_flag, str(mpi_info["ppn"])]

    # Rankfile takes precedence over --hosts + --cpu-bind when active
    if cfg.use_rankfile and cfg.rankfile_flag and "rankfile_path" in mpi_info:
        cmd += [cfg.rankfile_flag, mpi_info["rankfile_path"]]
    else:
        # Host / node list
        if "hosts" in mpi_info and cfg.hosts_flag is not None:
            host_list = mpi_info["hosts"].split(",")
            if len(host_list) >= cfg.hostfile_threshold and cfg.hostfile_flag:
                cmd += [cfg.hostfile_flag, _write_hostfile(host_list)]
            else:
                cmd += [cfg.hosts_flag, mpi_info["hosts"]]

        # CPU affinity
        if (
            "cpu_binding" in mpi_info
            and cfg.cpu_bind_method != "none"
            and cfg.cpu_bind_flag
        ):
            cpu_str = mpi_info["cpu_binding"]
            if cfg.cpu_bind_method == "list":
                # Intel MPI / MPICH / Cray / aprun:  --cpu-bind list:0:1:2:3
                cmd += [cfg.cpu_bind_flag, f"list:{cpu_str}"]
            elif cfg.cpu_bind_method == "bind-to":
                # OpenMPI:  --bind-to core --map-by <map_by>
                cmd += [cfg.cpu_bind_flag, "core", "--map-by", cfg.openmpi_map_by]

    # rank 0 = master/gateway, ranks 1..np_workers = workers
    cmd += [sys.executable, _MPI_POOL_SCRIPT, "--socket-path", socket_path]
    return cmd


@executor_registry.register("async_mpi_processpool", type="async")
class AsyncMPIPoolExecutor:
    def __init__(
        self,
        logger: Logger,
        mpi_info: Dict[str, str],
        cpu_to_pid: Dict[Tuple[str, int], int],
        gpu_selector: str = "ZE_AFFINITY_MASK",
        mpi_config: Optional[MPIConfig] = None,
        **kwargs,
    ):
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = kwargs.pop("return_stdout", False)
        self._mpi_info: Dict[str, str] = mpi_info
        self._mpi_config = mpi_config if mpi_config is not None else MPIConfig()
        self._cpu_to_pid = cpu_to_pid

        self._socket_path = str(f"/tmp/mpi_pool_{uuid.uuid4().hex}.ipc")

        launch_env = os.environ.copy()
        cwd = os.getcwd()
        existing_pythonpath = launch_env.get("PYTHONPATH", "")
        launch_env["PYTHONPATH"] = f"{cwd}:{existing_pythonpath}" if existing_pythonpath else cwd
        self.logger.info(f"MPI pool mpi_info: {self._mpi_info}")
        cmd = _build_mpirun_cmd(self._mpi_info, self._mpi_config, self._socket_path)
        self.logger.info(f"MPI pool cmd: {' '.join(cmd)}")
        self._server_proc = subprocess.Popen(cmd, cwd=cwd, env=launch_env)

        # ROUTER socket; rank 0 DEALER connects with identity b"mpi_pool"
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(f"ipc://{self._socket_path}")

        self._rank0_identity: Optional[bytes] = None
        self._ready = asyncio.Event()
        self._shutdown = asyncio.Event()
        self._pending: Dict[int, AsyncFuture] = {}
        self._msg_counter = itertools.count()
        self._recv_task: Optional[asyncio.Task] = None

        self.logger.info("Initialized AsyncMPIPoolExecutor!")

    def _ensure_recv_loop(self):
        if self._recv_task is None or self._recv_task.done():
            self._recv_task = asyncio.ensure_future(self._recv_loop())

    async def _recv_loop(self):
        while True:
            try:
                identity, data = await self._sock.recv_multipart()
            except zmq.ZMQError:
                break
            msg = cloudpickle.loads(data)
            if msg[0] == "ready":
                self._rank0_identity = identity
                self._ready.set()
                continue

            if identity == self._rank0_identity and msg == "done":
                self._shutdown.set()
                self.logger.debug("MPI pool shutdown complete")
                continue

            _, msg_id, status, value = msg  # ("result", msg_id, "ok"/"err", value)
            future = self._pending.pop(msg_id, None)
            if future is None or future.cancelled():
                continue
            if status == "err":
                future.set_exception(value)
            else:
                future.set_result(value)

    def submit(
        self,
        job_resource: JobResource,
        fn: Union[Callable, str],
        task_args: Tuple = (),
        task_kwargs: Dict = {},
        env: Dict[str, Any] = {},
        **kwargs,
    ) -> AsyncFuture:
        np = sum([res.cpu_count for res in job_resource.resources])
        if np > 1:
            raise ValueError("AsyncMPIPoolExecutor can only execute serial tasks")

        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus[0]

        if cpu_id is None:
            raise ValueError(
                "Not setting cpu affinity could cause deadlocks. Raising error....."
            )

        if req.gpu_count > 0:
            if isinstance(req, NodeResourceCount):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpu_count])
                self.logger.warning(
                    "Received non-zero gpu request using NodeResourceCount. Oversubscribing"
                )
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env = {**env, self._gpu_selector: gpu_ids}

        worker_id = self._cpu_to_pid[(job_resource.nodes[0], cpu_id)]

        self._ensure_recv_loop()

        loop = asyncio.get_event_loop()
        future: AsyncFuture = loop.create_future()
        msg_id = next(self._msg_counter)
        self._pending[msg_id] = future

        payload = cloudpickle.dumps(
            (
                "task",
                msg_id,
                worker_id,
                cloudpickle.dumps(fn),
                cloudpickle.dumps(task_args),
                cloudpickle.dumps(task_kwargs),
                env,
            )
        )

        async def _send():
            if not self._ready.is_set():
                await self._ready.wait()
            await self._sock.send_multipart([self._rank0_identity, payload])

        asyncio.ensure_future(_send())
        return future

    async def ashutdown(self, wait: bool = True):
        self.logger.info("ashutdown: starting")

        async def _send_shutdown():
            if not self._ready.is_set():
                self.logger.info("ashutdown: waiting for MPI pool ready")
                await self._ready.wait()
            self.logger.info("ashutdown: sending shutdown command to MPI pool")
            await self._sock.send_multipart(
                [self._rank0_identity, cloudpickle.dumps(("shutdown",))]
            )

        if self._ready.is_set():
            await _send_shutdown()
            self.logger.info("ashutdown: waiting for 'done' ack from MPI pool")
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=30)
                self.logger.info("ashutdown: received 'done' ack")
            except asyncio.TimeoutError:
                self.logger.warning(
                    "ashutdown: timed out waiting for MPI pool 'done'; forcing termination"
                )
        else:
            self.logger.warning("ashutdown: MPI pool never became ready, skipping shutdown command")

        if self._recv_task and not self._recv_task.done():
            self.logger.info("ashutdown: cancelling recv loop")
            self._recv_task.cancel()

        self.logger.info("ashutdown: closing ZMQ socket")
        self._sock.close()
        self.logger.info("ashutdown: terminating ZMQ context")
        self._ctx.term()
        self.logger.info("ashutdown: ZMQ teardown complete")

        # Wait for the MPI process to exit; SIGTERM if it hasn't already.
        self.logger.info(f"ashutdown: waiting for MPI process (pid={self._server_proc.pid}) to exit")
        try:
            self._server_proc.wait(timeout=5)
            self.logger.info("ashutdown: MPI process exited normally")
        except subprocess.TimeoutExpired:
            self.logger.info("ashutdown: MPI pool still running; sending SIGTERM")
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=10)
                self.logger.info("ashutdown: MPI process exited after SIGTERM")
            except subprocess.TimeoutExpired:
                self.logger.warning("ashutdown: MPI pool did not exit after SIGTERM; sending SIGKILL")
                self._server_proc.kill()
                self._server_proc.wait()
                self.logger.info("ashutdown: MPI process killed")

        socket_file = pathlib.Path(self._socket_path)
        if socket_file.exists():
            socket_file.unlink()
            self.logger.info(f"ashutdown: cleaned up IPC socket {self._socket_path}")

        self.logger.info("ashutdown: complete")
