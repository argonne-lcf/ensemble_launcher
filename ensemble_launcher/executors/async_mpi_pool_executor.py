import asyncio
import itertools
import pathlib
import subprocess
import sys
import uuid
from asyncio import Future as AsyncFuture
from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple, Union

import cloudpickle
import zmq
import zmq.asyncio

from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from .utils import executor_registry

_MPI_POOL_SCRIPT = str(pathlib.Path(__file__).parent / "mpi_pool.py")


def _build_mpirun_cmd(mpi_info: Dict[str, str], np_workers: int, socket_path: str) -> list:
    cmd = ["mpirun"]
    for flag, value in mpi_info.items():
        cmd.extend([flag, value])
    # rank 0 = master/gateway, ranks 1..np_workers = workers
    cmd.extend([sys.executable, _MPI_POOL_SCRIPT, "--socket-path", socket_path])
    return cmd


@executor_registry.register("async_mpi_processpool", type="async")
class AsyncMPIPoolExecutor:
    def __init__(
        self,
        logger: Logger,
        mpi_info: Dict[str, str],
        gpu_selector: str = "ZE_AFFINITY_MASK",
        **kwargs,
    ):
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = kwargs.pop("return_stdout", False)
        self._mpi_info: Dict[str, str] = mpi_info

        hosts = self._mpi_info["--hosts"].split(",")
        ppn = int(self._mpi_info.get("-ppn", "0"))
        np = int(self._mpi_info.get("-np", len(hosts) * ppn))

        if ppn == 0 and np == 0:
            raise ValueError("mpi info needs either ppn or np")
        if ppn == 0:
            ppn = np // len(hosts)
        if np == 0:
            np = ppn * len(hosts)

        cpu_binding = self._mpi_info.get("--cpu-bind", None)
        cpu_binding = (
            list(map(int, cpu_binding.split("list:")[-1].split(":")))
            if cpu_binding is not None
            else list(range(ppn))
        )

        # Workers are ranks 1..np; rank 0 is the master/gateway
        self._cpu_to_pid = {
            (hname, cpu_binding[local_rank]): hid * ppn + local_rank + 1
            for hid, hname in enumerate(hosts)
            for local_rank in range(ppn)
        }

        self._socket_path = str(
            f"/tmp/mpi_pool_{uuid.uuid4().hex}.ipc"
        )

        cmd = _build_mpirun_cmd(self._mpi_info, np, self._socket_path)
        self.logger.info(f"Launching MPI pool: {' '.join(cmd)}")
        self._server_proc = subprocess.Popen(cmd)

        # ROUTER socket; rank 0 DEALER connects with identity b"mpi_pool"
        self._ctx = zmq.asyncio.Context()
        self._sock = self._ctx.socket(zmq.ROUTER)
        self._sock.bind(f"ipc://{self._socket_path}")

        self._rank0_identity: Optional[bytes] = None
        self._ready = asyncio.Event()
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
            _, msg_id, status, value = msg   # ("result", msg_id, "ok"/"err", value)
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

        payload = cloudpickle.dumps((
            "task",
            msg_id,
            worker_id,
            cloudpickle.dumps(fn),
            cloudpickle.dumps(task_args),
            cloudpickle.dumps(task_kwargs),
            env,
        ))

        async def _send():
            if not self._ready.is_set():
                await self._ready.wait()
            await self._sock.send_multipart([self._rank0_identity, payload])

        asyncio.ensure_future(_send())
        return future

    def shutdown(self, wait: bool = True):
        async def _send_shutdown():
            if not self._ready.is_set():
                await self._ready.wait()
            await self._sock.send_multipart(
                [self._rank0_identity, cloudpickle.dumps(("shutdown",))]
            )

        if self._ready.is_set():
            asyncio.ensure_future(_send_shutdown())

        if self._recv_task and not self._recv_task.done():
            self._recv_task.cancel()

        self._sock.close()
        self._ctx.term()

        self._server_proc.terminate()
        try:
            self._server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.logger.warning("MPI pool did not exit cleanly; killing.")
            self._server_proc.kill()
            self._server_proc.wait()

        socket_file = pathlib.Path(self._socket_path)
        if socket_file.exists():
            socket_file.unlink()
