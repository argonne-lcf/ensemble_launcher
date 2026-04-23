import abc
import asyncio
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future as ConcurrentFuture
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union, Deque
from collections import deque

import cloudpickle

from ensemble_launcher.checkpointing import CommCheckpointData
from ensemble_launcher.checkpointing.checkpointer import _get_comm_state_class
from ensemble_launcher.comm.messages import IResultBatch, Result, TaskUpdate
from ensemble_launcher.ensemble import Task
from ensemble_launcher.logging import setup_logger

# ---------------------------------------------------------------------------
# Transport abstraction
# ---------------------------------------------------------------------------


class _ClientTransport(abc.ABC):
    """Abstract send/recv transport used by ClusterClient."""

    @abc.abstractmethod
    def connect(self, address: str, identity: bytes) -> None: ...

    @abc.abstractmethod
    def send(self, data: bytes) -> None: ...

    @abc.abstractmethod
    async def recv(self) -> bytes:
        """Await and return the next message."""
        ...

    @abc.abstractmethod
    def close(self) -> None: ...


class _ZMQTransport(_ClientTransport):
    def __init__(self):
        self._context = None
        self._socket = None

    def connect(self, address: str, identity: bytes) -> None:
        import zmq
        import zmq.asyncio

        self._context = zmq.asyncio.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, identity)
        self._socket.connect(f"tcp://{address}")

    def send(self, data: bytes) -> None:
        # zmq.asyncio.Socket.send is synchronous (only recv is overridden).
        self._socket.send(data)

    async def recv(self) -> bytes:
        return await self._socket.recv()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
        if self._context is not None:
            self._context.term()


# Maps comm_state_type names → transport class
_TRANSPORT_REGISTRY: Dict[str, Type[_ClientTransport]] = {
    "AsyncZMQCommState": _ZMQTransport,
}


# ---------------------------------------------------------------------------
# Worker pipeline
# ---------------------------------------------------------------------------


class _WorkerPipeline:
    """One independent send/recv pipeline: own thread, event loop, and ZMQ socket."""

    def __init__(self, worker_id: str, transport: _ClientTransport):
        self.worker_id = worker_id
        self.transport = transport
        self.pending: Dict[str, ConcurrentFuture] = {}
        self.lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_ready = threading.Event()
        self._recv_task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        # Profiling accumulators
        self.recv_time = 0.0
        self.deser_time = 0.0
        self.set_result_time = 0.0
        self.total_results = 0
        self._logger = None

    def start(self, node_address: str, logger) -> None:
        """Start the pipeline's dedicated thread and wait until its event loop is ready."""
        self._logger = logger
        self._thread = threading.Thread(
            target=self._run, args=(node_address, logger), daemon=True
        )
        self._thread.start()
        self._loop_ready.wait()

    def _run(self, node_address: str, logger) -> None:
        """Thread target: create event loop, connect transport, run recv loop."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self.transport.connect(node_address, self.worker_id.encode())
        self._loop_ready.set()
        try:
            self._loop.run_until_complete(self._recv_loop(logger))
        except Exception as e:
            logger.error(f"Pipeline {self.worker_id} exited with error: {e}")
        finally:
            self._loop.close()

    async def _recv_loop(self, logger) -> None:
        self._recv_task = asyncio.current_task()
        try:
            while True:
                t0 = time.perf_counter()
                raw = await self.transport.recv()
                self.recv_time += time.perf_counter() - t0
                asyncio.create_task(self._deserialize_and_set(raw))
        except asyncio.CancelledError:
            logger.info(
                f"{self.worker_id} recv={self.recv_time:.3f}s "
                f"deser={self.deser_time:.3f}s "
                f"set_result={self.set_result_time:.3f}s"
            )

    async def _deserialize_and_set(self, raw: bytes) -> None:
        t0 = time.perf_counter()
        msg: IResultBatch = await asyncio.get_running_loop().run_in_executor(
            None, cloudpickle.loads, raw
        )
        t1 = time.perf_counter()
        for result in msg.data:
            self.set_result(result)
        self.total_results += len(msg.data)
        self.deser_time += t1 - t0
        self.set_result_time += time.perf_counter() - t1
        if self._logger is not None:
            self._logger.info(
                f"{self.worker_id}: received {len(msg.data)} results "
                f"(total={self.total_results})"
            )

    def set_result(self, result: Result) -> None:
        with self.lock:
            fut = self.pending.pop(result.task_id, None)
        if fut is None or fut.done():
            return
        if result.success:
            fut.set_result(result.data)
        else:
            fut.set_exception(Exception(result.exception or "Task failed"))

    def send(self, data: bytes) -> None:
        """Thread-safe send: schedules onto this pipeline's own event loop."""
        self._loop.call_soon_threadsafe(self.transport.send, data)

    def stop(self) -> None:
        """Cancel the recv task, join the thread, and close the transport."""
        if self._loop is not None and self._recv_task is not None:
            self._loop.call_soon_threadsafe(self._recv_task.cancel)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self.transport.close()


# ---------------------------------------------------------------------------
# Node resolution
# ---------------------------------------------------------------------------


def _wait_for_path(path: str, timeout: float, poll_interval: float = 1.0) -> None:
    """Block until *path* exists (file or non-empty directory), or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while True:
        if os.path.isfile(path):
            return
        if os.path.isdir(path) and os.listdir(path):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for checkpoint path: {path}"
            )
        time.sleep(min(poll_interval, remaining))


def _resolve_node_id(checkpoint_dir: str, node_id: str) -> str:
    """Resolve a symbolic node_id to a concrete node_id.

    ``"global"`` resolves to the shortest node_id found in *checkpoint_dir*
    (the global master always has the shortest name, e.g. ``"main"``).
    Any other value is returned unchanged.
    """
    if node_id not in ["global", "local"]:
        return node_id
    else:
        if node_id == "global":
            return os.listdir(checkpoint_dir)[0]
        elif node_id == "local":
            raise NotImplementedError("local connection is not implemented")
        else:
            raise ValueError(f"Unknown node_id {node_id}")


# ---------------------------------------------------------------------------
# ClusterClient
# ---------------------------------------------------------------------------


class ClusterClient:
    """Backend-agnostic client for submitting tasks to a running orchestrator.

    Connects to a node by reading its comm checkpoint, which contains the
    actual ``host:port`` address and transport backend.  Node IDs follow the
    scheduler naming convention: ``"main"``, ``"main.w0"``, ``"main.m0.w1"``, etc.

    Set ``n_workers > 1`` to open multiple parallel send/recv pipelines, each
    with its own ZMQ socket and result-deserialization task.  Tasks are
    distributed round-robin across pipelines; results are routed back to the
    pipeline that submitted them via the ZMQ DEALER identity.

    Examples::

        # Connect to the global master (default):
        with ClusterClient("/scratch/ckpt") as client:
            fut = client.submit(task)

        # Four parallel pipelines:
        with ClusterClient("/scratch/ckpt", n_workers=4) as client:
            futs = client.map(fn, items)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        node_id: str = "global",
        client_id: Optional[str] = None,
        n_workers: int = 1,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
        checkpoint_timeout: float = 60.0,
        task_buffer_size: int = 10000,
        task_flush_interval: float = 5.0,
    ):
        """
        Args:
            checkpoint_dir:      Directory containing orchestrator checkpoint files.
            node_id:             Which node to connect to. ``"global"`` (default) resolves
                                 to the global master (shortest node_id in the checkpoint
                                 dir, e.g. ``"main"``). Pass any scheduler node id such as
                                 ``"main.w0"`` or ``"main.m0.w2"`` to connect directly to
                                 that node.
            client_id:           Optional client identity string; auto-generated if omitted.
            n_workers:           Number of parallel send/recv pipelines (default 1).
            log_dir:             Directory for log files.  When provided a file
                                 ``{log_dir}/{client_id}.log`` is created.  When
                                 ``None`` (default) logging goes to the root handler.
            log_level:           Logging level (default ``logging.INFO``).
            checkpoint_timeout:  Seconds to wait for checkpoint files to appear before
                                 raising ``TimeoutError`` (default 60).  Useful when the
                                 client is started before the orchestrator has written its
                                 first checkpoint.
            task_buffer_size:    Flush the outgoing task buffer when it reaches this many
                                 tasks (default 10000).
            task_flush_interval: Seconds between periodic flushes of the task buffer
                                 (default 5.0).
        """
        self._client_id = client_id or f"client:{uuid.uuid4().hex[:8]}"
        self._n_workers = n_workers
        self._task_buffer_size = task_buffer_size
        self._task_flush_interval = task_flush_interval
        self._task_buffer: Deque[Tuple[Task, ConcurrentFuture]] = deque()
        self._task_buffer_lock = threading.Lock()
        self._flush_stop_event = threading.Event()
        self._flush_thread: Optional[threading.Thread] = None
        self.logger = setup_logger(
            __name__, self._client_id, log_dir=log_dir, level=log_level
        )
        self._node_id = None

        # Wait for the checkpoint directory to be populated before resolving the node id.
        self.logger.info(
            f"Waiting for checkpoint directory '{checkpoint_dir}' "
            f"(timeout={checkpoint_timeout:.0f}s)"
        )
        _wait_for_path(checkpoint_dir, timeout=checkpoint_timeout)

        resolved_id = _resolve_node_id(checkpoint_dir, node_id)
        self._node_id = resolved_id
        comm_path = os.path.join(
            checkpoint_dir, *self._node_id.split("."), f"{resolved_id}_comm.json"
        )

        # Wait for the specific comm checkpoint file to be written.
        self.logger.info(f"Waiting for comm checkpoint '{comm_path}'")
        _wait_for_path(comm_path, timeout=checkpoint_timeout)
        with open(comm_path, "r") as f:
            comm_data = CommCheckpointData.model_validate_json(f.read())

        transport_cls = _TRANSPORT_REGISTRY.get(comm_data.comm_state_type)
        if transport_cls is None:
            raise ValueError(
                f"No transport registered for comm type '{comm_data.comm_state_type}'. "
                f"Known types: {list(_TRANSPORT_REGISTRY)}"
            )
        comm_cls = _get_comm_state_class(comm_data.comm_state_type)
        comm_state = comm_cls.deserialize(comm_data.comm_state_json)
        self._node_address = comm_state.my_address

        # Create one pipeline per worker; transports are connected in _run_event_loop.
        self._pipelines: List[_WorkerPipeline] = [
            _WorkerPipeline(
                worker_id=f"{self._client_id}:w{i}",
                transport=transport_cls(),
            )
            for i in range(n_workers)
        ]

    def start(self) -> None:
        """Start each pipeline's dedicated thread and wait until all are connected."""
        for pipeline in self._pipelines:
            pipeline.start(self._node_address, self.logger)
        self._flush_stop_event.clear()
        self._flush_thread = threading.Thread(
            target=self._flush_timer_loop, daemon=True
        )
        self._flush_thread.start()
        self.logger.info(
            f"Started {self._n_workers} pipeline(s) connected to "
            f"{self._node_id} at {self._node_address}"
        )

    def teardown(self) -> None:
        """Flush remaining tasks, stop the flush timer, and stop all pipelines."""
        self.flush()
        self._flush_stop_event.set()
        if self._flush_thread is not None:
            self._flush_thread.join(timeout=5.0)
        for pipeline in self._pipelines:
            pipeline.stop()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_task(
        self,
        task_or_callable: Union[Task, Callable, str],
        args: Tuple = (),
        kwargs: Dict = {},
        nnodes: int = 1,
        ppn: int = 1,
        ngpus_per_process: int = 0,
    ) -> Task:
        """Wrap a callable or shell string in a Task with the given resource spec."""
        if isinstance(task_or_callable, Task):
            return task_or_callable
        if callable(task_or_callable) or isinstance(task_or_callable, str):
            return Task(
                task_id=str(uuid.uuid4()),
                nnodes=nnodes,
                ppn=ppn,
                ngpus_per_process=ngpus_per_process,
                executable=task_or_callable,
                args=args,
                kwargs=kwargs,
            )
        raise TypeError(
            f"submit() expects a Task, callable, or str; got {type(task_or_callable)}"
        )

    def _send_batch(self, tasks: List[Task]) -> List[ConcurrentFuture]:
        """Buffer tasks and return futures. Flushes immediately when buffer is full."""
        futures: List[ConcurrentFuture] = []
        with self._task_buffer_lock:
            for task in tasks:
                fut: ConcurrentFuture = ConcurrentFuture()
                self._task_buffer.append((task, fut))
                futures.append(fut)
            if len(self._task_buffer) >= self._task_buffer_size:
                self._flush_task_buffer()
        return futures

    def _flush_task_buffer(self) -> None:
        """Drain the buffer and send one TaskUpdate per pipeline. Caller must hold _task_buffer_lock."""
        if not self._task_buffer:
            return
        items = list(self._task_buffer)
        self._task_buffer.clear()

        pipeline_batches: List[List[Task]] = [[] for _ in self._pipelines]
        for i, (task, fut) in enumerate(items):
            pipeline = self._pipelines[i % self._n_workers]
            with pipeline.lock:
                pipeline.pending[task.task_id] = fut
            pipeline_batches[i % self._n_workers].append(task)

        for pipeline, batch in zip(self._pipelines, pipeline_batches):
            if batch:
                data = cloudpickle.dumps(
                    TaskUpdate(sender=pipeline.worker_id, added_tasks=batch)
                )
                pipeline.send(data)
        self.logger.info(f"Flushed {len(items)} tasks across {self._n_workers} pipeline(s)")

    def flush(self) -> None:
        """Force-flush all buffered tasks. Safe to call from any thread."""
        with self._task_buffer_lock:
            self._flush_task_buffer()

    def _flush_timer_loop(self) -> None:
        """Periodically flush the task buffer until stop event is set."""
        while not self._flush_stop_event.is_set():
            self._flush_stop_event.wait(self._task_flush_interval)
            if not self._flush_stop_event.is_set():
                self.flush()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        task: Union[Task, Callable, str],
        *args,
        nnodes: int = 1,
        ppn: int = 1,
        ngpus_per_process: int = 0,
        **kwargs,
    ) -> ConcurrentFuture:
        """Send a task to the node. Returns a Future resolved on completion.

        *task* may be:
        - a :class:`Task` (used as-is; resource args and *args*/*kwargs* are ignored),
        - a callable — wrapped in a Task with the given resource spec,
        - a shell command string — wrapped in a Task with the given resource spec.

        Args:
            task: Task object, callable, or shell command string.
            *args: Positional arguments forwarded to the callable.
            nnodes: Number of nodes to request (ignored when *task* is a Task).
            ppn: Processes per node (ignored when *task* is a Task).
            ngpus_per_process: GPUs per process (ignored when *task* is a Task).
            **kwargs: Keyword arguments forwarded to the callable.
        """
        return self._send_batch(
            [self._to_task(task, args, kwargs, nnodes, ppn, ngpus_per_process)]
        )[0]

    def submit_batch(self, tasks: List[Task]):
        """Submit a batch of tasks to the cluster."""
        return self._send_batch(tasks)

    def map(
        self,
        fn: Union[Callable, str],
        iterable: Iterable,
        nnodes: int = 1,
        ppn: int = 1,
        ngpus_per_process: int = 0,
        **kwargs,
    ) -> List[ConcurrentFuture]:
        """Submit *fn* applied to each element of *iterable* in a single batch.

        All tasks are sent in one :class:`TaskUpdate` message per pipeline.
        Returns a list of :class:`~concurrent.futures.Future` objects in the
        same order as *iterable*.

        Args:
            fn: Callable or shell command string to apply to each item.
            iterable: Items to map over; each becomes the sole positional arg.
            nnodes: Number of nodes per task.
            ppn: Processes per node per task.
            ngpus_per_process: GPUs per process per task.
            **kwargs: Keyword arguments forwarded to *fn* for every call.

        Example::

            futs = client.map(my_func, [1, 2, 3], ppn=4)
            results = [f.result() for f in futs]
        """
        tasks = [
            self._to_task(
                fn,
                item if isinstance(item, tuple) else (item,),
                kwargs,
                nnodes,
                ppn,
                ngpus_per_process,
            )
            for item in iterable
        ]
        return self._send_batch(tasks)

    def __enter__(self) -> "ClusterClient":
        self.start()
        return self

    def __exit__(self, *_):
        self.teardown()
