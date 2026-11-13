import asyncio
import collections
import logging
import os
import secrets
import threading
import uuid
from collections import deque
from concurrent.futures import Future as ConcurrentFuture
from concurrent.futures import wait as wait_futures
from typing import Callable, Deque, Dict, Iterable, List, Optional, Tuple, Union

from ensemble_launcher.checkpointing import CommCheckpointData
from ensemble_launcher.comm import AsyncCommState, transport_registry
from ensemble_launcher.comm.messages import Message, Result, TaskUpdate
from ensemble_launcher.ensemble import Task
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.orchestrator.discovery import _resolve_node_id, _wait_for_path

_SEEN_MAX = 10000  # message ids remembered to drop resends


class ClusterClient:
    """Submit tasks to a running orchestrator node and get ``concurrent.futures.Future``s.

    Connects to a node by reading its comm checkpoint (address, transport, secret). Node ids
    follow the scheduler naming: ``"main"``, ``"main.w0"``, ``"main.m0.w1"``, ...

    One connection, one background thread running one asyncio loop. The loop runs the
    connection's front end, receives result batches, decodes them and resolves the futures
    (keyed by task uid), and flushes the task buffer on a timer. The socket and the ACK
    protocol live in the connection's IOProcess -- in a process of its own when
    ``io_process`` is on (with ``req_res``) -- so decoding a large batch here never delays an
    ACK. ``submit`` / ``map`` only buffer; ``flush`` (automatic when the buffer is full and
    every ``task_flush_interval``) sends the buffer as one ``TaskUpdate``.

    Examples::

        with ClusterClient("/scratch/ckpt") as client:     # the global master
            fut = client.submit(task)
            futs = client.map(fn, items)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        node_id: str = "global",
        client_id: Optional[str] = None,
        log_level: int = logging.INFO,
        checkpoint_timeout: float = 60.0,
        task_buffer_size: int = 10000,
        task_flush_interval: float = 0.5,
        io_process: Optional[bool] = None,
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
            log_level:           Logging level (default ``logging.INFO``).
            checkpoint_timeout:  Seconds to wait for checkpoint files to appear before
                                 raising ``TimeoutError`` (default 60), for a client started
                                 before the orchestrator wrote its first checkpoint.
            task_buffer_size:    Flush the task buffer when it reaches this many tasks
                                 (default 10000).
            task_flush_interval: Seconds between periodic flushes of the task buffer
                                 (default 0.5).
            io_process:          Run the connection's IOProcess in its own process. None
                                 (default) follows the cluster's ``comm_io_process``. Only
                                 takes effect when the cluster uses ``req_res``.
        """
        self._client_id = client_id or f"client-{secrets.token_hex(4)}"
        self._task_buffer_size = task_buffer_size
        self._task_flush_interval = task_flush_interval
        self.logger = setup_logger(__name__, self._client_id, level=log_level)

        self.logger.info(
            f"Waiting for checkpoint directory '{checkpoint_dir}' "
            f"(timeout={checkpoint_timeout:.0f}s)"
        )
        _wait_for_path(checkpoint_dir, timeout=checkpoint_timeout)
        self._node_id = _resolve_node_id(checkpoint_dir, node_id)
        node_dir = os.path.join(checkpoint_dir, *self._node_id.split("."))

        comm_path = os.path.join(node_dir, f"{self._node_id}_comm.json")
        self.logger.info(f"Waiting for comm checkpoint '{comm_path}'")
        _wait_for_path(comm_path, timeout=checkpoint_timeout)
        with open(comm_path, "r") as f:
            comm_data = CommCheckpointData.model_validate_json(f.read())

        secret_path = os.path.join(node_dir, f"{self._node_id}_secret")
        cluster_secret = None
        if os.path.exists(secret_path):
            with open(secret_path, "r") as f:
                cluster_secret = f.read().strip()
            self.logger.info("Loaded cluster secret from checkpoint")
        else:
            self.logger.warning("No cluster secret file found — connecting without authentication")

        comm_state = AsyncCommState.deserialize(comm_data.comm_state_json)
        self._req_res = comm_state.req_res
        self._send_retries = comm_state.send_retries
        self._send_timeout = comm_state.send_timeout
        transport = comm_state.data_transport_state
        if transport is None:
            raise RuntimeError(f"{self._node_id} has no data transport to connect to")
        self._node_address = next(
            (cs.address for cs in transport.server_connections.values() if cs.address is not None),
            None,
        )
        conn_cls = transport_registry[transport.transport_type]["client_connection"]
        self._conn = conn_cls(
            identity=self._client_id,
            secret_id=cluster_secret or secrets.token_hex(4),
            remote_address=self._node_address,
            req_res=self._req_res,
            io_process=comm_state.io_process if io_process is None else io_process,
        )

        # Shared with the loop thread, guarded by _lock.
        self._lock = threading.Lock()
        self._pending: Dict[bytes, ConcurrentFuture] = {}  # task uid -> future
        self._sends: set = set()  # in-flight _send calls, waited for by teardown
        self._buffer: Deque[Tuple[Task, ConcurrentFuture, Optional[List[ConcurrentFuture]]]] = deque()
        # Loop thread only.
        self._seen: collections.OrderedDict = collections.OrderedDict()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None
        self._thread: Optional[threading.Thread] = None
        self._ready = threading.Event()
        self._start_error: Optional[BaseException] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Open the connection on the background loop; returns once it is connected."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=asyncio.run, args=(self._main(),), daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._start_error is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
            raise self._start_error
        self.logger.info(f"Connected to {self._node_id} at {self._node_address}")

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._main_task = asyncio.current_task()
        try:
            await self._conn.open()
        except BaseException as e:
            self._start_error = e
            self._ready.set()
            return
        self._ready.set()
        try:
            await asyncio.gather(self._recv_loop(), self._flush_loop())
        except asyncio.CancelledError:
            pass
        finally:
            await self._conn.close()

    def teardown(self) -> None:
        """Send what is still buffered, then close the connection and stop the thread."""
        if self._thread is None:
            return
        self.flush()
        with self._lock:
            sends = list(self._sends)
        wait_futures(sends, timeout=self._send_timeout * max(1, self._send_retries) + 5.0)
        loop, task = self._loop, self._main_task
        if loop is not None and task is not None and not loop.is_closed():
            loop.call_soon_threadsafe(task.cancel)
        self._thread.join(timeout=10.0)
        self._thread = None

    def __enter__(self) -> "ClusterClient":
        self.start()
        return self

    def __exit__(self, *_):
        self.teardown()

    # ------------------------------------------------------------------
    # Loop thread
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        """Result batches -> futures. Decoded inline: the ACKs are the IOProcess's job."""
        while True:
            frames = await self._conn.recv()
            try:
                msg = Message.from_byte_array(frames[1:])  # frames[0]: the sender's identity
                if self._req_res and msg.message_id is not None:
                    if msg.message_id in self._seen:  # a resend after a lost ACK
                        continue
                    self._seen[msg.message_id] = None
                    if len(self._seen) > _SEEN_MAX:
                        self._seen.popitem(last=False)
                results = msg.to_results()
            except Exception as e:
                self.logger.error(f"Could not decode a result message: {e!r}")
                continue
            for result in results:
                self._resolve(result)
            self.logger.debug(f"Received {len(results)} results")

    async def _flush_loop(self) -> None:
        while True:
            await asyncio.sleep(self._task_flush_interval)
            self.flush()

    async def _send(self, frames: List[bytes], uids: List[bytes]) -> None:
        """Send one TaskUpdate; with req_res retry on ACK timeout, then fail its tasks."""
        attempts = self._send_retries if self._req_res else 1
        for attempt in range(attempts):
            if await self._conn.send(frames, timeout=self._send_timeout):
                return
            if self._req_res:
                self.logger.warning(f"Send failed, attempt {attempt + 1}/{attempts}")
        self.logger.error(f"Send of {len(uids)} tasks failed after {attempts} attempt(s)")
        for uid in uids:
            self._resolve(Result(uid=uid, success=False,
                                 exception=f"Send failed after {attempts} attempt(s)"))  # fmt: skip

    def _send_done(self, sent: ConcurrentFuture) -> None:
        with self._lock:
            self._sends.discard(sent)

    def _resolve(self, result: Result) -> None:
        with self._lock:
            fut = self._pending.pop(result.uid, None)
        if fut is None or fut.done():
            return
        if result.success:
            fut.set_result(result.data)
        else:
            fut.set_exception(Exception(result.exception or "Task failed"))

    # ------------------------------------------------------------------
    # Buffering (any thread)
    # ------------------------------------------------------------------

    def _send_batch(
        self,
        tasks: List[Task],
        dependencies: Optional[List[Optional[List[ConcurrentFuture]]]] = None,
    ) -> List[ConcurrentFuture]:
        """Buffer tasks and return their futures; flush when the buffer is full."""
        futures: List[ConcurrentFuture] = []
        deps = dependencies if dependencies is not None else [None] * len(tasks)
        with self._lock:
            for task, d in zip(tasks, deps):
                fut: ConcurrentFuture = ConcurrentFuture()
                self._buffer.append((task, fut, d))
                futures.append(fut)
            full = len(self._buffer) >= self._task_buffer_size
        if full:
            self.flush()
        return futures

    def flush(self) -> None:
        """Send every buffered task whose dependencies are done, as one TaskUpdate. Tasks
        still waiting on dependencies stay buffered for a later flush. Safe from any thread."""
        if self._loop is None:
            return  # not started; start() and the timer will send them
        with self._lock:
            if not self._buffer:
                return
            ready, waiting = [], deque()
            for task, fut, deps in self._buffer:
                if deps and not all(d.done() for d in deps):
                    waiting.append((task, fut, deps))
                    continue
                self._pending[task.uid] = fut
                ready.append(task)
            self._buffer = waiting
        if not ready:
            return
        msg = TaskUpdate.from_tasks(ready, sender=self._client_id)
        if self._req_res:
            msg.message_id = uuid.uuid4().hex
        sent = asyncio.run_coroutine_threadsafe(
            self._send(msg.to_byte_array(), [t.uid for t in ready]), self._loop
        )
        with self._lock:
            self._sends.add(sent)
        sent.add_done_callback(self._send_done)
        self.logger.info(f"Flushed {len(ready)} tasks ({len(waiting)} waiting on dependencies)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _to_task(
        self,
        task_or_callable: Union[Task, Callable, str],
        args: Tuple = (),
        kwargs: Optional[Dict] = None,
        nnodes: int = 1,
        ppn: int = 1,
        ngpus_per_process: Union[int, float] = 0,
        serialize_executable_by_value: bool = True,
    ) -> Task:
        """Wrap a callable or shell string in a Task with the given resource spec."""
        if isinstance(task_or_callable, Task):
            return task_or_callable
        if callable(task_or_callable) or isinstance(task_or_callable, str):
            return Task(
                task_id=uuid.uuid4().hex,  # 32 chars: fits META_DTYPE
                nnodes=nnodes,
                ppn=ppn,
                ngpus_per_process=ngpus_per_process,
                executable=task_or_callable,
                args=args,
                kwargs=kwargs or {},
                serialize_executable_by_value=serialize_executable_by_value,
            )
        raise TypeError(
            f"submit() expects a Task, callable, or str; got {type(task_or_callable)}"
        )

    def submit(
        self,
        task: Union[Task, Callable, str],
        *args,
        nnodes: int = 1,
        ppn: int = 1,
        ngpus_per_process: Union[int, float] = 0,
        dependencies: Optional[List[ConcurrentFuture]] = None,
        serialize_executable_by_value: bool = True,
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
            dependencies: Futures that must be done before this task is sent.
            serialize_executable_by_value: When False, serialize the callable
                by reference (module + qualname) using pickle instead of
                cloudpickle. This preserves per-worker caches across calls.
            **kwargs: Keyword arguments forwarded to the callable.
        """
        return self._send_batch(
            [self._to_task(task, args, kwargs, nnodes, ppn, ngpus_per_process,
                           serialize_executable_by_value=serialize_executable_by_value)],
            dependencies=[dependencies],
        )[0]  # fmt: skip

    def submit_batch(
        self,
        tasks: List[Task],
        dependencies: Optional[List[List[ConcurrentFuture]]] = None,
    ) -> List[ConcurrentFuture]:
        """Submit a batch of tasks to the cluster."""
        return self._send_batch(tasks, dependencies=dependencies)

    def map(
        self,
        fn: Union[Callable, str],
        iterable: Iterable,
        nnodes: int = 1,
        ppn: int = 1,
        ngpus_per_process: Union[int, float] = 0,
        dependencies: Optional[List[ConcurrentFuture]] = None,
        serialize_executable_by_value: bool = True,
        **kwargs,
    ) -> List[ConcurrentFuture]:
        """Submit *fn* applied to each element of *iterable* in a single batch.

        Returns a list of :class:`~concurrent.futures.Future` objects in the
        same order as *iterable*.

        Args:
            fn: Callable or shell command string to apply to each item.
            iterable: Items to map over; each becomes the sole positional arg
                (a tuple item is unpacked as the positional args).
            nnodes: Number of nodes per task.
            ppn: Processes per node per task.
            ngpus_per_process: GPUs per process per task.
            dependencies: Futures every task must wait for.
            serialize_executable_by_value: When False, serialize the callable
                by reference (module + qualname) using pickle instead of
                cloudpickle. This preserves per-worker caches across calls.
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
                serialize_executable_by_value=serialize_executable_by_value,
            )
            for item in iterable
        ]
        return self._send_batch(
            tasks, dependencies=[dependencies] * len(tasks) if dependencies else None
        )
