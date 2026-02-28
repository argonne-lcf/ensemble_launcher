import abc
import logging
import os
import threading
import uuid
from concurrent.futures import Future as ConcurrentFuture
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type, Union

import cloudpickle

from ensemble_launcher.checkpointing import CommCheckpointData
from ensemble_launcher.checkpointing.checkpointer import _get_comm_state_class
from ensemble_launcher.comm.messages import Result, TaskUpdate
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
    def recv(self, timeout_ms: int = 100) -> Optional[bytes]:
        """Return the next message or None on timeout."""
        ...

    @abc.abstractmethod
    def close(self) -> None: ...


class _ZMQTransport(_ClientTransport):
    def __init__(self):
        self._context = None
        self._socket = None

    def connect(self, address: str, identity: bytes) -> None:
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.DEALER)
        self._socket.setsockopt(zmq.IDENTITY, identity)
        self._socket.connect(f"tcp://{address}")

    def send(self, data: bytes) -> None:
        self._socket.send(data)

    def recv(self, timeout_ms: int = 100) -> Optional[bytes]:
        import zmq

        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        try:
            return self._socket.recv()
        except zmq.Again:
            return None

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
# Node resolution
# ---------------------------------------------------------------------------


def _resolve_node_id(checkpoint_dir: str, node_id: str) -> str:
    """Resolve a symbolic node_id to a concrete node_id.

    ``"global"`` resolves to the shortest node_id found in *checkpoint_dir*
    (the global master always has the shortest name, e.g. ``"main"``).
    Any other value is returned unchanged.
    """
    if node_id != "global":
        return node_id

    candidates = [
        fname[: -len("_comm.json")]
        for fname in os.listdir(checkpoint_dir)
        if fname.endswith("_comm.json")
    ]
    if not candidates:
        raise FileNotFoundError(f"No comm checkpoints found in '{checkpoint_dir}'")
    return min(candidates, key=len)


# ---------------------------------------------------------------------------
# ClusterClient
# ---------------------------------------------------------------------------


class ClusterClient:
    """Backend-agnostic client for submitting tasks to a running orchestrator.

    Connects to a node by reading its comm checkpoint, which contains the
    actual ``host:port`` address and transport backend.  Node IDs follow the
    scheduler naming convention: ``"main"``, ``"main.w0"``, ``"main.m0.w1"``, etc.

    Examples::

        # Connect to the global master (default):
        with ClusterClient("/scratch/ckpt") as client:
            fut = client.submit(task)

        # Connect to a specific worker node:
        with ClusterClient("/scratch/ckpt", node_id="main.w0") as client:
            fut = client.submit(task)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        node_id: str = "global",
        client_id: Optional[str] = None,
        log_dir: str = "logs",
        log_level: int = logging.INFO,
    ):
        """
        Args:
            checkpoint_dir: Directory containing orchestrator checkpoint files.
            node_id:        Which node to connect to. ``"global"`` (default) resolves
                            to the global master (shortest node_id in the checkpoint
                            dir, e.g. ``"main"``). Pass any scheduler node id such as
                            ``"main.w0"`` or ``"main.m0.w2"`` to connect directly to
                            that node.
            client_id:      Optional client identity string; auto-generated if omitted.
            log_dir:        Directory for log files.  When provided a file
                            ``{log_dir}/{client_id}.log`` is created.  When
                            ``None`` (default) logging goes to the root handler.
            log_level:      Logging level (default ``logging.INFO``).
        """
        self._client_id = client_id or f"client:{uuid.uuid4().hex[:8]}"
        self.logger = setup_logger(
            __name__, self._client_id, log_dir=log_dir, level=log_level
        )
        self._pending: Dict[str, ConcurrentFuture] = {}
        self._lock = threading.Lock()
        self._recv_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._node_id = None

        resolved_id = _resolve_node_id(checkpoint_dir, node_id)
        self._node_id = resolved_id
        comm_path = os.path.join(checkpoint_dir, f"{resolved_id}_comm.json")
        if not os.path.exists(comm_path):
            raise FileNotFoundError(
                f"No comm checkpoint found for node '{resolved_id}' in '{checkpoint_dir}'"
            )
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
        self._transport: _ClientTransport = transport_cls()

    def start(self):
        """Connect to the node and start the result-receive thread."""
        self._transport.connect(self._node_address, self._client_id.encode())
        self.logger.info(f"Connected to {self._node_id} at {self._node_address}")
        self._recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._recv_thread.start()
        self.logger.info("Started recv thread")

    def teardown(self):
        """Shut down the transport and receive thread."""
        self._stop_event.set()
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=5.0)
        self._transport.close()

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
        """Register futures for *tasks* and send them in a single TaskUpdate."""
        futures: List[ConcurrentFuture] = []
        with self._lock:
            for task in tasks:
                fut: ConcurrentFuture = ConcurrentFuture()
                self._pending[task.task_id] = fut
                futures.append(fut)
        task_update = TaskUpdate(sender=self._client_id, added_tasks=tasks)
        self._transport.send(cloudpickle.dumps(task_update))
        return futures

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

        All tasks are sent in one :class:`TaskUpdate` message.  Returns a list
        of :class:`~concurrent.futures.Future` objects in the same order as
        *iterable*.

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
            self._to_task(fn, (item,), kwargs, nnodes, ppn, ngpus_per_process)
            for item in iterable
        ]
        return self._send_batch(tasks)

    def __enter__(self) -> "ClusterClient":
        self.start()
        return self

    def __exit__(self, *_):
        self.teardown()

    def _recv_loop(self):
        """Background thread: receive Result messages and resolve pending Futures."""
        while not self._stop_event.is_set():
            raw = self._transport.recv(timeout_ms=100)
            if raw is None:
                continue
            msg = cloudpickle.loads(raw)
            if not isinstance(msg, Result):
                continue
            with self._lock:
                fut = self._pending.pop(msg.task_id, None)
            if fut is None or fut.done():
                continue
            if msg.success:
                fut.set_result(msg.data)
            else:
                fut.set_exception(Exception(msg.exception or "Task failed"))
