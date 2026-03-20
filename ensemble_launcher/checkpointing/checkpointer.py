"""Non-blocking checkpointer for scheduler state, comm state, and task results.

Each component is written to its own file so they can be updated independently:

  ``{node_id}_meta.json``       — CheckpointData (metadata / manifest)
  ``{node_id}_scheduler.json``  — SchedulerState
  ``{node_id}_comm.json``       — CommCheckpointData (type name + serialised state)
  ``{node_id}_tasks.json``      — TasksCheckpointData (cloudpickled task dict)
  ``{node_id}_results.json``    — ResultCheckpointData (cloudpickled result dict)

All arguments to ``write_checkpoint`` are optional; only the provided components
are written.  ``read_checkpoint`` returns ``None`` when no metadata file exists,
or a ``(scheduler_state, comm_state, tasks)`` tuple where absent components are
``None``.

All file I/O is offloaded to a thread-pool executor via ``run_in_executor`` so
the asyncio event loop is never blocked.  Every write uses an atomic
rename-after-write pattern (write ``.tmp`` → ``os.replace``) to ensure files
are never left in a partially-written state.
"""

from __future__ import annotations

import asyncio
import base64
import os
import socket
import tempfile
import time
from logging import Logger
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import cloudpickle
from pydantic import BaseModel

from ensemble_launcher.comm.async_base import AsyncCommState
from ensemble_launcher.scheduler.state import SchedulerState

if TYPE_CHECKING:
    from ensemble_launcher.comm.messages import Result
    from ensemble_launcher.ensemble import Task


# ---------------------------------------------------------------------------
# On-disk data models
# ---------------------------------------------------------------------------


class CheckpointData(BaseModel):
    """Metadata written to ``{node_id}_meta.json``.

    Acts as a manifest: records the timestamp and which component files are
    present so callers can quickly determine checkpoint completeness without
    opening every file.
    """

    timestamp: float
    node_id: str
    has_scheduler: bool = False
    has_comm: bool = False
    has_tasks: bool = False
    pid: int = os.getpid()
    hostname: str = socket.gethostname()


class CommCheckpointData(BaseModel):
    """Written to ``{node_id}_comm.json``.

    Bundles the concrete class name with the serialised state so the correct
    ``AsyncCommState`` subclass can be instantiated on read-back.
    """

    comm_state_type: str
    comm_state_json: str


class TasksCheckpointData(BaseModel):
    """Written to ``{node_id}_tasks.json``."""

    tasks_b64: str


class ResultCheckpointData(BaseModel):
    """Written to ``{node_id}_results.json``."""

    results_b64: str


# ---------------------------------------------------------------------------
# Comm-state deserialisation registry
# ---------------------------------------------------------------------------

_COMM_STATE_REGISTRY: Dict[str, type] = {}


def _get_comm_state_class(type_name: str) -> type:
    """Return the ``AsyncCommState`` subclass registered under *type_name*.

    Performs a lazy import of known backends on the first miss.
    """
    if type_name not in _COMM_STATE_REGISTRY:
        try:
            from ensemble_launcher.comm.async_zmq import AsyncZMQCommState

            _COMM_STATE_REGISTRY[AsyncZMQCommState.__name__] = AsyncZMQCommState
        except ImportError:
            pass
    if type_name not in _COMM_STATE_REGISTRY:
        raise ValueError(
            f"Unknown comm state type '{type_name}'. "
            f"Known types: {list(_COMM_STATE_REGISTRY)}"
        )
    return _COMM_STATE_REGISTRY[type_name]


# Eagerly register built-in comm state types.
try:
    from ensemble_launcher.comm.async_zmq import AsyncZMQCommState

    _COMM_STATE_REGISTRY[AsyncZMQCommState.__name__] = AsyncZMQCommState
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------


class Checkpointer:
    """Non-blocking checkpointer for a single node's scheduler, comm, and result state.

    Each component is stored in its own file so they can be written and read
    independently.  All arguments to ``write_checkpoint`` are optional.

    Args:
        node_id:        Logical ID of the node being checkpointed.
        checkpoint_dir: Directory where checkpoint files are stored.
        logger:         Logger instance.

    Typical usage in a worker::

        checkpointer = Checkpointer("test.w0", "/scratch/ckpt", logger)

        # Write only what you have — all args are optional:
        await checkpointer.write_checkpoint(
            scheduler_state=scheduler.get_state(node_id),
            comm_state=comm.get_state(),
        )

        # Write tasks separately when the dict changes:
        await checkpointer.write_checkpoint(tasks=scheduler.tasks)

        # Persist results after each task completes:
        await checkpointer.write_results(completed_results)

        # On restart:
        checkpoint = await checkpointer.read_checkpoint()
        if checkpoint is not None:
            sched_state, comm_state, tasks = checkpoint
            # Each element may be None if it was not previously written.
    """

    def __init__(
        self,
        node_id: str,
        checkpoint_dir: str,
        logger: Logger,
    ) -> None:
        self.node_id = node_id
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_sub_dir = os.path.join(checkpoint_dir, *self.node_id.split("."))
        self.logger = logger
        try:
            os.makedirs(self.checkpoint_sub_dir, exist_ok=True)
            self.logger.info(f"Writing scheduler state to {self.scheduler_path}")
            self.logger.info(f"Writing comm state {self.comm_path}")
        except FileExistsError:
            if not os.path.isdir(self.checkpoint_sub_dir):
                raise RuntimeError(
                    f"Checkpoint path {self.checkpoint_sub_dir!r} exists but is not a directory"
                )

    # ------------------------------------------------------------------
    # File paths
    # ------------------------------------------------------------------

    @property
    def meta_path(self) -> str:
        return os.path.join(self.checkpoint_sub_dir, f"{self.node_id}_meta.json")

    @property
    def scheduler_path(self) -> str:
        return os.path.join(self.checkpoint_sub_dir, f"{self.node_id}_scheduler.json")

    @property
    def comm_path(self) -> str:
        return os.path.join(self.checkpoint_sub_dir, f"{self.node_id}_comm.json")

    @property
    def tasks_path(self) -> str:
        return os.path.join(self.checkpoint_sub_dir, f"{self.node_id}_tasks.json")

    @property
    def results_path(self) -> str:
        return os.path.join(self.checkpoint_sub_dir, f"{self.node_id}_results.json")

    # ------------------------------------------------------------------
    # Synchronous I/O helpers (dispatched to a thread via run_in_executor)
    # ------------------------------------------------------------------

    def _write_json_atomic(self, path: str, json_str: str) -> None:
        """Atomically write *json_str* to *path* via a sibling ``.tmp`` file."""
        target_dir = os.path.dirname(path)
        max_retries = 3
        last_exc: Optional[Exception] = None
        for i in range(max_retries):
            tmp = None
            try:
                os.makedirs(target_dir, exist_ok=True)
                fd, tmp = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
                try:
                    with os.fdopen(fd, "w") as fh:
                        fh.write(json_str)
                        fh.flush()
                        os.fsync(fh.fileno())
                except Exception:
                    os.close(fd)
                    raise
                os.replace(tmp, path)
                return
            except Exception as e:
                last_exc = e
                if tmp and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                self.logger.warning(f"Writing {path} failed with error {e}. retry {i}.....")
        self.logger.error(f"Failed to write {path} after {max_retries} attempts: {last_exc}")

    def _read_json(self, path: str) -> Optional[str]:
        """Return the raw JSON string at *path*, or ``None`` if absent."""
        if not os.path.exists(path):
            return None
        with open(path, "r") as fh:
            return fh.read()

    def _delete_file(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Internal synchronous write helpers (one per component)
    # ------------------------------------------------------------------

    def _write_components_sync(
        self,
        scheduler_state: Optional[SchedulerState],
        comm_state: Optional[AsyncCommState],
        tasks: Optional[Dict[str, "Task"]],
    ) -> CheckpointData:
        """Write whichever components are not None and return updated metadata."""
        # Load existing metadata so we don't lose has_* flags for components
        # that were written in a previous call.
        meta: Optional[CheckpointData] = None
        raw_meta = self._read_json(self.meta_path)
        if raw_meta is not None:
            try:
                meta = CheckpointData.model_validate_json(raw_meta)
            except Exception:
                pass

        has_scheduler = meta.has_scheduler if meta else False
        has_comm = meta.has_comm if meta else False
        has_tasks = meta.has_tasks if meta else False

        if scheduler_state is not None:
            self._write_json_atomic(
                self.scheduler_path, scheduler_state.model_dump_json()
            )
            has_scheduler = True

        if comm_state is not None:
            comm_data = CommCheckpointData(
                comm_state_type=type(comm_state).__name__,
                comm_state_json=comm_state.serialize(),
            )
            self._write_json_atomic(self.comm_path, comm_data.model_dump_json())
            has_comm = True

        if tasks is not None:
            tasks_data = TasksCheckpointData(
                tasks_b64=base64.b64encode(cloudpickle.dumps(tasks)).decode("ascii")
            )
            self._write_json_atomic(self.tasks_path, tasks_data.model_dump_json())
            has_tasks = True

        new_meta = CheckpointData(
            timestamp=time.time(),
            node_id=self.node_id,
            has_scheduler=has_scheduler,
            has_comm=has_comm,
            has_tasks=has_tasks,
        )
        self._write_json_atomic(self.meta_path, new_meta.model_dump_json())
        return new_meta

    def _read_components_sync(
        self,
    ) -> Optional[
        Tuple[
            Optional[SchedulerState],
            Optional[AsyncCommState],
            Optional[Dict[str, "Task"]],
        ]
    ]:
        """Read all available components; return None if no metadata exists."""
        raw_meta = self._read_json(self.meta_path)
        if raw_meta is None:
            return None

        meta = CheckpointData.model_validate_json(raw_meta)

        scheduler_state: Optional[SchedulerState] = None
        if meta.has_scheduler:
            raw = self._read_json(self.scheduler_path)
            if raw is not None:
                scheduler_state = SchedulerState.model_validate_json(raw)

        comm_state: Optional[AsyncCommState] = None
        if meta.has_comm:
            raw = self._read_json(self.comm_path)
            if raw is not None:
                comm_data = CommCheckpointData.model_validate_json(raw)
                comm_cls = _get_comm_state_class(comm_data.comm_state_type)
                comm_state = comm_cls.deserialize(comm_data.comm_state_json)

        tasks: Optional[Dict[str, "Task"]] = None
        if meta.has_tasks:
            raw = self._read_json(self.tasks_path)
            if raw is not None:
                tasks_data = TasksCheckpointData.model_validate_json(raw)
                tasks = cloudpickle.loads(base64.b64decode(tasks_data.tasks_b64))

        return scheduler_state, comm_state, tasks

    # ------------------------------------------------------------------
    # State checkpoint primitives
    # ------------------------------------------------------------------

    async def write_checkpoint(
        self,
        scheduler_state: Optional[SchedulerState] = None,
        comm_state: Optional[AsyncCommState] = None,
        tasks: Optional[Dict[str, "Task"]] = None,
    ) -> None:
        """Non-blocking write of any combination of scheduler state, comm state,
        and tasks to their respective files.

        Only the provided (non-``None``) components are written; previously
        saved components are preserved.  The metadata file is always updated.

        Args:
            scheduler_state: From ``scheduler.get_state(node_id)``.
            comm_state:      From ``comm.get_state()``.
            tasks:           ``{task_id: Task}`` dict; cloudpickled to handle
                             arbitrary ``Task.executable`` callables.
        """
        loop = asyncio.get_event_loop()
        meta = await loop.run_in_executor(
            None, self._write_components_sync, scheduler_state, comm_state, tasks
        )
        self.logger.debug(
            f"[Checkpointer] Checkpoint written for {self.node_id} "
            f"(scheduler={meta.has_scheduler}, comm={meta.has_comm}, tasks={meta.has_tasks})"
        )

    async def read_checkpoint(
        self,
    ) -> Optional[
        Tuple[
            Optional[SchedulerState],
            Optional[AsyncCommState],
            Optional[Dict[str, "Task"]],
        ]
    ]:
        """Non-blocking read of all checkpoint components from disk.

        Returns:
            ``(scheduler_state, comm_state, tasks)`` if a metadata file exists
            (each element may be ``None`` if that component was not written),
            or ``None`` if no checkpoint exists at all.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, self._read_components_sync)
        if result is None:
            self.logger.debug(f"[Checkpointer] No checkpoint found for {self.node_id}")
            return None

        scheduler_state, comm_state, tasks = result
        self.logger.debug(
            f"[Checkpointer] Checkpoint restored for {self.node_id} "
            f"(scheduler={'yes' if scheduler_state else 'no'}, "
            f"comm={'yes' if comm_state else 'no'}, "
            f"tasks={'yes' if tasks else 'no'})"
        )
        return result

    # ------------------------------------------------------------------
    # Result checkpoint primitives (for workers)
    # ------------------------------------------------------------------

    async def write_results(self, results: Dict[str, "Result"]) -> None:
        """Non-blocking write of completed task results.

        Designed to be called after each task completes so that a restarted
        worker can skip already-finished tasks.

        Args:
            results: ``{task_id: Result}``; cloudpickled because ``Result.data``
                     may be an arbitrary Python object.
        """
        data = ResultCheckpointData(
            results_b64=base64.b64encode(cloudpickle.dumps(results)).decode("ascii")
        )
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, self._write_json_atomic, self.results_path, data.model_dump_json()
        )
        self.logger.debug(
            f"[Checkpointer] Results written for {self.node_id} ({len(results)} results)"
        )

    async def read_results(self) -> Optional[Dict[str, "Result"]]:
        """Non-blocking read of the results checkpoint.

        Returns:
            ``{task_id: Result}`` if a results file exists, ``None`` otherwise.
        """
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(None, self._read_json, self.results_path)
        if raw is None:
            self.logger.debug(
                f"[Checkpointer] No results checkpoint found for {self.node_id}"
            )
            return None

        data = ResultCheckpointData.model_validate_json(raw)
        results: Dict[str, "Result"] = cloudpickle.loads(
            base64.b64decode(data.results_b64)
        )
        self.logger.debug(
            f"[Checkpointer] Results restored for {self.node_id} ({len(results)} results)"
        )
        return results

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def checkpoint_exists(self) -> bool:
        """Return ``True`` if a metadata file exists (synchronous)."""
        return os.path.exists(self.meta_path)

    def results_exist(self) -> bool:
        """Return ``True`` if a results file exists (synchronous)."""
        return os.path.exists(self.results_path)

    async def delete_checkpoint(self) -> None:
        """Non-blocking deletion of all checkpoint component files."""
        loop = asyncio.get_event_loop()
        for path in (
            self.meta_path,
            self.scheduler_path,
            self.comm_path,
            self.tasks_path,
        ):
            await loop.run_in_executor(None, self._delete_file, path)
        self.logger.debug(f"[Checkpointer] Checkpoint deleted for {self.node_id}")

    async def delete_results(self) -> None:
        """Non-blocking deletion of the results file."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._delete_file, self.results_path)
        self.logger.debug(f"[Checkpointer] Results deleted for {self.node_id}")
