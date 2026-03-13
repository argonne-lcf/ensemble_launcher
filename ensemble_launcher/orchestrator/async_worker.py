import asyncio
import json
import os
import random
import signal
import socket
import time
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional, Tuple, Union

from ensemble_launcher.checkpointing import Checkpointer
from ensemble_launcher.comm import (
    AsyncComm,
    AsyncZMQComm,
    AsyncZMQCommState,
    HeartBeat,
    NodeInfo,
    NodeUpdate,
    Result,
    ResultBatch,
    Status,
    Stop,
    StopType,
    TaskUpdate,
)
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.executors import (
    AsyncMPIExecutor,
    AsyncProcessPoolExecutor,
    AsyncThreadPoolExecutor,
    executor_registry,
)
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.profiling import EventRegistry, get_registry
from ensemble_launcher.scheduler import AsyncTaskScheduler
from ensemble_launcher.scheduler.resource import JobResource

from .node import Node

AsyncFuture = asyncio.Future


class AsyncWorker(Node):
    """Leaf-level worker node that executes tasks using a local executor.

    Receives its resource allocation and task assignments from a parent master,
    runs tasks through an AsyncTaskScheduler, and returns results and final
    status back to the parent when all work is complete.
    """

    type = "AsyncWorker"

    def __init__(
        self,
        id: str,
        config: LauncherConfig,
        Nodes: Optional[JobResource] = None,
        tasks: Optional[Dict[str, Task]] = None,
        parent: Optional[NodeInfo] = None,
        children: Optional[Dict[str, NodeInfo]] = None,
        parent_comm: Optional[AsyncComm] = None,
    ):
        super().__init__(id, parent=parent, children=children)
        self._config = config
        self._init_tasks: Dict[str, Task] = tasks if tasks is not None else {}
        self._init_nodes = Nodes
        self._parent_comm = parent_comm

        ##lazy init in run function
        self._comm = None
        ##lazy init in run function
        self._executor: Dict[
            str,
            Union[AsyncProcessPoolExecutor, AsyncMPIExecutor, AsyncThreadPoolExecutor],
        ] = None
        self._default_executor_name: str = None

        self._scheduler = None

        self.logger = None

        # Initialize event registry for perfetto profiling
        self._event_registry: Optional[EventRegistry] = None

        self._stop_submission = asyncio.Event()
        self._stop_reporting = asyncio.Event()

        self._submission_task = None
        self._reporting_task = None

        self._task_futures: Dict[str, AsyncFuture] = {}
        self._task_id_to_executor: Dict[str, str] = {}
        self._event_loop = None

        self._checkpointer: Optional[Checkpointer] = None
        # Cached checkpoint data — populated early in _lazy_init() before sockets are set up.
        self._ckpt_data: Optional[tuple] = None  # (sched_state, comm_state, tasks)
        self._ckpt_results: Optional[dict] = None  # {task_id: Result}

        # Cluster mode state
        self._stop_task_update = asyncio.Event()
        self._task_update_task = None
        self._client_handler_task: Optional[asyncio.Task] = None
        self._client_task_map: Dict[str, str] = {}  # task_id -> client_id

        # Cluster mode / workstealing mode
        self._stop_signal_received = asyncio.Event()

        # Parent fault detection
        self._last_parent_ack_time: float = time.time()
        self._parent_dead_event: asyncio.Event = asyncio.Event()

    @asynccontextmanager
    async def _timer(self, event_name: str):
        """Timer that records to event registry for Perfetto export."""
        if self._event_registry is not None:
            with self._event_registry.measure(
                event_name, "async_worker", node_id=self.node_id, pid=os.getpid()
            ):
                yield
        else:
            yield

    @property
    def nodes(self) -> JobResource:
        """Node resource allocation owned by the scheduler cluster."""
        return self._scheduler.cluster.nodes

    @nodes.setter
    def nodes(self, value: JobResource) -> None:
        self._scheduler.cluster.update_nodes(value)

    @property
    def parent_comm(self) -> Optional[AsyncComm]:
        """Communication channel to the parent node."""
        return self._parent_comm

    @parent_comm.setter
    def parent_comm(self, value: AsyncComm) -> None:
        self._parent_comm = value

    @property
    def comm(self) -> Optional[AsyncComm]:
        """Communication channel for this worker."""
        return self._comm

    @property
    def tasks(self) -> Dict[str, Task]:
        """All tasks owned by this worker's scheduler."""
        return self._scheduler.tasks

    @property
    def init_nodes(self) -> JobResource:
        return self._init_nodes

    @property
    def init_tasks(self) -> JobResource:
        return self._init_tasks

    # -----------------------------------------------------------------
    #                       Initialization
    # -----------------------------------------------------------------

    def _setup_logger(self) -> None:
        """Configure the logger, optionally writing to a per-worker log file."""
        log_dir = (
            os.path.join(os.getcwd(), "logs") if self._config.worker_logs else None
        )
        self.logger = setup_logger(
            __name__, self.node_id, log_dir=log_dir, level=self._config.log_level
        )

    def _create_comm(self) -> None:
        """Instantiate the communication backend (currently only async_zmq is supported)."""
        if self._config.comm_name == "async_zmq":
            self.logger.info(f"{self.node_id}: Starting comm init")
            self._comm = AsyncZMQComm(
                self.logger.getChild("comm"),
                self.info(),
                parent_address=self.parent_comm.my_address
                if self.parent_comm
                else None,
            )
            self.logger.info(f"{self.node_id}: Done with comm init")
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")

    def _create_monitor_tasks(self) -> None:
        """Start the submission, reporting, and cluster monitor asyncio tasks."""
        ##start submission loop
        self._submission_task = asyncio.create_task(self._submit_ready_tasks())

        ##start reporting loop
        self._reporting_task = asyncio.create_task(self.report_status())

        if self._config.cluster and self.parent:
            self._task_update_task = asyncio.create_task(self._task_update_monitor())

        if self._config.cluster:
            self._client_handler_task = asyncio.create_task(
                self._client_request_handler()
            )

    async def _lazy_init(self) -> None:
        """Set up all resources needed before task execution begins.

        In order: logging, event loop capture, comm, parent sync (heartbeat +
        node/task updates), scheduler, executor, and monitor task creation.
        """
        if self._config.profile == "perfetto":
            self._event_registry = get_registry()
            self._event_registry.enable()

        self._event_loop = asyncio.get_running_loop()
        self._event_loop.add_signal_handler(
            signal.SIGTERM, self._stop_signal_received.set
        )
        # lazy logger creation
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(
            f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds"
        )

        try:
            self.logger.info(f"My cpu affinity: {os.sched_getaffinity(0)}")
        except Exception:
            pass

        # Read checkpoint data early so both comm state and scheduler state
        # can be restored from the cached data at the appropriate points below.
        await self._read_checkpoint_data()

        # Lazy comm creation
        self._create_comm()

        # Restore saved comm state before binding so parent can reconnect.
        self._restore_comm_state()

        # start parent endpoint monitors
        await self._comm.start_monitors()

        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

        # Jitter before syncing to avoid thundering herd at startup.
        await asyncio.sleep(random.uniform(0, 0.1 * self._config.report_interval))

        # Syncronize with parent
        await self._sync_with_parent()

        # Init scheduler
        self._scheduler = AsyncTaskScheduler(
            self.logger.getChild("scheduler"), self._init_tasks, self._init_nodes
        )

        # Validate that nodes are initialized
        if not self.nodes:
            self.logger.error(f"{self.node_id}: Nodes not initialized!")
            raise RuntimeError(
                f"{self.node_id}: Nodes must be initialized before execution"
            )

        # Restore scheduler state from cached checkpoint data (comm already restored).
        self._restore_scheduler_checkpoint()

        self._scheduler.start_monitoring()  # start the scheduler monitoring

        self.logger.info(f"Running {list(self.tasks.keys())} tasks")
        self.logger.debug(f"Sorted tasks size {self._scheduler._sorted_tasks.qsize()}")

        ##lazy executor creation
        if isinstance(self._config.task_executor_name, list):
            assert all(
                [
                    executor_name in executor_registry.async_executors
                    for executor_name in self._config.task_executor_name
                ]
            ), (
                f"Executor {self._config.task_executor_name} not found in async executors {executor_registry.async_executors}"
            )
        else:
            assert (
                self._config.task_executor_name in executor_registry.async_executors
            ), (
                f"Executor {self._config.task_executor_name} not found in async executors {executor_registry.async_executors}"
            )

        kwargs = {}
        kwargs["logger"] = self.logger.getChild("executor")
        kwargs["gpu_selector"] = self._config.gpu_selector
        kwargs["max_workers"] = self.nodes.resources[0].cpu_count
        kwargs["return_stdout"] = self._config.return_stdout
        if self._config.task_executor_name == "async_mpi":
            kwargs["cpu_binding_option"] = self._config.cpu_binding_option
            kwargs["use_ppn"] = self._config.use_mpi_ppn

        if isinstance(self._config.task_executor_name, list):
            self._executor = {}
            for exec_name in self._config.task_executor_name:
                self._executor[exec_name] = executor_registry.create_executor(
                    exec_name, kwargs=kwargs
                )
            self._default_executor_name = self._config.task_executor_name[0]
        else:
            self._executor = {
                self._config.task_executor_name: executor_registry.create_executor(
                    self._config.task_executor_name, kwargs=kwargs
                )
            }
            self._default_executor_name = self._config.task_executor_name

        self.logger.info(
            f"Created {self._config.task_executor_name} executors (Default = {self._default_executor_name})"
        )

        # Start global monitor tasks
        self._create_monitor_tasks()

    # --------------------------------------------------------------------------
    #                               Checkpointing
    # --------------------------------------------------------------------------

    async def _read_checkpoint_data(self) -> None:
        """Create checkpointer (if configured) and eagerly read the full checkpoint.

        Called immediately after logger setup — before comm is created and
        before any sockets are bound — so that both comm state and scheduler
        state are available in ``self._ckpt_data`` when needed.
        """
        if not self._config.checkpoint_dir:
            return
        self._checkpointer = Checkpointer(
            self.node_id,
            self._config.checkpoint_dir,
            self.logger.getChild("checkpointer"),
        )
        if not self._checkpointer.checkpoint_exists():
            return
        self._ckpt_data = await self._checkpointer.read_checkpoint()
        self._ckpt_results = await self._checkpointer.read_results()

    def _restore_comm_state(self) -> None:
        """Restore comm from cached checkpoint data using the comm's set_state.

        Must be called after _create_comm() and before setup_zmq_sockets() so
        the node re-binds to its previous address.
        """
        if self._ckpt_data is None:
            return
        _, comm_state, _ = self._ckpt_data
        if comm_state is None:
            return
        comm_logger = self._comm.logger
        self._comm = type(self._comm).set_state(comm_state)
        self._comm.logger = comm_logger
        self.logger.info(
            f"{self.node_id}: Comm state restored from checkpoint"
            f" (address: {self._comm.my_address})"
        )

    def _restore_scheduler_checkpoint(self) -> bool:
        """Restore scheduler state from cached checkpoint data.

        Called after scheduler creation. Returns True if scheduler state was
        successfully restored.
        """
        if self._ckpt_data is None:
            return False
        scheduler_state, _, _ = self._ckpt_data
        if scheduler_state is None:
            return False
        self._scheduler.set_state(scheduler_state, self._ckpt_results or {})
        self.logger.info(f"{self.node_id}: Scheduler state restored from checkpoint")
        return True

    async def _write_checkpoint(self) -> None:
        """Write scheduler state, comm state, and completed results to checkpoint."""
        if self._checkpointer is None:
            return
        await self._checkpointer.write_checkpoint(
            scheduler_state=self._scheduler.get_state(self.node_id),
            comm_state=self._comm.get_state(),
        )
        completed_results = {
            task_id: Result(
                task_id=task_id,
                data=task.result,
                exception=str(task.exception),
            )
            for task_id, task in self.tasks.items()
            if task.status in (TaskStatus.SUCCESS, TaskStatus.FAILED)
        }
        if completed_results:
            await self._checkpointer.write_results(completed_results)

    # --------------------------------------------------------------------------
    #                               Parent Synchronization
    # --------------------------------------------------------------------------

    async def _receive_initial_tasks(self) -> None:
        """Receive initial task assignment from parent into _init_tasks. Overridable by subclasses."""
        task_update: TaskUpdate = await self._comm.recv_message_from_parent(
            TaskUpdate, timeout=5.0
        )
        if task_update is not None:
            self.logger.info(
                f"{self.node_id}: Received task update from parent containing {len(task_update.added_tasks)}"
            )
            for task in task_update.added_tasks:
                self._init_tasks[task.task_id] = task

    async def _sync_with_parent(self) -> None:
        """Perform initial handshake with the parent: heartbeat, node update, task update."""
        # sync heart beat with parent
        async with self._timer("heartbeat_sync"):
            if self.parent and not await self._comm.sync_heartbeat_with_parent(
                timeout=30.0
            ):
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            self._last_parent_ack_time = time.time()
            self.logger.info(f"{self.node_id}: Synced heartbeat with parent")

        node_update: NodeUpdate = await self._comm.recv_message_from_parent(
            NodeUpdate, timeout=10.0
        )
        if node_update is not None:
            self.logger.info(f"{self.node_id}: Received node update from parent")
            if node_update.nodes:
                self._init_nodes = node_update.nodes
                self.logger.info(
                    f"{self.node_id}: Updated nodes list with {len(self._init_nodes.nodes)} nodes"
                )
                self.logger.info(f"{self.node_id}: Nodes details: {self._init_nodes}")
            else:
                self.logger.warning(
                    f"{self.node_id}: Received empty node update from parent"
                )
        else:
            self.logger.warning(
                f"{self.node_id}: No node update received from parent at start"
            )

        await self._receive_initial_tasks()

    # --------------------------------------------------------------------------
    #                               Callbacks
    # --------------------------------------------------------------------------

    def create_done_callback(self, task: Task) -> Callable[[AsyncFuture], None]:
        """Return a done-callback that dispatches _task_callback into the event loop."""

        def done_callback(future: AsyncFuture) -> None:
            self._event_loop.call_soon_threadsafe(self._task_callback, task, future)

        return done_callback

    def _task_callback(self, task: Task, future: AsyncFuture) -> None:
        """Process a completed task future: record status, free resources, forward result."""
        task_id = task.task_id
        if self._config.profile == "perfetto" and self._event_registry is not None:
            self._event_registry.record_async_end(
                name=task.task_id,
                category="task_execution",
                node_id=self.node_id,
                pid=os.getpid(),
                async_id=task.task_id,
            )
        if task_id in self.tasks:
            exception = future.exception()
            task.end_time = time.time()
            if exception is None:
                task.status = TaskStatus.SUCCESS
                task.result = future.result()
            else:
                task.status = TaskStatus.FAILED
                task.exception = str(exception)

            self._scheduler.free(task_id, task.status)

            # In cluster mode: eagerly send individual Result to client or parent
            if self._config.cluster:
                task_result = Result(
                    sender=self.node_id,
                    task_id=task_id,
                    data=task.result if exception is None else None,
                    success=(exception is None),
                    exception=str(exception) if exception else None,
                )
                if task_id in self._client_task_map:
                    client_id = self._client_task_map.pop(task_id)
                    asyncio.create_task(
                        self._comm.send_message_to_child(client_id, task_result)
                    )
                elif self.parent:
                    asyncio.create_task(self._comm.send_message_to_parent(task_result))

    # -------------------------------------------------------------------------
    #                               Monitors
    # -------------------------------------------------------------------------

    def get_status(self) -> Status:
        """Return a Status snapshot of running/failed/successful tasks and free resources."""
        return Status(
            nrunning_tasks=len(self._scheduler.running_tasks),
            nfailed_tasks=len(self._scheduler.failed_tasks),
            nsuccessful_tasks=len(self._scheduler.successful_tasks),
            nfree_cores=self._scheduler.cluster.free_cpus,
            nfree_gpus=self._scheduler.cluster.free_gpus,
        )

    def _update_tasks(
        self, taskupdate: TaskUpdate
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """Apply a TaskUpdate: add new tasks to the scheduler and cancel/delete removed ones.

        Returns (add_status, del_status) dicts mapping task_id -> success bool.
        """
        ##Add the tasks to scheduler
        add_status = {}
        del_status = {}
        for new_task in taskupdate.added_tasks:
            self.logger.debug(f"Adding new task {new_task}")
            add_status[new_task.task_id] = self._scheduler.add_task(new_task)
            if not add_status[new_task.task_id]:
                self.logger.error(f"Failed to add new task {new_task.task_id}")
            else:
                self.logger.debug(f"Added new task {new_task.task_id}")

        ##delete tasks if needed
        for task in taskupdate.deleted_tasks:
            if task.task_id in self._scheduler._running_tasks:
                executor_name = self._task_id_to_executor[task.task_id]
                self._executor[executor_name].stop(
                    task_id=self._executor_task_ids[task.task_id]
                )
                self._task_futures[task.task_id].cancel()
            del_status[task.task_id] = self._scheduler.delete_task(task)

        return (add_status, del_status)

    async def _client_request_handler(self) -> None:
        """Cluster mode: handle messages from any ClusterClient connected to this worker."""
        while not self._stop_task_update.is_set():
            item = await self._comm.recv_client_message()
            if item is None:
                continue
            client_id, msg = item
            if isinstance(msg, TaskUpdate):
                for task in msg.added_tasks:
                    self._client_task_map[task.task_id] = client_id
                self._update_tasks(msg)

    async def _task_update_monitor(self) -> None:
        """Receive TaskUpdate messages from parent and incorporate new tasks."""
        while not self._stop_task_update.is_set():
            try:
                task_update = await self._comm.recv_message_from_parent(
                    TaskUpdate, block=True
                )
                if task_update is not None:
                    self._update_tasks(task_update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Task update monitor error: {e}")
                await asyncio.sleep(0.1)

    async def _submit_ready_tasks(self) -> None:
        """Consume the scheduler's ready_tasks queue and submit each task to the executor."""
        self.logger.info("Starting task submission loop")

        while not self._stop_submission.is_set():
            try:
                task_id, req = await self._scheduler.ready_tasks.get()

                task = self.tasks[task_id]
                self.logger.debug(
                    f"Submitting task {task_id}: {task.executable} with resources {req.resources} {task.env}"
                )
                task.status = TaskStatus.READY
                task.start_time = time.time()
                if (
                    self._config.profile == "perfetto"
                    and self._event_registry is not None
                ):
                    self._event_registry.record_async_begin(
                        name=task.task_id,
                        category="task_execution",
                        node_id=self.node_id,
                        pid=os.getpid(),
                        async_id=task.task_id,
                    )
                    self._event_registry.record_counter(
                        name="tasks_submitted",
                        category="task_execution",
                        value=1,
                        pid=os.getpid(),
                        node_id=self.node_id,
                    )

                set_exception = False
                if task.executor_name is not None:
                    if task.executor_name in self._executor:
                        future = self._executor[task.executor_name].submit(
                            req,
                            task.executable,
                            task_args=task.args,
                            task_kwargs=task.kwargs,
                            env=task.env,
                        )
                        self._task_id_to_executor[task_id] = task.executor_name
                    else:
                        self.logger.warning(
                            f"Failed to submit {task_id}. {self.node_id} doesn't have {task.executor_name} executor"
                        )
                        future = self._event_loop.create_future()
                        set_exception = True

                else:
                    default_executor = self._executor[self._default_executor_name]
                    future = default_executor.submit(
                        req,
                        task.executable,
                        task_args=task.args,
                        task_kwargs=task.kwargs,
                        env=task.env,
                    )
                    self._task_id_to_executor[task_id] = self._default_executor_name
                future.add_done_callback(self.create_done_callback(task))
                self._task_futures[task_id] = future
                task.status = TaskStatus.RUNNING
                if set_exception:
                    future.set_exception(
                        Exception(
                            f"Failed due to unavailability of {task.executor_name} at {self.node_id}"
                        )
                    )
            except asyncio.CancelledError:
                self.logger.info("Submission loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in task submission loop: {e}", exc_info=True)
                raise e

    async def report_status(self) -> None:
        """Periodically send a Status snapshot to the parent at the configured interval."""
        while not self._stop_reporting.is_set():
            try:
                status = self.get_status()
                if self.parent:
                    await self._comm.send_message_to_parent(status)
                    self.logger.info(status)
                    # Check for HeartBeat ACK from parent (sent by parent on Status receipt)
                    threshold = (
                        self._config.dead_node_factor
                        * self._config.report_interval
                        / 2.0
                    )
                    ack = await self._comm.recv_message_from_parent(HeartBeat)
                    if ack is not None:
                        self._last_parent_ack_time = time.time()
                    else:
                        age = time.time() - self._last_parent_ack_time
                        if age > threshold:
                            self.logger.error(
                                f"{self.node_id}: No heartbeat from parent for {age:.1f}s "
                                f"(threshold {threshold:.1f}s) — self-terminating"
                            )
                            self._parent_dead_event.set()
                else:
                    self.logger.info(status)

                # Periodic checkpoint: scheduler state, comm state, and completed results.
                if self._checkpointer is not None:
                    asyncio.create_task(self._write_checkpoint())

                # Use wait with timeout so we can exit quickly when stopped
                try:
                    jitter = random.uniform(-0.05, 0.05) * self._config.report_interval
                    await asyncio.wait_for(
                        self._stop_reporting.wait(),
                        timeout=self._config.report_interval + jitter,
                    )
                    break  # Exit if stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue loop after interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.info(f"Reporting loop failed with error {e}")
                await asyncio.sleep(0.1)

    async def _wait_for_stop_condition(self) -> None:
        """Wait for the condition that ends this worker's execution loop.

        Races conditions with asyncio.FIRST_COMPLETED:
        1. SIGTERM on the process.
        2. Parent dead locally — consecutive send failures exceeded threshold.
        3. Work done (non-cluster mode only) — task pool exhausted.
        4. Stop(TERMINATE/KILL) received from parent (non-root only).
        """
        import sys

        stop_tasks: Dict[str, asyncio.Task] = {}

        stop_tasks["stop_signal"] = asyncio.create_task(
            self._stop_signal_received.wait()
        )
        stop_tasks["parent_dead"] = asyncio.create_task(self._parent_dead_event.wait())

        if not self._config.cluster:
            stop_tasks["work_done"] = asyncio.create_task(
                self._scheduler.wait_for_completion()
            )

        if self.parent is not None:

            async def _recv_stop_from_parent():
                msg = await self._comm.recv_message_from_parent(Stop, block=True)
                if msg.type == StopType.KILL:
                    sys.exit(1)
                elif msg.type == StopType.TERMINATE:
                    return

            stop_tasks["parent_stop"] = asyncio.create_task(_recv_stop_from_parent())

        done, pending = await asyncio.wait(
            set(stop_tasks.values()), return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        if self._parent_dead_event.is_set():
            sys.exit(1)

    # -------------------------------------------------------------------------
    #                               Teardown
    # -------------------------------------------------------------------------

    async def _stop_monitor_tasks(self) -> None:
        """Signal stop events then cancel and await each monitor asyncio task.

        Sets the stop events for the submission loop, status reporter, and
        parent task-update listener before cancelling the corresponding tasks,
        ensuring each coroutine exits cleanly via ``CancelledError``.
        """
        self._stop_submission.set()
        self._stop_reporting.set()
        self._stop_task_update.set()
        for task in [
            self._submission_task,
            self._reporting_task,
            self._task_update_task,
            self._client_handler_task,
        ]:
            if task is not None and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    async def _send_final_results_and_status(self) -> ResultBatch:
        """Package completed task results, send them to the parent, then send the final status.

        Iterates over all tasks owned by this worker, collects results for
        those that succeeded or failed, and sends a ``ResultBatch`` to the
        parent.  Immediately afterwards sends a ``Status`` message tagged
        ``"final"`` so the parent knows this worker is finished.  If the
        status send fails the status is also written to a local JSON file as
        a fallback.
        """
        async with self._timer("final_status"):
            ##also send the final status
            final_status = self.get_status()
            final_status.tag = "final"
            success = await self._comm.send_message_to_parent(final_status)
            if self.parent:
                if success:
                    self.logger.info(f"{self.node_id}: Sent final status to parent")
                    msg = await self._comm.recv_message_from_parent(
                        HeartBeat, timeout=5.0
                    )
                    if msg is None:
                        self.logger.warning(
                            "Did not get the final status update ack from parent in 5 sec!"
                        )
                    else:
                        self.logger.info("Successfully received ack from parent")
                else:
                    self.logger.warning(
                        f"{self.node_id}: Failed to send final status to parent"
                    )
                    fname = os.path.join(os.getcwd(), f"{self.node_id}_status.json")
                    self.logger.info(f"{final_status}")
                    final_status.to_file(fname)
        result_batch = ResultBatch(sender=self.node_id)
        for task_id, task in self.tasks.items():
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                task_result = Result(
                    task_id=task_id,
                    data=self.tasks[task_id].result,
                    exception=str(self.tasks[task_id].exception),
                )
                result_batch.add_result(task_result)
            else:
                self.logger.warning(f"Task {task_id} status {task.status}")

        max_retries = 3
        if self.parent:
            for attempt in range(max_retries):
                success = await self._comm.send_message_to_parent(result_batch)
                if not success:
                    self.logger.warning(
                        f"{self.node_id}: Failed to send results to parent "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                ack = await self._comm.recv_message_from_parent(
                    HeartBeat, timeout=5.0
                )
                if ack is not None:
                    self.logger.info(
                        f"{self.node_id}: Successfully sent results and received ack from parent"
                    )
                    break
                self.logger.warning(
                    f"{self.node_id}: No ack for result batch from parent "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
            else:
                self.logger.warning(
                    f"{self.node_id}: Failed to get result batch ack after {max_retries} attempts"
                )

        return result_batch

    async def stop(self) -> None:
        """Gracefully shut down the worker in a fixed teardown order.

        1. Stop the scheduler's resource monitor so no new tasks are dispatched
           to the executor.
        2. Cancel all monitor asyncio tasks: submission loop, status reporter,
           parent task-update listener, and cluster client handler.
        3. Export Perfetto profiling traces if profiling was enabled.
        4. Close the comm layer and shut down the executor.

        Note: results and final status are sent to the parent inside
        ``run()`` (via ``_send_final_results_and_status``) before ``stop()``
        is called, so this method does not perform any result forwarding.
        """

        if self._checkpointer:
            await self._write_checkpoint()
            self.logger.info(f"{self.node_id}: Final checkpoint written")

        ##stop scheduler monitoring first
        await self._scheduler.stop_monitoring()
        await self._stop_monitor_tasks()

        if self._config.profile == "perfetto" and self._event_registry is not None:
            os.makedirs(os.path.join(os.getcwd(), "profiles"), exist_ok=True)
            # Export to Perfetto format
            fname = os.path.join(
                os.getcwd(), "profiles", f"{self.node_id}_perfetto.json"
            )
            self.logger.info(f"Exporting Perfetto trace to {fname}")
            self._event_registry.export_perfetto(fname)

            # Also export statistics
            stats = self._event_registry.get_statistics()
            fname = os.path.join(os.getcwd(), "profiles", f"{self.node_id}_stats.json")
            self.logger.info(f"Exporting event statistics to {fname}")
            with open(fname, "w") as f:
                json.dump(stats, f, indent=2)

        await self._comm.close()
        for executor in self._executor.values():
            executor.shutdown()

        return

    # -------------------------------------------------------------------------
    #                               Entry point
    # -------------------------------------------------------------------------

    async def run(self) -> ResultBatch:
        """Main entry point: initialise, execute tasks, collect results, stop."""
        async with self._timer("init"):
            ##lazy init
            await self._lazy_init()

        self.logger.info("Started waiting for stop condition")
        await self._wait_for_stop_condition()

        async with self._timer("result_collection"):
            if not self._parent_dead_event.is_set():
                all_results = await self._send_final_results_and_status()

        await self.stop()

        self.logger.info(f"{self.node_id} stopped")
        return all_results

    def create_an_event_loop(self) -> None:
        """Entry point for a new child process: run the async event loop."""
        asyncio.run(self.run())

    # -------------------------------------------------------------------------
    #                       Serialization and Deserialization
    # -------------------------------------------------------------------------

    def asdict(self, include_tasks: bool = False) -> dict:
        """Serialise this worker to a JSON-compatible dict for cross-process transfer."""
        obj_dict = {
            "type": self.type,
            "node_id": self.node_id,
            "config": self._config.model_dump_json(),
            "parent": self.parent.serialize() if self.parent else None,
            "children": {
                child_id: child.serialize() for child_id, child in self.children.items()
            },
            "parent_comm": self.parent_comm.get_state().serialize()
            if self.parent_comm
            else None,
        }

        if include_tasks:
            raise NotImplementedError(
                "Including tasks in serialization is not implemented yet."
            )

        return obj_dict

    @classmethod
    def fromdict(cls, data: dict) -> "AsyncWorker":
        """Reconstruct an AsyncWorker from a serialised dict (inverse of asdict)."""
        config = LauncherConfig.model_validate_json(data["config"])
        parent = (
            NodeInfo.deserialize(data["parent"]) if data["parent"] is not None else None
        )
        print(socket.gethostname(), data["children"])
        children = {
            child_id: NodeInfo.deserialize(child_json)
            for child_id, child_json in data["children"].items()
        }

        if config.comm_name == "async_zmq":
            parent_comm = (
                AsyncZMQComm.set_state(
                    AsyncZMQCommState.deserialize(data["parent_comm"])
                )
                if data["parent_comm"] is not None
                else None
            )
        else:
            raise ValueError(f"Unsupported comm type {config.comm_name}")

        worker = cls(
            id=data["node_id"],
            config=config,
            Nodes=None,  # Nodes will be received via NodeUpdate message
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm,
        )
        return worker
