import asyncio
import json
import os
import random
import signal
import socket
import time
from collections import deque
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from typing import Callable, Dict, Optional, Tuple, Union

from ensemble_launcher.checkpointing import Checkpointer
from ensemble_launcher.comm import (
    AsyncComm,
    AsyncZMQComm,
    AsyncZMQCommState,
    NodeInfo,
    NodeRequest,
    NodeUpdate,
    Result,
    ResultBatch,
    Status,
    Stop,
    StopType,
    TaskRequest,
    TaskUpdate,
)
from ensemble_launcher.comm.messages import IResultBatch, Ready, ResultAck
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
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

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

        # Node update monitor state
        self._stop_node_update = asyncio.Event()
        self._node_update_task = None

        # Cluster mode / workstealing mode
        self._stop_signal_received = asyncio.Event()

        # Intermediate result batch queue
        self._iresult_q: Dict[str, deque] = {}
        self._streamed_task_ids: set = set()

        # Parent-ready handshake: set when parent sends a Ready message
        self._parent_ready_event = asyncio.Event()
        self._parent_ready_monitor_task: Optional[asyncio.Task] = None

        self._flush_task: Optional[asyncio.Task] = None

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
            os.path.join(os.getcwd(), self._config.log_dir)
            if self._config.worker_logs
            else None
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
                parent_comm=self.parent_comm,
                parent_address=self.parent_comm.my_address
                if self.parent_comm
                else None,
                heartbeat_interval=self._config.heartbeat_interval,
                heartbeat_dead_threshold=self._config.heartbeat_dead_threshold,
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

        if self.parent:
            self._node_update_task = asyncio.create_task(self._node_update_monitor())
            self._parent_ready_monitor_task = asyncio.create_task(
                self._parent_ready_monitor()
            )

        if self._config.cluster and self.parent:
            self._task_update_task = asyncio.create_task(self._task_update_monitor())

        if self._config.cluster:
            self._client_handler_task = asyncio.create_task(
                self._client_request_handler()
            )
        if self.parent or self._config.cluster:
            self._flush_task = asyncio.create_task(self._periodic_flush_loop())

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
        await self._restore_comm_state()

        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

        # start parent endpoint monitors
        await self._comm.start_monitors()

        # Jitter before syncing to avoid thundering herd at startup.
        await asyncio.sleep(random.uniform(0, 0.1 * self._config.report_interval))

        # Syncronize with parent
        await self._sync_with_parent()

        executor_names = (
            self._config.task_executor_name
            if isinstance(self._config.task_executor_name, list)
            else [self._config.task_executor_name]
        )

        original_head_node = self._init_nodes.resources[0]
        if "async_mpi_processpool" in executor_names:
            self.logger.info(
                f"{self.node_id}: Reserving first core for gateway in MPI Pool executor. "
                f"No Tasks will be scheduled on cpu {self._init_nodes.resources[0].cpus[0]} of host {self._init_nodes.nodes[0]}"
            )
            if isinstance(original_head_node, NodeResourceCount):
                trimmed_head_node = NodeResourceCount(
                    ncpus=original_head_node.ncpus - 1, ngpus=original_head_node.ngpus
                )
            else:
                trimmed_head_node = NodeResourceList(
                    cpus=original_head_node.cpus[1:], gpus=original_head_node.gpus
                )

            self._init_nodes.resources[0] = trimmed_head_node

        # Init scheduler
        self._scheduler = AsyncTaskScheduler(
            self.logger.getChild("scheduler"),
            self._init_tasks,
            self._init_nodes,
            policy_config=self._config.policy_config,
            policy=self._config.task_scheduler_policy,
        )
        self.logger.debug("Scheduler init complete")

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
        self.logger.debug(f"Pending tasks size {len(self._scheduler._pending_tasks)}")

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
        kwargs["log_dir"] = self._config.log_dir

        ##Async mpi specific options
        kwargs["mpi_config"] = self._config.mpi_config

        # Async mpi pool specific options
        np = sum(
            [original_head_node.cpu_count]
            + [node.cpu_count for node in self.nodes.resources[1:]]
        )
        kwargs["mpi_info"] = {"np": np}
        all_nodes_identical = (
            all([original_head_node == node for node in self.nodes.resources[1:]])
            or len(self.nodes.nodes) <= 1
        )
        if all_nodes_identical:
            kwargs["mpi_info"]["ppn"] = original_head_node.cpu_count
            kwargs["mpi_info"]["hosts"] = ",".join(self.nodes.nodes)
            if self._config.mpi_config.cpu_bind_method != "none":
                kwargs["mpi_info"]["cpu_binding"] = ":".join(
                    list(map(str, original_head_node.cpus))
                )
        else:
            # if (
            #     "async_mpi_processpool" in executor_names
            #     and self._config.mpi_config.rankfile_flag is None
            # ):
            ## TODO: implement rankfile
            if "async_mpi_processpool" in executor_names:
                raise ValueError(
                    f"{self.node_id}: Not all nodes are identical"
                    f" and MPI flavour {self._config.mpi_config.flavor} doesn't support rankfile."
                    f"Impossible to initialize async_mpi_processpool."
                )

        kwargs["cpu_to_pid"] = {}
        pid = 0
        for host, res in zip(
            self.nodes.nodes, [original_head_node] + self.nodes.resources[1:]
        ):
            for cpu_id in res.cpus:
                kwargs["cpu_to_pid"][(host, cpu_id)] = pid
                pid += 1

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

    async def _restore_comm_state(self) -> None:
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

        ## Don't restore the node_info. This will be updated once the scheduler is restored
        await self._comm.update_node_info(self.info())

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

        ##Forward the result to the appropriate place.
        ##task_id to client map should be restored from scheduler state
        for result in self._ckpt_results.values():
            self._forward_result(result)

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
        """Request initial task assignment from parent and retry until received."""
        max_retries = 100 if not self._config.cluster else 2
        for attempt in range(max_retries):
            if self._comm.parent_dead_event.is_set():
                break
            await self._comm.send_message_to_parent(
                TaskRequest(
                    sender=self.node_id, ntasks=0, free_resources=self._init_nodes
                )
            )
            task_update = await self._comm.recv_message_from_parent(
                TaskUpdate, timeout=5.0
            )
            if task_update is not None:
                self.logger.info(
                    f"{self.node_id}: Received task update from parent containing {len(task_update.added_tasks)}"
                )
                for task in task_update.added_tasks:
                    self._init_tasks[task.task_id] = task
                break
            self.logger.warning(
                f"{self.node_id}: TaskRequest attempt {attempt + 1}/{max_retries} failed, retrying..."
            )

    async def _sync_with_parent(self) -> None:
        """Perform initial handshake with the parent: heartbeat, node update, task update."""
        if self.parent is None:
            return
        # sync heart beat with parent
        async with self._timer("heartbeat_sync"):
            if not await self._comm.sync_heartbeat_with_parent(timeout=1000.0):
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
            self.logger.info(f"{self.node_id}: Synced heartbeat with parent")

        max_retries = 100
        for attempt in range(max_retries):
            if self._comm.parent_dead_event.is_set():
                raise RuntimeError(
                    f"{self.node_id}: Parent died while waiting for NodeUpdate"
                )
            await self._comm.send_message_to_parent(NodeRequest(sender=self.node_id))
            node_update = await self._comm.recv_message_from_parent(
                NodeUpdate, timeout=5.0
            )
            if node_update is not None and node_update.nodes:
                self._init_nodes = node_update.nodes
                self.logger.info(
                    f"{self.node_id}: Received NodeUpdate with {len(self._init_nodes.nodes)} nodes"
                )
                break
            self.logger.warning(
                f"{self.node_id}: NodeRequest attempt {attempt + 1}/{max_retries} failed, retrying..."
            )
        else:
            raise RuntimeError(
                f"{self.node_id}: Failed to receive NodeUpdate after {max_retries} retries"
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

    async def _flush_dest_queue(self, dest_id: str) -> float:
        """Send all buffered results for dest_id as an IResultBatch (or nothing if empty)."""
        result_q = self._iresult_q.get(dest_id)
        if not result_q:
            return time.time()
        results = []
        while result_q:
            results.append(result_q.popleft())
        if not results:
            return
        msg = IResultBatch(sender=self.node_id, data=results)
        if dest_id.startswith("client:"):
            self.logger.info(f"Flushing queue of size {len(results)} to {dest_id}")
            await self._comm.send_message_to_child(dest_id, msg)
        else:
            await self._comm.send_message_to_parent(msg)

        return time.time()

    async def _periodic_flush_loop(self) -> None:
        """Background task: flush all buffered intermediate result queues on a fixed interval.

        Sleep duration is jittered ±5% to avoid thundering herd across workers.
        """
        while True:
            jitter = random.uniform(-0.05, 0.05) * self._config.result_flush_interval
            await asyncio.sleep(self._config.result_flush_interval + jitter)
            for dest_id in list(self._iresult_q.keys()):
                await self._flush_dest_queue(dest_id)

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

            # Buffer Result in per-destination queue; flush as IResultBatch (both modes)
            task_result = Result(
                sender=self.node_id,
                task_id=task_id,
                data=task.result if exception is None else None,
                success=(exception is None),
                exception=str(exception) if exception else None,
            )
            self._forward_result(task_result)

    def _forward_result(self, result: Result):
        task_id = result.task_id
        dest_id = self._scheduler.get_task_client(task_id)
        if dest_id is None and self.parent:
            dest_id = self.parent.node_id

        if dest_id is not None:
            if dest_id not in self._iresult_q:
                self._iresult_q[dest_id] = deque()

            result_q = self._iresult_q[dest_id]
            result_q.append(result)
            self._streamed_task_ids.add(task_id)
            if len(result_q) >= self._config.result_buffer_size:
                asyncio.create_task(self._flush_dest_queue(dest_id))

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
        self, taskupdate: TaskUpdate, client_id: Optional[str] = None
    ) -> Tuple[Dict[str, bool], Dict[str, bool]]:
        """Apply a TaskUpdate: add new tasks to the scheduler and cancel/delete removed ones.

        Returns (add_status, del_status) dicts mapping task_id -> success bool.
        """
        ##Add the tasks to scheduler
        add_status = {}
        del_status = {}
        for new_task in taskupdate.added_tasks:
            self.logger.debug(f"Adding new task {new_task}")
            add_status[new_task.task_id] = self._scheduler.add_task(
                new_task, client_id=client_id
            )
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
                self._update_tasks(msg, client_id=client_id)

    async def _parent_ready_monitor(self) -> None:
        """Wait for a Ready message from parent and set the parent-ready event."""
        try:
            msg = await self._comm.recv_message_from_parent(Ready, block=True)
            if msg is not None:
                self.logger.info(f"{self.node_id}: Received Ready from parent")
                self._parent_ready_event.set()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"{self.node_id}: Parent ready monitor error: {e}")

    async def _node_update_monitor(self) -> None:
        """Receive NodeUpdate messages from parent and update resource allocation."""
        while not self._stop_node_update.is_set():
            try:
                node_update = await self._comm.recv_message_from_parent(
                    NodeUpdate, block=True
                )
                if node_update is not None and node_update.nodes:
                    self.nodes = node_update.nodes
                    self.logger.info(
                        f"{self.node_id}: Node update received: {len(node_update.nodes.nodes)} nodes"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Node update monitor error: {e}")
                await asyncio.sleep(0.1)

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
                task.status = TaskStatus.RUNNING
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

                self.logger.info(
                    f"Submitted task {task_id}: {task.executable} with resources {req.resources} to {self._task_id_to_executor[task_id]} executor"
                )
                future.add_done_callback(self.create_done_callback(task))
                self._task_futures[task_id] = future
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
        if self._comm.parent_dead_event is not None:
            stop_tasks["parent_dead"] = asyncio.create_task(
                self._comm.parent_dead_event.wait()
            )

        if not self._config.cluster:
            stop_tasks["work_done"] = asyncio.create_task(
                self._scheduler.wait_for_completion()
            )

        if self.parent is not None:

            async def _recv_stop_from_parent():
                msg = await self._comm.recv_message_from_parent(Stop, block=True)
                if msg.type == StopType.KILL:
                    await self.stop()
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

        if (
            self._comm.parent_dead_event is not None
            and self._comm.parent_dead_event.is_set()
        ):
            await self.stop()
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
        self._stop_node_update.set()
        for task in [
            self._submission_task,
            self._reporting_task,
            self._task_update_task,
            self._node_update_task,
            self._client_handler_task,
            self._flush_task,
            self._parent_ready_monitor_task,
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
        # Flush any remaining buffered intermediate results before final teardown.
        for dest_id in list(self._iresult_q.keys()):
            await self._flush_dest_queue(dest_id)

        async with self._timer("final_status"):
            ##also send the final status
            final_status = self.get_status()
            final_status.tag = "final"
            if self.parent:
                max_retries = 10
                for i in range(max_retries):
                    success = await self._comm.send_message_to_parent(final_status)
                    if success:
                        self.logger.info(f"{self.node_id}: Sent final status to parent")
                        msg = await self._comm.recv_message_from_parent(
                            ResultAck, timeout=5.0 + i * 5.0
                        )
                        if msg is None:
                            self.logger.warning(
                                "Did not get the final status update ack from parent in 5 sec!"
                            )
                        else:
                            self.logger.info("Successfully received ack from parent")
                            break
                    else:
                        self.logger.warning(
                            f"{self.node_id}: Failed to send final status to parent"
                        )
                        fname = os.path.join(os.getcwd(), f"{self.node_id}_status.json")
                        self.logger.info(f"{final_status}")
                        final_status.to_file(fname)
                        break
        result_batch = ResultBatch(sender=self.node_id)
        for task_id, task in self.tasks.items():
            if task_id in self._streamed_task_ids:
                continue  # already sent via IResultBatch
            if task.status == TaskStatus.SUCCESS or task.status == TaskStatus.FAILED:
                task_result = Result(
                    task_id=task_id,
                    data=self.tasks[task_id].result,
                    exception=str(self.tasks[task_id].exception),
                )
                result_batch.add_result(task_result)
            else:
                self.logger.warning(f"Task {task_id} status {task.status}")

        max_retries = 10
        if self.parent:
            # Wait for parent to signal it is ready before sending the result
            # batch to avoid a thundering-herd of simultaneous sends.
            self.logger.info(f"{self.node_id}: Waiting for Ready signal from parent")
            await self._parent_ready_event.wait()
            jitter = random.uniform(0, 5.0)
            self.logger.info(
                f"{self.node_id}: Parent is ready; sleeping {jitter:.2f}s before sending results"
            )
            await asyncio.sleep(jitter)
            for attempt in range(max_retries):
                success = await self._comm.send_message_to_parent(result_batch)
                if not success:
                    self.logger.warning(
                        f"{self.node_id}: Failed to send results to parent "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    continue
                ack = await self._comm.recv_message_from_parent(
                    ResultAck, timeout=5.0 + attempt * 5.0
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
            if hasattr(executor, "ashutdown"):
                await executor.ashutdown()
            else:
                executor.shutdown()

        self.logger.info(f"{self.node_id}: Closing logger")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # -------------------------------------------------------------------------
    #                               Entry point
    # -------------------------------------------------------------------------

    async def run(self) -> ResultBatch:
        """Main entry point: initialise, execute tasks, collect results, stop."""
        try:
            async with self._timer("init"):
                ##lazy init
                await self._lazy_init()

            self.logger.info("Started waiting for stop condition")
            await self._wait_for_stop_condition()

            async with self._timer("result_collection"):
                if (
                    self._comm.parent_dead_event is None
                    or not self._comm.parent_dead_event.is_set()
                ):
                    all_results = await self._send_final_results_and_status()
        finally:
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

        obj_dict["secret_id"] = self._secret_id

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
        worker._secret_id = data["secret_id"]
        return worker
