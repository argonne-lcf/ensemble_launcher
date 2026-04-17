import asyncio
import base64
import json
import os
import random
import signal
import socket
import time
import uuid
from collections import deque
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from typing import Callable, Dict, List, Optional, Union

from ensemble_launcher.checkpointing import Checkpointer
from ensemble_launcher.comm import (
    AsyncComm,
    AsyncZMQComm,
    AsyncZMQCommState,
    NodeInfo,
)
from ensemble_launcher.comm.messages import (
    IResultBatch,
    NodeRequest,
    NodeUpdate,
    Ready,
    Result,
    ResultAck,
    ResultBatch,
    Status,
    Stop,
    StopType,
    TaskRequest,
    TaskUpdate,
)
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task, TaskStatus
from ensemble_launcher.executors import Executor, executor_registry
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.profiling import EventRegistry, get_registry
from ensemble_launcher.scheduler import AsyncChildrenScheduler, ChildrenAssignment
from ensemble_launcher.scheduler.child_state import ChildState
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from .async_worker import AsyncWorker
from .node import Node
from .utils import (
    async_load_str,
    async_load_str_file,
    async_simple_load_str,
    async_simple_load_str_file,
)

AsyncFuture = asyncio.Future


class AsyncMaster(Node):
    """Hierarchical master node that manages a layer of child workers or sub-masters.

    Responsible for scheduling tasks onto children, launching child processes,
    collecting results, and aggregating status up to its parent (or writing to
    disk if it is the root). Supports both flat and multi-level tree deployments.
    """

    type = "AsyncMaster"

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
        self._init_tasks = tasks
        self._init_nodes = Nodes
        self._config = config
        self._parent_comm = parent_comm

        ##lazily created in run
        self._executor = None
        self._comm = None

        self._scheduler = None

        ##maps
        self._children_futures: Dict[str, Union[AsyncFuture, ConcurrentFuture]] = {}
        self._children_results: Dict[str, Result] = {}
        self._results: Dict[str, List[ResultBatch]] = {}
        self._child_objs: Dict[str, Node] = {}

        self.logger = None

        # Initialize event registry for perfetto profiling
        self._event_registry: Optional[EventRegistry] = None

        # asyncio event
        # _all_children_done_event is owned by the scheduler after _lazy_init;
        # use __fallback_done_event before the scheduler is created.
        self.__fallback_done_event = asyncio.Event()
        self._stop_reporting_event = asyncio.Event()
        self._event_loop = None  # Will be set in run()

        # result and aggregate tasks
        self._aggregate_task = None
        self._child_result_batch_task: Dict[
            str, asyncio.Task
        ] = {}  # child_id -> collect task
        self._child_forwarder_task: Dict[
            str, asyncio.Task
        ] = {}  # child_id -> result monitor task

        self._child_status_task: Dict[
            str, asyncio.Task
        ] = {}  # child_id -> status monitor task

        self._child_liveness_task: Dict[
            str, asyncio.Task
        ] = {}  # child_id -> liveness monitor task

        self._checkpointer: Optional[Checkpointer] = None
        # Cached checkpoint data — populated early in _run() before sockets are set up.
        self._ckpt_data: Optional[tuple] = None  # (sched_state, comm_state, tasks)

        # Cluster mode state
        self._stop_task_update = asyncio.Event()
        self._task_update_task: Optional[asyncio.Task] = None
        self._client_monitor_task: Optional[asyncio.Task] = None
        self._reporting_task: asyncio.Task = None
        self._stop_signal_received = asyncio.Event()

        # Per-child task request monitor tasks
        self._child_task_request_task: Dict[str, asyncio.Task] = {}

        # Tasks already sent to children via _route_tasks (before initial sync completes).
        # Used by _build_init_task_update to avoid double-sending.
        self._routed_task_ids: Dict[str, set] = {}  # child_id -> set of task_ids

        # Intermediate result buffer: used when forwarding to a parent master (fan-in aggregation)
        self._iresult_q: Dict[str, deque] = {}
        self._flush_task: Optional[asyncio.Task] = None
        # Intermediate task buffer: batches tasks dispatched to children (fan-out)
        self._itask_q: Dict[str, deque] = {}
        self._task_flush_task: Optional[asyncio.Task] = None
        # Batch mode root: accumulates results streamed via IResultBatch during execution
        self._batch_streaming_results: List[Result] = []

        # Parent-ready handshake: set when parent sends a Ready message
        self._parent_ready_event = asyncio.Event()
        self._parent_ready_monitor_task: Optional[asyncio.Task] = None

        self._total_received: int = 0
        self._total_streamed = 0

    @asynccontextmanager
    async def _timer(self, event_name: str):
        """Timer that records to event registry for Perfetto export."""
        if self._event_registry is not None:
            with self._event_registry.measure(
                event_name, "async_master", node_id=self.node_id, pid=os.getpid()
            ):
                yield
        else:
            yield

    @property
    def _all_children_done_event(self) -> asyncio.Event:
        """Delegate to scheduler's event once available; use fallback before _lazy_init."""
        if self._scheduler is not None:
            return self._scheduler._all_children_done_event
        return self.__fallback_done_event

    @property
    def nodes(self) -> JobResource:
        """Node resource allocation owned by the scheduler cluster."""
        return self._scheduler.cluster.nodes

    @nodes.setter
    def nodes(self, value: JobResource) -> None:
        self._scheduler.cluster.update_nodes(value)

    @property
    def parent_comm(self) -> Optional[AsyncComm]:
        """Communication channel to the parent node, or None if this is the root."""
        return self._parent_comm

    @parent_comm.setter
    def parent_comm(self, value: AsyncComm) -> None:
        self._parent_comm = value

    @property
    def comm(self) -> Optional[AsyncComm]:
        """Communication channel for this master (connecting to parent and children)."""
        return self._comm

    @property
    def tasks(self) -> Dict[str, Task]:
        """All tasks owned by this master's scheduler."""
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
        """Configure the logger for this master, optionally writing to a per-node log file."""
        log_dir = (
            os.path.join(os.getcwd(), self._config.log_dir)
            if self._config.master_logs
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

    def _create_scheduler(self) -> AsyncChildrenScheduler:
        """Instantiate the worker scheduler.  Override to change init args."""
        return AsyncChildrenScheduler(
            self.logger.getChild("scheduler"),
            self._init_nodes,
            self._config,
            tasks=self._init_tasks,
            node_id=self.node_id,
            level=self.level,
        )

    def _get_child_class(
        self, child_assignment: Optional[ChildrenAssignment] = None
    ) -> type:
        """Return the Node class to use for children at the next level."""
        if (
            child_assignment is not None
            and child_assignment.get("child_class", None) is not None
        ):
            child_class = child_assignment["child_class"]
            if child_class.lower() == "asyncworker":
                return AsyncWorker
            elif child_class.lower() == "asyncmaster":
                return AsyncMaster
            elif child_class.lower() == "asyncworkstealingmaster":
                from .async_workstealing_master import AsyncWorkStealingMaster

                return AsyncWorkStealingMaster
            else:
                raise RuntimeError(f"Unknown child_class {child_class}")
        else:
            if self.level + 1 == self._config.policy_config.nlevels:
                return AsyncWorker
            from .async_workstealing_master import AsyncWorkStealingMaster

            return (
                AsyncWorkStealingMaster
                if self._config.enable_workstealing
                and self.level + 1 == self._config.policy_config.nlevels - 1
                else AsyncMaster
            )

    def _instantiate_children(
        self,
        include_tasks: bool,
        target_ids: set,
    ) -> Dict[str, Node]:
        """Instantiate Node objects for target_ids using current scheduler assignments."""
        NodeClass = self._get_child_class()
        children = {}
        for child_id in target_ids:
            alloc = self._scheduler.child_assignments[child_id]
            child_config = self._config
            if "task_executor_name" in alloc:
                child_config = self._config.model_copy(
                    update={"task_executor_name": alloc["task_executor_name"]}
                )
                self.logger.info(
                    f"{self.node_id}: Child {child_id} using task_executor_name: {alloc['task_executor_name']}"
                )
            children[child_id] = NodeClass(
                child_id,
                config=child_config,
                Nodes=alloc["job_resource"],
                tasks={
                    task_id: self._scheduler.tasks[task_id]
                    for task_id in alloc["task_ids"]
                }
                if include_tasks
                else {},
                parent=None,
            )
        return children

    def _apply_resource_headroom(self) -> None:
        """Reserve one CPU on the first child's head node for this master process."""
        first_child_id = next(iter(self._scheduler.child_assignments))
        first_job_resource = self._scheduler.get_child_assignment(first_child_id)[
            "job_resource"
        ]
        first_node = first_job_resource.resources[0]
        if isinstance(first_node, NodeResourceList):
            first_job_resource.resources[0] = NodeResourceList(
                cpus=first_node.cpus[1:], gpus=first_node.gpus
            )
        else:
            first_job_resource.resources[0] = NodeResourceCount(
                ncpus=first_node.cpu_count - 1, ngpus=first_node.gpu_count
            )

    def _create_children(
        self,
        include_tasks: bool = False,
        partial: bool = False,
        nodes: Optional[JobResource] = None,
    ) -> Dict[str, Node]:
        """Assign tasks via the scheduler and instantiate child Node objects.

        Uses self._scheduler.tasks as the task source.
        partial=True: additive — preserves running children, offsets new wids, and
            returns only newly created children.
        nodes: restrict assignment to these nodes (e.g. recovered nodes on retry).
        """
        existing_ids = set(self._scheduler.children_names) if partial else set()

        # Phase 1: determine resource layout and register children (no tasks yet).
        self._scheduler.assign_resources(
            self.level, self.node_id, reset=not partial, nodes=nodes
        )

        if not partial and not self._config.overload_orchestrator_core:
            self._apply_resource_headroom()

        # Phase 2: distribute tasks from the unassigned pool to registered children.
        self._scheduler.assign_task_ids(self._scheduler.unassigned_task_ids)

        target_ids = set(self._scheduler.child_assignments.keys()) - existing_ids
        return self._instantiate_children(include_tasks, target_ids)

    async def _init_child(self, child_id: str, child: Node) -> None:
        """Register a single child and start its per-child monitoring tasks.

        Initialises the comm cache for this child (via update_node_info) so that
        the collect / forwarder tasks can safely block on the message queue from
        the moment they are created.
        """
        self._child_objs[child_id] = child
        self.add_child(child_id, child.info())
        child.set_parent(self.info())
        child.parent_comm = self.comm.pickable_copy()
        # Extend the comm's node-info so its cache gains an entry for this child.
        await self._comm.update_node_info(self.info())
        # Per-child collect task (waits for the child's final ResultBatch).
        task = asyncio.create_task(
            self._collect_final_result_batch_from_child(child_id)
        )
        self._child_result_batch_task[child_id] = task
        # Per-child status monitor (continuously drains Status messages).
        self._child_status_task[child_id] = asyncio.create_task(
            self._child_status_monitor(child_id)
        )
        # Per-child liveness monitor (waits for HB process dead event).
        self._child_liveness_task[child_id] = asyncio.create_task(
            self._child_liveness_monitor(child_id)
        )
        # Per-child task request monitor (handles initial or continuous TaskRequests).
        self._child_task_request_task[child_id] = asyncio.create_task(
            self._monitor_single_child_task_requests(child_id)
        )
        self._child_forwarder_task[child_id] = asyncio.create_task(
            self._child_result_monitor(child_id)
        )

    def _create_monitor_tasks(self) -> None:
        """Start long-running asyncio monitor tasks (status reporter, cluster monitors).

        Per-child collect/forwarder tasks are started in _init_child.
        The aggregate task is created at the end of _lazy_init once all
        children (including any recreated ones) have been registered.
        """
        self._reporting_task = asyncio.create_task(self._report_status())
        if self.parent:
            self._parent_ready_monitor_task = asyncio.create_task(
                self._parent_ready_monitor()
            )
        if self._config.cluster:
            self._client_monitor_task = asyncio.create_task(
                self._client_request_monitor()
            )
            if self.parent:
                self._task_update_task = asyncio.create_task(
                    self._parent_task_update_monitor()
                )
            self._task_flush_task = asyncio.create_task(
                self._periodic_task_flush_loop()
            )
        if self.parent:
            self._flush_task = asyncio.create_task(self._periodic_flush_loop())

    def _cpu_bind_mpi_kwargs(self, child_resource) -> dict:
        """Return mpi_kwargs that pin a child process to its allocated CPUs.

        Child processes are launched as a single MPI rank (NodeResourceCount),
        so CPU binding cannot go through _build_resource_cmd (which only applies
        to NodeResourceList resources).  Instead we inject it as additional MPI
        options via mpi_kwargs, using MPIConfig to produce the correct flags.
        """
        cfg = self._config.mpi_config

        if cfg.cpu_bind_method == "none" or not cfg.cpu_bind_flag:
            return {}

        if cfg.cpu_bind_method == "bind-to":
            # OpenMPI: --bind-to core --map-by <map_by>
            # No need for explicit CPU IDs — MPI assigns cores automatically.
            return {cfg.cpu_bind_flag: "core", "--map-by": cfg.openmpi_map_by}

        # "list" method: build the colon-separated core-ID string
        if isinstance(child_resource, NodeResourceList):
            cpus = ":".join(map(str, child_resource.cpus))
        else:
            # NodeResourceCount — fall back to 0..N-1 (best effort)
            cpus = ":".join(map(str, range(child_resource.cpu_count)))

        if cfg.cpu_bind_method == "list":
            return {cfg.cpu_bind_flag: f"list:{cpus}"}

        self.logger.warning(
            f"Unknown cpu_bind_method '{cfg.cpu_bind_method}'. Not setting child affinity."
        )
        return {}

    async def _launch_child(
        self, child_name: str, child_obj: Node, child_idx: int
    ) -> None:
        """Launch a single child process.

        Args:
            child_name: The ID of the child to launch.
            child_obj: The child Node object.
            child_idx: Index of the child, used to set EL_CHILDID in the environment.
        """

        if self._config.child_executor_name == "async_mpi":
            child_nodes = child_obj.init_nodes
            head_node = child_nodes.nodes[0]

            # Serialize child object
            child_dict = child_obj.asdict()
            json_str = json.dumps(child_dict, default=str)

            # Write JSON to a temp file to avoid ARG_MAX limits on large payloads
            json_fname = os.path.join(
                self._executor.tmp_dir, f"child_{uuid.uuid4()}.json"
            )
            if hasattr(self._executor, "write_file_to_nodes"):
                await self._executor.write_file_to_nodes(
                    json_fname, json_str, [head_node]
                )
            else:
                with open(json_fname, "w") as _f:
                    _f.write(json_str)
            load_str_embed = async_simple_load_str_file.replace(
                "json_file_path", f"'{json_fname}'"
            )

            req = JobResource(resources=[NodeResourceCount(ncpus=1)], nodes=[head_node])
            env = os.environ.copy()
            env["EL_CHILDID"] = str(child_idx)

            mpi_kwargs = self._cpu_bind_mpi_kwargs(child_obj.init_nodes.resources[0])
            self.logger.info(
                f"Launching child {child_name} with mpi_kwargs={mpi_kwargs}"
            )

            future = self._executor.submit(
                req, ["python", "-c", load_str_embed], env=env, mpi_kwargs=mpi_kwargs
            )
            cb = self._create_done_callback([child_name])
            if cb is not None:
                future.add_done_callback(cb)
            self._children_futures[child_name] = future
        else:
            self.logger.info("Not using MPIexec")
            child_nodes = child_obj.init_nodes.nodes
            req = JobResource(
                resources=[
                    NodeResourceList(cpus=child_obj.init_nodes.resources[0].cpus)
                ],
                nodes=child_nodes[:1],
            )
            self.logger.info("Created req")
            env = os.environ.copy()
            env["EL_CHILDID"] = str(child_idx)

            self.logger.info(f"Submitting job,{child_obj}")
            future = self._executor.submit(req, child_obj.create_an_event_loop, env=env)
            self.logger.info("Submitted to the executor")
            cb = self._create_done_callback([child_name])
            if cb is not None:
                future.add_done_callback(cb)
            self.logger.info("Added callback")
            self._children_futures[child_name] = future

    async def _launch_children(self, child_names: List[str]) -> None:
        """Submit all named children to the executor."""
        children = {}
        for child_name in child_names:
            children[child_name] = self._child_objs[child_name]

        if self._config.child_executor_name == "async_mpi":
            first_headnode = next(iter(children.values())).init_nodes.resources[0]
            worker_equality = all(
                [
                    child.init_nodes.resources[0] == first_headnode
                    for child in children.values()
                ]
            )
            if len(child_names) > 1 and (
                (worker_equality and self._config.sequential_child_launch is None)
                or (
                    self._config.sequential_child_launch is not None
                    and not self._config.sequential_child_launch
                )
            ):
                self.logger.info("Using one shot mpiexec")
                if not worker_equality:
                    self.logger.warning(
                        "All workers are not equal. Using one shot mpiexec can cause issues"
                    )
                ##launch all children in a single shot
                child_head_nodes = []
                child_resources = []
                child_obj_dict = {}

                for child_name, child_obj in children.items():
                    head_node = child_obj.init_nodes.nodes[0]
                    child_head_nodes.append(head_node)
                    child_resources.append(NodeResourceCount(ncpus=1))
                    child_obj_dict[head_node + "-" + child_name] = child_obj

                # Build combined dictionary structure
                common_keys = ["type", "parent", "parent_comm"]
                if all(
                    [
                        "task_executor_name" not in cdict
                        for cdict in self._scheduler.child_assignments.values()
                    ]
                ):
                    common_keys.append("config")
                self.logger.info(f"common keys: {common_keys}")
                first_child = next(iter(child_obj_dict.values()))
                first_dict = first_child.asdict()

                # Initialize with common keys from first child
                final_dict = {key: first_dict[key] for key in common_keys}

                # Initialize per-host keys as empty dicts
                for key in first_dict.keys():
                    if key not in common_keys:
                        final_dict[key] = {}

                # Populate per-host values
                for hostname_child_id, child_obj in child_obj_dict.items():
                    hostname = hostname_child_id.split("-")[0]
                    child_dict = child_obj.asdict()
                    for key, value in child_dict.items():
                        if key not in common_keys:
                            if hostname not in final_dict[key]:
                                final_dict[key][hostname] = [value]
                            else:
                                final_dict[key][hostname].append(value)

                self.logger.info(f"Final dict: {final_dict}")
                # Write JSON to a temp file to avoid ARG_MAX limits on large payloads
                json_str = json.dumps(final_dict, default=str)
                json_fname = os.path.join(
                    self._executor.tmp_dir, f"workers_{uuid.uuid4()}.json"
                )
                if hasattr(self._executor, "write_file_to_nodes"):
                    await self._executor.write_file_to_nodes(
                        json_fname, json_str, child_head_nodes
                    )
                else:
                    with open(json_fname, "w") as _f:
                        _f.write(json_str)
                common_keys_str = ",".join(common_keys)
                load_str_embed = async_load_str_file.replace(
                    "json_file_path", f"'{json_fname}'"
                )
                load_str_embed = load_str_embed.replace(
                    "common_keys_str", f"'{common_keys_str}'"
                )

                req = JobResource(resources=child_resources, nodes=child_head_nodes)
                env = os.environ.copy()

                # CPU binding: use first child's resource as the representative.
                # All workers are equal when we reach here (worker_equality check above),
                # so every rank on every node gets the same core list.
                if worker_equality:
                    mpi_kwargs = self._cpu_bind_mpi_kwargs(
                        first_child.init_nodes.resources[0]
                    )
                else:
                    mpi_kwargs = {}
                    self.logger.warning("Workers are not equal — skipping CPU binding.")
                self.logger.info(f"One-shot launch mpi_kwargs={mpi_kwargs}")

                future = self._executor.submit(
                    req,
                    ["python", "-c", load_str_embed],
                    env=env,
                    mpi_kwargs=mpi_kwargs,
                )

                # Generate one UUID for all children in this one-shot launch
                child_info = []
                for child_id in children.keys():
                    child_info.append(child_id)
                    self._children_futures[child_id] = future
                cb = self._create_done_callback(child_info)
                if cb is not None:
                    future.add_done_callback(cb)
            else:
                ##launch children in parallel using gather
                launch_tasks = [
                    self._launch_child(child_name, child_obj, child_idx)
                    for child_idx, (child_name, child_obj) in enumerate(
                        children.items()
                    )
                ]
                await asyncio.gather(*launch_tasks)
        else:
            ##launch children in parallel using gather
            launch_tasks = [
                self._launch_child(child_name, child_obj, child_idx)
                for child_idx, (child_name, child_obj) in enumerate(children.items())
            ]
            await asyncio.gather(*launch_tasks)

    async def _lazy_init(self) -> None:
        """Set up all resources needed before task execution begins.

        In order: logging, event loop capture, comm, parent sync, scheduler,
        executor, children creation/launch, and monitor task creation.
        """
        if self._config.profile == "perfetto":
            self._event_registry = get_registry()
            self._event_registry.enable()
            os.environ["EL_ENABLE_PROFILING"] = "1"

        # Store event loop for thread-safe event signaling from callbacks
        self._event_loop = asyncio.get_event_loop()
        self._event_loop.add_signal_handler(
            signal.SIGTERM, self._stop_signal_received.set
        )

        # create logger
        tick = time.perf_counter()
        self._setup_logger()
        tock = time.perf_counter()
        self.logger.info(
            f"{self.node_id}: Logger setup time: {tock - tick:.4f} seconds"
        )

        try:
            self.logger.info(
                f"My cpu affinity: {os.sched_getaffinity(0)}, my hostname: {socket.gethostname()}"
            )
        except Exception:
            pass

        # Read checkpoint data early so both comm address and scheduler state
        # can be restored from the cached data at the appropriate points below.
        await self._read_checkpoint_data()

        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm()  ###This will only create picklable objects

        # Restore saved comm state before binding so children can reconnect.
        await self._restore_comm_state()

        # for zmq, setup the sockets
        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

        # Start parent comm end point monitor
        await self._comm.start_monitors(parent_only=True)

        # Receive node update from parent if it has a parent
        self.logger.info("Syncing with parent")
        if self.parent:
            await self._sync_with_parent()
        self.logger.info("Done Syncing with parent")

        # create scheduler
        self._scheduler = self._create_scheduler()

        # Validate that nodes are initialized
        if not self.nodes:
            self.logger.error(f"{self.node_id}: Nodes not initialized!")
            raise RuntimeError(
                f"{self.node_id}: Nodes must be initialized before execution"
            )

        self.logger.info(
            f"{self.node_id}: Have {len(self.tasks)} tasks after update from parent"
        )

        # Restore scheduler state from cached checkpoint data (comm already restored).
        # Restores the node and task assignments of the children
        ckpt_restored = self._restore_scheduler_checkpoint()

        # If there are still tasks un-assigned try to assign them
        if ckpt_restored and len(self._scheduler.unassigned_task_ids):
            self._scheduler.assign_task_ids(self._scheduler.unassigned_task_ids)

        # Check executor validity
        assert self._config.child_executor_name in executor_registry.async_executors, (
            f"Executor {self._config.child_executor_name} not found in async executors {executor_registry.async_executors}"
        )

        kwargs = {}
        kwargs["logger"] = self.logger.getChild("executor")
        kwargs["max_workers"] = self.nodes.resources[0].cpu_count
        if self._config.child_executor_name == "async_mpi":
            kwargs["mpi_config"] = self._config.mpi_config

        # Create executor
        self._executor: Executor = executor_registry.create_executor(
            self._config.child_executor_name, kwargs=kwargs
        )
        self.logger.info(f"Created {self._config.child_executor_name} executor")

        # Create children: use restored assignment topology if available, else run policy.
        if ckpt_restored:
            children = self._instantiate_children(
                False, set(self._scheduler.children_names)
            )
            self.logger.info(
                f"{self.node_id}: Recreated {len(children)} children from checkpoint: {list(children.keys())}"
            )
        else:
            children = self._create_children()
            self.logger.info(
                f"{self.node_id}: Created {len(children)} children: {list(children.keys())}"
            )
        for child_id, child in children.items():
            await self._init_child(child_id, child)

        # Start the shared comm monitor for all child sockets (idempotent).
        await self._comm.start_monitors(children_only=True)

        # Start global monitors (status reporting, result aggregation, cluster tasks).
        self._create_monitor_tasks()

        # Launch and sync children, retrying failures up to 2 times
        children_names = self._scheduler.children_names
        results = await self._launch_and_sync_children(children_names)
        failed_names = [
            name for name, r in zip(children_names, results) if r is not None
        ]

        max_retries = 3
        for attempt in range(max_retries):
            recover_tasks = []
            for name in failed_names:
                recover_tasks.append(self._recover_dead_child(name))
            results = await asyncio.gather(*recover_tasks)

            new_failed_names = []
            for name, success in zip(failed_names, results):
                if not success:
                    new_failed_names.append(name)

            failed_names = new_failed_names

            if len(failed_names) == 0:
                break

        if failed_names:
            self.logger.warning(
                f"{self.node_id}: {len(failed_names)} children still failed after "
                f"{max_retries} retries"
            )

        # All children are launched and synced — signal each one that the master
        # is ready so they can send their final ResultBatch when done.
        for child_id in list(self._child_result_batch_task.keys()):
            await self._comm.send_message_to_child(child_id, Ready(sender=self.node_id))
            self.logger.info(f"{self.node_id}: Sent Ready to {child_id}")

        # Create aggregate task once, after all children (including recreated ones) are registered.
        self._aggregate_task = asyncio.create_task(
            self._aggregate_and_send_result_batch()
        )

        return None

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
            self.node_id, self._config.checkpoint_dir, self.logger
        )
        if not self._checkpointer.checkpoint_exists():
            return
        self._ckpt_data = await self._checkpointer.read_checkpoint()

    async def _restore_comm_state(self) -> None:
        """Restore comm from cached checkpoint data using the comm's set_state.

        Must be called after _create_comm() and before setup_zmq_sockets() so
        the node re-binds to its previous address and reconnecting children
        can find it.
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

        ## Don't restore the node_info. This will be updated once the scheduler is end
        await self._comm.update_node_info(self.info())
        self.logger.info(
            f"{self._secret_id}, {self._comm._node_info.secret_id}, {self.info().secret_id}"
        )

    def _restore_scheduler_checkpoint(self) -> bool:
        """Restore scheduler state from cached checkpoint data.

        Called after scheduler creation. Returns True if child assignment
        topology was successfully restored.
        """
        if self._ckpt_data is None:
            return False
        ckpt_sched_state, _, _ = self._ckpt_data
        if ckpt_sched_state is None or not ckpt_sched_state.children_task_ids:
            return False
        self._scheduler.set_state(ckpt_sched_state)
        self.logger.info(
            f"{self.node_id}: Scheduler state restored from checkpoint "
            f"({len(ckpt_sched_state.children_task_ids)} children)"
        )
        return True

    async def _write_checkpoint(self) -> None:
        """Write scheduler state and comm state to checkpoint."""
        if self._checkpointer is None:
            return
        await self._checkpointer.write_checkpoint(
            scheduler_state=self._scheduler.get_state(self.node_id),
            comm_state=self._comm.get_state(),
        )

    # --------------------------------------------------------------------------
    #                               Parent Synchronization
    # --------------------------------------------------------------------------

    async def _sync_with_parent(self) -> None:
        """Perform initial handshake with the parent: heartbeat, node update, task update."""
        # sync heart beat with parent
        if self.parent is None:
            return
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

        # Request initial task assignment from parent
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

        return

    # --------------------------------------------------------------------------
    #                               Child Synchronization
    # --------------------------------------------------------------------------

    def _build_init_node_update(self, child_id: str) -> NodeUpdate:
        """Build the initial NodeUpdate message to send to a child at startup."""
        child_nodes = self._scheduler.get_child_assignment(child_id)["job_resource"]
        return NodeUpdate(sender=self.node_id, nodes=child_nodes)

    def _build_init_task_update(self, child_id: str) -> TaskUpdate:
        """Build the initial TaskUpdate message containing all tasks assigned to a child."""
        already_sent = self._routed_task_ids.get(child_id, set())
        new_tasks = [
            self._scheduler.tasks[task_id]
            for task_id in self._scheduler.get_child_assignment(child_id)["task_ids"]
            if task_id not in already_sent
        ]
        if not new_tasks:
            return None
        return TaskUpdate(sender=self.node_id, added_tasks=new_tasks)

    async def _get_child_exception(self, child_id: str) -> Optional[Result]:
        """
        Collect and handle exception from a single child process.

        Args:
            child_id: The ID of the child to check for exceptions

        Returns:
            Result: A Result object with the exception if the child failed, None otherwise.
                    The Result has the child_id as sender and the exception stored as a string
                    in its exception attribute.
        """
        future = self._children_futures.get(child_id)
        if future is None:
            self.logger.warning(f"Child {child_id} not found in futures")
            return None

        # Stop the child if not done
        if not future.done():
            self.logger.info(f"Stopping child {child_id}")
            future.cancel()

        # Collect exception without waiting
        if future.done():
            try:
                exception = future.exception()
                if exception is not None:
                    self.logger.error(
                        f"Child {child_id} failed with exception: {exception}"
                    )
                    exception_result = Result(sender=child_id, data=[])
                    exception_result.exception = str(exception)
                    return exception_result
                else:
                    result = future.result()
                    self.logger.error(
                        f"Child {child_id}: No child exception found! Got {result}"
                    )
            except asyncio.CancelledError:
                pass

        return None

    async def _sync_with_child(
        self,
        child_id: str,
        node_update: Optional[NodeUpdate],
    ) -> Optional[Result]:
        """
        Sync with a single child and send initial node/task updates.

        Args:
            child_id: The ID of the child to sync with
            node_update: NodeUpdate message to send to the child

        Returns:
            None if successful, Result object with exception if failed
        """
        # Sync heartbeat with child
        if not await self._comm.sync_heartbeat_with_child(
            child_id=child_id, timeout=600.0
        ):
            self.logger.error(f"Failed to sync heartbeat with child {child_id}")
            return await self._get_child_exception(child_id)
        self.logger.info(f"Successfully synced heart beat with {child_id}")
        self._scheduler.mark_child_running(child_id)
        child_dead_event = self._comm._child_dead_events.get(child_id, None)
        if child_dead_event is not None and child_dead_event.is_set():
            child_dead_event.clear()

        # Wait for NodeRequest from child before sending NodeUpdate
        node_req = await self._comm.recv_message_from_child(
            NodeRequest, child_id=child_id, timeout=600.0
        )
        if node_req is None:
            self.logger.warning(
                f"{self.node_id}: No NodeRequest from {child_id}, sending NodeUpdate anyway"
            )

        # Send NodeUpdate (now triggered by child's request)
        if node_update is not None:
            await self._comm.send_message_to_child(child_id, node_update)
            self.logger.info(
                f"{self.node_id}: Sent node update to {child_id} containing {len(node_update.nodes.nodes)} nodes"
            )

        return None

    async def _sync_with_children(
        self, child_names: List[str]
    ) -> List[Optional[Result]]:
        """Sync with all children and send initial node/task updates."""
        # Prepare updates for each child
        sync_tasks = []
        for child_id in child_names:
            # Create node update
            node_update = self._build_init_node_update(child_id)

            ##Task update handled by the seperate tast request monitor

            # Add sync task
            sync_tasks.append(self._sync_with_child(child_id, node_update))

        # Sync with all children in parallel
        results = await asyncio.gather(*sync_tasks, return_exceptions=True)
        return results

    async def _monitor_single_child_task_requests(self, child_id: str) -> None:
        """Base: receive initial TaskRequest from child and respond with pre-assigned tasks."""
        task_req = await self._comm.recv_message_from_child(
            TaskRequest, child_id=child_id, block=True
        )
        if task_req is not None:
            task_update = self._build_init_task_update(child_id)
            if task_update is not None:
                await self._comm.send_message_to_child(child_id, task_update)
                self.logger.info(
                    f"{self.node_id}: Sent initial task update to {child_id} "
                    f"containing {len(task_update.added_tasks)} tasks"
                )

    async def _launch_and_sync_children(
        self, child_names: List[str]
    ) -> List[Optional[Result]]:
        """Launch children and perform the initial sync handshake with each.

        Returns a list parallel to child_names where each entry is None on
        success, or a Result carrying the exception on failure.
        """
        self.logger.debug(f"launching {child_names}")
        await self._launch_children(child_names)
        self.logger.debug(f"Done launching {child_names}")
        results = await self._sync_with_children(child_names)
        self.logger.debug(f"Done syncing {child_names}")
        return results

    # --------------------------------------------------------------------------
    #                               Task Routing
    # --------------------------------------------------------------------------

    async def _route_tasks(
        self, tasks: List[Task], client_id: Optional[str] = None
    ) -> List[Optional[str]]:
        """Route a list of tasks to the best child via scheduler policy. Returns chosen child_id."""
        for task in tasks:
            self._scheduler.add_task(task, client_id=client_id)
        child_assignments, task_to_child, unassigned_tasks = (
            self._scheduler.assign_task_ids({task.task_id for task in tasks})
        )

        for child_id, added_tasks in child_assignments.items():
            if len(added_tasks) == 0:
                continue
            await self._scheduler.wait_for_child_ready(child_id)
            if child_id not in self._itask_q:
                self._itask_q[child_id] = deque()
            task_q = self._itask_q[child_id]
            for task_id in added_tasks:
                task_q.append(self.tasks[task_id])
            self._routed_task_ids.setdefault(child_id, set()).update(added_tasks)
            self.logger.debug(f"Buffering {len(added_tasks)} Tasks for {child_id}")
            if len(task_q) >= self._config.task_buffer_size or not self._config.cluster:
                await self._flush_child_task_queue(child_id)

        for task_id in unassigned_tasks:
            self.logger.warning(f"Can't route task {task_id} to any child")
            result = Result(
                sender=self.node_id,
                task_id=task_id,
                success=False,
                exception=f"Routing failed at {self.node_id}",
            )
            asyncio.create_task(self._forward_result(result))
        return [task_to_child.get(task.task_id, None) for task in tasks]

    # --------------------------------------------------------------------------
    #                               Callbacks
    # --------------------------------------------------------------------------

    def _create_done_callback(
        self, child_ids: List[str]
    ) -> Optional[Callable[[ConcurrentFuture], None]]:
        """Return a done-callback for the given child futures, or None.

        The base AsyncMaster does not need a callback: _teardown_child awaits
        the future directly (asyncio.wrap_future), and crash detection is
        handled by the _report_status heartbeat loop.  Subclasses that need
        eager notification (e.g. AsyncWorkStealingMaster) override this.
        """
        return None

    # -------------------------------------------------------------------------
    #                               Monitors
    # -------------------------------------------------------------------------

    async def _collect_final_result_batch_from_child(self, child_id: str) -> None:
        """Collect result and final status from a single child."""
        try:
            # Wait for result from child
            result_batch: ResultBatch = await self._comm.recv_message_from_child(
                ResultBatch, child_id=child_id, block=True
            )

            if result_batch is not None:
                self._results[child_id] = [result_batch]
                self.logger.info(f"{self.node_id}: Received result from {child_id}")
                # Update task statuses from the final result batch
                for r in result_batch.data:
                    self._scheduler.set_task_status(
                        r.task_id,
                        TaskStatus.SUCCESS if r.success else TaskStatus.FAILED,
                    )
                await self._comm.send_message_to_child(
                    child_id, ResultAck(sender=self.node_id)
                )
                # Source c (authoritative): mark SUCCESS from RUNNING or RECOVERING.
                state = self._scheduler.get_child_state(child_id)
                if state in {ChildState.RUNNING, ChildState.RECOVERING}:
                    self._scheduler.mark_child_success(child_id)
            else:
                self.logger.warning(
                    f"{self.node_id}: No result received from {child_id}"
                )
                self._results[child_id] = []
            # Final status is collected by the per-child status monitor.
        except Exception as e:
            self.logger.error(
                f"{self.node_id}: Error collecting result from {child_id}: {e}"
            )
            self._results[child_id] = []

    async def _aggregate_and_send_result_batch(self) -> None:
        """Wait for all result collection tasks, aggregate, and send to parent.

        Dynamically tracks self._child_result_batch_task so that tasks added
        during child recovery are also awaited before aggregation proceeds.
        """
        # Wait for all result collection tasks to complete, including any added
        # during recovery (which updates self._child_result_batch_task at runtime).
        seen_tasks: set = set()
        while True:
            current_tasks = set(self._child_result_batch_task.values())
            new_tasks = current_tasks - seen_tasks
            if new_tasks:
                seen_tasks.update(new_tasks)
                await asyncio.gather(*new_tasks, return_exceptions=True)
                # Loop immediately: recovery may have added more tasks while we waited.
                continue
            # No new tasks this iteration — check if we are truly done.
            # _all_children_done_event won't be set while any child is RECOVERING,
            # so no separate guard is needed.
            if self._all_children_done_event.is_set() and all(
                t.done() for t in self._child_result_batch_task.values()
            ):
                break
            # Children still running or recovery in progress; yield and recheck.
            await asyncio.sleep(0.05)
        self.logger.info(f"{self.node_id}: All result collection tasks completed")

        # Wait for all forwarder tasks to finish draining IResultBatch messages.
        # This ensures all streamed results have been forwarded/accumulated before
        # we aggregate the final batch.
        forwarder_tasks = list(self._child_forwarder_task.values())
        if forwarder_tasks:
            await asyncio.gather(*forwarder_tasks, return_exceptions=True)
        self.logger.info(f"{self.node_id}: All forwarder tasks completed")

        # Cancel the reporting task
        self._stop_reporting_event.set()
        self._reporting_task.cancel()
        try:
            await self._reporting_task
        except Exception:
            pass

        self.logger.info(f"{self.node_id}: Stopped reporting loop")

        # Report final status to parent (or write to file if root / parent dead)
        async with self._timer("report_to_parent"):
            if self.parent and (
                self._comm.parent_dead_event is None
                or not self._comm.parent_dead_event.is_set()
            ):
                final_status = self._scheduler.aggregate_status()
                final_status.tag = "final"
                max_retries = 10
                for i in range(max_retries):
                    success = await self._comm.send_message_to_parent(final_status)
                    if not success:
                        self.logger.warning(
                            f"{self.node_id}: Failed to send final status to parent"
                        )
                    else:
                        self.logger.info(
                            f"{self.node_id}: Successfully reported final status to parent"
                        )
                        ack = await self._comm.recv_message_from_parent(
                            ResultAck, timeout=5.0 + i * 5.0
                        )
                        if ack is None:
                            self.logger.warning(
                                "Did not receive ack for final status from parent in 5sec"
                            )
                        else:
                            self.logger.info(
                                "Received ack from parent for final status"
                            )
                            break
            else:
                try:
                    status = self._scheduler.aggregate_status()
                    status.tag = "final"
                    fname = os.path.join(os.getcwd(), f"{self.node_id}_status.json")
                    status.to_file(fname)
                    self.logger.info(
                        f"{self.node_id}: Successfully dumped final status to {fname}"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"{self.node_id}: Reporting final status failed with exception {e}"
                    )

        # Aggregate results
        async with self._timer("aggregate_results"):
            if len(self._scheduler.unassigned_task_ids) > 0:
                failed_results = [
                    Result(
                        sender=self.node_id,
                        task_id=task_id,
                        success=False,
                        exception=f"Failed to assign to children at {self.node_id}",
                    )
                    for task_id in self._scheduler.unassigned_task_ids
                ]
            else:
                failed_results = []
            result_batch = ResultBatch(sender=self.node_id, data=failed_results)
            # Include results streamed during execution (batch mode root master only)
            for r in self._batch_streaming_results:
                result_batch.add_result(r)
            for child_id, child_results in self._results.items():
                for rb in child_results:
                    result_batch += rb
            self.logger.info(
                f"{self.node_id}: Aggregated results from {len(self._results)} children"
            )

        # Send to parent with ACK and retries (skip if parent is dead)
        max_retries = 10
        if self.parent and (
            self._comm.parent_dead_event is None
            or not self._comm.parent_dead_event.is_set()
        ):
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
        elif self.parent:
            self.logger.warning(f"{self.node_id}: Parent is dead, skipping result send")

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

    async def _report_status(self) -> None:
        """Periodically aggregate child status and forward to parent."""
        while not self._stop_reporting_event.is_set():
            try:
                status = self._scheduler.aggregate_status()
                if self.parent:
                    await self._comm.send_message_to_parent(status)
                    self.logger.info(status)
                else:
                    self.logger.info(status)

                # Periodic checkpoint: scheduler state, comm state, and collected results.
                if self._checkpointer is not None:
                    asyncio.create_task(self._write_checkpoint())
                # Use wait with timeout so we can exit quickly when stopped
                try:
                    jitter = random.uniform(-0.05, 0.05) * self._config.report_interval
                    await asyncio.wait_for(
                        self._stop_reporting_event.wait(),
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

    async def _client_request_monitor(self) -> None:
        """Cluster mode: handle messages from any ClusterClient connected to this master."""
        while not self._all_children_done_event.is_set():
            item = await self._comm.recv_client_message()
            if item is None:
                continue
            client_id, msg = item
            if isinstance(msg, TaskUpdate):
                self._total_received += len(msg.added_tasks)
                self.logger.info(
                    f"Received TaskUpdate from client. Total tasks: {self._total_received}"
                )
                await self._route_tasks(msg.added_tasks, client_id=client_id)

    async def _flush_dest_queue(self, dest_id: str) -> None:
        """Send all buffered results for dest_id to the parent master as a single IResultBatch."""
        result_q = self._iresult_q.get(dest_id)
        if not result_q:
            return
        results = []
        while result_q:
            results.append(result_q.popleft())
        await self._comm.send_message_to_parent(
            IResultBatch(sender=self.node_id, data=results)
        )

    async def _periodic_flush_loop(self) -> None:
        """Flush buffered parent-bound results on a fixed interval with jitter.

        Only runs on masters that have a parent (i.e. there are results to forward upward).
        """
        while True:
            jitter = random.uniform(-0.05, 0.05) * self._config.result_flush_interval
            await asyncio.sleep(self._config.result_flush_interval + jitter)
            for dest_id in list(self._iresult_q.keys()):
                await self._flush_dest_queue(dest_id)

    async def _flush_child_task_queue(self, child_id: str) -> None:
        """Send all buffered tasks for child_id to that child as a single TaskUpdate."""
        task_q = self._itask_q.get(child_id)
        if not task_q:
            return
        tasks = []
        while task_q:
            tasks.append(task_q.popleft())
        await self._comm.send_message_to_child(
            child_id,
            TaskUpdate(sender=self.node_id, added_tasks=tasks),
        )
        self.logger.info(f"Sent TaskUpdate of size {len(tasks)} to {child_id}")

    async def _periodic_task_flush_loop(self) -> None:
        """Flush buffered child-bound tasks on a fixed interval with jitter.

        Runs on all masters that have children (i.e. there are tasks to dispatch downward).
        """
        while True:
            jitter = random.uniform(-0.05, 0.05) * self._config.task_flush_interval
            await asyncio.sleep(self._config.task_flush_interval + jitter)
            for child_id in list(self._itask_q.keys()):
                await self._flush_child_task_queue(child_id)

    async def _forward_result(self, result: Union[Result, IResultBatch]) -> None:
        """Route a result to its destination.

        - Client-bound results (root master): forwarded directly as IResultBatch.
        - Parent-bound results (intermediate master): buffered and flushed periodically
          to aggregate fan-in across children before sending up the hierarchy.
        """
        items = result.data if isinstance(result, IResultBatch) else [result]

        for single_result in items:
            self._scheduler.set_task_status(
                single_result.task_id,
                TaskStatus.SUCCESS if single_result.success else TaskStatus.FAILED,
            )

        dest_results: Dict[str, List[Result]] = {}
        for single_result in items:
            dest_id = self._scheduler.get_task_client(single_result.task_id) or (
                self.parent.node_id if self.parent else None
            )
            if dest_id is None:
                if not self._config.cluster:
                    # Batch mode root master: accumulate locally
                    self._batch_streaming_results.append(single_result)
                else:
                    self.logger.warning(
                        f"Can't find a destination for task {single_result.task_id}"
                    )
                continue
            dest_results.setdefault(dest_id, []).append(single_result)

        for dest_id, results in dest_results.items():
            if dest_id.startswith("client:"):
                # Client attached at this level: send directly, no buffering needed
                self._total_streamed += len(results)
                await self._comm.send_message_to_child(
                    dest_id, IResultBatch(sender=self.node_id, data=results)
                )
            else:
                # Parent-bound result: buffer to aggregate fan-in before sending up
                if dest_id not in self._iresult_q:
                    self._iresult_q[dest_id] = deque()
                result_q = self._iresult_q[dest_id]
                for r in results:
                    result_q.append(r)
                if len(result_q) >= self._config.result_buffer_size:
                    await self._flush_dest_queue(dest_id)

    async def _child_result_monitor(self, child_id: str) -> None:
        """Receive IResultBatch messages from child and forward to client or parent."""
        child_done = self._scheduler.get_done_event(child_id)
        done_task = asyncio.create_task(child_done.wait())
        while True:
            recv_task = asyncio.create_task(
                self._comm.recv_message_from_child(
                    IResultBatch, child_id=child_id, block=True
                )
            )
            done, _ = await asyncio.wait(
                {recv_task, done_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if recv_task in done:
                msg = recv_task.result()
                if msg is not None:
                    await self._forward_result(msg)
            else:
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass
            if done_task in done:
                break
        # Drain remaining results after child exits
        while True:
            msg = await self._comm.recv_message_from_child(
                IResultBatch, child_id=child_id
            )
            if msg is None:
                break
            await self._forward_result(msg)
        # Flush any parent-bound results that didn't reach the buffer threshold.
        for dest_id in list(self._iresult_q.keys()):
            await self._flush_dest_queue(dest_id)

    async def _child_status_monitor(self, child_id: str) -> None:
        """Continuously collect status messages from a single child and update scheduler."""
        child_done = self._scheduler.get_done_event(child_id)
        try:
            while not child_done.is_set():
                status = await self._comm.recv_message_from_child(
                    Status, child_id=child_id, block=True
                )
                if status is not None:
                    self._scheduler.set_child_status(child_id, status)
                    if status.tag == "final":
                        self.logger.info(
                            f"Received final status from {child_id}. Sending ACK"
                        )
                        await self._comm.send_message_to_child(child_id, ResultAck())
                        return
            # Drain any remaining status messages after child process exits
            while True:
                status = await self._comm.recv_message_from_child(
                    Status, child_id=child_id
                )
                if status is None:
                    break
                self._scheduler.set_child_status(child_id, status)
        except asyncio.CancelledError:
            pass

    async def _child_liveness_monitor(self, child_id: str) -> None:
        """Wait for child dead event and trigger recovery."""
        while True:
            try:
                ev = self._comm._child_dead_events.get(child_id)
                await ev.wait()
                state = self._scheduler.get_child_state(child_id)
                if state is None or self._scheduler.is_child_terminal(child_id):
                    return
                self.logger.warning(
                    f"{self.node_id}: Child {child_id} declared dead by HB process"
                )
                if self._config.restart_children_on_failure:
                    self._scheduler.mark_child_recovering(child_id)
                    max_retries = 3
                    success = False
                    for i in range(max_retries):
                        success = await self._recover_dead_child(child_id)
                        if success:
                            self.logger.info(
                                f"Child recovered successfully after {i}/{max_retries} retries."
                            )
                            # Signal the recovered child that the master is ready.
                            await self._comm.send_message_to_child(
                                child_id, Ready(sender=self.node_id)
                            )
                            self.logger.info(
                                f"{self.node_id}: Sent Ready to recovered child {child_id}"
                            )
                            break
                    if not success:
                        self._scheduler.mark_child_failed(child_id)
                        break
                else:
                    self._scheduler.mark_child_failed(child_id)
                    break
            except Exception as e:
                self.logger.warning(f"liveness monitor failed with Exception {e}")
                await asyncio.sleep(10.0)

    async def _recover_dead_child(self, child_id: str) -> bool:
        """Tear down a dead child and relaunch on the same resources."""
        try:
            self.logger.info(f"{self.node_id}: Recovering dead child {child_id}")

            # Capture assignment BEFORE teardown removes it from the scheduler
            assignment = self._scheduler.get_child_assignment(child_id)

            # cancels the futures, tasks and removes all events from scheduler and comm
            await self._teardown_child(child_id)

            # Re-register child: Allocate the resources in the cluster. NOTREADY, then immediately READY so _launch_child
            # can transition READY → RUNNING.
            allocated, _ = self._scheduler.cluster.allocate(assignment["job_resource"])
            if not allocated:
                self.logger.error(
                    f"Can't allocate the identical child resources while recovering"
                )
                raise RuntimeError(
                    "Can't allocate the identical child resources while recovering"
                )
            else:
                self.logger.info(
                    f"Reallocated resources for {child_id}: {assignment['job_resource']}"
                )
            self._scheduler.register_child(child_id, assignment)  # → NOTREADY
            self._scheduler.mark_child_ready(child_id)  # NOTREADY → READY

            # Instantiate a fresh child object — never reuse the old one since it
            # carries stale asyncio.Event / comm state from the previous run, which
            # causes pickling failures or immediate early-exit in the subprocess.
            fresh_children = self._instantiate_children(
                include_tasks=False, target_ids={child_id}
            )
            fresh_child_obj = fresh_children[child_id]

            await self._init_child(child_id, fresh_child_obj)

            self.logger.info(f"Completed init child {child_id}")

            results = await self._launch_and_sync_children([child_id])
            result = results[0]
            if result is not None:
                self.logger.error(
                    f"{self.node_id}: Failed to recover child {child_id}: {result}"
                )
                return False
            else:
                self.logger.info(f"Completed launching and sync with child {child_id}")
                task_ids = self._scheduler.child_assignments[child_id]["task_ids"]
                for task_id in task_ids:
                    self._scheduler.discard_unassigned(task_id)

            return True
        except Exception as e:
            self.logger.error(f"{self.node_id}: Recovery failed for {child_id}: {e}")
            return False

    async def _parent_task_update_monitor(self) -> None:
        """Non-root master only: receive TaskUpdates from parent and route tasks to children."""
        while not self._stop_task_update.is_set():
            try:
                task_update = await self._comm.recv_message_from_parent(
                    TaskUpdate, block=True
                )
                if task_update is not None:
                    await self._route_tasks(task_update.added_tasks)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Task update monitor error: {e}")
                await asyncio.sleep(0.1)

    # -------------------------------------------------------------------------
    #                               Teardown
    # -------------------------------------------------------------------------

    async def _teardown_child(self, child_id: str) -> None:
        """Cancel per-child tasks, remove the child from all bookkeeping, and prune
        the comm cache so no stale entries remain."""
        # Cancel the future and wait for the scheduler to mark the child done.
        child_fut = self._children_futures.get(child_id, None)
        if child_fut is None:
            self.logger.warning(
                f"_teardown_child: no future found for {child_id}, skipping cancel"
            )
        else:
            if not child_fut.done():
                success = child_fut.cancel()
                if success:
                    self.logger.info(f"Cancelling {child_id} future successful")
                else:
                    self.logger.warning(
                        f"Cancelling {child_id} future was unsuccessful"
                    )

        # Cancel per-child monitor tasks and await them to avoid dangling coroutines.
        tasks_to_cancel = []
        task_dicts = [
            self._child_result_batch_task,
            self._child_status_task,
            self._child_forwarder_task,
            self._child_task_request_task,
        ]
        if not self._scheduler.is_child_recovering(child_id):
            task_dicts.append(self._child_liveness_task)
        for task_dict in task_dicts:
            t = task_dict.pop(child_id, None)
            if t is not None:
                t.cancel()
                tasks_to_cancel.append(t)
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        self._scheduler.remove_child(child_id)
        self.remove_child(child_id)
        self._child_objs.pop(child_id, None)
        self._children_futures.pop(child_id, None)
        self._routed_task_ids.pop(child_id, None)
        self._iresult_q.pop(child_id, None)
        self._itask_q.pop(child_id, None)
        await self._comm.update_node_info(self.info())

    async def stop(self) -> None:
        """Gracefully shut down the master in a fixed teardown order.

        1. Stop global monitor tasks: status reporter, cluster client monitor
           (if cluster mode), and parent task-update monitor (if non-root).
        2. Tear down each child — cancels the child's future, waits for the
           scheduler to mark it done, and cancels its per-child result/status
           monitor tasks.
        3. Wait (up to 30 s) for ``_all_children_done_event`` to confirm every
           child process has exited.
        4. Close the comm layer and shut down the executor.
        5. Export Perfetto profiling traces if profiling was enabled.
        """

        if (
            self._parent_ready_monitor_task
            and not self._parent_ready_monitor_task.done()
        ):
            self._parent_ready_monitor_task.cancel()
            try:
                await self._parent_ready_monitor_task
            except asyncio.CancelledError:
                pass

        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass

        if self._client_monitor_task and not self._client_monitor_task.done():
            self._client_monitor_task.cancel()
            try:
                await self._client_monitor_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Stopped client monitor task")

        if self._task_update_task:
            self._stop_task_update.set()
            self._task_update_task.cancel()
            try:
                await self._task_update_task
            except asyncio.CancelledError:
                pass
            self.logger.info("Stopped parent task update monitor")

        # Tear down children and wait for them to be done
        for child_name in self._scheduler.children_names:
            await self._teardown_child(child_name)

        # Wait for all children to complete with timeout
        try:
            await asyncio.wait_for(self._all_children_done_event.wait(), timeout=30.0)
            self.logger.info(f"{self.node_id}: All children have completed execution")
        except asyncio.TimeoutError:
            self.logger.warning(
                f"{self.node_id}: Timeout waiting for children to complete"
            )

        # write a final checkpoint
        if self._checkpointer is not None:
            await self._write_checkpoint()
            self.logger.info(f"{self.node_id}: Final checkpoint written")

        # stop comm and executor
        await self._comm.close()
        self.logger.info(f"Shutting down executor")
        self._executor.shutdown()
        self.logger.info(f"Done Shutting down executor")

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
        self.logger.info(f"Done stopping {self.node_id}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # -------------------------------------------------------------------------
    #                               Entry point
    # -------------------------------------------------------------------------

    async def _wait_for_finish(self) -> None:
        """Wait for all work to complete, then trigger result aggregation.

        Races conditions with asyncio.FIRST_COMPLETED:
        1. Work done — all children have returned their final ResultBatch.
        2. Parent dead locally — consecutive send failures exceeded threshold.
        3. SIGTERM on this process.
        4. Stop(TERMINATE/KILL) received from parent (non-root only).

        On parent-dead exit, forwards Stop(KILL) to all children. On any other
        non-work-done exit, forwards Stop(TERMINATE) cascading the shutdown down
        the subtree. In both cases skips sending results/status to the dead parent.
        """
        import sys

        stop_tasks: Dict[str, asyncio.Task] = {}
        received_stop_type: List[Optional[StopType]] = [None]

        stop_tasks["work_done"] = asyncio.create_task(
            self._all_children_done_event.wait()
        )
        if self._comm.parent_dead_event is not None:
            stop_tasks["parent_dead"] = asyncio.create_task(
                self._comm.parent_dead_event.wait()
            )
        stop_tasks["stop_signal"] = asyncio.create_task(
            self._stop_signal_received.wait()
        )

        if self.parent is not None:

            async def _recv_stop_from_parent():
                msg = await self._comm.recv_message_from_parent(Stop, block=True)
                if msg is not None:
                    received_stop_type[0] = msg.type

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

        # Propagate stop to children if work didn't finish naturally.
        # Parent dead → KILL children. Received stop from parent → mirror its type.
        # Any other reason (SIGTERM etc.) → TERMINATE cleanly.
        stop_type: Optional[StopType] = None
        if not self._all_children_done_event.is_set():
            if (
                self._comm.parent_dead_event is not None
                and self._comm.parent_dead_event.is_set()
            ):
                stop_type = StopType.KILL
            elif received_stop_type[0] is not None:
                stop_type = received_stop_type[0]
            else:
                stop_type = StopType.TERMINATE
            self.logger.info(
                f"{self.node_id}: Propagating {stop_type.value} to children"
            )
            for child_id in list(self.children.keys()):
                try:
                    await self._comm.send_message_to_child(
                        child_id, Stop(sender=self.node_id, type=stop_type)
                    )
                except Exception:
                    pass

        # If we KILLed children they won't send results — cancel all result tasks.
        if not self._all_children_done_event.is_set() and stop_type == StopType.KILL:
            for t in list(self._child_result_batch_task.values()):
                t.cancel()
            await asyncio.gather(
                *self._child_result_batch_task.values(),
                return_exceptions=True,
            )
            self._aggregate_task.cancel()
            try:
                await self._aggregate_task
            except (asyncio.CancelledError, Exception):
                pass
        else:
            await self._aggregate_task

        # Force-exit after propagating if we received a KILL from our parent.
        if received_stop_type[0] == StopType.KILL:
            await self.stop()
            sys.exit(1)

    async def run(self) -> ResultBatch:
        """Main entry point: initialise, wait for all work to finish, stop, return results."""
        try:
            async with self._timer("init"):
                await self._lazy_init()

            # Wait for aggregation to complete
            await self._wait_for_finish()
        finally:
            await self.stop()

        # Return aggregated results
        result_batch = ResultBatch(sender=self.node_id)
        for r in self._batch_streaming_results:
            result_batch.add_result(r)
        for child_results in self._results.values():
            for rb in child_results:
                result_batch += rb
        return result_batch

    def create_an_event_loop(self) -> None:
        """Entry point for a new child process: run the async event loop."""
        asyncio.run(self.run())

    # -------------------------------------------------------------------------
    #                       Serialization and Deserialization
    # -------------------------------------------------------------------------

    def asdict(self, include_tasks: bool = False) -> dict:
        """Serialise this master to a JSON-compatible dict for cross-process transfer."""
        obj_dict = {
            "type": self.type,
            "node_id": self.node_id,
            "config": self._config.model_dump_json(),
            "parent": self.parent.serialize() if self.parent else None,
            "children": {
                child_id: child.serialize() for child_id, child in self.children.items()
            },
            "parent_comm": self.parent_comm.get_state().serialize()
            if self.parent_comm is not None
            else None,
        }

        if include_tasks:
            raise NotImplementedError(
                "Including tasks in serialization is not implemented yet."
            )

        obj_dict["secret_id"] = self._secret_id
        return obj_dict

    @classmethod
    def fromdict(cls, data: dict) -> "AsyncMaster":
        """Reconstruct an AsyncMaster from a serialised dict (inverse of asdict)."""
        config = LauncherConfig.model_validate_json(data["config"])
        parent = NodeInfo.deserialize(data["parent"]) if data["parent"] else None
        children = {
            child_id: NodeInfo.deserialize(child_json)
            for child_id, child_json in data["children"].items()
        }

        if config.comm_name == "async_zmq":
            # AsyncZMQComm might need special handling due to non-picklable attributes
            parent_comm = (
                AsyncZMQComm.set_state(
                    AsyncZMQCommState.deserialize(data["parent_comm"])
                )
                if data["parent_comm"]
                else None
            )
        else:
            raise ValueError(f"Unsupported comm type {config.comm_name}")

        master = cls(
            id=data["node_id"],
            config=config,
            Nodes=None,  # Nodes will be received via NodeUpdate message
            tasks={},  # Tasks are not included in serialization
            parent=parent,
            children=children,
            parent_comm=parent_comm,
        )
        master._secret_id = data["secret_id"]
        return master
