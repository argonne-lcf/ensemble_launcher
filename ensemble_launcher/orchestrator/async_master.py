import asyncio
import functools
import json
import os
import random
import signal
import socket
import time
import uuid
from concurrent.futures import Future as ConcurrentFuture
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from ensemble_launcher.checkpointing import Checkpointer
from ensemble_launcher.comm import (
    AsyncComm,
    ClientConnection,
    NodeInfo,
    transport_registry,
)
from ensemble_launcher.comm.messages import (
    Done,
    IResultBatch,
    NodeUpdate,
    Ready,
    Result,
    ResultBatch,
    Status,
    Stop,
    StopType,
    TaskUpdate,
)
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.executors import Executor, executor_registry
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.orchestrator.policy_endpoint import KIND_CHILDREN, PolicyEndpoint
from ensemble_launcher.scheduler import AsyncChildrenScheduler, Child, SchedulerState
from ensemble_launcher.scheduler.child_state import ChildState
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from .async_worker import AsyncWorker
from .node import Node
from ensemble_launcher.taskstore import TaskStore

from .utils import (
    as_task_update,
    async_load_str_file,
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
    # Push: tasks are placed as they arrive. AsyncWorkStealingMaster pulls instead.
    _placement = AsyncChildrenScheduler.PUSH

    def __init__(
        self,
        id: str,
        config: LauncherConfig,
        Nodes: Optional[JobResource] = None,
        tasks: Optional[Dict[str, Task]] = None,
        parent: Optional[NodeInfo] = None,
        children: Optional[Dict[str, NodeInfo]] = None,
        parent_data_conn: Optional[ClientConnection] = None,
        parent_hb_conn: Optional[ClientConnection] = None,
    ):
        super().__init__(id, parent=parent, children=children)
        # Initial tasks as one TaskUpdate (a dict/list of Tasks, e.g. from EnsembleLauncher,
        # is encoded once here). Only the root is built with tasks: they become its first
        # `scheduler.queue()`. Every other master gets its tasks pushed by its parent.
        self._init_tasks: Optional[TaskUpdate] = as_task_update(tasks, sender=id)
        self._init_nodes = Nodes
        self._config = config
        self._parent_data_conn = parent_data_conn
        self._parent_hb_conn = parent_hb_conn

        ##lazily created in run
        self._executor = None
        self._comm = None

        self._scheduler = None

        ##maps
        self._children_futures: Dict[str, Union[AsyncFuture, ConcurrentFuture]] = {}
        self._children_results: Dict[str, Result] = {}
        self._child_objs: Dict[str, Node] = {}

        self.logger = None


        # asyncio event
        # _all_children_done_event is owned by the scheduler after _lazy_init;
        # use __fallback_done_event before the scheduler is created.
        self.__fallback_done_event = asyncio.Event()
        self._stop_reporting_event = asyncio.Event()
        self._event_loop = None  # Will be set in run()

        # Per-child tasks: waits for the child's Done, and receives its streamed results.
        self._child_done_task: Dict[str, asyncio.Task] = {}  # child_id -> Done wait
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

        # Externally tunable children-policy state (see policy_endpoint.py).
        self._policy_endpoint: Optional[PolicyEndpoint] = None
        self._policy_version_written: int = -1

        # Cluster mode state
        self._stop_task_update = asyncio.Event()
        self._task_update_task: Optional[asyncio.Task] = None
        self._client_monitor_task: Optional[asyncio.Task] = None
        self._reporting_task: asyncio.Task = None
        self._stop_signal_received = asyncio.Event()

        # Per-child task request monitor tasks
        self._child_task_request_task: Dict[str, asyncio.Task] = {}

        # Per-child task sender (consumes the child's outbox) and its send window.
        self._child_sender_task: Dict[str, asyncio.Task] = {}
        self._task_window: Dict[str, asyncio.Semaphore] = {}
        # Fails tasks no child could take (scheduler.unplaced).
        self._unplaced_task: Optional[asyncio.Task] = None
        # Set once the initial tasks (if any) are queued; wait_drained() waits for it.
        self._initial_queued = asyncio.Event()

        # Result senders, one per destination ("parent" or a client id): each consumes a
        # scheduler result outbox, with its own send window and in-flight sends.
        self._result_sender_task: Dict[str, asyncio.Task] = {}
        self._result_window: Dict[str, asyncio.Semaphore] = {}
        self._result_outbox: Dict[str, asyncio.Queue] = {}  # dest -> the outbox it sends
        self._result_flush_inflight: Dict[str, Set[asyncio.Task]] = {}
        # Messages a sender took and that are not delivered yet, with the number of outbox
        # items each holds (they are marked done on delivery, so outbox.join() means "sent").
        self._result_unacked: Dict[str, List[Tuple[IResultBatch, int]]] = {}
        # Starts a result sender for each client the scheduler announces.
        self._new_client_task: Optional[asyncio.Task] = None
        # TaskUpdate sends in flight per child (capped by its _task_window).
        self._task_flush_inflight: Dict[str, Set[asyncio.Task]] = {}
        # Periodic checkpointing, on its own cadence rather than the status loop's. The
        # in-flight handle keeps a slow write from overlapping the next one.
        self._checkpoint_task: Optional[asyncio.Task] = None
        self._checkpoint_write: Optional[asyncio.Task] = None
        # Parent-ready handshake: set when parent sends a Ready message
        self._parent_ready_event = asyncio.Event()
        self._parent_ready_monitor_task: Optional[asyncio.Task] = None

        self._total_received: int = 0
        self._total_streamed = 0

    def _child_state(self, child_id: str) -> Optional[ChildState]:
        """The child's lifecycle state, or None once it has been removed."""
        child = self._scheduler.state.children.get(child_id)
        return child.state if child is not None else None

    def _set_child_status(self, child_id: str, status: Status) -> None:
        child = self._scheduler.state.children.get(child_id)
        if child is not None:
            child.status = status

    @property
    def _all_children_done_event(self) -> asyncio.Event:
        """Delegate to scheduler's event once available; use fallback before _lazy_init."""
        if self._scheduler is not None:
            return self._scheduler.state.all_done
        return self.__fallback_done_event

    @property
    def nodes(self) -> JobResource:
        """Node resource allocation owned by the scheduler cluster."""
        return self._scheduler.cluster.nodes

    @nodes.setter
    def nodes(self, value: JobResource) -> None:
        self._scheduler.cluster.update_nodes(value)

    @property
    def parent_data_conn(self):
        """Connection to the parent node, or None if this is the root."""
        return self._parent_data_conn

    @parent_data_conn.setter
    def parent_data_conn(self, value) -> None:
        self._parent_data_conn = value

    @property
    def parent_hb_conn(self):
        """Connection to the parent node, or None if this is the root."""
        return self._parent_hb_conn

    @parent_hb_conn.setter
    def parent_hb_conn(self, value) -> None:
        self._parent_hb_conn = value

    @property
    def comm(self) -> Optional[AsyncComm]:
        """Communication channel for this master (connecting to parent and children)."""
        return self._comm

    @property
    def tasks(self) -> TaskStore:
        """All tasks this master holds, keyed by uid."""
        return self._scheduler.tasks

    @property
    def init_nodes(self) -> JobResource:
        return self._init_nodes

    @property
    def init_tasks(self) -> Optional[TaskUpdate]:
        return self._init_tasks

    # -----------------------------------------------------------------
    #                       Initialization
    # -----------------------------------------------------------------

    def _setup_logger(self) -> None:
        """Configure the logger for this master, optionally writing to a per-node log file."""
        self.logger = setup_logger(
            __name__, self.node_id, level=self._config.log_level,
            log_to_file=self._config.master_logs,
        )

    def _create_comm(self) -> None:
        """Instantiate the communication backend."""
        self.logger.info(f"{self.node_id}: Starting comm init")
        self._comm = AsyncComm(
            self.logger.getChild("comm"),
            self.info(),
            parent_conn=self.parent_data_conn,
            hb_parent_conn=self.parent_hb_conn,
            heartbeat_interval=self._config.heartbeat_interval,
            heartbeat_dead_threshold=self._config.heartbeat_dead_threshold,
            cluster_secret=self._config.cluster_secret,
            req_res=self._config.req_res,
            send_retries=self._config.send_retries,
            send_timeout=self._config.send_timeout,
            io_process=self._config.comm_io_process,
        )
        self.logger.info(f"{self.node_id}: Done with comm init")

    def _create_scheduler(self) -> AsyncChildrenScheduler:
        """Instantiate the worker scheduler.  Override to change init args."""
        return AsyncChildrenScheduler(
            self.logger.getChild("scheduler"),
            self._init_nodes,
            self._config,
            node_id=self.node_id,
            level=self.level,
            placement=self._placement,
        )

    def _get_child_class(
        self, child_assignment: Optional[Child] = None
    ) -> type:
        """Return the Node class to use for children at the next level."""
        if child_assignment is not None and child_assignment.child_class is not None:
            child_class = child_assignment.child_class
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

    def _instantiate_children(self, target_ids: set) -> Dict[str, Node]:
        """Instantiate Node objects for target_ids. Children are built without tasks: their
        senders push the tasks once each child is running."""
        NodeClass = self._get_child_class()
        children = {}
        for child_id in target_ids:
            alloc = self._scheduler.state.children[child_id]
            child_config = self._config
            if alloc.task_executor_name is not None:
                child_config = self._config.model_copy(
                    update={"task_executor_name": alloc.task_executor_name}
                )
                self.logger.info(
                    f"{self.node_id}: Child {child_id} using task_executor_name: {alloc.task_executor_name}"
                )
            children[child_id] = NodeClass(
                child_id,
                config=child_config,
                Nodes=alloc.job_resource,
                tasks=None,
                parent=None,
            )
        return children

    def _apply_resource_headroom(self) -> None:
        """Reserve one CPU on the first child's head node for this master process."""
        first_job_resource = next(iter(self._scheduler.state.children.values())).job_resource
        first_node = first_job_resource.resources[0]
        if isinstance(first_node, NodeResourceList):
            first_job_resource.resources[0] = first_node.with_cpus(first_node.cpus[1:])
        else:
            first_job_resource.resources[0] = NodeResourceCount(
                ncpus=first_node.cpu_count - 1, ngpus=first_node.gpu_count
            )

    def _create_children(
        self,
        partial: bool = False,
        nodes: Optional[JobResource] = None,
    ) -> Dict[str, Node]:
        """Lay out children via the scheduler and instantiate their Node objects.

        Placement is not done here: queued tasks are placed by the scheduler's assigner.
        partial=True: additive — preserves running children, offsets new wids, and
            returns only newly created children.
        nodes: restrict the layout to these nodes (e.g. recovered nodes on retry).
        """
        existing_ids = set(self._scheduler.state.children) if partial else set()
        self._scheduler.create_children(
            self.level, self.node_id, reset=not partial, nodes=nodes
        )
        if not partial and not self._config.overload_orchestrator_core:
            self._apply_resource_headroom()
        target_ids = set(self._scheduler.state.children) - existing_ids
        return self._instantiate_children(target_ids)

    async def _init_child(self, child_id: str, child: Node) -> None:
        """Register a single child and start its per-child monitoring tasks.

        Initialises the comm cache for this child (via update_node_info) so that
        the collect / forwarder tasks can safely block on the message queue from
        the moment they are created.
        """
        self._child_objs[child_id] = child
        self.add_child(child_id, child.info())
        child.set_parent(self.info())
        self._comm.update_node_info(self.info())
        child_data_conn, child_hb_conn = await self._comm.create_child_pipe(
            child_id=child.info().node_id, child_secret_id=child.info().secret_id
        )
        child.parent_data_conn = child_data_conn
        child.parent_hb_conn = child_hb_conn
        # Extend the comm's node-info so its cache gains an entry for this child.

        # Per-child Done wait: the child is finished once every result it sent was delivered.
        self._child_done_task[child_id] = asyncio.create_task(
            self._wait_for_child_done(child_id)
        )
        # Per-child status monitor (continuously drains Status messages).
        self._child_status_task[child_id] = asyncio.create_task(
            self._child_status_monitor(child_id)
        )
        # Per-child liveness monitor (waits for HB process dead event).
        self._child_liveness_task[child_id] = asyncio.create_task(
            self._child_liveness_monitor(child_id)
        )
        self._child_forwarder_task[child_id] = asyncio.create_task(
            self._child_result_monitor(child_id)
        )
        # Per-child task sender: sends the child's outbox once it is running.
        self._task_window[child_id] = asyncio.Semaphore(
            max(1, self._config.max_inflight_task_sends)
        )
        self._child_sender_task[child_id] = asyncio.create_task(
            self._child_sender(child_id)
        )

    def _create_monitor_tasks(self) -> None:
        """Start long-running asyncio monitor tasks (status reporter, cluster monitors).

        Per-child collect/forwarder tasks are started in _init_child.
        The aggregate task is created at the end of _lazy_init once all
        children (including any recreated ones) have been registered.
        """
        self._reporting_task = asyncio.create_task(self._report_status())

        if self._checkpointer is not None:
            self._checkpoint_task = asyncio.create_task(
                self._periodic_checkpoint_loop()
            )

        self._unplaced_task = asyncio.create_task(self._unplaced_monitor())
        if self.parent:
            # Every sub-master listens for tasks from its parent, in every mode.
            self._task_update_task = asyncio.create_task(
                self._parent_task_update_monitor()
            )
        if self._config.cluster:
            self._client_monitor_task = asyncio.create_task(
                self._client_request_monitor()
            )
        # Results go out as they arrive: one sender for the parent (the root keeps its
        # parent outbox as the run's results), and one per client as clients appear.
        if self.parent:
            self._start_result_sender("parent", self._scheduler.parent_outbox)
        self._new_client_task = asyncio.create_task(self._new_client_monitor())

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
        sorted_cpus = sorted(
            child_resource.cpus
            if isinstance(child_resource, NodeResourceList)
            else list(range(child_resource.cpu_count))
        )

        cpus = f"{sorted_cpus[0]}-{sorted_cpus[-1]}"

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
            self.logger.info(f"head node: {head_node}")

            # Serialize child object
            child_dict = child_obj.asdict()
            json_str = json.dumps(child_dict, default=str)

            # Write JSON to a temp file to avoid ARG_MAX limits on large payloads
            json_fname = os.path.join(
                self._executor.tmp_dir, f"child_{uuid.uuid4()}.json"
            )
            if hasattr(self._executor, "write_file_to_nodes"):
                success = await self._executor.write_file_to_nodes(
                    json_fname, json_str, [head_node]
                )
                if not success:
                    self.logger.error("Copying file to nodes failed!!")
                    raise RuntimeError
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
                common_keys = [
                    "type",
                    "parent",
                ]  # , "parent_data_conn", "parent_hb_conn"]
                if all(
                    [
                        cdict.task_executor_name is None
                        for cdict in self._scheduler.state.children.values()
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
                    success = await self._executor.write_file_to_nodes(
                        json_fname, json_str, child_head_nodes
                    )
                    if not success:
                        self.logger.error("Copying file to nodes failed!!")
                        raise RuntimeError
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
                self.logger.info("Launching children in serial")
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
        except Exception as e:
            self.logger.warning(f"get affinity failed with exception: {e}")

        # Read checkpoint data early so both comm address and scheduler state
        # can be restored from the cached data at the appropriate points below.
        await self._read_checkpoint_data()

        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm()  ###This will only create picklable objects

        # Restore saved comm state before binding so children can reconnect.
        ## Removed this as we are restarting the whole subtree
        # await self._restore_comm_state()

        # Start parent comm end point monitor
        await self._comm.start_monitors()

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

        # The root's initial tasks are just its first update. A sub-master is built without
        # tasks; its parent pushes them -- unless its children policy needs tasks to lay out
        # children (bin packing), in which case it takes its first update now.
        if self._init_tasks is not None:
            await self._scheduler.queue(self._init_tasks)
            self._init_tasks = None  # the store owns the bytes now
        elif self.parent and self._scheduler.policy.layout_needs_tasks:
            await self._receive_first_task_update()
        self._initial_queued.set()
        self.logger.info(f"{self.node_id}: Have {len(self.tasks)} tasks queued")

        # Restore the children (and the tasks each held) from cached checkpoint data. Must
        # come after the initial tasks are queued: each restored child claims its tasks from
        # the pool, so the assigner only places what no child held.
        ckpt_restored = self._restore_scheduler_checkpoint()

        await self._create_policy_endpoint()

        # Check executor validity
        assert self._config.child_executor_name in executor_registry.async_executors, (
            f"Executor {self._config.child_executor_name} not found in async executors {executor_registry.async_executors}"
        )

        kwargs = {}
        kwargs["logger"] = self.logger.getChild("executor")
        kwargs["max_workers"] = self.nodes.resources[0].cpu_count
        if self._config.child_executor_name == "async_mpi":
            kwargs["mpi_config"] = self._config.mpi_config

        # Create executor. The child executor launches sub-masters and workers rather than
        # tasks, but it is built from the same registry, so it takes its configuration from
        # the same place -- `executor_configs` keyed by the executor's name.
        self._executor: Executor = executor_registry.create_executor(
            self._config.child_executor_name,
            kwargs=self._config.executor_kwargs(
                self._config.child_executor_name, **kwargs
            ),
        )
        self.logger.info(f"Created {self._config.child_executor_name} executor")

        # Create children: use restored assignment topology if available, else run policy.
        if ckpt_restored:
            children = self._instantiate_children(set(self._scheduler.state.children))
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

        # Children exist: start placing queued tasks. Each child's sender sends its share
        # once the child is running.
        self._scheduler.start()

        # Launch and sync children, retrying failures up to 2 times
        children_names = list(self._scheduler.state.children)
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
                f"{max_retries} retries; giving up on them"
            )
            for name in failed_names:
                self._give_up_on_child(name)

        # Start global monitors: result senders, unplaced tasks, client/parent task updates,
        # status reporting and checkpointing.
        self._create_monitor_tasks()

        # All children are launched and synced — signal each one that the master
        # is ready so they can send Done when they finish.
        for child_id in list(self._child_done_task.keys()):
            await self._comm.send_message_to_child(
                child_id, Ready(sender=self.node_id), retries=self._config.send_retries
            )
            self.logger.info(f"{self.node_id}: Sent Ready to {child_id}")

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
            self.logger,
            cluster_secret=self._config.cluster_secret,
        )
        if not self._checkpointer.checkpoint_exists():
            return
        self._ckpt_data = await self._checkpointer.read_checkpoint(SchedulerState)

    async def _restore_comm_state(self) -> None:
        """Restore comm from cached checkpoint data using the comm's set_state.

        Must be called after _create_comm() and before setup_connections() so
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
        self._comm.update_node_info(self.info())
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
        if ckpt_sched_state is None or not ckpt_sched_state.children:
            return False
        self._scheduler.set_state(ckpt_sched_state)
        self.logger.info(
            f"{self.node_id}: Scheduler state restored from checkpoint "
            f"({len(ckpt_sched_state.children)} children)"
        )
        return True

    async def _create_policy_endpoint(self) -> None:
        """Serve this master's children-policy state to external ``PolicyClient``s.

        Started here rather than in ``_create_monitor_tasks`` so the address file is
        on disk before the node reports ready — a client that races node startup then
        blocks on the file rather than finding a half-initialized endpoint.

        No ``rescore_cb`` is registered: the children-policy analogue of re-scoring a
        queue is re-running resource/task assignment, which can revoke allocations and
        recycle live child processes. New children-policy state instead takes effect at
        the next natural assignment point.
        """
        if not (self._config.cluster or self._config.enable_policy_client):
            return
        if self._checkpointer is None:
            self.logger.warning(
                f"{self.node_id}: policy endpoint requested but checkpoint_dir is "
                "unset; PolicyClient discovery is file-based, so none will be served"
            )
            return

        self._policy_endpoint = PolicyEndpoint(
            self.node_id,
            self._checkpointer,
            self.logger.getChild("policy"),
        )
        self._policy_endpoint.register(KIND_CHILDREN, self._scheduler.policy)
        self._policy_endpoint.restore(await self._checkpointer.read_policy_state())
        await self._policy_endpoint.start()

    async def _write_checkpoint(self) -> None:
        """Write scheduler state and comm state to checkpoint."""
        if self._checkpointer is None:
            return
        await self._checkpointer.write_checkpoint(
            scheduler_state=self._scheduler.get_state(),
            comm_state=self._comm.get_state(),
        )
        await self._write_policy_checkpoint()

    async def _write_policy_checkpoint(self) -> None:
        """Persist tuned policy state, skipping the write when nothing changed.

        ``version`` only advances on a successful ``set``, so an untuned run never
        pays for this on top of every report interval.
        """
        if self._policy_endpoint is None:
            return
        version = self._policy_endpoint.version
        if version == self._policy_version_written:
            return
        await self._checkpointer.write_policy_state(self._policy_endpoint.snapshot())
        self._policy_version_written = version

    # --------------------------------------------------------------------------
    #                               Parent Synchronization
    # --------------------------------------------------------------------------

    async def _sync_with_parent(self) -> None:
        """Perform initial handshake with the parent: heartbeat, node update, task update."""
        # sync heart beat with parent
        if self.parent is None:
            return

        # Start the task to detect parent ready, which is set when the parent init is done and is ready to accept results
        self._parent_ready_monitor_task = asyncio.create_task(
            self._parent_ready_monitor()
        )

        if not await self._comm.sync_heartbeat_with_parent(timeout=1000.0):
            raise TimeoutError(f"{self.node_id}: Can't connect to parent")
        self.logger.info(f"{self.node_id}: Synced heartbeat with parent")

        # The parent pushes our NodeUpdate after the heartbeat sync (guaranteed delivery);
        # just wait for it. Tasks are pushed later and read by _parent_task_update_monitor.
        self._init_nodes = (await self._recv_from_parent(NodeUpdate)).nodes
        self.logger.info(
            f"{self.node_id}: Received NodeUpdate with {len(self._init_nodes.nodes)} nodes"
        )

    async def _recv_from_parent(self, cls):
        """Wait for the next ``cls`` message from the parent; raise if the parent dies."""
        while True:
            if self._comm.parent_dead_event.is_set():
                raise RuntimeError(f"{self.node_id}: Parent died while waiting for {cls.__name__}")
            msg = await self._comm.recv_message_from_parent(cls, timeout=5.0)
            if msg is not None:
                return msg

    async def _receive_first_task_update(self) -> None:
        """For children policies that lay out children from tasks (bin packing): take the
        parent's first TaskUpdate before creating children."""
        update = await self._recv_from_parent(TaskUpdate)
        await self._scheduler.queue(update)
        self.logger.info(f"{self.node_id}: Received first {update.ntasks} tasks for layout")

    # --------------------------------------------------------------------------
    #                               Child Synchronization
    # --------------------------------------------------------------------------

    def _build_init_node_update(self, child_id: str) -> NodeUpdate:
        """Build the initial NodeUpdate message to send to a child at startup."""
        child_nodes = self._scheduler.state.children[child_id].job_resource
        return NodeUpdate(sender=self.node_id, nodes=child_nodes)

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
        self.logger.debug(f"Syncing with child {child_id}")
        if not await self._comm.sync_heartbeat_with_child(
            child_id=child_id, timeout=600.0
        ):
            self.logger.error(f"Failed to sync heartbeat with child {child_id}")
            return await self._get_child_exception(child_id)
        self.logger.info(f"Successfully synced heart beat with {child_id}")
        child_dead_event = self._comm._child_dead_events.get(child_id, None)
        if child_dead_event is not None and child_dead_event.is_set():
            child_dead_event.clear()

        # Push the NodeUpdate and make sure it arrived; only then is the child RUNNING (and
        # its sender free to send tasks): a child must have its nodes before its tasks.
        if node_update is not None:
            sent = await self._comm.send_message_to_child(
                child_id, node_update, retries=self._config.send_retries
            )
            if not sent:
                self.logger.error(f"{self.node_id}: NodeUpdate to {child_id} was not delivered")
                failure = await self._get_child_exception(child_id)
                if failure is None:
                    failure = Result(sender=child_id, data=[])
                    failure.exception = f"NodeUpdate to {child_id} was not delivered"
                return failure
            self.logger.info(
                f"{self.node_id}: Sent node update to {child_id} containing {len(node_update.nodes.nodes)} nodes"
            )
        self._scheduler.transition(child_id, ChildState.RUNNING)
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

            self.logger.debug(f"Built NodeUpdate for child {child_id}")
            ##Task update handled by the seperate tast request monitor

            # Add sync task
            sync_tasks.append(self._sync_with_child(child_id, node_update))

        # Sync with all children in parallel
        results = await asyncio.gather(*sync_tasks, return_exceptions=True)
        return results

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

    async def _route_tasks(self, update: TaskUpdate, client_id: Optional[str] = None) -> None:
        """Hand a TaskUpdate to the scheduler. Placement happens in its assigner; each
        child's sender sends what lands in its outbox, and `_unplaced_monitor` fails what no
        child could take."""
        await self._scheduler.queue(update, client_id=client_id)

    async def _unplaced_monitor(self) -> None:
        """Fail tasks the scheduler could not place on any child."""
        while True:
            try:
                uids = await self._scheduler.unplaced.get()
                self._fail_tasks(uids, f"Routing failed at {self.node_id}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"{self.node_id}: Unplaced monitor error: {e}")

    def _fail_tasks(self, uids, reason: str, child_id: Optional[str] = None) -> None:
        """Report tasks as failed, as results routed like any other (to their client or the
        parent). ``child_id`` is the child holding them, if any."""
        store = self._scheduler.tasks
        failed = [
            Result(
                sender=self.node_id,
                uid=uid,
                task_id=store[uid].task_id if uid in store else None,
                success=False,
                exception=reason,
            )
            for uid in uids
        ]
        if failed:
            self.logger.warning(f"{self.node_id}: Failing {len(failed)} tasks: {reason}")
            self._scheduler.queue_result(IResultBatch.from_results(failed), child_id)

    def wait_drained(self):
        """Wait until every task this master was given has a result. Reports; it never stops
        the master -- `EnsembleLauncher` uses it to end a batch run with `request_stop`."""

        async def _wait():
            await self._initial_queued.wait()
            while self._scheduler is None:
                await asyncio.sleep(0.05)
            await self._scheduler.drained.wait()

        return _wait()

    def request_stop(self) -> None:
        """Stop this master the same way SIGTERM does."""
        self._stop_signal_received.set()

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

    async def _wait_for_child_done(self, child_id: str) -> None:
        """Wait for the child's Done: every result it sent has been delivered (they are in
        this master's IResultBatch buffer already), so the child is finished."""
        child = self._scheduler.state.children.get(child_id)
        if child is None:
            return
        # A child given up on (FAILED) never sends Done: stop waiting when it turns terminal.
        recv = asyncio.ensure_future(
            self._comm.recv_message_from_child(Done, child_id=child_id, block=True)
        )
        ended = asyncio.ensure_future(child.done.wait())
        try:
            await asyncio.wait({recv, ended}, return_when=asyncio.FIRST_COMPLETED)
            if not recv.done():
                self.logger.info(f"{self.node_id}: {child_id} ended ({child.state.name}) without Done")
                return
            msg = recv.result()
            if msg is not None:
                self.logger.info(f"{self.node_id}: {child_id} is done")
                if not child.terminal:
                    # Anything it still holds (e.g. placed after it took Stop) goes to
                    # unplaced and is failed.
                    self._scheduler.transition(child_id, ChildState.SUCCESS)
            else:
                self.logger.warning(f"{self.node_id}: No Done received from {child_id}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.logger.error(f"{self.node_id}: Error waiting for Done from {child_id}: {e}")
        finally:
            for t in (recv, ended):
                if not t.done():
                    t.cancel()

    # -------------------------------------------------------------------------
    #                           Stop path (_wait_for_finish)
    # -------------------------------------------------------------------------

    async def _wait_for_children_done(self, killed: bool) -> None:
        """Wait for every child's Done -- including children a recovery in progress adds --
        then stop the status reporter. After a KILL, children send nothing: cancel instead."""
        if killed:
            tasks = list(self._child_done_task.values())
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            seen: set = set()
            while True:
                new = set(self._child_done_task.values()) - seen
                if new:
                    seen.update(new)
                    await asyncio.gather(*new, return_exceptions=True)
                    continue  # recovery may have added more while we waited
                # all_done is not set while any child is RECOVERING.
                if self._all_children_done_event.is_set() and all(
                    t.done() for t in self._child_done_task.values()
                ):
                    break
                await asyncio.sleep(0.05)
        self.logger.info(f"{self.node_id}: All children are done")

        self._stop_reporting_event.set()
        if self._reporting_task is not None:
            self._reporting_task.cancel()
            try:
                await self._reporting_task
            except (asyncio.CancelledError, Exception):
                pass

    async def _drain_forwarders(self) -> None:
        """Each child's result monitor drains what that child streamed before its Done."""
        tasks = list(self._child_forwarder_task.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.logger.info(f"{self.node_id}: All forwarders drained")

    def _fail_unfinished(self) -> None:
        """Give every task still unfinished -- in the pool, or held by a child (unsent in its
        outbox, lost, or held by a killed child) -- a failed result, routed like any other."""
        for child_id, uids in self._scheduler.state.unfinished():
            self._fail_tasks(uids, f"Stopped before finishing at {self.node_id}", child_id)

    def _parent_alive(self) -> bool:
        return self.parent is not None and (
            self._comm.parent_dead_event is None or not self._comm.parent_dead_event.is_set()
        )

    async def _send_final_status_and_done(self) -> None:
        """Final Status, then Done (a non-root master); the root writes its status file.

        Runs after `_flush_results`, so Done means every result went up. If the parent is gone
        or a send fails, results still unsent are written to ``{node}_results.bin``."""
        status = self._scheduler.state.aggregate_status()
        status.tag = "final"
        if not self.parent:
            try:
                fname = os.path.join(os.getcwd(), f"{self.node_id}_status.json")
                status.to_file(fname)
                self.logger.info(f"{self.node_id}: Dumped final status to {fname}")
            except Exception as e:
                self.logger.warning(f"{self.node_id}: Writing final status failed: {e}")
            return

        delivered = False
        if self._parent_alive():
            if not await self._comm.send_message_to_parent(status):
                self.logger.warning(f"{self.node_id}: Failed to send final status to parent")
            if self._results_all_sent("parent"):
                delivered = await self._comm.send_message_to_parent(
                    Done(sender=self.node_id), retries=self._config.send_retries
                )
                if delivered:
                    self.logger.info(f"{self.node_id}: Sent Done to parent")
        if not delivered:
            self.logger.warning(f"{self.node_id}: Could not finish with parent; saving results")
            self._save_unsent_results()

    def _results_all_sent(self, dest: str) -> bool:
        """Nothing queued for ``dest`` and every message taken for it was delivered."""
        outbox = self._result_outbox.get(dest)
        return (outbox is None or outbox.empty()) and not self._result_unacked.get(dest)

    def _save_unsent_results(self) -> None:
        """Write the results not delivered to the parent -- taken by its sender but not
        ACKed, or still queued -- to ``{node}_results.bin``."""
        outbox = self._scheduler.parent_outbox
        pieces = [msg for msg, _ in self._result_unacked.pop("parent", [])]
        while not outbox.empty():
            pieces.append(outbox.get_nowait())
        if not pieces:
            return
        batch = ResultBatch.concat(pieces, sender=self.node_id)
        fname = os.path.join(os.getcwd(), f"{self.node_id}_results.bin")
        with open(fname, "wb") as f:
            f.write(batch.to_bytes())
        self.logger.info(f"{self.node_id}: Saved {batch.nresults} unsent results to {fname}")

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
                status = self._scheduler.state.aggregate_status()
                if self.parent:
                    await self._comm.send_message_to_parent(status)
                    self.logger.info(status)
                else:
                    self.logger.info(status)
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

    @property
    def _checkpoint_interval(self) -> float:
        """How often to checkpoint, defaulting to the status cadence.

        Checkpointing used to ride the status loop, which tied how fresh a peer's view of this
        node is to how much state gets serialised and fsynced. They scale differently: a
        ``Status`` is a handful of integers, while a checkpoint rewrites every task id this
        node has handled. Leaving the interval at ``None`` keeps the old coupled behaviour.
        """
        if self._config.checkpoint_interval is None:
            return self._config.report_interval
        return self._config.checkpoint_interval

    def _start_checkpoint_write(self) -> None:
        """Start a checkpoint write unless the previous one is still running.

        A checkpoint is a snapshot of the moment it is taken, so a tick skipped because the
        last write is still in flight costs nothing -- the next one captures a newer state
        anyway. Skipping also keeps writes from overlapping, which would race on the shared
        metadata file that each write reads before rewriting.
        """
        if self._checkpoint_write is not None and not self._checkpoint_write.done():
            self.logger.debug(
                f"{self.node_id}: Checkpoint still writing, skipping this interval"
            )
            return
        self._checkpoint_write = asyncio.create_task(self._write_checkpoint())
        self._checkpoint_write.add_done_callback(self._checkpoint_write_done)

    def _checkpoint_write_done(self, task: asyncio.Task) -> None:
        """Surface a failed checkpoint write, which is otherwise silent."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self.logger.error(f"{self.node_id}: Checkpoint write failed: {exc}")

    async def _periodic_checkpoint_loop(self) -> None:
        """Write checkpoints on their own interval, independent of status reporting.

        Writes before waiting, not after. A ``ClusterClient`` blocks on this node's comm file
        appearing before it can connect, so the first checkpoint has to land as promptly as it
        did when this rode the status loop -- not one (possibly much longer) interval later.
        """
        while not self._stop_reporting_event.is_set():
            try:
                self._start_checkpoint_write()
                jitter = random.uniform(-0.05, 0.05) * self._checkpoint_interval
                try:
                    await asyncio.wait_for(
                        self._stop_reporting_event.wait(),
                        timeout=self._checkpoint_interval + jitter,
                    )
                    break  # Exit if stop event was set
                except asyncio.TimeoutError:
                    pass  # Continue loop after interval
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.info(f"Checkpoint loop failed with error {e}")
                await asyncio.sleep(0.1)

    async def _client_request_monitor(self) -> None:
        """Cluster mode: handle messages from any ClusterClient connected to this master."""
        while not self._all_children_done_event.is_set():
            try:
                item = await self._comm.recv_client_message()
                if item is None:
                    continue
                client_id, msg = item
                if isinstance(msg, TaskUpdate):
                    self._total_received += msg.ntasks
                    self.logger.info(
                        f"Received TaskUpdate from client. Total tasks: {self._total_received}"
                    )
                    await self._route_tasks(msg, client_id=client_id)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Client task update monitor failed with error: {e}")
                await asyncio.sleep(0.5)

    def _send_result_batch(self, dest_id: str, msg: IResultBatch, retries: int = 1):
        """Return the send coroutine for an IResultBatch bound for `dest_id`."""
        if dest_id.startswith("client-"):
            return self._comm.send_message_to_child(dest_id, msg, retries=retries)
        return self._comm.send_message_to_parent(msg, retries=retries)

    def _start_result_sender(self, dest: str, outbox: asyncio.Queue) -> None:
        """Start the sender for ``dest`` ("parent" or a client id) on its result outbox."""
        self._result_outbox[dest] = outbox
        self._result_window[dest] = asyncio.Semaphore(max(1, self._config.max_inflight_flushes))
        self._result_sender_task[dest] = asyncio.create_task(self._result_sender(dest))

    async def _new_client_monitor(self) -> None:
        """Start a result sender for each client the scheduler announces."""
        while True:
            client = await self._scheduler.new_clients.get()
            self._start_result_sender(client, self._scheduler.client_outboxes[client])

    async def _result_sender(self, dest: str) -> None:
        """Send ``dest``'s result outbox as it fills.

        Takes a window slot first, then an item, then everything else already queued, and
        sends them as one IResultBatch: whatever arrives during a send leaves together in
        the next one. The slot is released by `_result_flush_done`.
        """
        outbox, window = self._result_outbox[dest], self._result_window[dest]
        while True:
            await window.acquire()
            pieces = [await outbox.get()]
            while not outbox.empty():
                pieces.append(outbox.get_nowait())
            msg = IResultBatch.concat(pieces, sender=self.node_id)
            self._result_unacked.setdefault(dest, []).append((msg, len(pieces)))
            self._schedule_result_flush(dest, msg, len(pieces), window)

    def _schedule_result_flush(
        self, dest: str, msg: IResultBatch, nitems: int, window: asyncio.Semaphore
    ) -> None:
        """Start one IResultBatch send to ``dest``, in a window slot the caller holds."""
        if self._result_window.get(dest) is not window:
            return  # the sender is gone (stopping)
        t = asyncio.create_task(self._send_result_batch(dest, msg))
        self._result_flush_inflight.setdefault(dest, set()).add(t)
        t.add_done_callback(
            functools.partial(self._result_flush_done, dest, msg, nitems, window)
        )

    def _result_flush_done(
        self, dest: str, msg: IResultBatch, nitems: int, window: asyncio.Semaphore,
        task: asyncio.Task,
    ) -> None:
        """A result send finished: free its slot, or resend the same message in it.

        The window is the one the send was started in, never looked up again, so a late
        callback cannot release another sender's slot. The resend is the same object, so it
        keeps its message_id and the receiver drops the copy if only the ACK was lost.
        """
        inflight = self._result_flush_inflight.get(dest)
        if inflight is not None:
            inflight.discard(task)
        if task.cancelled():
            window.release()
            return
        exc = task.exception()
        if exc is None and task.result():
            if dest.startswith("client-"):
                self._total_streamed += msg.nresults
            unacked = self._result_unacked.get(dest)
            if unacked is not None:
                unacked[:] = [(m, n) for m, n in unacked if m is not msg]
            outbox = self._result_outbox.get(dest)
            if outbox is not None:
                for _ in range(nitems):
                    outbox.task_done()
            window.release()
            return
        self.logger.warning(
            f"Failed to send IResultBatch of size {msg.nresults} to {dest}"
            f"{f' ({exc})' if exc else ''}, resending"
        )
        # Same slot, so nothing newer overtakes it. A send that raised failed at once, so
        # wait an ACK timeout first rather than spin.
        delay = self._config.send_timeout if exc is not None else 0.0
        asyncio.get_running_loop().call_later(
            delay, self._schedule_result_flush, dest, msg, nitems, window
        )

    async def _flush_results(self) -> None:
        """Wait until every result sender has delivered everything queued for it.

        Senders keep resending failures, so this is bounded: ``send_retries`` ACK timeouts
        per destination, after which the stop goes on. The parent is skipped once dead.
        (The root has no parent sender: its parent outbox is the run's results.)
        """
        timeout = max(1, self._config.send_retries) * self._config.send_timeout
        dests = [
            d for d in self._result_outbox if d != "parent" or self._parent_alive()
        ]

        async def _one(dest: str) -> None:
            try:
                await asyncio.wait_for(self._result_outbox[dest].join(), timeout)
            except asyncio.TimeoutError:
                self.logger.error(f"{self.node_id}: Results for {dest} not delivered in {timeout}s")

        await asyncio.gather(*(_one(d) for d in dests))

    async def _child_sender(self, child_id: str) -> None:
        """Send this child's outbox, one item per TaskUpdate, once the child is running.

        Takes a window slot *before* waiting for an item, so while sends are in flight new
        items simply wait in the outbox. The slot is released by `_task_flush_done`. The
        update is built on the loop, like every other store access.
        """
        child = self._scheduler.state.children[child_id]
        window = self._task_window[child_id]
        build = self._scheduler.tasks.build_update
        await child.ready.wait()  # set once its NodeUpdate was delivered
        while True:
            await window.acquire()
            uids = [uid for uid in await child.outbox.get() if uid in child]  # drop deleted ones
            if not uids:
                window.release()
                continue
            self._schedule_task_flush(child_id, build(uids, sender=self.node_id))

    def _schedule_task_flush(
        self, child_id: str, msg: TaskUpdate, window: Optional[asyncio.Semaphore] = None
    ) -> None:
        """Start one TaskUpdate send to child_id, in a window slot the caller holds.

        A resend passes the window its first send held. If the child was torn down since (its
        window dropped, or replaced by a reinstated child's), the resend is dropped: recovery
        refilled the new child's outbox, and the new process could not dedupe the old message.
        """
        current = self._task_window.get(child_id)
        if window is None:
            window = current
        if window is None or window is not current:
            return
        t = asyncio.create_task(self._comm.send_message_to_child(child_id, msg))
        self._task_flush_inflight.setdefault(child_id, set()).add(t)
        t.add_done_callback(functools.partial(self._task_flush_done, child_id, msg, window))

    def _task_flush_done(self, child_id: str, msg: TaskUpdate, window: asyncio.Semaphore, task: asyncio.Task) -> None:
        """A TaskUpdate send finished: free its slot, or resend the same message in it.

        Synchronous, so the child-state check and the resend are atomic with respect to the
        event loop. The resend is the same object, so it keeps its message_id and the child
        drops the copy if only the ACK was lost.
        """
        inflight = self._task_flush_inflight.get(child_id)
        if inflight is not None:
            inflight.discard(task)
        if task.cancelled():
            window.release()
            return
        exc = task.exception()
        if exc is None and task.result():
            self.logger.info(f"Sent TaskUpdate of size {msg.ntasks} to {child_id}")
            self._scheduler.delivered(msg.uids())
            window.release()
            return
        state = self._child_state(child_id)
        if state in (ChildState.SUCCESS, ChildState.FAILED):
            self.logger.error(
                f"Failed to send TaskUpdate of size {msg.ntasks} to {child_id} (state={state})"
            )
            # Only what it still holds: a FAILED child's tasks were requeued or failed already.
            child = self._scheduler.state.children.get(child_id)
            held = [u for u in msg.uids() if child is not None and u in child]
            self._fail_tasks(held, f"Send to {child_id} failed at {self.node_id}", child_id)
        elif state in (None, ChildState.RECOVERING):
            # Recovery owns these tasks: the reinstated child's outbox holds them again.
            self.logger.info(f"Send to {child_id} failed while it recovers; dropping it")
        else:
            self.logger.info(
                f"Send to {child_id} failed (state={state}){f' ({exc})' if exc else ''}, "
                f"resending {msg.ntasks} tasks"
            )
            # Same slot, so nothing newer overtakes it. A send that raised failed at once,
            # so wait an ACK timeout first rather than spin.
            delay = self._config.send_timeout if exc is not None else 0.0
            asyncio.get_running_loop().call_later(
                delay, self._schedule_task_flush, child_id, msg, window
            )
            return
        window.release()

    async def _child_result_monitor(self, child_id: str) -> None:
        """Receive IResultBatch messages from child and forward to client or parent."""
        child_done = self._scheduler.state.children[child_id].done
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
                    try:
                        self._scheduler.queue_result(msg, child_id)
                    except Exception as e:
                        self.logger.error(
                            f"{self.node_id}: Failed to forward result from {child_id}: {e}",
                            exc_info=True,
                        )
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
            try:
                self._scheduler.queue_result(msg, child_id)
            except Exception as e:
                self.logger.error(
                    f"{self.node_id}: Failed to forward result from {child_id}: {e}",
                    exc_info=True,
                )

    async def _child_status_monitor(self, child_id: str) -> None:
        """Continuously collect status messages from a single child and update scheduler."""
        child_done = self._scheduler.state.children[child_id].done
        try:
            while not child_done.is_set():
                status = await self._comm.recv_message_from_child(
                    Status, child_id=child_id, block=True
                )
                if status is not None:
                    self._set_child_status(child_id, status)
                    if status.tag == "final":
                        self.logger.info(
                            f"Received final status from {child_id}"
                        )
                        return
            # Drain any remaining status messages after child process exits
            while True:
                status = await self._comm.recv_message_from_child(
                    Status, child_id=child_id
                )
                if status is None:
                    break
                self._set_child_status(child_id, status)
        except asyncio.CancelledError:
            pass

    async def _child_liveness_monitor(self, child_id: str) -> None:
        """Wait for child dead event and trigger recovery."""
        while True:
            try:
                ev = self._comm._child_dead_events.get(child_id)
                await ev.wait()
                state = self._child_state(child_id)
                if state in (None, ChildState.SUCCESS, ChildState.FAILED):
                    return
                self.logger.warning(
                    f"{self.node_id}: Child {child_id} declared dead by HB process"
                )
                if self._config.restart_children_on_failure:
                    self._scheduler.transition(child_id, ChildState.RECOVERING)
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
                                child_id,
                                Ready(sender=self.node_id),
                                retries=self._config.send_retries,
                            )
                            self.logger.info(
                                f"{self.node_id}: Sent Ready to recovered child {child_id}"
                            )
                            # _init_child started a fresh monitor for the new process.
                            return
                    if not success:
                        self._give_up_on_child(child_id)
                        break
                else:
                    self._give_up_on_child(child_id)
                    break
            except Exception as e:
                self.logger.warning(f"liveness monitor failed with Exception {e}")
                await asyncio.sleep(10.0)

    def _give_up_on_child(self, child_id: str) -> None:
        """Mark a child FAILED for good -- whether it died running or never got running (a
        reinstated child whose relaunch failed is READY). The scheduler requeues its tasks on
        the other children (failed through `unplaced` if none can take them), and its Done
        wait ends, so the stop path doesn't wait on it."""
        child = self._scheduler.state.children.get(child_id)
        if child is None or child.terminal:
            return
        self.logger.warning(
            f"{self.node_id}: Giving up on {child_id}; requeueing its {len(child)} tasks"
        )
        self._scheduler.transition(child_id, ChildState.FAILED, requeue=True)

    async def _recover_dead_child(self, child_id: str) -> bool:
        """Tear down a dead child and relaunch on the same resources."""
        try:
            self.logger.info(f"{self.node_id}: Recovering dead child {child_id}")

            # Cancels the child's futures and tasks and removes it from the scheduler and
            # comm; the removed record keeps its resources and unfinished tasks.
            child = await self._teardown_child(child_id)
            if child is None:
                raise RuntimeError(f"{child_id} is not a child of {self.node_id}")

            # Back on the same resources with the same tasks, READY so _launch_child can
            # move it to RUNNING.
            if not self._scheduler.reinstate_child(child):
                self.logger.error(
                    "Can't allocate the identical child resources while recovering"
                )
                # The child can't come back: place its tasks (back in the pool) on the others.
                self._scheduler.requeue(child.uids)
                raise RuntimeError(
                    "Can't allocate the identical child resources while recovering"
                )
            self.logger.info(f"Reallocated resources for {child_id}: {child.job_resource}")

            # Instantiate a fresh child object — never reuse the old one since it
            # carries stale asyncio.Event / comm state from the previous run, which
            # causes pickling failures or immediate early-exit in the subprocess.
            fresh_children = self._instantiate_children({child_id})
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
                    await self._route_tasks(task_update)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Task update monitor error: {e}")
                await asyncio.sleep(0.1)

    # -------------------------------------------------------------------------
    #                               Teardown
    # -------------------------------------------------------------------------

    async def _teardown_child(self, child_id: str) -> Optional[Child]:
        """Cancel per-child tasks, remove the child from all bookkeeping, and prune
        the comm cache so no stale entries remain. Returns the scheduler's removed record."""
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
            self._child_done_task,
            self._child_status_task,
            self._child_forwarder_task,
            self._child_task_request_task,
            self._child_sender_task,
        ]
        if self._child_state(child_id) != ChildState.RECOVERING:
            task_dicts.append(self._child_liveness_task)
        for task_dict in task_dicts:
            t = task_dict.pop(child_id, None)
            if t is not None:
                t.cancel()
                tasks_to_cancel.append(t)
        for t in self._task_flush_inflight.pop(child_id, set()):
            t.cancel()
            tasks_to_cancel.append(t)
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

        removed = self._scheduler.remove_child(child_id)
        self.remove_child(child_id)
        self._child_objs.pop(child_id, None)
        self._children_futures.pop(child_id, None)
        self._task_window.pop(child_id, None)
        self._comm.update_node_info(self.info())
        return removed

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

        if self._scheduler is not None:
            await self._scheduler.stop()  # the assigner

        for loop_task in (
            self._new_client_task,
            self._unplaced_task,
            *self._result_sender_task.values(),
            *self._child_sender_task.values(),
        ):
            if loop_task and not loop_task.done():
                loop_task.cancel()
                try:
                    await loop_task
                except asyncio.CancelledError:
                    pass
        # Senders are gone: a resend still on a timer finds no window and is dropped.
        self._result_window.clear()
        self._task_window.clear()

        inflight_flushes = [
            t
            for inflight in (
                *self._result_flush_inflight.values(),
                *self._task_flush_inflight.values(),
            )
            for t in inflight
        ]
        for t in inflight_flushes:
            t.cancel()
        if inflight_flushes:
            await asyncio.gather(*inflight_flushes, return_exceptions=True)

        if self._checkpoint_task and not self._checkpoint_task.done():
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except asyncio.CancelledError:
                pass

        # Let any write already running finish before the final checkpoint below, so the two
        # do not race to replace the same files.
        if self._checkpoint_write and not self._checkpoint_write.done():
            try:
                await self._checkpoint_write
            except Exception:
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

        if self._policy_endpoint is not None:
            await self._policy_endpoint.stop()

        # Tear down children and wait for them to be done
        for child_name in list(self._scheduler.state.children):
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
        self.logger.info("Shutting down executor")
        if hasattr(self._executor, "ashutdown"):
            await self._executor.ashutdown()
        else:
            self._executor.shutdown()
        self.logger.info("Done Shutting down executor")

        # Drop held task bytes and delete any offloaded segment files.
        if self._scheduler is not None:
            self._scheduler.tasks.close()

        self.logger.info(f"Done stopping {self.node_id}")
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    # -------------------------------------------------------------------------
    #                               Entry point
    # -------------------------------------------------------------------------

    async def _wait_for_finish(self) -> None:
        """The master's one stop path, top to bottom.

        A master never decides by itself that it is finished. It stops on the first of: its
        parent's Stop, its stop signal (SIGTERM, or ``request_stop()`` -- EnsembleLauncher ends a
        batch run this way once ``wait_drained()`` resolves), or its parent dying. Then it stops
        its children, takes in what they sent, fails whatever is left, delivers every result,
        and reports Done.
        """
        import sys

        received = await self._wait_for_stop_condition()
        stop_type = await self._propagate_stop(received)
        if received == StopType.KILL:
            # Our parent wants us gone now: no result exchange, keep what we have on disk.
            await self._wait_for_children_done(killed=True)
            self._save_unsent_results()
            await self.stop()
            sys.exit(1)
        await self._wait_for_children_done(killed=stop_type == StopType.KILL)
        await self._drain_forwarders()
        self._fail_unfinished()
        await self._flush_results()
        await self._send_final_status_and_done()

    async def _wait_for_stop_condition(self) -> Optional[StopType]:
        """Wait for the first stop condition. Returns the Stop type the parent sent, if that
        is what happened."""
        stop_tasks: Dict[str, asyncio.Task] = {}
        received_stop_type: List[Optional[StopType]] = [None]

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

        _, pending = await asyncio.wait(
            set(stop_tasks.values()), return_when=asyncio.FIRST_COMPLETED
        )
        for t in pending:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        return received_stop_type[0]

    async def _propagate_stop(self, received: Optional[StopType]) -> Optional[StopType]:
        """Stop the children: KILL if our parent died, the type our parent sent, else
        TERMINATE. Returns what was sent (None if every child had already finished)."""
        if self._all_children_done_event.is_set():
            return None
        if self._comm.parent_dead_event is not None and self._comm.parent_dead_event.is_set():
            stop_type = StopType.KILL
        elif received is not None:
            stop_type = received
        else:
            stop_type = StopType.TERMINATE
        self.logger.info(f"{self.node_id}: Propagating {stop_type.value} to children")
        for child_id in list(self.children.keys()):
            try:
                await self._comm.send_message_to_child(
                    child_id,
                    Stop(sender=self.node_id, type=stop_type),
                    retries=self._config.send_retries,
                )
            except Exception:
                pass
        return stop_type

    async def run(self) -> ResultBatch:
        """Main entry point: initialise, wait for the stop path to finish, stop, return results.

        The root returns every result with no client (its parent outbox, which nothing sends);
        a sub-master has streamed everything up and returns an empty batch."""
        try:
            await self._lazy_init()
            await self._wait_for_finish()
        finally:
            await self.stop()

        pieces = []
        if not self.parent and self._scheduler is not None:
            outbox = self._scheduler.parent_outbox
            while not outbox.empty():
                pieces.append(outbox.get_nowait())
        return ResultBatch.concat(pieces, sender=self.node_id)

    def create_an_event_loop(self) -> None:
        """Entry point for a new child process: run the async event loop."""
        from ensemble_launcher.loop import run
        run(self.run())

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
            "parent_data_conn": self.parent_data_conn.get_state().serialize()
            if self.parent_data_conn is not None
            else None,
            "parent_hb_conn": self.parent_hb_conn.get_state().serialize()
            if self.parent_hb_conn is not None
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

        ##Get the serial state
        parent_data_conn_data: Dict = data.get("parent_data_conn")
        parent_hb_conn_data: Dict = data.get("parent_hb_conn")
        ##
        parent_data_conn = None
        parent_hb_conn = None
        if parent_data_conn_data is not None:
            tt = transport_registry.get(
                json.loads(parent_data_conn_data)["transport_type"]
            )

            parent_data_conn_state = tt["client_connection_state"].deserialize(
                parent_data_conn_data
            )
            parent_data_conn = tt["client_connection"].set_state(parent_data_conn_state)
        ##
        if parent_hb_conn_data is not None:
            tt = transport_registry.get(
                json.loads(parent_hb_conn_data)["transport_type"]
            )
            parent_hb_conn_state = tt["client_connection_state"].deserialize(
                parent_hb_conn_data
            )
            parent_hb_conn = tt["client_connection"].set_state(parent_hb_conn_state)

        master = cls(
            id=data["node_id"],
            config=config,
            Nodes=None,
            tasks={},
            parent=parent,
            children=children,
            parent_data_conn=parent_data_conn,
            parent_hb_conn=parent_hb_conn,
        )
        master._secret_id = data["secret_id"]
        return master
