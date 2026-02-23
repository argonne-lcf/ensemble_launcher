import asyncio
import base64
import json
import logging
import os
import socket
import time
from concurrent.futures import Future as ConcurrentFuture
from contextlib import asynccontextmanager
from dataclasses import asdict
from typing import Callable, Dict, List, Optional, Union

from ensemble_launcher.comm import AsyncComm, AsyncZMQComm, NodeInfo
from ensemble_launcher.comm.messages import (
    NodeUpdate,
    Result,
    ResultBatch,
    Status,
    TaskUpdate,
)
from ensemble_launcher.config import LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.executors import Executor, executor_registry
from ensemble_launcher.profiling import EventRegistry, get_registry
from ensemble_launcher.scheduler import AsyncWorkerScheduler
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from .async_worker import AsyncWorker
from .node import Node
from .utils import async_load_str, async_simple_load_str

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
        self._all_children_done_event = asyncio.Event()
        self._stop_reporting_event = asyncio.Event()
        self._event_loop = None  # Will be set in run()

        # result and aggregate tasks
        self._result_tasks = []
        self._aggregate_task = None
        self._child_result_task: Dict[
            str, asyncio.Task
        ] = {}  # child_id -> collect task
        self._child_forwarder_task: Dict[
            str, asyncio.Task
        ] = {}  # child_id -> result monitor task

        # Cluster mode state
        self._stop_task_update = asyncio.Event()
        self._task_update_task: Optional[asyncio.Task] = None
        self._client_monitor_task: Optional[asyncio.Task] = None
        self._reporting_task: asyncio.Task = None
        self._client_task_map: Dict[str, str] = {}  # task_id -> client_id

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
        if self._config.master_logs:
            os.makedirs(os.path.join(os.getcwd(), "logs"), exist_ok=True)
            # Configure file handler for this specific self.self.logger
            file_handler = logging.FileHandler(
                os.path.join(os.getcwd(), f"logs/master-{self.node_id}.log")
            )
            file_handler.setLevel(self._config.log_level)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            file_handler.setFormatter(formatter)
            # Create instance self.self.logger and add handler
            self.logger = logging.getLogger(f"{__name__}.{self.node_id}")
            self.logger.addHandler(file_handler)
            self.logger.setLevel(self._config.log_level)
        else:
            self.logger = logging.getLogger(__name__)

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

    def _create_scheduler(self) -> AsyncWorkerScheduler:
        """Instantiate the worker scheduler.  Override to change init args."""
        return AsyncWorkerScheduler(
            self.logger.getChild("scheduler"),
            self._init_nodes,
            self._config,
            tasks=self._init_tasks,
        )

    def _get_child_class(self) -> type:
        """Return the Node class to use for children at the next level."""
        if self.level + 1 == self._config.nlevels:
            return AsyncWorker
        from .async_workstealing_master import AsyncWorkStealingMaster

        return (
            AsyncWorkStealingMaster if self._config.enable_workstealing else AsyncMaster
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

        if not partial:
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
        task = asyncio.create_task(self._collect_final_result_from_child(child_id))
        self._result_tasks.append(task)
        self._child_result_task[child_id] = task
        if self._config.cluster:
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
        if self._config.cluster:
            self._client_monitor_task = asyncio.create_task(
                self._client_request_monitor()
            )
            if self.parent:
                self._task_update_task = asyncio.create_task(
                    self._parent_task_update_monitor()
                )

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
            json_str_b64 = base64.b64encode(json_str.encode("utf-8")).decode("ascii")

            # Create embedded command string for this child (simple version, no per-host logic)
            load_str_embed = async_simple_load_str.replace(
                "json_str_b64", f"b'{json_str_b64}'"
            )

            req = JobResource(resources=[NodeResourceCount(ncpus=1)], nodes=[head_node])
            env = os.environ.copy()
            env["EL_CHILDID"] = str(child_idx)

            ##get mpi kwargs
            if isinstance(child_obj.init_nodes.resources[0], NodeResourceList):
                cpus = ":".join(map(str, child_obj.init_nodes.resources[0].cpus))
            else:
                cpus = ":".join(
                    map(str, list(range(child_obj.init_nodes.resources[0].cpu_count)))
                )
            if self._config.cpu_binding_option == "--cpu-bind":
                mpi_kwargs = {self._config.cpu_binding_option: f"list:{cpus}"}
                self.logger.info(f"Setting cpu affinity to child:{mpi_kwargs}")
            else:
                mpi_kwargs = {}
                self.logger.warning(
                    f"Unknown cpu binding option {self._config.cpu_binding_option}. Ignoring child pinning."
                )

            self.logger.info(f"Launching child {child_name} using MPI executor")
            future = self._executor.submit(
                req, ["python", "-c", load_str_embed], env=env, mpi_kwargs=mpi_kwargs
            )
            future.add_done_callback(self._create_done_callback([child_name]))
            self._children_futures[child_name] = future
            self._scheduler.mark_child_running(child_name)
        else:
            child_nodes = child_obj.init_nodes.nodes
            req = JobResource(
                resources=[
                    NodeResourceList(cpus=child_obj.init_nodes.resources[0].cpus)
                ],
                nodes=child_nodes[:1],
            )
            env = os.environ.copy()
            env["EL_CHILDID"] = str(child_idx)

            future = self._executor.submit(req, child_obj.create_an_event_loop, env=env)
            future.add_done_callback(self._create_done_callback([child_name]))
            self._children_futures[child_name] = future
            self._scheduler.mark_child_running(child_name)

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
            if not self._config.sequential_child_launch and worker_equality:
                ##launch all children in a single shot
                child_head_nodes = []
                child_resources = []
                child_obj_dict = {}

                for child_name, child_obj in children.items():
                    head_node = child_obj.init_nodes.nodes[0]
                    child_head_nodes.append(head_node)
                    child_resources.append(NodeResourceCount(ncpus=1))
                    child_obj_dict[head_node] = child_obj

                # Build combined dictionary structure
                common_keys = ["type", "parent", "parent_comm"]
                if all(
                    [
                        "task_executor_name" not in cdict
                        for cdict in self._child_assignment.values()
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
                for hostname, child_obj in child_obj_dict.items():
                    child_dict = child_obj.asdict()
                    for key, value in child_dict.items():
                        if key not in common_keys:
                            self.logger.info(f"{key} not int common keys")
                            final_dict[key][hostname] = value

                self.logger.info(f"Final dict: {final_dict}")
                # Create embedded command string
                json_str = json.dumps(final_dict, default=str)
                json_str_b64 = base64.b64encode(json_str.encode("utf-8")).decode(
                    "ascii"
                )
                common_keys_str = ",".join(common_keys)
                load_str_embed = async_load_str.replace(
                    "json_str_b64", f"b'{json_str_b64}'"
                )
                load_str_embed = load_str_embed.replace(
                    "common_keys_str", f"'{common_keys_str}'"
                )

                req = JobResource(resources=child_resources, nodes=child_head_nodes)
                env = os.environ.copy()

                self.logger.info("Launching worker using one shot mpiexec")
                ##get mpi kwargs
                if isinstance(first_child.nodes.resources[0], NodeResourceList):
                    cpus = ":".join(map(str, first_child.nodes.resources[0].cpus))
                else:
                    cpus = ":".join(
                        map(str, list(range(first_child.nodes.resources[0].cpu_count)))
                    )
                if self._config.cpu_binding_option == "--cpu-bind":
                    mpi_kwargs = {self._config.cpu_binding_option: f"list:{cpus}"}
                else:
                    mpi_kwargs = {}
                    self.logger.warning(
                        f"Unknown cpu binding option {self._config.cpu_binding_option}. Ignoring child pinning."
                    )

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
                    self._scheduler.mark_child_running(child_id)
                future.add_done_callback(self._create_done_callback(child_info))
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

        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm()  ###This will only create picklable objects

        # Start parent comm end point monitor
        await self._comm.start_monitors(parent_only=True)

        # for zmq, setup the sockets
        if self._config.comm_name == "async_zmq":
            await self._comm.setup_zmq_sockets()

        # Receive node update from parent if it has a parent
        if self.parent:
            await self._sync_with_parent()

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

        # Check executor validity
        assert self._config.child_executor_name in executor_registry.async_executors, (
            f"Executor {self._config.child_executor_name} not found in async executors {executor_registry.async_executors}"
        )

        kwargs = {}
        kwargs["logger"] = self.logger.getChild("executor")
        kwargs["max_workers"] = self.nodes.resources[0].cpu_count
        if self._config.child_executor_name == "async_mpi":
            kwargs["cpu_binding_option"] = self._config.cpu_binding_option
            kwargs["use_ppn"] = self._config.use_mpi_ppn

        # Create executor
        self._executor: Executor = executor_registry.create_executor(
            self._config.child_executor_name, kwargs=kwargs
        )
        self.logger.info(f"Created {self._config.child_executor_name} executor")

        # Create children and initialise each one (tree registration + per-child tasks).
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

        max_retries = 2
        for attempt in range(max_retries):
            if not failed_names:
                break
            self.logger.warning(
                f"{self.node_id}: Retrying {len(failed_names)} failed children "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            results = await self._launch_and_sync_children(failed_names)
            failed_names = [
                name for name, r in zip(failed_names, results) if r is not None
            ]

        if failed_names:
            self.logger.warning(
                f"{self.node_id}: {len(failed_names)} children still failed after "
                f"{max_retries} retries, attempting recreation: {failed_names}"
            )

            # Collect nodes from failed children before tearing them down.
            recovered_resources = []
            recovered_node_list = []
            for child_id in failed_names:
                jr = self._scheduler.get_child_assignment(child_id)["job_resource"]
                recovered_resources.extend(jr.resources)
                if jr.nodes:
                    recovered_node_list.extend(jr.nodes)
            recovered_nodes = JobResource(
                resources=recovered_resources,
                nodes=recovered_node_list if recovered_node_list else None,
            )

            # _teardown_child → remove_child restores the child's task_ids to
            # the scheduler's unassigned pool automatically.
            for child_id in failed_names:
                await self._teardown_child(child_id)

            if self._scheduler.unassigned_task_ids:
                # Recreate children using only the nodes freed from the failed children.
                new_children = self._create_children(
                    nodes=recovered_nodes, partial=True
                )
                for child_id, child in new_children.items():
                    await self._init_child(child_id, child)

                results = await self._launch_and_sync_children(
                    list(new_children.keys())
                )
                still_failed = [
                    n for n, r in zip(new_children.keys(), results) if r is not None
                ]
                if still_failed:
                    self.logger.error(
                        f"{self.node_id}: Could not recover children after recreation: {still_failed}"
                    )
                    for child_id in still_failed:
                        await self._teardown_child(child_id)

        # Create aggregate task once, after all children (including recreated ones) are registered.
        self._aggregate_task = asyncio.create_task(
            self._aggregate_and_send_results(self._result_tasks)
        )
        return None

    # --------------------------------------------------------------------------
    #                               Parent Synchronization
    # --------------------------------------------------------------------------

    async def _sync_with_parent(self) -> None:
        """Perform initial handshake with the parent: heartbeat, node update, task update."""
        # sync heart beat with parent
        async with self._timer("heartbeat_sync"):
            if self.parent and not await self._comm.sync_heartbeat_with_parent(
                timeout=30.0
            ):
                raise TimeoutError(f"{self.node_id}: Can't connect to parent")
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
                self.logger.debug(f"{self.node_id}: Nodes details: {self._init_nodes}")
            else:
                self.logger.warning(
                    f"{self.node_id}: Received empty node update from parent"
                )
        else:
            self.logger.warning(
                f"{self.node_id}: No node update received from parent at start"
            )

        # Receive task update from parent
        task_update: TaskUpdate = await self._comm.recv_message_from_parent(
            TaskUpdate, timeout=5.0
        )
        if task_update is not None:
            self.logger.info(
                f"{self.node_id}: Received task update from parent containing {len(task_update.added_tasks)}"
            )
            for task in task_update.added_tasks:
                self._init_tasks[task.task_id] = task

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
        new_tasks = [
            self._scheduler.tasks[task_id]
            for task_id in self._scheduler.get_child_assignment(child_id)["task_ids"]
        ]
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
        task_update: Optional[TaskUpdate],
    ) -> Optional[Result]:
        """
        Sync with a single child and send initial node/task updates.

        Args:
            child_id: The ID of the child to sync with
            node_update: NodeUpdate message to send to the child
            task_update: TaskUpdate message to send to the child

        Returns:
            None if successful, Result object with exception if failed
        """
        # Sync heartbeat with child
        if not await self._comm.sync_heartbeat_with_child(
            child_id=child_id, timeout=30.0
        ):
            self.logger.error(f"Failed to sync heartbeat with child {child_id}")
            return await self._get_child_exception(child_id)

        # Send node update first
        if node_update is not None:
            await self._comm.send_message_to_child(child_id, node_update)
            self.logger.info(
                f"{self.node_id}: Sent node update to {child_id} containing {len(node_update.nodes.nodes)} nodes"
            )

        # Then send task update
        if task_update is not None:
            await self._comm.send_message_to_child(child_id, task_update)
            self.logger.info(
                f"{self.node_id}: Sent task update to {child_id} containing {len(task_update.added_tasks)} tasks"
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

            # Create task update
            task_update = self._build_init_task_update(child_id)

            # Add sync task
            sync_tasks.append(self._sync_with_child(child_id, node_update, task_update))

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
        await self._launch_children(child_names)
        results = await self._sync_with_children(child_names)
        return results

    # --------------------------------------------------------------------------
    #                               Task Routing
    # --------------------------------------------------------------------------

    def _route_task(self, task: Task) -> str:
        """Route a task to the best child via scheduler policy. Returns chosen child_id."""
        self._scheduler.add_task(task)
        child_assignments = self._scheduler.assign_task_ids({task.task_id})

        if not child_assignments:
            raise RuntimeError(
                f"Policy could not assign task {task.task_id} to any worker"
            )

        best_child = next(iter(child_assignments))
        asyncio.create_task(
            self._comm.send_message_to_child(
                best_child, TaskUpdate(sender=self.node_id, added_tasks=[task])
            )
        )
        return best_child

    # --------------------------------------------------------------------------
    #                               Callbacks
    # --------------------------------------------------------------------------

    def _mark_children_done(self, child_ids: List[str]) -> None:
        """Mark children as done (runs in event loop via call_soon_threadsafe).

        Args:
            child_ids: List of child_ids for completed children.
        """
        for child_id in child_ids:
            self._scheduler.mark_child_done(child_id)
        if self._scheduler.all_children_done:
            self._all_children_done_event.set()

    def _create_done_callback(
        self, child_ids: List[str]
    ) -> Callable[[ConcurrentFuture], None]:
        """Create a done-callback that marks children complete when their future resolves.

        The callback is invoked from an executor thread and uses call_soon_threadsafe
        to safely dispatch _mark_children_done into the event loop.

        Args:
            child_ids: Child node IDs associated with the completing future.
        """

        def _done_callback(future: AsyncFuture):
            if self._event_loop is not None:
                self._event_loop.call_soon_threadsafe(
                    self._mark_children_done, child_ids
                )
            else:
                self.logger.warning("No event loop stored, can't mark child done!")

        return _done_callback

    # -------------------------------------------------------------------------
    #                               Monitors
    # -------------------------------------------------------------------------

    async def _collect_final_result_from_child(self, child_id: str) -> None:
        """Collect result and final status from a single child."""
        try:
            # Wait for result from child
            result_batch: ResultBatch = await self._comm.recv_message_from_child(
                ResultBatch, child_id=child_id, block=True
            )

            if result_batch is not None:
                self._results[child_id] = [result_batch]
                self.logger.info(f"{self.node_id}: Received result from {child_id}")
            else:
                self.logger.warning(
                    f"{self.node_id}: No result received from {child_id}"
                )
                self._results[child_id] = []

            # Collect final status from child
            # First check if we already have final status
            if self._scheduler.has_final_status(child_id):
                self.logger.debug(
                    f"{self.node_id}: Child {child_id} already has final status"
                )
            else:
                # Drain status queue to get final status
                empty_count = 0
                while empty_count < 2:
                    status = await self._comm.recv_message_from_child(
                        Status, child_id=child_id, timeout=0.01
                    )
                    if status is not None:
                        empty_count = 0  # Reset counter on successful recv
                        self._scheduler.set_child_status(child_id, status)
                        if status.tag == "final":
                            break
                    else:
                        empty_count += 1  # Increment when queue is empty

                if not self._scheduler.has_final_status(child_id):
                    self.logger.warning(
                        f"{self.node_id}: Failed to receive final status from {child_id}"
                    )
        except Exception as e:
            self.logger.error(
                f"{self.node_id}: Error collecting result from {child_id}: {e}"
            )
            self._results[child_id] = []

    async def _aggregate_and_send_results(
        self, result_tasks: List[asyncio.Task]
    ) -> None:
        """Wait for all result collection tasks, aggregate, and send to parent."""
        # Wait for all result collection tasks to complete
        await asyncio.gather(*result_tasks, return_exceptions=True)
        self.logger.info(f"{self.node_id}: All result collection tasks completed")

        # Aggregate results
        async with self._timer("aggregate_results"):
            result_batch = ResultBatch(sender=self.node_id)
            for child_id, child_results in self._results.items():
                for rb in child_results:
                    result_batch += rb
            self.logger.info(
                f"{self.node_id}: Aggregated results from {len(self._results)} children"
            )

        # Send to parent
        if self.parent:
            success = await self._comm.send_message_to_parent(result_batch)
            if not success:
                self.logger.warning(f"{self.node_id}: Failed to send results to parent")
            else:
                self.logger.info(f"{self.node_id}: Successfully sent results to parent")

        # Report final status to parent
        async with self._timer("report_to_parent"):
            if self.parent:
                final_status = self._scheduler.aggregate_status()
                final_status.tag = "final"
                success = await self._comm.send_message_to_parent(final_status)
                if not success:
                    self.logger.warning(
                        f"{self.node_id}: Failed to send final status to parent"
                    )
                else:
                    self.logger.info(
                        f"{self.node_id}: Successfully reported final status to parent"
                    )
            else:
                try:
                    status = self._scheduler.aggregate_status()
                    status.tag = "final"
                    # Write to a json file
                    fname = os.path.join(os.getcwd(), f"{self.node_id}_status.json")
                    status.to_file(fname)
                    self.logger.info(
                        f"{self.node_id}: Successfully reported final status"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"{self.node_id}: Reporting final status failed with exception {e}"
                    )

    async def _report_status(self) -> None:
        """Periodically collect status from children, aggregate, and forward to parent."""
        while not self._stop_reporting_event.is_set():
            try:
                for child_id in self.children:
                    status = await self._comm.recv_message_from_child(
                        Status, child_id=child_id
                    )
                    if status is not None:
                        self._scheduler.set_child_status(child_id, status)
                status = self._scheduler.aggregate_status()
                if self.parent:
                    await self._comm.send_message_to_parent(status)
                    self.logger.info(status)
                else:
                    self.logger.info(status)
                # Use wait with timeout so we can exit quickly when stopped
                try:
                    await asyncio.wait_for(
                        self._stop_reporting_event.wait(),
                        timeout=self._config.report_interval,
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
            item = await self._comm.recv_client_message(timeout=0.1)
            if item is None:
                continue
            client_id, msg = item
            if isinstance(msg, TaskUpdate):
                for task in msg.added_tasks:
                    self._route_task(task)
                    self._client_task_map[task.task_id] = client_id

    async def _forward_result(self, result: Result) -> None:
        """Route a result to its originating client if dynamically submitted, otherwise to parent."""
        if result.task_id in self._client_task_map:
            client_id = self._client_task_map.pop(result.task_id)
            await self._comm.send_message_to_child(client_id, result)
        elif self.parent:
            await self._comm.send_message_to_parent(result)

    async def _child_result_monitor(self, child_id: str) -> None:
        """Receive Result messages from child and forward to client or parent."""
        child_done = self._scheduler.get_done_event(child_id)
        while not child_done.is_set():
            result = await self._comm.recv_message_from_child(
                Result, child_id=child_id, block=True
            )
            if result is not None:
                await self._forward_result(result)
        # Drain remaining results after child exits
        while True:
            result = await self._comm.recv_message_from_child(Result, child_id=child_id)
            if result is None:
                break
            await self._forward_result(result)

    async def _parent_task_update_monitor(self) -> None:
        """Non-root master only: receive TaskUpdates from parent and route tasks to children."""
        while not self._stop_task_update.is_set():
            try:
                task_update = await self._comm.recv_message_from_parent(
                    TaskUpdate, block=True
                )
                if task_update is not None:
                    for task in task_update.added_tasks:
                        self._route_task(task)
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
        ##Cancel the future and wait for scheduler to mark it done
        child_fut = self._children_futures.get(child_id, None)
        if child_fut is None:
            raise RuntimeError(f"{child_id} Child future doesn't exist!")
        if not child_fut.done():
            child_fut.cancel()
        await self._scheduler.wait_for_child(child_id)

        old_task = self._child_result_task.pop(child_id, None)
        if old_task is not None:
            old_task.cancel()
            if old_task in self._result_tasks:
                self._result_tasks.remove(old_task)
        old_fwd = self._child_forwarder_task.pop(child_id, None)
        if old_fwd is not None:
            old_fwd.cancel()
        self._scheduler.remove_child(child_id)
        self.remove_child(child_id)
        self._child_objs.pop(child_id, None)
        await self._comm.update_node_info(self.info())

    async def stop(self) -> None:
        """Gracefully shut down all children, monitors, comm, and executor."""
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

        # stop global monitors
        self._stop_reporting_event.set()
        self._reporting_task.cancel()
        try:
            await self._reporting_task
        except Exception:
            pass
        self.logger.info(f"{self.node_id}: Stopped reporting loop")

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

        # stop comm and executor
        await self._comm.close()
        self._executor.shutdown()

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

    # -------------------------------------------------------------------------
    #                               Entry point
    # -------------------------------------------------------------------------

    async def _wait_for_finish(self) -> None:
        """Wait for all work to complete. Overridable by subclasses."""
        await self._aggregate_task

    async def run(self) -> ResultBatch:
        """Main entry point: initialise, wait for all work to finish, stop, return results."""
        async with self._timer("init"):
            await self._lazy_init()

        # Wait for aggregation to complete
        await self._wait_for_finish()

        await self.stop()

        # Return aggregated results
        result_batch = ResultBatch(sender=self.node_id)
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
            "parent": asdict(self.parent) if self.parent else None,
            "children": {
                child_id: asdict(child) for child_id, child in self.children.items()
            },
            "parent_comm": self.parent_comm.asdict() if self.parent_comm else None,
        }

        if include_tasks:
            raise NotImplementedError(
                "Including tasks in serialization is not implemented yet."
            )

        return obj_dict

    @classmethod
    def fromdict(cls, data: dict) -> "AsyncMaster":
        """Reconstruct an AsyncMaster from a serialised dict (inverse of asdict)."""
        config = LauncherConfig.model_validate_json(data["config"])
        parent = NodeInfo(**data["parent"]) if data["parent"] else None
        children = {
            child_id: NodeInfo(**child_dict)
            for child_id, child_dict in data["children"].items()
        }

        if config.comm_name == "async_zmq":
            # AsyncZMQComm might need special handling due to non-picklable attributes
            parent_comm = (
                AsyncZMQComm.fromdict(data["parent_comm"])
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
        return master
