import asyncio
import collections
import multiprocessing
import os
import queue
import time
import uuid
from asyncio import Queue
from logging import Logger
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel, SerializeAsAny

from ensemble_launcher.profiling import EventRegistry, get_registry

from .hb import HeartBeatProcess
from .messages import Message, all_messages

_CRITICAL_MSG_TYPES = frozenset({5, 3, 9, 8, 6})  # TaskUpdate, ResultBatch, Stop, Ready, NodeUpdate
from .nodeinfo import NodeInfo
from .pipe import (
    AsyncConnection,
    AsyncTransport,
    AsyncTransportState,
    ClientConnection,
    ClientConnectionState,
    transport_registry,
)

T = TypeVar("T", bound="AsyncCommState")


class AsyncCommState(BaseModel):
    node_info: NodeInfo
    parent_transport_type: Optional[str] = None
    child_transport_type: str = "zmq"
    parent_conn_state: Optional[SerializeAsAny[ClientConnectionState]] = None
    hb_parent_conn_state: Optional[SerializeAsAny[ClientConnectionState]] = None
    data_transport_state: Optional[SerializeAsAny[AsyncTransportState]] = None
    hb_transport_state: Optional[SerializeAsAny[AsyncTransportState]] = None
    req_res: bool = False
    send_retries: int = 3
    send_timeout: float = 5.0

    def serialize(self, *args, **kwargs) -> str:
        return self.model_dump_json(*args, **kwargs)

    @classmethod
    def deserialize(cls: Type[T], data: str) -> T:
        import json

        raw = json.loads(data)

        child_tt = raw.get("child_transport_type", "zmq")
        parent_tt = raw.get("parent_transport_type")

        child_entry = transport_registry.get(child_tt)
        parent_entry = transport_registry.get(parent_tt) if parent_tt else None

        if child_entry:
            ts_cls = child_entry.get("transport_state")
            if ts_cls:
                for key in ("data_transport_state", "hb_transport_state"):
                    nested = raw.get(key)
                    if isinstance(nested, dict):
                        raw[key] = ts_cls.model_validate(nested)

        if parent_entry:
            cs_cls = parent_entry.get("client_connection_state")
            if cs_cls:
                for key in ("parent_conn_state", "hb_parent_conn_state"):
                    nested = raw.get(key)
                    if isinstance(nested, dict):
                        raw[key] = cs_cls.model_validate(nested)

        return cls.model_validate(raw)


def _decode_identity(raw: bytes) -> Tuple[str, str, Optional[str]]:
    full_id = raw.decode()
    parts = full_id.split(":", 1)
    return full_id, parts[0], parts[1] if len(parts) > 1 else None


def _encode_identity(node_id: str, secret_id: str) -> str:
    return f"{node_id}:{secret_id}"


class AsyncMessageRoutingQueue:
    """An async routing queue that organizes messages by type using separate LifoQueues. Not thread-safe."""

    def __init__(
        self, logger: Logger, message_types: Optional[List[Type[Message]]] = None
    ):
        self.logger = logger
        self._queues: Dict[Type[Message], Queue] = {}
        self._message_types = message_types
        if message_types is not None:
            for msg_type in message_types:
                self._queues[msg_type] = Queue()

    async def put(self, message: Message):
        msg_type = type(message)
        if msg_type not in self._queues:
            self._queues[msg_type] = Queue()
            self.logger.debug(
                f"Created new queue for message type: {msg_type.__name__}"
            )
        await self._queues[msg_type].put(message)

    def put_nowait(self, message: Message):
        msg_type = type(message)
        if msg_type not in self._queues:
            self._queues[msg_type] = Queue()
            self.logger.debug(
                f"Created new queue for message type: {msg_type.__name__}"
            )
        self._queues[msg_type].put_nowait(message)

    async def get(
        self, msg_type: Type[Message], timeout: Optional[float] = None
    ) -> Optional[Message]:
        if msg_type not in self._queues:
            self.logger.warning(f"No messages of type {msg_type.__name__} available")
            return None
        try:
            self.logger.debug(
                f"Waiting for message of type {msg_type.__name__} with timeout {timeout}"
            )
            msg = await asyncio.wait_for(self._queues[msg_type].get(), timeout=timeout)
            self.logger.debug(
                f"Retrieved message of type {msg_type.__name__} with timeout {timeout}"
            )
            return msg
        except asyncio.TimeoutError:
            self.logger.debug(
                f"No messages of type {msg_type.__name__} available within timeout {timeout}s"
            )
            return None
        except asyncio.QueueEmpty:
            self.logger.debug(f"Queue of type {msg_type.__name__} is empty")
            return None

    def get_nowait(self, msg_type: Type[Message]) -> Optional[Message]:
        if msg_type not in self._queues:
            self.logger.warning(f"No messages of type {msg_type.__name__} available")
            return None
        try:
            msg = self._queues[msg_type].get_nowait()
            self.logger.debug(
                f"Retrieved message of type {msg_type.__name__} without blocking"
            )
            return msg
        except asyncio.QueueEmpty:
            self.logger.debug(
                f"No messages of type {msg_type.__name__} available in queue"
            )
            return None

    def clear(self, msg_type: Optional[Type[Message]] = None):
        if msg_type is not None:
            if msg_type in self._queues:
                try:
                    while True:
                        self._queues[msg_type].get_nowait()
                except asyncio.QueueEmpty:
                    pass
        else:
            for queue_obj in self._queues.values():
                try:
                    while True:
                        queue_obj.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queues.clear()

    def empty(self, msg_type: Optional[Type[Message]] = None) -> bool:
        if msg_type is not None:
            if msg_type not in self._queues:
                self.logger.warning(
                    f"No messages of type {msg_type.__name__} available"
                )
                return True
            return self._queues[msg_type].empty()
        else:
            return all(queue_obj.empty() for queue_obj in self._queues.values())


class AsyncComm:
    def __init__(
        self,
        logger: Logger,
        node_info: NodeInfo,
        parent_conn: Optional[ClientConnection] = None,
        hb_parent_conn: Optional[ClientConnection] = None,
        child_transport: str = "zmq",
        heartbeat_interval: float = 1.0,
        heartbeat_dead_threshold: float = 30.0,
        cluster_secret: Optional[str] = None,
        skip_hb: bool = False,
        req_res: bool = False,
        send_retries: int = 3,
        send_timeout: float = 5.0,
    ):
        self.logger = logger
        self._skip_hb = skip_hb
        self._node_info = node_info
        self._cluster_secret = cluster_secret
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_conn = parent_conn
        self._hb_parent_conn = hb_parent_conn
        self._child_transport = child_transport
        self._req_res = req_res
        self._send_retries = send_retries
        self._send_timeout = send_timeout
        self._seen_message_ids: collections.OrderedDict = collections.OrderedDict()
        self._seen_message_ids_max: int = 10000

        entry = transport_registry.get(child_transport)
        if entry is None:
            raise ValueError(
                f"Unknown transport: {child_transport!r}. "
                f"Available: {transport_registry.available}"
            )
        transport_cls = entry["transport"]
        self._data_transport: AsyncTransport = transport_cls()
        try:
            self._hb_transport: AsyncTransport = transport_cls(lightweight=True)
        except TypeError:
            self._hb_transport: AsyncTransport = transport_cls()

        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self._recv_tasks: Dict[int, asyncio.Task] = {}

        self._cache: Dict[str, AsyncMessageRoutingQueue] = {}
        self._stop_event = asyncio.Event()
        self._client_queue: asyncio.Queue = asyncio.Queue()

        self._parent_monitor_started = False
        self._child_monitor_started = False
        self._monitor_tasks: List[asyncio.Task] = []

        self.parent_dead_event: Optional[asyncio.Event] = asyncio.Event()
        self._child_dead_events: Dict[str, asyncio.Event] = {}

        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry: EventRegistry = get_registry()

        # Heartbeat state
        self._hb_process: Optional[multiprocessing.Process] = None
        self._hb_control_queue: Optional[multiprocessing.Queue] = None
        self._hb_notify_queue: Optional[multiprocessing.Queue] = None
        self._hb_watcher_task: Optional[asyncio.Task] = None
        self._hb_process_started: bool = False

        self._hb_parent_ready: Optional[asyncio.Event] = asyncio.Event()
        self._hb_child_ready: Dict[str, asyncio.Event] = {}
        self._heartbeat_dead_threshold: float = heartbeat_dead_threshold
        self.update_node_info(node_info=node_info)

    # -----------------------------------------------------------------
    #  Properties
    # -----------------------------------------------------------------

    @property
    def my_address(self) -> Optional[str]:
        for conn in self._data_transport.get_server_connections():
            if conn.address is not None:
                return conn.address
        return None

    @property
    def parent_address(self) -> Optional[str]:
        if self._parent_conn is None:
            return None
        return getattr(self._parent_conn, "remote_address", None)

    @property
    def my_hb_address(self) -> Optional[str]:
        for conn in self._hb_transport.get_server_connections():
            if conn.address is not None:
                return conn.address
        return None

    @property
    def parent_hb_address(self) -> Optional[str]:
        if self._hb_parent_conn is None:
            return None
        return self._hb_parent_conn.remote_address

    @property
    def data_transport(self) -> AsyncTransport:
        return self._data_transport

    @property
    def hb_transport(self) -> AsyncTransport:
        return self._hb_transport

    # -----------------------------------------------------------------
    #  Child pipe creation
    # -----------------------------------------------------------------

    def _make_client_validator(self):
        cluster_secret = self._cluster_secret

        def validator(sender_id: str, sender_secret: Optional[str]) -> bool:
            if not sender_id.startswith("client-"):
                return False
            if cluster_secret is not None:
                return sender_secret == cluster_secret
            return True

        return validator

    async def create_child_pipe(
        self, child_id: str, child_secret_id: str
    ) -> Tuple[ClientConnection, ClientConnection]:
        known_data_conns = set(
            id(c) for c in self._data_transport.get_server_connections()
        )
        data_server, data_client = self._data_transport.create_child_pipe(
            self._node_info.node_id,
            self._node_info.secret_id,
            child_id,
            child_secret_id,
            req_res=self._req_res,
        )
        data_server.set_unknown_sender_validator(self._make_client_validator())
        if id(data_server) not in known_data_conns:
            if not data_server.is_open:
                await data_server.open()
            self._monitor_tasks.append(
                asyncio.create_task(self._monitor_children_data_conn(data_server))
            )

        known_hb_conns = set(id(c) for c in self._hb_transport.get_server_connections())
        hb_server, hb_client = self._hb_transport.create_child_pipe(
            self._node_info.node_id,
            self._node_info.secret_id,
            child_id,
            child_secret_id,
        )
        if id(hb_server) not in known_hb_conns and self._hb_control_queue is not None:
            self._hb_control_queue.put(("add_server_connection", hb_server))
        return data_client, hb_client

    # -----------------------------------------------------------------
    #  Cache management
    # -----------------------------------------------------------------

    def update_node_info(self, node_info: NodeInfo):
        self._node_info = node_info
        ## Add children
        for child_id in node_info.children_ids:
            if child_id not in self._hb_child_ready:
                self._hb_child_ready[child_id] = asyncio.Event()
                self._child_dead_events[child_id] = asyncio.Event()
                self._cache[child_id] = AsyncMessageRoutingQueue(
                    logger=self.logger, message_types=all_messages
                )
                if self._hb_control_queue is not None:
                    child_secret = node_info.children_secret_ids[child_id]
                    self._hb_control_queue.put(("add_child", child_id, child_secret))
                elif self._hb_process_started:
                    self._hb_child_ready[child_id].set()

        # Remove children
        child_ids = set(self._hb_child_ready.keys())
        for child_id in child_ids:
            if child_id not in node_info.children_ids:
                self._hb_child_ready.pop(child_id, None)
                self._child_dead_events.pop(child_id, None)
                self._cache.pop(child_id, None)
                if self._hb_control_queue is not None:
                    self._hb_control_queue.put(("remove_child", child_id))

    async def clear_cache(self):
        for routing_queue in self._cache.values():
            routing_queue.clear()
        self._cache.clear()

    # -----------------------------------------------------------------
    #  req_res helpers
    # -----------------------------------------------------------------

    def _assign_message_id(self, msg: Message) -> None:
        if self._req_res and msg.message_id is None:
            msg.message_id = uuid.uuid4().hex

    def _is_duplicate(self, msg: Message) -> bool:
        if not self._req_res or msg.message_id is None:
            return False
        if msg.message_id in self._seen_message_ids:
            return True
        self._seen_message_ids[msg.message_id] = None
        if len(self._seen_message_ids) > self._seen_message_ids_max:
            self._seen_message_ids.popitem(last=False)
        return False

    async def _send_with_retry(
        self,
        send_coro_factory: Callable,
        description: str,
        msg: Message = None,
    ) -> bool:
        use_retry = self._req_res and (
            msg is None or msg.MSG_TYPE_ID in _CRITICAL_MSG_TYPES
        )
        if not use_retry:
            return await send_coro_factory()

        for attempt in range(self._send_retries):
            success = await send_coro_factory()
            if success:
                return True
            self.logger.warning(
                f"{self._node_info.node_id}: {description} failed attempt "
                f"{attempt + 1}/{self._send_retries}"
            )
        self.logger.error(
            f"{self._node_info.node_id}: {description} failed after "
            f"{self._send_retries} retries"
        )
        return False

    # -----------------------------------------------------------------
    #  Monitors
    # -----------------------------------------------------------------

    async def start_monitors(self, **kwargs):

        if self._parent_conn is not None and not self._parent_conn.is_open:
            await self._parent_conn.open()
            if self._node_info.parent_id not in self._cache:
                self._cache[self._node_info.parent_id] = AsyncMessageRoutingQueue(
                    logger=self.logger, message_types=all_messages
                )
            self._monitor_tasks.append(
                asyncio.create_task(self._monitor_parent_data_conn())
            )
        if self._node_info.parent_id is not None:
            self.logger.info(
                f"{self._node_info.node_id}: Connected to parent at {self.parent_address}"
            )

        # Eagerly create the local server connections (ZMQ supports this;
        # MP transport does not, so we guard with try/except).
        try:
            data_conn = self._data_transport.get_server_connection(
                self._node_info.node_id, self._node_info.secret_id,
                req_res=self._req_res,
            )
            if not data_conn.is_open:
                await data_conn.open()
                self.logger.info(
                    f"{self._node_info.node_id}: Data server connection bound to {self.my_address}"
                )
                self._monitor_tasks.append(
                    asyncio.create_task(self._monitor_children_data_conn(data_conn))
                )
        except NotImplementedError:
            pass
        # HB process (only for transports that support standalone server connections)
        if not self._hb_process_started and (self._hb_transport.transport_type == "mp" or self._skip_hb):
            self.logger.warning(
                f"{self._node_info.node_id}: MP transport does not support HB process, "
                "setting all HB events as ready"
            )
            if self._hb_parent_ready is not None:
                self._hb_parent_ready.set()
            for ev in self._hb_child_ready.values():
                ev.set()
            self._hb_process_started = True

        if not self._hb_process_started and self._hb_transport.transport_type != "mp":
            initial_children = {}
            for child_id in self._node_info.children_ids:
                secret = self._node_info.children_secret_ids.get(child_id)
                initial_children[child_id] = secret
            self._hb_transport.get_server_connection(
                self._node_info.node_id, self._node_info.secret_id
            )
            hb_server_conns = self._hb_transport.get_server_connections()
            self._hb_control_queue = multiprocessing.Queue()
            self._hb_notify_queue = multiprocessing.Queue()
            hb_proc = HeartBeatProcess(
                node_id=self._node_info.node_id,
                secret_id=self._node_info.secret_id,
                parent_id=self._node_info.parent_id,
                hb_parent_conn=self._hb_parent_conn,
                hb_server_conns=hb_server_conns,
                heartbeat_interval=self.heartbeat_interval,
                heartbeat_dead_threshold=self._heartbeat_dead_threshold,
                control_queue=self._hb_control_queue,
                notify_queue=self._hb_notify_queue,
                initial_children=initial_children if initial_children else None,
            )
            self._hb_process = multiprocessing.Process(
                target=hb_proc,
                daemon=True,
                name=f"hb-{self._node_info.node_id}",
            )
            self._hb_process.start()
            loop = asyncio.get_running_loop()
            num_expected_addrs = len(hb_server_conns)
            for _ in range(num_expected_addrs):
                try:
                    msg = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, self._hb_notify_queue.get, True, 10.0
                        ),
                        timeout=15.0,
                    )
                    if msg[0] == "hb_address":
                        actual_addr = msg[1]
                        for conn in self._hb_transport.get_server_connections():
                            conn._address = actual_addr
                        self.logger.info(
                            f"{self._node_info.node_id}: HB process bound to {actual_addr}"
                        )
                except (asyncio.TimeoutError, queue.Empty):
                    self.logger.warning(
                        f"{self._node_info.node_id}: HB process did not report bound address"
                    )
            self._hb_watcher_task = asyncio.create_task(self._hb_watcher())
            self._hb_process_started = True
        self.logger.info(f"Start monitors done {self._node_info.node_id}")

    # ------------------------------------------------------------------ #
    # HB process watcher                                                  #
    # ------------------------------------------------------------------ #

    async def _hb_watcher(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._stop_event.is_set():
            try:
                msg = await loop.run_in_executor(
                    None, self._hb_notify_queue.get, True, 0.5
                )
            except queue.Empty:
                if self._hb_process is not None and not self._hb_process.is_alive():
                    self.logger.warning(
                        f"{self._node_info.node_id}: HB process died, setting all dead events"
                    )
                    if self.parent_dead_event is not None:
                        self.parent_dead_event.set()
                    for ev in self._child_dead_events.values():
                        ev.set()
                    break
                continue
            except Exception:
                break

            kind = msg[0]
            self.logger.debug(
                f"{self._node_info.node_id}: HB watcher received {msg}, "
                f"process alive={self._hb_process.is_alive() if self._hb_process else 'N/A'}"
            )
            if kind == "ready_child":
                child_id = msg[1]
                ev = self._hb_child_ready.get(child_id)
                if ev is not None and not ev.is_set():
                    ev.set()
            elif kind == "dead_child":
                child_id = msg[1]
                self.logger.warning(
                    f"{self._node_info.node_id}: HB watcher received dead_child for {child_id}"
                )
                ev = self._child_dead_events.get(child_id)
                if ev is not None:
                    ev.set()
                else:
                    self.logger.warning(
                        f"{self._node_info.node_id}: No dead event for child {child_id}"
                    )
            elif kind == "ready_parent":
                if (
                    self._hb_parent_ready is not None
                    and not self._hb_parent_ready.is_set()
                ):
                    self._hb_parent_ready.set()
            elif kind == "dead_parent":
                if self.parent_dead_event is not None:
                    self.parent_dead_event.set()
            elif kind == "hb_address":
                actual_addr = msg[1]
                for conn in self._hb_transport.get_server_connections():
                    conn._address = actual_addr
                self.logger.info(
                    f"{self._node_info.node_id}: HB address updated to {actual_addr}"
                )

    # ------------------------------------------------------------------ #
    # Sync heartbeat                                                     #
    # ------------------------------------------------------------------ #

    async def sync_heartbeat_with_parent(self, timeout: Optional[float] = None) -> bool:
        if self._node_info.parent_id is None:
            return True
        if self._hb_parent_ready is None:
            return True
        try:
            await asyncio.wait_for(self._hb_parent_ready.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def sync_heartbeat_with_child(
        self, child_id: str, timeout: Optional[float] = None
    ) -> bool:
        ev = self._hb_child_ready.get(child_id)
        if ev is None:
            return True
        try:
            await asyncio.wait_for(ev.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ------------------------------------------------------------------ #
    # Deserialization + dispatch                                          #
    # ------------------------------------------------------------------ #

    async def _deserialize_and_dispatch_parent(
        self, raw_data: list, loop: asyncio.AbstractEventLoop, parent_id: str
    ) -> None:
        try:
            data_frames = raw_data[1:]
            if len(data_frames) == 1:
                msg = Message.from_bytes(data_frames[0])
            else:
                msg = Message.from_byte_array(data_frames)
            if self._is_duplicate(msg):
                self.logger.debug(
                    f"{self._node_info.node_id}: Dropping duplicate {msg.message_id} from parent"
                )
                return
            self._cache[parent_id].put_nowait(msg)
            self.logger.debug(
                f"{self._node_info.node_id}: Cached message from parent: {type(msg).__name__}"
            )
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Failed to deserialize message from parent: {e}"
            )

    async def _deserialize_and_dispatch_child(self, raw_data: list) -> None:
        full_id, sender_id, _ = _decode_identity(raw_data[0])
        try:
            data_frames = raw_data[1:]
            if len(data_frames) == 1:
                msg = Message.from_bytes(data_frames[0])
            else:
                msg = Message.from_byte_array(data_frames)
            if self._is_duplicate(msg):
                self.logger.debug(
                    f"{self._node_info.node_id}: Dropping duplicate {msg.message_id} from child {sender_id}"
                )
                return
            if sender_id.startswith("client-"):
                self._client_queue.put_nowait((full_id, msg))
                self.logger.debug(
                    f"{self._node_info.node_id}: Queued client message from {full_id}: {type(msg).__name__}"
                )
                return
            self._cache[sender_id].put_nowait(msg)
            self.logger.debug(
                f"{self._node_info.node_id}: Cached message from child {sender_id}: {type(msg).__name__}"
            )
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Failed to deserialize message from child {sender_id}: {e}"
            )

    # ------------------------------------------------------------------ #
    # Monitors                                                            #
    # ------------------------------------------------------------------ #

    async def _monitor_parent_data_conn(self) -> None:
        if self._parent_conn is None:
            return
        self._parent_monitor_started = True
        parent_id = self._node_info.parent_id
        loop = asyncio.get_running_loop()
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self._parent_conn.recv()
                failures = 0
                asyncio.create_task(
                    self._deserialize_and_dispatch_parent(raw_data, loop, parent_id)
                )
            except Exception as e:
                failures += 1
                self.logger.warning(
                    f"{self._node_info.node_id}: Error receiving from parent failed {failures} times: {e}"
                )
                await asyncio.sleep(0.01)

    async def _monitor_children_data_conn(self, conn: AsyncConnection) -> None:
        self._child_monitor_started = True
        while not self._stop_event.is_set():
            try:
                raw_data = await conn.recv()
                asyncio.create_task(self._deserialize_and_dispatch_child(raw_data))
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(
                    f"{self._node_info.node_id}: Error dispatching child message: {e}"
                )

    # ------------------------------------------------------------------ #
    # Send / recv                                                         #
    # ------------------------------------------------------------------ #

    async def send_message_to_parent(self, msg: Message) -> bool:
        if self._node_info.parent_id is None:
            self.logger.warning(
                f"{self._node_info.node_id}: No parent connection available"
            )
            return False
        try:
            self._assign_message_id(msg)
            packed = msg.to_byte_array()

            async def _do_send():
                return await self._parent_conn.send(packed, timeout=self._send_timeout)

            success = await self._send_with_retry(
                _do_send, f"send {type(msg).__name__} to parent", msg=msg
            )
            if success:
                self.logger.debug(
                    f"{self._node_info.node_id}: Sent message to parent: {type(msg).__name__}"
                )
            return success
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Sending message to parent failed with {e}"
            )
            return False

    async def recv_message_from_parent(
        self,
        cls: Type[Message],
        block: bool = False,
        timeout: Optional[float] = None,
        unpack: bool = False,
    ) -> Message | None:
        parent_id = self._node_info.parent_id
        if parent_id is None or parent_id not in self._cache:
            self.logger.warning("No parent available to receive message from.")
            return None
        if block is False and timeout is None:
            msg = self._cache[parent_id].get_nowait(cls)
        else:
            msg = await self._cache[parent_id].get(cls, timeout=timeout)
        if msg is not None and unpack:
            await asyncio.get_running_loop().run_in_executor(None, msg.unpack)
        return msg

    async def send_message_to_child(self, child_id: str, msg: Message) -> bool:
        if (
            not child_id.startswith("client-")
            and child_id not in self._node_info.children_ids
        ):
            self.logger.error(
                f"{self._node_info.node_id}: No connection to child {child_id}"
            )
            raise RuntimeError(f"No connection to child {child_id}")
        try:
            self._assign_message_id(msg)
            packed = msg.to_byte_array()
            conn = self._data_transport.get_server_connection(
                self._node_info.node_id, self._node_info.secret_id
            )
            if conn is None:
                raise RuntimeError(f"No server connection for {child_id}")
            if child_id.startswith("client-"):
                target_id = child_id
            else:
                target_id = _encode_identity(
                    child_id, self._node_info.children_secret_ids[child_id]
                )

            async def _do_send():
                return await conn.send(packed, target_id, timeout=self._send_timeout)

            success = await self._send_with_retry(
                _do_send, f"send {type(msg).__name__} to child {child_id}", msg=msg
            )
            if success:
                self.logger.debug(
                    f"{self._node_info.node_id}: Sent {type(msg).__name__} to child {target_id}"
                )
            return success
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Sending message to child {child_id} failed with {e}"
            )
            return False

    async def recv_message_from_child(
        self,
        cls: Type[Message],
        child_id: str,
        block: bool = False,
        timeout: Optional[float] = None,
        unpack: bool = False,
    ) -> Message | None:
        if child_id not in self._cache:
            self.logger.warning(
                f"{child_id} not in cache. Current keys {self._cache.keys()}"
            )
            return None
        if block is False and timeout is None:
            msg = self._cache[child_id].get_nowait(cls)
        else:
            msg = await self._cache[child_id].get(cls, timeout=timeout)
        if msg is not None and unpack:
            await asyncio.get_running_loop().run_in_executor(None, msg.unpack)
        return msg

    async def recv_client_message(
        self, timeout: Optional[float] = None
    ) -> Optional[Tuple[str, "Message"]]:
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._client_queue.get(), timeout=timeout)
            return await self._client_queue.get()
        except asyncio.TimeoutError:
            return None

    # ------------------------------------------------------------------ #
    # State / serialization                                               #
    # ------------------------------------------------------------------ #

    def get_state(self) -> AsyncCommState:
        return AsyncCommState(
            node_info=self._node_info,
            parent_transport_type=self._parent_conn.transport_type
            if self._parent_conn
            else None,
            child_transport_type=self._data_transport.transport_type,
            parent_conn_state=self._parent_conn.get_state()
            if self._parent_conn
            else None,
            hb_parent_conn_state=self._hb_parent_conn.get_state()
            if self._hb_parent_conn
            else None,
            data_transport_state=self._data_transport.get_state(),
            hb_transport_state=self._hb_transport.get_state(),
            req_res=self._req_res,
            send_retries=self._send_retries,
            send_timeout=self._send_timeout,
        )

    @classmethod
    def set_state(cls, state: AsyncCommState) -> "AsyncComm":
        parent_conn = None
        hb_parent_conn = None

        if state.parent_transport_type and state.parent_conn_state is not None:
            parent_entry = transport_registry[state.parent_transport_type]
            parent_conn = parent_entry["client_connection"].set_state(
                state.parent_conn_state
            )

        if state.parent_transport_type and state.hb_parent_conn_state is not None:
            parent_entry = transport_registry[state.parent_transport_type]
            hb_parent_conn = parent_entry["client_connection"].set_state(
                state.hb_parent_conn_state
            )

        child_tt = state.child_transport_type
        child_entry = transport_registry[child_tt]

        ret = cls(
            logger=None,
            node_info=state.node_info,
            parent_conn=parent_conn,
            hb_parent_conn=hb_parent_conn,
            child_transport=child_tt,
            req_res=state.req_res,
            send_retries=state.send_retries,
            send_timeout=state.send_timeout,
        )
        if state.data_transport_state is not None:
            ret._data_transport = child_entry["transport"].set_state(
                state.data_transport_state
            )
        if state.hb_transport_state is not None:
            ret._hb_transport = child_entry["transport"].set_state(
                state.hb_transport_state
            )
        return ret

    # ------------------------------------------------------------------ #
    # Cleanup                                                             #
    # ------------------------------------------------------------------ #

    async def close(self):
        self._stop_event.set()

        for t in self._recv_tasks.values():
            if not t.done():
                t.cancel()
        if self._recv_tasks:
            await asyncio.gather(*self._recv_tasks.values(), return_exceptions=True)
        self._recv_tasks.clear()

        for t in self._monitor_tasks:
            if not t.done():
                t.cancel()
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        self._monitor_tasks.clear()

        if self._hb_watcher_task is not None and not self._hb_watcher_task.done():
            self._hb_watcher_task.cancel()
            try:
                await self._hb_watcher_task
            except asyncio.CancelledError:
                pass

        if self._hb_control_queue is not None:
            try:
                self._hb_control_queue.put(("stop",))
            except Exception:
                pass
        if self._hb_process is not None:
            self._hb_process.join(timeout=5.0)
            if self._hb_process.is_alive():
                self._hb_process.terminate()
                self._hb_process.join(timeout=2.0)
            if self._hb_process.is_alive():
                self._hb_process.kill()
                self._hb_process.join(timeout=1.0)

        for q in (self._hb_control_queue, self._hb_notify_queue):
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._hb_control_queue = None
        self._hb_notify_queue = None

        self.logger.info("Stopped HB process")

        await self.clear_cache()
        try:
            if self._parent_conn and self._parent_conn.is_open:
                await self._parent_conn.close()
            for conn in self._data_transport.get_server_connections():
                if conn.is_open:
                    await conn.close()
            for conn in self._hb_transport.get_server_connections():
                if conn.is_open:
                    await conn.close()
            for conn in self._data_transport.get_client_connections():
                if conn.is_open:
                    await conn.close()
            for conn in self._hb_transport.get_client_connections():
                if conn.is_open:
                    await conn.close()
        except Exception as e:
            self.logger.warning(f"{self._node_info.node_id}: Error during cleanup: {e}")
        self.logger.info("Done stopping comm")
