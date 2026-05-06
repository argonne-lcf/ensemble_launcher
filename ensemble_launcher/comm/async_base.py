import asyncio
import os
import random
import threading
import time
from asyncio import Queue
from logging import Logger
from typing import Dict, List, Optional, Tuple, Type, TypeVar

import cloudpickle
from pydantic import BaseModel, SerializeAsAny

from ensemble_launcher.profiling import EventRegistry, get_registry

from .messages import Message, all_messages
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


_HB_PING = b"\x01"


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
    ):
        self.logger = logger
        self._node_info = node_info
        self._cluster_secret = cluster_secret
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_conn = parent_conn
        self._hb_parent_conn = hb_parent_conn
        self._child_transport = child_transport

        entry = transport_registry.get(child_transport)
        if entry is None:
            raise ValueError(
                f"Unknown transport: {child_transport!r}. "
                f"Available: {transport_registry.available}"
            )
        transport_cls = entry["transport"]
        self._data_transport: AsyncTransport = transport_cls()
        self._hb_transport: AsyncTransport = transport_cls()

        self._recv_queue: asyncio.Queue = asyncio.Queue()
        self._recv_tasks: Dict[int, asyncio.Task] = {}

        self._hb_recv_queue: asyncio.Queue = asyncio.Queue()
        self._hb_recv_tasks: Dict[int, asyncio.Task] = {}

        self._cache: Dict[str, AsyncMessageRoutingQueue] = {}
        self._stop_event = None
        self._client_queue: asyncio.Queue = asyncio.Queue()

        self._parent_monitor_started = False
        self._child_monitor_started = False
        self._monitor_tasks: List[asyncio.Task] = []

        self.parent_dead_event: Optional[asyncio.Event] = None
        self._child_dead_events: Dict[str, asyncio.Event] = {}

        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry: EventRegistry = get_registry()

        # Heartbeat state
        self._parent_hb_thread: Optional[threading.Thread] = None
        self._parent_hb_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._parent_hb_asyncio_stop: Optional[asyncio.Event] = None
        self._parent_hb_started: bool = False

        self._children_hb_thread: Optional[threading.Thread] = None
        self._children_hb_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._children_hb_asyncio_stop: Optional[asyncio.Event] = None
        self._children_hb_started: bool = False

        self._last_parent_hb_time: Optional[float] = None
        self._last_child_hb_time: Dict[str, float] = {}
        self._hb_parent_ready: Optional[asyncio.Event] = None
        self._hb_child_ready: Dict[str, asyncio.Event] = {}
        self._heartbeat_dead_threshold: float = heartbeat_dead_threshold

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
        return self._parent_conn.remote_address

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

    def create_child_pipe(
        self, child_id: str, child_secret_id: str
    ) -> Tuple[ClientConnection, ClientConnection]:
        data_server, data_client = self._data_transport.create_child_pipe(
            self._node_info.node_id,
            self._node_info.secret_id,
            child_id,
            child_secret_id,
        )
        data_server.set_unknown_sender_validator(self._make_client_validator())
        _, hb_client = self._hb_transport.create_child_pipe(
            self._node_info.node_id,
            self._node_info.secret_id,
            child_id,
            child_secret_id,
        )
        return data_client, hb_client

    # -----------------------------------------------------------------
    #  Cache management
    # -----------------------------------------------------------------

    async def init_cache(self):
        for child_id in self._node_info.children_ids:
            if child_id in self._cache:
                continue
            self.logger.info(f"Initializing cache for child_id: {child_id}")
            self._cache[child_id] = AsyncMessageRoutingQueue(
                logger=self.logger, message_types=all_messages
            )

        if self._node_info.parent_id and self._node_info.parent_id not in self._cache:
            self.logger.info(
                f"Initializing cache for parent_id: {self._node_info.parent_id}"
            )
            self._cache[self._node_info.parent_id] = AsyncMessageRoutingQueue(
                logger=self.logger, message_types=all_messages
            )

    async def update_node_info(self, node_info: NodeInfo):
        added_children = set(node_info.children_ids) - set(self._node_info.children_ids)
        removed_children = set(self._node_info.children_ids) - set(
            node_info.children_ids
        )

        for child_id in added_children:
            self._child_dead_events[child_id] = asyncio.Event()
            self._hb_child_ready[child_id] = asyncio.Event()
            self._last_child_hb_time[child_id] = None
            child_secret = node_info.children_secret_ids.get(child_id)
            if child_secret:
                for conn in self._data_transport.get_server_connections():
                    conn.add_expected_remote(child_id, child_secret)
                for conn in self._hb_transport.get_server_connections():
                    conn.add_expected_remote(child_id, child_secret)

        for child_id in removed_children:
            self._hb_child_ready.pop(child_id, None)
            self._last_child_hb_time.pop(child_id, None)
            self._child_dead_events.pop(child_id, None)
            self._cache.pop(child_id, None)
            for conn in self._data_transport.get_server_connections():
                conn.remove_expected_remote(child_id)
            for conn in self._hb_transport.get_server_connections():
                conn.remove_expected_remote(child_id)

        self._node_info = node_info
        await self.init_cache()

    async def clear_cache(self):
        for routing_queue in self._cache.values():
            routing_queue.clear()
        self._cache.clear()

    # -----------------------------------------------------------------
    #  Monitors
    # -----------------------------------------------------------------

    async def start_monitors(self, **kwargs):
        await self.init_cache()
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

        if self._parent_conn is not None and not self._parent_conn.is_open:
            await self._parent_conn.open()
            self.logger.info(
                f"{self._node_info.node_id}: Connected to parent at {self.parent_address}"
            )

        if self._parent_conn is not None:
            self.logger.info(f"My hb secret: {self._hb_parent_conn._secret_id}")
            self.logger.info(f"My data secret: {self._parent_conn._secret_id}")

        if self._hb_transport is not None:
            self.logger.info(
                f"Expected hb remotes: {[conn.expected_remotes for conn in self._hb_transport.get_server_connections()]}"
            )
            self.logger.info(
                f"Expected data remotes: {[conn.expected_remotes for conn in self._data_transport.get_server_connections()]}"
            )

        parent_only = kwargs.get("parent_only", False)
        children_only = kwargs.get("children_only", False)

        if not children_only:
            if (
                self._node_info.parent_id is not None
                and not self._parent_monitor_started
            ):
                self._monitor_tasks.append(asyncio.create_task(self._monitor_parent()))
                self._parent_monitor_started = True

        if not parent_only:
            for conn in self._data_transport.get_server_connections():
                if not conn.is_open:
                    await conn.open()
                    self.logger.info(
                        f"{self._node_info.node_id}: Data connection bound to {self.my_address}"
                    )

                if id(conn) not in self._recv_tasks:
                    self._recv_tasks[id(conn)] = asyncio.create_task(
                        self._recv_loop(conn)
                    )
            if not self._child_monitor_started:
                self._monitor_tasks.append(
                    asyncio.create_task(self._monitor_children())
                )
                self._child_monitor_started = True

        main_loop = asyncio.get_running_loop()

        # Parent HB thread
        if (
            not children_only
            and self._hb_parent_conn is not None
            and not self._parent_hb_started
        ):
            self._parent_hb_started = True
            if self._node_info.parent_id:
                self.parent_dead_event = asyncio.Event()
                self._hb_parent_ready = asyncio.Event()
            self._parent_hb_thread = threading.Thread(
                target=self._parent_hb_thread_main,
                args=(main_loop,),
                daemon=True,
                name=f"hb-parent-{self._node_info.node_id}",
            )
            self._parent_hb_thread.start()

        # Children HB thread
        if not parent_only and not self._children_hb_started:
            self._children_hb_started = True
            for child_id in self._node_info.children_ids:
                self._child_dead_events[child_id] = asyncio.Event()
                self._hb_child_ready[child_id] = asyncio.Event()
                self._last_child_hb_time[child_id] = None

            addr_q: asyncio.Queue = asyncio.Queue()
            self._children_hb_thread = threading.Thread(
                target=self._children_hb_thread_main,
                args=(main_loop, addr_q),
                daemon=True,
                name=f"hb-children-{self._node_info.node_id}",
            )
            self._children_hb_thread.start()

            try:
                actual_hb_addr = await asyncio.wait_for(addr_q.get(), timeout=10.0)
                if actual_hb_addr is not None:
                    self.logger.info(
                        f"{self._node_info.node_id}: Children HB thread bound to {actual_hb_addr}"
                    )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"{self._node_info.node_id}: Children HB thread did not report bound address"
                )

    # ------------------------------------------------------------------ #
    # HB threads                                                          #
    # ------------------------------------------------------------------ #

    def _parent_hb_thread_main(self, main_loop: asyncio.AbstractEventLoop) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._parent_hb_thread_loop = loop
        try:
            loop.run_until_complete(self._parent_hb_coroutine(main_loop))
        finally:
            loop.close()
            self._parent_hb_thread_loop = None

    async def _parent_hb_coroutine(self, main_loop: asyncio.AbstractEventLoop) -> None:
        stop = asyncio.Event()
        self._parent_hb_asyncio_stop = stop

        hb_conn = self._hb_parent_conn
        await hb_conn.open()
        self.logger.info(f"Connected hb thread to {hb_conn.remote_address}")

        self._last_parent_hb_time = time.time()
        try:
            while not stop.is_set():
                try:
                    await hb_conn.send(_HB_PING)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(
                        f"{self._node_info.node_id}:{self._node_info.secret_id}: HB send error: {e}"
                    )

                try:
                    raw = await asyncio.wait_for(hb_conn.recv(), timeout=1.0)
                    if raw is not None and len(raw) >= 2 and raw[1] == _HB_PING:
                        if (
                            self._hb_parent_ready is not None
                            and not self._hb_parent_ready.is_set()
                        ):
                            try:
                                main_loop.call_soon_threadsafe(self._hb_parent_ready.set)
                            except RuntimeError:
                                pass
                        self._last_parent_hb_time = time.time()
                except (asyncio.TimeoutError, TimeoutError):
                    # Parent didn't respond in time. Pass so the threshold logic can evaluate.
                    pass 
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    # If the parent abruptly dies, recv() might raise EOFError or ConnectionResetError
                    self.logger.warning(
                        f"{self._node_info.node_id}:{self._node_info.secret_id}: HB recv error: {e}"
                    )

                if (
                    time.time() - self._last_parent_hb_time
                    > self._heartbeat_dead_threshold
                ):
                    self.logger.warning(
                        f"{self._node_info.node_id}: Parent HB dead — setting parent_dead_event"
                    )
                    if self.parent_dead_event is not None:
                        try:
                            main_loop.call_soon_threadsafe(self.parent_dead_event.set)
                        except RuntimeError:
                            pass
                    break
                jitter = self.heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
                await asyncio.sleep(jitter)
        finally:
            await hb_conn.close()

    def _children_hb_thread_main(
        self, main_loop: asyncio.AbstractEventLoop, addr_q: asyncio.Queue
    ) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._children_hb_thread_loop = loop
        try:
            loop.run_until_complete(self._children_hb_coroutine(main_loop, addr_q))
        finally:
            loop.close()
            self._children_hb_thread_loop = None

    async def _children_hb_coroutine(
        self, main_loop: asyncio.AbstractEventLoop, addr_q: asyncio.Queue
    ) -> None:
        stop = asyncio.Event()
        self._children_hb_asyncio_stop = stop

        for conn in self._hb_transport.get_server_connections():
            if not conn.is_open:
                await conn.open()

        actual_hb_address = self.my_hb_address
        main_loop.call_soon_threadsafe(addr_q.put_nowait, actual_hb_address)

        recv_tasks = {}
        for conn in self._hb_transport.get_server_connections():
            recv_tasks[id(conn)] = asyncio.create_task(self._hb_recv_loop(conn, stop))

        async def dispatch():
            while not stop.is_set():
                try:
                    parts = await self._hb_recv_queue.get()
                    full_id, sender_id, _ = _decode_identity(parts[0])
                    hb_conn = self._hb_transport.get_server_connection(
                        self._node_info.node_id, self._node_info.secret_id
                    )
                    if hb_conn is None:
                        self.logger.warning(
                            f"{self._node_info.node_id}: HB from unknown identity {full_id}, ignoring"
                        )
                        continue
                    self._last_child_hb_time[sender_id] = time.time()
                    ev = self._hb_child_ready.get(sender_id)
                    if ev is not None and not ev.is_set():
                        try:
                            main_loop.call_soon_threadsafe(ev.set)
                        except RuntimeError:
                            pass
                    await hb_conn.send(_HB_PING, full_id)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.warning(
                        f"{self._node_info.node_id}: HB dispatch error: {e}"
                    )

        async def dead_check():
            while not stop.is_set():
                jitter = self.heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
                await asyncio.sleep(jitter)
                for child_id, last in list(self._last_child_hb_time.items()):
                    if (
                        last is not None
                        and time.time() - last > self._heartbeat_dead_threshold
                    ):
                        self.logger.warning(
                            f"{self._node_info.node_id}: Child {child_id} HB dead — setting dead event"
                        )
                        ev = self._child_dead_events.get(child_id)
                        if ev is not None:
                            try:
                                main_loop.call_soon_threadsafe(ev.set)
                            except RuntimeError:
                                pass

        dispatch_task = asyncio.create_task(dispatch())
        dead_check_task = asyncio.create_task(dead_check())

        await stop.wait()
        for t in recv_tasks.values():
            t.cancel()
        dispatch_task.cancel()
        dead_check_task.cancel()
        await asyncio.gather(
            *recv_tasks.values(), dispatch_task, dead_check_task, return_exceptions=True
        )

        for conn in self._hb_transport.get_server_connections():
            if conn.is_open:
                await conn.close()

    async def _hb_recv_loop(self, conn: AsyncConnection, stop: asyncio.Event) -> None:
        while not stop.is_set():
            try:
                raw = await conn.recv()
                self._hb_recv_queue.put_nowait(raw)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"{self._node_info.node_id}: HB recv error: {e}")

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
            msg = cloudpickle.loads(raw_data[1])
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
            if sender_id.startswith("client-"):
                msg = cloudpickle.loads(raw_data[1])
                self._client_queue.put_nowait((full_id, msg))
                self.logger.debug(
                    f"{self._node_info.node_id}: Queued client message from {full_id}: {type(msg).__name__}"
                )
                return
            msg = cloudpickle.loads(raw_data[1])
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

    async def _monitor_parent(self) -> None:
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

    async def _recv_loop(self, conn: AsyncConnection) -> None:
        while not self._stop_event.is_set():
            try:
                raw_data = await conn.recv()
                self._recv_queue.put_nowait(raw_data)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stop_event.is_set():
                    self.logger.warning(
                        f"{self._node_info.node_id}: recv loop error: {e}"
                    )
                    await asyncio.sleep(0.01)

    async def _monitor_children(self) -> None:
        while not self._stop_event.is_set():
            try:
                raw_data = await self._recv_queue.get()
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
            await self._parent_conn.send(cloudpickle.dumps(msg))
            self.logger.debug(
                f"{self._node_info.node_id}: Sent message to parent: {type(msg).__name__}"
            )
            return True
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Sending message to parent failed with {e}"
            )
            return False

    async def recv_message_from_parent(
        self, cls: Type[Message], block: bool = False, timeout: Optional[float] = None
    ) -> Message | None:
        parent_id = self._node_info.parent_id
        if parent_id is None or parent_id not in self._cache:
            self.logger.warning("No parent available to receive message from.")
            return None
        if block is False and timeout is None:
            return self._cache[parent_id].get_nowait(cls)
        return await self._cache[parent_id].get(cls, timeout=timeout)

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
            packed = cloudpickle.dumps(msg)
            conn = self._data_transport.get_server_connection(
                self._node_info.node_id, self._node_info.secret_id
            )
            if conn is None:
                raise RuntimeError(f"No server connection for {child_id}")
            if child_id.startswith("client-"):
                await conn.send(packed, child_id)
            else:
                target_id = _encode_identity(
                    child_id, self._node_info.children_secret_ids[child_id]
                )
                await conn.send(packed, target_id)
                self.logger.debug(
                    f"{self._node_info.node_id}: Sent message of type {type(msg).__name__} to child {target_id}"
                )
            return True
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: hello Sending message to child {child_id} failed with {e}"
            )
            return False

    async def recv_message_from_child(
        self,
        cls: Type[Message],
        child_id: str,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> Message | None:
        if child_id not in self._cache:
            self.logger.warning(
                f"{child_id} not in cache. Current keys {self._cache.keys()}"
            )
            return None
        if block is False and timeout is None:
            return self._cache[child_id].get_nowait(cls)
        return await self._cache[child_id].get(cls, timeout=timeout)

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

        if (
            self._parent_hb_thread_loop is not None
            and self._parent_hb_asyncio_stop is not None
        ):
            self._parent_hb_thread_loop.call_soon_threadsafe(
                self._parent_hb_asyncio_stop.set
            )
        if self._parent_hb_thread is not None:
            self._parent_hb_thread.join(timeout=5.0)

        if (
            self._children_hb_thread_loop is not None
            and self._children_hb_asyncio_stop is not None
        ):
            self._children_hb_thread_loop.call_soon_threadsafe(
                self._children_hb_asyncio_stop.set
            )
        if self._children_hb_thread is not None:
            self._children_hb_thread.join(timeout=5.0)

        self.logger.info("Stopped HB threads")

        await self.clear_cache()
        try:
            if self._parent_conn and self._parent_conn.is_open:
                await self._parent_conn.close()
            for conn in self._data_transport.get_server_connections():
                if conn.is_open:
                    await conn.close()
        except Exception as e:
            self.logger.warning(f"{self._node_info.node_id}: Error during cleanup: {e}")
        self.logger.info("Done stopping comm")
