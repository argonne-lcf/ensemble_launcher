import asyncio
import functools
import os
import queue as _queue
import random
import socket
import threading
import time
from asyncio import Queue
from dataclasses import dataclass as _dataclass
from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

import cloudpickle

from ensemble_launcher.logging import setup_logger

from .async_base import AsyncComm, AsyncCommState
from .nodeinfo import NodeInfo

try:
    import zmq
    from zmq.asyncio import Context, Poller, Socket

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

# logger = logging.getLogger(__name__)


@_dataclass
class _HeartBeat:
    alive: bool = True


class AsyncZMQCommState(AsyncCommState):
    node_info: NodeInfo
    my_address: str
    parent_address: Optional[str] = None
    my_hb_address: Optional[str] = None
    parent_hb_address: Optional[str] = None


class AsyncZMQComm(AsyncComm):
    def __init__(
        self,
        logger: Logger,
        node_info: NodeInfo,
        parent_comm: "AsyncZMQComm" = None,
        heartbeat_interval: int = 1,
        heartbeat_dead_threshold: float = 30.0,
        parent_address: str = None,  ###parent comm is not always pickleble
        parent_hb_address: str = None,
    ):

        super().__init__(logger, node_info, parent_comm, heartbeat_interval)
        if not ZMQ_AVAILABLE:
            self.logger.error(f"zmq is not available")
            raise ModuleNotFoundError

        # ZMQ specific attributes
        self.parent_address = (
            self._parent_comm.my_address
            if self._parent_comm is not None
            else parent_address
        )
        self.my_address = f"{socket.gethostname() if 'local' not in socket.gethostname() else 'localhost'}:{5555 + random.randint(1, 1000)}"

        # Heartbeat subprocess addresses
        self.my_hb_address: str = f"{self.my_address.rsplit(':', 1)[0]}:{int(self.my_address.rsplit(':', 1)[1]) + 500}"
        self.parent_hb_address: Optional[str] = (
            parent_comm.my_hb_address
            if parent_comm is not None
            else parent_hb_address
            if parent_hb_address is not None
            else None
        )

        self.zmq_context = None
        self.router_socket = None
        self.dealer_socket = None

        self._router_cache = None

        self._stop_event = None
        self._client_queue: asyncio.Queue = (
            asyncio.Queue()
        )  # (client_id, Message) tuples
        self._parent_monitor_started = False
        self._child_monitor_started = False
        self._monitor_tasks: List[asyncio.Task] = []

        # Heartbeat state
        # Parent and children HB I/O each run in their own dedicated thread with a
        # separate asyncio loop; dead-detection monitoring stays in the main loop.
        self._parent_hb_thread: Optional[threading.Thread] = None
        self._parent_hb_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._parent_hb_asyncio_stop: Optional[asyncio.Event] = (
            None  # lives in parent thread loop
        )
        self._parent_hb_started: bool = False

        self._children_hb_thread: Optional[threading.Thread] = None
        self._children_hb_thread_loop: Optional[asyncio.AbstractEventLoop] = None
        self._children_hb_asyncio_stop: Optional[asyncio.Event] = (
            None  # lives in children thread loop
        )
        self._children_hb_started: bool = False

        self._hb_tasks: List[asyncio.Task] = []  # main-loop monitoring tasks
        self._hb_stop: Optional[asyncio.Event] = None  # main-loop stop signal
        # Timestamps written by HB threads, read by main-loop monitors (GIL-safe)
        self._last_parent_hb_time: Optional[float] = None
        self._last_child_hb_time: Dict[str, float] = {}
        # asyncio.Events on main loop — set via call_soon_threadsafe from HB threads
        self._hb_parent_ready: Optional[asyncio.Event] = None
        self._hb_child_ready: Dict[str, asyncio.Event] = {}
        self._heartbeat_dead_threshold: float = heartbeat_dead_threshold

    async def update_node_info(self, node_info: NodeInfo):
        added_children = set(node_info.children_ids) - set(self._node_info.children_ids)
        removed_children = set(self._node_info.children_ids) - set(
            node_info.children_ids
        )

        for child_id in added_children:
            self._child_dead_events[child_id] = asyncio.Event()
            self._hb_child_ready[child_id] = asyncio.Event()
            self._last_child_hb_time[child_id] = None
            if self._children_hb_started:
                t = asyncio.create_task(
                    self._hb_run_child_hb_loop(child_id),
                    name=f"hb-child-{child_id}",
                )
                self._hb_tasks.append(t)

        for child_id in removed_children:
            self._hb_child_ready.pop(child_id, None)
            self._last_child_hb_time.pop(child_id, None)
            self._child_dead_events.pop(child_id, None)

        if self._router_cache is not None:
            for child_id in removed_children:
                self._router_cache.pop(child_id, None)
        await super().update_node_info(node_info)

    async def init_cache(self):
        await super().init_cache()
        await self._init_router_cache()

    async def _init_router_cache(self):
        if self._router_cache is None:
            self._router_cache: Dict[str, Queue] = {}

        for child_id in self._node_info.children_ids:
            if child_id not in self._router_cache:
                self._router_cache[child_id] = Queue()

        if self._node_info.parent_id:
            if self._node_info.parent_id not in self._router_cache:
                self._router_cache[self._node_info.parent_id] = Queue()

    async def setup_zmq_sockets(self):
        if not self._router_cache:
            await self._init_router_cache()

        self.zmq_context = Context()
        self.router_socket = self.zmq_context.socket(zmq.ROUTER, socket_class=Socket)
        self.router_socket.setsockopt(
            zmq.IDENTITY, f"{self._node_info.node_id}".encode()
        )
        self.router_socket.setsockopt(zmq.SNDHWM, 10000)
        self.router_socket.setsockopt(zmq.RCVHWM, 10000)
        try:
            self.router_socket.bind(f"tcp://{self.my_address}")
            # self.router_socket.bind(f"tcp://*:{self.my_address.split(':')[-1]}")
            self.logger.info(
                f"{self._node_info.node_id}: Successfully bound to {self.my_address}"
            )
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                # Try binding up to 3 times with different ports
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        port = int(self.my_address.split(":")[-1]) + random.randint(
                            1, 1000
                        )
                        self.logger.info(
                            f"{self._node_info.node_id}: Attempt {attempt + 1}/{max_attempts}: Trying to bind to port {port} instead."
                        )
                        self.my_address = f"{self.my_address.rsplit(':', 1)[0]}:{port}"
                        self.router_socket.bind(f"tcp://{self.my_address}")
                        self.logger.info(
                            f"{self._node_info.node_id}: Successfully bound to {self.my_address}"
                        )
                        break  # Break out of the retry loop if binding succeeds
                    except zmq.error.ZMQError as retry_error:
                        if (
                            "Address already in use" in str(retry_error)
                            and attempt < max_attempts - 1
                        ):
                            self.logger.warning(
                                f"{self._node_info.node_id}: Port {port} also in use, retrying..."
                            )
                            continue
                        else:
                            raise retry_error
            else:
                raise e

        if self.parent_address is not None:
            self.dealer_socket = self.zmq_context.socket(
                zmq.DEALER, socket_class=Socket
            )
            self.dealer_socket.setsockopt(
                zmq.IDENTITY,
                f"{self._node_info.node_id}:{self._node_info.secret_id}".encode(),
            )
            self.dealer_socket.setsockopt(zmq.SNDHWM, 10000)
            self.dealer_socket.setsockopt(zmq.RCVHWM, 10000)
            self.dealer_socket.connect(f"tcp://{self.parent_address}")
            self.logger.info(
                f"{self._node_info.node_id}: connected to:{self.parent_address}"
            )

    async def start_monitors(self, **kwargs):
        """Start background tasks to monitor ZMQ sockets."""
        await super().start_monitors(**kwargs)

        if kwargs.get("parent_only", False):
            if (
                self._node_info.parent_id is not None
                and not self._parent_monitor_started
            ):
                self._monitor_tasks.append(
                    asyncio.create_task(self._monitor_parent_socket())
                )
                self._parent_monitor_started = True
        elif kwargs.get("children_only", False):
            if not self._child_monitor_started:
                self._monitor_tasks.append(
                    asyncio.create_task(self._monitor_child_sockets())
                )
                self._child_monitor_started = True
        else:
            if (
                self._node_info.parent_id is not None
                and not self._parent_monitor_started
            ):
                self._monitor_tasks.append(
                    asyncio.create_task(self._monitor_parent_socket())
                )
                self._parent_monitor_started = True
            if not self._child_monitor_started:
                self._monitor_tasks.append(
                    asyncio.create_task(self._monitor_child_sockets())
                )
                self._child_monitor_started = True

        parent_only = kwargs.get("parent_only", False)
        children_only = kwargs.get("children_only", False)
        main_loop = asyncio.get_running_loop()

        if self._hb_stop is None:
            self._hb_stop = asyncio.Event()

        # --- Parent HB thread (DEALER → parent) ---
        if not children_only and self.parent_hb_address and not self._parent_hb_started:
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
            self._hb_tasks.append(
                asyncio.create_task(
                    self._hb_run_parent_hb_loop(),
                    name=f"hb-parent-loop-{self._node_info.node_id}",
                )
            )

        # --- Children HB thread (ROUTER ← children) ---
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

            # Wait for the thread to bind and report its address
            try:
                actual_hb_addr = await asyncio.wait_for(addr_q.get(), timeout=10.0)
                if actual_hb_addr is not None:
                    self.my_hb_address = actual_hb_addr
                    self.logger.info(
                        f"{self._node_info.node_id}: Children HB thread bound to {self.my_hb_address}"
                    )
            except asyncio.TimeoutError:
                self.logger.warning(
                    f"{self._node_info.node_id}: Children HB thread did not report bound address"
                )

            for child_id in list(self._last_child_hb_time.keys()):
                self._hb_tasks.append(
                    asyncio.create_task(
                        self._hb_run_child_hb_loop(child_id),
                        name=f"hb-child-{child_id}",
                    )
                )

    # ------------------------------------------------------------------ #
    # HB thread — dedicated loop for ZMQ send/recv so main loop is never  #
    # starved of heartbeats under heavy task load                          #
    # ------------------------------------------------------------------ #

    def _parent_hb_thread_main(self, main_loop: asyncio.AbstractEventLoop) -> None:
        """Entry point for the parent HB thread. Creates its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._parent_hb_thread_loop = loop
        try:
            loop.run_until_complete(self._parent_hb_coroutine(main_loop))
        finally:
            loop.close()
            self._parent_hb_thread_loop = None

    async def _parent_hb_coroutine(self, main_loop: asyncio.AbstractEventLoop) -> None:
        """Runs inside the parent HB thread's event loop: sends HBs to parent."""
        stop = asyncio.Event()
        self._parent_hb_asyncio_stop = stop

        ctx = Context()
        dealer = ctx.socket(zmq.DEALER, socket_class=Socket)
        dealer.setsockopt(
            zmq.IDENTITY,
            f"{self._node_info.node_id}:{self._node_info.secret_id}".encode(),
        )
        dealer.connect(f"tcp://{self.parent_hb_address}")

        task = asyncio.create_task(
            self._hb_send_to_parent_thread(dealer, main_loop, stop)
        )
        await stop.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        dealer.close()
        ctx.term()

    def _children_hb_thread_main(
        self, main_loop: asyncio.AbstractEventLoop, addr_q: asyncio.Queue
    ) -> None:
        """Entry point for the children HB thread. Creates its own event loop."""
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
        """Runs inside the children HB thread's event loop: receives HBs from children."""
        stop = asyncio.Event()
        self._children_hb_asyncio_stop = stop

        ctx = Context()
        router = ctx.socket(zmq.ROUTER, socket_class=Socket)
        router.setsockopt(zmq.IDENTITY, self._node_info.node_id.encode())

        actual_hb_address = self.my_hb_address
        try:
            router.bind(f"tcp://{actual_hb_address}")
        except Exception as e:
            if "Address already in use" in str(e):
                host = actual_hb_address.rsplit(":", 1)[0]
                base_port = int(actual_hb_address.rsplit(":", 1)[1])
                for _ in range(10):
                    try:
                        port = base_port + random.randint(1, 1000)
                        actual_hb_address = f"{host}:{port}"
                        router.bind(f"tcp://{actual_hb_address}")
                        break
                    except Exception:
                        continue
                else:
                    main_loop.call_soon_threadsafe(addr_q.put_nowait, None)
                    router.close()
                    ctx.term()
                    return
            else:
                main_loop.call_soon_threadsafe(addr_q.put_nowait, None)
                router.close()
                ctx.term()
                return

        # Report actual bound address back to the main loop
        main_loop.call_soon_threadsafe(addr_q.put_nowait, actual_hb_address)

        task = asyncio.create_task(
            self._hb_recv_from_children_thread(router, main_loop, stop)
        )
        await stop.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        router.close()
        ctx.term()

    async def _hb_recv_from_children_thread(
        self,
        router: Any,
        main_loop: asyncio.AbstractEventLoop,
        stop: asyncio.Event,
    ) -> None:
        """Receive HBs from children and echo back. Runs in HB thread's loop."""
        while not stop.is_set():
            try:
                parts = await router.recv_multipart()
                sender_id = (
                    parts[0].decode().split(":", 1)[0]
                )  # strip :secret_id suffix from ZMQ identity
                # Parse child's secret_id from the HB payload.
                # The child verifies the echo matches its own secret_id (see _hb_send_to_parent_thread).
                child_secret_id, _ = cloudpickle.loads(parts[1])
                expected = self._node_info.children_secret_ids.get(sender_id)
                if expected is not None and child_secret_id != expected:
                    self.logger.warning(
                        f"{self._node_info.node_id}: HB from {sender_id} has unexpected secret_id — "
                        f"expecting {expected}, got {child_secret_id}"
                        f"stale connection, ignoring"
                    )
                    continue
                self._last_child_hb_time[sender_id] = time.time()  # GIL-safe
                ev = self._hb_child_ready.get(sender_id)
                if ev is not None and not ev.is_set():
                    try:
                        main_loop.call_soon_threadsafe(ev.set)
                    except RuntimeError:
                        pass  # main loop already closed
                # Echo secret_id back as a single pickled frame so the child
                # can validate it with `secret_id == self._node_info.parent_secret_id`.
                await router.send_multipart(
                    [
                        parts[0],
                        cloudpickle.dumps((self._node_info.secret_id, _HeartBeat())),
                    ]
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(f"{self._node_info.node_id}: HB recv error: {e}")

    async def _hb_send_to_parent_thread(
        self,
        dealer: Any,
        main_loop: asyncio.AbstractEventLoop,
        stop: asyncio.Event,
    ) -> None:
        """Send periodic HBs to parent and await reply. Runs in HB thread's loop."""
        hb_bytes = cloudpickle.dumps((self._node_info.secret_id, _HeartBeat()))
        while not stop.is_set():
            try:
                await dealer.send(hb_bytes)
                msg = await asyncio.wait_for(dealer.recv(), timeout=1.0)
                if msg is not None:
                    parent_secret_id, _ = cloudpickle.loads(msg)
                    if parent_secret_id == self._node_info.parent_secret_id:
                        if (
                            self._hb_parent_ready is not None
                            and not self._hb_parent_ready.is_set()
                        ):
                            try:
                                main_loop.call_soon_threadsafe(
                                    self._hb_parent_ready.set
                                )
                            except RuntimeError:
                                pass  # main loop already closed
                        self._last_parent_hb_time = time.time()  # GIL-safe
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.warning(
                    f"{self._node_info.node_id}:{self._node_info.secret_id}: HB send error: {e}"
                )
            if time.time() - self._last_parent_hb_time > self._heartbeat_dead_threshold:
                break
            jitter = self.heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
            await asyncio.sleep(jitter)

    async def _hb_run_parent_hb_loop(self) -> None:
        """Check whether the parent has gone silent; set parent_dead_event if so."""
        # The parent need to be marked ready before deciding its death
        await self._hb_parent_ready.wait()
        while not self._hb_stop.is_set():
            jitter = self.heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
            try:
                await asyncio.wait_for(self._hb_stop.wait(), timeout=jitter)
                break  # stop was set
            except asyncio.TimeoutError:
                pass
            if self._last_parent_hb_time is not None:
                if (
                    time.time() - self._last_parent_hb_time
                    > self._heartbeat_dead_threshold
                ):
                    self.logger.warning(
                        f"{self._node_info.node_id}: Parent HB dead — setting parent_dead_event"
                    )
                    if self.parent_dead_event is not None:
                        self.parent_dead_event.set()
                    self._hb_stop.set()
                    break

    async def _hb_run_child_hb_loop(self, child_id: str) -> None:
        """Check whether a child has gone silent; set its dead event if so."""
        # The child need to be ready first to mark it dead or alive
        await self._hb_child_ready.get(child_id).wait()
        while not self._hb_stop.is_set():
            jitter = self.heartbeat_interval * (1 + random.uniform(-0.1, 0.1))
            try:
                await asyncio.wait_for(self._hb_stop.wait(), timeout=jitter)
                break  # stop was set
            except asyncio.TimeoutError:
                pass
            last = self._last_child_hb_time.get(child_id)
            if last is not None and time.time() - last > self._heartbeat_dead_threshold:
                self.logger.warning(
                    f"{self._node_info.node_id}: Child {child_id} HB dead — setting dead event"
                )
                ev = self._child_dead_events.get(child_id)
                if ev is not None:
                    ev.set()
                break

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

    async def _deserialize_and_dispatch_parent(
        self, raw_data: list, loop: asyncio.AbstractEventLoop, parent_id: str
    ) -> None:
        """Deserialize a raw frame from the parent and put it into the appropriate cache.

        raw_data layout (after ROUTER strips the routing identity):
            [0] secret_id bytes  — sender's secret_id for staleness detection
            [1] pickled payload
        """
        from .messages import Message

        try:
            sender_secret_id = raw_data[0].decode()
            expected = self._node_info.parent_secret_id
            if expected is not None and sender_secret_id != expected:
                self.logger.warning(
                    f"{self._node_info.node_id}: Discarding message from parent {parent_id} — "
                    f"secret_id mismatch (stale connection)"
                )
                return
            msg = await loop.run_in_executor(None, cloudpickle.loads, raw_data[1])
            if isinstance(msg, Message):
                self._cache[parent_id].put_nowait(msg)
                self.logger.debug(
                    f"{self._node_info.node_id}: Cached message from parent: {type(msg).__name__}"
                )
            else:
                self._router_cache[parent_id].put_nowait(msg)
                self.logger.debug(
                    f"{self._node_info.node_id}: Cached raw data from parent."
                )
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Failed to deserialize message from parent: {e}"
            )

    async def _deserialize_and_dispatch_child(
        self, raw_data: list, loop: asyncio.AbstractEventLoop
    ) -> None:
        """Deserialize a raw multipart frame from a child and put it into the appropriate cache.

        raw_data layout for regular children:
            [0] sender_id bytes
            [1] secret_id bytes  — sender's secret_id for staleness detection
            [2] pickled payload
        Cluster clients (sender_id starts with "client:") skip the secret_id frame:
            [0] client_id bytes
            [1] pickled payload
        """
        from .messages import Message

        full_id = raw_data[0].decode()
        sender_id = (
            full_id if full_id.startswith("client:") else full_id.split(":", 1)[0]
        )
        try:
            if sender_id.startswith("client:"):
                # Clients do not include a secret_id frame.
                msg = await loop.run_in_executor(None, cloudpickle.loads, raw_data[1])
                self._client_queue.put_nowait((sender_id, msg))
                self.logger.debug(
                    f"{self._node_info.node_id}: Queued client message from {sender_id}: {type(msg).__name__}"
                )
                return
            sender_secret_id = raw_data[1].decode()
            expected = self._node_info.children_secret_ids.get(sender_id)
            if expected is not None and sender_secret_id != expected:
                self.logger.warning(
                    f"{self._node_info.node_id}: Discarding message from child {sender_id} — "
                    f"secret_id mismatch (stale connection)"
                )
                return
            msg = await loop.run_in_executor(None, cloudpickle.loads, raw_data[2])
            if isinstance(msg, Message):
                self._cache[sender_id].put_nowait(msg)
                self.logger.debug(
                    f"{self._node_info.node_id}: Cached message from child {sender_id}: {type(msg).__name__}"
                )
            else:
                self._router_cache[sender_id].put_nowait(msg)
                self.logger.info(
                    f"{self._node_info.node_id}: Cached raw data from child {sender_id}."
                )
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Failed to deserialize message from child {sender_id}: {e}"
            )

    async def _monitor_parent_socket(self) -> None:
        """
        Monitor the parent dealer socket for incoming messages and cache them directly to _cache.
        Deserialization is offloaded to a task so the socket is drained without blocking on unpickling.
        """
        await self._init_router_cache()
        parent_id = self._node_info.parent_id
        loop = asyncio.get_running_loop()
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self.dealer_socket.recv_multipart()
                failures = 0
                asyncio.create_task(
                    self._deserialize_and_dispatch_parent(raw_data, loop, parent_id)
                )
            except Exception as e:
                failures += 1
                self.logger.warning(
                    f"{self._node_info.node_id}: Error receiving from parent failed {failures} times: {e}"
                )
                await asyncio.sleep(0.01)  # Backoff after repeated failures

    async def _monitor_child_sockets(self) -> None:
        """
        Monitor the child router sockets for incoming messages and cache them directly to _cache.
        Deserialization is offloaded to a task so the socket is drained without blocking on unpickling.
        """
        await self._init_router_cache()
        loop = asyncio.get_running_loop()
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self.router_socket.recv_multipart()
                failures = 0
                asyncio.create_task(
                    self._deserialize_and_dispatch_child(raw_data, loop)
                )
            except Exception as e:
                failures += 1
                self.logger.warning(
                    f"{self._node_info.node_id}: Error receiving from child failed {failures} times: {e}"
                )
                await asyncio.sleep(0.01)  # Backoff after repeated failures

    async def _send_to_parent(self, data: Any) -> bool:
        if self._node_info.parent_id is None:
            self.logger.warning(
                f"{self._node_info.node_id}: No parent connection available to {self._node_info.parent_id}"
            )
            return False

        try:
            await self.dealer_socket.send_multipart(
                [self._node_info.secret_id.encode(), cloudpickle.dumps(data)]
            )
            self.logger.debug(
                f"{self._node_info.node_id}: Sent message to parent: {data}"
            )
            return True
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Sending message to parent failed with {e}"
            )
            return False

    async def _recv_from_parent(self, timeout: Optional[float] = None) -> Any:
        if self._node_info.parent_id is None:
            self.logger.error(
                f"{self._node_info.node_id}: No parent connection available"
            )
            raise RuntimeError("No parent connection available")

        try:
            # Check ZMQ-specific FIFO cache first for raw data
            parent_id = self._node_info.parent_id
            self.logger.debug(
                f"{self._node_info.node_id}: Waiting to receive message from parent {parent_id} with timeout {timeout}"
            )
            return await asyncio.wait_for(
                self._router_cache[parent_id].get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.debug(
                f"{self._node_info.node_id}: No message received from parent within timeout {timeout} seconds."
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Receiving message from parent failed with exception {e}!"
            )
            return None

    async def _send_to_child(self, child_id: str, data: Any) -> bool:
        if (
            not child_id.startswith("client:")
            and child_id not in self._node_info.children_ids
        ):
            self.logger.error(
                f"{self._node_info.node_id}: No connection to child {child_id}"
            )
            raise RuntimeError(f"No connection to child {child_id}")

        try:
            if child_id.startswith("client:"):
                self.logger.debug(
                    f"{self._node_info.node_id}: Sent message to child {child_id}"
                )
                await self.router_socket.send_multipart(
                    [f"{child_id}".encode(), cloudpickle.dumps(data)]
                )
            else:
                child_zmq_id = f"{child_id}:{self._node_info.children_secret_ids[child_id]}".encode()
                await self.router_socket.send_multipart(
                    [
                        child_zmq_id,
                        self._node_info.secret_id.encode(),
                        cloudpickle.dumps(data),
                    ]
                )
            self.logger.debug(
                f"{self._node_info.node_id}: Sent message to child {child_id}"
            )
            return True
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Sending message to child {child_id} failed with {e}"
            )
            return False

    async def _recv_from_child(
        self, child_id: str, timeout: Optional[float] = None
    ) -> Any:
        if child_id not in self._node_info.children_ids:
            self.logger.error(
                f"{self._node_info.node_id}: No connection to child {child_id}"
            )
            raise RuntimeError(f"No connection to child {child_id}")

        try:
            return await asyncio.wait_for(
                self._router_cache[child_id].get(), timeout=timeout
            )
        except asyncio.TimeoutError:
            self.logger.debug(
                f"{self._node_info.node_id}: No message received from child {child_id} within timeout {timeout} seconds."
            )
            return None
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Receiving message from child {child_id} failed with exception {e}!"
            )
            return None

    async def recv_client_message(
        self, timeout: Optional[float] = None
    ) -> Optional[Tuple[str, "Message"]]:
        """Return the next (client_id, message) from any connected client, or None on timeout."""
        try:
            if timeout is not None:
                return await asyncio.wait_for(self._client_queue.get(), timeout=timeout)
            return await self._client_queue.get()
        except asyncio.TimeoutError:
            return None

    async def close(self):
        """Clean up ZMQ resources."""
        self._stop_event.set()  ##signal monitors to stop

        # Cancel monitor tasks BEFORE closing sockets — prevents ZMQError on
        # pending recv Futures being garbage-collected with an unhandled exception.
        for t in self._monitor_tasks:
            if not t.done():
                t.cancel()
        if self._monitor_tasks:
            await asyncio.gather(*self._monitor_tasks, return_exceptions=True)
        self._monitor_tasks.clear()

        # Stop main-loop HB monitoring tasks
        if self._hb_stop is not None:
            self._hb_stop.set()
        for t in self._hb_tasks:
            if not t.done():
                t.cancel()
        if self._hb_tasks:
            await asyncio.gather(*self._hb_tasks, return_exceptions=True)
        self._hb_tasks.clear()

        # Stop parent HB thread
        if (
            self._parent_hb_thread_loop is not None
            and self._parent_hb_asyncio_stop is not None
        ):
            self._parent_hb_thread_loop.call_soon_threadsafe(
                self._parent_hb_asyncio_stop.set
            )
        if self._parent_hb_thread is not None:
            self._parent_hb_thread.join(timeout=5.0)

        # Stop children HB thread
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

        await super().clear_cache()
        try:
            # Clear ZMQ-specific FIFO cache
            for cache_queue in self._router_cache.values():
                try:
                    while True:
                        cache_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._router_cache.clear()

            # Close ZMQ resources
            if self.router_socket:
                self.router_socket.close()
            if self.dealer_socket:
                self.dealer_socket.close()
            if self.zmq_context:
                self.zmq_context.term()
        except Exception as e:
            self.logger.warning(
                f"{self._node_info.node_id}: Error during ZMQ cleanup: {e}"
            )
        self.logger.info(f"Done stopping comm")

    def pickable_copy(self) -> "AsyncZMQComm":
        state = self.get_state()
        state = AsyncZMQCommState.deserialize(state.serialize())
        return AsyncZMQComm.set_state(state)

    def get_state(self) -> AsyncZMQCommState:
        return AsyncZMQCommState(
            node_info=self._node_info,
            my_address=self.my_address,
            parent_address=self.parent_address,
            my_hb_address=self.my_hb_address,
            parent_hb_address=self.parent_hb_address,
        )

    @classmethod
    def set_state(cls, state: AsyncZMQCommState) -> "AsyncZMQComm":
        ret = AsyncZMQComm(
            logger=None,
            node_info=state.node_info,
            parent_address=state.parent_address,
            parent_hb_address=state.parent_hb_address,
        )
        ret.my_address = state.my_address
        if state.my_hb_address:
            ret.my_hb_address = state.my_hb_address
        return ret
