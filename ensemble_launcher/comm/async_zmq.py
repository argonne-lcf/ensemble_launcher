import asyncio
import random
import socket
import time
from asyncio import Queue
from dataclasses import asdict
from logging import Logger
from typing import Any, Dict, Optional, Tuple

import cloudpickle

from .async_base import AsyncComm, AsyncCommState
from .nodeinfo import NodeInfo

try:
    import zmq
    from zmq.asyncio import Context, Poller, Socket

    ZMQ_AVAILABLE = True
except ImportError:
    ZMQ_AVAILABLE = False

# logger = logging.getLogger(__name__)


class AsyncZMQCommState(AsyncCommState):
    node_info: NodeInfo
    my_address: str
    parent_address: Optional[str] = None


class AsyncZMQComm(AsyncComm):
    def __init__(
        self,
        logger: Logger,
        node_info: NodeInfo,
        parent_comm: "AsyncZMQComm" = None,
        heartbeat_interval: int = 1,
        parent_address: str = None,  ###parent comm is not always pickleble
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

    async def update_node_info(self, node_info: NodeInfo):
        removed_children = set(self._node_info.children_ids) - set(
            node_info.children_ids
        )
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
                zmq.IDENTITY, f"{self._node_info.node_id}".encode()
            )
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
                asyncio.create_task(self._monitor_parent_socket())
                self._parent_monitor_started = True
        elif kwargs.get("children_only", False):
            if not self._child_monitor_started:
                asyncio.create_task(self._monitor_child_sockets())
                self._child_monitor_started = True
        else:
            if (
                self._node_info.parent_id is not None
                and not self._parent_monitor_started
            ):
                asyncio.create_task(self._monitor_parent_socket())
                self._parent_monitor_started = True
            if not self._child_monitor_started:
                asyncio.create_task(self._monitor_child_sockets())
                self._child_monitor_started = True

    async def _monitor_parent_socket(self) -> None:
        """
        Monitor the parent dealer socket for incoming messages and cache them directly to _cache.
        """
        from .messages import Message

        await self._init_router_cache()
        parent_id = self._node_info.parent_id
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self.dealer_socket.recv()
                msg = cloudpickle.loads(raw_data)

                # Push to _cache if it's a Message object
                if isinstance(msg, Message):
                    failures = 0  # Reset on success
                    self._cache[parent_id].put_nowait(msg)
                    self.logger.debug(
                        f"{self._node_info.node_id}: Cached message from parent: {type(msg).__name__}"
                    )
                else:
                    # Still cache raw data for non-Message types
                    self._router_cache[parent_id].put_nowait(msg)
                    self.logger.debug(
                        f"{self._node_info.node_id}: Cached raw data from parent."
                    )
            except Exception as e:
                failures += 1
                self.logger.warning(
                    f"{self._node_info.node_id}: Error caching data from parent failed {failures} times: {e}"
                )
                await asyncio.sleep(0.01)  # Backoff after repeated failures

    async def _monitor_child_sockets(self) -> None:
        """
        Monitor the child router sockets for incoming messages and cache them directly to _cache.
        """
        from .messages import Message

        await self._init_router_cache()
        failures = 0
        while not self._stop_event.is_set():
            try:
                raw_data = await self.router_socket.recv_multipart()
                sender_id = raw_data[0].decode()  # Convert bytes to string for child_id
                msg = cloudpickle.loads(raw_data[1])  # Unpickle the raw data

                # Push to _cache if it's a Message object
                if isinstance(msg, Message):
                    failures = 0  # Reset on success
                    if sender_id.startswith("client:"):
                        self._client_queue.put_nowait((sender_id, msg))
                        self.logger.debug(
                            f"{self._node_info.node_id}: Queued client message from {sender_id}: {type(msg).__name__}"
                        )
                    else:
                        self._cache[sender_id].put_nowait(msg)
                        self.logger.debug(
                            f"{self._node_info.node_id}: Cached message from child {sender_id}: {type(msg).__name__}"
                        )
                else:
                    # Still cache raw data for non-Message types
                    self._router_cache[sender_id].put_nowait(msg)
                    self.logger.info(
                        f"{self._node_info.node_id}: Cached raw data from child {sender_id}."
                    )
            except Exception as e:
                failures += 1
                self.logger.warning(
                    f"{self._node_info.node_id}: Error caching data from child failed {failures} times: {e}"
                )
                await asyncio.sleep(0.01)  # Backoff after repeated failures

    async def _send_to_parent(self, data: Any) -> bool:
        if self._node_info.parent_id is None:
            self.logger.warning(
                f"{self._node_info.node_id}: No parent connection available to {self._node_info.parent_id}"
            )
            return False

        try:
            self.dealer_socket.send(cloudpickle.dumps(data))
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
            self.router_socket.send_multipart(
                [f"{child_id}".encode(), cloudpickle.dumps(data)]
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

    def pickable_copy(self) -> "AsyncZMQComm":
        state = self.get_state()
        state = AsyncZMQCommState.deserialize(state.serialize())
        return AsyncZMQComm.set_state(state)

    def get_state(self) -> AsyncZMQCommState:
        return AsyncZMQCommState(
            node_info=self._node_info,
            my_address=self.my_address,
            parent_address=self.parent_address,
        )

    @classmethod
    def set_state(self, state: AsyncZMQCommState) -> "AsyncZMQComm":
        ret = AsyncZMQComm(
            logger=None,
            node_info=state.node_info,
            parent_address=state.parent_address,
        )
        ret.my_address = state.my_address
        return ret
