import asyncio
import os
import random
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

from ensemble_launcher.logging import setup_logger

T = TypeVar("T", bound="AsyncConnectionState")

_ACK = b"_ACK"


class IdentityVerificationError(Exception):
    pass


class AsyncConnectionState(BaseModel):
    transport_type: str
    identity: str
    secret_id: str
    req_res: bool = False

    def serialize(self, *args, **kwargs) -> str:
        return self.model_dump_json(*args, **kwargs)

    @classmethod
    def deserialize(cls: Type[T], data: str) -> T:
        return cls.model_validate_json(data)


class ServerConnectionState(AsyncConnectionState):
    address: Optional[str] = None
    expected_remotes: Dict[str, str] = Field(default_factory=dict)


class ClientConnectionState(AsyncConnectionState):
    address: Optional[str] = None
    remote_address: Optional[str] = None
    remote_identity: Optional[str] = None
    remote_secret_id: Optional[str] = None


class AsyncConnection(ABC):
    transport_type: str = ""

    def __init__(self, identity: str, secret_id: str, req_res: bool = False):
        self._identity = identity
        self._secret_id = secret_id
        self._is_open = False
        self._req_res = req_res
        self._ack_futures: Dict[int, asyncio.Future] = {}
        self._msg_queue: asyncio.Queue = None
        self._close: asyncio.Event = None
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self._msg_counter: int = 0
        self._recv_task: asyncio.Task = None

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def identity(self):
        return self._identity

    @property
    def secret_id(self):
        return self._secret_id

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._msg_queue = asyncio.Queue()
        self._ack_futures = {}
        self._close = asyncio.Event()
        os.makedirs(f"{os.getcwd()}/logs/connections", exist_ok=True)
        self.logger = setup_logger(
            name=f"connection-{self.identity}",
            log_dir=f"{os.getcwd()}/logs/connections",
        )
        await self._raw_open()
        self._is_open = True
        if self._req_res:
            self._recv_task = self._loop.create_task(self._recv_loop())

    async def close(self):
        self._close.set()
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        self._is_open = False
        await self._raw_close()

    async def _recv_loop(self):
        while not self._close.is_set():
            try:
                frames = await asyncio.wait_for(self._raw_recv(), timeout=1.0)
                _, sender_id, sender_secret = decode_identity(frames[0])
                if not self.verify_sender(sender_id, sender_secret):
                    self.logger.warning(
                        f"{self._identity}: Discarding message from {sender_id} "
                        f"— secret_id mismatch (stale connection)"
                    )
                    continue

                if self._req_res and frames[-1] == _ACK:
                    msg_id = int.from_bytes(frames[-2], "big")
                    fut = self._ack_futures.pop(msg_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(True)
                    elif fut is None:
                        self.logger.warning(
                            f"Received ACK for unknown msg_id={msg_id} from {sender_id}"
                        )
                elif self._req_res:
                    msg_id_bytes = frames[1]
                    sender_full_id = frames[0].decode()
                    await self._raw_send(
                        _ACK,
                        msg_id=int.from_bytes(msg_id_bytes, "big"),
                        target_id=sender_full_id,
                    )
                    await self._msg_queue.put([frames[0]] + frames[2:])
                else:
                    await self._msg_queue.put(frames)

            except (TimeoutError, asyncio.TimeoutError):
                pass
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"_recv_loop failed with exception {e}")

    async def recv(self, timeout: Optional[float] = None) -> List[bytes]:
        if self._req_res:
            return await asyncio.wait_for(self._msg_queue.get(), timeout=timeout)
        return await asyncio.wait_for(self._raw_recv(), timeout=timeout)

    async def send(
        self, msg: Union[bytes, List[bytes]], target_id: Optional[str] = None, timeout: float = 5.0
    ) -> bool:
        self._msg_counter += 1

        if self._req_res:
            msg_id = self._msg_counter
            self._ack_futures[msg_id] = self._loop.create_future()
            success = await self._raw_send(msg, msg_id=msg_id, target_id=target_id)
        else:
            success = await self._raw_send(msg, target_id=target_id)

        if not success:
            self.logger.warning("Raw send failed")
            if self._req_res:
                self._ack_futures.pop(self._msg_counter, None)
            return False

        if self._req_res:
            try:
                return await asyncio.wait_for(
                    self._ack_futures[msg_id], timeout=timeout
                )
            except (TimeoutError, asyncio.TimeoutError):
                self._ack_futures.pop(msg_id, None)
                self.logger.warning(f"ACK timeout for msg_id={msg_id} to {target_id}")
                return False
        else:
            return True

    # ---------------------- Sync wrappers ----------------------------------
    def sopen(self):
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self.open(), self._loop)
        return fut.result()

    def sclose(self):
        if self._loop and self._loop_thread:
            fut = asyncio.run_coroutine_threadsafe(self.close(), self._loop)
            fut.result()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=10.0)

    def srecv(self) -> List[bytes]:
        return asyncio.run_coroutine_threadsafe(self.recv(), self._loop).result()

    def ssend(self, msg: bytes, target_id: Optional[str] = None, timeout: float = 5.0):
        return asyncio.run_coroutine_threadsafe(
            self.send(msg=msg, target_id=target_id, timeout=timeout), self._loop
        ).result()

    # ----------------------- Abstract methods ---------------------------------

    @abstractmethod
    async def _raw_open(self):
        pass

    @abstractmethod
    async def _raw_close(self) -> None:
        pass

    @abstractmethod
    async def _raw_send(
        self,
        data: bytes,
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        pass

    @abstractmethod
    async def _raw_recv(self) -> List[bytes]:
        pass

    @abstractmethod
    def get_state(self) -> AsyncConnectionState:
        pass

    @classmethod
    @abstractmethod
    def set_state(cls, state: AsyncConnectionState) -> "AsyncConnection":
        pass

    @abstractmethod
    def verify_sender(self, sender_id, sender_secret) -> bool:
        pass


def decode_identity(raw: bytes):
    full_id = raw.decode()
    parts = full_id.split(":", 1)
    return full_id, parts[0], parts[1] if len(parts) > 1 else None


class ServerConnection(AsyncConnection):
    def __init__(
        self,
        identity: str,
        secret_id: str,
        expected_remotes: Optional[Dict[str, str]] = None,
        req_res: bool = False,
    ):
        super().__init__(identity=identity, secret_id=secret_id, req_res=req_res)
        self._expected_remotes: Dict[str, str] = dict(expected_remotes or {})
        self._unknown_sender_validator: Optional[
            Callable[[str, Optional[str]], bool]
        ] = None

    @property
    def address(self) -> Optional[str]:
        return None

    @property
    def expected_remotes(self) -> Dict[str, str]:
        return dict(self._expected_remotes)

    def add_expected_remote(self, node_id: str, secret_id: str) -> None:
        self._expected_remotes[node_id] = secret_id

    def remove_expected_remote(self, node_id: str) -> None:
        self._expected_remotes.pop(node_id, None)

    def set_unknown_sender_validator(
        self, validator: Callable[[str, Optional[str]], bool]
    ) -> None:
        self._unknown_sender_validator = validator

    def verify_sender(
        self, sender_id: str, sender_secret: Optional[str] = None
    ) -> bool:
        if not self._expected_remotes:
            return True
        expected_secret = self._expected_remotes.get(sender_id)
        if expected_secret is None:
            if self._unknown_sender_validator is not None:
                return self._unknown_sender_validator(sender_id, sender_secret)
            return False
        return sender_secret == expected_secret


class ClientConnection(AsyncConnection):
    def __init__(
        self,
        identity: str,
        secret_id: str,
        remote_identity: Optional[str] = None,
        remote_secret_id: Optional[str] = None,
        req_res: bool = False,
    ):
        super().__init__(identity=identity, secret_id=secret_id, req_res=req_res)
        self._remote_identity = remote_identity
        self._remote_secret_id = remote_secret_id

    @property
    def address(self) -> Optional[str]:
        return None

    @property
    def remote_address(self) -> Optional[str]:
        return None

    @property
    def remote_identity(self) -> Optional[str]:
        return self._remote_identity

    @property
    def remote_secret_id_value(self) -> Optional[str]:
        return self._remote_secret_id

    def verify_sender(
        self, sender_id: str, sender_secret: Optional[str] = None
    ) -> bool:
        if self._remote_identity is None:
            return True
        if sender_id != self._remote_identity:
            return False
        if (
            self._remote_secret_id is not None
            and sender_secret != self._remote_secret_id
        ):
            return False
        return True


# ---------------------------------------------------------------------------
#  ZMQ ROUTER (Server) implementation
# ---------------------------------------------------------------------------


class AsyncZMQRouterConnectionState(ServerConnectionState):
    transport_type: str = "zmq"
    address: str
    lightweight: bool = False


class AsyncZMQRouterConnection(ServerConnection):
    transport_type: str = "zmq"

    def __init__(
        self,
        identity: str,
        secret_id: str,
        address: str,
        expected_remotes: Optional[Dict[str, str]] = None,
        req_res: bool = False,
        lightweight: bool = False,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            expected_remotes=expected_remotes,
            req_res=req_res,
        )
        self._address = address
        self._lightweight = lightweight
        self._context = None
        self._socket = None

    @property
    def address(self) -> str:
        return self._address

    async def _raw_open(self) -> None:
        import zmq
        from zmq.asyncio import Context, Socket

        self._context = Context()
        self._socket = self._context.socket(zmq.ROUTER, socket_class=Socket)
        self._socket.setsockopt(
            zmq.IDENTITY, f"{self._identity}:{self._secret_id}".encode()
        )

        if self._lightweight:
            self._context.set(zmq.IO_THREADS, 1)
            self._socket.setsockopt(zmq.SNDHWM, 100)
            self._socket.setsockopt(zmq.RCVHWM, 100)
            self._socket.setsockopt(zmq.SNDTIMEO, 100)
        else:
            self._context.set(zmq.IO_THREADS, 4)
            self._socket.setsockopt(zmq.SNDHWM, 10000)
            self._socket.setsockopt(zmq.RCVHWM, 10000)
            self._socket.setsockopt(zmq.HEARTBEAT_IVL, 5000)
            self._socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 15000)
            self._socket.setsockopt(zmq.HEARTBEAT_TTL, 15000)
            self._socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)

        try:
            self._socket.bind(f"tcp://{self._address}")
        except zmq.error.ZMQError as e:
            if "Address already in use" in str(e):
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        port = int(self._address.split(":")[-1]) + random.randint(
                            1, 1000
                        )
                        self._address = f"{self._address.rsplit(':', 1)[0]}:{port}"
                        self._socket.bind(f"tcp://{self._address}")
                        break
                    except zmq.error.ZMQError as retry_error:
                        if (
                            "Address already in use" in str(retry_error)
                            and attempt < max_attempts - 1
                        ):
                            continue
                        else:
                            raise retry_error
            else:
                raise e
        endpoint = self._socket.getsockopt(zmq.LAST_ENDPOINT).decode()
        self._address = endpoint.replace("tcp://", "")
        self.logger.info(f"Server bound to {self._address}")

    async def _raw_close(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        """Send frames via ZMQ Router.

        Sends: [target_id, identity_frame, msg_id(8B)?, *data_frames]
        """
        identity_frame = f"{self._identity}:{self._secret_id}".encode()
        if msg_id is not None:
            frames = [
                target_id.encode(),
                identity_frame,
                msg_id.to_bytes(8, "big"),
            ]
        else:
            frames = [target_id.encode(), identity_frame]
        if isinstance(data, list):
            frames.extend(data)
        else:
            frames.append(data)
        try:
            await self._socket.send_multipart(frames)
        except Exception:
            if self._lightweight:
                return False
            raise
        return True

    async def _raw_recv(self) -> List[bytes]:
        """Receive frames via ZMQ Router.

        Returns: [dealer_routing_id, msg_id(8B)?, *data_frames]
        ZMQ prepends the dealer's routing id automatically.
        """
        return await self._socket.recv_multipart()

    def get_state(self) -> AsyncZMQRouterConnectionState:
        return AsyncZMQRouterConnectionState(
            identity=self._identity,
            secret_id=self._secret_id,
            address=self._address,
            expected_remotes=self.expected_remotes,
            req_res=self._req_res,
            lightweight=self._lightweight,
        )

    @classmethod
    def set_state(
        cls, state: AsyncZMQRouterConnectionState
    ) -> "AsyncZMQRouterConnection":
        return cls(
            identity=state.identity,
            secret_id=state.secret_id,
            address=state.address,
            expected_remotes=state.expected_remotes,
            req_res=state.req_res,
            lightweight=state.lightweight,
        )


# ---------------------------------------------------------------------------
#  ZMQ DEALER (Client) implementation
# ---------------------------------------------------------------------------


class AsyncZMQDealerConnectionState(ClientConnectionState):
    transport_type: str = "zmq"
    remote_address: str
    lightweight: bool = False


class AsyncZMQDealerConnection(ClientConnection):
    transport_type: str = "zmq"

    def __init__(
        self,
        identity: str,
        secret_id: str,
        remote_address: str,
        remote_identity: Optional[str] = None,
        remote_secret_id: Optional[str] = None,
        req_res: bool = False,
        lightweight: bool = False,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            remote_identity=remote_identity,
            remote_secret_id=remote_secret_id,
            req_res=req_res,
        )
        self._remote_address = remote_address
        self._lightweight = lightweight
        self._context = None
        self._socket = None

    @property
    def remote_address(self) -> str:
        return self._remote_address

    async def _raw_open(self) -> None:
        import zmq
        from zmq.asyncio import Context, Socket

        self._context = Context()
        self._socket = self._context.socket(zmq.DEALER, socket_class=Socket)
        self._socket.setsockopt(
            zmq.IDENTITY, f"{self._identity}:{self._secret_id}".encode()
        )

        if self._lightweight:
            self._context.set(zmq.IO_THREADS, 1)
            self._socket.setsockopt(zmq.SNDHWM, 100)
            self._socket.setsockopt(zmq.RCVHWM, 100)
        else:
            self._context.set(zmq.IO_THREADS, 4)
            self._socket.setsockopt(zmq.SNDHWM, 10000)
            self._socket.setsockopt(zmq.RCVHWM, 10000)
            self._socket.setsockopt(zmq.HEARTBEAT_IVL, 5000)
            self._socket.setsockopt(zmq.HEARTBEAT_TIMEOUT, 15000)
            self._socket.setsockopt(zmq.HEARTBEAT_TTL, 15000)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 10)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 5)
            self._socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 3)

        self._socket.connect(f"tcp://{self._remote_address}")
        self.logger.info(f"Connected to {self.remote_address}")

    async def _raw_close(self) -> None:
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        """Send frames via ZMQ Dealer.

        Sends: [msg_id(8B)?, *data_frames]
        """
        if msg_id is not None:
            frames = [msg_id.to_bytes(8, "big")]
        else:
            frames = []
        if isinstance(data, list):
            frames.extend(data)
        else:
            frames.append(data)
        await self._socket.send_multipart(frames)
        return True

    async def _raw_recv(self) -> List[bytes]:
        """Receive frames via ZMQ Dealer.

        Returns: [identity_frame, msg_id(8B)?, *data_frames]
        ZMQ strips the routing id; the first frame is the sender's identity.
        """
        return await self._socket.recv_multipart()

    def get_state(self) -> AsyncZMQDealerConnectionState:
        return AsyncZMQDealerConnectionState(
            identity=self._identity,
            secret_id=self._secret_id,
            remote_address=self._remote_address,
            remote_identity=self._remote_identity,
            remote_secret_id=self._remote_secret_id,
            req_res=self._req_res,
            lightweight=self._lightweight,
        )

    @classmethod
    def set_state(
        cls, state: AsyncZMQDealerConnectionState
    ) -> "AsyncZMQDealerConnection":
        return cls(
            identity=state.identity,
            secret_id=state.secret_id,
            remote_address=state.remote_address,
            remote_identity=state.remote_identity,
            remote_secret_id=state.remote_secret_id,
            req_res=state.req_res,
            lightweight=state.lightweight,
        )
