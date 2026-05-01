import logging
import random
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T", bound="AsyncConnectionState")

logger = logging.getLogger(__name__)


class IdentityVerificationError(Exception):
    pass


class AsyncConnectionState(BaseModel):
    transport_type: str
    identity: str
    secret_id: str

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

    def __init__(self, identity: str, secret_id: str):
        self._identity = identity
        self._secret_id = secret_id
        self._is_open = False

    @property
    def is_open(self) -> bool:
        return self._is_open

    @abstractmethod
    async def open(self) -> None:
        pass

    @abstractmethod
    async def send(self, data: bytes) -> bool:
        pass

    @abstractmethod
    async def recv(self) -> List[bytes]:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass

    @abstractmethod
    def get_state(self) -> AsyncConnectionState:
        pass

    @classmethod
    @abstractmethod
    def set_state(cls, state: AsyncConnectionState) -> "AsyncConnection":
        pass


def _decode_identity(raw: bytes):
    full_id = raw.decode()
    parts = full_id.split(":", 1)
    return full_id, parts[0], parts[1] if len(parts) > 1 else None


class ServerConnection(AsyncConnection):
    def __init__(
        self,
        identity: str,
        secret_id: str,
        expected_remotes: Optional[Dict[str, str]] = None,
    ):
        super().__init__(identity=identity, secret_id=secret_id)
        self._expected_remotes: Dict[str, str] = dict(expected_remotes or {})
        self._remotes_lock = threading.Lock()
        self._unknown_sender_validator: Optional[
            Callable[[str, Optional[str]], bool]
        ] = None

    @property
    def address(self) -> Optional[str]:
        return None

    @property
    def expected_remotes(self) -> Dict[str, str]:
        with self._remotes_lock:
            return dict(self._expected_remotes)

    def add_expected_remote(self, node_id: str, secret_id: str) -> None:
        with self._remotes_lock:
            self._expected_remotes[node_id] = secret_id

    def remove_expected_remote(self, node_id: str) -> None:
        with self._remotes_lock:
            self._expected_remotes.pop(node_id, None)

    def set_unknown_sender_validator(
        self, validator: Callable[[str, Optional[str]], bool]
    ) -> None:
        self._unknown_sender_validator = validator

    def verify_sender(
        self, sender_id: str, sender_secret: Optional[str] = None
    ) -> bool:
        with self._remotes_lock:
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
    ):
        super().__init__(identity=identity, secret_id=secret_id)
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
        if self._remote_secret_id is not None and sender_secret != self._remote_secret_id:
            return False
        return True


class AsyncZMQRouterConnectionState(ServerConnectionState):
    transport_type: str = "zmq"
    address: str


class AsyncZMQRouterConnection(ServerConnection):
    transport_type: str = "zmq"

    def __init__(
        self,
        identity: str,
        secret_id: str,
        address: str,
        expected_remotes: Optional[Dict[str, str]] = None,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            expected_remotes=expected_remotes,
        )
        self._address = address
        self._context = None
        self._socket = None

    @property
    def address(self) -> str:
        return self._address

    async def open(self) -> None:
        if self._is_open:
            return

        import zmq
        from zmq.asyncio import Context, Socket

        self._context = Context()
        self._socket = self._context.socket(zmq.ROUTER, socket_class=Socket)
        self._socket.setsockopt(
            zmq.IDENTITY, f"{self._identity}:{self._secret_id}".encode()
        )
        self._socket.setsockopt(zmq.SNDHWM, 10000)
        self._socket.setsockopt(zmq.RCVHWM, 10000)

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

        self._is_open = True

    async def send(self, data: bytes, target_id: str) -> bool:
        await self._socket.send_multipart(
            [target_id.encode(), f"{self._identity}:{self._secret_id}".encode(), data]
        )
        return True

    async def recv(self) -> List[bytes]:
        while True:
            frames = await self._socket.recv_multipart()
            if len(frames) >= 1:
                _, sender_id, sender_secret = _decode_identity(frames[0])
                if not self.verify_sender(sender_id, sender_secret):
                    logger.warning(
                        f"{self._identity}: Discarding message from {sender_id} "
                        f"— secret_id mismatch (stale connection)"
                    )
                    continue
            return frames

    async def close(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def get_state(self) -> AsyncZMQRouterConnectionState:
        return AsyncZMQRouterConnectionState(
            identity=self._identity,
            secret_id=self._secret_id,
            address=self._address,
            expected_remotes=self.expected_remotes,
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
        )


class AsyncZMQDealerConnectionState(ClientConnectionState):
    transport_type: str = "zmq"
    remote_address: str


class AsyncZMQDealerConnection(ClientConnection):
    transport_type: str = "zmq"

    def __init__(
        self,
        identity: str,
        secret_id: str,
        remote_address: str,
        remote_identity: Optional[str] = None,
        remote_secret_id: Optional[str] = None,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            remote_identity=remote_identity,
            remote_secret_id=remote_secret_id,
        )
        self._remote_address = remote_address
        self._context = None
        self._socket = None

    @property
    def remote_address(self) -> str:
        return self._remote_address

    async def open(self) -> None:
        if self._is_open:
            return

        import zmq
        from zmq.asyncio import Context, Socket

        self._context = Context()
        self._socket = self._context.socket(zmq.DEALER, socket_class=Socket)
        self._socket.setsockopt(
            zmq.IDENTITY, f"{self._identity}:{self._secret_id}".encode()
        )
        self._socket.setsockopt(zmq.SNDHWM, 10000)
        self._socket.setsockopt(zmq.RCVHWM, 10000)
        self._socket.connect(f"tcp://{self._remote_address}")
        self._is_open = True

    async def send(self, data: bytes) -> bool:
        await self._socket.send(data)
        return True

    async def recv(self) -> List[bytes]:
        while True:
            frames = await self._socket.recv_multipart()
            if self._remote_identity is not None and len(frames) >= 1:
                _, sender_id, sender_secret = _decode_identity(frames[0])
                if not self.verify_sender(sender_id, sender_secret):
                    logger.warning(
                        f"{self._identity}: Discarding message from {sender_id} "
                        f"— secret_id mismatch (stale connection)"
                    )
                    continue
            return frames

    async def close(self) -> None:
        if not self._is_open:
            return
        self._is_open = False
        if self._socket:
            self._socket.close()
            self._socket = None
        if self._context:
            self._context.term()
            self._context = None

    def get_state(self) -> AsyncZMQDealerConnectionState:
        return AsyncZMQDealerConnectionState(
            identity=self._identity,
            secret_id=self._secret_id,
            remote_address=self._remote_address,
            remote_identity=self._remote_identity,
            remote_secret_id=self._remote_secret_id,
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
        )
