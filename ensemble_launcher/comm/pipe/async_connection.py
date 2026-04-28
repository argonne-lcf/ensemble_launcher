import random
from abc import ABC, abstractmethod
from typing import List, Optional, Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound="AsyncConnectionState")


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


class ClientConnectionState(AsyncConnectionState):
    address: Optional[str] = None
    remote_address: Optional[str] = None


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
    async def send(self, data: List[bytes]) -> bool:
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


class ServerConnection(AsyncConnection):
    @property
    def address(self) -> Optional[str]:
        return None


class ClientConnection(AsyncConnection):
    @property
    def address(self) -> Optional[str]:
        return None

    @property
    def remote_address(self) -> Optional[str]:
        return None


class AsyncZMQRouterConnectionState(ServerConnectionState):
    transport_type: str = "zmq"
    address: str


class AsyncZMQRouterConnection(ServerConnection):
    transport_type: str = "zmq"

    def __init__(self, identity: str, secret_id: str, address: str):
        super().__init__(identity=identity, secret_id=secret_id)
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
        return await self._socket.recv_multipart()

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
        )

    @classmethod
    def set_state(
        cls, state: AsyncZMQRouterConnectionState
    ) -> "AsyncZMQRouterConnection":
        return cls(
            identity=state.identity, secret_id=state.secret_id, address=state.address
        )


class AsyncZMQDealerConnectionState(ClientConnectionState):
    transport_type: str = "zmq"
    remote_address: str


class AsyncZMQDealerConnection(ClientConnection):
    transport_type: str = "zmq"

    def __init__(self, identity: str, secret_id: str, remote_address: str):
        super().__init__(identity=identity, secret_id=secret_id)
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
        return await self._socket.recv_multipart()

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
        )

    @classmethod
    def set_state(
        cls, state: AsyncZMQDealerConnectionState
    ) -> "AsyncZMQDealerConnection":
        return cls(
            identity=state.identity,
            secret_id=state.secret_id,
            remote_address=state.remote_address,
        )
