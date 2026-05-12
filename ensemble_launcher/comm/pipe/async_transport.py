import random
import socket
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Type, TypeVar

from pydantic import BaseModel

from .async_connection import (
    AsyncZMQDealerConnection,
    AsyncZMQDealerConnectionState,
    AsyncZMQRouterConnection,
    AsyncZMQRouterConnectionState,
    ClientConnection,
    ClientConnectionState,
    ServerConnection,
    ServerConnectionState,
)
from .registry import transport_registry

T = TypeVar("T", bound="AsyncTransportState")


class AsyncTransportState(BaseModel):
    transport_type: str
    server_connections: Dict[str, ServerConnectionState] = {}
    client_connections: Dict[str, ClientConnectionState] = {}

    def serialize(self, *args, **kwargs) -> str:
        return self.model_dump_json(*args, **kwargs)

    @classmethod
    def deserialize(cls: Type[T], data: str) -> T:
        return cls.model_validate_json(data)


class AsyncTransport(ABC):
    transport_type: str = ""

    def __init__(self):
        self._server_connections: Dict[str, ServerConnection] = {}
        self._client_connections: Dict[str, ClientConnection] = {}

    def get_server_connection(
        self, identity: str, secret_id: str, **kwargs
    ) -> Optional[ServerConnection]:
        key = f"{identity}:{secret_id}"
        conn = self._server_connections.get(key)
        if conn is None and kwargs:
            conn = self._create_server_connection(identity, secret_id, **kwargs)
            self._server_connections[key] = conn
        return conn

    def get_client_connection(
        self, identity: str, secret_id: str, **kwargs
    ) -> Optional[ClientConnection]:
        key = f"{identity}:{secret_id}"
        conn = self._client_connections.get(key)
        if conn is None and kwargs:
            conn = self._create_client_connection(identity, secret_id, **kwargs)
            self._client_connections[key] = conn
        return conn

    def get_server_connections(self) -> List[ServerConnection]:
        seen = set()
        result = []
        for conn in self._server_connections.values():
            if id(conn) not in seen:
                seen.add(id(conn))
                result.append(conn)
        return result

    def get_client_connections(self) -> List[ClientConnection]:
        return list(self._client_connections.values())

    @abstractmethod
    def _create_server_connection(
        self, identity: str, secret_id: str, **kwargs
    ) -> ServerConnection:
        pass

    @abstractmethod
    def _create_client_connection(
        self, identity: str, secret_id: str, **kwargs
    ) -> ClientConnection:
        pass

    @abstractmethod
    def create_child_pipe(
        self,
        parent_id: str,
        parent_secret: str,
        child_id: str,
        child_secret: str,
    ) -> Tuple[ServerConnection, ClientConnection]:
        pass

    @abstractmethod
    def get_state(self) -> Optional[AsyncTransportState]:
        pass

    @classmethod
    @abstractmethod
    def set_state(cls, state: AsyncTransportState) -> "AsyncTransport":
        pass


class AsyncZMQTransportState(AsyncTransportState):
    transport_type: str = "zmq"
    server_connections: Dict[str, AsyncZMQRouterConnectionState] = {}
    client_connections: Dict[str, AsyncZMQDealerConnectionState] = {}
    hostname: str


@transport_registry.register(
    "zmq",
    transport_state=AsyncZMQTransportState,
    server_connection=AsyncZMQRouterConnection,
    server_connection_state=AsyncZMQRouterConnectionState,
    client_connection=AsyncZMQDealerConnection,
    client_connection_state=AsyncZMQDealerConnectionState,
)
class AsyncZMQTransport(AsyncTransport):
    transport_type: str = "zmq"

    def __init__(self):
        super().__init__()
        hn = socket.gethostname()
        hostname = "localhost" if "local" in hn else hn
        self._hostname = hostname
        self._server_connections: Dict[str, AsyncZMQRouterConnection] = {}
        self._client_connections: Dict[str, AsyncZMQDealerConnection] = {}

    def _create_server_connection(
        self, identity: str, secret_id: str, **kwargs
    ) -> AsyncZMQRouterConnection:
        key = f"{identity}:{secret_id}"
        router = self._server_connections.get(key, None)
        if router is None:
            address = (
                kwargs.get("address")
                or f"{self._hostname}:{5555 + random.randint(1, 5000)}"
            )
            expected_remotes = kwargs.get("expected_remotes", None)
            router = AsyncZMQRouterConnection(
                identity=identity,
                secret_id=secret_id,
                address=address,
                expected_remotes=expected_remotes,
            )
            self._server_connections[key] = router
        return router

    def _create_client_connection(
        self, identity: str, secret_id: str, **kwargs
    ) -> AsyncZMQDealerConnection:
        key = f"{identity}:{secret_id}"
        dealer = self._client_connections.get(key, None)
        if dealer is None:
            remote_address = kwargs["remote_address"]
            dealer = AsyncZMQDealerConnection(
                identity=identity,
                secret_id=secret_id,
                remote_address=remote_address,
                remote_identity=kwargs.get("remote_identity"),
                remote_secret_id=kwargs.get("remote_secret_id"),
            )
            self._client_connections[key] = dealer
        return dealer

    def create_child_pipe(
        self,
        parent_id: str,
        parent_secret: str,
        child_id: str,
        child_secret: str,
    ) -> Tuple[AsyncZMQRouterConnection, AsyncZMQDealerConnection]:
        server = self.get_server_connection(parent_id, parent_secret, address=None)
        server.add_expected_remote(child_id, child_secret)
        client = self.get_client_connection(
            child_id,
            child_secret,
            remote_address=server.address,
            remote_identity=parent_id,
            remote_secret_id=parent_secret,
        )
        return server, client

    def create_connection(
        self,
        cls: type,
        identity: str,
        secret_id: str,
        address: Optional[str] = None,
        remote_address: Optional[str] = None,
    ):
        if cls is AsyncZMQRouterConnection:
            return self._create_server_connection(identity, secret_id, address=address)
        elif cls is AsyncZMQDealerConnection:
            if remote_address is None:
                raise ValueError(f"Need remote address to create {cls}")
            return self._create_client_connection(
                identity, secret_id, remote_address=remote_address
            )
        raise ValueError(f"Unknown connection type: {cls}")

    def get_state(self) -> AsyncZMQTransportState:
        return AsyncZMQTransportState(
            server_connections={
                k: v.get_state() for k, v in self._server_connections.items()
            },
            client_connections={
                k: v.get_state() for k, v in self._client_connections.items()
            },
            hostname=self._hostname,
        )

    @classmethod
    def set_state(cls, state: AsyncZMQTransportState) -> "AsyncZMQTransport":
        transport = cls()
        transport._hostname = state.hostname
        server_states = state.server_connections
        client_states = state.client_connections
        transport._server_connections = {
            k: AsyncZMQRouterConnection.set_state(v) for k, v in server_states.items()
        }
        transport._client_connections = {
            k: AsyncZMQDealerConnection.set_state(v) for k, v in client_states.items()
        }
        return transport
