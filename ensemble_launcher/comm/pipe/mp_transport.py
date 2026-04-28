import multiprocessing
from typing import Dict, Optional, Tuple

from .async_connection import ClientConnection, ServerConnection
from .async_transport import AsyncTransport, AsyncTransportState
from .mp_connection import AsyncMPConnection, AsyncMPConnectionState
from .registry import transport_registry


class AsyncMPTransportState(AsyncTransportState):
    transport_type: str = "mp"
    server_connections: Dict[str, AsyncMPConnectionState] = {}
    client_connections: Dict[str, AsyncMPConnectionState] = {}
    pairs: Dict[str, str] = {}


@transport_registry.register(
    "mp",
    transport_state=AsyncMPTransportState,
    server_connection=AsyncMPConnection,
    server_connection_state=AsyncMPConnectionState,
    client_connection=AsyncMPConnection,
    client_connection_state=AsyncMPConnectionState,
)
class AsyncMPTransport(AsyncTransport):
    transport_type: str = "mp"

    def __init__(self):
        super().__init__()
        self._server_connections: Dict[str, AsyncMPConnection] = {}
        self._client_connections: Dict[str, AsyncMPConnection] = {}
        self._pairs: Dict[str, str] = {}

    def _create_server_connection(self, **kwargs) -> ServerConnection:
        raise NotImplementedError("MP connections are created via create_child_pipe")

    def _create_client_connection(self, **kwargs) -> ClientConnection:
        raise NotImplementedError("MP connections are created via create_child_pipe")

    def create_child_pipe(
        self,
        parent_id: str,
        parent_secret: str,
        child_id: str,
        child_secret: str,
    ) -> Tuple[AsyncMPConnection, AsyncMPConnection]:
        raw_a, raw_b = multiprocessing.Pipe()
        server_conn = AsyncMPConnection(
            identity=parent_id, secret_id=parent_secret, pipe_conn=raw_a
        )
        client_conn = AsyncMPConnection(
            identity=child_id, secret_id=child_secret, pipe_conn=raw_b
        )
        server_key = f"{parent_id}:{parent_secret}"
        client_key = f"{child_id}:{child_secret}"
        self._server_connections[server_key] = server_conn
        self._client_connections[client_key] = client_conn
        self._pairs[server_key] = client_key
        return server_conn, client_conn

    def get_state(self) -> Optional[AsyncMPTransportState]:
        return AsyncMPTransportState(
            transport_type="mp",
            server_connections={
                k: v.get_state() for k, v in self._server_connections.items()
            },
            client_connections={
                k: v.get_state() for k, v in self._client_connections.items()
            },
            pairs=self._pairs.copy(),
        )

    @classmethod
    def set_state(cls, state: AsyncMPTransportState) -> "AsyncMPTransport":
        transport = cls()
        transport._pairs = state.pairs
        for server_key, client_key in state.pairs.items():
            server_id = server_key.split(":")[0]
            server_secret = server_key.split(":")[1]
            #
            client_id = client_key.split(":")[0]
            client_secret = client_key.split(":")[1]
            transport.create_child_pipe(
                server_id, server_secret, client_id, client_secret
            )
        return transport
