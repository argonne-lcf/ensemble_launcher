"""The ZMQ transport: makes and tracks a node's ROUTER / DEALER connections."""

import socket
from typing import Dict, Optional, Tuple

from ..async_transport import AsyncTransport, AsyncTransportState, get_hsn_ip_cli
from ..registry import transport_registry
from ..utils import find_free_port
from .connection import (
    AsyncZMQDealerConnection,
    AsyncZMQDealerConnectionState,
    AsyncZMQRouterConnection,
    AsyncZMQRouterConnectionState,
)


class AsyncZMQTransportState(AsyncTransportState):
    transport_type: str = "zmq"
    server_connections: Dict[str, AsyncZMQRouterConnectionState] = {}
    client_connections: Dict[str, AsyncZMQDealerConnectionState] = {}
    hostname: str
    io_process: bool = False


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

    def __init__(self, lightweight: bool = False, io_process: bool = False):
        """``io_process``: req_res connections made here run their IOProcess (socket, ACKs,
        sends) in a process of its own."""
        super().__init__()
        self._io_process = io_process
        hn = get_hsn_ip_cli() or socket.gethostname()
        hostname = "localhost" if "local" in hn else hn
        self._hostname = hostname
        self._lightweight = lightweight
        self._server_connections: Dict[str, AsyncZMQRouterConnection] = {}
        self._client_connections: Dict[str, AsyncZMQDealerConnection] = {}

    def _create_server_connection(
        self, identity: str, secret_id: str, req_res: bool = False, **kwargs
    ) -> AsyncZMQRouterConnection:
        key = f"{identity}:{secret_id}"
        router = self._server_connections.get(key, None)

        if router is None:
            free_port = find_free_port((10000, 30000), self._hostname)
            address = (
                kwargs.get("address")
                or f"{self._hostname}:{free_port}"
            )
            expected_remotes = kwargs.get("expected_remotes", None)
            router = AsyncZMQRouterConnection(
                identity=identity,
                secret_id=secret_id,
                address=address,
                req_res=req_res,
                expected_remotes=expected_remotes,
                lightweight=self._lightweight,
                io_process=self._io_process,
            )
            self._server_connections[key] = router
        return router

    def _create_client_connection(
        self, identity: str, secret_id: str, req_res: bool = False, **kwargs
    ) -> AsyncZMQDealerConnection:
        key = f"{identity}:{secret_id}"
        dealer = self._client_connections.get(key, None)
        if dealer is None:
            remote_address = kwargs["remote_address"]
            dealer = AsyncZMQDealerConnection(
                identity=identity,
                secret_id=secret_id,
                req_res=req_res,
                remote_address=remote_address,
                remote_identity=kwargs.get("remote_identity"),
                remote_secret_id=kwargs.get("remote_secret_id"),
                lightweight=self._lightweight,
                io_process=self._io_process,
            )
            self._client_connections[key] = dealer
        return dealer

    def create_child_pipe(
        self,
        parent_id: str,
        parent_secret: str,
        child_id: str,
        child_secret: str,
        req_res: bool = False,
    ) -> Tuple[AsyncZMQRouterConnection, AsyncZMQDealerConnection]:
        server = self.get_server_connection(
            parent_id, parent_secret, req_res=req_res, address=None
        )
        server.add_expected_remote(child_id, child_secret)
        client = self.get_client_connection(
            child_id,
            child_secret,
            req_res=req_res,
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
        req_res: bool = False,
    ):
        if cls is AsyncZMQRouterConnection:
            return self._create_server_connection(
                identity, secret_id, req_res=req_res, address=address
            )
        elif cls is AsyncZMQDealerConnection:
            if remote_address is None:
                raise ValueError(f"Need remote address to create {cls}")
            return self._create_client_connection(
                identity, secret_id, req_res=req_res, remote_address=remote_address
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
            io_process=self._io_process,
        )

    @classmethod
    def set_state(cls, state: AsyncZMQTransportState) -> "AsyncZMQTransport":
        transport = cls(io_process=state.io_process)
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
