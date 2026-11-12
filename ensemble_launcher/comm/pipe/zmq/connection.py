"""The ZMQ connections (front ends): ROUTER as a server, DEALER as a client. Their sockets
live in `ZMQRouterIOProcess` / `ZMQDealerIOProcess` (``zmq_io_process.py``)."""

from typing import Dict, Optional

from ..async_connection import (
    ClientConnection,
    ClientConnectionState,
    ServerConnection,
    ServerConnectionState,
)
from .io_process import ZMQDealerIOProcess, ZMQRouterIOProcess


# ---------------------------------------------------------------------------
#  ZMQ ROUTER (Server)
# ---------------------------------------------------------------------------


class AsyncZMQRouterConnectionState(ServerConnectionState):
    transport_type: str = "zmq"
    address: str
    lightweight: bool = False
    io_process: bool = False


class AsyncZMQRouterConnection(ServerConnection):
    transport_type: str = "zmq"
    supports_io_process = True

    def __init__(
        self,
        identity: str,
        secret_id: str,
        address: str,
        expected_remotes: Optional[Dict[str, str]] = None,
        req_res: bool = False,
        lightweight: bool = False,
        io_process: bool = False,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            expected_remotes=expected_remotes,
            req_res=req_res,
            io_process=io_process,
        )
        self._address = address
        self._lightweight = lightweight

    @property
    def address(self) -> str:
        return self._address

    def _make_io(self) -> ZMQRouterIOProcess:
        return ZMQRouterIOProcess(
            self._identity,
            self._secret_id,
            req_res=self._req_res,
            expected_remotes=self._expected_remotes,
            sender_rule=self._unknown_sender_validator,
            address=self._address,
            lightweight=self._lightweight,
        )

    def _on_started(self, address: Optional[str]) -> None:
        if address:
            self._address = address

    def get_state(self) -> AsyncZMQRouterConnectionState:
        return AsyncZMQRouterConnectionState(
            identity=self._identity,
            secret_id=self._secret_id,
            address=self._address,
            expected_remotes=self.expected_remotes,
            req_res=self._req_res,
            lightweight=self._lightweight,
            io_process=self._io_process,
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
            io_process=state.io_process,
        )



# ---------------------------------------------------------------------------
#  ZMQ DEALER (Client)
# ---------------------------------------------------------------------------


class AsyncZMQDealerConnectionState(ClientConnectionState):
    transport_type: str = "zmq"
    remote_address: str
    lightweight: bool = False
    io_process: bool = False


class AsyncZMQDealerConnection(ClientConnection):
    transport_type: str = "zmq"
    supports_io_process = True

    def __init__(
        self,
        identity: str,
        secret_id: str,
        remote_address: str,
        remote_identity: Optional[str] = None,
        remote_secret_id: Optional[str] = None,
        req_res: bool = False,
        lightweight: bool = False,
        io_process: bool = False,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            remote_identity=remote_identity,
            remote_secret_id=remote_secret_id,
            req_res=req_res,
            io_process=io_process,
        )
        self._remote_address = remote_address
        self._lightweight = lightweight

    @property
    def remote_address(self) -> str:
        return self._remote_address

    def _make_io(self) -> ZMQDealerIOProcess:
        return ZMQDealerIOProcess(
            self._identity,
            self._secret_id,
            req_res=self._req_res,
            remote_identity=self._remote_identity,
            remote_secret_id=self._remote_secret_id,
            remote_address=self._remote_address,
            lightweight=self._lightweight,
        )

    def get_state(self) -> AsyncZMQDealerConnectionState:
        return AsyncZMQDealerConnectionState(
            identity=self._identity,
            secret_id=self._secret_id,
            remote_address=self._remote_address,
            remote_identity=self._remote_identity,
            remote_secret_id=self._remote_secret_id,
            req_res=self._req_res,
            lightweight=self._lightweight,
            io_process=self._io_process,
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
            io_process=state.io_process,
        )



__all__ = [
    "AsyncZMQRouterConnection",
    "AsyncZMQRouterConnectionState",
    "AsyncZMQDealerConnection",
    "AsyncZMQDealerConnectionState",
]
