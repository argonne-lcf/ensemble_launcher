import logging
from typing import Dict, Optional

from ..async_connection import (
    ServerConnection,
    ServerConnectionState,
)
from .io_process import MPIOProcess

logger = logging.getLogger(__name__)


class AsyncMPConnectionState(ServerConnectionState):
    transport_type: str = "mp"


class AsyncMPConnection(ServerConnection):
    """Wraps one end of a multiprocessing.Pipe (its IOProcess always runs in process)."""

    transport_type: str = "mp"

    def __init__(
        self,
        identity: str,
        secret_id: str,
        pipe_conn,
        expected_remotes: Optional[Dict[str, str]] = None,
        req_res: bool = False,
    ):
        super().__init__(
            identity=identity,
            secret_id=secret_id,
            expected_remotes=expected_remotes,
            req_res=req_res,
        )
        self._conn = pipe_conn

    def _make_io(self) -> MPIOProcess:
        return MPIOProcess(
            self._identity,
            self._secret_id,
            req_res=self._req_res,
            expected_remotes=self._expected_remotes,
            sender_rule=self._unknown_sender_validator,
            pipe_conn=self._conn,
        )

    def get_state(self) -> AsyncMPConnectionState:
        return AsyncMPConnectionState(
            transport_type="mp",
            identity=self._identity,
            secret_id=self._secret_id,
            req_res=self._req_res,
        )

    @classmethod
    def set_state(cls, state: AsyncMPConnectionState) -> "AsyncMPConnection":
        raise NotImplementedError("Can't set state directly for mp")
