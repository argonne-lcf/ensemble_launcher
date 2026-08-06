import asyncio
import logging
from typing import Dict, List, Optional, Union

from .async_connection import (
    ServerConnection,
    ServerConnectionState,
)

logger = logging.getLogger(__name__)


class AsyncMPConnectionState(ServerConnectionState):
    transport_type: str = "mp"


class AsyncMPConnection(ServerConnection):
    """Wraps one end of a multiprocessing.Pipe.

    Sends/receives lists of byte frames to match the ZMQ multipart convention.
    """

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
        self._identity_frame = f"{identity}:{secret_id}".encode()

    async def _raw_open(self) -> None:
        pass

    async def close(self):
        self._conn.close()
        await super().close()

    async def _raw_close(self) -> None:
        if not self._conn.closed:
            self._conn.close()

    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        """Send frames via multiprocessing pipe.

        Data frames are joined into a single blob before sending.
        Sends: [identity_frame, msg_id(8B)?, blob]
        """
        blob = b"".join(data) if isinstance(data, list) else data
        if msg_id is not None:
            frames = [self._identity_frame, msg_id.to_bytes(8, "big"), blob]
        else:
            frames = [self._identity_frame, blob]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._conn.send, frames)
        return True

    def _poll_recv(self):
        while not self._conn.closed:
            if self._conn.poll(0.5):
                return self._conn.recv()
        raise OSError("pipe closed")

    async def _raw_recv(self) -> List[bytes]:
        """Receive frames via multiprocessing pipe.

        Returns: [identity_frame, msg_id(8B)?, blob]
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._poll_recv)

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
