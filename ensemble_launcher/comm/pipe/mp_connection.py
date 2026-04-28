import asyncio
from typing import List

from .async_connection import (
    ServerConnection,
    ServerConnectionState,
)


class AsyncMPConnectionState(ServerConnectionState):
    transport_type: str = "mp"


class AsyncMPConnection(ServerConnection):
    """Wraps one end of a multiprocessing.Pipe.

    Frames messages to match ZMQ wire format: send prepends own identity:secret_id,
    recv returns [sender_identity_frame, data].
    """

    transport_type: str = "mp"

    def __init__(self, identity: str, secret_id: str, pipe_conn):
        super().__init__(identity=identity, secret_id=secret_id)
        self._conn = pipe_conn
        self._identity_frame = f"{identity}:{secret_id}".encode()
        self._is_open = True

    @property
    def is_open(self) -> bool:
        return self._is_open and not self._conn.closed

    async def open(self) -> None:
        pass

    async def send(self, data: bytes, target_id: str = None) -> bool:
        header = len(self._identity_frame).to_bytes(4, "big")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._conn.send_bytes, header + self._identity_frame + data
        )
        return True

    async def recv(self) -> List[bytes]:
        loop = asyncio.get_running_loop()
        raw = await loop.run_in_executor(None, self._conn.recv_bytes)
        id_len = int.from_bytes(raw[:4], "big")
        identity = raw[4 : 4 + id_len]
        data = raw[4 + id_len :]
        return [identity, data]

    async def close(self) -> None:
        self._is_open = False
        self._conn.close()

    def get_state(self) -> AsyncMPConnectionState:
        return AsyncMPConnectionState(
            transport_type="mp", identity=self._identity, secret_id=self._secret_id
        )

    @classmethod
    def set_state(cls, state: AsyncMPConnectionState) -> "AsyncMPConnection":
        raise NotImplementedError("Can't set state directly for mp")
