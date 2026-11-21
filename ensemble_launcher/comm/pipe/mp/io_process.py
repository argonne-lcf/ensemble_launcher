"""The multiprocessing.Pipe IOProcess (in process only)."""

import asyncio
import threading
from typing import List, Optional, Union

from ..io_process import IOProcess


class MPIOProcess(IOProcess):
    """One end of a multiprocessing.Pipe, as an IOProcess. In process only: the pipe end
    belongs to its owner.

    Sends/receives lists of byte frames to match the ZMQ multipart convention.
    """

    server = True

    def __init__(self, *args, pipe_conn, **kwargs):
        super().__init__(*args, **kwargs)
        self._conn = pipe_conn
        self._identity_frame = f"{self._identity}:{self._secret_id}".encode()
        self._stop_recv: Optional[threading.Event] = None

    @property
    def address(self) -> Optional[str]:
        return None

    async def _raw_open(self) -> None:
        self._stop_recv = threading.Event()

    async def _raw_close(self) -> None:
        if self._stop_recv is not None:
            self._stop_recv.set()
        if not self._conn.closed:
            self._conn.close()

    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool:
        """Send frames via multiprocessing pipe.

        Sends: [identity_frame, msg_id(8B)?, *data_frames]
        """
        if msg_id is not None:
            frames = [self._identity_frame, msg_id.to_bytes(8, "big")]
        else:
            frames = [self._identity_frame]
        if isinstance(data, list):
            frames.extend(data)
        else:
            frames.append(data)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._conn.send, frames)
        return True

    def _poll_recv(self):
        while not self._stop_recv.is_set() and not self._conn.closed:
            if self._conn.poll(0.5):
                return self._conn.recv()
        raise OSError("connection stopped")

    async def _raw_recv(self) -> List[bytes]:
        """Receive frames via multiprocessing pipe.

        Returns: [identity_frame, msg_id(8B)?, blob]
        """
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, self._poll_recv)
        except asyncio.CancelledError:
            self._stop_recv.set()
            raise



__all__ = ["MPIOProcess"]
