"""Connections: a front end (`AsyncConnection`) plus an `IOProcess` that owns the socket.

This module holds the transport-independent front ends; each transport has its own
``<transport>_connection.py`` (front ends) and ``<transport>_io_process.py`` (IOProcesses).

The front end is the API the rest of the code uses -- ``open`` / ``send`` / ``recv`` / state /
expected remotes. It knows nothing about sockets or ACKs: ``send`` puts a request on the
IOProcess's input queue and awaits its result; ``recv`` reads messages the IOProcess put on its
output queue. The IOProcess (``io_process.py``) runs on the same event loop, or -- with
``io_process=True`` and ``req_res=True`` on the ZMQ connections -- in its own process, so a busy
event loop can't delay ACKs.
"""

import asyncio
import threading
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Type, TypeVar, Union

from pydantic import BaseModel, Field

from ensemble_launcher.logging import setup_logger

from .io_process import (
    _ACK,  # noqa: F401  (re-exported for older imports)
    IOProcess,
    LocalIOQueue,
    SenderRule,
    decode_identity,
    start_io_process,
    verify_client_sender,
    verify_server_sender,
)

T = TypeVar("T", bound="AsyncConnectionState")


class IdentityVerificationError(Exception):
    pass


class AsyncConnectionState(BaseModel):
    transport_type: str
    identity: str
    secret_id: str
    req_res: bool = False

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
    """The front end: puts requests on its IOProcess's input queue and listens to its output
    queue. Subclasses say how to build their IOProcess (`_make_io`)."""

    transport_type: str = ""
    # Whether this connection's IOProcess can be spawned in its own process.
    supports_io_process: bool = False

    def __init__(
        self,
        identity: str,
        secret_id: str,
        req_res: bool = False,
        io_process: bool = False,
    ):
        self._identity = identity
        self._secret_id = secret_id
        self._is_open = False
        self._req_res = req_res
        self._io_process = io_process
        self._msg_queue: asyncio.Queue = None
        self._loop: asyncio.AbstractEventLoop = None
        self._loop_thread: threading.Thread = None
        self.logger = None
        # The IOProcess and the two queues to it.
        self._in_q = None
        self._out_q = None
        self._io_task: Optional[asyncio.Task] = None  # in process
        self._proc = None  # out of process
        self._io_ctx = None
        self._io_paths: List[str] = []
        self._listen_task: Optional[asyncio.Task] = None
        self._pending: Dict[int, asyncio.Future] = {}  # req_id -> send result
        self._req_counter = 0

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def identity(self):
        return self._identity

    @property
    def secret_id(self):
        return self._secret_id

    @property
    def out_of_process(self) -> bool:
        """The IOProcess runs in its own process (only req_res connections that support it)."""
        return self._io_process and self._req_res and self.supports_io_process

    # ------------------------------------------------------------------ #
    #  Open / close                                                        #
    # ------------------------------------------------------------------ #

    async def open(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._msg_queue = asyncio.Queue()
        self._pending = {}
        self.logger = setup_logger(
            name=f"connection-{self.identity}",
            subdir="connections",
        )
        io = self._make_io()
        if self.out_of_process:
            self._proc, self._in_q, self._out_q, self._io_ctx, self._io_paths = (
                start_io_process(io)
            )
        else:
            io.logger = self.logger
            self._in_q, self._out_q = LocalIOQueue(), LocalIOQueue()
            self._io_task = self._loop.create_task(io.run(self._in_q, self._out_q))
        address = await self._wait_ready(timeout=60.0)
        self._on_started(address)
        self._is_open = True
        self._listen_task = self._loop.create_task(self._listen())
        if self._io_task is not None:
            self._io_task.add_done_callback(self._on_io_task_done)

    async def _wait_ready(self, timeout: float) -> Optional[str]:
        """Wait for the IOProcess's ``ready``; raise if it died first."""
        deadline = self._loop.time() + timeout
        while self._loop.time() < deadline:
            item = await self._out_q.get(timeout=0.5)
            if item is not None:
                if item[0] == "ready":
                    return item[1]
                continue
            if self._io_task is not None and self._io_task.done():
                await self._teardown_io()
                self._io_task.result()  # raises what the IOProcess raised
                raise RuntimeError(f"{self.identity}: IOProcess ended before it was ready")
            if self._proc is not None and not self._proc.is_alive():
                code = self._proc.exitcode
                await self._teardown_io()
                raise RuntimeError(f"{self.identity}: IOProcess exited ({code}) before ready")
        await self._teardown_io()
        raise TimeoutError(f"{self.identity}: IOProcess not ready after {timeout}s")

    async def close(self):
        if self._in_q is None:
            return
        self._is_open = False
        try:
            self._in_q.put_nowait(("control", "close", ()))
        except Exception:
            pass
        if self._io_task is not None and not self._io_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._io_task), timeout=5.0)
            except (asyncio.TimeoutError, TimeoutError, Exception):
                self._io_task.cancel()
        if self._proc is not None:
            for _ in range(100):  # up to 5 s
                if not self._proc.is_alive():
                    break
                await asyncio.sleep(0.05)
            if self._proc.is_alive():
                self._proc.terminate()
                await asyncio.to_thread(self._proc.join, 1.0)
        await self._teardown_io()

    async def _teardown_io(self) -> None:
        if self._listen_task is not None:
            self._listen_task.cancel()
            try:
                await self._listen_task
            except (asyncio.CancelledError, Exception):
                pass
            self._listen_task = None
        self._fail_pending()
        for q in (self._in_q, self._out_q):
            if q is not None:
                q.close()
        self._in_q = self._out_q = None
        if self._io_ctx is not None:
            self._io_ctx.term()
            self._io_ctx = None
        import os

        for path in self._io_paths:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._io_paths = []

    # ------------------------------------------------------------------ #
    #  The output queue                                                    #
    # ------------------------------------------------------------------ #

    async def _listen(self) -> None:
        """Received messages go to ``_msg_queue``; send results resolve their futures. Out of
        process, also notice the IOProcess dying."""
        timeout = 1.0 if self._proc is not None else None
        while True:
            item = await self._out_q.get(timeout=timeout)
            if item is None:
                if self._proc is not None and not self._proc.is_alive():
                    self._on_io_dead(f"IOProcess exited ({self._proc.exitcode})")
                    return
                continue
            kind = item[0]
            if kind == "data":
                self._msg_queue.put_nowait(item[1])
            elif kind == "sent":
                fut = self._pending.pop(item[1], None)
                if fut is not None and not fut.done():
                    fut.set_result(item[2])

    def _on_io_task_done(self, task: asyncio.Task) -> None:
        if self._is_open and not task.cancelled():
            exc = task.exception()
            self._on_io_dead(f"IOProcess ended: {exc!r}" if exc else "IOProcess ended")

    def _on_io_dead(self, why: str) -> None:
        if self.logger is not None:
            self.logger.error(f"{self.identity}: {why}")
        self._is_open = False
        self._fail_pending()

    def _fail_pending(self) -> None:
        for fut in self._pending.values():
            if not fut.done():
                fut.set_result(False)
        self._pending.clear()

    # ------------------------------------------------------------------ #
    #  API                                                                 #
    # ------------------------------------------------------------------ #

    async def recv(self, timeout: Optional[float] = None) -> List[bytes]:
        return await asyncio.wait_for(self._msg_queue.get(), timeout=timeout)

    async def send(
        self, msg: Union[bytes, List[bytes]], target_id: Optional[str] = None, timeout: float = 5.0
    ) -> bool:
        """Send ``msg``; with req_res, True once the peer ACKed it within ``timeout``. The
        IOProcess does the send and the ACK wait; this only waits for its answer."""
        if self._in_q is None or not self._is_open:
            if self.logger is not None:
                self.logger.warning(f"{self.identity}: send on a closed connection")
            return False
        self._req_counter += 1
        req_id = self._req_counter
        fut = self._loop.create_future()
        self._pending[req_id] = fut
        frames = msg if isinstance(msg, list) else [msg]
        self._in_q.put_nowait(("send", req_id, target_id, timeout, frames))
        return await fut

    def _control(self, op: str, *args) -> None:
        if self._in_q is not None and self._is_open:
            self._in_q.put_nowait(("control", op, args))

    # ---------------------- Sync wrappers ----------------------------------
    def sopen(self):
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        fut = asyncio.run_coroutine_threadsafe(self.open(), self._loop)
        return fut.result()

    def sclose(self):
        if self._loop and self._loop_thread:
            fut = asyncio.run_coroutine_threadsafe(self.close(), self._loop)
            fut.result()
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._loop_thread.join(timeout=10.0)

    def srecv(self) -> List[bytes]:
        return asyncio.run_coroutine_threadsafe(self.recv(), self._loop).result()

    def ssend(self, msg: bytes, target_id: Optional[str] = None, timeout: float = 5.0):
        return asyncio.run_coroutine_threadsafe(
            self.send(msg=msg, target_id=target_id, timeout=timeout), self._loop
        ).result()

    # ----------------------- For subclasses ---------------------------------

    @abstractmethod
    def _make_io(self) -> IOProcess:
        """This connection's IOProcess, built from its fields."""

    def _on_started(self, address: Optional[str]) -> None:
        """The IOProcess is up; ``address`` is what it actually bound to (servers)."""

    @abstractmethod
    def get_state(self) -> AsyncConnectionState:
        pass

    @classmethod
    @abstractmethod
    def set_state(cls, state: AsyncConnectionState) -> "AsyncConnection":
        pass

    @abstractmethod
    def verify_sender(self, sender_id, sender_secret) -> bool:
        pass


class ServerConnection(AsyncConnection):
    def __init__(
        self,
        identity: str,
        secret_id: str,
        expected_remotes: Optional[Dict[str, str]] = None,
        req_res: bool = False,
        io_process: bool = False,
    ):
        super().__init__(
            identity=identity, secret_id=secret_id, req_res=req_res, io_process=io_process
        )
        self._expected_remotes: Dict[str, str] = dict(expected_remotes or {})
        self._unknown_sender_validator: Optional[
            Callable[[str, Optional[str]], bool]
        ] = None

    @property
    def address(self) -> Optional[str]:
        return None

    @property
    def expected_remotes(self) -> Dict[str, str]:
        return dict(self._expected_remotes)

    def add_expected_remote(self, node_id: str, secret_id: str) -> None:
        self._expected_remotes[node_id] = secret_id
        self._control("add_remote", node_id, secret_id)

    def remove_expected_remote(self, node_id: str) -> None:
        self._expected_remotes.pop(node_id, None)
        self._control("remove_remote", node_id)

    def set_unknown_sender_validator(
        self, validator: Callable[[str, Optional[str]], bool]
    ) -> None:
        """Out of process, this must be a `SenderRule` (it has to reach the IOProcess). May be
        called after ``open()``: the running IOProcess gets the new rule."""
        self._unknown_sender_validator = validator
        if self._in_q is not None and self._is_open:
            if self.out_of_process:
                if not isinstance(validator, SenderRule):
                    raise TypeError(
                        "An out-of-process connection needs a SenderRule as its "
                        f"unknown-sender validator, not {type(validator).__name__}"
                    )
                self._control("set_rule", validator.model_dump())
            else:
                self._control("set_rule", validator)

    def verify_sender(
        self, sender_id: str, sender_secret: Optional[str] = None
    ) -> bool:
        return verify_server_sender(
            self._expected_remotes, self._unknown_sender_validator, sender_id, sender_secret
        )


class ClientConnection(AsyncConnection):
    def __init__(
        self,
        identity: str,
        secret_id: str,
        remote_identity: Optional[str] = None,
        remote_secret_id: Optional[str] = None,
        req_res: bool = False,
        io_process: bool = False,
    ):
        super().__init__(
            identity=identity, secret_id=secret_id, req_res=req_res, io_process=io_process
        )
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
        return verify_client_sender(
            self._remote_identity, self._remote_secret_id, sender_id, sender_secret
        )


__all__ = [
    "AsyncConnection",
    "AsyncConnectionState",
    "ServerConnection",
    "ServerConnectionState",
    "ClientConnection",
    "ClientConnectionState",
    "IdentityVerificationError",
    "SenderRule",
    "decode_identity",
]
