"""The socket half of a connection, joined to its front end by a queue.

An `IOProcess` owns a connection's socket and everything that touches it: the receive loop,
sender verification, the ACK protocol (sending ACKs for received messages, and waiting for the
peer's ACK for our own sends, with the timeout) and msg ids. The front end (`AsyncConnection`)
and the IOProcess each hold one end of a two-way queue; each end ``put_nowait``s to the other
end and ``get``s what the other end put:

    IOProcess -> front end  ("ready", address)       started; the actual bound address
                            ("data", frames)         a received message, verified and ACKed
                            ("sent", req_id, ok)     final result of one send
    front end -> IOProcess  ("send", req_id, target, timeout, frames)
                            ("control", op, args)    "add_remote" / "remove_remote" / "close"

The IOProcess runs either on the caller's event loop, with a `LocalIOQueue` pair (two
`asyncio.Queue`s), or in its own process, with a `ZMQIOQueue` on each side (a PUSH and a PULL
socket over two ``ipc://`` paths). It doesn't know which: the items are the whole contract. Out of process, a busy event loop in
the owner can no longer delay ACKs.
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import secrets
import signal
import struct
import tempfile
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel

_ACK = b"_ACK"


def decode_identity(raw: bytes):
    full_id = raw.decode()
    parts = full_id.split(":", 1)
    return full_id, parts[0], parts[1] if len(parts) > 1 else None


# --------------------------------------------------------------------------- #
#  Sender verification (shared by the front ends and the IOProcess)             #
# --------------------------------------------------------------------------- #


class SenderRule(BaseModel):
    """Accept a sender whose id starts with ``prefix`` (and whose secret is ``secret``, if
    set). A picklable stand-in for a validator closure, so it can reach a spawned IOProcess."""

    prefix: str
    secret: Optional[str] = None

    def __call__(self, sender_id: str, sender_secret: Optional[str]) -> bool:
        if not sender_id.startswith(self.prefix):
            return False
        return self.secret is None or sender_secret == self.secret


def verify_server_sender(
    expected_remotes: Dict[str, str],
    validator: Optional[Callable[[str, Optional[str]], bool]],
    sender_id: str,
    sender_secret: Optional[str],
) -> bool:
    """A server accepts anyone when it expects no one; otherwise known remotes with their
    secret, and unknown senders only if ``validator`` says so."""
    if not expected_remotes:
        return True
    expected_secret = expected_remotes.get(sender_id)
    if expected_secret is None:
        return validator(sender_id, sender_secret) if validator is not None else False
    return sender_secret == expected_secret


def verify_client_sender(
    remote_identity: Optional[str],
    remote_secret_id: Optional[str],
    sender_id: str,
    sender_secret: Optional[str],
) -> bool:
    """A client accepts only its remote (and its secret, if known); anyone if it has none."""
    if remote_identity is None:
        return True
    if sender_id != remote_identity:
        return False
    if remote_secret_id is not None and sender_secret != remote_secret_id:
        return False
    return True


# --------------------------------------------------------------------------- #
#  Queues                                                                       #
# --------------------------------------------------------------------------- #


class LocalIOQueue:
    """One end of an in-process two-way queue: ``put_nowait`` feeds the other end's ``get``.
    Make both ends with `pair`."""

    def __init__(self, send_q: asyncio.Queue, recv_q: asyncio.Queue):
        self._send_q = send_q
        self._recv_q = recv_q

    @classmethod
    def pair(cls) -> Tuple["LocalIOQueue", "LocalIOQueue"]:
        a_to_b, b_to_a = asyncio.Queue(), asyncio.Queue()
        return cls(a_to_b, b_to_a), cls(b_to_a, a_to_b)

    def open(self) -> None:
        pass

    def put_nowait(self, item: tuple) -> None:
        self._send_q.put_nowait(item)

    async def get(self, timeout: Optional[float] = None) -> Optional[tuple]:
        if timeout is None:
            return await self._recv_q.get()
        try:
            return await asyncio.wait_for(self._recv_q.get(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None

    def close(self, linger: int = 0) -> None:
        pass


def _encode(item: tuple) -> List[Any]:
    kind = item[0]
    if kind == "data":
        return [b"data", *item[1]]
    if kind == "sent":
        return [b"sent", item[1].to_bytes(8, "big"), b"\x01" if item[2] else b"\x00"]
    if kind == "send":
        _, req_id, target, timeout, frames = item
        return [
            b"send",
            req_id.to_bytes(8, "big"),
            (target or "").encode(),
            struct.pack("!d", timeout),
            *frames,
        ]
    if kind == "ready":
        return [b"ready", (item[1] or "").encode()]
    if kind == "control":
        return [b"control", item[1].encode(), json.dumps(list(item[2])).encode()]
    raise ValueError(f"Unknown queue item {kind!r}")


def _decode(frames: List[bytes]) -> tuple:
    kind = frames[0]
    if kind == b"data":
        return ("data", frames[1:])
    if kind == b"sent":
        return ("sent", int.from_bytes(frames[1], "big"), frames[2] == b"\x01")
    if kind == b"send":
        target = frames[2].decode() or None
        return (
            "send",
            int.from_bytes(frames[1], "big"),
            target,
            struct.unpack("!d", frames[3])[0],
            frames[4:],
        )
    if kind == b"ready":
        return ("ready", frames[1].decode() or None)
    if kind == b"control":
        return ("control", frames[1].decode(), tuple(json.loads(frames[2])))
    raise ValueError(f"Unknown queue item {kind!r}")


class ZMQIOQueue:
    """One end of a cross-process two-way queue: a PUSH socket on ``send_path``
    (``put_nowait``) and a PULL socket on ``recv_path`` (``get``). The other end is built with
    the same two paths reversed, so each end's puts reach the other end's gets; one end binds,
    the other connects. HWM 0: the hop is local, so nothing is dropped or blocks. Items travel
    as multipart frames, data frames as they are. Only the paths are needed to build an end, so
    the connect side is made from them in the spawned process."""

    def __init__(self, send_path: str, recv_path: str, bind: bool):
        self._send_path = send_path
        self._recv_path = recv_path
        self._bind = bind
        self._ctx = None
        self._send_sock = None
        self._recv_sock = None

    def open(self) -> None:
        import zmq
        import zmq.asyncio

        self._ctx = zmq.asyncio.Context()
        self._send_sock = self._ctx.socket(zmq.PUSH)
        self._recv_sock = self._ctx.socket(zmq.PULL)
        for path, sock in ((self._send_path, self._send_sock), (self._recv_path, self._recv_sock)):
            sock.setsockopt(zmq.SNDHWM, 0)
            sock.setsockopt(zmq.RCVHWM, 0)
            sock.setsockopt(zmq.LINGER, 0)
            (sock.bind if self._bind else sock.connect)(f"ipc://{path}")

    def put_nowait(self, item: tuple) -> None:
        fut = self._send_sock.send_multipart(_encode(item))
        if hasattr(fut, "add_done_callback"):
            fut.add_done_callback(_log_send_error)

    async def get(self, timeout: Optional[float] = None) -> Optional[tuple]:
        if timeout is not None and not await self._recv_sock.poll(int(timeout * 1000)):
            return None
        return _decode(await self._recv_sock.recv_multipart())

    def close(self, linger: int = 0) -> None:
        """``linger`` (ms) lets queued puts reach the other end before the socket goes."""
        for sock in (self._send_sock, self._recv_sock):
            if sock is not None:
                sock.close(linger if sock is self._send_sock else 0)
        self._send_sock = self._recv_sock = None
        if self._ctx is not None:
            self._ctx.term()
            self._ctx = None


def _log_send_error(fut) -> None:
    if not fut.cancelled() and fut.exception() is not None:
        logging.getLogger(__name__).warning(f"IO queue put failed: {fut.exception()}")


def new_ipc_path(suffix: str) -> str:
    """A short ``ipc://`` socket path (unix sockets allow at most 107 characters)."""
    name = f"el-{secrets.token_hex(4)}-{suffix}.ipc"
    path = os.path.join(tempfile.gettempdir(), name)
    return path if len(path) <= 100 else os.path.join("/tmp", name)


# --------------------------------------------------------------------------- #
#  IOProcess                                                                    #
# --------------------------------------------------------------------------- #


class IOProcess(ABC):
    """A connection's socket, receive loop, sender verification and ACK protocol.

    Subclasses provide the transport (`_raw_open` / `_raw_close` / `_raw_send` / `_raw_recv`,
    `address`). ``server`` picks the sender check: servers use ``expected_remotes`` plus
    ``sender_rule``; clients use ``remote_identity`` / ``remote_secret_id``.
    """

    server: bool = True

    def __init__(
        self,
        identity: str,
        secret_id: str,
        req_res: bool = False,
        expected_remotes: Optional[Dict[str, str]] = None,
        sender_rule: Optional[Callable[[str, Optional[str]], bool]] = None,
        remote_identity: Optional[str] = None,
        remote_secret_id: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self._identity = identity
        self._secret_id = secret_id
        self._req_res = req_res
        self._expected_remotes: Dict[str, str] = dict(expected_remotes or {})
        self._sender_rule = sender_rule
        self._remote_identity = remote_identity
        self._remote_secret_id = remote_secret_id
        self.logger = logger or logging.getLogger(f"io-{identity}")
        self._msg_counter = 0
        self._ack_futures: Dict[int, asyncio.Future] = {}
        self._stop: Optional[asyncio.Event] = None
        self._send_tasks: set = set()

    # ---------------- what a spawned process needs to rebuild this ---------------- #

    def kwargs(self) -> Dict[str, Any]:
        """Constructor arguments (minus the logger) -- picklable, for a spawned process."""
        return {
            "identity": self._identity,
            "secret_id": self._secret_id,
            "req_res": self._req_res,
            "expected_remotes": dict(self._expected_remotes),
            "sender_rule": self._sender_rule,
            "remote_identity": self._remote_identity,
            "remote_secret_id": self._remote_secret_id,
        }

    # ------------------------------- main loop -------------------------------- #

    async def run(self, q) -> None:
        """Open the socket, report ``ready``, and serve the queue until ``close``."""
        self._stop = asyncio.Event()
        await self._raw_open()
        q.put_nowait(("ready", self.address))
        recv = asyncio.create_task(self._recv_loop(q))
        inp = asyncio.create_task(self._input_loop(q))
        try:
            await self._stop.wait()
        finally:
            for t in (recv, inp, *self._send_tasks):
                t.cancel()
            await asyncio.gather(recv, inp, *self._send_tasks, return_exceptions=True)
            for fut in self._ack_futures.values():
                if not fut.done():
                    fut.set_result(False)
            self._ack_futures.clear()
            await self._raw_close()

    async def _recv_loop(self, q) -> None:
        """Read the socket; verify the sender; resolve ACKs for our sends, ACK and forward
        everything else. Without req_res, frames are forwarded as they come."""
        while True:
            try:
                frames = await self._raw_recv()
                if not self._req_res:
                    q.put_nowait(("data", frames))
                    continue
                _, sender_id, sender_secret = decode_identity(frames[0])
                if not self.verify_sender(sender_id, sender_secret):
                    self.logger.warning(
                        f"{self._identity}: Discarding message from {sender_id} "
                        f"— secret_id mismatch (stale connection)"
                    )
                    continue
                if frames[-1] == _ACK:
                    msg_id = int.from_bytes(frames[-2], "big")
                    fut = self._ack_futures.pop(msg_id, None)
                    if fut is not None and not fut.done():
                        fut.set_result(True)
                    elif fut is None:
                        self.logger.warning(
                            f"Received ACK for unknown msg_id={msg_id} from {sender_id}"
                        )
                    continue
                await self._raw_send(
                    _ACK,
                    msg_id=int.from_bytes(frames[1], "big"),
                    target_id=frames[0].decode(),
                )
                q.put_nowait(("data", [frames[0]] + frames[2:]))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"_recv_loop failed with exception {e}")

    async def _input_loop(self, q) -> None:
        """Serve send requests (each in its own task, so a slow ACK wait blocks nothing) and
        control requests."""
        while True:
            item = await q.get()
            kind = item[0]
            if kind == "send":
                _, req_id, target, timeout, frames = item
                task = asyncio.create_task(self._send(frames, target, timeout))
                self._send_tasks.add(task)
                task.add_done_callback(
                    lambda t, rid=req_id: self._send_done(q, rid, t)
                )
            elif kind == "control":
                op, args = item[1], item[2]
                if op == "add_remote":
                    self.add_remote(*args)
                elif op == "remove_remote":
                    self.remove_remote(*args)
                elif op == "set_rule":
                    rule = args[0]
                    self._sender_rule = SenderRule(**rule) if isinstance(rule, dict) else rule
                elif op == "close":
                    self._stop.set()
                    return

    def _send_done(self, q, req_id: int, task: asyncio.Task) -> None:
        self._send_tasks.discard(task)
        if task.cancelled():
            return
        ok = False if task.exception() is not None else bool(task.result())
        if task.exception() is not None:
            self.logger.warning(f"Send failed: {task.exception()}")
        q.put_nowait(("sent", req_id, ok))

    async def _send(
        self, data: Union[bytes, List[bytes]], target_id: Optional[str], timeout: float
    ) -> bool:
        """One send: with req_res, a msg id, the raw send, then wait for the peer's ACK."""
        self._msg_counter += 1
        if not self._req_res:
            return await self._raw_send(data, target_id=target_id)
        msg_id = self._msg_counter
        fut = asyncio.get_running_loop().create_future()
        self._ack_futures[msg_id] = fut
        if not await self._raw_send(data, msg_id=msg_id, target_id=target_id):
            self.logger.warning("Raw send failed")
            self._ack_futures.pop(msg_id, None)
            return False
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            self._ack_futures.pop(msg_id, None)
            self.logger.warning(f"ACK timeout for msg_id={msg_id} to {target_id}")
            return False

    # ------------------------------ verification ------------------------------ #

    def add_remote(self, node_id: str, secret_id: str) -> None:
        self._expected_remotes[node_id] = secret_id

    def remove_remote(self, node_id: str) -> None:
        self._expected_remotes.pop(node_id, None)

    def verify_sender(self, sender_id: str, sender_secret: Optional[str] = None) -> bool:
        if self.server:
            return verify_server_sender(
                self._expected_remotes, self._sender_rule, sender_id, sender_secret
            )
        return verify_client_sender(
            self._remote_identity, self._remote_secret_id, sender_id, sender_secret
        )

    # ------------------------------- transport -------------------------------- #

    @property
    @abstractmethod
    def address(self) -> Optional[str]: ...

    @abstractmethod
    async def _raw_open(self) -> None: ...

    @abstractmethod
    async def _raw_close(self) -> None: ...

    @abstractmethod
    async def _raw_send(
        self,
        data: Union[bytes, List[bytes]],
        msg_id: Optional[int] = None,
        target_id: Optional[str] = None,
    ) -> bool: ...

    @abstractmethod
    async def _raw_recv(self) -> List[bytes]: ...


# --------------------------------------------------------------------------- #
#  Running an IOProcess in its own process                                      #
# --------------------------------------------------------------------------- #


def start_io_process(io: IOProcess) -> Tuple[Any, ZMQIOQueue, List[str]]:
    """Spawn ``io``'s class in a new process, joined to the caller by a `ZMQIOQueue`.

    Returns ``(process, q, paths)``: the caller's (bind) end of the queue, already open; on
    close the caller closes it and unlinks ``paths``.
    """
    import multiprocessing as mp

    kwargs = io.kwargs()
    rule = kwargs.get("sender_rule")
    if rule is not None and not isinstance(rule, SenderRule):
        raise TypeError(
            "An out-of-process connection needs a SenderRule as its unknown-sender "
            f"validator, not {type(rule).__name__}"
        )
    in_path, out_path = new_ipc_path("in"), new_ipc_path("out")  # into / out of the IOProcess
    q = ZMQIOQueue(send_path=in_path, recv_path=out_path, bind=True)
    q.open()
    class_path = f"{type(io).__module__}:{type(io).__qualname__}"
    proc = mp.get_context("spawn").Process(
        target=_io_main,
        args=(class_path, kwargs, out_path, in_path, os.getpid()),  # reversed: its send, its recv
        name=f"io-{kwargs['identity']}",
        daemon=True,
    )
    proc.start()
    return proc, q, [in_path, out_path]


def _io_main(class_path: str, kwargs: dict, send_path: str, recv_path: str, parent_pid: int):
    """Entry point of a spawned IOProcess: serve the queue until ``close`` or the owner
    dies. Signals are left to the owner, which closes us."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    module, _, qualname = class_path.partition(":")
    cls = getattr(importlib.import_module(module), qualname)

    from ensemble_launcher.logging import setup_logger

    logger = setup_logger(name=f"connection-{kwargs['identity']}-io", subdir="connections")
    io = cls(**kwargs, logger=logger)

    async def main():
        q = ZMQIOQueue(send_path, recv_path, bind=False)
        q.open()

        async def watch_parent():
            while os.getppid() == parent_pid:
                await asyncio.sleep(1.0)
            logger.warning(f"{kwargs['identity']}: owner {parent_pid} is gone; exiting")
            io._stop.set()

        watcher = None
        try:
            run = asyncio.create_task(io.run(q))
            while io._stop is None and not run.done():
                await asyncio.sleep(0)
            watcher = asyncio.create_task(watch_parent())
            await run
        finally:
            if watcher is not None:
                watcher.cancel()
            q.close(linger=1000)  # let the last results reach the owner

    asyncio.run(main())
