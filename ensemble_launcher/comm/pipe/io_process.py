"""The socket half of a connection, joined to its front end by two queues.

An `IOProcess` owns a connection's socket and everything that touches it: the receive loop,
sender verification, the ACK protocol (sending ACKs for received messages, and waiting for the
peer's ACK for our own sends, with the timeout) and msg ids. It reads an **input queue** (send
requests, control) and writes an **output queue** (received messages, send results):

    output  ("ready", address)                   started; the actual bound address
            ("data", frames)                     a received message, verified and ACKed
            ("sent", req_id, ok)                 final result of one send
    input   ("send", req_id, target, timeout, frames)
            ("control", op, args)                "add_remote" / "remove_remote" / "close"

The front end (`AsyncConnection`) only puts on the input queue and listens to the output queue.
The IOProcess runs either on the caller's event loop, with `LocalIOQueue`s (plain
`asyncio.Queue`), or in its own process, with `ZMQIOQueue`s (ZMQ PUSH/PULL over ``ipc://``). It
doesn't know which: the queue items are the whole contract. Out of process, a busy event loop in
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
    """In-process queue: a plain `asyncio.Queue` of item tuples."""

    def __init__(self):
        self._q: asyncio.Queue = asyncio.Queue()

    def put_nowait(self, item: tuple) -> None:
        self._q.put_nowait(item)

    async def get(self, timeout: Optional[float] = None) -> Optional[tuple]:
        if timeout is None:
            return await self._q.get()
        try:
            return await asyncio.wait_for(self._q.get(), timeout)
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
    """One direction of a cross-process queue: a ZMQ PUSH (put) or PULL (get) socket on an
    ``ipc://`` path. HWM 0 on both ends: the hop is local, so nothing is dropped or blocks.
    Items travel as multipart frames, data frames as they are."""

    def __init__(self, ctx, path: str, push: bool, bind: bool):
        import zmq

        self._sock = ctx.socket(zmq.PUSH if push else zmq.PULL)
        self._sock.setsockopt(zmq.SNDHWM, 0)
        self._sock.setsockopt(zmq.RCVHWM, 0)
        self._sock.setsockopt(zmq.LINGER, 0)
        (self._sock.bind if bind else self._sock.connect)(f"ipc://{path}")

    def put_nowait(self, item: tuple) -> None:
        fut = self._sock.send_multipart(_encode(item))
        if hasattr(fut, "add_done_callback"):
            fut.add_done_callback(_log_send_error)

    async def get(self, timeout: Optional[float] = None) -> Optional[tuple]:
        if timeout is not None and not await self._sock.poll(int(timeout * 1000)):
            return None
        return _decode(await self._sock.recv_multipart())

    def close(self, linger: int = 0) -> None:
        self._sock.close(linger)


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

    async def run(self, in_q, out_q) -> None:
        """Open the socket, report ``ready``, and serve both queues until ``close``."""
        self._stop = asyncio.Event()
        await self._raw_open()
        out_q.put_nowait(("ready", self.address))
        recv = asyncio.create_task(self._recv_loop(out_q))
        inp = asyncio.create_task(self._input_loop(in_q, out_q))
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

    async def _recv_loop(self, out_q) -> None:
        """Read the socket; verify the sender; resolve ACKs for our sends, ACK and forward
        everything else. Without req_res, frames are forwarded as they come."""
        while True:
            try:
                frames = await self._raw_recv()
                if not self._req_res:
                    out_q.put_nowait(("data", frames))
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
                out_q.put_nowait(("data", [frames[0]] + frames[2:]))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"_recv_loop failed with exception {e}")

    async def _input_loop(self, in_q, out_q) -> None:
        """Serve send requests (each in its own task, so a slow ACK wait blocks nothing) and
        control requests."""
        while True:
            item = await in_q.get()
            kind = item[0]
            if kind == "send":
                _, req_id, target, timeout, frames = item
                task = asyncio.create_task(self._send(frames, target, timeout))
                self._send_tasks.add(task)
                task.add_done_callback(
                    lambda t, rid=req_id: self._send_done(out_q, rid, t)
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

    def _send_done(self, out_q, req_id: int, task: asyncio.Task) -> None:
        self._send_tasks.discard(task)
        if task.cancelled():
            return
        ok = False if task.exception() is not None else bool(task.result())
        if task.exception() is not None:
            self.logger.warning(f"Send failed: {task.exception()}")
        out_q.put_nowait(("sent", req_id, ok))

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


def start_io_process(io: IOProcess) -> Tuple[Any, ZMQIOQueue, ZMQIOQueue, Any, List[str]]:
    """Spawn ``io``'s class in a new process, joined by two ipc queues.

    Returns ``(process, in_q, out_q, ctx, paths)``; the caller puts on ``in_q``, gets from
    ``out_q``, and on close closes both, terminates the context and unlinks ``paths``.
    """
    import multiprocessing as mp

    import zmq.asyncio

    kwargs = io.kwargs()
    rule = kwargs.get("sender_rule")
    if rule is not None and not isinstance(rule, SenderRule):
        raise TypeError(
            "An out-of-process connection needs a SenderRule as its unknown-sender "
            f"validator, not {type(rule).__name__}"
        )
    ctx = zmq.asyncio.Context()
    in_path, out_path = new_ipc_path("in"), new_ipc_path("out")
    in_q = ZMQIOQueue(ctx, in_path, push=True, bind=True)
    out_q = ZMQIOQueue(ctx, out_path, push=False, bind=True)
    class_path = f"{type(io).__module__}:{type(io).__qualname__}"
    proc = mp.get_context("spawn").Process(
        target=_io_main,
        args=(class_path, kwargs, in_path, out_path, os.getpid()),
        name=f"io-{kwargs['identity']}",
        daemon=True,
    )
    proc.start()
    return proc, in_q, out_q, ctx, [in_path, out_path]


def _io_main(class_path: str, kwargs: dict, in_path: str, out_path: str, parent_pid: int):
    """Entry point of a spawned IOProcess: serve the queues until ``close`` or the owner
    dies. Signals are left to the owner, which closes us."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    module, _, qualname = class_path.partition(":")
    cls = getattr(importlib.import_module(module), qualname)

    from ensemble_launcher.logging import setup_logger

    logger = setup_logger(name=f"connection-{kwargs['identity']}-io", subdir="connections")
    io = cls(**kwargs, logger=logger)

    async def main():
        import zmq.asyncio

        ctx = zmq.asyncio.Context()
        in_q = ZMQIOQueue(ctx, in_path, push=False, bind=False)
        out_q = ZMQIOQueue(ctx, out_path, push=True, bind=False)

        async def watch_parent():
            while os.getppid() == parent_pid:
                await asyncio.sleep(1.0)
            logger.warning(f"{kwargs['identity']}: owner {parent_pid} is gone; exiting")
            io._stop.set()

        watcher = None
        try:
            run = asyncio.create_task(io.run(in_q, out_q))
            while io._stop is None and not run.done():
                await asyncio.sleep(0)
            watcher = asyncio.create_task(watch_parent())
            await run
        finally:
            if watcher is not None:
                watcher.cancel()
            in_q.close()
            out_q.close(linger=1000)  # let the last results reach the owner
            ctx.term()

    asyncio.run(main())
