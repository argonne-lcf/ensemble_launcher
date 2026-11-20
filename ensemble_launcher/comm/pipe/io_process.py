"""The socket half of a connection, joined to its front end by a queue.

An `IOProcess` owns a connection's socket and everything that touches it: the receive loop,
sender verification, the ACK protocol (sending ACKs for received messages, and waiting for the
peer's ACK for our own sends, with the timeout) and msg ids. The front end (`AsyncConnection`)
and the IOProcess each hold one end of a two-way queue; each end ``put_nowait``s to the other
end and ``get``s what the other end put:

    IOProcess -> front end  ("ready", address)       started; the actual bound address
                            ("data", frames)         a received message, verified and ACKed
                            ("sent", req_id, ok)     final result of one send
                            ("peer", alive)          the peer was reached / lost (optional,
                                                     see `IOProcess.provides_peer_liveness`)
    front end -> IOProcess  ("send", req_id, target, timeout, frames)
                            ("control", op, args)    "add_remote" / "remove_remote" / "close"

Fast path: a queue end may be given ``routes={kind: asyncio.Queue}``; received items of a routed
kind go straight into that queue, decoded (the tuple ``get`` would have returned), and never
reach ``get``. The front end routes ``"data"`` to the connection's message queue, the IOProcess
routes ``"send"`` to its send requests, so the bulk traffic skips the dispatch loops.

The IOProcess runs either on the caller's event loop, with a `LocalIOQueue` pair (two
`asyncio.Queue`s), or in its own process, with a `SocketIOQueue` on each side (two Unix stream
sockets, length-framed items, read with a `BufferedProtocol`). `ZMQIOQueue` (PUSH/PULL over
``ipc://``) has the same interface and is kept for comparison (``benchmark_io_queue.py``). The
IOProcess doesn't know which it has: the items are the whole contract. Out of process, a busy event loop in
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
import socket
import struct
import tempfile
from abc import ABC, abstractmethod
from asyncio import BufferedProtocol
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


Routes = Optional[Dict[str, asyncio.Queue]]


class LocalIOQueue:
    """One end of an in-process two-way queue: ``put_nowait`` feeds the other end's ``get``,
    or, for a kind the other end routes, its route queue. Make both ends with `pair`."""

    def __init__(self, send_q: asyncio.Queue, recv_q: asyncio.Queue, peer_routes: Routes = None):
        self._send_q = send_q
        self._recv_q = recv_q
        self._peer_routes = peer_routes or {}

    @classmethod
    def pair(
        cls, a_routes: Routes = None, b_routes: Routes = None
    ) -> Tuple["LocalIOQueue", "LocalIOQueue"]:
        """Two connected ends; ``a_routes`` / ``b_routes`` are each end's receive routes."""
        a_to_b, b_to_a = asyncio.Queue(), asyncio.Queue()
        return cls(a_to_b, b_to_a, peer_routes=b_routes), cls(b_to_a, a_to_b, peer_routes=a_routes)

    async def open(self) -> None:
        pass

    def put_nowait(self, item: tuple) -> None:
        route = self._peer_routes.get(item[0])
        (route if route is not None else self._send_q).put_nowait(item)

    async def get(self, timeout: Optional[float] = None) -> Optional[tuple]:
        if timeout is None:
            return await self._recv_q.get()
        try:
            return await asyncio.wait_for(self._recv_q.get(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None

    async def close(self, linger: int = 0) -> None:
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
    if kind == "peer":
        return [b"peer", b"\x01" if item[1] else b"\x00"]
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
    if kind == b"peer":
        return ("peer", frames[1] == b"\x01")
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

    def __init__(self, send_path: str, recv_path: str, bind: bool, routes: Routes = None):
        self._send_path = send_path
        self._recv_path = recv_path
        self._bind = bind
        self._routes = routes or {}  # applied in get(): someone must keep calling it
        self._ctx = None
        self._send_sock = None
        self._recv_sock = None

    async def open(self) -> None:
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
        while True:
            if timeout is not None and not await self._recv_sock.poll(int(timeout * 1000)):
                return None
            item = _decode(await self._recv_sock.recv_multipart())
            route = self._routes.get(item[0])
            if route is None:
                return item
            route.put_nowait(item)

    async def close(self, linger: int = 0) -> None:
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


# Wire format of a SocketIOQueue item: [magic 1B][nframes 4B][len 4B x nframes][frames...]
_MAGIC = 0xE1
_HEAD = struct.Struct("!BI")
_MAX_ITEM = 1 << 36  # sanity cap: a header announcing more than 64 GiB means a broken stream
_JOIN_BELOW = 4096  # items up to this many payload bytes are sent as one joined buffer


class _Proto(BufferedProtocol):
    """One Unix stream connection of a `SocketIOQueue`. Received bytes go straight into a
    growable buffer (`get_buffer`); each complete item's frames go on ``items``. ``None`` on
    ``items`` means the connection is gone."""

    INITIAL = 1 << 18  # 256 KiB
    MIN_FREE = 1 << 16  # always offer at least this much room to a read
    SOCK_BUF = 1 << 22  # kernel socket buffers (macOS defaults to 8 KiB, Linux ~208 KiB)
    HIGH_WATER = 1 << 22  # transport write buffer that triggers pause_writing

    def __init__(self, connected: asyncio.Future, routes: Routes = None):
        super().__init__()
        self._connected = connected
        self.transport: Optional[asyncio.Transport] = None
        self.items: asyncio.Queue = asyncio.Queue()  # raw frames of unrouted items; get() decodes
        self._routes = {k.encode(): q for k, q in (routes or {}).items()}  # kind frame -> queue
        self._buf = bytearray(self.INITIAL)
        self._w = 0  # end of the received bytes
        self._r = 0  # start of the first unparsed item
        self._need = 0  # size of the item being received, counted from _r (0: unknown yet)
        # Send side: while the transport is paused, buffers wait here (see `send`).
        self._paused = False
        self._held: list = []
        self.held_bytes = 0

    def connection_made(self, transport):
        if self._connected.done():  # only one peer per path
            transport.abort()
            return
        self.transport = transport
        sock = transport.get_extra_info("socket")
        for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
            try:
                sock.setsockopt(socket.SOL_SOCKET, opt, self.SOCK_BUF)
            except OSError:
                pass  # keep the default
        transport.set_write_buffer_limits(high=self.HIGH_WATER)
        self._connected.set_result(self)

    # --- sending ---------------------------------------------------------- #

    def send(self, buffers: list) -> None:
        """Write ``buffers`` in order. While the transport is paused they are held here and
        handed over in one batch on `resume_writing`: the transport sums its whole buffer on
        every write, so writing item by item into a backed-up transport is quadratic."""
        if self._paused or self._held:
            self._held.extend(buffers)
            self.held_bytes += sum(memoryview(b).nbytes for b in buffers)
        else:
            self.transport.writelines(buffers)

    def pause_writing(self) -> None:
        self._paused = True

    def resume_writing(self) -> None:
        self._paused = False
        held, self._held, self.held_bytes = self._held, [], 0
        if held and not self.transport.is_closing():
            self.transport.writelines(held)  # may pause again; later sends are held again

    # --- receiving -------------------------------------------------------- #

    def get_buffer(self, sizehint: int) -> memoryview:
        buf, r, w = self._buf, self._r, self._w
        pending = w - r  # bytes of an incomplete item
        want = max(self._need, pending + self.MIN_FREE)  # room needed, counted from r
        if len(buf) - r < want:  # the tail is too short
            if want > len(buf):  # too small even if compacted: grow
                new = bytearray(max(want, 2 * len(buf)))
                new[:pending] = buf[r:w]
                self._buf = buf = new
            else:  # big enough: slide the partial item to the front
                buf[:pending] = buf[r:w]
            self._r, self._w = 0, pending
        return memoryview(buf)[self._w :]

    def buffer_updated(self, nbytes: int) -> None:
        self._w += nbytes
        buf, r, w = self._buf, self._r, self._w
        mv = memoryview(buf)
        try:
            while w - r >= _HEAD.size:
                magic, nframes = _HEAD.unpack_from(buf, r)
                if magic != _MAGIC:
                    return self._fail("stream out of sync (bad magic byte)")
                head = _HEAD.size + 4 * nframes
                if w - r < head:
                    self._need = head
                    break
                lens = struct.unpack_from(f"!{nframes}I", buf, r + _HEAD.size)
                total = head + sum(lens)
                if total > _MAX_ITEM:
                    return self._fail(f"item of {total} bytes announced")
                if w - r < total:
                    self._need = total
                    break
                off, frames = r + head, []
                for ln in lens:
                    frames.append(bytes(mv[off : off + ln]))  # one copy, out of the reused buffer
                    off += ln
                route = self._routes.get(frames[0])  # the kind frame
                if route is None:
                    self.items.put_nowait(frames)
                else:  # fast path: decoded, straight to its consumer
                    route.put_nowait(_decode(frames))
                r += total
                self._need = 0
        finally:
            mv.release()
        if r == w:
            r = self._w = 0
        self._r = r

    def _fail(self, why: str) -> None:
        logging.getLogger(__name__).error(f"SocketIOQueue: {why}; closing the connection")
        self.transport.abort()  # connection_lost follows and wakes the reader

    def eof_received(self) -> bool:
        if self._w != self._r:
            logging.getLogger(__name__).error(
                f"SocketIOQueue: peer closed in the middle of an item ({self._w - self._r} bytes)"
            )
        return False  # close our side too

    def connection_lost(self, exc: Optional[Exception]) -> None:
        if not self._connected.done():
            self._connected.set_exception(exc or ConnectionError("connection closed"))
        self.items.put_nowait(None)


def _proto_of(fut: Optional[asyncio.Future]) -> Optional[_Proto]:
    if fut is None or not fut.done() or fut.cancelled() or fut.exception() is not None:
        return None
    return fut.result()


class SocketIOQueue:
    """One end of a cross-process two-way queue over two Unix stream sockets: items
    ``put_nowait`` here are written to ``send_path``; ``get`` returns items read from
    ``recv_path``. The other end is built with the two paths reversed; one end binds (listens),
    the other connects. Items are framed with a length header and received through a
    `BufferedProtocol`, so many items cost one read and one callback.

    Lifecycle: ``await open()`` -- the bind side only starts listening, so it can be called
    before the peer exists -- then ``put_nowait`` / ``get``, then ``await close(linger)``.
    Nothing waits for the peer explicitly: items put before it connects are kept and written,
    in order, once it does; ``get`` waits for it. ``get`` returns None on timeout and once the
    connection is gone (after waiting the timeout, so a polling caller doesn't spin)."""

    def __init__(self, send_path: str, recv_path: str, bind: bool, routes: Routes = None):
        self._send_path = send_path
        self._recv_path = recv_path
        self._bind = bind
        self._servers: list = []
        self._send_fut: Optional[asyncio.Future] = None
        self._recv_fut: Optional[asyncio.Future] = None
        self._send_proto: Optional[_Proto] = None
        self._recv_proto: Optional[_Proto] = None
        self._early: list = []  # items put before the peer connected, in order
        self._routes = routes  # receive routes, given to the receiving connection's _Proto

    async def open(self) -> None:
        def factory(fut: asyncio.Future, routes: Routes):
            return lambda: _Proto(fut, routes)

        loop = asyncio.get_running_loop()
        self._send_fut, self._recv_fut = loop.create_future(), loop.create_future()
        ends = (
            (self._send_path, factory(self._send_fut, None)),
            (self._recv_path, factory(self._recv_fut, self._routes)),
        )
        self._send_fut.add_done_callback(self._on_connect)
        self._recv_fut.add_done_callback(self._on_connect)
        if self._bind:
            for path, make in ends:
                self._servers.append(await loop.create_unix_server(make, path))
        else:
            for path, make in ends:
                await loop.create_unix_connection(make, path)

    def _on_connect(self, _fut: asyncio.Future) -> None:
        if self._send_proto is None and _proto_of(self._send_fut) is not None:
            self._attach_send()
        if self._send_fut.done() and self._recv_fut.done():
            self._close_servers()  # both peers are in (or failed): stop listening

    def _attach_send(self) -> None:
        """The send connection is up: write what was put before it, in order."""
        self._send_proto = _proto_of(self._send_fut)
        early, self._early = self._early, []
        for item in early:
            self._write(item)

    def put_nowait(self, item: tuple) -> None:
        if self._send_proto is None:
            if _proto_of(self._send_fut) is None:
                self._early.append(item)  # written by _attach_send once the peer connects
                return
            self._attach_send()  # connected, but the callback hasn't run yet
        self._write(item)

    def _write(self, item: tuple) -> None:
        if self._send_proto.transport.is_closing():
            return  # the peer is gone; the reader side reports it
        frames = _encode(item)
        lens = [memoryview(f).nbytes for f in frames]
        head = struct.pack(f"!BI{len(lens)}I", _MAGIC, len(lens), *lens)
        if sum(lens) <= _JOIN_BELOW:  # a small item: one buffer instead of 1 + nframes
            self._send_proto.send([b"".join([head, *frames])])
        else:  # large frames go as they are, not copied
            self._send_proto.send([head, *frames])

    async def get(self, timeout: Optional[float] = None) -> Optional[tuple]:
        async def next_frames():
            if self._recv_proto is None:
                try:
                    self._recv_proto = await asyncio.shield(self._recv_fut)  # waits for the peer
                except (asyncio.CancelledError, ConnectionError, OSError):
                    if not self._recv_fut.done():
                        raise  # our own timeout cancelled the wait
                    return None  # never connected: closed or failed
            return await self._recv_proto.items.get()

        try:
            frames = await asyncio.wait_for(next_frames(), timeout)
        except (asyncio.TimeoutError, TimeoutError):
            return None
        if frames is None:  # connection gone: keep saying so, without spinning a poller
            if self._recv_proto is not None:
                self._recv_proto.items.put_nowait(None)
            if timeout is not None:
                await asyncio.sleep(timeout)
            return None
        return _decode(frames)

    async def close(self, linger: int = 0) -> None:
        """``linger`` (ms): wait up to this long for queued puts to reach the socket."""
        self._close_servers()
        send = self._send_proto or _proto_of(self._send_fut)
        recv = self._recv_proto or _proto_of(self._recv_fut)
        if send is not None and send.transport is not None:
            deadline = asyncio.get_running_loop().time() + linger / 1000
            while (
                (send.held_bytes or send.transport.get_write_buffer_size())
                and not send.transport.is_closing()
                and asyncio.get_running_loop().time() < deadline
            ):
                await asyncio.sleep(0.005)
            send.transport.close()
        if recv is not None and recv.transport is not None:
            recv.transport.close()
        for fut in (self._send_fut, self._recv_fut):
            if fut is not None and not fut.done():
                fut.cancel()

    def _close_servers(self) -> None:
        for server in self._servers:
            server.close()
        self._servers = []


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

    **Peer liveness (optional).** Not every transport can tell that its peer is gone, so this
    is a capability: a transport that can sets `provides_peer_liveness` and implements
    `_raw_watch_peer`, calling `_peer_up` / `_peer_down` as it learns. The IOProcess reports
    each change as ``("peer", alive)``; the front end turns it into its ``peer_connected`` /
    ``peer_lost`` events. A transport without it never reports anything.
    """

    server: bool = True
    # Whether `_raw_watch_peer` reports the peer coming and going (see the class docstring).
    provides_peer_liveness: bool = False

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
        self._q = None  # the queue to the front end, while running
        self._peer_alive: Optional[bool] = None  # last reported peer state
        # "send" items the queue routes here (the fast path) instead of to _input_loop
        self.send_requests: asyncio.Queue = asyncio.Queue()

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
        self._q = q
        await self._raw_open()
        q.put_nowait(("ready", self.address))
        recv = asyncio.create_task(self._recv_loop(q))
        inp = asyncio.create_task(self._input_loop(q))
        sends = asyncio.create_task(self._send_loop(q))
        watch = asyncio.create_task(self._raw_watch_peer()) if self.provides_peer_liveness else None
        try:
            await self._stop.wait()
        finally:
            if watch is not None:  # first, so our own close is never reported as the peer going
                watch.cancel()
                await asyncio.gather(watch, return_exceptions=True)
            for t in (recv, inp, sends, *self._send_tasks):
                t.cancel()
            await asyncio.gather(recv, inp, sends, *self._send_tasks, return_exceptions=True)
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
            if item is None:  # the queue is gone: the owner closed it or died
                self._stop.set()
                return
            kind = item[0]
            if kind == "send":  # only when the queue doesn't route sends to _send_loop
                self._start_send(q, item)
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

    async def _send_loop(self, q) -> None:
        """Send requests routed past `_input_loop` (``routes={"send": self.send_requests}``)."""
        while True:
            self._start_send(q, await self.send_requests.get())

    def _start_send(self, q, item: tuple) -> None:
        """One send in its own task, so a slow ACK wait blocks nothing."""
        _, req_id, target, timeout, frames = item
        task = asyncio.create_task(self._send(frames, target, timeout))
        self._send_tasks.add(task)
        task.add_done_callback(lambda t, rid=req_id: self._send_done(q, rid, t))

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

    # ------------------------------ peer liveness ----------------------------- #

    def _peer_up(self) -> None:
        """The transport reached its peer."""
        self._report_peer(True)

    def _peer_down(self) -> None:
        """The transport lost its peer (it died, hung, or the link broke)."""
        self._report_peer(False)

    def _report_peer(self, alive: bool) -> None:
        if alive == self._peer_alive or self._q is None:
            return
        self._peer_alive = alive
        self._q.put_nowait(("peer", alive))

    async def _raw_watch_peer(self) -> None:
        """Watch the peer, calling `_peer_up` / `_peer_down` (only run when
        `provides_peer_liveness`; cancelled before `_raw_close`)."""

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


async def start_io_process(io: IOProcess, routes: Routes = None) -> Tuple[Any, SocketIOQueue, List[str]]:
    """Spawn ``io``'s class in a new process, joined to the caller by a `SocketIOQueue`.

    Returns ``(process, q, paths)``: the caller's (bind) end of the queue, already open; on
    close the caller closes it and unlinks ``paths``. ``routes`` are that end's receive routes;
    the IOProcess end routes ``"send"`` to its `IOProcess.send_requests`.
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
    q = SocketIOQueue(send_path=in_path, recv_path=out_path, bind=True, routes=routes)
    await q.open()  # listen before the child connects; puts until then are kept
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
        q = SocketIOQueue(send_path, recv_path, bind=False, routes={"send": io.send_requests})
        await q.open()

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
            await q.close(linger=1000)  # let the last results reach the owner

    asyncio.run(main())
