"""`SocketIOQueue`: the cross-process two-way queue over Unix stream sockets.

Each end puts items the other end gets, in order, whatever their size; the bind end can open
and put before its peer exists; ``get`` returns None on timeout and once the peer is gone.
"""

import asyncio
import multiprocessing as mp
import os

import pytest

from ensemble_launcher.comm.pipe.io_process import SocketIOQueue, new_ipc_path

pytestmark = pytest.mark.core


def _items(n_small=2000, big_sizes=(3 << 20, 9 << 20)):
    """Every queue item kind, many small ones and a few multi-MB data frames."""
    out = [("ready", "tcp://host:1234")]
    for i in range(n_small):
        out.append(("data", [b"id", os.urandom(i % 97)]))
        out.append(("sent", i, i % 3 != 0))
    for size in big_sizes:
        out.append(("send", 7, "child-1", 2.5, [os.urandom(size), b"tail"]))
    out.append(("control", "add_remote", ("node", "secret")))
    return out


def _normal(item):
    """Items as get() returns them: frames as bytes, args as a tuple."""
    kind = item[0]
    if kind == "data":
        return (kind, [bytes(f) for f in item[1]])
    if kind == "send":
        return item[:4] + ([bytes(f) for f in item[4]],)
    return item


async def _pair():
    a_path, b_path = new_ipc_path("a"), new_ipc_path("b")
    bind = SocketIOQueue(a_path, b_path, bind=True)
    await bind.open()  # listens; the peer doesn't exist yet
    bind.put_nowait(("ready", "early"))  # before the peer exists: kept, sent once it connects
    peer = SocketIOQueue(b_path, a_path, bind=False)
    await peer.open()
    assert await peer.get(timeout=5) == ("ready", "early")
    return bind, peer, (a_path, b_path)


def _cleanup(paths):
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


@pytest.mark.asyncio
async def test_items_both_ways_in_order():
    bind, peer, paths = await _pair()
    try:
        items = _items()
        for item in items:
            bind.put_nowait(item)
            peer.put_nowait(item)
        for q in (peer, bind):  # each end gets what the other put
            got = [await q.get(timeout=10) for _ in items]
            assert got == [_normal(i) for i in items]
        assert await bind.get(timeout=0.05) is None  # nothing left: a timeout
    finally:
        await bind.close(linger=1000)
        await peer.close()
        _cleanup(paths)


@pytest.mark.asyncio
async def test_get_reports_a_closed_peer():
    bind, peer, paths = await _pair()
    try:
        peer.put_nowait(("sent", 1, True))
        await peer.close(linger=1000)
        assert await bind.get(timeout=5) == ("sent", 1, True)  # delivered before the close
        assert await bind.get(timeout=0.1) is None  # then: gone
        assert await bind.get(timeout=0.1) is None  # and stays gone
    finally:
        await bind.close()
        _cleanup(paths)


def _echo_main(send_path, recv_path):
    """A spawned peer: connect, send every item back, stop at ("control", "close", ())."""

    async def main():
        q = SocketIOQueue(send_path, recv_path, bind=False)
        await q.open()
        while (item := await q.get(timeout=30)) is not None:
            q.put_nowait(item)
            if item[0] == "control" and item[1] == "close":
                break
        await q.close(linger=2000)

    asyncio.run(main())


@pytest.mark.asyncio
async def test_spawned_peer_echoes_and_dies():
    to_child, from_child = new_ipc_path("in"), new_ipc_path("out")
    q = SocketIOQueue(to_child, from_child, bind=True)
    await q.open()  # before the child exists, as start_io_process does
    items = _items(n_small=500) + [("control", "close", ())]
    for item in items:  # all put before the child even starts
        q.put_nowait(item)
    proc = mp.get_context("spawn").Process(target=_echo_main, args=(from_child, to_child))
    proc.start()
    try:
        got = [await q.get(timeout=30) for _ in items]
        assert got == [_normal(i) for i in items]
        await asyncio.to_thread(proc.join, 10)
        assert proc.exitcode == 0
        assert await q.get(timeout=0.2) is None  # the child is gone
    finally:
        if proc.is_alive():
            proc.terminate()
        await q.close()
        _cleanup((to_child, from_child))
