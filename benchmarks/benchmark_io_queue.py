"""Cross-process IO queue: `SocketIOQueue` vs `ZMQIOQueue`.

The queue between a connection's front end and its out-of-process IOProcess. Each case spawns
a peer process (as `start_io_process` does: the owner binds and opens first, the peer gets the
two paths reversed and connects) and moves real queue items through it:

- one-way throughput: the owner puts N ``("data", [identity, payload])`` items and stops the
  clock when the peer reports it has received all of them (it reports items and bytes, which
  are checked);
- round-trip latency: put one 64 B ``("data", ...)`` item, wait for the peer's echo, repeat.

``socket-routed`` is the same `SocketIOQueue` with the receiving end routing ``"data"`` to a
queue of its own (``routes={"data": q}``), as the connection front end does: data items skip
``get()`` and are decoded in ``buffer_updated``. Both variants consume data as it arrives.

    python benchmark_io_queue.py                         # every case, both queues
    python benchmark_io_queue.py --queues socket --cases small large
    python benchmark_io_queue.py --loop asyncio          # default: uvloop if installed

Payloads are random bytes (``numpy`` RNG), so nothing compresses or dedupes.
"""

import argparse
import asyncio
import multiprocessing as mp
import os
import statistics
import time

import numpy as np

from ensemble_launcher.comm.pipe.io_process import SocketIOQueue, ZMQIOQueue, new_ipc_path

# name -> (class, receiving end routes "data" to its own queue)
QUEUES = {
    "zmq": (ZMQIOQueue, False),
    "socket": (SocketIOQueue, False),
    "socket-routed": (SocketIOQueue, True),
}
# name -> (number of items, payload bytes per item); "pingpong" is the latency case
CASES = {
    "small": (200_000, 64),
    "medium": (20_000, 16 << 10),
    "large": (64, 16 << 20),
    "pingpong": (5_000, 64),
}
STOP = ("control", "stop", ())


def _make(queue: str, send_path: str, recv_path: str, bind: bool):
    """``(queue end, data route queue or None)``."""
    cls, routed = QUEUES[queue]
    data_q = asyncio.Queue() if routed else None
    return cls(send_path, recv_path, bind=bind, routes={"data": data_q} if routed else None), data_q


def _run(coro, loop_kind: str):
    if loop_kind == "uvloop":
        import uvloop

        return uvloop.run(coro)
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
#  Peer process                                                               #
# --------------------------------------------------------------------------- #


def _peer(queue: str, send_path: str, recv_path: str, mode: str, loop_kind: str) -> None:
    """``sink``: count items until STOP, then report the count. ``echo``: send every item
    back until STOP."""

    async def main():
        q, data_q = _make(queue, send_path, recv_path, bind=False)
        await q.open()
        q.put_nowait(("ready", "peer"))  # handshake: connected and running
        n = nbytes = 0

        def on_data(item):
            nonlocal n, nbytes
            if mode == "echo":
                q.put_nowait(item)
            else:
                nbytes += sum(len(f) for f in item[1])
            n += 1

        if data_q is None:  # everything through get()
            while (item := await q.get(timeout=120)) is not None and item != STOP:
                on_data(item)
        else:  # data from its route queue, as it arrives; STOP still through get()
            async def consume():
                while True:
                    on_data(await data_q.get())

            consumer = asyncio.create_task(consume())
            await q.get(timeout=120)  # STOP: every data item before it is already routed
            while not data_q.empty():
                await asyncio.sleep(0)
            consumer.cancel()
        if mode == "sink":
            q.put_nowait(("ready", f"{n}:{nbytes}"))  # what actually arrived
        await q.close(linger=5000)

    _run(main(), loop_kind)


# --------------------------------------------------------------------------- #
#  Owner                                                                      #
# --------------------------------------------------------------------------- #


async def _with_peer(queue: str, mode: str, loop_kind: str, body):
    to_peer, from_peer = new_ipc_path("in"), new_ipc_path("out")
    q, data_q = _make(queue, to_peer, from_peer, bind=True)
    await q.open()  # bind before the peer exists, as start_io_process does
    proc = mp.get_context("spawn").Process(
        target=_peer, args=(queue, from_peer, to_peer, mode, loop_kind), daemon=True
    )
    proc.start()
    try:
        ready = await q.get(timeout=120)
        assert ready == ("ready", "peer"), ready
        return await body(q, data_q)
    finally:
        await asyncio.to_thread(proc.join, 30)
        if proc.is_alive():
            proc.terminate()
        await q.close()
        for p in (to_peer, from_peer):
            try:
                os.unlink(p)
            except OSError:
                pass


async def one_way(queue: str, n: int, size: int, loop_kind: str) -> dict:
    payload = np.random.default_rng(0).bytes(size)
    item = ("data", [b"client-1", payload])

    async def body(q, _data_q):
        t0 = time.perf_counter()
        for _ in range(n):
            q.put_nowait(item)
        t_put = time.perf_counter() - t0
        q.put_nowait(STOP)
        reply = await q.get(timeout=600)
        dt = time.perf_counter() - t0
        expect = f"{n}:{n * (size + len(b'client-1'))}"
        assert reply == ("ready", expect), (reply, expect)  # every item and byte arrived
        return {"seconds": dt, "put_seconds": t_put}

    r = await _with_peer(queue, "sink", loop_kind, body)
    r["items_per_s"] = n / r["seconds"]
    r["us_per_item"] = 1e6 * r["seconds"] / n
    r["MB_per_s"] = n * size / r["seconds"] / 1e6
    return r


async def ping_pong(queue: str, n: int, size: int, loop_kind: str) -> dict:
    payload = np.random.default_rng(0).bytes(size)
    item = ("data", [b"client-1", payload])

    async def body(q, data_q):
        rtts = []
        for _ in range(n):
            t0 = time.perf_counter()
            q.put_nowait(item)
            if data_q is None:
                got = await q.get(timeout=30)
            else:
                got = await asyncio.wait_for(data_q.get(), 30)
            rtts.append(time.perf_counter() - t0)
            assert got == item, got
        q.put_nowait(STOP)
        return rtts

    rtts = await _with_peer(queue, "echo", loop_kind, body)
    rtts = sorted(rtts[len(rtts) // 10 :])  # drop the first 10% (warm-up)
    return {
        "median_us": 1e6 * statistics.median(rtts),
        "p99_us": 1e6 * rtts[int(0.99 * (len(rtts) - 1))],
    }


def main() -> None:
    try:
        import uvloop  # noqa: F401

        default_loop = "uvloop"
    except ImportError:
        default_loop = "asyncio"
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--queues", nargs="+", choices=sorted(QUEUES), default=["zmq", "socket"])
    ap.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    ap.add_argument("--loop", choices=["uvloop", "asyncio"], default=default_loop)
    ap.add_argument("--repeat", type=int, default=3, help="runs per case; the best is reported")
    args = ap.parse_args()

    print(f"loop={args.loop}  repeat={args.repeat} (best of)")
    for case in args.cases:
        n, size = CASES[case]
        print(f"\n== {case}: {n:,} items x {size:,} B")
        for queue in args.queues:
            if case == "pingpong":
                runs = [_run(ping_pong(queue, n, size, args.loop), args.loop) for _ in range(args.repeat)]
                best = min(runs, key=lambda r: r["median_us"])
                print(f"  {queue:14s} round trip  median {best['median_us']:7.1f} us"
                      f"   p99 {best['p99_us']:7.1f} us")  # fmt: skip
            else:
                runs = [_run(one_way(queue, n, size, args.loop), args.loop) for _ in range(args.repeat)]
                best = min(runs, key=lambda r: r["seconds"])
                print(f"  {queue:14s} {best['items_per_s']:>12,.0f} items/s  {best['us_per_item']:8.2f} us/item"
                      f"  {best['MB_per_s']:9.1f} MB/s   (puts {best['put_seconds']:.3f}s of {best['seconds']:.3f}s)")  # fmt: skip


if __name__ == "__main__":
    main()
