"""ACK round trip and throughput of a req_res ROUTER/DEALER pair, IOProcess in vs. out of process.

The receiver (ROUTER) runs in this process; the sender (DEALER) in a spawned one, so its timings
don't depend on the receiver's loop. With ``--busy``, the receiver's event loop is kept busy with
Python work in ``--busy-ms`` slices (like placement holding a master's loop): in process, ACKs
wait for the loop; out of process, they don't.

    python benchmarks/benchmark_connection_ack.py --n 2000 --busy-ms 200
"""

import argparse
import asyncio
import multiprocessing as mp
import statistics
import time
import uuid


def _sender(address, srv_id, srv_secret, cli_id, cli_secret, n, window, timeout, out_q):
    from ensemble_launcher.comm.pipe import AsyncZMQDealerConnection

    async def main():
        client = AsyncZMQDealerConnection(
            identity=cli_id, secret_id=cli_secret, remote_address=address,
            remote_identity=srv_id, remote_secret_id=srv_secret, req_res=True,
        )  # fmt: skip
        await client.open()
        await asyncio.sleep(0.2)
        payload = b"x" * 1024
        lat, fails = [], 0

        async def one():
            nonlocal fails
            t0 = time.perf_counter()
            ok = await client.send(payload, timeout=timeout)
            lat.append(time.perf_counter() - t0)
            fails += not ok

        t0 = time.perf_counter()
        for start in range(0, n, window):
            await asyncio.gather(*(one() for _ in range(min(window, n - start))))
        wall = time.perf_counter() - t0
        await client.close()
        out_q.put((lat, fails, wall))

    asyncio.run(main())


async def _run(io_process: bool, busy_ms: float, n: int, window: int, timeout: float):
    from ensemble_launcher.comm.pipe import AsyncZMQRouterConnection

    tag = uuid.uuid4().hex[:6]
    srv_id, srv_secret, cli_id, cli_secret = f"srv-{tag}", "ss", f"cli-{tag}", "cs"
    server = AsyncZMQRouterConnection(
        identity=srv_id, secret_id=srv_secret, address="127.0.0.1:0",
        expected_remotes={cli_id: cli_secret}, req_res=True, io_process=io_process,
    )  # fmt: skip
    await server.open()

    stop = asyncio.Event()

    async def drain():
        while not stop.is_set():
            try:
                await server.recv(timeout=0.2)
            except (asyncio.TimeoutError, TimeoutError):
                pass

    async def hog():  # the loop does Python work in busy_ms slices
        while not stop.is_set():
            end = time.perf_counter() + busy_ms / 1000
            x = 0
            while time.perf_counter() < end:
                x += 1
            await asyncio.sleep(0)

    tasks = [asyncio.create_task(drain())]
    if busy_ms > 0:
        tasks.append(asyncio.create_task(hog()))

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    proc = ctx.Process(
        target=_sender,
        args=(server.address, srv_id, srv_secret, cli_id, cli_secret, n, window, timeout, out_q),
    )
    proc.start()
    while proc.is_alive() and out_q.empty():
        await asyncio.sleep(0.05)
    lat, fails, wall = out_q.get()
    proc.join()
    stop.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    await server.close()
    return lat, fails, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--window", type=int, default=16, help="concurrent sends")
    ap.add_argument("--timeout", type=float, default=1.0, help="ACK timeout (s)")
    ap.add_argument("--busy-ms", type=float, default=200.0)
    args = ap.parse_args()

    print(f"{args.n} x 1 KiB sends, window {args.window}, ACK timeout {args.timeout}s")
    print(f"{'receiver':<10} {'IOProcess':<15} {'msgs/s':>9} {'p50 ms':>8} {'p99 ms':>8} {'max ms':>8} {'timeouts':>9}")
    for busy in (0.0, args.busy_ms):
        for io_process in (False, True):
            lat, fails, wall = asyncio.run(
                _run(io_process, busy, args.n, args.window, args.timeout)
            )
            lat_ms = sorted(x * 1e3 for x in lat)
            p99 = lat_ms[int(0.99 * (len(lat_ms) - 1))]
            print(
                f"{'busy' if busy else 'idle':<10} {'out of process' if io_process else 'in process':<15} "
                f"{args.n / wall:>9.0f} {statistics.median(lat_ms):>8.2f} {p99:>8.2f} "
                f"{lat_ms[-1]:>8.2f} {fails:>9}"
            )


if __name__ == "__main__":
    main()
