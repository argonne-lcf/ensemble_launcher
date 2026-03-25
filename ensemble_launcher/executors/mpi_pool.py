"""
MPI pool: rank 0 = gateway (ZMQ IPC <-> MPI), ranks 1..N-1 = workers.

Launch with:
    mpirun -n <N+1> python mpi_pool.py --socket-path <ipc_path>
where N is the number of worker ranks (rank 0 is the master/gateway).
"""

import argparse
import asyncio
import collections
import os

import cloudpickle
import zmq
import zmq.asyncio
from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

TAG_TASK   = 1
TAG_RESULT = 2
TAG_STOP   = 3


# ── Workers (ranks 1..N-1) ────────────────────────────────────────────────────

def run_worker():
    status = MPI.Status()
    while True:
        data = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.tag == TAG_STOP:
            break
        msg_id, fn_bytes, args_bytes, kwargs_bytes, env = cloudpickle.loads(data)
        original_env = os.environ.copy()
        os.environ.update(env)
        try:
            result = cloudpickle.loads(fn_bytes)(
                *cloudpickle.loads(args_bytes), **cloudpickle.loads(kwargs_bytes)
            )
            out = cloudpickle.dumps(("result", msg_id, "ok", result))
        except Exception as e:
            out = cloudpickle.dumps(("result", msg_id, "err", e))
        finally:
            os.environ.clear()
            os.environ.update(original_env)
        COMM.send(out, dest=0, tag=TAG_RESULT)


# ── Master (rank 0) ───────────────────────────────────────────────────────────

def _dispatch(available, task_queue):
    """Send queued tasks to available workers via MPI."""
    while task_queue and available:
        msg_id, target, payload = task_queue[0]
        if target is not None and target not in available:
            break  # target worker busy; preserve original targeted-dispatch semantics
        worker = target if target is not None else next(iter(available))
        available.discard(worker)
        task_queue.popleft()
        COMM.send(payload, dest=worker, tag=TAG_TASK)


async def _client_loop(sock, available, task_queue, stop):
    """Receive tasks from AsyncMPIPoolExecutor, dispatch to workers."""
    while True:
        data = await sock.recv()
        msg = cloudpickle.loads(data)

        if msg[0] == "shutdown":
            for w in range(1, SIZE):
                COMM.send(None, dest=w, tag=TAG_STOP)
            stop.set()
            return

        _, msg_id, target, fn_bytes, args_bytes, kwargs_bytes, env = msg
        payload = cloudpickle.dumps((msg_id, fn_bytes, args_bytes, kwargs_bytes, env))
        task_queue.append((msg_id, target, payload))
        _dispatch(available, task_queue)


async def _mpi_loop(sock, available, task_queue, stop):
    """Poll for MPI results and forward them to the client."""
    status = MPI.Status()
    while not stop.is_set():
        if COMM.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
            src = status.Get_source()
            data = COMM.recv(source=src, tag=TAG_RESULT)
            available.add(src)
            await sock.send(data)  # result bytes pass through unchanged
            _dispatch(available, task_queue)
        else:
            await asyncio.sleep(0.001)


async def run_master(socket_path):
    ctx = zmq.asyncio.Context()
    sock = ctx.socket(zmq.DEALER)
    sock.identity = b"mpi_pool"
    sock.connect(f"ipc://{socket_path}")

    # Announce ready to the client
    await sock.send(cloudpickle.dumps(("ready",)))
    print(f"[rank 0] MPI pool ready, {SIZE - 1} workers, socket={socket_path}", flush=True)

    available  = set(range(1, SIZE))
    task_queue = collections.deque()
    stop       = asyncio.Event()

    await asyncio.gather(
        _client_loop(sock, available, task_queue, stop),
        _mpi_loop   (sock, available, task_queue, stop),
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-path", required=True)
    args = parser.parse_args()

    if RANK == 0:
        asyncio.run(run_master(args.socket_path))
    else:
        run_worker()
