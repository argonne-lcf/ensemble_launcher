"""
MPI pool: rank 0 = gateway (ZMQ IPC <-> MPI) + local task executor,
          ranks 1..N-1 = MPI workers.

Rank 0 manages the ZMQ<->MPI bridge and also executes tasks locally via a
ThreadPoolExecutor when targeted (i.e. when its CPU is in _cpu_to_pid).
The async_worker trims rank 0's CPU from the scheduler so it is never
double-scheduled; mpi_info still includes rank 0's CPU for correct binding.

Three dedicated ZMQ sockets separate tasks, results, and control messages
so that task/result payloads pass through the gateway as opaque blobs —
no cloudpickle serialization or deserialization on the hot path.

Launch with:
    mpirun -n <N> python mpi_pool.py --socket-base <ipc_base>
where N includes rank 0.
"""

import argparse
import asyncio
import os
import signal
import struct
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.getcwd())

import cloudpickle
import zmq
import zmq.asyncio
from mpi4py import MPI

from ensemble_launcher.logging import setup_logger

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

TAG_TASK = 1
TAG_RESULT = 2
TAG_STOP = 3
TAG_DONE = 4

# Routing header: (target: int32) — batch header, worker_id only
_HEADER_FMT = "!i"

logger = None  # Initialized in main after parsing args

# ── Workers (ranks 1..N-1) ────────────────────────────────────────────────────


def run_worker():
    logger.info(f"[rank {RANK}] Worker started, waiting for tasks")
    status = MPI.Status()
    while True:
        data = COMM.recv(source=0, tag=MPI.ANY_TAG, status=status)
        if status.tag == TAG_STOP:
            logger.debug(f"[rank {RANK}] Received STOP signal, shutting down")
            COMM.send("done", dest=0, tag=TAG_DONE)
            break
        msg_id, fn, args, kwargs, env = cloudpickle.loads(data)
        logger.debug(f"[rank {RANK}] Received task {msg_id}")
        original_env = os.environ.copy()
        os.environ.update(env)
        try:
            result = fn(*args, **kwargs)
            logger.debug(f"[rank {RANK}] Task {msg_id} completed successfully")
            out = cloudpickle.dumps(("result", msg_id, "ok", result))
        except Exception as e:
            logger.error(f"[rank {RANK}] Task {msg_id} failed: {e}")
            out = cloudpickle.dumps(("result", msg_id, "err", e))
        finally:
            os.environ.clear()
            os.environ.update(original_env)
        COMM.send(out, dest=0, tag=TAG_RESULT)


# ── Master (rank 0) ───────────────────────────────────────────────────────────


def _dispatch(header, payload, local_queue):
    """Deserialize task header and dispatch payload to the target worker."""
    (target,) = struct.unpack(_HEADER_FMT, header)
    if target == 0:
        local_queue.put_nowait(payload)
    else:
        COMM.send(payload, dest=target, tag=TAG_TASK)


async def _client_loop(task_sock, local_queue, stop):
    """Receive tasks from AsyncMPIPoolExecutor via the task socket, forward to workers."""
    logger.info("[client_loop] Started")
    recv_fut = asyncio.ensure_future(task_sock.recv_multipart())
    stop_fut = asyncio.ensure_future(stop.wait())
    try:
        while True:
            done, _ = await asyncio.wait(
                [recv_fut, stop_fut], return_when=asyncio.FIRST_COMPLETED
            )
            if stop_fut in done:
                recv_fut.cancel()
                break
            # frames = [header1, payload1, header2, payload2, ...]
            frames = recv_fut.result()
            for i in range(0, len(frames), 2):
                _dispatch(frames[i], frames[i + 1], local_queue)
            recv_fut = asyncio.ensure_future(task_sock.recv_multipart())
    finally:
        logger.info("[client_loop] Exiting")


async def _msg_loop(msg_sock, stop):
    """Listen for control messages (shutdown) on the message socket."""
    logger.info("[msg_loop] Started")
    recv_fut = asyncio.ensure_future(msg_sock.recv())
    stop_fut = asyncio.ensure_future(stop.wait())
    try:
        while True:
            done, _ = await asyncio.wait(
                [recv_fut, stop_fut], return_when=asyncio.FIRST_COMPLETED
            )
            if stop_fut in done:
                recv_fut.cancel()
                break
            msg = cloudpickle.loads(recv_fut.result())
            if msg[0] == "shutdown":
                logger.info("[msg_loop] Received shutdown command, setting stop event")
                stop.set()
                return
            recv_fut = asyncio.ensure_future(msg_sock.recv())
    finally:
        logger.info("[msg_loop] Exiting")


async def _mpi_loop(result_sock, stop):
    """Poll for MPI results and forward them to the client via the result socket."""
    logger.info("[mpi_loop] Started")
    status = MPI.Status()
    while not stop.is_set():
        if COMM.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
            src = status.Get_source()
            data = COMM.recv(source=src, tag=TAG_RESULT)
            logger.debug(f"[mpi_loop] Received result from rank {src}")
            await result_sock.send(data)  # result bytes pass through unchanged
        else:
            await asyncio.sleep(0.01)
    logger.info("[mpi_loop] Exiting — stop event was set")


async def _rank0_executor(local_queue, result_sock, stop):
    """Execute tasks locally on rank 0 using a thread pool."""
    logger.info("[rank0_executor] Started")
    loop = asyncio.get_event_loop()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        while True:
            get_task = asyncio.ensure_future(local_queue.get())
            stop_task = asyncio.ensure_future(stop.wait())
            done, pending = await asyncio.wait(
                [get_task, stop_task], return_when=asyncio.FIRST_COMPLETED
            )
            for t in pending:
                t.cancel()
            if stop_task in done:
                logger.info("[rank0_executor] Stop event received, breaking")
                break
            payload = get_task.result()  # raw bytes from task socket
            msg_id, fn, args, kwargs, env = cloudpickle.loads(payload)
            logger.debug(f"[rank 0] Executing task {msg_id} locally")

            def _execute():
                original_env = os.environ.copy()
                os.environ.update(env)
                try:
                    result = fn(*args, **kwargs)
                    return cloudpickle.dumps(("result", msg_id, "ok", result))
                except Exception as e:
                    return cloudpickle.dumps(("result", msg_id, "err", e))
                finally:
                    os.environ.clear()
                    os.environ.update(original_env)

            out = await loop.run_in_executor(pool, _execute)
            await result_sock.send(out)
    finally:
        logger.info("[rank0_executor] Shutting down thread pool")
        pool.shutdown(wait=True)
        logger.info("[rank0_executor] Thread pool shut down")


def _stop_workers():
    """Send TAG_STOP to every worker and drain TAG_DONE acknowledgements.

    Called after the stop event is set, outside the async loops, so blocking
    MPI calls won't stall the event loop.  Any pending TAG_RESULT messages
    are drained first so workers aren't blocked trying to send results.
    """
    status = MPI.Status()
    # Drain any in-flight results so workers can proceed to receive STOP
    drained = 0
    while COMM.Iprobe(source=MPI.ANY_SOURCE, tag=TAG_RESULT, status=status):
        COMM.recv(source=status.Get_source(), tag=TAG_RESULT)
        drained += 1
    logger.info(f"[_stop_workers] Drained {drained} in-flight results")

    logger.info(f"[_stop_workers] Sending TAG_STOP to {SIZE - 1} workers")
    for w in range(1, SIZE):
        COMM.send(None, dest=w, tag=TAG_STOP)
        logger.info(f"[_stop_workers] Sent TAG_STOP to rank {w}")

    logger.info("[_stop_workers] Waiting for TAG_DONE from all workers")
    for w in range(1, SIZE):
        COMM.recv(source=w, tag=TAG_DONE)
        logger.info(f"[_stop_workers] Rank {w} confirmed shutdown")


async def run_master(socket_base):
    logger.info(f"[rank 0] Starting master, SIZE={SIZE}, socket_base={socket_base}")
    ctx = zmq.asyncio.Context()

    task_sock = ctx.socket(zmq.PULL)
    task_sock.connect(f"ipc://{socket_base}_task.ipc")

    result_sock = ctx.socket(zmq.PUSH)
    result_sock.connect(f"ipc://{socket_base}_result.ipc")

    msg_sock = ctx.socket(zmq.DEALER)
    msg_sock.identity = b"mpi_pool"
    msg_sock.connect(f"ipc://{socket_base}_msg.ipc")

    # Announce ready to the client on the message socket
    await msg_sock.send(cloudpickle.dumps(("ready",)))
    logger.info(
        f"[rank 0] MPI pool ready, {SIZE - 1} workers, socket_base={socket_base}"
    )
    print(
        f"[rank 0] MPI pool ready, {SIZE - 1} workers, socket_base={socket_base}",
        flush=True,
    )

    local_queue = asyncio.Queue()
    stop = asyncio.Event()

    # Register SIGTERM so that a terminate() from the parent triggers
    # graceful shutdown instead of an abrupt kill.
    def _on_sigterm():
        logger.info("[rank 0] SIGTERM received, setting stop event")
        stop.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, _on_sigterm)

    try:
        logger.info(
            "[rank 0] Entering asyncio.gather for client_loop, msg_loop, mpi_loop, rank0_executor"
        )
        await asyncio.gather(
            _client_loop(task_sock, local_queue, stop),
            _msg_loop(msg_sock, stop),
            _mpi_loop(result_sock, stop),
            _rank0_executor(local_queue, result_sock, stop),
        )
        logger.info("[rank 0] asyncio.gather returned")
    except Exception as e:
        logger.error(f"[rank 0] asyncio.gather raised: {e}")
        raise
    finally:
        logger.info("[rank 0] Entered finally block — beginning teardown")

        logger.info("[rank 0] Stopping MPI workers")
        _stop_workers()
        logger.info("[rank 0] All MPI workers stopped")

        logger.info("[rank 0] Sending 'done' to client")
        try:
            await msg_sock.send(cloudpickle.dumps("done"))
            logger.info("[rank 0] 'done' sent successfully")
        except zmq.ZMQError as e:
            logger.warning(f"[rank 0] Could not send 'done': {e}")

        logger.info("[rank 0] Closing ZMQ sockets")
        task_sock.close()
        result_sock.close()
        msg_sock.close()
        logger.info("[rank 0] ZMQ sockets closed, terminating context")
        ctx.term()
        logger.info("[rank 0] ZMQ context terminated — master shut down cleanly")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket-base", required=True)
    parser.add_argument("--log-dir", default="logs")
    args = parser.parse_args()

    logger = setup_logger(__name__, node_id="mpi_pool", log_dir=args.log_dir)

    if RANK == 0:
        asyncio.run(run_master(args.socket_base))
    else:
        run_worker()

    logger.info(f"[rank {RANK}] Calling MPI.Finalize()")
    MPI.Finalize()
    logger.info(f"[rank {RANK}] MPI.Finalize() complete, exiting")
