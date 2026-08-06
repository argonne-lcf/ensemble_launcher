import argparse
import asyncio
from asyncio import run as asyncio_run
import logging
import multiprocessing as mp
import os
import statistics
import time

from ensemble_launcher.comm import (
    AsyncComm,
    NodeInfo,
    Result,
    Stop,
)
from ensemble_launcher.loop import run as uv_run

"""
The benchmark tests the robustness of the EL heartbeat mechanism.
The goal of this benchmark is to test the robustness of hb mechanism under increasing load of main communication.
Here, robustness is evaluated by
    - setting a constant hb interval and thresholds.
    - increase the number of messages between the processes.
    - Measure the message load at which the false positive is detected
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _make_payload(payload_size):
    if payload_size <= 0:
        return None
    import numpy as np

    return np.random.rand(payload_size)


async def _child_async(
    child_idx,
    mps,
    duration,
    payload_size,
    data_conn,
    hb_conn,
    skip_child_monitors=False,
    heartbeat_interval=1.0,
    heartbeat_dead_threshold=30.0,
):
    child_id = f"child-{child_idx}"
    my_node_info = NodeInfo(
        node_id=child_id,
        secret_id=f"child_secret_{child_idx}",
        parent_id="parent",
        parent_secret_id="parent_secret",
    )
    try:
        comm = AsyncComm(
            logger=logger,
            node_info=my_node_info,
            parent_conn=data_conn,
            hb_parent_conn=hb_conn,
            skip_hb=skip_child_monitors,
            heartbeat_interval=heartbeat_interval,
            heartbeat_dead_threshold=heartbeat_dead_threshold,
        )
    except Exception:
        comm = AsyncComm(
            logger=logger,
            node_info=my_node_info,
            parent_conn=data_conn,
            hb_parent_conn=hb_conn,
        )
    await comm.start_monitors(parent_only=True)

    if not skip_child_monitors:
        await comm.sync_heartbeat_with_parent()

    dt = 1.0 / mps
    payload = _make_payload(payload_size)

    end = time.monotonic() + duration
    counter = 0
    try:
        msg = Result(data=payload, task_id=f"{child_id}-{counter}", success=True).pack()
    except Exception:
        msg = Result(data=payload, task_id=f"{child_id}-{counter}", success=True)
    while time.monotonic() < end:
        counter += 1
        await comm.send_message_to_parent(msg)
        msg = await comm.recv_message_from_parent(Result, block=True)
        await asyncio.sleep(dt)
        if counter % 100 == 0:
            logger.info(f"[{child_id}] Sent/recv {counter} messages")

    logger.info(f"[{child_id}] Waiting for stop")
    await comm.recv_message_from_parent(Stop, block=True)
    logger.info(f"[{child_id}] Received stop")
    await comm.close()


def child_main(
    child_idx,
    mps,
    duration,
    payload_size,
    data_conn,
    hb_conn,
    skip_child_monitors=False,
    heartbeat_interval=1.0,
    heartbeat_dead_threshold=30.0,
    use_uv = False,
):
    if use_uv:
        uv_run(
            _child_async(
                child_idx,
                mps,
                duration,
                payload_size,
                data_conn,
                hb_conn,
                skip_child_monitors,
                heartbeat_interval,
                heartbeat_dead_threshold,
            )
        )
    else:
        asyncio_run(
            _child_async(
                child_idx,
                mps,
                duration,
                payload_size,
                data_conn,
                hb_conn,
                skip_child_monitors,
                heartbeat_interval,
                heartbeat_dead_threshold,
            )
        )



async def busy_wait(duration):
    import cloudpickle

    blob = cloudpickle.dumps(os.urandom(1_000_000_000))
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        # cloudpickle.loads(blob)
        pass


async def main(
    num_children,
    mps,
    duration,
    payload_size,
    hb_interval,
    hb_threshold,
    use_mpi,
    skip_child_monitors=False,
    use_uv = False,
):
    children_ids = [f"child-{i}" for i in range(num_children)]
    children_secret_ids = {
        f"child-{i}": f"child_secret_{i}" for i in range(num_children)
    }

    my_node_info = NodeInfo(
        node_id="parent",
        secret_id="parent_secret",
        children_ids=children_ids,
        children_secret_ids=children_secret_ids,
    )
    comm = AsyncComm(
        logger=logger,
        node_info=my_node_info,
        heartbeat_dead_threshold=hb_threshold,
        heartbeat_interval=hb_interval,
    )

    child_conns = {}
    for i in range(num_children):
        child_id = f"child-{i}"
        result = comm.create_child_pipe(
            child_id=child_id, child_secret_id=f"child_secret_{i}"
        )
        if asyncio.iscoroutine(result):
            result = await result
        data_conn, hb_conn = result
        child_conns[child_id] = (data_conn, hb_conn)

    await comm.start_monitors(children_only=True)

    processes = {}
    mpi_tasks = {}

    if use_mpi:
        from ensemble_launcher.config import MPIConfig
        from ensemble_launcher.executors import AsyncMPIExecutor
        from ensemble_launcher.helper_functions import get_nodes
        from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

        nodes = get_nodes()
        if len(nodes) < 2:
            raise RuntimeError(
                f"--mpi requires at least 2 nodes, got {len(nodes)}: {nodes}"
            )
        remote_node = nodes[1]
        logger.info(f"Launching children on remote node: {remote_node}")

        executor = AsyncMPIExecutor(
            mpi_config=MPIConfig(flavor="mpich", cpu_bind_flag=""),
            return_stdout=True,
        )
        for i in range(num_children):
            child_id = f"child-{i}"
            data_conn, hb_conn = child_conns[child_id]
            job_resource = JobResource(
                resources=[NodeResourceList(cpus=(i,))],
                nodes=[remote_node],
            )
            task = executor.submit(
                job_resource=job_resource,
                task=child_main,
                task_args=(i, mps, duration, payload_size, data_conn, hb_conn, False, hb_interval, hb_threshold, use_uv),
            )
            mpi_tasks[child_id] = task
    else:
        for i in range(num_children):
            child_id = f"child-{i}"
            data_conn, hb_conn = child_conns[child_id]
            p = mp.Process(
                target=child_main,
                args=(
                    i,
                    mps,
                    duration,
                    payload_size,
                    data_conn,
                    hb_conn,
                    skip_child_monitors,
                    hb_interval,
                    hb_threshold,
                    use_uv,
                ),
            )
            p.start()
            processes[child_id] = p

    success = True
    false_positives = []
    try:
        if not skip_child_monitors:
            for child_id in children_ids:
                ok = await comm.sync_heartbeat_with_child(
                    child_id=child_id, timeout=30.0
                )
                if not ok:
                    logger.error(f"Failed to sync heartbeat with {child_id}")
                    success = False
            logger.info(f"Synced heartbeat with all {num_children} children")

        # busy_wait_task = asyncio.create_task(busy_wait(duration))

        payload = _make_payload(payload_size)
        end = time.monotonic() + duration
        dt = 1.0 / mps
        times = []
        while time.monotonic() < end:
            tic = time.perf_counter()
            for child_id in children_ids:
                msg = await comm.recv_message_from_child(
                    Result, child_id=child_id, block=True, timeout=2.0
                )
                if msg is not None:
                    # reply = Result(data=payload, task_id="ack", success=True)
                    await comm.send_message_to_child(child_id=child_id, msg=msg)
            toc = time.perf_counter()
            times.append(toc - tic)
            await asyncio.sleep(dt)

            for child_id in children_ids:
                if comm._child_dead_events[child_id].is_set():
                    if child_id not in false_positives:
                        false_positives.append(child_id)
                        logger.warning(
                            f"False positive: {child_id} reported dead at "
                            f"t={duration - (end - time.monotonic()):.1f}s"
                        )

        for child_id in children_ids:
            await comm.send_message_to_child(child_id=child_id, msg=Stop())
    except Exception as e:
        logger.error(f"Main failed with exception {e}")
        success = False
    finally:
        # busy_wait_task.cancel()
        await comm.close()
        if use_mpi:
            for child_id, task in mpi_tasks.items():
                try:
                    await asyncio.wait_for(task, timeout=10.0)
                except (asyncio.TimeoutError, Exception) as e:
                    logger.warning(f"MPI child {child_id} cleanup: {e}")
            await executor.ashutdown(wait=True)
        else:
            for child_id, p in processes.items():
                p.join(timeout=10.0)
                if p.is_alive():
                    p.kill()

    if false_positives:
        success = False

    return success, false_positives, statistics.mean(times)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Heartbeat robustness benchmark")
    parser.add_argument("--children", type=int, default=1, help="Number of children")
    parser.add_argument(
        "--mps", type=int, default=10, help="Messages per second per child"
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Test duration in seconds"
    )
    parser.add_argument(
        "--payload-size", type=int, default=0, help="Payload size in bytes (0 = empty)"
    )
    parser.add_argument(
        "--hb-interval", type=float, default=1.0, help="Heartbeat interval in seconds"
    )
    parser.add_argument(
        "--hb-threshold",
        type=float,
        default=5.0,
        help="Heartbeat dead threshold in seconds",
    )
    parser.add_argument(
        "--mpi",
        action="store_true",
        help="Launch children on remote node via AsyncMPIExecutor",
    )
    parser.add_argument(
        "--use-uv",
        action="store_true",
        help="Launch children on remote node via AsyncMPIExecutor",
    )
    parser.add_argument(
        "--skip-child-monitors",
        action="store_true",
        help="Skip HB process and data server on child nodes",
    )
    args = parser.parse_args()

    if args.use_uv:
        success, false_positives, mean_dt = uv_run(
            main(
                args.children,
                args.mps,
                args.duration,
                args.payload_size,
                args.hb_interval,
                args.hb_threshold,
                args.mpi,
                args.skip_child_monitors,
                args.use_uv,
            )
        )
    else:
        success, false_positives, mean_dt = asyncio_run(
            main(
                args.children,
                args.mps,
                args.duration,
                args.payload_size,
                args.hb_interval,
                args.hb_threshold,
                args.mpi,
                args.skip_child_monitors,
                args.use_uv,
            )
        )
    print(
        f"\nChildren: {args.children}, MPS/child: {args.mps}, "
        f"Total MPS: {args.children * args.mps}, Duration: {args.duration}s, "
        f"Payload: {args.payload_size}B, "
        f"HB interval: {args.hb_interval}s, HB threshold: {args.hb_threshold}s, "
        f"MPI: {args.mpi}"
    )
    print(f"Mean round trip time: {mean_dt}")
    if success:
        print("PASSED — no false positive death detections")
    else:
        print(f"FAILED — false positives: {false_positives}")
