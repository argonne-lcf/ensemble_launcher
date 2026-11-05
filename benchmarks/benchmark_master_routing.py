"""How long a sub-master spends routing tasks down and results up, and how much of it is GC.

Drives a real ``AsyncMaster`` (real ``AsyncChildrenScheduler``, real routing and flush code)
with the comm layer stubbed out: nothing is launched and no socket is opened. The master is
set up as ``main.m0``, a sub-master with a parent and ``--nchildren`` workers, and each batch
goes through what that node does in a real run:

- ``decode_tasks``   ``TaskUpdate.from_byte_array`` on frames as they arrive from the parent
- ``route``          ``_route_tasks``: scheduler bookkeeping and the per-child queues
- ``task_flush``     one ``TaskUpdate`` per child, encoded with ``to_byte_array``
- ``decode_results`` ``IResultBatch.from_byte_array`` on one batch per child
- ``forward``        ``_forward_result``: status updates and the queue to the parent
- ``result_flush``   the ``IResultBatch`` to the parent, encoded with ``to_byte_array``

The master keeps every task it routes, as in a real run, so the live heap -- and with it
the cost of each full collection -- grows batch by batch. Per stage it prints wall time,
the number of GC collections and the time spent in them::

    python benchmark_master_routing.py --ntasks 100000 --nbatches 10 --nchildren 128

``--gc off`` / ``--gc freeze`` give the baselines to compare against. ``--pyspy DIR`` samples
the benchmark itself with py-spy (needs ``sudo`` on macOS) from the first batch to the last
and writes ``DIR/merged_perfetto.json`` for ui.perfetto.dev; setup is left out of the trace.
"""

import argparse
import asyncio
import gc
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

from utils import noop

from ensemble_launcher.comm import IResultBatch, Message, NodeInfo, Result, TaskUpdate
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import AsyncMaster
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

STAGES = ("decode_tasks", "route", "task_flush", "decode_results", "forward", "result_flush")


# --------------------------------------------------------------------------- #
#  GC accounting                                                              #
# --------------------------------------------------------------------------- #


class GCStats:
    """Collections and their durations, attributed to whichever stage is running."""

    def __init__(self):
        self.stage = None
        self._t0 = None
        self.count = defaultdict(int)
        self.seconds = defaultdict(float)
        self.longest = defaultdict(float)
        self.full = defaultdict(int)  # generation-2 collections
        gc.callbacks.append(self._callback)

    def _callback(self, phase, info):
        if phase == "start":
            self._t0 = time.perf_counter()
            return
        if self._t0 is None or self.stage is None:
            return
        dt = time.perf_counter() - self._t0
        self._t0 = None
        self.count[self.stage] += 1
        self.seconds[self.stage] += dt
        self.longest[self.stage] = max(self.longest[self.stage], dt)
        if info.get("generation") == 2:
            self.full[self.stage] += 1

    def close(self):
        gc.callbacks.remove(self._callback)


class Timer:
    def __init__(self, gc_stats: GCStats):
        self.gc_stats = gc_stats
        self.wall = defaultdict(float)

    @contextmanager
    def stage(self, name):
        self.gc_stats.stage = name
        tic = time.perf_counter()
        try:
            yield
        finally:
            self.wall[name] += time.perf_counter() - tic
            self.gc_stats.stage = None


# --------------------------------------------------------------------------- #
#  Comm stub                                                                  #
# --------------------------------------------------------------------------- #


class EncodingComm:
    """Stands in for AsyncZMQComm: encodes what the master sends, then drops it.

    Encoding is the master's share of a send; delivery and the receiver's decode are not
    on this node's timeline, so they are left out.
    """

    def __init__(self):
        self.messages = 0
        self.bytes = 0

    def _encode(self, msg):
        frames = msg.to_byte_array()
        self.messages += 1
        self.bytes += sum(len(f) for f in frames)
        return True

    async def send_message_to_child(self, child_id, msg, retries=1):
        return self._encode(msg)

    async def send_message_to_parent(self, msg, retries=1):
        return self._encode(msg)


# --------------------------------------------------------------------------- #
#  Setup                                                                      #
# --------------------------------------------------------------------------- #


def make_master(nchildren: int, cpus_per_node: int) -> AsyncMaster:
    """A sub-master ``main.m0`` with ``nchildren`` ready workers, one per (fake) node."""
    config = LauncherConfig(
        nlevels=2,
        policy_config=PolicyConfig(nlevels=2, nchildren=nchildren),
        cluster=False,  # _route_tasks flushes every batch straight away
        overload_orchestrator_core=True,
    )
    parent = NodeInfo(node_id="main", secret_id="bench", level=0)
    node = NodeResourceList.from_config(
        SystemConfig(name="bench", ncpus=cpus_per_node, cpus=list(range(cpus_per_node)))
    )
    nodes = JobResource(
        resources=[node] * nchildren, nodes=[f"node{i}" for i in range(nchildren)]
    )
    m = AsyncMaster("main.m0", config, Nodes=nodes, parent=parent)
    m.logger = logging.getLogger("bench.master")
    m._comm = EncodingComm()
    m._scheduler = m._create_scheduler()
    m._scheduler.assign_resources(m.level, m.node_id)  # registers children as READY
    for child_id in m._scheduler.children_names:
        m._scheduler.mark_child_running(child_id)
    return m


def task_update_frames(batch: int, ntasks: int) -> list:
    """One encoded TaskUpdate of ``ntasks`` no-op tasks, as the parent would send it."""
    tasks = [
        Task(task_id=f"b{batch}-t{i}", nnodes=1, ppn=1, executable=noop)
        for i in range(ntasks)
    ]
    return TaskUpdate(sender="main", added_tasks=tasks).to_byte_array()


def result_frames(m: AsyncMaster, task_ids) -> list:
    """One encoded IResultBatch per child, covering the tasks it was given."""
    by_child = defaultdict(list)
    for tid in task_ids:
        by_child[m._scheduler._task_to_child[tid]].append(tid)
    return [
        IResultBatch(
            sender=child_id,
            data=[Result(sender=child_id, task_id=tid, data=None) for tid in tids],
        ).to_byte_array()
        for child_id, tids in by_child.items()
    ]


async def drain(inflight: dict) -> None:
    """Wait for every flush send the master started."""
    while True:
        pending = [t for ts in inflight.values() for t in ts]
        if not pending:
            return
        await asyncio.gather(*pending, return_exceptions=True)
        await asyncio.sleep(0)  # let the done-callbacks remove them


# --------------------------------------------------------------------------- #
#  py-spy on ourselves                                                        #
# --------------------------------------------------------------------------- #


class SelfProfile:
    """``py-spy record`` on this process, in the layout ``merge_traces`` reads."""

    NODE = "bench"

    def __init__(self, out_dir: str, rate: int, native: bool):
        self.out_dir = os.path.abspath(out_dir)
        os.makedirs(self.out_dir, exist_ok=True)
        self.trace = os.path.join(self.out_dir, f"{self.NODE}_pyspy.json")
        pyspy = shutil.which("py-spy")
        if pyspy is None:
            sys.exit("--pyspy needs py-spy on PATH (pip install py-spy)")
        cmd = [
            pyspy, "record", "--pid", str(os.getpid()), "--format", "chrometrace",
            "-o", self.trace, "--rate", str(rate), "--threads", "--idle",
        ]  # fmt: skip
        if native:
            cmd.append("--native")
        start_us = time.time_ns() // 1000
        self.proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with open(os.path.join(self.out_dir, f"{self.NODE}_pyspy.meta.json"), "w") as f:
            json.dump({"node_id": self.NODE, "hostname": os.uname().nodename,
                       "pid": self.proc.pid, "start_us": start_us}, f)  # fmt: skip
        time.sleep(1.0)  # give py-spy time to attach before the first batch
        if self.proc.poll() is not None:
            sys.exit(f"py-spy exited: {self.proc.stderr.read().decode().strip()}")

    def stop(self) -> None:
        from ensemble_launcher.profiling.merge_traces import merge_traces

        self.proc.send_signal(signal.SIGINT)  # py-spy writes its trace on SIGINT
        try:
            self.proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        if not os.path.exists(self.trace):
            print(f"py-spy wrote no trace: {self.proc.stderr.read().decode().strip()}")
            return
        merged = os.path.join(self.out_dir, "merged_perfetto.json")
        merge_traces([self.trace], merged)
        print(f"Trace: {merged}")


# --------------------------------------------------------------------------- #
#  Run                                                                        #
# --------------------------------------------------------------------------- #


async def run_batches(m: AsyncMaster, frames: list, timer: Timer, gc_stats: GCStats):
    rows = []
    parent_id = m.parent.node_id
    for batch, task_frames in enumerate(frames):
        before_wall = dict(timer.wall)
        before_gc = dict(gc_stats.seconds)

        with timer.stage("decode_tasks"):
            msg = Message.from_byte_array(task_frames)
        with timer.stage("route"):
            await m._route_tasks(msg.added_tasks)
        with timer.stage("task_flush"):
            await drain(m._task_flush_inflight)

        # Children report back. Building their frames is the children's work, not the
        # master's, so it happens outside every stage.
        task_ids = [t.task_id for t in msg.added_tasks]
        del msg
        replies = result_frames(m, task_ids)

        for reply in replies:
            with timer.stage("decode_results"):
                results = Message.from_byte_array(reply)
            with timer.stage("forward"):
                await m._forward_result(results)
        with timer.stage("result_flush"):
            m._schedule_result_flush(parent_id)
            await drain(m._result_flush_inflight)
        del replies, results

        wall = sum(timer.wall.values()) - sum(before_wall.values())
        gc_s = sum(gc_stats.seconds.values()) - sum(before_gc.values())
        rows.append((batch, len(m.tasks), wall, gc_s))
        print(
            f"  batch {batch:3d}  live tasks {len(m.tasks):>10,d}  "
            f"{wall:8.3f}s  gc {gc_s:7.3f}s ({100 * gc_s / wall if wall else 0:4.1f}%)"
        )
    return rows


def report(timer: Timer, gc_stats: GCStats, comm: EncodingComm, ntotal: int) -> None:
    print()
    print(f"  {'stage':16s} {'wall':>9s} {'us/task':>8s} {'gc':>9s} {'gc %':>6s}"
          f" {'colls':>6s} {'full':>5s} {'longest':>9s}")  # fmt: skip
    total_wall = total_gc = 0.0
    for name in STAGES:
        wall, gcs = timer.wall[name], gc_stats.seconds[name]
        total_wall += wall
        total_gc += gcs
        print(
            f"  {name:16s} {wall:8.3f}s {1e6 * wall / ntotal:8.2f} {gcs:8.3f}s"
            f" {100 * gcs / wall if wall else 0:5.1f}% {gc_stats.count[name]:6d}"
            f" {gc_stats.full[name]:5d} {gc_stats.longest[name]:8.3f}s"
        )
    print(
        f"  {'total':16s} {total_wall:8.3f}s {1e6 * total_wall / ntotal:8.2f} {total_gc:8.3f}s"
        f" {100 * total_gc / total_wall if total_wall else 0:5.1f}%"
    )
    print(f"  sent {comm.messages} messages, {comm.bytes / 1e6:.1f} MB")


async def main(args) -> None:
    logging.basicConfig(level=logging.ERROR)
    m = make_master(args.nchildren, args.cpus_per_node)
    print(
        f"main.m0: {len(m._scheduler.children_names)} children, "
        f"{args.nbatches} batches x {args.ntasks:,d} tasks, gc={args.gc}, "
        f"python {sys.version.split()[0]}"
    )
    frames = [task_update_frames(b, args.ntasks) for b in range(args.nbatches)]

    gc.collect()
    if args.gc == "freeze":
        gc.freeze()  # setup objects out of the way; new ones are still collected
    elif args.gc == "off":
        gc.disable()

    profile = SelfProfile(args.pyspy, args.rate, args.native) if args.pyspy else None
    gc_stats = GCStats()
    timer = Timer(gc_stats)
    try:
        await run_batches(m, frames, timer, gc_stats)
    finally:
        gc_stats.close()
        gc.enable()
        if profile:
            profile.stop()
    report(timer, gc_stats, m._comm, args.ntasks * args.nbatches)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--ntasks", type=int, default=100_000, help="tasks per TaskUpdate")
    p.add_argument("--nbatches", type=int, default=10)
    p.add_argument("--nchildren", type=int, default=128)
    p.add_argument("--cpus-per-node", type=int, default=102)
    p.add_argument("--gc", choices=("on", "off", "freeze"), default="on")
    p.add_argument("--pyspy", metavar="DIR", help="profile the batches with py-spy into DIR")
    p.add_argument("--rate", type=int, default=200, help="py-spy samples per second")
    p.add_argument("--native", action="store_true", help="py-spy --native")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
