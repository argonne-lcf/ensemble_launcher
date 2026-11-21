"""Allocate/deallocate throughput of EL's cluster resource.

Builds an ``AsyncLocalClusterResource`` with ``--nnodes`` nodes and pushes
``--tasks-per-node`` one-core tasks per node through it in waves: allocate until nothing
fits, then deallocate everything held. Prints progress every few seconds (Ctrl-C stops early
and still prints the totals), then the time per allocate and per deallocate.

    python benchmark_resources.py --nnodes 10000 --tasks-per-node 10000
    python benchmark_resources.py --ngpus-per-task 0.25
"""
import argparse
import logging
import time

from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)
from ensemble_launcher.scheduler.resource.cluster import AsyncLocalClusterResource


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--nnodes", type=int, default=10_000)
    p.add_argument("--tasks-per-node", type=int, default=100)
    p.add_argument("--ncpus", type=int, default=102, help="cores per node")
    p.add_argument("--ngpus", type=int, default=12, help="GPUs per node")
    p.add_argument("--ncpus-per-task", type=int, default=1)
    p.add_argument("--ngpus-per-task", type=float, default=0.0)
    p.add_argument("--report", type=float, default=5.0, help="seconds between progress lines")
    a = p.parse_args()

    t = time.perf_counter()
    nodes = JobResource(
        resources=[NodeResourceList(cpus=tuple(range(a.ncpus)), gpus=tuple(range(a.ngpus)))
                   for _ in range(a.nnodes)],
        nodes=[f"node{i}" for i in range(a.nnodes)],
    )
    cluster = AsyncLocalClusterResource(logging.getLogger("bench"), nodes)
    req = JobResource(resources=[NodeResourceCount(ncpus=a.ncpus_per_task, ngpus=a.ngpus_per_task)])
    total = a.nnodes * a.tasks_per_node
    print(f"cluster: {a.nnodes:,} nodes x {a.ncpus} cores, {a.ngpus} GPUs "
          f"(built in {time.perf_counter() - t:.1f}s); {total:,} tasks of "
          f"{a.ncpus_per_task} core(s), {a.ngpus_per_task} GPU(s)")

    t_alloc = t_dealloc = 0.0
    n_alloc = n_dealloc = 0
    start = time.perf_counter()
    next_report = start + a.report

    def line(tag):
        el = time.perf_counter() - start
        print(f"{tag} {el:8.1f}s  {n_alloc:>13,d}/{total:,} allocated  "
              f"allocate {1e6 * t_alloc / max(n_alloc, 1):9.2f} us  "
              f"deallocate {1e6 * t_dealloc / max(n_dealloc, 1):9.2f} us", flush=True)

    def maybe_report():
        nonlocal next_report
        now = time.perf_counter()
        if now >= next_report:
            line("  ")
            next_report = now + a.report

    try:
        while n_alloc < total:
            held = []
            while n_alloc < total:  # fill: allocate until nothing fits
                t0 = time.perf_counter()
                ok, res = cluster.allocate(req)
                t_alloc += time.perf_counter() - t0
                if not ok:
                    break
                held.append(res)
                n_alloc += 1
                if n_alloc % 1000 == 0:
                    maybe_report()
            if not held:
                raise SystemExit("a task does not fit on an empty cluster")
            for i, res in enumerate(held, 1):  # drain: deallocate everything held
                t0 = time.perf_counter()
                cluster.deallocate(res)
                t_dealloc += time.perf_counter() - t0
                n_dealloc += 1
                if i % 1000 == 0:
                    maybe_report()
    except KeyboardInterrupt:
        print("  interrupted")
    line("done")


if __name__ == "__main__":
    main()