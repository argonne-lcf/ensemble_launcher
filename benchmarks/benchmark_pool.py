"""Throughput benchmark for a single-node worker pool in cluster mode.

Submits ``--nbatches`` batches of ``--ntasks`` no-op tasks through a
``ClusterClient`` and reports the mean per-batch throughput. Pass several
values to ``--ntasks`` to sweep the batch size; each value runs against a fresh
launcher so earlier sweeps don't skew later ones.

    python benchmark_pool.py --ntasks 1000 10000 --nbatches 5 --policy fifo_policy

``--ppn 1 1 1 2 3 4`` makes the workload heterogeneous: each task draws its core
count uniformly from the list (seeded, so every run submits the same tasks).
"""

import argparse
import concurrent.futures
import os
import random
import socket
import time
import uuid

from utils import noop

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.orchestrator import ClusterClient


def _create_tasks(ntasks, prefix="", ppn_choices=(1,), seed=0):
    rng = random.Random(seed)
    return [
        Task(
            task_id=f"{prefix}{idx}",
            nnodes=1,
            ppn=rng.choice(ppn_choices),
            executable=noop,
        )
        for idx in range(ntasks)
    ]


def _system_config():
    if os.cpu_count() == 104:
        cpus = list(range(104))
        cpus.pop(52)  # can't use these cores on Aurora
        cpus.pop(0)  # can't use these cores on Aurora
        gpus = list(range(12))
        return SystemConfig(name="system", cpus=cpus, gpus=gpus)
    cpus = list(range(os.cpu_count()))
    return SystemConfig(name="system", cpus=cpus, ncpus=len(cpus))


def benchmark_max_throughput(
    task_executor="async_processpool",
    ntasks=1000,
    nbatches=29,
    policy="fifo_policy",
    timeout=None,
    warmup=10.0,
    logger=None,
    ppn_choices=(1,),
    strict_priority=False,
):
    """Return mean throughput (tasks/sec) over ``nbatches`` batches of ``ntasks``."""
    logger = logger or setup_logger("Pool_benchmark")
    if timeout is None:
        timeout = max(60.0, ntasks / 100.0)

    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    os.makedirs(ckpt_dir)
    launcher_config = LauncherConfig(
        task_executor_name=task_executor,
        task_scheduler_policy=policy,
        cluster=True,
        worker_logs=False,
        policy_config=PolicyConfig(nlevels=0, strict_priority=strict_priority),
        result_flush_interval=0.05,
        checkpoint_dir=ckpt_dir,
    )

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=_system_config(),
        launcher_config=launcher_config,
        Nodes=[socket.gethostname()],
    )
    el.start()
    time.sleep(warmup)
    times = []
    try:
        with ClusterClient(checkpoint_dir=ckpt_dir) as client:
            # Warm-up batch: spins up the executor's worker processes.
            futures = client.submit_batch(tasks=_create_tasks(102, prefix="warm-"))
            concurrent.futures.wait(futures, timeout=60)

            for b in range(nbatches):
                futures = client.submit_batch(
                    tasks=_create_tasks(
                        ntasks, prefix=f"b{b}-", ppn_choices=ppn_choices, seed=b
                    )
                )
                tic = time.perf_counter()
                _done, not_done = concurrent.futures.wait(futures, timeout=timeout)
                toc = time.perf_counter()
                if not_done:
                    logger.warning(
                        f"batch {b}: {len(not_done)}/{ntasks} tasks not done after {timeout}s"
                    )
                times.append(toc - tic)
                logger.info(
                    f"batch {b}: {ntasks / times[-1]:.1f} tasks/sec ({times[-1]:.3f}s)"
                )
    finally:
        el.stop()

    throughput = ntasks / (sum(times) / len(times))
    logger.info(
        f"Throughput: {throughput:.1f} tasks/sec, ntasks: {ntasks}, "
        f"policy: {policy}, executor: {task_executor}, ppn: {list(ppn_choices)}, "
        f"strict_priority: {strict_priority}"
    )
    return throughput


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task-executor", default="async_processpool")
    parser.add_argument(
        "--ntasks",
        type=int,
        nargs="+",
        default=[1000],
        help="tasks per batch; pass several values to sweep",
    )
    parser.add_argument("--nbatches", type=int, default=29)
    parser.add_argument(
        "--policy", default="fifo_policy", help="task_scheduler_policy name"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="per-batch wait timeout in seconds (default: max(60, ntasks/100))",
    )
    parser.add_argument(
        "--warmup", type=float, default=10.0, help="seconds to wait after start()"
    )
    parser.add_argument(
        "--ppn",
        type=int,
        nargs="+",
        default=[1],
        help="cores per task, drawn uniformly per task (e.g. 1 1 1 2 3 4 for a mixed workload)",
    )
    parser.add_argument(
        "--strict-priority",
        action="store_true",
        help="start tasks in strict priority order (no backfill)",
    )
    args = parser.parse_args()

    logger = setup_logger("Pool_benchmark")
    results = {}
    for n in args.ntasks:
        results[n] = benchmark_max_throughput(
            task_executor=args.task_executor,
            ntasks=n,
            nbatches=args.nbatches,
            policy=args.policy,
            timeout=args.timeout,
            warmup=args.warmup,
            logger=logger,
            ppn_choices=args.ppn,
            strict_priority=args.strict_priority,
        )

    print(
        f"\npolicy={args.policy} executor={args.task_executor} nbatches={args.nbatches} "
        f"ppn={args.ppn} strict_priority={args.strict_priority}"
    )
    print(f"{'ntasks':>10} {'tasks/sec':>12}")
    for n, tp in results.items():
        print(f"{n:>10} {tp:>12.1f}")


if __name__ == "__main__":
    main()
