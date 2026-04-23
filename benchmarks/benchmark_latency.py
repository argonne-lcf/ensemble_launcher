import csv
import datetime
import os
import subprocess
import time

from ensemble_launcher.config import LauncherConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient
from utils import noop

from ensemble_launcher import EnsembleLauncher


def benchmark_roundtrip_latency(
    nodes: list[str], n_repeats: int = 100, nlevels: int = 1, nchildren: int = 2
):
    """
    Measure round-trip latency for a single no-op serial task in cluster mode.
    Submits the task, waits for the result, and records the elapsed time.
    Repeats n_repeats times then returns the latency array.
    """
    import uuid

    import numpy as np

    ckpt_dir = os.path.join(os.getcwd(), f"ckpt_{str(uuid.uuid4())}")
    ncpus = 12

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=ncpus, cpus=list(range(ncpus))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            child_executor_name="async_mpi",
            comm_name="async_zmq",
            nlevels=nlevels,
            return_stdout=True,
            master_logs=True,
            worker_logs=True,
            cpu_binding_option="",
            use_mpi_ppn=False,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            children_scheduler_policy="simple_split_children_policy",
            nchildren=nchildren,
        ),
        Nodes=nodes,
    )

    el.start()
    time.sleep(2.0)

    latencies = []
    with ClusterClient(checkpoint_dir=ckpt_dir, node_id="global") as client:
        for i in range(n_repeats):
            task = Task(task_id=f"latency-{i}", executable=noop, nnodes=1, ppn=1)
            t0 = time.perf_counter()
            fut = client.submit(task)
            fut.result(timeout=60.0)
            t1 = time.perf_counter()
            latencies.append(t1 - t0)

    el.stop()

    return np.array(latencies)


RESULTS_CSV = os.path.join(os.path.dirname(__file__), "latency_results.csv")
_CSV_FIELDS = [
    "timestamp",
    "git_commit",
    "nnodes",
    "nlevels",
    "nchildren",
    "n_repeats",
    "mean_ms",
    "std_ms",
]


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def record_result(nnodes: int, nlevels: int, nchildren: int, n_repeats: int, latencies):
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "nnodes": nnodes,
        "nlevels": nlevels,
        "nchildren": nchildren,
        "n_repeats": n_repeats,
        "mean_ms": f"{latencies.mean() * 1e3:.3f}",
        "std_ms": f"{latencies.std() * 1e3:.3f}",
    }
    write_header = not os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"  (recorded to {RESULTS_CSV})")


def benchmark_extra_level_latency(nodes: list[str], n_repeats: int = 100):
    """
    Compare round-trip latency on 2 nodes with nlevels=1 vs nlevels=2 to
    quantify the overhead added by an extra hierarchy level.
    """

    nodes2 = nodes[:2]

    print(f"Running nlevels=1 on 2 nodes ({n_repeats} repeats)...")
    lat1 = benchmark_roundtrip_latency(
        nodes2, n_repeats=n_repeats, nlevels=1, nchildren=2
    )
    print(f"Running nlevels=2 on 2 nodes ({n_repeats} repeats)...")
    lat2 = benchmark_roundtrip_latency(
        nodes2, n_repeats=n_repeats, nlevels=2, nchildren=2
    )

    overhead = lat2 - lat1
    print(f"\nLatency comparison (2 nodes, no-op serial task, {n_repeats} repeats):")
    print(
        f"  nlevels=1  mean: {lat1.mean() * 1e3:.3f} ms  std: {lat1.std() * 1e3:.3f} ms"
    )
    print(
        f"  nlevels=2  mean: {lat2.mean() * 1e3:.3f} ms  std: {lat2.std() * 1e3:.3f} ms"
    )
    print(
        f"  extra-level overhead  mean: {overhead.mean() * 1e3:.3f} ms  std: {overhead.std() * 1e3:.3f} ms"
    )
    record_result(
        len(nodes2), nlevels=1, nchildren=2, n_repeats=n_repeats, latencies=lat1
    )
    record_result(
        len(nodes2), nlevels=2, nchildren=2, n_repeats=n_repeats, latencies=lat2
    )


if __name__ == "__main__":
    import argparse

    from ensemble_launcher.helper_functions import get_nodes

    parser = argparse.ArgumentParser(
        description="Benchmark round-trip latency in cluster mode"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
        help="Number of measurement repeats (default: 100)",
    )
    parser.add_argument(
        "--extra-level",
        action="store_true",
        help="Run the 4-node extra-level latency comparison instead of the 2-node baseline",
    )
    args = parser.parse_args()

    nodes = get_nodes()

    if len(nodes) < 2:
        raise ValueError(
            f"At least 2 nodes are required for this benchmark, got {len(nodes)}."
        )
    if args.extra_level:
        benchmark_extra_level_latency(nodes=nodes, n_repeats=args.repeats)
    else:
        latencies = benchmark_roundtrip_latency(nodes=nodes, n_repeats=args.repeats)
        print(
            f"Round-trip latency over {args.repeats} runs (2-node no-op serial task):"
        )
        print(f"  mean : {latencies.mean() * 1e3:.3f} ms")
        print(f"  std  : {latencies.std() * 1e3:.3f} ms")
        record_result(
            len(nodes),
            nlevels=1,
            nchildren=2,
            n_repeats=args.repeats,
            latencies=latencies,
        )
