from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.orchestrator.async_worker import AsyncWorker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceCount
import multiprocessing as mp
import logging
import time
import argparse
import random

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def echo(sleep: float):
    import time
    time.sleep(sleep)
    return

def benchmark_worker(ntasks_per_core=10, min_sleep_time=1.0, max_sleep_time=10.0, seed=None):
    if seed is not None:
        random.seed(seed)
    
    # Generate random sleep times once
    sleep_times = [random.uniform(min_sleep_time, max_sleep_time) 
                   for _ in range(mp.cpu_count() * ntasks_per_core)]
    
    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))

    print(f"\nTask sleep times: min={min(sleep_times):.2f}s, max={max(sleep_times):.2f}s, avg={sum(sleep_times)/len(sleep_times):.2f}s")
    print(f"Total sleep time: {sum(sleep_times):.2f}s")
    print(f"Theoretical minimum (parallel): {sum(sleep_times)/mp.cpu_count():.2f}s")
    print("-" * 60)

    results = {}
    
    # Test both Worker and AsyncWorker
    for worker_type in ["sync", "async"]:
        results[worker_type] = {}
        print(f"\nTesting {worker_type.upper()} Worker:")
        print("-" * 60)
        
        for exec in ["multiprocessing", "mpi"]:
            # Create fresh tasks with the same sleep times
            tasks = {}
            for i in range(len(sleep_times)):
                tasks[f"task-{i}"] = Task(
                    task_id=f"task-{i}",
                    nnodes=1,
                    ppn=1,
                    executable=echo,
                    args=(sleep_times[i],)
                )
            
            if worker_type == "async":
                w = AsyncWorker(
                    "test-async",
                    LauncherConfig(task_executor_name=exec, worker_logs=False),
                    sys_info,
                    nodes,
                    tasks
                )
            else:
                w = Worker(
                    "test-sync",
                    LauncherConfig(task_executor_name=exec, worker_logs=False),
                    sys_info,
                    nodes,
                    tasks
                )

            print(f"  Running {exec}...", end=" ", flush=True)
            tic = time.perf_counter()
            res = w.run()
            toc = time.perf_counter()
            elapsed = toc - tic
            results[worker_type][exec] = elapsed
            print(f"{elapsed:.3f}s")
    
    return results, sleep_times

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark worker performance with random task runtimes')
    parser.add_argument('--ntasks-per-core', type=int, default=10, 
                       help='Number of tasks per CPU core (default: 10)')
    parser.add_argument('--min-sleep-time', type=float, default=1.0,
                       help='Minimum sleep time for each task in seconds (default: 1.0)')
    parser.add_argument('--max-sleep-time', type=float, default=10.0,
                       help='Maximum sleep time for each task in seconds (default: 10.0)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    print(f"\nRunning benchmark with:")
    print(f"  Tasks per core: {args.ntasks_per_core}")
    print(f"  Total tasks: {mp.cpu_count() * args.ntasks_per_core}")
    print(f"  CPU cores: {mp.cpu_count()}")
    print(f"  Sleep time range: {args.min_sleep_time}s - {args.max_sleep_time}s")
    print(f"  Random seed: {args.seed}")

    results, sleep_times = benchmark_worker(
        args.ntasks_per_core, 
        args.min_sleep_time, 
        args.max_sleep_time,
        args.seed
    )

    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    
    theoretical_min = sum(sleep_times) / mp.cpu_count()
    
    for worker_type in ["sync", "async"]:
        print(f"\n{worker_type.upper()} Worker:")
        print("-" * 60)
        for executor, time_taken in results[worker_type].items():
            efficiency = (theoretical_min / time_taken) * 100
            print(f"  {executor:>20}: {time_taken:>8.3f}s  (efficiency: {efficiency:>5.1f}%)")
    
    print("\n" + "=" * 60)
    print("COMPARISON (Async vs Sync)")
    print("=" * 60)
    for executor in ["multiprocessing", "mpi"]:
        sync_time = results["sync"][executor]
        async_time = results["async"][executor]
        speedup = sync_time / async_time
        faster_slower = "faster" if speedup > 1 else "slower"
        print(f"  {executor:>20}: {speedup:>6.3f}x {faster_slower}")
    print("=" * 60)
