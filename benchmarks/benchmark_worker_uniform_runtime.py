from ensemble_launcher.orchestrator import Worker
from ensemble_launcher.orchestrator import AsyncWorker
from ensemble_launcher.ensemble import Task
import socket
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.scheduler.resource import NodeResourceCount
import multiprocessing as mp
import logging
import time
import argparse
import asyncio
import csv
import json
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def echo(sleep: float):
    import time
    time.sleep(sleep)
    return


async def benchmark_async_worker(ntasks_per_core=10, sleep_time=1.0,async_worker=False):
    ##create tasks
    tasks = {}
    for i in range(mp.cpu_count()*ntasks_per_core):
        tasks[f"task-{i}"] = \
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo,
                 args=(sleep_time,))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))

    run_time = {}

    w = AsyncWorker("test",LauncherConfig(task_executor_name="async_processpool",comm_name="async_zmq",worker_logs=False),sys_info,nodes,tasks)

    tic = time.perf_counter()
    res = await w.run()
    toc = time.perf_counter()
    run_time["processpool"] = toc-tic
    
    return run_time


def benchmark_worker(ntasks_per_core=10, sleep_time=1.0,async_worker=False):
    ##create tasks
    tasks = {}
    for i in range(mp.cpu_count()*ntasks_per_core):
        tasks[f"task-{i}"] = \
            Task(task_id=f"task-{i}",
                 nnodes=1,
                 ppn=1,
                 executable=echo,
                 args=(sleep_time,))

    nodes = [socket.gethostname()]
    sys_info = NodeResourceCount.from_config(SystemConfig(name="local"))

    run_time = {}
    # for exec in ["multiprocessing","mpi"]:
    w = Worker("test",LauncherConfig(task_executor_name="multiprocessing",comm_name="zmq",worker_logs=False),sys_info,nodes,tasks)

    tic = time.perf_counter()
    res = w.run()
    toc = time.perf_counter()
    run_time["processpool"] = toc-tic
    
    return run_time

def run_strong_scaling_experiment(total_work_per_core=10.0, output_file="strong_scaling_results.csv"):
    """
    Run strong scaling experiment where total work per core is constant.
    Total work = ntasks_per_core * sleep_time
    
    Args:
        total_work_per_core: Total work time per core in seconds (default: 10.0)
        output_file: Output CSV file name
    """
    # Define test configurations: (ntasks_per_core, sleep_time)
    # Keep total_work_per_core constant
    configurations = []
    
    # Generate configurations from sleep_time 10s to 0.001s
    sleep_times = [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    for sleep_time in sleep_times:
        ntasks = int(total_work_per_core / sleep_time)
        if ntasks > 0 and ntasks <= 10000:
            configurations.append((ntasks, sleep_time))
    
    print(f"Strong Scaling Experiment")
    print(f"Total work per core: {total_work_per_core}s")
    print(f"Number of CPU cores: {mp.cpu_count()}")
    print(f"Total configurations: {len(configurations)}")
    print("=" * 60)
    
    results = []
    
    for i, (ntasks, sleep_time) in enumerate(configurations, 1):
        print(f"\n[{i}/{len(configurations)}] Running: {ntasks} tasks/core × {sleep_time}s sleep")
        print(f"  Total tasks: {ntasks * mp.cpu_count()}")
        print(f"  Ideal runtime: {total_work_per_core}s")
        
        # Run sync benchmark
        try:
            sync_run_time = benchmark_worker(ntasks, sleep_time)
            sync_time = sync_run_time.get("processpool", None)
        except Exception as e:
            print(f"  Sync benchmark failed: {e}")
            sync_time = None
        
        # Run async benchmark
        try:
            async_run_time = asyncio.run(benchmark_async_worker(ntasks, sleep_time))
            async_time = async_run_time.get("processpool", None)
        except Exception as e:
            print(f"  Async benchmark failed: {e}")
            async_time = None
        
        result = {
            'ntasks_per_core': ntasks,
            'sleep_time': sleep_time,
            'total_work_per_core': total_work_per_core,
            'ideal_runtime': total_work_per_core,
            'total_tasks': ntasks * mp.cpu_count(),
            'ncores': mp.cpu_count(),
            'sync_runtime': sync_time,
            'async_runtime': async_time,
            'sync_efficiency': total_work_per_core / sync_time if sync_time else None,
            'async_efficiency': total_work_per_core / async_time if async_time else None,
            'sync_overhead': sync_time - total_work_per_core if sync_time else None,
            'async_overhead': async_time - total_work_per_core if async_time else None,
        }
        
        results.append(result)
        
        if sync_time:
            print(f"  Sync runtime: {sync_time:.3f}s (efficiency: {result['sync_efficiency']:.2%})")
        if async_time:
            print(f"  Async runtime: {async_time:.3f}s (efficiency: {result['async_efficiency']:.2%})")
    
    # Save results to CSV
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print("\n" + "=" * 60)
    print(f"Results saved to: {output_path.absolute()}")
    
    # Also save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_work_per_core': total_work_per_core,
        'ncores': mp.cpu_count(),
        'hostname': socket.gethostname(),
        'nconfigurations': len(configurations)
    }
    
    metadata_file = output_path.with_suffix('.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_file.absolute()}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark worker performance')
    parser.add_argument('--ntasks-per-core', type=int, default=None, 
                       help='Number of tasks per CPU core (for single run)')
    parser.add_argument('--sleep-time', type=float, default=None,
                       help='Sleep time for each task in seconds (for single run)')
    parser.add_argument('--strong-scaling', action='store_true',
                       help='Run strong scaling experiment')
    parser.add_argument('--total-work', type=float, default=10.0,
                       help='Total work per core for strong scaling (default: 10.0s)')
    parser.add_argument('--output', type=str, default='strong_scaling_results.csv',
                       help='Output CSV file for strong scaling results')

    args = parser.parse_args()

    if args.strong_scaling:
        # Run strong scaling experiment
        run_strong_scaling_experiment(args.total_work, args.output)
    else:
        # Run single benchmark
        ntasks = args.ntasks_per_core or 1
        sleep = args.sleep_time or 1.0
        
        print(f"Running benchmark with:")
        print(f"  Tasks per core: {ntasks}")
        print(f"  Sleep time: {sleep}s")
        print(f"  Total tasks: {mp.cpu_count() * ntasks}")
        print("-" * 40)

        run_time = benchmark_worker(ntasks, sleep)
        async_run_time = asyncio.run(benchmark_async_worker(ntasks, sleep))

        print("\nBenchmark Results:")
        print("-" * 40)
        print("Sync Results:")
        for executor, time_taken in run_time.items():
            print(f"{executor:>15}: {time_taken:.3f} seconds")
        print("Async Results:")
        for executor, time_taken in async_run_time.items():
            print(f"async {executor:>10}: {time_taken:.3f} seconds")
        print("-" * 40)
