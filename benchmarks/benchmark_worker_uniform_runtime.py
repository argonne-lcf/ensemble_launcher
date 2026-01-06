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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark worker performance')
    parser.add_argument('--ntasks-per-core', type=int, default=1, 
                       help='Number of tasks per CPU core (default: 10)')
    parser.add_argument('--sleep-time', type=float, default=1.0,
                       help='Sleep time for each task in seconds (default: 1.0)')

    args = parser.parse_args()

    print(f"Running benchmark with:")
    print(f"  Tasks per core: {args.ntasks_per_core}")
    print(f"  Sleep time: {args.sleep_time}s")
    print(f"  Total tasks: {mp.cpu_count() * args.ntasks_per_core}")
    print("-" * 40)

    run_time = benchmark_worker(args.ntasks_per_core, args.sleep_time)
    async_run_time = asyncio.run(benchmark_async_worker(args.ntasks_per_core, args.sleep_time))

    print("\nBenchmark Results:")
    print("-" * 40)
    print("Sync Results:")
    for executor, time_taken in run_time.items():
        print(f"{executor:>15}: {time_taken:.3f} seconds")
    print("Async Results:")
    for executor, time_taken in async_run_time.items():
        print(f"async {executor:>10}: {time_taken:.3f} seconds")
    print("-" * 40)
