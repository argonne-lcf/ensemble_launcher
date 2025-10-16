import socket
import os
import psutil
import argparse
from time import sleep

import mpi4py.rc
mpi4py.rc.initialize = False
mpi4py.rc.finalize = False
from mpi4py import MPI

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    Args:
        sleep_time: Time to sleep in seconds
        debug: Debug mode
    Returns:
        args: Parsed arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep_time", type=int, default=10, help="Time to sleep in seconds")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    return parser.parse_args()

def main() -> None:
    """
    A simple MPI application which sleeps for a given time.
    In debug mode, it will print information about the node, CPU and GPU each rank is bound to.
    """
    # Initialize MPI
    if not MPI.Is_initialized():
        MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    rankl = int(os.getenv("PALS_LOCAL_RANKID"))

    # Parse command line arguments
    args = parse_args()

    # Print debug information
    if args.debug:
        hostname = socket.gethostname()
        p = psutil.Process()
        cpu_affinity = p.cpu_affinity()
        gpu_affinity = os.getenv("ZE_AFFINITY_MASK") or os.getenv("CUDA_VISIBLE_DEVICES")
        print(f"Hello from rank {rank}/{size} on node {hostname}, CPU cores {cpu_affinity} and GPU {gpu_affinity}", flush=True)

    # Sleep for the given time
    comm.Barrier()
    sleep(args.sleep_time)
    comm.Barrier()

    MPI.Finalize()

if __name__ == "__main__":
    main()