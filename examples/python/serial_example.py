from time import sleep
import argparse
import socket
import os
import psutil

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
    A simple Python script which runs in serial and sleeps for a given time.
    In debug mode, it will print information about the node, CPU and GPU it is bound to.
    """
    # Parse command line arguments
    args = parse_args()

    # Print debug information
    if args.debug:
        hostname = socket.gethostname()
        p = psutil.Process()
        cpu_affinity = p.cpu_affinity()
        gpu_affinity = os.getenv("ZE_AFFINITY_MASK") or os.getenv("CUDA_VISIBLE_DEVICES")
        print(f"Hello from node {hostname}, CPU cores {cpu_affinity} and GPU {gpu_affinity}", flush=True)
        
    # Sleep for the given time
    sleep(args.sleep_time)

if __name__ == "__main__":
    main()