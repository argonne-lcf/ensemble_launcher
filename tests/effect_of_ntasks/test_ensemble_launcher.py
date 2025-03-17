import time
import sys
import socket
sys.path.append('../..')
from ensemble_launcher import ensemble_launcher
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A script with command-line options.")
    parser.add_argument("--ntasks", "-i", type=str, help="Path to the input file")
    args = parser.parse_args()
    
    el = ensemble_launcher(f"config_{args.ntasks}.json")
    start_time = time.perf_counter()
    print(f'Launching node is {socket.gethostname()}')
    total_poll_time = el.run_tasks()
    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    print(f"{total_run_time=}")