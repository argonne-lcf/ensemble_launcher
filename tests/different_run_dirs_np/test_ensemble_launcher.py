import time
import sys
import socket
sys.path.append('../..')
from ensemble_launcher import ensemble_launcher

if __name__ == '__main__':
    el = ensemble_launcher("config.json")
    start_time = time.perf_counter()
    print(f'Launching node is {socket.gethostname()}')
    total_poll_time = el.run_tasks()
    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    print(f"{total_run_time=}")