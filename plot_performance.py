import os
import re
import sys

import matplotlib.pyplot as plt
import scienceplots

# Use scienceplots for better aesthetics
plt.style.use(['science','no-latex'])

def parse_log_file(log_file_path):
    """Parse the log file and extract relevant data."""
    data = {
        "timestamps": [],
        "finished_jobs": [],
        "free_cores": [],
        "total_cores": None,
        "free_gpus": [],
        "total_gpus": None,
        "running_tasks": [],
        "failed_tasks": [],
        "todo_tasks": []
    }

    with open(log_file_path, 'r') as file:
        for line in file:
            # Extract timestamp and finished jobs
            match = re.search(r'(\d+\.\d+),.*Finished: (\d+)', line)
            if match:
                data["timestamps"].append(float(match.group(1)))
                data["finished_jobs"].append(int(match.group(2)))

            # Extract core utilization
            core_match = re.search(r'Free Cores: (\d+)/(\d+)', line)
            if core_match:
                data["free_cores"].append(int(core_match.group(1)))
                data["total_cores"] = int(core_match.group(2))
            
            #Extract gpu utilization
            # Extract GPU utilization
            gpu_match = re.search(r'Free GPUs: (\d+)/(\d+)', line)
            if gpu_match:
                if "free_gpus" not in data:
                    data["free_gpus"] = []
                data["free_gpus"].append(int(gpu_match.group(1)))
                data["total_gpus"] = int(gpu_match.group(2))

            # Extract number of running tasks
            running_match = re.search(r'Running: (\d+)', line)
            if running_match:
                if "running_tasks" not in data:
                    data["running_tasks"] = []
                data["running_tasks"].append(int(running_match.group(1)))

            # Extract number of failed tasks
            failed_match = re.search(r'Failed: (\d+)', line)
            if failed_match:
                if "failed_tasks" not in data:
                    data["failed_tasks"] = []
                data["failed_tasks"].append(int(failed_match.group(1)))

            # Extract number of tasks to do
            todo_match = re.search(r'ToDo: (\d+)', line)
            if todo_match:
                if "todo_tasks" not in data:
                    data["todo_tasks"] = []
                data["todo_tasks"].append(int(todo_match.group(1)))            
    data["timestamps"] = [timestamp - data["timestamps"][0] for timestamp in data["timestamps"]]
    return data

def plot_finished_jobs_vs_time(timestamps, finished_jobs, output_dir):
    """Plot the number of finished jobs vs time using axes handle."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timestamps, finished_jobs, label="Finished Jobs", color="blue")
    ax.set_xlabel("Time (s)", fontsize=16)  # Increased label size
    ax.set_ylabel("Number of Finished Jobs", fontsize=16)  # Increased label size
    ax.set_title("Finished Jobs vs Time", fontsize=18)  # Increased title size
    ax.grid(True)
    ax.legend(fontsize=14)  # Increased legend font size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "finished_jobs_vs_time.png"))
    plt.close(fig)

def plot_utilization_vs_time(timestamps, free_cores, total_cores, free_gpus, total_gpus, output_dir):
    """Plot percent utilization for both CPU and GPU vs time."""
    cpu_utilization = [(1 - free / total_cores) * 100 for free in free_cores]
    gpu_utilization = [(1 - free / total_gpus) * 100 for free in free_gpus]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(timestamps, cpu_utilization, label="CPU Utilization (%)", color="green")
    ax.plot(timestamps, gpu_utilization, label="GPU Utilization (%)", color="orange")
    ax.set_xlabel("Time (s)", fontsize=16)  # Increased label size
    ax.set_ylabel("Percent Utilization (%)", fontsize=16)  # Increased label size
    ax.set_title("CPU and GPU Utilization vs Time", fontsize=18)  # Increased title size
    ax.grid(True)
    ax.legend(fontsize=14)  # Increased legend font size
    ax.tick_params(axis='both', which='major', labelsize=14)  # Increased tick label size
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "utilization_vs_time.png"))
    plt.close(fig)

def main(outputs_dir):
    log_file_path = os.path.join(outputs_dir, "log.txt")
    if not os.path.exists(log_file_path):
        print(f"Log file not found at {log_file_path}")
        return

    # Parse the log file
    data = parse_log_file(log_file_path)

    # Extract relevant data
    timestamps = data["timestamps"]
    finished_jobs = data["finished_jobs"]
    free_cores = data["free_cores"]
    total_cores = data["total_cores"]
    free_gpus = data["free_gpus"]
    total_gpus = data["total_gpus"]

    # Generate plots
    plot_finished_jobs_vs_time(timestamps, finished_jobs, outputs_dir)
    plot_utilization_vs_time(timestamps, free_cores, total_cores, free_gpus, total_gpus, outputs_dir)
    print(f"Plots saved in {outputs_dir}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_performance.py <outputs_directory>")
    else:
        main(sys.argv[1])