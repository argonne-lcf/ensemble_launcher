from plot_performance import *
import os
import json
import matplotlib.pyplot as plt
import scienceplots
from glob import glob
import numpy as np
import scipy

plt.style.use(['science', 'no-latex'])


# Define the range of nodes
nodes = [1, 2, 4, 8, 16, 32]

# Collect time-to-completion data
time_to_completion = []

for node in nodes:
    folder = f"{node}node/outputs/"
    paths = glob(folder+"log_*.txt")
    temp_time = []
    for log_file_path in paths:
        if os.path.exists(log_file_path):
            data = parse_log_file(log_file_path)
            timestamps = data.get("timestamps")
            finished_tasks = data.get("finished_jobs")
            all_tasks = data.get("all_tasks")
            if(finished_tasks[-1] == all_tasks[-1]):
                temp_time.append(timestamps[-1])  # Last timestamp value
        
    if len(temp_time) == 0:
        time_to_completion.append(np.nan)
    else:
        time_to_completion.append(np.mean(np.array(temp_time)))
    


# Plot the data
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(nodes, time_to_completion, marker='o', label="Time to Completion", color="blue")
ax.plot(nodes, [ time_to_completion[0] / 2**i for i in range(len(nodes))], linestyle='--', color='red', label="Ideal Speedup")
ax.set_xlabel("Number of Nodes", fontsize=16)
ax.set_ylabel("Time to Completion (s)", fontsize=16)
ax.set_title("Strong Scaling Performance", fontsize=18)
ax.grid(True)
ax.legend(fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
fig.tight_layout()

# Save the plot
output_dir = os.getcwd()
fig.savefig(os.path.join(output_dir, "strong_scaling_performance.png"))
plt.close(fig)


# Plot number of jobs finished vs time for all nodes (left) and running jobs vs time (right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
ntimes = 10000

for node in nodes:
    folder = f"{node}node/outputs/"
    time_range = ()
    paths = glob(folder+"log_*.txt")
    finished_jobs = np.zeros(ntimes)
    running_jobs = np.zeros(ntimes)
    count = 0
    for log_file_path in paths:
        if os.path.exists(log_file_path):
            data = parse_log_file(log_file_path)
            timestamps = np.array(data.get("timestamps"))
            finished_tasks = np.array(data.get("finished_jobs"))
            total_tasks = data.get("all_tasks")[-1]
            running_tasks = np.array(data.get("running_tasks"))
            if len(finished_tasks) == 0 or len(running_tasks) == 0:
                continue
            f_finished = scipy.interpolate.interp1d(timestamps, finished_tasks, fill_value="extrapolate")
            f_running = scipy.interpolate.interp1d(timestamps, running_tasks, fill_value="extrapolate")
            if len(time_range) == 0:
                time_range = (timestamps[0], timestamps[-1])
            finished_jobs += f_finished(np.linspace(time_range[0], time_range[1], ntimes))
            running_jobs += f_running(np.linspace(time_range[0], time_range[1], ntimes))
            count += 1
    if count == 0:
        continue
    else:
        finished_jobs /= count
        running_jobs /= count
        time_normalized = np.linspace(time_range[0], time_range[1], ntimes) / time_range[1]
        ax1.plot(time_normalized, finished_jobs * 100 / total_tasks, label=f"{node} Nodes", linewidth=2)
        ax2.plot(time_normalized, running_jobs, label=f"{node} Nodes", linewidth=2)

# Configure the left plot (Finished Jobs)
ax1.set_xlabel("Time/Total Time", fontsize=16)
ax1.set_ylabel("Finished Tasks (%)", fontsize=16)
ax1.set_title("Finished Jobs vs Time for Different Nodes", fontsize=18)
ax1.grid(True)
ax1.legend(fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

# Configure the right plot (Running Jobs)
ax2.set_xlabel("Time/Total Time", fontsize=16)
ax2.set_ylabel("Running Jobs", fontsize=16)
ax2.set_title("Running Jobs vs Time for Different Nodes", fontsize=18)
ax2.grid(True)
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()

# Save the plot
fig.savefig(os.path.join(output_dir, "jobs_vs_time_all_nodes.png"))
plt.close(fig)


# Plot CPU and GPU utilization for different nodes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

for node in nodes:
    folder = f"{node}node/outputs/"
    time_range = ()
    paths = glob(folder+"log_*.txt")
    cpu_utilization = np.zeros(ntimes)
    gpu_utilization = np.zeros(ntimes)
    count = 0
    
    for log_file_path in paths:
        if os.path.exists(log_file_path):
            data = parse_log_file(log_file_path)
            timestamps = data.get("timestamps", [])
            if len(time_range) == 0:
                time_range = (timestamps[0], timestamps[-1])
            
            free_cores = data.get("free_cores", [])
            total_cores = data.get("total_cores")
            free_gpus = data.get("free_gpus", [])
            total_gpus = data.get("total_gpus")
            
            if len(free_cores) > 0 and len(free_gpus) > 0:
                cpu_util = [(1 - free/total_cores)*100 for free in free_cores]
                gpu_util = [(1 - free/total_gpus)*100 for free in free_gpus]
                
                f_cpu = scipy.interpolate.interp1d(timestamps, cpu_util, fill_value="extrapolate")
                f_gpu = scipy.interpolate.interp1d(timestamps, gpu_util, fill_value="extrapolate")
                
                cpu_utilization += f_cpu(np.linspace(time_range[0], time_range[1], ntimes))
                gpu_utilization += f_gpu(np.linspace(time_range[0], time_range[1], ntimes))
                count += 1
    
    if count == 0:
        continue
    else:
        cpu_utilization /= count
        gpu_utilization /= count
        time_normalized = np.linspace(time_range[0], time_range[1], ntimes)/time_range[1]
        cpu_efficiency = np.trapz(cpu_utilization, time_normalized)
        gpu_efficiency = np.trapz(gpu_utilization, time_normalized)
        ax1.plot(time_normalized, cpu_utilization, label=f"{node} Nodes ({cpu_efficiency:.1f})", linewidth=2)
        ax2.plot(time_normalized, gpu_utilization, label=f"{node} Nodes ({gpu_efficiency:.1f})", linewidth=2)

ax1.set_xlabel("Time/Total Time", fontsize=16)
ax1.set_ylabel("CPU Utilization (%)", fontsize=16)
ax1.set_title("CPU Utilization vs Time", fontsize=18)
ax1.grid(True)
ax1.legend(fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=14)

ax2.set_xlabel("Time/Total Time", fontsize=16)
ax2.set_ylabel("GPU Utilization (%)", fontsize=16)
ax2.set_title("GPU Utilization vs Time", fontsize=18)
ax2.grid(True)
ax2.legend(fontsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)

fig.tight_layout()
fig.savefig(os.path.join(output_dir, "resource_utilization_all_nodes.png"))
plt.close(fig)