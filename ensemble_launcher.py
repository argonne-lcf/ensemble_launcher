import json
import os
import socket
import subprocess
import time

def launch_task(task, task_id, assigned_nodes):
    print(f"Launching task {task_id} on nodes {assigned_nodes}")

    num_nodes = len(assigned_nodes)
    ppn = task["ranks_per_node"]
    nranks = num_nodes*ppn

    task_cmd = task["cmd"]
    run_dir = task["run_dir"]

    hosts_str = ""
    for node in assigned_nodes:
        if node != assigned_nodes[0]:
            hosts_str += ","
        hosts_str += node

    os.makedirs(run_dir, exist_ok=True)

    cmd = f"mpiexec -n {nranks} --ppn {ppn} --hosts {hosts_str} {mpi_options} {task_cmd}"

    print(f"task {task_id}: {cmd}")
    p = subprocess.Popen(cmd,
                         executable="/bin/bash",
                         shell=True,
                         stdout=open(os.path.join(run_dir,'job.out'),'wb'),
                         stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL,
                         cwd=run_dir,
                         env=os.environ.copy(),)
    
    return p

def get_nodes():
    node_list = []
    node_file = os.getenv("PBS_NODEFILE")
    with open(node_file,"r") as f:
        node_list = f.readlines()
        node_list = [node.split("\n")[0] for node in node_list]
    return node_list

def set_application(app_type,kwargs=None):
    if app_type == "lammps":
        NX = 400
        executable = "$HOME/bin/lmp_aurora_kokkos"
        inputs = f"-in $HOME/bin/lj_lammps_template.in -k on g 1 -var nx {NX} -sf kk -pk kokkos neigh half neigh/qeq full newton on"

        NDEPTH = kwargs["NDEPTH"]
        NTHREADS = kwargs["NTHREADS"]
        mpi_options = f"--depth={NDEPTH} --cpu-bind depth --env OMP_NUM_THREADS={NTHREADS} --env OMP_PROC_BIND=spread --env OMP_PLACES=cores"
    else:
        executable = "sleep"
        inputs = "10"
        mpi_options = ""
    return executable, inputs, mpi_options

def make_task_list(executable, inputs,
                   num_nodes, ranks_per_node,
                   mpi_options="", gpu_affinity="",
                   out_dir="outputs"):
    tasks = {}
    for i in range(ntasks):
        tasks[str(i)] = { 'name':'lammps',
                          'cmd': f'{gpu_affinity} {executable} {inputs}',
                          'num_nodes': num_nodes,
                          'ranks_per_node': ranks_per_node,
                          'mpi_options': mpi_options,
                          'run_dir': os.getcwd()+f"/{out_dir}/{i}",
                          'status': 'ready'}
    return tasks

def report_status(progress_info):

    total_job_nodes = progress_info["total_job_nodes"]
    num_occupied_nodes = progress_info["total_job_nodes"] - len(progress_info["free_node_list"])
    ready_tasks_id = progress_info["ready_tasks_id"]
    running_tasks_id = progress_info["running_tasks_id"]
    
    print(f"Nodes occupied {num_occupied_nodes}/{total_job_nodes}, Tasks ready: {len(ready_tasks_id)}, Tasks running: {len(running_tasks_id)}")

def launch_ready_tasks(tasks, progress_info):

    #ready_tasks_id = progress_info["ready_tasks_id"]
    #running_tasks_id = progress_info["running_tasks_id"]
    #launched_procs = progress_info["launched_procs"]
    #free_node_list = progress_info["free_node_list"]
    
    # Launch ready tasks
    for tid in progress_info["ready_tasks_id"]:
        if tasks[tid]["num_nodes"] <= len(progress_info["free_node_list"]):
                
            assigned_nodes = progress_info["free_node_list"][0:tasks[tid]["num_nodes"]]
            p = launch_task(tasks[tid], tid, assigned_nodes=assigned_nodes)
            tasks[tid]['start_time'] = time.perf_counter()
            tasks[tid]['status'] = "running"
            progress_info["ready_tasks_id"].remove(tid)
            progress_info["running_tasks_id"].append(tid)
            progress_info["launched_procs"][tid] = {"process":p,
                                                    "assigned_nodes": assigned_nodes}
            for node in assigned_nodes:
                progress_info["free_node_list"].remove(node)
                    
            report_status(progress_info)
            
        # If launcher has run out of free nodes, return so polling can free nodes
        if len(progress_info["free_node_list"]) == 0:
            return tasks, progress_info
        
    return tasks, progress_info

def poll_running_tasks(tasks, progress_info):
    for tid in progress_info["running_tasks_id"]:
        popen_proc = progress_info["launched_procs"][tid]["process"]
        if popen_proc.poll() is not None:            
            tasks[tid]["end_time"] = time.perf_counter()
            if popen_proc.returncode == 0:
                tasks[tid]["status"] = "finished"
            else:
                tasks[tid]["status"] = "failed"
            progress_info["running_tasks_id"].remove(tid)
            assigned_nodes = progress_info["launched_procs"][tid]["assigned_nodes"]
            for node in assigned_nodes:
                progress_info["free_node_list"].append(node)
            progress_info["num_unfinished_tasks"] -= 1
            print(f"Task {tid} returned in {tasks[tid]['end_time'] - tasks[tid]['start_time']} seconds with status {tasks[tid]['status']}")
            report_status(progress_info)
    return tasks, progress_info


def run_tasks(tasks, free_node_list):

    poll_interval = 1
    total_poll_time = 0
    progress_info = {"ready_tasks_id": list(tasks.keys()),
                     "running_tasks_id": [],
                     "launched_procs": {},
                     "free_node_list": free_node_list,
                     "total_job_nodes": len(free_node_list),
                     "num_unfinished_tasks": len(tasks),
                     }
    
    report_status(progress_info)
    while progress_info["num_unfinished_tasks"] > 0:
        # Launch tasks ready to run
        if len(free_node_list) > 0:
            tasks, progress_info = launch_ready_tasks(tasks, progress_info)
            
        # Poll running tasks
        tasks, progress_info = poll_running_tasks(tasks, progress_info)
            
        time.sleep(poll_interval)
        total_poll_time += poll_interval
    return total_poll_time

def save_task_status(tasks, out_dir="outputs"):
    print(" ")
    for state in ['finished', 'failed', 'ready']:
        state_tasks = [task for task in tasks if tasks[task]['status'] == state]
        print(f"Tasks {state}: {len(state_tasks)} tasks")
    with open(os.getcwd()+f'/{out_dir}/tasks.json', 'w', encoding='utf-8') as f:
        json.dump(tasks, f, ensure_ascii=False, indent=4)
    
        
if __name__ == '__main__':

    start_time = time.perf_counter()
    
    # Get available nodes
    free_node_list = get_nodes()
    total_job_nodes = len(free_node_list)

    # Set the number of tasks
    tasks_per_node_factor = 2
    ntasks = total_job_nodes*tasks_per_node_factor

    # Set application and inputs
    gpu_affinity = "/soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh"
    executable, inputs, mpi_options = set_application("lammps", kwargs={"NDEPTH":8,
                                                           "NTHREADS":1})

    # Set per task resources
    num_nodes = 2
    ranks_per_node =  12

    print(f'Launching node is {socket.gethostname()}')
    # Make simple task list
    tasks = make_task_list(executable=executable,
                           inputs=inputs,
                           num_nodes=num_nodes,
                           ranks_per_node=ranks_per_node,
                           mpi_options=mpi_options,
                           gpu_affinity=gpu_affinity,
                           out_dir=f"outputs_{len(free_node_list)}")
    
    total_poll_time = run_tasks(tasks=tasks, free_node_list=free_node_list)

    save_task_status(tasks,
                     out_dir=f"outputs_{len(free_node_list)}")

    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    application_run_times = [tasks[task]["end_time"] - tasks[task]["start_time"] for task in tasks]
    mean_run_time = sum(application_run_times)/ntasks
    nslots = len(free_node_list)//num_nodes
    tasks_per_slot = ntasks//nslots
    fom = total_run_time - tasks_per_slot*mean_run_time
    
    print(f"{total_run_time=}")
    print(f"{mean_run_time=}")
    print(f"{total_poll_time=}")
    print(f"{nslots=}")
    print(f"{tasks_per_slot=}")
    print(f"{fom=}")
    
