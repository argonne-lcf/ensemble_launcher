import json
import os
import socket
import subprocess
import time
import sys


def unravel_index(flat_index, shape):
    unravel_result = []
    for dim in reversed(shape):
        flat_index, remainder = divmod(flat_index, dim)
        unravel_result.append(remainder)
    return tuple(reversed(unravel_result))

class ensemble_launcher:
    def __init__(self,config_file:str) -> None:
        self.__progress_info = {}
        self.__progress_info["total_nodes"] = self.get_nodes()
        self.__progress_info["free_nodes"] = self.get_nodes()
        self.__progress_info["busy_nodes"] = []
        self.__progress_info["running_tasks"] = []
        self.__progress_info["finished_tasks"] = []

        ##
        self.__start_time = time.perf_counter()
        self.__last_update_time = time.perf_counter()
        self.__config_file = config_file
        self.read_input_file()
        return None

    def read_input_file(self):
        with open(self.__config_file, "r") as file:
            data = json.load(file)
            self.__update_interval = data.get("update_interval",None)
            self.__poll_interval = data.get("poll_interval",600)
            self.__progress_info["ensemble_names"],self.__progress_info["tasks"]=\
                self.generate_ensembles(data["ensembles"])
    
    def update_ensembles(self):
        with open(self.__config_file, "r") as file:
            data = json.load(file)
            ensembles = data["ensembles"]
            updated_ensembles = {k:v for k,v in ensembles.items() if k not in self.__progress_info["ensemble_names"]}
            new_names,new_tasks = self.generate_ensembles(updated_ensembles)
            self.__progress_info["ensemble_names"].extend(new_names)
            self.__progress_info["tasks"].extend(new_tasks)

    def generate_ensembles(self, ensembles):
        ensemble_names = []
        tasks = []
        # Generate tasks based on ensemble configuration
        for name,ensemble in ensembles.items():
            multipliers = []  # To hold keys that correspond to lists
            num_elements = 1  # Total number of tasks to generate
            task_dim = []  # Dimensions of the task grid
            ensemble_names.append(name)
            task_template = {"ensemble_name":name}  # Template for the task (fixed part)
            for key, value in ensemble.items():
                if key != "bin_options":
                    task_template[key] = value
                else:
                    task_template["bin_options"] = {}
            # Process each key-value pair in the ensemble
            for key, value in ensemble["bin_options"].items():
                if isinstance(value, list):
                    # If value is a list, it's a multiplier
                    multipliers.append(key)
                    num_elements *= len(value)
                    task_dim.append(len(value))
                else:
                    # If value is not a list, it's part of the template
                    task_template["bin_options"][key] = value
            
            task_dim = tuple(task_dim)  # Convert to tuple for easy unpacking

            # Iterate over all combinations of task parameters
            for i in range(num_elements):
                task = {k: v for k, v in task_template.items()}  # Start with the task template
                idx = unravel_index(i, task_dim)  # Get the indices for the current combination

                # Assign values from the ensemble based on the indices
                for j, key in zip(idx, multipliers):
                    print(j,key)
                    task["bin_options"][key] = ensemble["bin_options"][key][j]

                if "status" not in task.keys():
                    task["status"] = "ready"
                # Append the generated task to the progress info
                tasks.append(task)
        return ensemble_names,tasks

    def push_task(self,task_info):
        self.__progress_info["tasks"].append(task_info)

    def get_nodes(self) -> list:
        node_list = []
        node_file = os.getenv("PBS_NODEFILE")
        if node_file is not None:
            with open(node_file,"r") as f:
                node_list = f.readlines()
                node_list = [node.split("\n")[0] for node in node_list]
        return node_list
    
    def build_task_cmd(self,task_info) -> None:
        task_info["cmd"] = f"{task_info.get('task_type_cmd','')}"
        ##Here, key is the option and value is its option
        for key,value in task_info.get("task_type_options",{}).items():
            task_info["cmd"] += f" {key} {value}"
        task_info["cmd"] += f" {task_info['bin']}"
        for key,value in task_info.get("bin_options",{}).items():
            task_info["cmd"] += f" {key} {value}"

    def launch_task(self, task_info:dict, assigned_nodes:list):
        # print(f"Launching task {task_info['task_id']} on nodes {assigned_nodes}")
    
        if task_info["task_type"].lower()=="mpi":
            hosts_str = ",".join(assigned_nodes)
            task_info["task_type_options"]["--hosts"] = hosts_str

        self.build_task_cmd(task_info)
        p = subprocess.Popen(task_info["cmd"],
                             executable="/bin/bash",
                             shell=True,
                             stdout=open(os.path.join(task_info.get("run_dir",os.getcwd()),'job.out'),'wb'),
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             cwd=task_info.get("run_dir",os.getcwd()),
                             env=os.environ.copy(),)
        
        return p
    
    def report_status(self):
    
        n_nodes = len(self.__progress_info["total_nodes"])
        n_busy_nodes = len(self.__progress_info["busy_nodes"])
        n_todo_tasks = len(self.__progress_info["tasks"])
        n_running_tasks = len(self.__progress_info["running_tasks"])
        
        print(f"Nodes occupied {n_busy_nodes}/{n_nodes}, Tasks ready: {n_todo_tasks}, Tasks running: {n_running_tasks}")
    
    def launch_ready_tasks(self):
        # Launch ready tasks
        for tid,task in enumerate(self.__progress_info["tasks"]):
            if task["status"] != "finished" and task["num_nodes"] <= len(self.__progress_info["free_nodes"]):        
                assigned_nodes = []
                for i in range(task["num_nodes"]):
                    assigned_nodes.append(self.__progress_info["free_nodes"].pop(0))
                task["process"] = self.launch_task(task, assigned_nodes=assigned_nodes)
                task['start_time'] = time.perf_counter()
                task['status'] = "running"
                task["assigned_nodes"] = assigned_nodes
                self.__progress_info["running_tasks"].append(tid)
                self.report_status()
                
            # If launcher has run out of free nodes, return so polling can free nodes
            if len(self.__progress_info["free_nodes"]) == 0:
                break
    
    def poll_running_tasks(self):
        tasks = self.__progress_info["tasks"]
        for tid in self.__progress_info["running_tasks"]:
            popen_proc = tasks[tid]["process"]
            if popen_proc.poll() is not None: 
                tasks[tid]["end_time"] = time.perf_counter()
                if popen_proc.returncode == 0:
                    tasks[tid]["status"] = "finished"
                    self.__progress_info["finished_tasks"].append(tid)
                else:
                    tasks[tid]["status"] = "failed"

                self.__progress_info["running_tasks"].remove(tid)
                self.__progress_info["free_nodes"] += tasks[tid]["assigned_nodes"]
                
                print(f"Task {tid} returned in {tasks[tid]['end_time'] - tasks[tid]['start_time']} seconds with status {tasks[tid]['status']}")
                self.report_status()
        return
    

    def run_tasks(self):
        total_poll_time = 0.0
        self.report_status()
        while len(self.__progress_info["finished_tasks"]) < len(self.__progress_info["tasks"]):
            # Launch tasks ready to run
            if len(self.__progress_info["free_nodes"]) > 0:
                self.launch_ready_tasks()
            self.poll_running_tasks()
            time.sleep(self.__poll_interval)
            total_poll_time += self.__poll_interval
            ##update tasks

            if self.__update_interval is not None \
                and time.perf_counter() - self.__last_update_time > self.__update_interval:
                self.update_ensembles()
                self.__last_update_time = time.perf_counter()
        return total_poll_time
    
    def save_task_status(self,out_dir="outputs"):
        tasks = self.__progress_info["tasks"]
        for state in ['finished', 'failed', 'ready']:
            state_tasks = [task for task in tasks if tasks[task]['status'] == state]
            print(f"Tasks {state}: {len(state_tasks)} tasks")
        with open(os.getcwd()+f'/{out_dir}/tasks.json', 'w', encoding='utf-8') as f:
            json.dump(tasks, f, ensure_ascii=False, indent=4)
    

if __name__ == '__main__':
    el = ensemble_launcher("config.json")
    start_time = time.perf_counter()
    print(f'Launching node is {socket.gethostname()}')
    total_poll_time = el.run_tasks()
    el.save_task_status()
    end_time = time.perf_counter()
    total_run_time = end_time - start_time
    print(f"{total_run_time=}")

    
