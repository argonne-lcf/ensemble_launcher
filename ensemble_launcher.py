import json
import os
import socket
import subprocess
import time
import copy
import sys
import platform


def unravel_index(flat_index, shape):
    unravel_result = []
    for dim in reversed(shape):
        flat_index, remainder = divmod(flat_index, dim)
        unravel_result.append(remainder)
    return tuple(reversed(unravel_result))

##NOTE: all the tasks need to be modified through ensemble object
class ensemble:
    def __init__(self,ensemble_name:str,ensemble_info:dict) -> None:
        self.__ensemble_name = ensemble_name
        self.__ensemble_info = ensemble_info
        self.__tasks = {}
        ##this is to make sure the task id is unique
        self.__bin_options_ordered = tuple(ensemble_info.get("bin_options").keys())
        ##this just fills the tasks list
        self.__tasks = self.__generate_ensemble()
        self.__ensemble_state = {}
        self.__ensemble_state["ready_task_ids"] = {task["id"]:"ready" for task in self.__tasks.values()}
        self.__ensemble_state["running_task_ids"] = {}
        return None

    @property
    def ensemble_name(self):
        return self.__ensemble_name
    
    def __generate_ensemble(self) -> dict:
        tasks = []
        ensemble = self.__ensemble_info
        name = self.__ensemble_name
        # Generate tasks based on ensemble configuration
        ##assert some necessary vals
        assert "task_type" in ensemble.keys()
        assert "num_nodes" in ensemble.keys()
        assert "bin" in ensemble.keys()
        ###
        multipliers = []  # To hold keys that correspond to lists
        num_elements = 1  # Total number of tasks to generate
        task_dim = []  # Dimensions of the task grid
        task_template = {"ensemble_name":name}  # Template for the task (fixed part)
        for key, value in ensemble.items():
            if key != "bin_options":
                task_template[key] = value
            else:
                task_template["bin_options"] = {}
        # Process each key-value pair in the ensemble
        for key, value in ensemble.get("bin_options",{}).items():
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
            task = copy.deepcopy(task_template)  # Start with the task template
            idx = unravel_index(i, task_dim)  # Get the indices for the current combination

            # Assign values from the ensemble based on the indices
            for j, key in zip(idx, multipliers):
                task["bin_options"][key] = ensemble["bin_options"][key][j]

            task = self.__set_defaults(task)
            # Append the generated task to the progress info
            tasks.append(task)

        return {task["id"]:task for task in tasks}
    
    def __generate_task_id(self, task:dict) -> str:
        bin_options_str = "-".join(f"{k}-{task['bin_options'][k]}" for k in self.__bin_options_ordered)
        ##NOTE: the string should always start with ensemble name
        unique_str = f"{task['ensemble_name']}-{bin_options_str}"
        return unique_str
    
    ###function sets the default values for a task
    def __set_defaults(self,task:dict) -> dict:
        ##some defaults
        if "status" not in task.keys():
            task["status"] = "ready"
                
        task["id"] = self.__generate_task_id(task)

        if "run_dir" not in task.keys():
            task["run_dir"] = os.getcwd()
                
        if "task_type_cmd" not in task.keys():
            task["task_type_cmd"] = ''
                
        if "task_type_options" not in task.keys():
            task["task_type_options"] = {}
                
        if "bin_options" not in task.keys():
            task["bin_options"] = {}
        
        if "num_cores_per_node" not in task.keys():
            task["num_cores_per_node"] = 1
        
        return task

    def get_task_ids(self):
        return list(self.__tasks.keys())
    
    def get_task_info(self,task_id:str):
        return self.__tasks[task_id]
    
    def get_finished_task_ids(self):
        tasks_ids = []
        for task_id,task in self.__tasks.items():
            if task["status"] == "finished":
                tasks_ids.append(task_id)
        return tasks_ids
    
    def get_failed_task_ids(self):
        tasks_ids = []
        for task_id,task in self.__tasks.items():
            if task["status"] == "failed":
                tasks_ids.append(task_id)
        return tasks_ids
    
    def get_running_task_ids(self):
        return list(self.__ensemble_state["running_task_ids"].keys())
    
    def get_ready_task_ids(self):
        return list(self.__ensemble_state["ready_task_ids"].keys())
    
    def update_task_status(self,task_id:str,status:str) -> None:
        if status == "running":
            self.__ensemble_state["running_task_ids"][task_id] = "running"
            self.__ensemble_state["ready_task_ids"].pop(task_id)
        elif status == "failed":
            retries = self.__tasks[task_id].get("retries",0)
            if retries < 1:
                ##retry the task
                self.__ensemble_state["ready_task_ids"][task_id] = "ready"
                self.__ensemble_state["running_task_ids"].pop(task_id)
                self.__tasks[task_id]["retries"] = retries + 1
            else:
                self.__tasks[task_id]["status"] = "failed"
                self.__ensemble_state["running_task_ids"].pop(task_id)
        elif status == "finished":
            self.__ensemble_state["running_task_ids"].pop(task_id)
        elif status == "deleted":
            self.__ensemble_state["running_task_ids"].pop(task_id,None)
            self.__ensemble_state["ready_task_ids"].pop(task_id,None)
            return None
        else:
            if status != "ready":
                raise ValueError(f"Invalid status {status}")
        self.__tasks[task_id]["status"] = status
        return None
    
    def update_task_info(self,task_id:str,info:dict) -> None:
        if "status" in info.keys():
            self.update_task_status(task_id,info["status"])
        self.__tasks[task_id].update(info)
        return None
    
    def update_host_nodes(self,task_id:str,hosts:list) -> None:
        self.__tasks[task_id]["task_type_options"]["--hosts"] = ",".join(hosts)
    
    def build_task_cmd(self,task_id:str) -> None:
        task_info = self.__tasks[task_id]
        task_info["cmd"] = f"{task_info['task_type_cmd']}"
        ##Here, key is the option and value is its option
        for key,value in task_info["task_type_options"].items():
            task_info["cmd"] += f" {key} {value}"
        task_info["cmd"] += f" {task_info['bin']}"
        for key,value in task_info["bin_options"].items():
            task_info["cmd"] += f" {key} {value}"
        return None

    def get_next_ready_task(self):
        if len(self.__ensemble_state["ready_task_ids"]) > 0:
            return self.__ensemble_state["ready_task_ids"][0]
        return None
    
    def update_ensemble(self,ensemble_info:dict) -> list:
        ##update the ensemble info
        self.__ensemble_info = ensemble_info
        ##regenerate the ensemble
        new_tasks = self.__generate_ensemble()
        old_tasks = self.__tasks
        self.__tasks = {}
        # Remove tasks that are already in old_tasks
        for task_id, task in new_tasks.items():
            if task_id in old_tasks:
                self.__tasks[task_id] = old_tasks.pop(task_id)
            else:
                self.__tasks[task_id] = task
        for old_task_id in old_tasks:
            self.update_task_status(old_task_id,"deleted")
        ##update the ready tasks
        for task_id,task in self.__tasks.items():
            if task["status"] == "ready":
                if task_id not in self.__ensemble_state["ready_task_ids"]:
                    self.__ensemble_state["ready_task_ids"][task_id] = "ready"
        return old_tasks
    
    def save_ensemble_status(self, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'{self.ensemble_name}.json'), 'w', encoding='utf-8') as f:
            json.dump({tid:{k:v for k,v in self.__tasks[tid].items() if k != "process"} for tid in self.__tasks.keys()}, f, ensure_ascii=False, indent=4)

class ensemble_launcher:
    def __init__(self,config_file:str) -> None:
        self.progress_info = {}
        self.progress_info["total_nodes"] = self.get_nodes()
        self.progress_info["free_nodes"] = self.get_nodes()
        self.progress_info["ncores_per_node"] = self.get_cores_per_node()
        self.progress_info["free_cores_per_node"] = {node:self.progress_info["ncores_per_node"] for node in self.progress_info["total_nodes"]}
        self.progress_info["busy_nodes"] = [] #this only has nodes with no cores free
        self.update_interval = None
        self.poll_interval = 60
        ##
        self.ensembles = {}
        ##
        self.start_time = time.perf_counter()
        self.last_update_time = time.perf_counter()
        self.config_file = config_file
        self.read_input_file()

        return None
    
    def read_input_file(self):
        with open(self.config_file, "r") as file:
            data = json.load(file)
            self.update_interval = data.get("update_interval",None)
            self.poll_interval = data.get("poll_interval",600)
            ensembles_info = data["ensembles"]
            for ensemble_name,ensemble_info in ensembles_info.items():
                self.ensembles[ensemble_name] = ensemble(ensemble_name,ensemble_info)
        return None


    def get_nodes(self) -> list:
        node_list = []
        node_file = os.getenv("PBS_NODEFILE")
        if node_file is not None and os.path.exists(node_file):
            with open(node_file, "r") as f:
                node_list = f.readlines()
                node_list = [node.strip() for node in node_list]
        else:
            node_list = [socket.gethostname()]
        return node_list
    
    def get_cores_per_node(self):
        if os.getenv("NCPUS") is not None:
            return int(os.getenv("NCPUS"))
        else:
            return os.cpu_count()
    
    def report_status(self):
        n_nodes = len(self.progress_info["total_nodes"])
        n_busy_nodes = len(self.progress_info["busy_nodes"])
        n_todo_tasks = 0
        n_running_tasks = 0
        for ensemble_name,ensemble in self.ensembles.items():
            n_todo_tasks += len(ensemble.get_ready_task_ids())
            n_running_tasks += len(ensemble.get_running_task_ids())
        
        print(f"Nodes fully occupied {n_busy_nodes}/{n_nodes}, Tasks ready: {n_todo_tasks}, Tasks running: {n_running_tasks}")

    
    def assign_task_nodes(self,task_id:str,ensemble:ensemble) -> list:
        assigned_nodes = []
        task = ensemble.get_task_info(task_id)
        for j in range(len(self.progress_info["free_nodes"])):
            if len(assigned_nodes) == task["num_nodes"]:
                break
            if self.progress_info["free_cores_per_node"][self.progress_info["free_nodes"][j]] >= task["num_cores_per_node"]\
                and self.progress_info["free_nodes"][j] not in self.progress_info["busy_nodes"]:
                node = self.progress_info["free_nodes"][j]
                self.progress_info["free_cores_per_node"][node] -= task["num_cores_per_node"]
                assigned_nodes.append(node)
                if self.progress_info["free_cores_per_node"][node] == 0:
                    self.progress_info["free_nodes"].remove(node)
                    self.progress_info["busy_nodes"].append(node)

        if len(assigned_nodes) < task["num_nodes"]:
            for node in assigned_nodes:
                self.progress_info["free_cores_per_node"][node] += task["num_cores_per_node"]
                if node in self.progress_info["busy_nodes"]:
                    self.progress_info["busy_nodes"].remove(node)
                    self.progress_info["free_nodes"].append(node)
            assigned_nodes = []
            
        return assigned_nodes

    def free_task_nodes(self,task_id:str,ensemble:ensemble) -> None:
        task = ensemble.get_task_info(task_id)
        self.free_task_nodes_base(task)
        return
    
    def free_task_nodes_base(self,task_info:dict) -> None:
        for node in task_info["assigned_nodes"]:
            self.progress_info["free_cores_per_node"][node] += task_info["num_cores_per_node"]
            if node in self.progress_info["busy_nodes"]:
                self.progress_info["busy_nodes"].remove(node)
                self.progress_info["free_nodes"].append(node)
        return

    
    def launch_task(self, task_id:str,ensemble:ensemble):
        ##check if run dir exists
        task_info = ensemble.get_task_info(task_id)
        os.makedirs(task_info["run_dir"],exist_ok=True)
        p = subprocess.Popen(task_info["cmd"],
                             executable="/bin/bash",
                             shell=True,
                             stdout=open(os.path.join(task_info["run_dir"],f'job-{task_info["id"]}.out'),'wb'),
                             stderr=subprocess.STDOUT,
                             stdin=subprocess.DEVNULL,
                             cwd=task_info.get("run_dir",os.getcwd()),
                             env=os.environ.copy(),)
        
        return p

    def launch_ready_tasks(self,ensemble:ensemble) -> None:
        for task_id in ensemble.get_ready_task_ids():
            assigned_nodes = self.assign_task_nodes(task_id,ensemble)
            if len(assigned_nodes) == 0:
                if len(self.progress_info["free_nodes"]) == 0:
                    break
                else:
                    continue
            task_info = ensemble.get_task_info(task_id)
            if task_info["task_type"] == "mpi":
                ensemble.update_host_nodes(task_id,assigned_nodes)
            ensemble.build_task_cmd(task_id)
            p = self.launch_task(task_id,ensemble)
            ensemble.update_task_info(task_id,{"process":p,
                                                "assigned_nodes":assigned_nodes,
                                                "start_time":time.perf_counter(),
                                                "status":"running"})
        return None
    
    def poll_running_tasks(self) -> None:
        for ensemble_name,ensemble in self.ensembles.items():
            task_ids = ensemble.get_running_task_ids()
            for task_id in task_ids:
                task = ensemble.get_task_info(task_id)
                popen_proc = task["process"]
                if popen_proc.poll() is not None:
                    if popen_proc.returncode == 0:
                        status = "finished"
                    else:
                        status = "failed"
                    self.free_task_nodes(task_id,ensemble)
                    ensemble.update_task_info(task_id,{"end_time":time.perf_counter(),
                                                       "status":status,
                                                       "process":None,
                                                       "assigned_nodes":[]})
                    
        return None
    
    def get_pending_tasks(self):
        pending_tasks = []
        for ensemble_name,ensemble in self.ensembles.items():
            pending_tasks = pending_tasks + ensemble.get_ready_task_ids() + ensemble.get_running_task_ids()
        return pending_tasks
    
    def delete_tasks(self,task_infos:dict) -> None:
        for task in task_infos.values():
            if task["status"] == "running":
                task["process"].terminate()
                task["process"].wait()
                self.free_task_nodes_base(task)
    
    def update_ensembles(self)-> dict:
        print("Updating ensembles")
        deleted_tasks = {}
        with open(self.config_file, "r") as file:
            data = json.load(file)
            ensemble_infos = data["ensembles"]
            for ensemble_name,ensemble_info in ensemble_infos.items():
                deleted_tasks.update(self.ensembles[ensemble_name].update_ensemble(ensemble_info))
        return deleted_tasks

    def run_tasks(self) -> None:
        while True:
            for ensemble_name,ensemble in self.ensembles.items():
                self.launch_ready_tasks(ensemble)
            self.poll_running_tasks()
            self.report_status()
            if self.update_interval is not None:
                if time.perf_counter() - self.last_update_time > self.update_interval:
                    deleted_tasks = self.update_ensembles()
                    self.delete_tasks(deleted_tasks)
                    self.last_update_time = time.perf_counter()
            time.sleep(self.poll_interval)
            if len(self.get_pending_tasks()) == 0:
                for ensemble_name,ensemble in self.ensembles.items():
                    ensemble.save_ensemble_status()
                break

        return None

    
