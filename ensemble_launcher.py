import json
import os
import socket
import subprocess
import time
import copy
import multiprocessing as mp
import psutil
import sys
import platform
import resource


def unravel_index(flat_index, shape):
    unravel_result = []
    for dim in reversed(shape):
        flat_index, remainder = divmod(flat_index, dim)
        unravel_result.append(remainder)
    return tuple(reversed(unravel_result))

##NOTE: all the tasks need to be modified through ensemble object
class ensemble:
    def __init__(self,ensemble_name:str,ensemble_info:dict,nparallel:int=1) -> None:
        self.__ensemble_name = ensemble_name
        self.__ensemble_info = ensemble_info
        self.__tasks = {}
        ##this is to make sure the task id is unique
        self.__list_options = None
        ##this just fills the tasks list
        self.__tasks = self.__generate_ensemble()
        self.__ensemble_state = {}
        self.__ntasks = len(self.__tasks)
        self.__np = nparallel
        self.__ensemble_state = {}
        self.__local_tasks = {}##this is dictionary of type pid: local_tasks.
        self.__split_tasks() 
        self.__initialize_ensemble_state()
        return None

    @property
    def ensemble_name(self):
        return self.__ensemble_name
    
    @property
    def ntasks(self):
        return self.__ntasks
    
    ##this function checks if ensemble config is correct
    def check_ensemble_info(self):
        ensemble = self.__ensemble_info
        ##assert some necessary vals
        assert "num_nodes" in ensemble.keys()
        assert "launcher" in ensemble.keys()
        assert "relation" in ensemble.keys()
        assert "cmd" in ensemble.keys()

    def __generate_ensemble(self) -> dict:
        """check ensemble config
        """
        self.check_ensemble_info()
        ensemble = self.__ensemble_info
        relation = ensemble["relation"]
        """this is one-to-one relationship between all the lists
        """
        if relation == "one-to-one":
            list_options = []
            non_list_options = []
            ntasks = None
            for key,value in ensemble.items():
                if isinstance(value,list):
                    list_options.append(key)
                    if ntasks is None:
                        ntasks = len(value)
                    else:
                        if len(ensemble[key]) != ntasks:
                            raise ValueError(f"Invalid option length for {key}")
                else:
                    non_list_options.append(key)
            
            self.__list_options = tuple(list_options)
            tasks = []
            for i in range(ntasks):
                task = {"ensemble_name":self.__ensemble_name}
                for opt in non_list_options:
                    task[opt] = ensemble[opt]
                for opt in list_options:
                    task[opt] = ensemble[opt][i]
                tasks.append(self.__set_defaults(task))
        elif relation == "many-to-many":
            list_options = []
            non_list_options = []
            ntasks = 1
            dim = []
            for key,value in ensemble.items():
                if isinstance(value,list):
                    list_options.append(key)
                    ntasks *= len(value)
                    dim.append(len(value))
                else:
                    non_list_options.append(key)
            self.__list_options = tuple(list_options)
            tasks = []
            for tid in range(ntasks):
                task = {"ensemble_name":self.__ensemble_name}
                loc = unravel_index(tid,dim)
                for id,opt in enumerate(list_options):
                    task[opt] = ensemble[opt][loc[id]]
                for opt in non_list_options:
                    task[opt] = ensemble[opt]
                tasks.append(self.__set_defaults(task))
        else:
            raise ValueError(f"Unknown relation {relation}")
        
        return {task["id"]:task for task in tasks}

    def __generate_task_id(self, task:dict) -> str:
        assert self.__list_options is not None
        bin_options_str = "-".join(f"{k}-{task[k]}" for k in self.__list_options)
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
        
        if "num_processes_per_node" not in task.keys():
            task["num_processes_per_node"] = 1
        
        return task

    def __split_tasks(self)->None:
        local_tasks = {}
        ntasks = [self.__ntasks//self.__np for pid in range(self.__np)]
        ntasks[-1] += sum(ntasks)%self.__np
        task_ids = self.get_task_ids()
        for pid in range(self.__np):
            self.__local_tasks[pid] = task_ids[sum(ntasks[:pid]):sum(ntasks[:pid])+ntasks[pid]]
            for task_id in self.__local_tasks[pid]:
                self.__tasks[task_id]["pid"] = pid
        return None

    def __initialize_ensemble_state(self) -> None:
        self.__ensemble_state = {}
        for pid in range(self.__np):
            self.__ensemble_state[pid] = {
                "ready_task_ids": {task_id: "ready" for task_id in self.__local_tasks[pid] if self.__tasks[task_id]["status"] == "ready"},
                "running_task_ids": {task_id: "running" for task_id in self.__local_tasks[pid] if self.__tasks[task_id]["status"] == "running"}
            }
        return None
 
    ##this is when ensemble itself decides the mapping
    def set_np(self,np:int)->None:
        self.__np = np
        self.__split_tasks()
        self.__initialize_ensemble_state()
        return None
    
    ###this when user wants to decide the mappings
    def set_local_tasks(self,local_tasks:dict)->None:
        self.__local_tasks = local_tasks
        self.__np = len(self.__local_tasks)
        for pid,task_ids in self.__local_tasks.itesm():
            for task_id in task_ids:
                self.__tasks[task_id]["pid"] = pid
        self.__initialize_ensemble_state()
        return None

    def get_task_ids(self,pid:int=None)->list:
        if pid is not None:
            return self.__local_tasks[pid]
        else:
            return list(self.__tasks.keys())
    
    def get_task_info(self,task_id:str)->dict:
        return self.__tasks[task_id]
    
    def get_finished_task_ids(self, pid: int = 0)->list:
        task_ids = []
        for task_id in self.__local_tasks[pid]:
            if self.__tasks[task_id]["status"] == "finished":
                task_ids.append(task_id)
        return task_ids
    
    def get_failed_task_ids(self, pid: int = 0)->list:
        task_ids = []
        if pid is not None:
            for task_id in self.__local_tasks[pid]:
                if self.__tasks[task_id]["status"] == "failed":
                    task_ids.append(task_id)
        return task_ids
    
    def get_running_task_ids(self,pid:int=0)->list:
        return list(self.__ensemble_state[pid]["running_task_ids"].keys())
    
    def get_ready_task_ids(self,pid:int=0)->list:
        return list(self.__ensemble_state[pid]["ready_task_ids"].keys())
    
    def update_task_status(self, task_id: str, status: str, pid: int = 0) -> None:
        if status == "running":
            assert self.__tasks[task_id]["status"] == "ready"
            self.__ensemble_state[pid]["running_task_ids"][task_id] = "running"
            self.__ensemble_state[pid]["ready_task_ids"].pop(task_id)
        elif status == "failed":
            assert self.__tasks[task_id]["status"] == "running"
            retries = self.__tasks[task_id].get("retries", 0)
            if retries < 1:
                # Retry the task
                self.__ensemble_state[pid]["ready_task_ids"][task_id] = "ready"
                self.__ensemble_state[pid]["running_task_ids"].pop(task_id)
                self.__tasks[task_id]["retries"] = retries + 1
                self.__tasks[task_id]["status"] = "ready"
                return None
            else:
                self.__tasks[task_id]["status"] = "failed"
                self.__ensemble_state[pid]["running_task_ids"].pop(task_id)
        elif status == "finished":
            assert self.__tasks[task_id]["status"] == "running"
            self.__ensemble_state[pid]["running_task_ids"].pop(task_id)
        elif status == "deleted":
            self.__ensemble_state[pid]["running_task_ids"].pop(task_id, None)
            self.__ensemble_state[pid]["ready_task_ids"].pop(task_id, None)
            return None
        else:
            if status != "ready":
                raise ValueError(f"Invalid status {status}")
        self.__tasks[task_id]["status"] = status
        return None
    
    def update_task_info(self,task_id:str,info:dict,pid:int=0) -> None:
        if "status" in info.keys():
            self.update_task_status(task_id,info["status"],pid=pid)
            info.pop("status")
        self.__tasks[task_id].update(info)
        return None
    
    def update_host_nodes(self,task_id:str,hosts:list) -> None:
        self.__tasks[task_id]["task_type_options"]["--hosts"] = ",".join(hosts)
    
    def build_task_cmd(self,task_id:str) -> None:
        task_info = self.__tasks[task_id]
        open_braces = [i for i, char in enumerate(task_info["cmd"]) if char == "{"]
        close_braces = [i for i, char in enumerate(task_info["cmd"]) if char == "}"]
        placeholders = [task_info["cmd"][open_braces[i] + 1:close_braces[i]] for i in range(len(open_braces))]
        for opt in placeholders:
            task_info["cmd"] = task_info["cmd"].format(**{key: task_info[key] for key in placeholders})
        if task_info["launcher"] == "mpi":
            if "mpirun -np " not in task_info["cmd"]:
                hosts = ",".join(task_info["assigned_nodes"])
                if "alcfwl" in hosts:
                    task_info["cmd"] = f"mpirun -np {task_info['num_nodes']*task_info['num_processes_per_node']} " + task_info["cmd"]
                else:
                    task_info["cmd"] = f"mpirun -np {task_info['num_nodes']*task_info['num_processes_per_node']} -ppn {task_info['num_processes_per_node']} --hosts {hosts} " + task_info["cmd"]
                
        else:
            if task_info["launcher"] != "bash":
                raise ValueError(f"Unknown launcher {task_info['launcher']}")
        return None

    def get_next_ready_task(self,pid:int=0):
        if len(self.__ensemble_state[pid]["ready_task_ids"]) > 0:
            return self.__ensemble_state[pid]["ready_task_ids"][0]
        return None
    
    def update_ensemble(self,ensemble_info:dict,rebalance_tasks:bool=False) -> list:
        ##update the ensemble info
        self.__ensemble_info = ensemble_info
        ##regenerate the ensemble
        updated_tasks = self.__generate_ensemble()
        old_tasks = self.__tasks
        old_local_tasks = self.__local_tasks
        self.__tasks = {}
        new_tasks = []
        # Remove tasks that are already in old_tasks
        for task_id, task in updated_tasks.items():
            if task_id in old_tasks:
                self.__tasks[task_id] = old_tasks.pop(task_id)
            else:
                self.__tasks[task_id] = task
                new_tasks.append(task_id)
        self.__ntasks = len(self.__tasks.keys())
        
        
        ##update ready tasks
        ##delete the old tasks
        for old_task_id,old_task in old_tasks.items():
            self.update_task_status(old_task_id,"deleted",pid=old_task["pid"])
        
        if rebalance_tasks:
            ##this will equally split the tasks among __np
            self.__split_tasks()
            self.__initialize_ensemble_state()
        else:
            ##if not, add new tasks to the pid=0
            for task_id in new_tasks:
                self.__ensemble_state[0]["ready_task_ids"][task_id] = "ready"

        return old_tasks
    
    def save_ensemble_status(self, out_dir="outputs"):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'{self.ensemble_name}.json'), 'w', encoding='utf-8') as f:
            json.dump({tid:{k:v for k,v in self.__tasks[tid].items() if k != "process"} for tid in self.__tasks.keys()}, f, ensure_ascii=False, indent=4)

class ensemble_launcher:
    def __init__(self,config_file:str,ncores_per_node:int=None) -> None:
        self.update_interval = None ##how often to update the ensembles in secs
        self.poll_interval = 60 ##how often to poll the running tasks in secs
        self.n_parallel = None ##number of parallel lacunchers in int
        self.pids_per_ensemble = {} ##pids of an ensemble
        ##
        self.ensembles = {}
        self.start_time = time.perf_counter()
        self.last_update_time = time.perf_counter()
        self.config_file = config_file
        self.read_input_file()
        self.pids_per_ensemble = {en:[0] for en in self.ensembles.keys()}
        if self.n_parallel > 1:
            self.distribute_procs()
        ##
        self.progress_info = {}
        self.progress_info["total_nodes"] = self.get_nodes()
        self.progress_info["my_nodes"] = self.split_nodes() ##this will be list of lists. Access should be same as a dict
        self.progress_info["my_free_nodes"] = self.progress_info["my_nodes"]
        self.progress_info["ncores_per_node"] = self.get_cores_per_node() if ncores_per_node is None else ncores_per_node
        self.progress_info["free_cores_per_node"] = {node:self.progress_info["ncores_per_node"] for node in self.progress_info["total_nodes"]}
        self.progress_info["my_busy_nodes"] = [[] for i in range(self.n_parallel)] #this only has nodes with no cores free
        ##
        return None
    
    def read_input_file(self):
        with open(self.config_file, "r") as file:
            data = json.load(file)
            self.update_interval = data.get("update_interval",None)
            self.poll_interval = data.get("poll_interval",600)
            self.n_parallel = data.get("n_parallel",1)
            self.logfile = data.get("logfile","log.txt")
            ensembles_info = data["ensembles"]
            for ensemble_name,ensemble_info in ensembles_info.items():
                self.ensembles[ensemble_name] = ensemble(ensemble_name,ensemble_info)
        return None
    
    ##pool all the tasks and split them among available processes
    def distribute_procs(self)->None:
        ntasks_per_proc = sum([e.ntasks for e in self.ensembles.values()])//self.n_parallel
        cum_tasks = 0
        for en,e in self.ensembles.items():
            start = cum_tasks//ntasks_per_proc
            end = (cum_tasks+e.ntasks)//ntasks_per_proc + 1
            if start == end:
                self.pids_per_ensemble[en] = [start]
            else:
                self.pids_per_ensemble[en] = [i for i in range(start,end+1)]
            self.pids_per_ensemble[en] = [i for i in self.pids_per_ensemble[en] if i < self.n_parallel]
            self.ensembles[en].set_np(len(self.pids_per_ensemble[en]))
            cum_tasks+=e.ntasks
            e.save_ensemble_status()
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
    
    def split_nodes(self)->list:
        my_nodes = []
        if len(self.progress_info["total_nodes"])<self.n_parallel:
            raise ValueError("Total number of nodes < number of parallel task launchers! Please set nparallel = 1")
        nn = len(self.progress_info["total_nodes"])//self.n_parallel
        for i in range(self.n_parallel):
            my_nodes.append(self.progress_info["total_nodes"][i*nn:(i+1)*nn])
        my_nodes[-1].extend(self.progress_info["total_nodes"][(i+1)*nn:])
        return my_nodes
    
    def get_cores_per_node(self):
        if os.getenv("NCPUS") is not None:
            return int(os.getenv("NCPUS"))
        else:
            return os.cpu_count()
    
    def report_status(self,my_pid:int=0):
        n_nodes = len(self.progress_info["total_nodes"])
        n_busy_nodes = len(self.progress_info["my_busy_nodes"][my_pid])
        n_todo_tasks = 0
        n_running_tasks = 0
        n_fds = 0
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        for ensemble_name,ensemble in self.ensembles.items():
            if my_pid in self.pids_per_ensemble[ensemble_name]:
                running_task_ids = ensemble.get_running_task_ids(self.pids_per_ensemble[ensemble_name].index(my_pid))
                n_todo_tasks += len(ensemble.get_ready_task_ids(self.pids_per_ensemble[ensemble_name].index(my_pid)))
                n_running_tasks += len(ensemble.get_running_task_ids(self.pids_per_ensemble[ensemble_name].index(my_pid)))
                for task_id in running_task_ids:
                    task_info = ensemble.get_task_info(task_id)
                    n_fds += self.get_nfd(task_info.get("process",None))

        fname = os.path.join(os.getcwd(),"outputs",self.logfile)
        with open(fname,"a") as f:
            f.write(f"Proc {my_pid}: nfds {n_fds}/{soft_limit} Nodes fully occupied {n_busy_nodes}/{n_nodes}, Tasks ready: {n_todo_tasks}, Tasks running: {n_running_tasks}\n")

    
    def assign_task_nodes(self,task_id:str,ensemble:ensemble,my_pid:int=0) -> list:
        assigned_nodes = []
        task = ensemble.get_task_info(task_id)
        j = 0
        while True:
            if len(assigned_nodes) == task["num_nodes"] or \
               len(self.progress_info["my_free_nodes"][my_pid]) == 0 or \
               j > len(self.progress_info["my_free_nodes"][my_pid]):
                break
            if self.progress_info["free_cores_per_node"][self.progress_info["my_free_nodes"][my_pid][j]] >= task["num_processes_per_node"]\
                and self.progress_info["my_free_nodes"][my_pid][j] not in self.progress_info["my_busy_nodes"][my_pid]:
                node = self.progress_info["my_free_nodes"][my_pid][j]
                self.progress_info["free_cores_per_node"][node] -= task["num_processes_per_node"]
                assigned_nodes.append(node)
                j += 1
                if self.progress_info["free_cores_per_node"][node] == 0:
                    self.progress_info["my_free_nodes"][my_pid].remove(node)
                    self.progress_info["my_busy_nodes"][my_pid].append(node)
                    j -= 1

        if len(assigned_nodes) < task["num_nodes"]:
            for node in assigned_nodes:
                self.progress_info["free_cores_per_node"][node] += task["num_processes_per_node"]
                if node in self.progress_info["my_busy_nodes"][my_pid]:
                    self.progress_info["my_busy_nodes"][my_pid].remove(node)
                    self.progress_info["my_free_nodes"][my_pid].append(node)
            assigned_nodes = []
            
        return assigned_nodes

    def free_task_nodes(self,task_id:str,ensemble:ensemble,my_pid:int=0) -> None:
        task = ensemble.get_task_info(task_id)
        self.free_task_nodes_base(task,my_pid)
        return
    
    def free_task_nodes_base(self,task_info:dict,my_pid:int=0) -> None:
        for node in task_info["assigned_nodes"]:
            self.progress_info["free_cores_per_node"][node] += task_info["num_processes_per_node"]
            if node in self.progress_info["my_busy_nodes"][my_pid]:
                self.progress_info["my_busy_nodes"][my_pid].remove(node)
                self.progress_info["my_free_nodes"][my_pid].append(node)
        return

    
    def launch_task(self, task_id:str,ensemble:ensemble):
        ##check if run dir exists
        task_info = ensemble.get_task_info(task_id)
        os.makedirs(task_info["run_dir"],exist_ok=True)
        p = subprocess.Popen(task_info["cmd"],
                             executable="/bin/bash",
                             shell=True,
                             stdout=open(os.path.join(task_info["run_dir"],"log.txt"),"w"),
                             stderr=open(os.path.join(task_info["run_dir"],"err.txt"),"w"),
                             stdin=subprocess.DEVNULL,
                             cwd=task_info.get("run_dir",os.getcwd()),
                             env=os.environ.copy(),
                             close_fds = True)
        return p

    def get_nfd(self,process)->int:
        if process is None:
            return 0
        pid = process.pid
        fd_path = f'/proc/{pid}/fd'
        try:
            open_fds = os.listdir(fd_path)
            return len(open_fds)
        except:
            return 0

    def launch_ready_tasks(self,ensemble:ensemble,local_pid:int=0,my_pid:int=0) -> int:
        launched_tasks = 0
        for task_id in ensemble.get_ready_task_ids(pid=local_pid):
            assigned_nodes = self.assign_task_nodes(task_id,ensemble,my_pid=my_pid)
            if len(assigned_nodes) == 0:
                if len(self.progress_info["my_free_nodes"][my_pid]) == 0:
                    break
                else:
                    continue
            ensemble.update_task_info(task_id,{ "assigned_nodes":assigned_nodes},pid=local_pid)
            ensemble.build_task_cmd(task_id)
            p = self.launch_task(task_id,ensemble)
            launched_tasks += 1
            fname = os.path.join(os.getcwd(),"outputs",self.logfile)
            with open(fname,"a") as f:
                f.write(f"{ensemble.ensemble_name}:launched {launched_tasks} tasks!\n")
            ensemble.update_task_info(task_id,{"process":p,
                                                "start_time":time.perf_counter(),
                                                "status":"running"},pid=local_pid)
            self.report_status()
        return launched_tasks
    
    def poll_running_tasks(self,my_pid:int=0) -> None:
        for ensemble_name,ensemble in self.ensembles.items():
            for local_pid,pid in enumerate(self.pids_per_ensemble[ensemble_name]):
                if pid == my_pid:
                    task_ids = ensemble.get_running_task_ids(local_pid)
                    for task_id in task_ids:
                        task = ensemble.get_task_info(task_id)
                        popen_proc = task["process"]
                        if popen_proc.poll() is not None:
                            if popen_proc.returncode == 0:
                                status = "finished"
                            else:
                                status = "failed"
                            out,err = popen_proc.communicate()
                            self.free_task_nodes(task_id,ensemble)
                            ensemble.update_task_info(task_id,{"end_time":time.perf_counter(),
                                                               "status":status,
                                                               "process":None,
                                                               "assigned_nodes":[],}
                                                            #    "stdout":out.decode(),
                                                            #    "stderr":err.decode()}
                                                               ,pid=local_pid)
                            ensemble.save_ensemble_status()
                    
        return None
    
    def get_pending_tasks(self,my_pid:int=0):
        pending_tasks = []
        for ensemble_name,ensemble in self.ensembles.items():
            for local_pid,pid in enumerate(self.pids_per_ensemble[ensemble_name]):
                if pid == my_pid:
                    pending_tasks = pending_tasks + ensemble.get_ready_task_ids(local_pid) + ensemble.get_running_task_ids(local_pid)
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

    def run_tasks_serial(self,my_pid:int=0) -> None:
        while True:
            count = 0
            for ensemble_name,ensemble in self.ensembles.items():
                for local_pid,pid in enumerate(self.pids_per_ensemble[ensemble_name]):
                    if my_pid == pid:
                        launched_tasks = self.launch_ready_tasks(ensemble,local_pid=local_pid,my_pid=my_pid)
            self.poll_running_tasks(my_pid=my_pid)
            self.report_status(my_pid)
            if self.update_interval is not None:
                if time.perf_counter() - self.last_update_time > self.update_interval:
                    deleted_tasks = self.update_ensembles()
                    self.delete_tasks(deleted_tasks)
                    self.last_update_time = time.perf_counter()
            time.sleep(self.poll_interval)
            if len(self.get_pending_tasks(my_pid)) == 0:
                for ensemble_name,ensemble in self.ensembles.items():
                    ensemble.save_ensemble_status()
                break

        return None
    
    def run_tasks_parallel(self):
        processes = []
        for i in range(self.n_parallel):
            p = mp.Process(target=self.run_tasks_serial,args=(i,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    
    def run_tasks(self) -> None:
        fname = os.path.join(os.getcwd(),"outputs",self.logfile)
        if os.path.exists(fname):
            os.remove(fname)
        os.makedirs(os.path.join(os.getcwd(),"outputs"),exist_ok=True)
        if self.n_parallel == 1:
            self.run_tasks_serial()
        else:
            self.run_tasks_parallel()
