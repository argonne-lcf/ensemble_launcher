import json
import os, stat
import socket
import subprocess
import time
import copy
import multiprocessing as mp
import resource
from helper_functions import *
import numpy as np
from ensemble import *
from worker import *
from master import *

class ensemble_launcher:
    def __init__(self,
                 config_file:str,
                 ncores_per_node:int=None,
                 ngpus_per_node:int=None,
                 parallel_backend="multiprocessing") -> None:
        assert parallel_backend == "multiprocessing"
        self.update_interval = None ##how often to update the ensembles in secs
        self.poll_interval = 60 ##how often to poll the running tasks in secs
        self.n_parallel = None ##number of parallel lacunchers in int
        ##
        self.ensembles = {}
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.config_file = config_file
        ##system info
        self.sys_info = {}
        self.read_input_file()
        ##update the system info. NOTE: precedence order config > inputs > functions
        if "ncores_per_node" not in self.sys_info.keys():
            self.sys_info["ncores_per_node"] = self.get_cores_per_node() if ncores_per_node is None else ncores_per_node
        if "ngpus_per_node" not in self.sys_info.keys():
            self.sys_info["ngpus_per_node"] = 0 if ngpus_per_node is None else ngpus_per_node
        ##
        self.total_nodes = self.get_nodes()
        ###split available nodes among workers
        self.worker_nodes = self.split_nodes()
        print("worker nodes",self.worker_nodes)
        ##split tasks among available workers
        self.pids_per_ensemble = {en:[0] for en in self.ensembles.keys()}
        if self.n_parallel > 1:
            self.distribute_procs()
        
        self.workers = []
        self.comm_config = {}
        self.masters = []

        self.progress_info = {}
        self.progress_info["nrunning_tasks"] = [0 for i in range(self.n_parallel)]
        self.progress_info["nready_tasks"] = [0 for i in range(self.n_parallel)]
        self.progress_info["nfailed_tasks"] = [0 for i in range(self.n_parallel)]
        self.progress_info["nfinished_tasks"] = [0 for i in range(self.n_parallel)]
        self.progress_info["nfree_cores"] = [0 for i in range(self.n_parallel)]
        self.progress_info["nfree_gpus"] = [0 for i in range(self.n_parallel)]
        
        return None
    
    """
    Function reads the input file and builds the ensembles
    NOTES:
    1. It is expected that when a system info is present in the input file. It should atleast have the name.
        The reason for this is to know how to build the launch command.
    """
    def read_input_file(self):
        with open(self.config_file, "r") as file:
            data = json.load(file)
            self.update_interval = data.get("update_interval",None)
            self.poll_interval = data.get("poll_interval",600)
            self.n_parallel = data.get("n_parallel",1)
            self.logfile = data.get("logfile","log.txt")
            if "sys_info" in data.keys():
                assert "name" in data["sys_info"]
                self.sys_info.update(data["sys_info"])
            else:
                self.sys_info["name"] = "local"
            self.sys_info.update()
            ensembles_info = data["ensembles"]
            for ensemble_name,ensemble_info in ensembles_info.items():
                self.ensembles[ensemble_name] = ensemble(ensemble_name,ensemble_info,system=self.sys_info["name"])
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
        node_file = os.getenv("PBS_NODEFILE",None)
        if node_file is not None and os.path.exists(node_file):
            with open(node_file, "r") as f:
                node_list = f.readlines()
                node_list = [(node.split(".")[0]).strip() for node in node_list]
        else:
            node_list = [socket.gethostname()]
        return node_list
    
    def split_nodes(self)->list:
        my_nodes = []
        if len(self.total_nodes)<self.n_parallel:
            raise ValueError("Total number of nodes < number of parallel task launchers! Please set nparallel = 1")
        nn = len(self.total_nodes)//self.n_parallel
        for i in range(self.n_parallel):
            my_nodes.append(self.total_nodes[i*nn:(i+1)*nn])
        my_nodes[-1].extend(self.total_nodes[(i+1)*nn:])
        return my_nodes
    
    def get_cores_per_node(self):
        if os.getenv("NCPUS") is not None:
            return int(os.getenv("NCPUS"))
        else:
            return os.cpu_count()
    
    def report_status(self):
        timestamp = time.time()
        n_fds = self.get_nfd(mp.current_process())
        n_nodes = len(self.total_nodes)
        n_busy_nodes = sum([1 for nodes in self.worker_nodes if len(nodes) > 0])
        total_cores = self.sys_info["ncores_per_node"] * n_nodes
        nfree_cores = sum(self.progress_info["nfree_cores"])
        total_gpu = self.sys_info["ngpus_per_node"] * n_nodes
        nfree_gpus = sum(self.progress_info["nfree_gpus"])
        total_tasks = sum([e.ntasks for e in self.ensembles.values()])
        n_todo_tasks = sum(self.progress_info["nready_tasks"])
        n_running_tasks = sum(self.progress_info["nrunning_tasks"])
        n_failed_tasks = sum(self.progress_info["nfailed_tasks"])
        n_finished_tasks = sum(self.progress_info["nfinished_tasks"])
        fname = self.logfile

        status_string = f"FDs: {n_fds}, Nodes: {n_busy_nodes}/{n_nodes}, Free Cores: {nfree_cores}/{total_cores}, Free GPUs: {nfree_gpus}/{total_gpu}, Tasks: {total_tasks}, ToDo: {n_todo_tasks}, Running: {n_running_tasks}, Failed: {n_failed_tasks}, Finished: {n_finished_tasks}"
        with open(fname, "a") as f:
            f.write(f"{timestamp},{status_string}\n")


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
        
    
    def update_ensembles(self)-> dict:
        print("Updating ensembles")
        deleted_tasks = {}
        with open(self.config_file, "r") as file:
            data = json.load(file)
            ensemble_infos = data["ensembles"]
            for ensemble_name,ensemble_info in ensemble_infos.items():
                deleted_tasks.update(self.ensembles[ensemble_name].update_ensemble(ensemble_info))
        return deleted_tasks
    
    ##this function creates and launches the workers
    def run_tasks(self):
        ##get worker tasks
        worker_tasks = {pid:{} for pid in range(self.n_parallel)}
        for en,e in self.ensembles.items():
            for local_id,pid in enumerate(self.pids_per_ensemble[en]):
                worker_tasks[pid].update(e.get_task_infos(local_id))

        ##create master
        my_master = master("master_0")
        self.masters.append(my_master)

        ##create workers and corresponding processes
        processes = []
        for pid in range(self.n_parallel):
            w = worker( f"worker_{pid}",
                        worker_tasks[pid],
                        self.worker_nodes[pid],
                        my_master,
                        self.sys_info)
            ##connect master and children
            parent_conn, child_conn = mp.Pipe()
            my_master.add_child(pid,parent_conn)
            w.add_parent(0,child_conn)
            self.workers.append(w)
            ##
            p = mp.Process(target=w.run_tasks)
            p.start()
            processes.append(p)
            for task_id,task_info in worker_tasks[pid].items():
                self.ensembles[task_info["ensemble_name"]].update_task_info(task_id,{"status":"running"})
        
        ndone = 0
        while True:
            for pid in range(self.n_parallel):
                ##there is default timeout of 60s
                msg = my_master.recv_from_child(pid)
                if msg == "DONE":
                    ndone += 1
                    tasks = my_master.recv_from_child(pid)
                    for task_id,task_info in tasks.items():
                        self.ensembles[task_info["ensemble_name"]].update_task_info(task_id,task_info)
                else:
                    ##receive metadata
                    for k,v in msg.items():
                        self.progress_info[k][pid] = v
            ##report status
            self.report_status()
            if ndone == self.n_parallel:
                for e in self.ensembles.values():
                    e.save_ensemble_status()
                break
            
                




    