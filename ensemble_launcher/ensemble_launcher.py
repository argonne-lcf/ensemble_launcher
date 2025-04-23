import json
import os, stat
import socket
import time
import multiprocessing as mp
import dragon
from .helper_functions import *
import numpy as np
from .ensemble import *
from .worker import *
from .master import *
import logging

class ensemble_launcher:
    def __init__(self,
                 config_file:str,
                 ncores_per_node:int=None,
                 ngpus_per_node:int=None,
                 parallel_backend="multiprocessing") -> None:
        self.update_interval = None ##how often to update the ensembles in secs
        self.poll_interval = 60 ##how often to poll the running tasks in secs
        self.parallel_backend = parallel_backend
        assert parallel_backend in ["multiprocessing","dragon"]
        if self.parallel_backend == "dragon":
            mp.set_start_method("dragon")
        ##
        self.ensembles = {}
        self.start_time = time.time()
        self.last_update_time = time.time()
        self.config_file = config_file
        ##system info
        self.sys_info = {}
        self.read_input_file()
        os.makedirs(os.path.join(os.getcwd(),"outputs"),exist_ok=True)
        self.configure_logger()
        
        ##update the system info. NOTE: precedence order config > inputs > functions
        if "ncores_per_node" not in self.sys_info.keys():
            self.sys_info["ncores_per_node"] = self.get_cores_per_node() if ncores_per_node is None else ncores_per_node
        if "ngpus_per_node" not in self.sys_info.keys():
            self.sys_info["ngpus_per_node"] = 0 if ngpus_per_node is None else ngpus_per_node
        ##
        self.total_nodes = self.get_nodes()
        # self.pids_per_ensemble = {en:[0] for en in self.ensembles.keys()}
        # if self.global_master.n_children > 1:
        #     self.distribute_procs()
        all_tasks = {}
        for en,e in self.ensembles.items():
            all_tasks.update(e.get_task_infos())
        ##this is the global master. This is only used to split the tasks among local masters.
        self.global_master = master(
            "global_master",
            all_tasks,
            self.total_nodes,
            self.sys_info,
            parallel_backend=self.parallel_backend,
            n_children= None,
            max_children_nnodes=self.max_nodes_per_master)
        
        print(f"Total number of local masters: {self.global_master.n_children}")
        for i in range(self.global_master.n_children):
            print(f"Local master {i} has {len(self.global_master.children_tasks[i])} tasks and {len(self.global_master.children_nodes[i])} nodes")
    
        self.comm_config = {}
        self.masters = []

        self.progress_info = {}
        self.progress_info["nrunning_tasks"] = [0 for i in range(self.global_master.n_children)]
        self.progress_info["nready_tasks"] = [0 for i in range(self.global_master.n_children)]
        self.progress_info["nfailed_tasks"] = [0 for i in range(self.global_master.n_children)]
        self.progress_info["nfinished_tasks"] = [0 for i in range(self.global_master.n_children)]
        self.progress_info["nfree_cores"] = [0 for i in range(self.global_master.n_children)]
        self.progress_info["nfree_gpus"] = [0 for i in range(self.global_master.n_children)]
        
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
            self.max_nodes_per_master = data.get("max_nodes_per_master",None)
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
    
    def configure_logger(self):
        self.logger = logging.getLogger(f"el")
        handler = logging.FileHandler(f'./outputs/el_log.txt', mode='w')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

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
    
    def get_cores_per_node(self):
        if os.getenv("NCPUS") is not None:
            return int(os.getenv("NCPUS"))
        else:
            return os.cpu_count()
    
    def report_status(self):

        progress_info = {}
        for k,v in self.progress_info.items():
            progress_info[k] = sum(v)
        nnodes = len(self.total_nodes)
        progress_info["total_cores"] = self.sys_info["ncores_per_node"]*nnodes
        progress_info["total_gpus"] = self.sys_info["ngpus_per_node"]*nnodes
        status_str = ",".join([f"{k}:{v}" for k,v in progress_info.items()])
        self.logger.info(status_str)


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
        deleted_tasks = {}
        with open(self.config_file, "r") as file:
            data = json.load(file)
            ensemble_infos = data["ensembles"]
            for ensemble_name,ensemble_info in ensemble_infos.items():
                deleted_tasks.update(self.ensembles[ensemble_name].update_ensemble(ensemble_info))
        return deleted_tasks
    
    def run_tasks(self):
        if len(self.total_nodes) > 128:
            self.logger.info("Running in multi level mode")
            self.global_master.run_local_masters()
        else:
            self.logger.info("Running in single level mode")
            self.global_master.run_workers()
            

        
            
                



#****************deprecated functions****************
    # ##pool all the tasks and split them among available processes
    # def distribute_procs(self)->None:
    #     ntasks_per_proc = sum([e.ntasks for e in self.ensembles.values()])//self.global_master.n_children
    #     cum_tasks = 0
    #     for en,e in self.ensembles.items():
    #         start = cum_tasks//ntasks_per_proc
    #         end = (cum_tasks+e.ntasks)//ntasks_per_proc + 1
    #         if start == end:
    #             self.pids_per_ensemble[en] = [start]
    #         else:
    #             self.pids_per_ensemble[en] = [i for i in range(start,end+1)]
    #         self.pids_per_ensemble[en] = [i for i in self.pids_per_ensemble[en] if i < self.global_master.n_children]
    #         self.ensembles[en].set_np(len(self.pids_per_ensemble[en]))
    #         cum_tasks+=e.ntasks
    #         e.save_ensemble_status()
    #     return None

    # def split_nodes(self)->list:
    #     my_nodes = []
    #     if len(self.total_nodes)<self.global_master.n_children:
    #         raise ValueError("Total number of nodes < number of parallel task launchers! Please set nparallel = 1")
    #     nn = len(self.total_nodes)//self.global_master.n_children
    #     for i in range(self.global_master.n_children):
    #         my_nodes.append(self.total_nodes[i*nn:(i+1)*nn])
    #     my_nodes[-1].extend(self.total_nodes[(i+1)*nn:])
    #     return my_nodes
    