import json
import os, stat
import socket
import time
import threading
import multiprocessing as mp
import dragon
from ensemble_launcher.helper_functions import *
import numpy as np
from ensemble_launcher.ensemble import *
from ensemble_launcher.worker import *
from ensemble_launcher.master import *
import logging

class ensemble_launcher:
    def __init__(self,
                 config_file:str,
                 ncores_per_node:int=None,
                 ngpus_per_node:int=None,
                 parallel_backend="multiprocessing",
                 logging_level=logging.INFO,
                 force_level:str=None) -> None:
        self.update_interval = None ##how often to update the ensembles in secs
        self.poll_interval = 60 ##how often to poll the running tasks in secs
        self.parallel_backend = parallel_backend
        self.logging_level = logging_level
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
        self.all_tasks = {}
        for en,e in self.ensembles.items():
            self.all_tasks.update(e.get_task_infos())
        
        if force_level is not None:
            assert force_level in ["single","double"]
            if force_level == "double":
                self.global_master = self._init_global_master(True)
            else:
                self.global_master = self._init_global_master(False)
        else:
            if len(self.total_nodes) > 128:    
                self.global_master = self._init_global_master(True)
            else:
                self.global_master = self._init_global_master(False)
        
        return None
    

    def _init_global_master(self,is_global_master:bool):
        if is_global_master:
            self.logger.info("Running in multi level mode")
        else:
            self.logger.info("Running in single level mode")

        return master(
                "global_master",
                self.all_tasks,
                self.total_nodes,
                self.sys_info,
                parallel_backend=self.parallel_backend,
                n_children=max(1,len(self.total_nodes)//128) if is_global_master else None,
                max_children_nnodes=self.max_nodes_per_master,
                is_global_master=is_global_master,
                logging_level=self.logging_level,
                update_interval=self.update_interval
            )

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
        self.logger.setLevel(self.logging_level)

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
        self.last_update_time = time.time()
    
        # Create and start the process
        parent_conn,child_conn = mp.Pipe()
        process = mp.Process(target=self.global_master.run_children,args=(child_conn,))
        process.start()
        if self.update_interval is not None:
            while process.is_alive():
                time.sleep(self.update_interval)
                ##delete the tasks
                deleted_tasks = self.update_ensembles()
                for task_id in deleted_tasks:
                    del self.all_tasks[task_id]
                ##add any new tasks
                new_tasks = {}
                for en,e in self.ensembles.items():
                    for task_id in e.get_task_ids():
                        if task_id not in self.all_tasks:
                            task_info = e.get_task_info(task_id)
                            new_tasks[task_id] = task_info
                            self.all_tasks[task_id] = task_info
                # Check if tasks have changed
                if len(deleted_tasks) > 0 or len(new_tasks) > 0:
                    self.logger.info(f"Tasks have been updated. {len(deleted_tasks)} tasks deleted. {len(new_tasks)} tasks added")
                    ###update the global master tasks
                    self.commit_update(parent_conn,deleted_tasks,new_tasks)
                else:
                    self.logger.debug("No changes in tasks detected.")
        process.join()
    
        return

    def commit_update(self,parent_conn,deleted_tasks:dict,new_tasks:dict):
        nsuccess = 0
        pid = 0
        self.logger.debug("Sending update to global master")
        ###this block the child process
        parent_conn.send(("SYNC",))
        msg = None
        while msg != "SYNCED":
            if parent_conn.poll(timeout=0.5):
                msg=parent_conn.recv()
        self.logger.debug(f"Synced with global master")
        parent_conn.send(("UPDATE",
                            deleted_tasks,
                            new_tasks))
        if parent_conn.poll(timeout=300):
            msg = parent_conn.recv()
        self.logger.info(f"Received {msg} from global master")

            

        
            
                



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
