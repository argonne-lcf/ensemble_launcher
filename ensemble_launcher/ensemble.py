import json
import os
import numpy as np


##NOTE: all the tasks need to be modified through ensemble object
class ensemble:
    def __init__(self,ensemble_name:str,ensemble_info:dict,nparallel:int=1,system:str="aurora") -> None:
        self.__ensemble_name = ensemble_name
        self.__ensemble_info = ensemble_info
        self.system = system
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
    
    def local_ntasks(self,pid:int):
        return len(self.__local_tasks[pid])
    
    ##this function checks if ensemble config is correct
    def check_ensemble_info(self):
        ensemble = self.__ensemble_info
        ##assert some necessary vals
        assert "num_nodes" in ensemble.keys()
        assert "launcher" in ensemble.keys()
        assert "relation" in ensemble.keys()
        assert "cmd_template" in ensemble.keys()

    def __generate_ensemble(self) -> dict:
        """check ensemble config
        """
        self.check_ensemble_info()
        ensemble = self.__ensemble_info
        relation = ensemble["relation"]
        """this is to generate the lists
        """
        for key,value in ensemble.items():
            if isinstance(value, str) and value.startswith("linspace"):
                args = eval(value[len("linspace"):])
                ensemble[key] = np.linspace(*args).tolist()
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
                task["index"] = i
                for opt in non_list_options:
                    task[opt] = ensemble[opt]
                for opt in list_options:
                    task[opt] = ensemble[opt][i]
                tasks.append(self.__set_defaults(task))
                if "run_dir" in list_options:
                    task["log_file"] = os.path.join(task["run_dir"],f"log.txt")
                    task["err_file"] = os.path.join(task["run_dir"],f"err.txt")
                    task["gpu_affinity_file"] = os.path.join(task["run_dir"],f"set_affinity.sh")
                    task["mpi_rankfile"] = os.path.join(task["run_dir"],f"rankfile.txt")
                else:
                    task["log_file"] = os.path.join(task["run_dir"],f"log_{i}.txt")
                    task["err_file"] = os.path.join(task["run_dir"],f"err_{i}.txt")
                    task["gpu_affinity_file"] = os.path.join(task["run_dir"],f"set_affinity_{i}.sh")
                    task["mpi_rankfile"] = os.path.join(task["run_dir"],f"rankfile_{i}.txt")
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
                task["index"] = tid
                loc = np.unravel_index(tid,dim)
                for id,opt in enumerate(list_options):
                    task[opt] = ensemble[opt][loc[id]]
                for opt in non_list_options:
                    task[opt] = ensemble[opt]
                tasks.append(self.__set_defaults(task))
                if "run_dir" in list_options:
                    task["log_file"] = os.path.join(task["run_dir"],f"log.txt")
                    task["err_file"] = os.path.join(task["run_dir"],f"err.txt")
                else:
                    task["log_file"] = os.path.join(task["run_dir"],f"log_{tid}.txt")
                    task["err_file"] = os.path.join(task["run_dir"],f"err_{tid}.txt")
        else:
            raise ValueError(f"Unknown relation {relation}")

        return {task["id"]:task for task in tasks}

    def __generate_task_id(self, task:dict) -> str:
        assert self.__list_options is not None
        bin_options_str = "-".join(f"{k}-{task[k]}" for k in self.__list_options)
        ##NOTE: the string should always start with ensemble name
        unique_str = f"{task['ensemble_name']}-{task['index']}-{bin_options_str}"
        return unique_str
    
    ###function sets the default values for a task
    def __set_defaults(self,task:dict) -> dict:
        ##some defaults
        if "status" not in task.keys():
            task["status"] = "ready"
                
        task["id"] = self.__generate_task_id(task)

        if "run_dir" not in task.keys():
            task["run_dir"] = os.getcwd()
        else:
            task["run_dir"] = os.path.join(os.getcwd(),task["run_dir"])

        if "launch_dir" not in task.keys():
            task["launch_dir"] = os.getcwd()
        else:
            task["launch_dir"] = os.path.join(os.getcwd(),task["launch_dir"])
        
        if "num_processes_per_node" not in task.keys():
            task["num_processes_per_node"] = 1
        
        if "env" not in task.keys():
            task["env"] = {}
        
        if "io" not in task.keys():
            task["io"] = False

        task["system"] = self.system
        
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
    
    def get_task_infos(self,pid:int=None)->list:
        task_infos = {}
        for task_id in self.get_task_ids(pid):
            task_infos[task_id] = self.__tasks[task_id]
        return task_infos
    
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
    
    def update_task_status(self, task_id: str, status: str, pid: int = 0, force:bool=False) -> None:
        if status == "running":
            if self.__tasks[task_id]["status"] != "running":
                assert force or self.__tasks[task_id]["status"] == "ready"
                self.__ensemble_state[pid]["running_task_ids"][task_id] = "running"
                self.__ensemble_state[pid]["ready_task_ids"].pop(task_id)
            else:
                return
        elif status == "failed":
            assert force or self.__tasks[task_id]["status"] == "running"
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
            assert force or self.__tasks[task_id]["status"] == "running"
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
    
    def update_task_info(self,task_id:str,info:dict,pid:int=0,force:bool=False) -> None:
        if "status" in info.keys():
            self.update_task_status(task_id,info["status"],pid=pid,force=force)
            info.pop("status")
        self.__tasks[task_id].update(info)
        return None
    
    def update_host_nodes(self,task_id:str,hosts:list) -> None:
        self.__tasks[task_id]["task_type_options"]["--hosts"] = ",".join(hosts)
    

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