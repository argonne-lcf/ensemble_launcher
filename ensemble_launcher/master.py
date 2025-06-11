from .worker import *
from .Node import *
try:
    import dragon
    DRAGON_AVAILABLE = True
except ImportError:
    DRAGON_AVAILABLE = False

import multiprocessing as mp
import os
import gc
import sys
import pickle

class master(Node):
    def __init__(self,
                 master_id:str,
                 my_tasks:dict,
                 my_nodes:list,
                 sys_info:dict,
                 my_master:Node=None,
                 parallel_backend="multiprocessing",
                 n_children:int=None,
                 max_children_nnodes:int=None,
                 is_global_master:bool=False,
                 comm_config:dict={"comm_layer":"multiprocessing"},
                 logger:bool=False,
                 logging_level=logging.INFO,
                 update_interval:int=None):
        super().__init__(master_id,
                         my_tasks,
                         my_nodes,
                         sys_info,
                         {"comm_layer":comm_config["comm_layer"],"role":"parent"},
                         logger=logger,
                         logging_level=logging_level,
                         update_interval=update_interval)
        self.my_master = my_master
        self.parallel_backend = parallel_backend
        self.is_global_master = is_global_master
        assert parallel_backend in ["multiprocessing","dragon","mpi"], f"Unsupported parallel backend: {parallel_backend}. Supported backends are 'multiprocessing', 'dragon', and 'mpi'."
        if self.parallel_backend == "dragon":
            if not DRAGON_AVAILABLE:
                raise ImportError("Dragon is not available. Please install dragon to use the dragon backend.")
            mp.set_start_method("dragon")

        # For tracking child processes and pipes
        self.processes = []
        self.policies = []
        

        ##
        ###this will limit number of nodes assigned to each child. 
        ##Note that this option will remove tasks that have num nodes > max_children_nnodes
        ##So, set this large enough to not remove any tasks
        self.max_children_nnodes = max_children_nnodes
        children_assignments = self.assign_children(n_children=n_children,max_children_nnodes=max_children_nnodes)
        self.n_children = len(children_assignments)
        self.children_nodes = {}
        self.children_names = []
        self.children_tasks = {}
        for pid in range(self.n_children):
            self.children_names.append(f"{self.node_id}_child_{pid}")
        total = 0
        for cid,assignment in children_assignments.items():
            self.children_tasks[self.children_names[cid]] = {task_id:self.my_tasks[task_id] for task_id in assignment["task_ids"]}
            self.children_nodes[self.children_names[cid]] = self.my_nodes[total : total + assignment["nnodes"]]
            total += assignment["nnodes"]
        ##
        self.progress_info = {}
        self.progress_info["nrunning_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nready_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nfailed_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nfinished_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nfree_cores"] = [0 for i in range(self.n_children)]
        self.progress_info["nfree_gpus"] = [0 for i in range(self.n_children)]

    def _initialize_children(self):
        """Initialize the appropriate children based on master type."""
        for child_name in self.children_names:
            if self.is_global_master:
                # Create local masters as children
                local_master = master(
                    child_name,
                    self.children_tasks[child_name],
                    self.children_nodes[child_name],
                    self.sys_info,
                    comm_config={"comm_layer":self.comm_config["comm_layer"],"role":"parent"},
                    parallel_backend=self.parallel_backend,
                    is_global_master=False,
                    logging_level=self.logging_level,
                    update_interval=self.update_interval
                )
                self.add_child(child_name,local_master)
            else:
                # Create workers as children
                w = worker(
                    child_name,
                    self.children_tasks[child_name],
                    self.children_nodes[child_name],
                    self.sys_info,
                    comm_config=self.comm_config,
                    update_interval=self.update_interval,
                    logging_level=self.logging_level
                )
                self.add_child(child_name,w)
        
            if self.parallel_backend == "dragon":
                policy = dragon.infrastructure.policy.Policy(
                    placement=dragon.infrastructure.policy.Policy.Placement.HOST_NAME,
                    host_name=self.children_nodes[child_name][0]
                )
                self.policies.append(policy)


    def assign_children(self,
                        n_children:int=None,
                        max_children_nnodes:int=None):
        """
        Assign tasks to workers based on resource requirements.
        """
        
        # Step 1: Sort tasks by decreasing number of nodes required
        sorted_tasks = sorted(self.my_tasks.items(), key=lambda x: x[1]['num_nodes'], reverse=True)
        
        ###remove tasks that have num nodes < total nodes of this master
        ##when max_children_nnodes is not None, remove tasks that have num nodes > max_children_nnodes
        removed_tasks = []
        if max_children_nnodes is not None:
            while sorted_tasks and sorted_tasks[0][1]["num_nodes"] > max_children_nnodes and sorted_tasks[0][1]["num_nodes"] > len(self.my_nodes):
                removed_tasks.append((sorted_tasks.pop(0))[0])
        else:
            while sorted_tasks and sorted_tasks[0][1]["num_nodes"] > len(self.my_nodes):
                removed_tasks.append((sorted_tasks.pop(0))[0])
        
        if len(removed_tasks) > 0:
            if self.logger:
                self.logger.warning(f"Can't schedule {','.join(removed_tasks)}!")
            else:
                print(f"Can't schedule {','.join(removed_tasks)}!")

        # Step 1: Calculate cumulative sum of nodes required for tasks
        cum_sum_nnodes = []
        for _, task in sorted_tasks:
            # Add the number of nodes required for the current task to the cumulative sum
            cum_sum_nnodes.append(cum_sum_nnodes[-1] + task["num_nodes"] if len(cum_sum_nnodes) > 0 else 0 + task["num_nodes"])
        
        # Step 2: Determine the number of workers
        if n_children is not None:
            nworkers = 0
            while nworkers < len(cum_sum_nnodes) and cum_sum_nnodes[nworkers] <= len(self.my_nodes) and nworkers < n_children:
                nworkers += 1
        else:
            nworkers = 0
            while nworkers < len(cum_sum_nnodes) and cum_sum_nnodes[nworkers] <= len(self.my_nodes):
                nworkers += 1
        
        # Step 3: Initialize worker assignments for the first set of tasks
        children_assignments = {
            wid: {
            "nnodes": task["num_nodes"],  # Total nodes available to the worker
            "ntasks": 1,                 # Number of tasks assigned to the worker
            "task_ids": [task_id]        # List of task IDs assigned to the worker
            }
            for wid, (task_id, task) in enumerate(sorted_tasks[:nworkers])
        }

        assigned_nnodes = sum([a["nnodes"] for _, a in children_assignments.items()])
        if assigned_nnodes < len(self.my_nodes):
            for i in range(len(self.my_nodes) - assigned_nnodes):
                children_assignments[i % nworkers]["nnodes"] += 1

        # Step 4: Assign remaining tasks to workers in a round-robin fashion
        for i, (task_id, task) in enumerate(sorted_tasks[nworkers:]):
            # Determine the worker ID in a round-robin manner
            worker_id = i % nworkers
            # Increment the number of tasks assigned to the worker
            children_assignments[worker_id]["ntasks"] += 1
            # Add the task ID to the worker's task list
            children_assignments[worker_id]["task_ids"].append(task_id)

        return children_assignments

    def report_status(self):
        progress_info = {}
        for k,v in self.progress_info.items():
            progress_info[k] = sum(v)
        if self.logger:
            self.logger.debug(f"Progress info: {progress_info}")
        self.send_to_parent(0,progress_info)
        nnodes = len(self.my_nodes)
        progress_info["total_cores"] = self.sys_info["ncores_per_node"]*nnodes
        progress_info["total_gpus"] = self.sys_info["ngpus_per_node"]*nnodes
        status_str = ",".join([f"{k}:{v}" for k,v in progress_info.items()])
        if self.logger: self.logger.info(f"{status_str}")

    def run_children(self, logger=True):
        """Wrapper function to run the appropriate type of children."""
        if self.is_global_master:
            return self._run_local_masters(logger=logger)
        else:
            return self._run_workers(logger=logger)

    def _run_workers(self,logger=False):
        """Run worker children (for local master)."""
        self._initialize_children()
        if self.comm_config["comm_layer"] == "zmq":
            self.setup_zmq_sockets()
        
        if logger: 
            self.configure_logger(self.logging_level)
            self.logger.info("Started running tasks")
        
        for wid in self.children_names:
            if self.logger: self.logger.debug(f"Worker {wid} has {len(self.children_tasks[wid])} tasks and {self.children_nodes[wid]} nodes")
        
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.path.join(os.path.dirname(__file__),'..')}:{env.get('PYTHONPATH', '')}"
        # Start all worker processes
        for pid,child_name in enumerate(self.children_names):
            if self.comm_config["comm_layer"] == "zmq":
                self.children[child_name].parent_address = self.my_address
            
            if self.parallel_backend == "dragon":
                p = dragon.native.process.Process(
                    target=self.children[child_name].run_tasks,
                    policy=self.policies[child_name],
                    env=env
                )
            elif self.parallel_backend == "multiprocessing":
                p = mp.Process(
                    target=self.children[child_name].run_tasks,args=(False,)
                )
                p.start()
                self.processes.append(p)
            elif self.parallel_backend == "mpi":
                os.makedirs(os.path.join(os.getcwd(),".obj"),exist_ok=True)
                ###dump the child object to a file
                for child_name,child_obj in self.children.items():
                    with open(os.path.join(os.getcwd(),".obj",f"{child_name}.pkl"),"wb") as f:
                        pickle.dump(child_obj,f)
                    ##launch the child process using mpiexec
                    cmd = f"mpiexec -n 1 --hosts {self.children_nodes[child_name][0]}"+\
                        f" python -m ensemble_launcher.worker"+\
                            f" {os.path.join(os.getcwd(),'.obj',f'{child_name}.pkl')} 1"
                    if self.logger: self.logger.debug(f"Running command: {cmd}")
                    p = subprocess.Popen(
                        cmd, shell=True, env=env
                    )
                    self.processes.append(p)
                                            # f" python -m {os.path.join(os.path.dirname(__file__), 'worker.py')}"+\
                ##wait for all children to connect
                nready = 0
                tstart = time.time()
                timeout = 600
                while nready < self.n_children:
                    for child_name in self.children_names:
                        msg = self.recv_from_child(child_name, timeout=0.5)
                        if msg == "READY":
                            nready += 1
                            if self.logger: self.logger.debug(f"Child {child_name} is ready")
                    time.sleep(0.5)
                    if time.time() - tstart > timeout:
                        if self.logger: self.logger.warning(f"Timeout waiting for children to be ready")
                        raise TimeoutError("Timeout waiting for children to be ready")


            for task_id, task_info in self.children_tasks[child_name].items():
                self.my_tasks[task_id].update({"status":"running"})
                task_info.update({"status":"running"})
                
        if self.logger: self.logger.info("Done forking processes")
        
        # Monitor worker processes
        return self._monitor_children()

    def _run_local_masters(self,logger=False):
        """Run local master children (for global master)."""
        self._initialize_children()
        if self.comm_config["comm_layer"] == "zmq":
            self.setup_zmq_sockets()
        if logger: 
            self.configure_logger(self.logging_level)
            self.logger.info("Started running tasks")

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{os.path.join(os.path.dirname(__file__),'..')}:{env.get('PYTHONPATH', '')}"
        # Start all local master processes
        for pid,child_name in enumerate(self.children_names):
            if self.comm_config["comm_layer"] == "zmq":
                self.children[child_name].parent_address = self.my_address
            if self.parallel_backend == "dragon":
                p = dragon.native.process.Process(
                    target=self.children[child_name].run_children, 
                    policy=self.policies[child_name], 
                    env=env
                )
                p.start()
                self.processes.append(p)
            elif self.parallel_backend == "multiprocessing":
                p = mp.Process(
                    target=self.children[child_name].run_children
                )
                p.start()
                self.processes.append(p)
            elif self.parallel_backend == "mpi":
                os.makedirs(os.path.join(os.getcwd(),".obj"),exist_ok=True)
                ###dump the child object to a file
                for child_name,child_obj in self.children.items():
                    with open(os.path.join(os.getcwd(),".obj",f"{child_name}.pkl"),"wb") as f:
                        pickle.dump(child_obj,f)
                    ##launch the child process using mpiexec
                    cmd = f"mpiexec -n 1 --hosts {self.children_nodes[child_name][0]}"+\
                        f" python -m ensemble_launcher.master"+\
                            f" {os.path.join(os.getcwd(),'.obj',f'{child_name}.pkl')} 1"
                    if self.logger: self.logger.debug(f"Running command: {cmd}")
                    p = subprocess.Popen(
                        cmd, shell=True, env=env
                    )
                    self.processes.append(p)

                ##wait for all children to connect
                nready = 0
                tstart = time.time()
                timeout = 600
                while nready < self.n_children:
                    for child_name in self.children_names:
                        msg = self.recv_from_child(child_name, timeout=0.5)
                        if msg == "READY":
                            nready += 1
                            if self.logger: self.logger.debug(f"Child {child_name} is ready")
                    time.sleep(0.5)
                    if time.time() - tstart > timeout:
                        if self.logger: self.logger.warning(f"Timeout waiting for children to be ready")
                        raise TimeoutError("Timeout waiting for children to be ready")

            for task_id, task_info in self.children_tasks[child_name].items():
                self.my_tasks[task_id].update({"status":"running"})
                task_info.update({"status":"running"})

        if self.logger: self.logger.info("Done forking masters")

        # Monitor local master processes
        return self._monitor_children()
    
    def _monitor_children(self):
        """Common monitoring code for both types of children."""
        ndone = 0
        done_children = []
        while True:
            ##First, recieve update from my master
            if self.update_interval is not None:
                if self.logger: self.logger.debug("Waiting for update from parent")
                msg = self.recv_from_parent(0,timeout=1)
                if isinstance(msg,tuple) and msg[0] == "KILL":
                    self.send_to_children(("KILL",))
                elif isinstance(msg,tuple) and msg[0] == "SYNC":
                    if self.logger: self.logger.debug("Received a sync message from parent")
                    self.send_to_parent(0,"SYNCED")
                    msg = self.blocking_recv_from_parent(0)
                    if isinstance(msg,tuple) and msg[0] == "UPDATE":
                        successful = self.commit_task_update(msg[1],msg[2])
                        if successful:
                            self.send_to_parent(0,"UPDATE SUCCESSFUL")
                        else:
                            self.send_to_parent(0,"UPDATE UNSUCCESSFUL")
                else:
                    if self.logger: self.logger.debug(f"Received unknown msg from parent: {msg}")

            for pid in range(self.n_children):
                if pid in done_children:
                    continue
                ##there is default timeout of 60s
                msg = self.recv_from_child(self.children_names[pid], timeout=0.5)
                if msg == "DONE":
                    ndone += 1
                    done_children.append(pid)
                else:
                    if msg is not None:
                        for k, v in msg.items():
                            self.progress_info[k][pid] = v
            
            if time.time() - self.last_update_time > 1:
                ##report status
                self.report_status()
                self.last_update_time = time.time()
            if ndone == self.n_children:
                break
        self.report_status()
        self.send_to_parent(0, "DONE")
        # self.send_to_parent(0, self.my_tasks)
        if self.logger: self.logger.info("Done running all tasks")
        # self.cleanup_resources()
        return

    def cleanup_resources(self):
        """Clean up all child processes and pipes"""
        if self.logger: self.logger.info("Cleaning up resources...")
    
        # Terminate any running child processes
        for i, p in enumerate(self.processes):
            if p.is_alive:
                try:
                    if self.logger: self.logger.info(f"Terminating process {i}")
                    if self.parallel_backend in ["dragon","multiprocessing"]:
                        p.kill()
                        p.join(timeout=0.5)
                    elif self.parallel_backend == "mpi":
                        p.terminate()
                        p.wait(timeout=0.5)
                except Exception as e:
                    if self.logger: self.logger.error(f"Error terminating process {i}: {e}")
        self.close()
    
        # Clear data structures
        self.children_obj = []
        self.children = {}
        self.parents = {}
        self.processes = []

        gc.collect()  # Force garbage collection to free up memory
        if self.logger: self.logger.info("Resource cleanup completed")
    # For backward compatibility
    def run_workers(self, parent_pipe=None):
        return self._run_workers(parent_pipe)
    
    def run_local_masters(self, parent_pipe=None):
        return self._run_local_masters(parent_pipe)
    
    def delete_tasks(self, deleted_tasks:dict)->dict:
        children_deleted_tasks = {child_name:{} for child_name in self.children_names}
        for pid,child_name in enumerate(self.children_names):
            for task_id,task_info in self.children_tasks[child_name].items():
                if task_id in deleted_tasks:
                    children_deleted_tasks[child_name][task_id] = task_info
            for task_id in children_deleted_tasks[child_name].keys():
                del self.my_tasks[task_id]
                del self.children_tasks[child_name][task_id]
        return children_deleted_tasks


    def add_tasks(self, new_tasks:dict):
        sorted_new_task_ids = sorted(new_tasks.keys(),key=lambda x:new_tasks[x]["num_nodes"],reverse=True)
        count = 0
        new_tasks_children = {child_name:{} for child_name in self.children_names}
        for idx,task_id in enumerate(sorted_new_task_ids):
            for pid in range(count%self.n_children,self.n_children):
                child_name = self.children_names[pid]
                if len(self.children_nodes[child_name]) >= new_tasks[task_id]["num_nodes"]:
                    new_tasks_children[child_name][task_id] = new_tasks[task_id]
                    self.children_tasks[child_name][task_id] = new_tasks[task_id]
                    count += 1
                    self.my_tasks[task_id] = new_tasks[task_id]
                    break
            if pid == self.n_children:
                if self.logger: self.logger.warning(f"Can't schedule task {task_id} {new_tasks[task_id]['num_nodes']} <= {len(self.children_nodes[0])}")
        return new_tasks_children
    
    def commit_task_update(self,deleted_tasks:dict,new_tasks:dict):
        deleted_children_tasks = self.delete_tasks(deleted_tasks)
        new_children_tasks = self.add_tasks(new_tasks)
        nsuccess = 0
        for child_name in self.children_names:
            if self.logger: self.logger.debug("Sending update to child "+child_name)
            ###this block the child process
            self.send_to_child(child_name,("SYNC",))
            msg = None
            while msg != "SYNCED":
                msg=self.recv_from_child(child_name,timeout=0.5)
                if self.logger: self.logger.debug(f"Syncing with child {child_name}:{msg}")
            if self.logger: self.logger.debug(f"Synced with child {child_name}:{msg}")
            self.send_to_child(child_name,("UPDATE",
                                    deleted_children_tasks[child_name],
                                    new_children_tasks[child_name]))
            msg = self.recv_from_child(child_name,timeout=300)
            if msg == "UPDATE SUCCESSFUL":
                if self.logger: self.logger.info(f"Updating child {child_name} successful")
                nsuccess += 1
            else:
                if self.logger: self.logger.warning(f"Update unsuccessful: {msg}")
        return nsuccess == self.n_children

    ##these are to make sure that the cleanup_resources is called
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_resources()
        return False
    
if __name__ == "__main__":
    master_obj_file = sys.argv[1]
    logger = int(sys.argv[2]) == 1
    with open(master_obj_file, "rb") as f:
        master_obj = pickle.load(f)
    if logger:
        master_obj.configure_logger(master_obj.logging_level)
        master_obj.logger.info("Loaded master object from file")
    master_obj.run_children(logger=False)


#************depreated functions***************
    # def split_nodes(self)->list:
    #     worker_nodes = []
    #     if len(self.my_nodes)<self.n_children:
    #         raise ValueError("Total number of nodes < number of parallel task launchers! Please set nparallel = 1")
    #     nn = len(self.my_nodes)//self.n_children
    #     for i in range(self.n_children):
    #         worker_nodes.append(self.my_nodes[i*nn:(i+1)*nn])
    #     worker_nodes[-1].extend(self.my_nodes[(i+1)*nn:])
    #     return worker_nodes
    
    # def split_tasks(self)->dict:
    #     nt_pw = len(self.my_tasks)//self.n_children
    #     worker_tasks = {}
    #     for wid in range(self.n_children-1):
    #         worker_tasks[wid] = {}
    #         for task_id in list(self.my_tasks.keys())[wid*nt_pw:(wid+1)*nt_pw]:
    #             worker_tasks[wid][task_id] = self.my_tasks[task_id]
    #     worker_tasks[self.n_children - 1] = {
    #         task_id: self.my_tasks[task_id]
    #         for task_id in list(self.my_tasks.keys())[(self.n_children - 1) * nt_pw:]
    #     }
    #     return worker_tasks