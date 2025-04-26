from .worker import *
from .Node import *
import dragon
import multiprocessing as mp
import os
import gc

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
                 logging_level=logging.INFO,
                 update_interval:int=None):
        super().__init__(master_id,
                         my_tasks,
                         my_nodes,
                         sys_info,
                         comm_config,
                         logger=False,
                         logging_level=logging_level,
                         update_interval=update_interval)
        self.my_master = my_master
        self.parallel_backend = parallel_backend
        self.is_global_master = is_global_master
        assert parallel_backend in ["multiprocessing","dragon"]

        # For tracking child processes and pipes
        self.processes = []
        self.policies = []
        
        ##for now I will just use n_children = number of nodes
        ##here, I am using children instead of workers because, 
        # a master can have either workers or local masters as children
        self.children_obj = []
        ##
        ###this will limit number of nodes assigned to each child. 
        ##Note that this option will remove tasks that have num nodes > max_children_nnodes
        ##So, set this large enough to not remove any tasks
        self.max_children_nnodes = max_children_nnodes
        children_assignments = self.assign_children(n_children=n_children,max_children_nnodes=max_children_nnodes)
        self.n_children = len(children_assignments)
        self.children_nodes = []
        self.child_pipes = []
        self.children_tasks = []
        total = 0
        for cid,assignment in children_assignments.items():
            self.children_tasks.append({task_id:self.my_tasks[task_id] for task_id in assignment["task_ids"]})
            self.children_nodes.append(self.my_nodes[total : total + assignment["nnodes"]])
            total += assignment["nnodes"]
        ##
        self.progress_info = {}
        self.progress_info["nrunning_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nready_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nfailed_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nfinished_tasks"] = [0 for i in range(self.n_children)]
        self.progress_info["nfree_cores"] = [0 for i in range(self.n_children)]
        self.progress_info["nfree_gpus"] = [0 for i in range(self.n_children)]
        
        # Create appropriate children based on master type
        self._initialize_children()

    def _initialize_children(self):
        """Initialize the appropriate children based on master type."""
        for pid in range(self.n_children):
            parent_conn, child_conn = mp.Pipe()
            self.add_child(pid, parent_conn)
            self.child_pipes.append(child_conn)
            
            if self.is_global_master:
                # Create local masters as children
                local_master = master(
                    f"{self.node_id}_local_master_{pid}",
                    self.children_tasks[pid],
                    self.children_nodes[pid],
                    self.sys_info,
                    parallel_backend=self.parallel_backend,
                    is_global_master=False,
                    logging_level=self.logging_level,
                    update_interval=self.update_interval
                )
                self.children_obj.append(local_master)
            else:
                # Create workers as children
                w = worker(
                    f"{self.node_id}_worker_{pid}",
                    self.children_tasks[pid],
                    self.children_nodes[pid],
                    self.sys_info,
                    update_interval=self.update_interval,
                    logging_level=self.logging_level
                )
                self.children_obj.append(w)
            
            if self.parallel_backend == "dragon":
                policy = dragon.infrastructure.policy.Policy(
                    placement=dragon.infrastructure.policy.Policy.Placement.HOST_NAME,
                    host_name=self.children_nodes[pid][0]
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
        self.send_to_parent(0,progress_info)
        nnodes = len(self.my_nodes)
        progress_info["total_cores"] = self.sys_info["ncores_per_node"]*nnodes
        progress_info["total_gpus"] = self.sys_info["ngpus_per_node"]*nnodes
        status_str = ",".join([f"{k}:{v}" for k,v in progress_info.items()])
        self.logger.info(f"{status_str}")

    def run_children(self, parent_pipe=None):
        """Wrapper function to run the appropriate type of children."""
        if self.is_global_master:
            return self._run_local_masters(parent_pipe)
        else:
            return self._run_workers(parent_pipe)

    def _run_workers(self, parent_pipe=None):
        """Run worker children (for local master)."""
        self.configure_logger(self.logging_level)
        if parent_pipe:
            self.add_parent(0, parent_pipe)
        self.logger.info("Started running tasks")
        
        for wid in range(self.n_children):
            self.logger.debug(f"Worker {wid} has {len(self.children_tasks[wid])} tasks and {self.children_nodes[wid]} nodes")
        
        # Start all worker processes
        for pid in range(self.n_children):
            if self.parallel_backend == "dragon":
                p = dragon.native.process.Process(
                    target=self.children_obj[pid].run_tasks, 
                    args=(self.child_pipes[pid],), 
                    policy=self.policies[pid]
                )
            else:
                p = mp.Process(
                    target=self.children_obj[pid].run_tasks, 
                    args=(self.child_pipes[pid],)
                )
            p.start()
            self.processes.append(p)
            
            for task_id, task_info in self.children_tasks[pid].items():
                self.my_tasks[task_id].update({"status":"running"})
                task_info.update({"status":"running"})
                
        self.logger.info("Done forking processes")
        
        # Monitor worker processes
        return self._monitor_children()

    def _run_local_masters(self, parent_pipe=None):
        """Run local master children (for global master)."""
        self.configure_logger(self.logging_level)
        if parent_pipe:
            self.add_parent(0, parent_pipe)
        self.logger.info("Started running tasks")
        
        # Start all local master processes
        for pid in range(self.n_children):
            if self.parallel_backend == "dragon":
                env = os.environ.copy()
                env["PYTHONPATH"] = f"{os.path.dirname(__file__)}:{env.get('PYTHONPATH', '')}"
                p = dragon.native.process.Process(
                    target=self.children_obj[pid].run_children, 
                    args=(self.child_pipes[pid],), 
                    policy=self.policies[pid], 
                    env=env
                )
            else:
                p = mp.Process(
                    target=self.children_obj[pid].run_children, 
                    args=(self.child_pipes[pid],)
                )
            p.start()
            self.processes.append(p)
            
            for task_id, task_info in self.children_tasks[pid].items():
                self.my_tasks[task_id].update({"status":"running"})
                task_info.update({"status":"running"})
                
        self.logger.info("Done forking masters")
        
        # Monitor local master processes
        return self._monitor_children()
    
    def _monitor_children(self):
        """Common monitoring code for both types of children."""
        ndone = 0
        done_children = []
        while True:
            ##First, recieve update from my master
            if self.update_interval is not None:
                msg = self.recv_from_parent(0,timeout=1)
                if isinstance(msg,tuple) and msg[0] == "KILL":
                    self.send_to_children(("KILL",))
                elif isinstance(msg,tuple) and msg[0] == "SYNC":
                    self.logger.debug("Received a sync message from parent")
                    self.send_to_parent(0,"SYNCED")
                    msg = self.blocking_recv_from_parent(0)
                    if isinstance(msg,tuple) and msg[0] == "UPDATE":
                        successful = self.commit_task_update(msg[1],msg[2])
                        if successful:
                            self.send_to_parent(0,"UPDATE SUCCESSFUL")
                        else:
                            self.send_to_parent(0,"UPDATE UNSUCCESSFUL")
                else:
                    self.logger.debug(f"Received unknown msg from parent: {msg}")

            for pid in range(self.n_children):
                if pid in done_children:
                    continue
                ##there is default timeout of 60s
                msg = self.recv_from_child(pid, timeout=0.5)
                if msg == "DONE":
                    ndone += 1
                    done_children.append(pid)
                    tasks = self.recv_from_child(pid, timeout=0.5)
                    if tasks is not None:
                        for task_id, task_info in tasks.items():
                            if task_id not in self.children_tasks[pid].keys():
                                self.logger.warning(f"{task_id} not in child {pid}")
                            else:
                                self.children_tasks[pid][task_id].update(task_info)
                                self.my_tasks[task_id].update(task_info)
                else:
                    if msg is not None:
                        for k, v in msg.items():
                            self.progress_info[k][pid] = v
            
            if time.time() - self.last_update_time > 5:
                ##report status
                self.report_status()
                self.last_update_time = time.time()
            if ndone == self.n_children:
                break
        self.send_to_parent(0, "DONE")
        self.send_to_parent(0, self.my_tasks)
        self.logger.info("Done running all tasks")
        self.cleanup_resources()
        return

    def cleanup_resources(self):
        """Clean up all child processes and pipes"""
        self.logger.info("Cleaning up resources...")
    
        # Terminate any running child processes
        for i, p in enumerate(self.processes):
            if p.is_alive():
                try:
                    self.logger.info(f"Terminating process {i}")
                    p.kill()
                    p.join(timeout=0.5)
                except Exception as e:
                    self.logger.error(f"Error terminating process {i}: {e}")
        self.close()
    
        # Clear data structures
        self.children_obj = []
        self.children = {}
        self.parents = {}
        self.processes = []

        gc.collect()  # Force garbage collection to free up memory
        self.logger.info("Resource cleanup completed")
    # For backward compatibility
    def run_workers(self, parent_pipe=None):
        return self._run_workers(parent_pipe)
    
    def run_local_masters(self, parent_pipe=None):
        return self._run_local_masters(parent_pipe)
    
    def delete_tasks(self, deleted_tasks:dict)->dict:
        children_deleted_tasks = {pid:{} for pid in range(self.n_children)}
        for pid in range(self.n_children):
            for task_id,task_info in self.children_tasks[pid].items():
                if task_id in deleted_tasks:
                    children_deleted_tasks[pid][task_id] = task_info
            for task_id in children_deleted_tasks[pid]:
                del self.my_tasks[task_id]
                del self.children_tasks[pid][task_id]
        return children_deleted_tasks


    def add_tasks(self, new_tasks:dict):
        sorted_new_task_ids = sorted(new_tasks.keys(),key=lambda x:new_tasks[x]["num_nodes"],reverse=True)
        count = 0
        new_tasks_children = {pid:{} for pid in range(self.n_children)}
        for idx,task_id in enumerate(sorted_new_task_ids):
            for pid in range(count%self.n_children,self.n_children):
                if len(self.children_nodes[pid]) >= new_tasks[task_id]["num_nodes"]:
                    new_tasks_children[pid][task_id] = new_tasks[task_id]
                    self.children_tasks[pid][task_id] = new_tasks[task_id]
                    count += 1
                    self.my_tasks[task_id] = new_tasks[task_id]
                    break
            if pid == self.n_children:
                self.logger.warning(f"Can't schedule task {task_id} {new_tasks[task_id]['num_nodes']} <= {len(self.children_nodes[0])}")
        return new_tasks_children
    
    def commit_task_update(self,deleted_tasks:dict,new_tasks:dict):
        deleted_children_tasks = self.delete_tasks(deleted_tasks)
        new_children_tasks = self.add_tasks(new_tasks)
        nsuccess = 0
        for pid in range(self.n_children):
            self.logger.debug("Sending update to child "+str(pid))
            ###this block the child process
            self.send_to_child(pid,("SYNC",))
            msg = None
            while msg != "SYNCED":
                msg=self.recv_from_child(pid,timeout=0.5)
                self.logger.debug(f"Syncing with child {pid}:{msg}")
            self.logger.debug(f"Synced with child {pid}:{msg}")
            self.send_to_child(pid,("UPDATE",
                                    deleted_children_tasks[pid],
                                    new_children_tasks[pid]))
            msg = self.recv_from_child(pid,timeout=300)
            if msg == "UPDATE SUCCESSFUL":
                self.logger.info(f"Updating child {pid} successful")
                nsuccess += 1
            else:
                self.logger.warning(f"Update unsuccessful: {msg}")
        return nsuccess == self.n_children

    ##these are to make sure that the cleanup_resources is called
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.cleanup_resources()
        return False
    



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