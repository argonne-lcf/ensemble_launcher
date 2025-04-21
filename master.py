from worker import *
from Node import *
import dragon
import multiprocessing as mp

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
                 comm_config:dict={"comm_layer":"multiprocessing",
                                    "parents":{},
                                    "children":{}}):
        super().__init__(master_id,comm_config,logger=False)
        self.my_tasks = my_tasks
        self.my_nodes = my_nodes
        self.my_master = my_master
        self.sys_info = sys_info
        self.parallel_backend = parallel_backend
        assert parallel_backend in ["multiprocessing","dragon"]
        ##for now I will just use n_children = number of nodes
        ##here, I am using children instead of workers because, 
        # a master can have either workers or local masters as children
        self.children = []
        ##
        ###this will limit number of nodes assigned to each child. 
        ##Note that this option will remove tasks that have num nodes > max_children_nnodes
        ##So, set this large enough to not remove any tasks
        self.max_children_nnodes = max_children_nnodes
        children_assignments = self.assign_children(n_children=n_children,max_children_nnodes=max_children_nnodes)
        self.n_children = len(children_assignments)
        self.children_nodes = []
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
        
        print(f"cum_sum_nnodes = {cum_sum_nnodes}")
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

    def run_workers(self,parent_pipe):
        self.configure_logger()
        self.add_parent(0,parent_pipe)
        self.logger.info("Started running tasks")
        if self.parallel_backend == "dragon":
            mp.set_start_method("dragon")
        ##create workers and corresponding processes
        processes = []
        policies = []
        for pid in range(self.n_children):
            w = worker( f"worker_{pid}",
                        self.children_tasks[pid],
                        self.children_nodes[pid],
                        self.sys_info)
            ##connect master and children
            parent_conn, child_conn = mp.Pipe()
            self.add_child(pid,parent_conn)
            self.children.append(w)
            ##
            if self.parallel_backend == "dragon":
                policies.append(dragon.infrastructure.policy.Policy(
                                        placement=dragon.infrastructure.policy.Policy.Placement.HOST_NAME,
                                        host_name=self.children_nodes[pid][0]
                                    ))
                p = dragon.native.process.Process(target=w.run_tasks,args=(child_conn,),policy=policies[-1])
            else:
                p = mp.Process(target=w.run_tasks,args=(child_conn,))
            p.start()
            processes.append(p)
            for task_id,task_info in self.children_tasks[pid].items():
                self.my_tasks[task_id].update({"status":"running"})
        self.logger.info("Done forking processes")
        ndone = 0
        done_workers = []
        
        while True:
            for pid in range(self.n_children):
                if pid in done_workers:
                    continue
                ##there is default timeout of 60s
                msg = self.recv_from_child(pid,timeout=0.5)
                if msg == "DONE":
                    ndone += 1
                    done_workers.append(pid)
                    tasks = self.recv_from_child(pid,timeout=0.5)
                    if tasks is not None:
                        for task_id,task_info in tasks.items():
                            if task_id not in self.children_tasks[pid].keys():
                                self.logger.warning(f"{task_id} not in worker {pid}")
                            else:
                                self.my_tasks[task_id].update(task_info)
                else:
                    if msg is not None:
                        for k,v in msg.items():
                            self.progress_info[k][pid] = v
                    else:
                        self.logger.warning(f"No message received from worker {pid}")
            ##report status
            self.logger.info("sending message to parent")
            self.report_status()
            if ndone == self.n_children:
                for p in processes:
                    p.join()
                break
        self.send_to_parent(0,"DONE")
        self.send_to_parent(0,self.my_tasks)
        self.logger.info("Done running all tasks")



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