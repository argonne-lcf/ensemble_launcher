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
        ##for now I will just use n_workers = number of nodes
        self.n_workers = len(my_nodes)
        self.workers = []
        ##
        self.worker_nodes = self.split_nodes()
        self.worker_tasks = self.split_tasks()
        ##
        self.progress_info = {}
        self.progress_info["nrunning_tasks"] = [0 for i in range(self.n_workers)]
        self.progress_info["nready_tasks"] = [0 for i in range(self.n_workers)]
        self.progress_info["nfailed_tasks"] = [0 for i in range(self.n_workers)]
        self.progress_info["nfinished_tasks"] = [0 for i in range(self.n_workers)]
        self.progress_info["nfree_cores"] = [0 for i in range(self.n_workers)]
        self.progress_info["nfree_gpus"] = [0 for i in range(self.n_workers)]


    def split_nodes(self)->list:
        worker_nodes = []
        if len(self.my_nodes)<self.n_workers:
            raise ValueError("Total number of nodes < number of parallel task launchers! Please set nparallel = 1")
        nn = len(self.my_nodes)//self.n_workers
        for i in range(self.n_workers):
            worker_nodes.append(self.my_nodes[i*nn:(i+1)*nn])
        worker_nodes[-1].extend(self.my_nodes[(i+1)*nn:])
        return worker_nodes
    
    def split_tasks(self)->dict:
        nt_pw = len(self.my_tasks)//self.n_workers
        worker_tasks = {}
        for wid in range(self.n_workers-1):
            worker_tasks[wid] = {}
            for task_id in list(self.my_tasks.keys())[wid*nt_pw:(wid+1)*nt_pw]:
                worker_tasks[wid][task_id] = self.my_tasks[task_id]
        worker_tasks[self.n_workers - 1] = {
            task_id: self.my_tasks[task_id]
            for task_id in list(self.my_tasks.keys())[(self.n_workers - 1) * nt_pw:]
        }
        return worker_tasks

    def report_status(self):
        progress_info = {}
        for k,v in self.progress_info.items():
            progress_info[k] = sum(v)
        self.send_to_parent(0,progress_info)        

    def run_tasks(self,parent_pipe):
        self.configure_logger()
        self.add_parent(0,parent_pipe)
        self.logger.info("Started running tasks")
        if self.parallel_backend == "dragon":
            mp.set_start_method("dragon")
        ##create workers and corresponding processes
        processes = []
        policies = []
        for pid in range(self.n_workers):
            w = worker( f"worker_{pid}",
                        self.worker_tasks[pid],
                        self.worker_nodes[pid],
                        self.sys_info)
            ##connect master and children
            parent_conn, child_conn = mp.Pipe()
            self.add_child(pid,parent_conn)
            self.workers.append(w)
            ##
            if self.parallel_backend == "dragon":
                policies.append(dragon.infrastructure.policy.Policy(
                                        placement=dragon.infrastructure.policy.Policy.Placement.HOST_NAME,
                                        host_name=self.worker_nodes[pid][0]
                                    ))
                p = dragon.native.process.Process(target=w.run_tasks,args=(child_conn,),policy=policies[-1])
            else:
                p = mp.Process(target=w.run_tasks,args=(child_conn,))
            p.start()
            processes.append(p)
            for task_id,task_info in self.worker_tasks[pid].items():
                self.my_tasks[task_id].update({"status":"running"})
        self.logger.info("Done forking processes")
        ndone = 0
        done_workers = []
        
        while True:
            for pid in range(self.n_workers):
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
                            if task_id not in self.worker_tasks[pid].keys():
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
            if ndone == self.n_workers:
                for p in processes:
                    p.join()
                break
        self.send_to_parent(0,"DONE")
        self.send_to_parent(0,self.my_tasks)
        self.logger.info("Done running all tasks")
