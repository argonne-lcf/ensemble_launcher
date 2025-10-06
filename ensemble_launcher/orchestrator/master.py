from .worker import Worker
from .node import Node
from ensemble_launcher.executors import executor_registry, MPIExecutor, Executor
from ensemble_launcher.scheduler import WorkerScheduler
from ensemble_launcher.scheduler.resource import LocalClusterResource, JobResource, NodeResourceList
from ensemble_launcher.config import SystemConfig, LauncherConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.comm import ZMQComm, MPComm, NodeInfo, Comm
from ensemble_launcher.comm.messages import Status, Result 

import logging
from itertools import accumulate
from typing import Optional, List, Dict, Any
import os
import cloudpickle
import time

logger = logging.getLogger(__name__)

class Master(Node):
    def __init__(self,
                id:str,
                config:LauncherConfig,
                system_info: SystemConfig,
                Nodes:List[str],
                tasks: Dict[str, Task],
                parent: Optional[NodeInfo] = None,
                children: Optional[Dict[str, NodeInfo]] = None,
                parent_comm: Optional[Comm] = None):
        super().__init__(id, parent=parent, children=children)
        self._tasks = tasks
        self._config = config
        self._parent_comm = parent_comm

        ##lazily created in run
        self._executor = None
        self._comm = None

        ##create a scheduler. maybe this can be removed??
        self._scheduler = WorkerScheduler(cluster=LocalClusterResource(Nodes,system_info))

        ##maps
        self._children_exec_ids: Dict[str, str] = {}
        self._child_assignment: Dict[str, Dict] = {}
        self._results: Dict[str, Any] = {}


    @property
    def nodes(self):
        return self._scheduler.cluster.nodes
    
    @property
    def parent_comm(self):
        return self._parent_comm
    
    @parent_comm.setter
    def parent_comm(self, value: Comm):
        self._parent_comm = value
    
    @property
    def comm(self):
        return self._comm

    def _create_comm(self):
        if self._config.comm_name == "multiprocessing":
            self._comm = MPComm(self.info(),self.parent_comm if self.parent_comm else None)
        elif self._config.comm_name == "zmq":
            ##sending parent address here because all zmq objects are not picklable
            self._comm = ZMQComm(self.info(),parent_address=self.parent_comm.my_address if self.parent_comm else None)
        else:
            raise ValueError(f"Unsupported comm {self._config.comm_name}")

    def _assign_children(self,
                        tasks:Dict[str, Task],
                        nodes:list):
        """
        Assign tasks to workers based on resource requirements.
        """
        
        # Step 1: Sort tasks by decreasing number of nodes required
        sorted_tasks = sorted(tasks.items(), key=lambda x: x[1].nnodes, reverse=True)
        
        ###remove tasks that have num nodes > total nodes of this master
        removed_tasks = []
        while sorted_tasks and sorted_tasks[0][1].nnodes > len(nodes):
            removed_tasks.append((sorted_tasks.pop(0))[0])
        
        if len(removed_tasks) > 0:
            logger.warning(f"Can't schedule {','.join(removed_tasks)}!")

        # Step 1: Calculate cumulative sum of nodes required for tasks
        cum_sum_nnodes = list(accumulate(task.nnodes for _, task in sorted_tasks))
        
        # Step 2: Determine the number of workers
        nworkers = 0
        while nworkers < len(cum_sum_nnodes) and cum_sum_nnodes[nworkers] <= len(nodes):
            nworkers += 1
        
        # Step 3: Initialize worker assignments for the first set of tasks
        nnodes = [task.nnodes for task_id,task in sorted_tasks[:nworkers]]
        if sum(nnodes) < len(nodes):
            nremaining = len(nodes) - sum(nnodes)
            for i in range(nremaining):
                nnodes[i%nworkers] += 1
        nnodes = list(accumulate(nnodes))
        children_assignments = {
            wid: {
            "nodes": nodes[:nnodes[wid]] if wid == 0 else nodes[nnodes[wid-1]:nnodes[wid]],
            "task_ids": [task_id]        # List of task IDs assigned to the worker
            }
            for wid, (task_id, task) in enumerate(sorted_tasks[:nworkers])
        }

        # Step 4: Assign remaining tasks to workers in a round-robin fashion
        for i, (task_id, task) in enumerate(sorted_tasks[nworkers:]):
            # Determine the worker ID in a round-robin manner
            worker_id = i % nworkers
            # Add the task ID to the worker's task list
            children_assignments[worker_id]["task_ids"].append(task_id)

        return children_assignments, removed_tasks

    def _create_children(self) -> Dict[str, Node]:
        assignments,remove_tasks = self._assign_children(self._tasks, self._scheduler.cluster.nodes)
        self._child_assignment = assignments

        children = {}
        if self.level + 1 == self._config.nlevels:
            for wid,alloc in assignments.items():
                child_id = self.node_id+f".w{wid}"
                #create a worker
                children[child_id] = \
                    Worker(
                        child_id,
                        config=self._config,
                        system_info=self._scheduler.cluster.system_info,
                        Nodes=alloc["nodes"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]},
                        parent=None
                    )
        else:
            #create a master again
            for wid,alloc in assignments.items():
                child_id = self.node_id+f".m{wid}"
                #create a worker
                children[child_id] = \
                    Master(
                        child_id,
                        config=self._config,
                        system_info=self._scheduler.cluster.system_info,
                        Nodes=alloc["nodes"],
                        tasks={task_id: self._tasks[task_id] for task_id in alloc["task_ids"]},
                        parent=None
                    )
        return children

    def _build_launch_cmd(self) -> str:
        ##save the state of the child obj        
        for child_name,child_obj in self.children.items():
                with open(os.path.join(os.getcwd(),".obj",f"{child_name}.pkl"),"wb") as f:
                    cloudpickle.dump(child_obj,f)

        if self.level + 1 == self._config.nlevels:
            cmd = f"python -m ensemble_launcher.orchestrator.worker"+\
                        f" {os.path.join(os.getcwd(),'.obj',f'{child_name}.pkl')}"
        else:   
            cmd = f"python -m ensemble_launcher.orchestrator.master"+\
                        f" {os.path.join(os.getcwd(),'.obj',f'{child_name}.pkl')}"
        return cmd
    
    def run(self):
        #create executor
        self._executor: Executor = executor_registry.create_executor(self._config.executor_name)

        ##create children
        children = self._create_children()
        logger.info(f"{self.node_id} Created {len(children)} children")

        #add children
        for child_id, child in children.items():
            self.add_child(child_id, child.info())
            child.set_parent(self.info())
        
        ##create comm: Need to do this after the setting the children to properly create pipes
        self._create_comm() ###This will only create picklable objects
        for child_id, child in children.items():
            child.parent_comm = self.comm
        
        for child_name,child_obj in children.items():
            child_nodes = child_obj.nodes
            req = JobResource(
                resources=[NodeResourceList(cpus=[1])], nodes=child_nodes[:1]
            )
            env = os.environ.copy()
            self._children_exec_ids[child_name] = self._executor.start(req, child_obj.run, env = env)
        
        ##lazy creation of non-pickable objects
        if self._config.comm_name == "zmq":
            self._comm.setup_zmq_sockets()
        
        ##send heart beat
        self._comm.send_heartbeat()

        success = []
        failure = []
        for child_id,exec_id in self._children_exec_ids.items():
            if child_id in success or child_id in failure:
                continue
            if self._executor.running(exec_id):
                logger.info(f"Waiting for child {child_id}.")
                self._comm.wait_for_child(child_id=child_id)
                success.append(child_id)
            else:
                if not self._executor.done(exec_id):
                    raise RuntimeError(f"Invalid state for {child_id}")
                else:
                    e = self._executor.exception(exec_id)
                    logger.error(f"{child_id} failed with exception {e}")
                    failure.append(child_id)

        if len(success) == len(self.children):
            logger.info(f"{self.node_id}: All children connected")
        else:
            logger.info(f"{self.node_id}: Children {failure} failed to start")
            self.stop()
            return

        self._monitor_children()
    
    def _monitor_children(self):
        next_report_time = time.time() + self._config.report_interval
        children_status = {}
        while True:
            done = []
            for child_id, exec_id in self._children_exec_ids.items():
                if child_id in done:
                    continue
                ##receive status updates
                status = self._comm.recv_message_from_child(Status,child_id=child_id,timeout=5.0)
                if status is not None:
                    children_status[child_id] = status

                if self._executor.done(exec_id):
                    done.append(child_id)
            
            ##send status to parent
            if time.time() > next_report_time:
                status = sum([s for s in children_status.values()])
                if self.parent:
                    self._comm.send_message_to_parent(status)
                else:
                    if isinstance(status,Status):
                        logger.info(f"Status: {status}")
                next_report_time = time.time() + self._config.report_interval

            if len(done) == len(self.children):
                logger.info(f"{self.node_id}: All children are done")
                break
    
    def results(self):
        """Collect results from all children workers."""
        results = {}
        for child_id in self.children.keys():
            results[child_id] = []

        done = set()
        expected_results = {
            child_id: len(self._child_assignment[wid]["task_ids"]) 
            for wid, child_id in enumerate(self.children.keys())
        }
        
        while len(done) < len(self.children):
            for child_id in self.children.keys():
                if child_id in done:
                    continue
                    
                # Check if we have all expected results for this child
                if len(results[child_id]) >= expected_results[child_id]:
                    done.add(child_id)
                    continue
                    
                result = self._comm.recv_message_from_child(Result, child_id, timeout=0.1)
                if result is not None:
                    results[child_id].append(result)
                    logger.debug(f"Received result from {child_id}: {len(results[child_id])}/{expected_results[child_id]}")
        
        logger.info(f"{self.node_id}: Collected all results from children")
        return results

    def stop(self):
        self._executor.shutdown()
        self._comm.close()