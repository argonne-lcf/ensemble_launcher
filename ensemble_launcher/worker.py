import os, stat, sys
import subprocess
import time
import copy
import multiprocessing as mp
import socket
import logging
import shutil
import pickle
import numpy as np
from .helper_functions import *
from .Node import Node

def _execute_task(task_info:dict, result_queue:mp.Queue) -> dict:
    """Function to execute in process pool - takes a dictionary with all task parameters"""
    env = os.environ.copy()
    env.update(task_info["env"])
    if task_info["io"]:
        os.makedirs(task_info["run_dir"],exist_ok=True)
    cmd = task_info["cmd"]
    log_file = task_info["log_file"]
    err_file = task_info["err_file"]
    launch_dir = task_info["launch_dir"]
    io = task_info["io"]
    timeout = task_info.get("timeout", None)
    start_time = time.time()
    task_info["start_time"] = start_time
    stdout = open(log_file, "w") if io else subprocess.DEVNULL
    stderr = open(err_file, "w") if io else subprocess.DEVNULL
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=stdout,
            stderr=stderr,
            stdin=subprocess.DEVNULL,
            cwd=launch_dir,
            env=env,
            timeout=timeout,
            close_fds=True
        )

        status = "finished" if result.returncode == 0 else "failed"
        task_info["end_time"] = time.time()
        task_info["status"] = status
        task_info["execution_time"] = time.time() - start_time
        task_info["returncode"] = result.returncode
        task_info["error"] = None if status == "finished" else f"Command failed with return code {result.returncode}"
    except Exception as e:
        task_info["status"] = "failed"
        task_info["end_time"] = time.time()
        task_info["execution_time"] = time.time() - start_time
        task_info["error"] = str(e)
    if io:
        if hasattr(stdout, 'close'): stdout.close()
        if hasattr(stderr, 'close'): stderr.close()
    result_queue.put(task_info)
    return task_info

class worker(Node):
    def __init__(self,
                worker_id:str,
                my_tasks:dict,
                my_nodes:list,
                sys_info:dict,
                comm_config:dict={"comm_layer":"multiprocessing"},
                launcher_config:dict={"mode":"mpi"},
                update_interval:int=None,
                logging_level=logging.INFO,
                heartbeat_interval:int=1):
        super().__init__(worker_id,
                         my_tasks,
                         my_nodes,
                         sys_info,
                         {"comm_layer":comm_config["comm_layer"],"role":"child"},
                         logger=False,
                         logging_level=logging_level,
                         update_interval=update_interval,
                         heartbeat_interval=heartbeat_interval)

        assert "name" in sys_info and "ncores_per_node" in sys_info and "ngpus_per_node" in sys_info
        
        # Store launcher configuration
        self.launcher_config = launcher_config
        ##resource info
        self.free_cores_per_node = {node:list(range(self.sys_info["ncores_per_node"])) 
                                                     for node in self.my_nodes}
        #
        if self.sys_info["name"] == "aurora":
            if os.getenv("ZE_FLAT_DEVICE_HIERARCHY", "COMPOSITE") == "FLAT":
                self.free_gpus_per_node = {node:[f"{i}" for i in range(self.sys_info["ngpus_per_node"])] 
                                                              for node in self.my_nodes}
            else:
                self.free_gpus_per_node = {node:["{}.{}".format(*list(np.unravel_index(i,(6,2)))) 
                                                              for i in range(self.sys_info["ngpus_per_node"])] 
                                                              for node in self.my_nodes}

            
        else:
            self.free_gpus_per_node = {node:list(range(self.sys_info["ngpus_per_node"])) for node in self.my_nodes}

        ### Mode determination from launcher_config
        self.mode = self.launcher_config.get("mode", "mpi")
        # Validate mode
        if self.mode not in ["mpi", "high throughput"]:
            if self.logger:
                self.logger.warning(f"Invalid mode '{self.mode}', defaulting to 'mpi'")
            self.mode = "mpi"

        # Ensure all tasks are compatible with the mode
        if self.mode == "high throughput":
            assert all([task_info["num_nodes"] == 1 and task_info["num_processes_per_node"] == 1
                         for task_info in self.my_tasks.values()]), "All tasks must be single-node, single-process"

        self.futures = {}
        self.process_pool = None
        self.result_queue = None
        self.manager = None
        ###
        self.tmp_dir = f"/tmp/worker-{worker_id}"
        

    def get_running_tasks(self) -> list:
        running_tasks = []
        for task_id, task_info in self.my_tasks.items():
            if task_info["status"] == "running":
                running_tasks.append(task_id)
        return running_tasks

    def get_failed_tasks(self) -> list:
        failed_tasks = []
        for task_id, task_info in self.my_tasks.items():
            if task_info["status"] == "failed":
                failed_tasks.append(task_id)
        return failed_tasks

    def get_ready_tasks(self) -> list:
        ready_tasks = []
        for task_id, task_info in self.my_tasks.items():
            if task_info["status"] == "ready":
                ready_tasks.append(task_id)
        return ready_tasks

    def get_finished_tasks(self) -> list:
        finished_tasks = []
        for task_id, task_info in self.my_tasks.items():
            if task_info["status"] == "finished":
                finished_tasks.append(task_id)
        return finished_tasks
    
    def get_pending_tasks(self) -> list:
        pending_tasks = []
        for task_id, task_info in self.my_tasks.items():
            if task_info["status"] in ["ready", "running"]:
                pending_tasks.append(task_id)
        return pending_tasks

    @property
    def my_free_nodes(self):
        my_free_nodes = []
        for node, cores in self.free_cores_per_node.items():
            if len(cores) != 0:
                my_free_nodes.append(node)
        return my_free_nodes

    @property
    def my_busy_nodes(self):
        my_busy_nodes = []
        for node, cores in self.free_cores_per_node.items():
            if len(cores) == 0:
                my_busy_nodes.append(node)
        return my_busy_nodes
    
    def assign_task_nodes(self,task:dict) -> list:
        assigned_nodes = []
        assigned_cores = {}
        assigned_gpus = {}
        j = 0
        while True:
            if len(assigned_nodes) == task["num_nodes"] or \
               len(self.my_free_nodes) == 0 or \
               j >= len(self.my_free_nodes):
                break

            node = self.my_free_nodes[j]
            visit_node = len(self.free_cores_per_node[node]) >= task["num_processes_per_node"]
            if task.get("num_gpus_per_process",0) > 0:
                visit_node = visit_node and len(self.free_gpus_per_node[node]) >= task["num_processes_per_node"]*task["num_gpus_per_process"]

            if visit_node:
                assigned_cores[node] = []
                ##smply pop the cores from the list at the start
                for i in range(task["num_processes_per_node"]):
                    assigned_cores[node].append(self.free_cores_per_node[node].pop(0))
                
                if task.get("num_gpus_per_process",0) > 0:
                    assigned_gpus[node] = []
                    for i in range(task["num_gpus_per_process"]*task["num_processes_per_node"]):
                        assigned_gpus[node].append(self.free_gpus_per_node[node].pop(0))
                assigned_nodes.append(node)

            j += 1
            if len(self.free_cores_per_node[node]) == 0:
                j -= 1

        if len(assigned_nodes) < task["num_nodes"]:
            for node in assigned_nodes:
                ##append the cores to the end
                self.free_cores_per_node[node].extend(assigned_cores[node])
                self.free_cores_per_node[node] = \
                    sorted(self.free_cores_per_node[node])
                del assigned_cores[node]

                if task.get("num_gpus_per_process",0) > 0:
                    self.free_gpus_per_node[node].extend(assigned_gpus[node])
                    self.free_gpus_per_node[node] = \
                        sorted(self.free_gpus_per_node[node])
                    del assigned_gpus[node]

            assigned_nodes = []
            
        return assigned_nodes,assigned_cores,assigned_gpus
    
    def free_task_nodes(self,task_info:dict) -> None:
        for node in task_info["assigned_nodes"]:
            ##append the cores to the end
            self.free_cores_per_node[node].extend(task_info["assigned_cores"][node])
            self.free_cores_per_node[node] = \
                    sorted(self.free_cores_per_node[node])

            if task_info.get("num_gpus_per_process",0) > 0:
                self.free_gpus_per_node[node].extend(task_info["assigned_gpus"][node])
                self.free_gpus_per_node[node] = \
                        sorted(self.free_gpus_per_node[node])
                if self.logger:
                    self.logger.info(f"Freed {task_info['assigned_gpus'][node]} GPUs from node {node} for task {task_info['id']}")
        return

    """
    Function modifies the task template based on the system and input values
    """
    def build_mpi_launcher_cmd(self, task_info:dict, cpu_bind = True) -> tuple:
        env = {}
        if task_info["system"] == "local":
            launcher_cmd = f"mpirun -np {task_info['num_nodes'] * task_info['num_processes_per_node']} "
            if "num_gpus_per_process" in task_info.keys():
                raise NotImplementedError("Unknown machine for scheduling tasks on GPUs, sorry!")
        else:
            launcher_options = task_info.get("launcher_options", {})
            launcher_cmd = "mpirun "
            if "np" in launcher_options:
                if launcher_options["np"] != task_info["num_nodes"] * task_info["num_processes_per_node"]:
                    raise ValueError("Mismatch in 'np' value between launcher_options and calculated options")
            launcher_cmd += f"-np {task_info['num_nodes']*task_info['num_processes_per_node']} "
            
            if "ppn" in launcher_options:
                if launcher_options["ppn"] != task_info["num_processes_per_node"]:
                    raise ValueError("Mismatch in 'ppn' value between launcher_options and calculated options")
            launcher_cmd += f"-ppn {task_info['num_processes_per_node']} "
            launcher_cmd += f"--hosts {','.join(task_info['assigned_nodes'])} "
            
            if cpu_bind:
                ###check for launcher options
                if "cpu-bind" in launcher_options:
                    if launcher_options["cpu-bind"] == "depth":
                        if "depth" not in launcher_options:
                            raise ValueError("'cpu-bind' value is 'depth' but 'depth' is not in launcher_options")
                        launcher_cmd += f"--depth={launcher_options['depth']} --cpu-bind={launcher_options['cpu-bind']} "
                    else:
                        launcher_cmd += f"--cpu-bind {launcher_options['cpu-bind']} "
                else:
                    if self.sys_info["name"] == "aurora":
                        ctoi = {i: i+1 for i in range(51)}
                        ctoi.update({i: i+2 for i in range(51,102)})
                        ctoi.update({i: i+3 for i in range(102,153)})
                        ctoi.update({i: i+4 for i in range(153,204)})
                    else:
                        ctoi = {i: i for i in range(self.sys_info["ncores_per_node"])}
                    common_cpus = set.intersection(*[set(cores) for cores in task_info["assigned_cores"].values()])
                    use_common_cpus = list(common_cpus) == task_info["assigned_cores"][task_info["assigned_nodes"][0]]
                    if use_common_cpus:
                        if self.sys_info["name"] == "aurora":
                            cores = []
                            for i in task_info["assigned_cores"][task_info["assigned_nodes"][0]]:
                                cores.append(f"{ctoi[i]}")
                            cores = ":".join(cores)
                        else:
                            cores = ":".join(map(str, task_info["assigned_cores"][task_info["assigned_nodes"][0]]))
                        launcher_cmd += f"--cpu-bind list:{cores} "
                    else:
                        ###user rankfile option
                        rankfile_path = task_info["mpi_rankfile"]
                        with open(rankfile_path, "w") as rankfile:
                            rank = 0
                            for node in task_info["assigned_nodes"]:
                                for core_set in task_info["assigned_cores"][node]:
                                    rankfile.write(f"rank {rank}={node} slot={ctoi[core_set]}\n")
                                    rank += 1
                        if self.logger: self.logger.warning(f"Over subscribing cores")
                        # launcher_cmd += f"--rankfile {rankfile_path} "
            
            ##append all other launcher options that are not checked above
            for key, value in launcher_options.items():
                if key != "np" and key != "ppn" and key != "hosts" and key != "cpu-bind" and key != "depth":
                    launcher_cmd += f"--{key} {value} "
            
            if "num_gpus_per_process" in task_info.keys():
                if task_info["system"] == "aurora":
                    common_gpus = set.intersection(*[set(gpus) for gpus in task_info["assigned_gpus"].values()])
                    use_common_gpus = sorted(list(common_gpus)) == sorted(task_info["assigned_gpus"][task_info["assigned_nodes"][0]])
                    if use_common_gpus:
                        if task_info["num_nodes"] == 1 and task_info["num_processes_per_node"] == 1:
                            ##here you don't need any compilcated bash script 
                            # you can just getaway with a simple environment variable
                            # launcher_cmd += f"ZE_AFFINITY_MASK={task_info['assigned_gpus'][task_info['assigned_nodes'][0]][0]} "
                            env.update({"ZE_AFFINITY_MASK": ",".join(task_info['assigned_gpus'][task_info['assigned_nodes'][0]])})
                        else:
                            bash_script = gen_affinity_bash_script_aurora_1(task_info["num_gpus_per_process"])
                            os.makedirs(task_info["run_dir"], exist_ok=True)
                            fname = task_info["gpu_affinity_file"]
                            if not os.path.exists(fname):
                                with open(fname, "w") as f:
                                    f.write(bash_script)
                                st = os.stat(fname)
                                os.chmod(fname,st.st_mode | stat.S_IEXEC)
                            launcher_cmd += f"{fname} "
                            ##set environment variables
                            env.update({"AVAILABLE_GPUS": ",".join(task_info["assigned_gpus"][task_info["assigned_nodes"][0]])})
                    else:
                        bash_script = gen_affinity_bash_script_aurora_2(task_info["num_gpus_per_process"])
                        os.makedirs(task_info["run_dir"], exist_ok=True)
                        fname = task_info["gpu_affinity_file"]
                        if not os.path.exists(fname):
                            with open(fname, "w") as f:
                                f.write(bash_script)
                            st = os.stat(fname)
                            os.chmod(fname,st.st_mode | stat.S_IEXEC)
                        launcher_cmd += f"{fname} "
                        ##Here you need to set the environment variables for each node
                        for node in task_info["assigned_nodes"]:
                            env.update({f"AVAILABLE_GPUS_{node}": ",".join(task_info["assigned_gpus"][node])})
                else:
                    raise NotImplementedError("Unknown machine for scheduling tasks on GPUs, sorry!")

        return launcher_cmd, env
    
    def build_bash_cmd(self, task_info:dict):
        launcher_cmd = ""
        env = {}
        if task_info.get("num_gpus_per_process", 0) > 0:
            assert task_info["system"] == "aurora", "High throughput tasks with GPUs are only supported on Aurora currently."
            env.update({"ZE_AFFINITY_MASK": ",".join(task_info['assigned_gpus'][task_info['assigned_nodes'][0]])})
        return launcher_cmd, env


    """
    Build the launch cmd based on cmd_template from the user
    """
    def build_task_cmd(self,task_info:dict) -> tuple:

        # Use the mode from launcher_config
        if self.mode == "high throughput":
            launcher_cmd, env = self.build_bash_cmd(task_info)
        else:  # mode == "mpi" or any other value defaults to mpi
            launcher_cmd, env = self.build_mpi_launcher_cmd(task_info, cpu_bind = "cpu-bind" not in task_info["cmd_template"])
        open_braces = [i for i, char in enumerate(task_info["cmd_template"]) if char == "{"]
        close_braces = [i for i, char in enumerate(task_info["cmd_template"]) if char == "}"]
        placeholders = [task_info["cmd_template"][open_braces[i] + 1:close_braces[i]] for i in range(len(open_braces))]
        ##put the options
        cmd = task_info["cmd_template"].format(**{key: task_info[key] for key in placeholders})
        ##pre launch cmd
        pre_cmd = task_info.get("pre_launch_cmd", None)
        if pre_cmd is not None:
            return pre_cmd + ";"+launcher_cmd+cmd, env
        else:
            return launcher_cmd+cmd, env
    
    def launch_task(self, task_info:dict):
        ##check if run dir exists
        env = os.environ.copy()
        env.update(task_info["env"])
        if task_info["io"]:
            os.makedirs(task_info["run_dir"],exist_ok=True)
            if self.logger: self.logger.info(f"Creating run dir {task_info['run_dir']}")
        if self.logger: self.logger.info(f"launching task {task_info['id']} with {task_info['cmd']}")
        env["TMPDIR"] = self.tmp_dir

        p = subprocess.Popen(task_info["cmd"],
                             executable="/bin/bash",
                             shell=True,
                             stdout=open(task_info["log_file"],"w") if task_info["io"] else subprocess.DEVNULL,
                             stderr=open(task_info["err_file"],"w") if task_info["io"] else subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL,
                             cwd=task_info["launch_dir"],
                             env=env,
                             close_fds = True)
        return p

    def launch_high_throughput_tasks(self,oversubscribe_gpus:bool=True) -> int:
        ready_tasks = self.get_ready_tasks()
        running_tasks = self.get_running_tasks()
        
        # Limit concurrent tasks to CPU count
        max_concurrent = mp.cpu_count()
        slots_available = max_concurrent - len(running_tasks)
        
        if slots_available <= 0:
            return 0
        
        # Only launch up to available slots
        tasks_to_launch = ready_tasks[:slots_available]
        task_infos = []
        
        for idx,task_id in enumerate(tasks_to_launch):
            task_info = self.my_tasks[task_id]
            ##assign nodes, cores and gpus
            assigned_node = self.my_free_nodes[idx % len(self.my_free_nodes)]
            task_info["assigned_nodes"] = [assigned_node]
            task_info["assigned_cores"] = {assigned_node:[]}
            ##assign gpus
            if task_info.get("num_gpus_per_process", 0) > 0:
                assigned_gpus = []
                if not oversubscribe_gpus:
                    while len(self.free_gpus_per_node[assigned_node]) > 0 and len(assigned_gpus) < task_info["num_gpus_per_process"]:
                        assigned_gpus.append(self.free_gpus_per_node[assigned_node].pop(0))
                    if len(assigned_gpus) < task_info["num_gpus_per_process"]:
                        continue
                else:
                    while len(assigned_gpus) < task_info["num_gpus_per_process"]:
                        assigned_gpus.append(self.free_gpus_per_node[assigned_node].pop(0))
                        self.free_gpus_per_node[assigned_node].append(assigned_gpus[-1])
                task_info["assigned_gpus"] = {assigned_node:assigned_gpus}
                if self.logger: self.logger.info(f"Assigned {assigned_gpus} GPUs to task {task_id} on node {assigned_node}")
            cmd,env = self.build_task_cmd(task_info)
            task_info["cmd"] = cmd
            task_info["env"].update(env)
            task_info["env"]["TMPDIR"] = self.tmp_dir

            task_info["status"] = "running"
            task_infos.append(task_info)

        if len(task_infos) > 0:
            self.process_pool.starmap_async(_execute_task, 
            [(task_info, self.result_queue) for task_info in task_infos],)

        return len(task_infos)

    """
    function checks if the requested resources are available on the node
    """
    def check_task_validity(self,task_info:dict)->bool:
        return task_info["num_processes_per_node"] <= self.sys_info["ncores_per_node"] \
            and task_info["num_processes_per_node"]*task_info.get("num_gpus_per_process",0) <= self.sys_info["ngpus_per_node"]

    """
    function loops through all the ready tasks and launches all the tasks that can be launched
    WARNING: When running in parallel, this function doesn't check process given by my_pid is managing the ensemble
    """
    def launch_ready_tasks(self) -> int:
        launched_tasks = 0
        ready_tasks = self.get_ready_tasks()
        for task_id in ready_tasks:
            task_info = self.my_tasks[task_id]
            ##idiot check
            valid_task = self.check_task_validity(task_info)
            if not valid_task:
                task_info["status"] = "failed"
                continue
            assigned_nodes,assigned_cores,assigned_gpus = \
                self.assign_task_nodes(task_info)
            if len(assigned_nodes) == 0:
                if len(self.my_free_nodes) == 0:
                    break
                else:
                    continue
            task_info.update({"assigned_nodes":assigned_nodes,
                            "assigned_cores":assigned_cores,
                                "assigned_gpus":assigned_gpus})
            cmd,env = self.build_task_cmd(task_info)
            copy_env = copy.deepcopy(task_info["env"])
            env.update(copy_env)
            task_info.update({"cmd":cmd,"env":env,"pre_launch_time":time.time()})
            p = self.launch_task(task_info)
            task_info.update({"process": p,
                              "start_time": time.time(),
                              "status": "running",
                              "pre_report_time": time.time()})
            task_info.update({"post_report_time": time.time()})
            launched_tasks += 1
        return launched_tasks
    
    def poll_running_tasks(self,my_pid:int=0) -> None:
        running_tasks = self.get_running_tasks()
        for task_id in running_tasks:
            task = self.my_tasks[task_id]
            popen_proc = task["process"]
            if popen_proc.poll() is not None:
                if popen_proc.returncode == 0:
                    status = "finished"
                else:
                    status = "failed"
                out,err = popen_proc.communicate()
                self.free_task_nodes(task)
                task.update({"end_time":time.time(),
                             "status":status,
                             "process":None,
                            "assigned_nodes":[]})
            else:
                if "timeout" in task and \
                   time.time() - task["start_time"] > task["timeout"]:
                    if self.logger: self.logger.warning(f"Task {task_id} timed out after {task['timeout']} seconds")
                    popen_proc.kill()
                    popen_proc.wait(timeout=10)
                    self.free_task_nodes(task)
                    task.update({"end_time":time.time(),
                                 "status":"failed",
                                 "process":None,
                                "assigned_nodes":[]})
        return None
    
    def check_for_messages(self) -> bool:
        """
        Check for messages from parent and handle them appropriately.
        Returns True if a kill signal was received.
        """
        kill_signal = False
        if self.update_interval is not None:
            msg = self.recv_from_parent(0, timeout=1)
            if isinstance(msg, tuple):
                if msg[0] == "KILL":
                    kill_signal = True
                elif msg[0] == "SYNC":
                    self.send_to_parent(0, "SYNCED")
                    msg = self.blocking_recv_from_parent(0)
                    if isinstance(msg, tuple) and msg[0] == "UPDATE":
                        self.commit_task_update(msg[1], msg[2])
                        self.send_to_parent(0, "UPDATE SUCCESSFUL")
            elif msg and self.logger:
                self.logger.debug(f"Received unknown msg from parent: {msg}")
        return kill_signal
    
    def setup_environment(self, logger=False):
        """Setup the environment for task execution."""
        os.environ.update(self.parent_env)
        if logger: 
            self.configure_logger(self.logging_level)
            if self.logger: self.logger.info(f"Running on {socket.gethostname()}")

        if self.sys_info["name"] == "aurora":
            if  "ZE_FLAT_DEVICE_HIERARCHY" not in os.environ:
                os.environ["ZE_FLAT_DEVICE_HIERARCHY"] = "COMPOSITE"
            else:
                if os.environ["ZE_FLAT_DEVICE_HIERARCHY"] != "COMPOSITE" and self.logger:
                    self.logger.warning("ZE_FLAT_DEVICE_HIERARCHY is not set to COMPOSITE, this may cause issues")
        if self.comm_config["comm_layer"] == "zmq":
            self.setup_zmq_sockets()
        self.last_update_time = time.time()
        os.makedirs(self.tmp_dir, exist_ok=True)
        ##this is needed so that pool can see all the cores. In addition to the one used by mpirun
        if self.mode == "high throughput":
            os.sched_setaffinity(0, range(mp.cpu_count()))

    def exit_sequence(self):
        """Handle cleanup and send completion signal."""
        self.report_status()
        self.cleanup_resources()
        self.send_to_parent(0, "DONE")

    def process_high_throughput_tasks(self,oversubscribe_gpus:bool=False):
        """Process tasks in high throughput mode."""
        tstart = time.time()
        ntasks = self.launch_high_throughput_tasks(oversubscribe_gpus=oversubscribe_gpus)
        tend = time.time()
        if ntasks > 0 and self.logger:
            self.logger.debug(f"Launched {ntasks} tasks in {tend - tstart:.2f} seconds.")
        # Poll running tasks
        while not self.result_queue.empty():
            update_task_info = self.result_queue.get()
            task_id = update_task_info["id"]
            self.my_tasks[task_id].update(update_task_info)
            if not oversubscribe_gpus:
                self.free_task_nodes(self.my_tasks[task_id])

    def process_standard_tasks(self):
        """Process tasks in standard mode."""
        self.launch_ready_tasks()
        self.poll_running_tasks()

    def run_tasks(self, logger=False) -> None:
        # Setup
        self.setup_environment(logger)
        
        # Use manager for high throughput mode only
        if self.mode == "high throughput":
            with mp.Manager() as manager:
                self.manager = manager
                self.result_queue = manager.Queue()
                self.process_pool = manager.Pool(processes=mp.cpu_count())
                if self.logger: self.logger.info("High throughput mode enabled. Launching all tasks at once.")

                # Main execution loop
                while True:
                    # Process tasks
                    self.process_high_throughput_tasks(oversubscribe_gpus=self.launcher_config.get("oversubscribe_gpus", False))
                    # Report status periodically
                    if time.time() - self.last_update_time > self.heartbeat_interval:
                        self.report_status()
                        self.last_update_time = time.time()
                    # Check for parent messages
                    kill_signal = self.check_for_messages()
                    # Exit conditions
                    if kill_signal or not self.get_pending_tasks():
                        self.exit_sequence()
                        break
                    time.sleep(0.1)
        else: 
            # Main execution loop - standard mode
            while True:
                # Process tasks
                self.process_standard_tasks()
                # Report status periodically
                if time.time() - self.last_update_time > self.heartbeat_interval:
                    self.report_status()
                    self.last_update_time = time.time()
                # Check for parent messages
                kill_signal = self.check_for_messages()
                # Exit conditions
                if kill_signal or not self.get_pending_tasks():
                    self.exit_sequence()
                    break
                time.sleep(0.1)

    def delete_tasks(self,task_infos:dict) -> None:
        task_ids = list(task_infos.keys())
        for task_id in task_ids:
            if self.my_tasks[task_id]["status"] == "running":
                p = self.my_tasks[task_id]["process"]
                p.kill()
                p.wait(timeout=10)
                self.free_task_nodes(self.my_tasks[task_id])
            del self.my_tasks[task_id]

    def add_tasks(self,new_tasks:dict):
        self.my_tasks.update(new_tasks)
    
    def commit_task_update(self,deleted_tasks:dict,new_tasks:dict):
        self.delete_tasks({task_id:self.my_tasks[task_id] for task_id in deleted_tasks})
        if self.logger: self.logger.info(f"Got {len(new_tasks)} new task!!")
        self.add_tasks(new_tasks)
        
    def report_status(self):
        num_failed = len(self.get_failed_tasks())
        num_finished = len(self.get_finished_tasks())
        num_running = len(self.get_running_tasks())
        num_ready = len(self.get_ready_tasks())
        nfree_cores = sum(len(cores) for cores in self.free_cores_per_node.values())
        nfree_gpus = sum(len(gpus) for gpus in self.free_gpus_per_node.values())
        info = {"nfailed_tasks": num_failed, 
                "nfinished_tasks": num_finished, 
                "nrunning_tasks": num_running, 
                "nready_tasks": num_ready,
                "nfree_cores": nfree_cores,
                "nfree_gpus": nfree_gpus}
        status_str = ",".join([f"{k}:{v}" for k,v in info.items()])
        if self.logger: self.logger.info(status_str)
        self.send_to_parent(0, info)
        return
    
    # cleanup_resources
    def cleanup_resources(self):
        if self.logger: self.logger.info("Cleaning up resources...")
        ##kill the dangling processes
        for task_id, task_info in self.my_tasks.items():
            if "process" in task_info and task_info["process"]:
                if task_info["process"].poll() is None:
                    if self.logger: self.logger.info(f"Process of {task_id} is still running. So, killing it...")
                    task_info["process"].kill()
                    task_info["process"].wait(timeout=10)
                    
        if self.process_pool is not None:
            if self.logger: self.logger.info("Closing process pool...")
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None

        if os.path.exists(self.tmp_dir):
            try:
                if self.logger: self.logger.info(f"Deleting tmpdir")
                shutil.rmtree(self.tmp_dir)
                if self.logger: self.logger.info(f"Done deleting tmpdir")
            except Exception as e:
                if self.logger: self.logger.error(f"Failed to delete tmp_dir {self.tmp_dir}: {e}")

if __name__ == "__main__":
    worker_obj_file = sys.argv[1]
    logger = int(sys.argv[2]) == 1

    with open(worker_obj_file, "rb") as f:
        worker_obj = pickle.load(f)
    if logger:
        worker_obj.configure_logger(worker_obj.logging_level)
        worker_obj.logger.info(f"Worker {worker_obj.node_id} started")
    worker_obj.run_tasks(False)
