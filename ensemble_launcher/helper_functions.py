import os
import socket
from typing import Dict, List, Optional, Any

def create_task_info(
    task_id: str,
    cmd_template: str,
    system: str,
    num_nodes: int = 1,
    num_processes_per_node: int = 1,
    num_gpus_per_process: int = 0,
    gpu_affinity: Optional[List[str]] = None,
    cpu_affinity: Optional[List[int]] = None,
    run_dir: Optional[str] = None,
    launch_dir: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    status: str = "ready",
    io: bool = True,
    launcher_options: Optional[Dict[str, Any]] = None,
    timeout: Optional[float] = None,
    pre_launch_cmd: Optional[str] = None,
    gpu_affinity_file: Optional[str] = None,
    mpi_rankfile: Optional[str] = None,
    log_file: Optional[str] = None,
    err_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Helper function to construct a task_info dictionary with common parameters.
    
    Args:
        task_id: Unique identifier for the task
        cmd_template: Command template to execute
        system: System name
        num_nodes: Number of nodes for the task (default: 1)
        num_processes_per_node: Number of processes per node (default: 1)
        num_gpus_per_process: Number of GPUs per process (default: 0)
        gpu_affinity: List of GPU affinities (default: None)
        cpu_affinity: List of CPU affinities (default: None)
        run_dir: Directory where task will run (default: current working directory + "run_dir")
        launch_dir: Directory from which task is launched (default: current working directory)
        env: Environment variables dictionary (default: None)
        status: Task status (default: "ready")
        io: Whether to enable I/O (default: True)
        launcher_options: MPI launcher options dictionary (default: None)
        timeout: Task timeout in seconds (default: None)
        pre_launch_cmd: Command to run before the main command (default: None)
    
    Returns:
        Dictionary containing task information
    """
    
    if run_dir is None:
        run_dir = os.path.join(os.getcwd(), "run_dir")
    
    if launch_dir is None:
        launch_dir = os.getcwd()
    
    if mpi_rankfile is None:
        mpi_rankfile = os.path.join(run_dir, "rankfile.txt")

    if gpu_affinity_file is None:
        gpu_affinity_file = os.path.join(run_dir, "gpu_affinity.sh")

    if io:
        log_file = log_file if log_file else os.path.join(run_dir, f"worker_{socket.gethostname()}.log")
        err_file = err_file if err_file else os.path.join(run_dir, f"worker_{socket.gethostname()}.err")

    task_info = {
        "id": task_id,
        "num_nodes": num_nodes,
        "num_processes_per_node": num_processes_per_node,
        "num_gpus_per_process": num_gpus_per_process,
        "cmd_template": cmd_template,
        "run_dir": run_dir,
        "launch_dir": launch_dir,
        "env": env,
        "status": status,
        "system": system,
        "io": io,
        "log_file": log_file,
        "err_file": err_file,
        "gpu_affinity_file": gpu_affinity_file,
        "mpi_rankfile": mpi_rankfile
    }
    
    # Add optional fields if provided
    if gpu_affinity is not None:
        task_info["gpu_affinity"] = gpu_affinity
    
    if cpu_affinity is not None:
        task_info["cpu_affinity"] = cpu_affinity
    
    if launcher_options is not None:
        task_info["launcher_options"] = launcher_options
    
    if timeout is not None:
        task_info["timeout"] = timeout
    
    if pre_launch_cmd is not None:
        task_info["pre_launch_cmd"] = pre_launch_cmd
    
    return task_info