import os
import socket
from typing import Dict, List, Optional, Any

"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use the same GPU
"""
def gen_affinity_bash_script_aurora_1(ngpus_per_process) -> str:
    bash_script = [
                      "#!/bin/bash",
                      "##get the free gpus from the environment variable",
                      r'IFS="," read -ra my_free_gpus <<< "$AVAILABLE_GPUS"',
                      "# Get the RankID from different launcher",
                      "if [[ -v MPI_LOCALRANKID ]]; then",
                      "   _MPI_RANKID=$MPI_LOCALRANKID ",
                      "elif [[ -v PALS_LOCAL_RANKID ]]; then",
                      "   _MPI_RANKID=$PALS_LOCAL_RANKID",
                      "fi",
                      "unset EnableWalkerPartition",
                      "export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
                      "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1",
                      "# Calculate the GPUs assigned to this rank",
                      f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                      f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                      r"echo $rank_gpus $_MPI_RANKID",
                      r"export ZE_AFFINITY_MASK=${rank_gpus}",
                      r"ulimit -c 0 # Until Aurora filesystem problems are fixed",
                      '"$@"'
                 ]
    return "\n".join(bash_script)


"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use different GPUs
"""
def gen_affinity_bash_script_aurora_2(ngpus_per_process) -> str:
   """
   the below bash script is adapted from gpu_tile_compact.sh script from aurora
   """
   bash_script = [
                    "#!/bin/bash",
                    "##get the hostname",
                    "hname=$(hostname)",
                    "##get the free gpus from the environment variable",
                    r'IFS="," read -ra my_free_gpus <<< "${AVAILABLE_GPUS_${hname}}"',
                    "# Get the RankID from different launcher",
                    "if [[ -v MPI_LOCALRANKID ]]; then",
                    "   _MPI_RANKID=$MPI_LOCALRANKID ",
                    "elif [[ -v PALS_LOCAL_RANKID ]]; then",
                    "   _MPI_RANKID=$PALS_LOCAL_RANKID",
                    "fi",
                    "unset EnableWalkerPartition",
                    "export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE",
                    "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1",
                    "# Calculate the GPUs assigned to this rank",
                    f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                    f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                    r"export ZE_AFFINITY_MASK=${rank_gpus}",
                    r"ulimit -c 0 # Until Aurora filesystem problems are fixed",
                    '"$@"'
                ]
   return "\n".join(bash_script)

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
    mpi_rankfile: Optional[str] = None
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
        "log_file": os.path.join(run_dir, f"worker_{socket.gethostname()}.log"),
        "err_file": os.path.join(run_dir, f"worker_{socket.gethostname()}.err"),
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