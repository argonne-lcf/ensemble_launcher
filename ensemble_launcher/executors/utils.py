from typing import Callable, Tuple, Dict, Any, Type
import cloudpickle
import base64
from ensemble_launcher.comm.queue import QueueProtocol
import logging

logger = logging.getLogger(__name__)

def run_callable_with_affinity(fn: Callable, 
                               args: Tuple = (), 
                               kwargs: Dict = {}, 
                               cpu_id:int = None, 
                               env: Dict[str, Any] = {}):
    """
    Function to run a callable on specific cpus and reset affinity after execution.
    """
    import os
    original_affinity = None
    if cpu_id is not None:
        try:
            original_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0, [cpu_id])
        except Exception as e:
            print(f"Setting affinity failed with exception {e}")
    original_env = os.environ.copy()
    os.environ.update(env)
    result = fn(*args,**kwargs)
    os.environ.clear()
    os.environ.update(original_env)
    # Reset affinity
    if cpu_id is not None and original_affinity is not None:
        try:
            os.sched_setaffinity(0, original_affinity)
        except Exception as e:
            print(f"Resetting affinity failed with exception {e}")
    return result


def run_cmd(cmd: str, 
            args: Tuple = (), 
            kwargs: Dict = {}, 
            cpu_id:int = None, 
            env: Dict[str, Any] = {}):
    import os
    import subprocess

    cmd = [s.strip() for s in cmd.split()]

    original_affinity = None
    if cpu_id is not None:
        try:
            original_affinity = os.sched_getaffinity(0)
            os.sched_setaffinity(0,[cpu_id])
        except Exception as e:
            print(f"Setting affinity failed with exception {e}")
    merged_env = os.environ.copy()
    merged_env.update(env)
    kwargs_list = []
    for k,v in kwargs.items():
        kwargs_list.extend([str(k),str(v)])
    result = subprocess.run([cmd] + list(args) + kwargs_list , capture_output=True, text=True,env=merged_env)

    # Reset affinity
    if cpu_id is not None and original_affinity is not None:
        try:
            os.sched_setaffinity(0, original_affinity)
        except Exception as e:
            print(f"Resetting affinity failed with exception {e}")
    return result.stdout

def return_wrapper(queue:QueueProtocol, fn: Callable, args: Tuple = (), kwargs: Dict = {}):
    result = fn(*args, **kwargs)
    queue.put(result)
    return 

def serialize_callable(fn: Callable, args: Tuple, kwargs: Dict) -> str:
    """serialize the function and args"""
    fb = cloudpickle.dumps((fn,args,kwargs))
    return base64.encodebytes(fb).decode("utf-8")

def generate_python_exec_command(fn: Callable, args: Tuple, kwargs: Dict) -> str:
    """
    Generates a Python command string that deserializes and executes a given function with arguments.
    The returned string can be run using `python -c "<command>"`.
    """
    s = serialize_callable(fn, args, kwargs)
    script = [
        "import cloudpickle",
        "import base64",
        f"s = {repr(s)}",
        "fb = base64.decodebytes(s.encode('utf-8'))",
        "fn,args,kwargs = cloudpickle.loads(fb)",
        "fn(*args,**kwargs)"
    ]
    return ";".join(script)
    
"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use the same GPU
"""
def gen_affinity_bash_script_1(ngpus_per_process: int, gpu_selector: str) -> str:
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
                      "export ZE_FLAT_DEVICE_HIERARCHY=FLAT" if gpu_selector=="ZE_AFFINITY_MASK" else "",
                      "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" if gpu_selector=="ZE_AFFINITY_MASK" else "",
                      "# Calculate the GPUs assigned to this rank",
                      f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                      f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                      r"echo $rank_gpus $_MPI_RANKID",
                      f"export {gpu_selector}" + r"=${rank_gpus}",
                      '"$@"'
                 ]
    return "\n".join(bash_script)


"""
this function generates a bash script that can be used to set the affinity mask for the GPUs
when all ranks on various nodes use different GPUs
"""
def gen_affinity_bash_script_2(ngpus_per_process: int, gpu_selector: str) -> str:
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
                    "export ZE_FLAT_DEVICE_HIERARCHY=FLAT" if gpu_selector == "ZE_AFFINITY_MASK" else "",
                    "export ZE_ENABLE_PCI_ID_DEVICE_ORDER=1" if gpu_selector == "ZE_AFFINITY_MASK" else "",
                    "# Calculate the GPUs assigned to this rank",
                    f"start_idx=$((_MPI_RANKID * {ngpus_per_process}))",
                    f"rank_gpus=$(IFS=','; echo \"${{my_free_gpus[@]:${{start_idx}}:{ngpus_per_process}}}\")",
                    f"export {gpu_selector}"+r"=${rank_gpus}",
                    '"$@"'
                ]
   return "\n".join(bash_script)


class ExecutorRegistry:
    def __init__(self):
        self._available_executors: Dict = {}

    def register(self,name:str):
        def decorator(cls: Type[Any]):
            self._available_executors[name] = cls
            return cls
        return decorator
    
    def create_executor(self,name:str ,args:Tuple = (), kwargs:Dict ={}):
        try:
            return self._available_executors[name](*args, **kwargs)
        except KeyError:
            logger.error(f"Executor '{name}' not found in registry. Available {self._available_executors.keys()}")
            raise

executor_registry = ExecutorRegistry()            