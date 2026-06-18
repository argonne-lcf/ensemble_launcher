import os
import socket
from typing import Dict, List, Optional, Any
from ensemble_launcher.comm import Result
import json
from subprocess import check_output, DEVNULL



def get_nodes():
    
    fname = os.getenv("PBS_NODEFILE","/dev/null")
    with open(fname) as f:
        lines = f.readlines()
    
    if len(lines) > 0:
        return [line.split(".")[0] for line in lines]
    else:
        return [socket.gethostname()]


def str_to_num(s: str) -> int | float:
    """Return a number from of the correct type from a string

    :param s: GPU ID
    :type s: str
    :return: number corresponding to input string
    :rtype: int or float
    """
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            raise ValueError("GPU mask needs to take ints or floats as the GPU IDs.") from None


def get_gpus() -> list[int | float]:
    """Get the list of GPUs available on the system node
    """
    # NVIDIA
    try:
        mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if mask:
            gpus = []
            for gpu in mask.split(","):
                gpus.append(str_to_num(gpu))
        else:
            output = check_output(["nvidia-smi", "-L"], stderr=DEVNULL).decode("utf-8").splitlines()
            gpus = list(range(len(output)))
        return gpus
    except:
        pass

    # Intel
    try:
        mask = os.environ.get("ZE_AFFINITY_MASK", "")
        if mask:
            gpus = []
            for gpu in mask.split(","):
                gpus.append(str_to_num(gpu))
        else:
            output = check_output(["xpu-smi", "discovery"], stderr=DEVNULL).decode("utf-8").splitlines()
            hierarchy_mode = os.environ.get("ZE_FLAT_DEVICE_HIERARCHY", "FLAT")
            gpus = []
            for i in range(len(output)):
                if hierarchy_mode == "FLAT":
                    gpus.append(i * 2)
                    gpus.append(i * 2 + 1)
                elif hierarchy_mode == "COMPOSITE":
                    gpus.append(i + 0.0)
                    gpus.append(i + 0.1)
        return gpus
    except:
        pass

    return []


def write_results_to_json(results: Result, fname: str = "./results.json"):
    """Fuction that writes the aggregated results to a json"""
    results_dict = {}
    for r in results.data:
        if isinstance(r.data, bytes):
            data = r.data.decode('utf-8')
        else:
            data = r.data
        
        # Handle newlines in the data
        if isinstance(data, str) and '\n' in data:
            results_dict[r.task_id] = data.split('\n')
        else:
            results_dict[r.task_id] = data
    
    with open(fname,"w") as f:
        json.dump(results_dict,f,indent=4)