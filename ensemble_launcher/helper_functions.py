import os
import socket
from typing import Dict, List, Optional, Any
from ensemble_launcher.comm import Result
import json



def get_nodes():
    
    fname = os.getenv("PBS_NODEFILE","/dev/null")
    with open(fname) as f:
        lines = f.readlines()
    
    if len(lines) > 0:
        return [line.split(".")[0] for line in lines]
    else:
        return [socket.gethostname()]


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