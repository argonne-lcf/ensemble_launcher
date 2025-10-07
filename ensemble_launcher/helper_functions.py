import os
import socket
from typing import Dict, List, Optional, Any



def get_nodes(self):
    
    fname = os.getenv("PBS_NODEFILE","/dev/null")
    with open(fname) as f:
        lines = f.readline()
    
    return [line.split(".")[0] for line in lines]