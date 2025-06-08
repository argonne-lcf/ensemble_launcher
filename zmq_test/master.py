import zmq
import subprocess
import os
import socket
import sys
import time

sys.path.append(os.getcwd())


def get_nodes() -> list:
    node_list = []
    node_file = os.getenv("PBS_NODEFILE")
    if node_file is not None and os.path.exists(node_file):
        with open(node_file, "r") as f:
            node_list = f.readlines()
            node_list = [node.split(".")[0].strip() for node in node_list]
    else:
        node_list = [socket.gethostname()]
    # node_list.pop(node_list.index(socket.gethostname()))
    return node_list

def get_nfd(pid)->int:
    if pid is None:
        return 0
    fd_path = f'/proc/{pid}/fd'
    try:
        open_fds = os.listdir(fd_path)
        return len(open_fds)
    except:
        return 0


if __name__ == "__main__":

    context = zmq.Context()

    rs = context.socket(zmq.ROUTER)
    parent_name = socket.gethostname()
    if "local" in parent_name:
        parent_name="localhost"
    rs.bind(f"tcp://{parent_name}:5555")

    nodes = get_nodes()

    processes = []
    ##launch processes on nodes
    for nid,node in enumerate(nodes):
        p = subprocess.Popen(f"mpirun -np 1 -host {node} python3 worker.py {nid} {parent_name} 5555",
                             executable="/bin/bash",
                             cwd=os.getcwd(),
                             shell=True)
        processes.append(p)
    
    expected_connections = len(nodes)

    current_connections = set()
    nloops = 0 
    ##wait for connected children
    while len(current_connections) < expected_connections:
        print(nloops)
        msg_parts = rs.recv_multipart()
        print(msg_parts)
        if msg_parts[-1].decode() == "READY":
            current_connections.add(msg_parts[1])
        nloops+=1

    for nid in range(len(nodes)):
        rs.send_multipart([f"{nid}".encode(),f"hello,node:{nid}".encode()])
    
    count = 0
    while count < 10:
        nfd = 0
        for p in processes:
            nfd += get_nfd(p.pid)
        print(f"number of fds on master {nfd}!")
        count += 1
        time.sleep(1)
    ndone = 0
    while ndone < len(nodes):
        msg_parts = rs.recv_multipart()
        if msg_parts[-1].decode() == "DONE":
            print(f"{msg_parts[1].decode()} is {msg_parts[-1].decode()}!")
            ndone += 1
        else:
            print(f"i received this {msg_parts[-1].decode()} from {msg_parts[1].decode()}!")

    rs.close()
    context.term()
