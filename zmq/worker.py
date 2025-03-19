import zmq
import subprocess
import os
import time 
import sys

worker_id = sys.argv[1]

parent_id = sys.argv[2]

port = sys.argv[3]


context = zmq.Context()

rs = context.socket(zmq.DEALER)
rs.setsockopt(zmq.IDENTITY,f"{worker_id}".encode())
print("connecting to:",parent_id,port)
rs.connect(f"tcp://{parent_id}:{port}")
rs.send_multipart([f"{worker_id}".encode(),b"READY"])
message = rs.recv()

nprocs = os.cpu_count()

procs = []
for pid in range(nprocs-1):
    p = subprocess.Popen(f"mpirun -np 1 ./test_script.sh -h {message.decode()},proc:{pid} {worker_id}",
                      executable="/bin/bash",
                      stdout=subprocess.PIPE,
                      stderr=subprocess.PIPE,
                      cwd=os.getcwd(),
                      shell=True)
    procs.append(p)

ndone = 0
with open(f"log_{worker_id}.txt","w") as f:
    while ndone < nprocs:
        for pid,p in enumerate(procs):
            return_code = p.poll()
            if return_code is not None:
                out, err = p.communicate()
                ndone += 1
                if out:
                    f.write(out.decode() + "\n")

rs.send_multipart([f"{worker_id}".encode(),b"DONE"])

rs.close()
context.term()

