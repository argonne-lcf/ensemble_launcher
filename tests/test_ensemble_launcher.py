import time
import sys
import socket
import os
from glob import glob
from ensemble_launcher import ensemble_launcher
import json

def get_num_nodes():
    """
    Get the number of nodes from the PBS_NODEFILE environment variable.
    """
    try:
        with open(os.getenv("PBS_NODEFILE", "/dev/null"), "r") as f:
            lines = f.readlines()
            num_nodes = len(lines)
        return max(num_nodes, 1)
    except Exception as e:
        print(f"Error reading PBS_NODEFILE: {e}")
        return 1

def test_cpu():
    nprocs = 1
    ngpus = 0

    num_nodes = get_num_nodes()
    ensembles = {}
    ensembles["poll_interval"] = 1
    ensembles["update_interval"] = None

    ensembles["sys_info"] = {
        "name":"aurora",
        "ncores_per_node":104,
        "ngpus_per_node":12
    }

    ensembles["ensembles"] = {
        "name1":{
                "num_nodes":1,
                "num_processes_per_node":nprocs,
                "launcher":"mpi",
                "relation":"one-to-one",
                "pre_launch_cmd":"echo 'Pre-launch command executed'",
                "cmd_template":"./test_script.sh -h 0 -l {opts1}",
                "opts1":f"linspace(0, {num_nodes*104//nprocs} , {num_nodes*104//nprocs})",
                "run_dir":"./run_dir/name1",
        },
        "name2":{
                "num_nodes":list(range(1, num_nodes+1)),
                "num_processes_per_node":nprocs,
                "launcher":"mpi",
                "relation":"one-to-one",
                "pre_launch_cmd":"echo 'Pre-launch command executed'",
                "cmd_template":"./test_script.sh -h 0",
                "run_dir":"./run_dir/name2",
        }
    }

    with open("config.json", "w") as f:
        json.dump(ensembles, f, indent=4)

    el = ensemble_launcher("config.json")
    total_poll_time = el.run_tasks()

    logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))
    logfiles += list(glob(os.path.join("./run_dir","name2", "log_*.txt")))

    num_nodes = get_num_nodes()
    assert len(logfiles) == ((num_nodes*104//nprocs)+num_nodes)
    count = 0
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "started sleep" in line:
                    count += 1
    
    assert count == (num_nodes*104 + sum([i*nprocs for i in list(range(1,num_nodes+1))]))


# def test_gpu():
#     nprocs = 6
#     ngpus = 1

#     num_nodes = get_num_nodes()
#     ensembles = {}
#     ensembles["poll_interval"] = 1
#     ensembles["update_interval"] = None

#     ensembles["sys_info"] = {
#         "name":"aurora",
#         "ncores_per_node":104,
#         "ngpus_per_node":12
#     }

#     ensembles["ensembles"] = {
#         "name1":{
#                 "num_nodes":1,
#                 "num_processes_per_node":nprocs,
#                 "num_gpus_per_process":ngpus,
#                 "launcher":"mpi",
#                 "relation":"one-to-one",
#                 "pre_launch_cmd":"echo 'Pre-launch command executed'",
#                 "cmd_template":"./test_script.sh -h 0 -l {opts1}",
#                 "opts1":f"linspace(0, {num_nodes*104//nprocs} , {num_nodes*104//nprocs})",
#                 "run_dir":"./run_dir/name1",
#         },
#         "name2":{
#                 "num_nodes":list(range(1, num_nodes+1)),
#                 "num_processes_per_node":nprocs,
#                 "num_gpus_per_process":ngpus*2,
#                 "launcher":"mpi",
#                 "relation":"one-to-one",
#                 "pre_launch_cmd":"echo 'Pre-launch command executed'",
#                 "cmd_template":"./test_script.sh -h 0",
#                 "run_dir":"./run_dir/name2",
#         }
#     }

#     with open("config.json", "w") as f:
#         json.dump(ensembles, f, indent=4))

#     el = ensemble_launcher("config.json")
#     total_poll_time = el.run_tasks()

#     logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))
#     logfiles += list(glob(os.path.join("./run_dir","name2", "log_*.txt")))

#     num_nodes = get_num_nodes()
#     assert len(logfiles) == ((num_nodes*104//nprocs)+num_nodes)
#     count = 0
#     for logfile in logfiles:
#         with open(logfile, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if "started sleep" in line:
#                     count += 1
    
#     assert count == (num_nodes*104 + num_nodes*nprocs)

# def test_cpu_and_gpu():
#     nprocs = 6
#     ngpus = 1

#     num_nodes = get_num_nodes()
#     ensembles = {}
#     ensembles["poll_interval"] = 1
#     ensembles["update_interval"] = None

#     ensembles["sys_info"] = {
#         "name":"aurora",
#         "ncores_per_node":104,
#         "ngpus_per_node":12
#     }

#     ensembles["ensembles"] = {
#         "name1":{
#                 "num_nodes":1,
#                 "num_processes_per_node":nprocs,
#                 "launcher":"mpi",
#                 "relation":"one-to-one",
#                 "pre_launch_cmd":"echo 'Pre-launch command executed'",
#                 "cmd_template":"./test_script.sh -h 0 -l {opts1}",
#                 "opts1":f"linspace(0, {num_nodes*104//nprocs} , {num_nodes*104//nprocs})",
#                 "run_dir":"./run_dir/name1",
#         },
#         "name2":{
#                 "num_nodes":list(range(1, num_nodes+1)),
#                 "num_processes_per_node":nprocs,
#                 "num_gpus_per_process":ngpus,
#                 "launcher":"mpi",
#                 "relation":"one-to-one",
#                 "pre_launch_cmd":"echo 'Pre-launch command executed'",
#                 "cmd_template":"./test_script.sh -h 0",
#                 "run_dir":"./run_dir/name2",
#         }
#     }

#     with open("config.json", "w") as f:
#         json.dump(ensembles, f, indent=4))

#     el = ensemble_launcher("config.json")
#     total_poll_time = el.run_tasks()

#     logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))
#     logfiles += list(glob(os.path.join("./run_dir","name2", "log_*.txt")))

#     num_nodes = get_num_nodes()
#     assert len(logfiles) == ((num_nodes*104//nprocs)+num_nodes)
#     count = 0
#     for logfile in logfiles:
#         with open(logfile, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 if "started sleep" in line:
#                     count += 1
    
#     assert count == (num_nodes*104 + num_nodes*nprocs)

    
if __name__ == "__main__":
    test_cpu()
    # test_gpu()
    # test_cpu_and_gpu()