import os
from glob import glob
from ensemble_launcher import ensemble_launcher
import json
import logging
import threading
import time
import sys

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

#**************************************************************************************************
def test_cpu(comm_config={"comm_layer":"multiprocessing"},parallel_backend="multiprocessing"):
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    nprocs = 1
    ngpus = 0

    num_nodes = get_num_nodes()
    ensembles = {}
    ensembles["poll_interval"] = 1
    ensembles["update_interval"] = None

    # ensembles["sys_info"] = {
    #     "name":"aurora",
    #     "ncores_per_node":104,
    #     "ngpus_per_node":12
    # }
    ensembles["comm_config"] = comm_config
    ensembles["ensembles"] = {
        "name1":{
                "num_nodes":1,
                "num_processes_per_node":nprocs,
                "launcher":"mpi",
                "relation":"one-to-one",
                "pre_launch_cmd":"echo 'Pre-launch command executed'",
                "cmd_template":"./test_script.sh -h 0 -l {opts1}",
                "opts1":f"linspace(0, {num_nodes*12//nprocs} , {num_nodes*12//nprocs})",
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

    el = ensemble_launcher("config.json",logging_level=logging.DEBUG,parallel_backend=parallel_backend)
    total_poll_time = el.run_tasks()

    logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))
    logfiles += list(glob(os.path.join("./run_dir","name2", "log_*.txt")))

    num_nodes = get_num_nodes()
    assert len(logfiles) == ((num_nodes*12//nprocs)+num_nodes)
    count = 0
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "started sleep" in line:
                    count += 1

    assert count == ((num_nodes*12//nprocs)*nprocs + sum([i*nprocs for i in list(range(1,num_nodes+1))]))
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")

    ##forcing multilevel launcher
    el = ensemble_launcher("config.json",logging_level=logging.DEBUG, force_level="double", parallel_backend=parallel_backend)
    total_poll_time = el.run_tasks()

    logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))
    logfiles += list(glob(os.path.join("./run_dir","name2", "log_*.txt")))

    num_nodes = get_num_nodes()
    assert len(logfiles) == ((num_nodes*12//nprocs)+num_nodes)
    count = 0
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "started sleep" in line:
                    count += 1

    assert count == ((num_nodes*12//nprocs)*nprocs + sum([i*nprocs for i in list(range(1,num_nodes+1))]))
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")


#**************************************************************************************************
def test_gpu(comm_config={"comm_layer":"multiprocessing"},parallel_backend="multiprocessing"):
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    nprocs = 6
    ngpus = 1

    num_nodes = get_num_nodes()
    ensembles = {}
    ensembles["poll_interval"] = 1
    ensembles["update_interval"] = None

    ensembles["sys_info"] = {
        "name":"aurora",
        "ncores_per_node":104,
        "ngpus_per_node":12
    }
    ensembles["comm_config"] = comm_config

    ensembles["ensembles"] = {
        "name1":{
                "num_nodes":1,
                "num_processes_per_node":nprocs,
                "num_gpus_per_process":ngpus,
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
                "num_gpus_per_process":ngpus*2,
                "launcher":"mpi",
                "relation":"one-to-one",
                "pre_launch_cmd":"echo 'Pre-launch command executed'",
                "cmd_template":"./test_script.sh -h 0",
                "run_dir":"./run_dir/name2",
        }
    }

    with open("config.json", "w") as f:
        json.dump(ensembles, f, indent=4)

    el = ensemble_launcher("config.json",logging_level=logging.DEBUG,parallel_backend=parallel_backend)
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
    
    assert count == ((num_nodes*104//nprocs)*nprocs + sum([i*nprocs for i in list(range(1,num_nodes+1))]))
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    
    ###forcing multi level launcher
    el = ensemble_launcher("config.json",logging_level=logging.DEBUG,force_level="double", parallel_backend=parallel_backend)
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
    
    assert count == ((num_nodes*104//nprocs)*nprocs + sum([i*nprocs for i in list(range(1,num_nodes+1))]))
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")

#**************************************************************************************************
def test_cpu_and_gpu(comm_config={"comm_layer":"multiprocessing"},parallel_backend="multiprocessing"):
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    nprocs = 6
    ngpus = 1

    num_nodes = get_num_nodes()
    ensembles = {}
    ensembles["poll_interval"] = 1
    ensembles["update_interval"] = None

    ensembles["sys_info"] = {
        "name":"aurora",
        "ncores_per_node":104,
        "ngpus_per_node":12
    }

    ensembles["comm_config"] = comm_config

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
                "num_gpus_per_process":ngpus,
                "launcher":"mpi",
                "relation":"one-to-one",
                "pre_launch_cmd":"echo 'Pre-launch command executed'",
                "cmd_template":"./test_script.sh -h 0",
                "run_dir":"./run_dir/name2",
        }
    }

    with open("config.json", "w") as f:
        json.dump(ensembles, f, indent=4)

    el = ensemble_launcher("config.json",logging_level=logging.DEBUG,parallel_backend=parallel_backend)
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
    
    assert count == ((num_nodes*104//nprocs)*nprocs + sum([i*nprocs for i in list(range(1,num_nodes+1))]))
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    
    ##forcing multilevel launcher
    el = ensemble_launcher("config.json",logging_level=logging.DEBUG,force_level="double", parallel_backend=parallel_backend)
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
    
    assert count == ((num_nodes*104//nprocs)*nprocs + sum([i*nprocs for i in list(range(1,num_nodes+1))]))
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")

#**************************************************************************************************
def test_ensemble_update(comm_config={"comm_layer":"multiprocessing"}, parallel_backend="multiprocessing"):
    def write_ensemble(num_nodes,nprocs,ngpus):
        ensembles = {}
        ensembles["poll_interval"] = 1
        ensembles["update_interval"] = 1

        # ensembles["sys_info"] = {
        # "name":"aurora",
        # "ncores_per_node":104,
        # "ngpus_per_node":12
        # }
        ensembles["comm_config"] = comm_config

        ensembles["ensembles"] = {
            "name1":{
                    "num_nodes":1,
                    "num_processes_per_node":nprocs,
                    "launcher":"mpi",
                    "relation":"one-to-one",
                    "pre_launch_cmd":"echo 'Pre-launch command executed'",
                    "cmd_template":"./test_script.sh -h 0 -l {opts1}",
                    "opts1":f"linspace(0, {num_nodes*3//nprocs} , {num_nodes*3//nprocs})",
                    "run_dir":"./run_dir/name1",
            },
        }

        with open("config.json", "w") as f:
            json.dump(ensembles, f, indent=4)

    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    nprocs = 1
    ngpus = 0

    num_nodes = get_num_nodes()

    write_ensemble(num_nodes,nprocs,ngpus)
    
    # Create a function to run the ensemble launcher in a thread
    def run_ensemble():
        el = ensemble_launcher("config.json", logging_level=logging.DEBUG, parallel_backend=parallel_backend)
        return el.run_tasks()

    # Start a thread to run the ensemble launcher
    thread = threading.Thread(target=run_ensemble)
    thread.start()
    time.sleep(10)
    num_nodes *= 2
    write_ensemble(num_nodes,nprocs,ngpus)
    thread.join()  # Wait for thread to complete

    logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))

    assert len(logfiles) == ((num_nodes*3//nprocs))
    count = 0
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "started sleep" in line:
                    count += 1

    # assert count == ((num_nodes*12//nprocs)*nprocs )
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")

    ##forcing multilevel launcher
    # Create a function to run the ensemble launcher in a thread
    def run_ensemble_multi_level():
        el = ensemble_launcher("config.json", logging_level=logging.DEBUG, force_level="double", parallel_backend=parallel_backend)
        return el.run_tasks()

    write_ensemble(num_nodes,nprocs,ngpus)
    # Start a thread to run the ensemble launcher
    thread = threading.Thread(target=run_ensemble_multi_level)
    thread.start()
    time.sleep(10)
    num_nodes *= 2
    write_ensemble(num_nodes,nprocs,ngpus)
    thread.join()  # Wait for thread to complete

    logfiles = list(glob(os.path.join("./run_dir","name1", "log_*.txt")))

    assert len(logfiles) == ((num_nodes*3//nprocs))
    count = 0
    for logfile in logfiles:
        with open(logfile, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "started sleep" in line:
                    count += 1
    
    assert count == ((num_nodes*3//nprocs)*nprocs)
    if os.path.exists("./run_dir"):
        os.system("rm -rf ./run_dir")
    
if __name__ == "__main__":
    print("Running tests...This may take a while")
    comm_layers = [
        {"name": "multiprocessing", "config": {"comm_layer": "multiprocessing"}},
        {"name": "zmq", "config": {"comm_layer": "zmq"}}
    ]
    
    test_functions = [
        # test_cpu,
        # test_gpu,
        # test_cpu_and_gpu,
        test_ensemble_update
    ]
    
    # for comm_layer in comm_layers:
    #     for test_func in test_functions:
    #         print(f"Running test: {test_func.__name__} with {comm_layer['name']} comm layer")
    #         test_func(comm_config=comm_layer["config"])
    comm_layer = comm_layers[-1]
    for test_func in test_functions:
        print(f"Running test: {test_func.__name__} with {comm_layer['name']} comm layer and MPI backend")
        test_func(comm_config=comm_layer["config"], parallel_backend="mpi")
    print("All tests passed successfully!")