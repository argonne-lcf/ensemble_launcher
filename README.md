# ensemble_launcher
A lightweight tool for launching and managing ensembles

## Installation

To install `ensemble_launcher`, clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-repo/ensemble_launcher.git
cd ensemble_launcher
pip install -r requirements.txt
```

## Usage

1. **Create a Configuration File**: Define your ensembles and tasks in a JSON file. Below is an example configuration file (`tests/simple_test/config.json`) and an explanation of its options:

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "local",
        "ncores_per_nodes": 1,
        "ngpus_per_node": 1
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 1,
            "num_processes_per_node": 1,
            "num_gpus_per_process": 1,
            "launcher": "mpi",
            "launcher_options": {
                "np": 1,
                "ppn": 1,
                "cpu-bind": "depth",
                "depth": 1
            },
            "relation": "one-to-one",
            "cmd_template": "./exe -a {arg1} -b {arg2}",
            "arg1": "linspace(0, 10, 5)",
            "arg2": "linspace(0, 1, 5)",
            "run_dir": "./run_dir",
            "env":{
                "var":"value"
            }
        }
    }
}
```

### Explanation of Configuration Options:

- **poll_interval**: Time interval (in seconds) to check the status of running tasks.
- **update_interval**: Time interval (in seconds) to update the ensemble configuration. Set to `null` to disable updates.
- **sys_info**: System-specific information:
  - **name**: Name of the system (e.g., `local`, `aurora`).
  - **ncores_per_nodes**: Number of CPU cores available per node.
  - **ngpus_per_node**: Number of GPUs available per node.
- **ensembles**: A dictionary defining the ensembles to be executed:
  - **example_ensemble**: Name of the ensemble.
    - **num_nodes**: Number of nodes required per task in the ensemble. Can be varied for each task
    - **num_processes_per_node**: Number of processes per node used per task.
    - **num_gpus_per_process**: Number of GPUs allocated per process per task.
    - **launcher**: Task launcher type (`mpi` or `bash`).
    - **launcher_options**: Additional options for the launcher:
      - **np**: Total number of processes.
      - **ppn**: Processes per node.
      - **cpu-bind**: CPU binding strategy (e.g., `depth`, `list`).
      - **depth**: Depth of CPU binding.
    - **relation**: Relationship between task parameters (`one-to-one` or `many-to-many`).
    - **pre_launch_cmd**: A linux cmd to be executed before launching running the below. (eg. cp -r * ./run_dir)
    - **cmd_template**: Template for the command to execute, with placeholders for task-specific arguments. Variable arguments should be surrounded by `{}`.
    - **arg1**, **arg2**: Task-specific arguments, which can be defined using functions like `linspace`.
    - **run_dir**: Directory where task outputs and logs will be stored.
    - **env**: A dictionary of environment variables to set for the tasks:
      - **key**: Name of the environment variable.
      - **value**: Value of the environment variable. Can be a static value or dynamically generated.

2. **Run the Launcher**: Use the provided Python script to launch tasks:
    ```bash
    cd tests/simple_test
    python tests/simple_test/test_ensemble_launcher.py
    ```

3. **Monitor Progress**: Check the `outputs` directory for logs and status updates.

## Examples

Following are example .json config files for various mpiexec commands used at ALCF

Example 1: 2 nodes, 4 ranks/node, 1 thread/rank

```bash
  mpiexec -n 8 -ppn 4 --depth 1 --cpu-bind=depth <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 2,
            "num_processes_per_node": 4,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "depth",
                "depth": 1
            },
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir",
        }
    }
}
```
Example 2: 2 nodes, 2 ranks/node, 2 thread/rank
```bash
  OMP_PLACES=threads OMP_NUM_THREADS=2 mpiexec -n 4 -ppn 2 --depth 2 --cpu-bind=depth <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 2,
            "num_processes_per_node": 2,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "depth",
                "depth": 2
            },
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir",
            "env":{
              "OMP_PLACES": "threads",
              "OMP_NUM_THREADS": 2,
            }
        }
    }
}
```
Example 3: 2 nodes, 2 ranks/node, 1 thread/rank, compact fashion
```bash
  mpiexec -n 4 -ppn 2 --cpu-bind=list:0:104 <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 2,
            "num_processes_per_node": 2,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "list:0:104"
            },
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir"
        }
    }
}
```

Example 4: 1 node, 12 ranks/node

```bash
  mpiexec -n 12 -ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 1,
            "num_processes_per_node": 12,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99"
            },
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir"
        }
    }
}
```

Example 5: 1 node, 12 ranks/node, 1 thread/rank, 1 rank/GPU tile

```bash
mpiexec -n 12 -ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 gpu_tile_compact.sh <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 1,
            "num_processes_per_node": 12,
            "num_gpus_per_process":1,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99"
            },
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir",
            "env":{
              "ZE_FLAT_DEVICE_HIERARCHY":"COMPOSITE"
            }
        }
    }
}
```

Example 6: 1 node, 6 ranks/node, 1 thread/rank, 1 rank/GPU device

```bash
mpiexec -n 12 -ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 gpu_dev_compact.sh <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 1,
            "num_processes_per_node": 6,
            "num_gpus_per_process":2,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99"
            },
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir",
            "env":{
              "ZE_FLAT_DEVICE_HIERARCHY":"COMPOSITE"
            }
        }
    }
}
```

Example 7: 1 node, 12 ranks/node, 1 thread/rank, and any other MPI options

```bash
mpiexec -n 12 -ppn 12 --cpu-bind=list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99 <other mpi options> <app> <app_args>
```

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": 1,
            "num_processes_per_node": 12,
            "launcher": "mpi",
            "launcher_options": {
                "cpu-bind": "list:0-7:8-15:16-23:24-31:32-39:40-47:52-59:60-67:68-75:76-83:84-91:92-99"
            },
            "relation": "one-to-one",
            "cmd_template": "<any other mpi options> <app> <constant args> <variable args>",
            "<variable args>": [1,2,....],
            "run_dir": "./run_dir",
        }
    }
}
```

Examples of other general ensembles

Example 1: An ensemble with N tasks. Each task usese 1 node, 12 ranks/node, and they have identical args

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": [1,1,1,1,1........N],
            "num_processes_per_node": 12,
            "launcher": "mpi",
            "relation": "one-to-one",
            "cmd_template": "<any other mpi options> <app> <constant args>",
            "run_dir": "./run_dir",
        }
    }
}
```

Example 2: A scaling test using 1-128 nodes and 104 ranks/node

```json
{
    "poll_interval": 1,
    "update_interval": null,
    "sys_info": {
        "name": "aurora",
        "ncores_per_nodes": 104,
        "ngpus_per_node": 12
    },
    "ensembles": {
        "example_ensemble": {
            "num_nodes": [1,2,4,8,16,32,64,128],
            "num_processes_per_node": 104,
            "launcher": "mpi",
            "relation": "one-to-one",
            "cmd_template": "<app> <constant args> <variable args>",
            "<variable_args>":[1,...]
            "run_dir": ["./run_dir_1","./run_dir_2","./run_dir_4","./run_dir_8","./run_dir_16","./run_dir_32","./run_dir_64","./run_dir_128"],
        }
    }
}
```

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## Support

If you encounter any issues, feel free to open an issue on the GitHub repository or contact the maintainers.



