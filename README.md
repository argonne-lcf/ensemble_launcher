# ensemble_launcher
A lightweight tool for launching and managing ensembles with features like:

- **Flexible Configurations**: Set up one-to-one or many-to-many task relationships effortlessly.
- **Parallel Execution**: Run tasks across multiple processes for better performance.
- **Custom Commands**: Define your own task-specific commands and environment settings.
- **Smart Resource Management**: Automatically allocate nodes, cores, and GPUs as needed.
- **Cluster Support**: Works seamlessly on local systems and clusters like Aurora, including GPU affinity.
- **Resilience**: Retries failed tasks and cleans up resources automatically.
- **Progress Logs**: Tracks task status and resource usage in real-time.
- **Modular Design**: Easy to extend and integrate into your workflows.
- **Simple Setup**: Use JSON files to configure tasks quickly.
- **Flexible Launchers**: Supports both MPI and Bash for task execution.
- **Live Updates**: Update ensembles on the fly while they’re running.

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
    - **num_nodes**: Number of nodes required for the ensemble. Can be varied for each task
    - **num_processes_per_node**: Number of processes to run on each node.
    - **num_gpus_per_process**: Number of GPUs allocated per process.
    - **launcher**: Task launcher type (`mpi` or `bash`).
    - **launcher_options**: Additional options for the launcher:
      - **np**: Total number of processes.
      - **ppn**: Processes per node.
      - **cpu-bind**: CPU binding strategy (e.g., `depth`, `list`).
      - **depth**: Depth of CPU binding.
    - **relation**: Relationship between task parameters (`one-to-one` or `many-to-many`).
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

- **Basic Example**: A simple one-to-one task configuration is available in `tests/simple_test/`.
- **Advanced Example**: Can be found in `examples/`

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request.

## Support

If you encounter any issues, feel free to open an issue on the GitHub repository or contact the maintainers.



