# Quick Start

## 1. Define Your Ensemble

Create a JSON configuration file describing your task ensemble:

```json
{
    "ensembles": {
        "example_ensemble": {
            "nnodes": 1,
            "ppn": 1,
            "cmd_template": "./exe -a {arg1} -b {arg2}",
            "arg1": "linspace(0, 10, 5)",
            "arg2": "linspace(0, 1, 5)",
            "relation": "one-to-one"
        }
    }
}
```

This configuration specifies:

- Tasks running on a single node with a single process per node
- Tasks executed with `./exe -a {arg1} -b {arg2}` taking two input arguments
- 5 linearly spaced values between 0--10 for `arg1` and 0--1 for `arg2`
- `one-to-one` relationship: 5 tasks, one for each pair of values

**Supported Relations:**

| Relation | Description |
|---|---|
| `one-to-one` | Pair parameters element-wise (N tasks) |
| `many-to-many` | Cartesian product of parameters (N x M tasks) |

## 2. Create a Launcher Script

```python
from ensemble_launcher import EnsembleLauncher

if __name__ == '__main__':
    el = EnsembleLauncher("config.json")
    results = el.run()

    from ensemble_launcher import write_results_to_json
    write_results_to_json(results, "results.json")
```

## 3. Execute

```bash
python3 launcher_script.py
```

## Using Python Callables

You can also define tasks as Python functions instead of shell commands:

```python
from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.ensemble import Task

def my_simulation(param_a, param_b):
    return result

tasks = {
    "task-1": Task(
        task_id="task-1",
        nnodes=1,
        ppn=1,
        executable=my_simulation,
        args=(10, 0.5)
    )
}

el = EnsembleLauncher(ensemble_file=tasks)
results = el.run()
```

Internally, dictionary-based ensemble definitions are also converted to `Task` objects.
