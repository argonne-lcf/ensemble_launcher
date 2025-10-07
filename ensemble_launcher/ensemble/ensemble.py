import json
import os
import numpy as np
import enum
from typing import Optional, Union, Callable, Tuple, Dict, List, Any
from pydantic import BaseModel, Field


class TaskStatus(enum.Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"

class Task(BaseModel):
    task_id: str
    nnodes: int
    ppn: int
    executable: Union[str, Callable]
    ngpus_per_process: int = 0
    args: Tuple = Field(default_factory=tuple)
    kwargs: Dict = Field(default_factory=dict)
    env: Dict = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.NOT_READY
    estimated_runtime: float = 0.0
    exception: Optional[str] = None  # Store exception message as string
    result: Optional[Any] = None
    cpu_affinity: List[int] = Field(default_factory=list)
    gpu_affinity: List[Union[int, str]] = Field(default_factory=list)
    run_dir: Union[str, os.PathLike] = Field(default="")
    

class TaskFactory:
    """A stateless generator of tasks from the ensemble json file"""
    
    @staticmethod
    def get_tasks(ensemble_name: str, ensemble_info: dict) -> Dict[str, Task]:
        """Return dictionary of Task objects"""
        tasks, list_options = TaskFactory._generate_ensemble(ensemble_name, ensemble_info)
        task_objects = {}
        for task_id, task_dict in tasks.items():
            # Replace placeholders in cmd_template with actual task values
            cmd = task_dict["cmd_template"]
            for option in list_options:
                cmd = cmd.replace(f"{{{option}}}", str(task_dict[option]))
            
            task_objects[task_id] = Task(
                task_id=task_dict["id"],
                nnodes=task_dict["nnodes"],
                ppn=task_dict["ppn"],
                ngpus_per_process= task_dict.get("ngpus_per_process",0),
                executable=cmd,
                env=task_dict.get("env", {}),
                run_dir=task_dict["run_dir"],
                cpu_affinity=[int(i) for i in task_dict["cpu_affinity"].split(",")] if "cpu_affinity" in task_dict else [],
                gpu_affinity=task_dict["gpu_affinity"].split(",") if "gpu_affinity" in task_dict else []
            )
        return task_objects

    @staticmethod
    def check_ensemble_info(ensemble_info: dict):
        assert "nnodes" in ensemble_info.keys()
        assert "relation" in ensemble_info.keys()
        assert "cmd_template" in ensemble_info.keys()

    @staticmethod
    def _generate_ensemble(ensemble_name: str, ensemble_info: dict) -> dict:
        """check ensemble config"""
        TaskFactory.check_ensemble_info(ensemble_info)
        ensemble = ensemble_info.copy()
        relation = ensemble["relation"]
        
        # Generate lists from linspace expressions
        for key, value in ensemble.items():
            if isinstance(value, str) and value.startswith("linspace"):
                args = eval(value[len("linspace"):])
                ensemble[key] = np.linspace(*args).tolist()
        
        if relation == "one-to-one":
            list_options = []
            non_list_options = []
            ntasks = None
            for key, value in ensemble.items():
                if isinstance(value, list):
                    list_options.append(key)
                    if ntasks is None:
                        ntasks = len(value)
                    else:
                        if len(ensemble[key]) != ntasks:
                            raise ValueError(f"Invalid option length for {key}")
                else:
                    non_list_options.append(key)
            
            tasks = []
            for i in range(ntasks):
                task = {"ensemble_name": ensemble_name}
                task["index"] = i
                for opt in non_list_options:
                    task[opt] = ensemble[opt]
                for opt in list_options:
                    task[opt] = ensemble[opt][i]
                tasks.append(TaskFactory._set_defaults(task, tuple(list_options)))
                
        elif relation == "many-to-many":
            list_options = []
            non_list_options = []
            ntasks = 1
            dim = []
            for key, value in ensemble.items():
                if isinstance(value, list):
                    list_options.append(key)
                    ntasks *= len(value)
                    dim.append(len(value))
                else:
                    non_list_options.append(key)
            
            tasks = []
            for tid in range(ntasks):
                task = {"ensemble_name": ensemble_name}
                task["index"] = tid
                loc = np.unravel_index(tid, dim)
                for id, opt in enumerate(list_options):
                    task[opt] = ensemble[opt][loc[id]]
                for opt in non_list_options:
                    task[opt] = ensemble[opt]
                tasks.append(TaskFactory._set_defaults(task, tuple(list_options)))
        else:
            raise ValueError(f"Unknown relation {relation}")

        return {task["id"]: task for task in tasks}, list_options

    @staticmethod
    def _generate_task_id(task: dict, list_options: tuple) -> str:
        bin_options_str = "-".join(f"{k}-{task[k]}" for k in list_options)
        unique_str = f"{task['ensemble_name']}-{task['index']}-{bin_options_str}"
        return unique_str
    
    @staticmethod
    def _set_defaults(task: dict, list_options: tuple) -> dict:
        task["id"] = TaskFactory._generate_task_id(task, list_options)

        if "run_dir" not in task.keys():
            task["run_dir"] = os.getcwd()
        else:
            task["run_dir"] = os.path.join(os.getcwd(), task["run_dir"])

        if "ppn" not in task.keys():
            task["ppn"] = 1
        
        if "env" not in task.keys():
            task["env"] = {}
        
        return task
