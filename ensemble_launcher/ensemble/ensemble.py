from __future__ import annotations

import asyncio
import enum
import os
import struct
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import pickle

import cloudpickle
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from ensemble_launcher.scheduler.resource import JobResource


class TaskStatus(enum.Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


_DEEP_FIELDS = ("executable", "args", "kwargs", "env", "result")
_DEEP_FIELDS_NO_EXEC = ("args", "kwargs", "env", "result")


class Task(BaseModel):
    task_id: str
    nnodes: int
    ppn: int
    executable: Union[str, Callable] = None
    ngpus_per_process: Union[int, float] = 0
    args: Tuple = Field(default_factory=tuple)
    kwargs: Dict = Field(default_factory=dict)
    env: Dict = Field(default_factory=dict)
    status: TaskStatus = TaskStatus.NOT_READY
    estimated_runtime: float = 0.0
    exception: Optional[str] = None
    result: Optional[Any] = None
    cpu_affinity: List[int] = Field(default_factory=list)
    gpu_affinity: List[Union[int, str]] = Field(default_factory=list)
    run_dir: Optional[Union[str, os.PathLike]] = None
    start_time: Any = None
    end_time: Any = None
    executor_name: Optional[str] = None
    tag: Optional[str] = None
    stdout_file: Optional[str] = None
    stderr_file: Optional[str] = None
    serialize_executable_by_value: bool = True

    _packed: bool = PrivateAttr(default=False)
    _raw_deep: Optional[bytes] = PrivateAttr(default=None)

    # ------------------------------------------------------------------ #
    #  pack / unpack                                                      #
    # ------------------------------------------------------------------ #

    @property
    def is_packed(self) -> bool:
        return self._packed

    def _check_unpacked(self, field: str) -> None:
        if self._packed:
            raise RuntimeError(
                f"Task.{field} not accessible in packed state — call unpack() first"
            )

    def pack(self) -> Task:
        if self._packed:
            return self
        if self.serialize_executable_by_value:
            self._raw_deep = cloudpickle.dumps(
                {f: getattr(self, f) for f in _DEEP_FIELDS}
            )
        else:
            exec_blob = pickle.dumps(self.executable)
            rest_blob = cloudpickle.dumps(
                {f: getattr(self, f) for f in _DEEP_FIELDS_NO_EXEC}
            )
            self._raw_deep = struct.pack("!I", len(exec_blob)) + exec_blob + rest_blob
        for f in _DEEP_FIELDS:
            object.__setattr__(self, f, None)
        self._packed = True
        return self

    def unpack(self) -> None:
        if not self._packed:
            return
        if self.serialize_executable_by_value:
            deep = cloudpickle.loads(self._raw_deep)
        else:
            (exec_len,) = struct.unpack_from("!I", self._raw_deep, 0)
            executable = pickle.loads(self._raw_deep[4 : 4 + exec_len])
            deep = cloudpickle.loads(self._raw_deep[4 + exec_len :])
            deep["executable"] = executable
        for f in _DEEP_FIELDS:
            object.__setattr__(self, f, deep[f])
        self._raw_deep = None
        self._packed = False

    # ------------------------------------------------------------------ #
    #  to_bytes / from_bytes                                              #
    # ------------------------------------------------------------------ #

    def to_byte_array(self) -> list:
        self.pack()
        shallow = cloudpickle.dumps(self.__dict__.copy())
        header = struct.pack("!I", len(shallow))
        return [header, shallow, self._raw_deep]

    @classmethod
    def from_byte_array(cls, frames: list) -> Task:
        shallow_dict = cloudpickle.loads(frames[1])
        deep_blob = frames[2]
        task = cls.__new__(cls)
        object.__setattr__(task, "__dict__", shallow_dict)
        object.__setattr__(task, "__pydantic_fields_set__", set())
        task.__pydantic_private__ = {"_packed": True, "_raw_deep": deep_blob}
        return task

    def to_bytes(self) -> bytes:
        return b"".join(self.to_byte_array())

    @classmethod
    def from_bytes(cls, data: bytes) -> Task:
        (shallow_len,) = struct.unpack_from("!I", data, 0)
        shallow_dict = cloudpickle.loads(data[4 : 4 + shallow_len])
        deep_blob = data[4 + shallow_len :]
        task = cls.__new__(cls)
        object.__setattr__(task, "__dict__", shallow_dict)
        object.__setattr__(task, "__pydantic_fields_set__", set())
        task.__pydantic_private__ = {"_packed": True, "_raw_deep": deep_blob}
        return task

    def get_resource_requirements(self) -> "JobResource":
        """Build JobResource requirements from this Task."""
        from ensemble_launcher.scheduler.resource import (
            JobResource,
            NodeResourceCount,
            NodeResourceList,
        )

        req = JobResource(
            resources=[
                NodeResourceCount(
                    ncpus=self.ppn, ngpus=int(self.ngpus_per_process * self.ppn)
                )
                for i in range(self.nnodes)
            ]
        )
        if len(self.cpu_affinity) > 0 or len(self.gpu_affinity) > 0:
            ncpus = self.ppn * self.nnodes
            ngpus = int(ncpus * self.ngpus_per_process)
            if ncpus >= 1 and ngpus >= 1:
                if self.cpu_affinity and (
                    self.ngpus_per_process > 0 and not self.gpu_affinity
                ):
                    # Ignore cpu_affinity if gpu_affinity is not set
                    return req

                if self.gpu_affinity and not self.cpu_affinity:
                    # Ignore gpu_affinity if cpu_affinity is not set
                    return req

                req = JobResource(
                    resources=[
                        NodeResourceList(
                            cpus=tuple(self.cpu_affinity), gpus=tuple(self.gpu_affinity)
                        )
                        for node in range(self.nnodes)
                    ]
                )
            elif ncpus >= 1 and ngpus == 0:
                if len(self.cpu_affinity) == self.ppn:
                    req = JobResource(
                        resources=[
                            NodeResourceList(
                                cpus=tuple(self.cpu_affinity),
                            )
                            for node in range(self.nnodes)
                        ]
                    )
                else:
                    # This is a multi threading case where we might need more physical core per process
                    pass
            else:
                pass

        return req


_TASK_REQUIRED = {"task_id", "nnodes", "ppn", "executable"}
TaskKwargs = TypedDict(
    "TaskKwargs",
    {k: v.annotation for k, v in Task.model_fields.items() if k not in _TASK_REQUIRED},
    total=False,
)


class _AsyncWrapper:
    """Picklable sync wrapper around an ``async def`` callable.

    A nested closure (e.g. defined inside ``model_post_init``) cannot be
    pickled by the standard ``pickle`` module used by ``ProcessPoolExecutor``.
    A top-level class instance is picklable as long as the wrapped function is
    also picklable (i.e. defined at module level).
    """

    __slots__ = ("_fn", "_loop")

    def __init__(self, fn: Callable, loop: Optional[Any] = None) -> None:
        self._fn = fn
        self._loop = loop

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._loop is not None:
            return self._loop.run_until_complete(self._fn(*args, **kwargs))
        from ensemble_launcher.loop import run
        return run(self._fn(*args, **kwargs))


class AsyncTask(Task):
    """A Task whose executable is an ``async def`` callable.

    After initialisation ``executable`` is transparently replaced with a sync
    wrapper so the task works with ``AsyncProcessPoolExecutor`` without any
    changes to the worker.

    If ``loop`` is set the wrapper calls ``loop.run_until_complete``; otherwise
    it uses ``asyncio.run``.

    Usage::

        async def my_sim(x: float) -> float:
            await asyncio.sleep(0)
            return x ** 2

        task = AsyncTask(task_id="t0", nnodes=1, ppn=1,
                         executable=my_sim, args=(3.0,))

        el = EnsembleLauncher(ensemble_file={"t0": task}, ...)
    """

    loop: Optional[Any] = None

    def model_post_init(self, _: Any) -> None:
        object.__setattr__(
            self, "executable", _AsyncWrapper(self.executable, self.loop)
        )


class TaskFactory:
    """A stateless generator of tasks from the ensemble json file"""

    @staticmethod
    def get_tasks(ensemble_name: str, ensemble_info: dict) -> Dict[str, Task]:
        """Return dictionary of Task objects"""
        tasks, list_options = TaskFactory._generate_ensemble(
            ensemble_name, ensemble_info
        )
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
                ngpus_per_process=task_dict.get("ngpus_per_process", 0),
                executable=cmd,
                env=task_dict.get("env", {}),
                run_dir=task_dict["run_dir"],
                cpu_affinity=[int(i) for i in task_dict["cpu_affinity"].split(",")]
                if "cpu_affinity" in task_dict
                else [],
                gpu_affinity=task_dict["gpu_affinity"].split(",")
                if "gpu_affinity" in task_dict
                else [],
                stdout_file=task_dict.get("stdout_file", None),
                stderr_file=task_dict.get("stderr_file", None),
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
                args = eval(value[len("linspace") :])
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
