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
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ensemble_launcher.scheduler.resource import JobResource


class TaskStatus(enum.Enum):
    NOT_READY = "not_ready"
    READY = "ready"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"


UID_BYTES = 16

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
    driver_only: bool = False
    """Reserve nnodes but run a single driver process on one node.
    The driver launches across the reservation itself (e.g. an ASE calculator
    that shells out to mpiexec). The full reservation — node list and per-node
    CPU/GPU ids — is exported to the driver through the environment."""

    uid: bytes = Field(default_factory=lambda: os.urandom(UID_BYTES))
    """Internal 16-byte id EL routes and stores the task by. Set when the Task is created;
    ``task_id`` stays the user's name for it and need not be unique."""

    def get_resource_requirements(self) -> "JobResource":
        """Build JobResource requirements from this Task."""
        return resource_requirements(
            self.nnodes, self.ppn, self.ngpus_per_process, self.cpu_affinity, self.gpu_affinity
        )


def resource_requirements(
    nnodes: int,
    ppn: int,
    ngpus_per_process: Union[int, float],
    cpu_affinity: List[int],
    gpu_affinity: List[Union[int, str]],
) -> "JobResource":
    """The JobResource a task with these fields needs. Shared by `Task` and `TaskMeta`."""
    from ensemble_launcher.scheduler.resource import (
        JobResource,
        NodeResourceCount,
        NodeResourceList,
    )

    req = JobResource(
        resources=[
            NodeResourceCount(ncpus=ppn, ngpus=ngpus_per_process * ppn)
            for i in range(nnodes)
        ]
    )
    if len(cpu_affinity) > 0 or len(gpu_affinity) > 0:
        ncpus = ppn * nnodes
        ngpus = ncpus * ngpus_per_process
        if ncpus >= 1 and ngpus > 0:
            if cpu_affinity and (ngpus_per_process > 0 and not gpu_affinity):
                # Ignore cpu_affinity if gpu_affinity is not set
                return req

            if gpu_affinity and not cpu_affinity:
                # Ignore gpu_affinity if cpu_affinity is not set
                return req

            # Distribute the (possibly fractional) total GPU demand evenly
            # across the explicitly requested ids, so a fractional
            # ngpus_per_process still works with gpu_affinity set.
            gpu_amount_per_id = ngpus / len(gpu_affinity)
            req = JobResource(
                resources=[
                    NodeResourceList.request(
                        cpus=tuple(cpu_affinity),
                        gpus=gpu_affinity,
                        gpu_fraction=gpu_amount_per_id,
                    )
                    for node in range(nnodes)
                ]
            )
        elif ncpus >= 1 and ngpus == 0:
            if len(cpu_affinity) == ppn:
                req = JobResource(
                    resources=[
                        NodeResourceList(cpus=tuple(cpu_affinity))
                        for node in range(nnodes)
                    ]
                )
            else:
                # This is a multi threading case where we might need more physical core per process
                pass
        else:
            pass

    return req


# --------------------------------------------------------------------------- #
#  Per-task encoding, used by TaskUpdate (comm/messages.py)                    #
# --------------------------------------------------------------------------- #

# Every field except the deep ones and uid, in a fixed order. A task's metadata record is
# pickle(tuple of these values); policies and schedulers read them through TaskMeta.
META_FIELDS: Tuple[str, ...] = tuple(
    f for f in Task.model_fields if f not in _DEEP_FIELDS and f != "uid"
)


def encode_meta(task: Task) -> bytes:
    """A task's metadata record: everything a policy may read, nothing it may not."""
    d = task.__dict__
    return pickle.dumps(tuple(d[f] for f in META_FIELDS), protocol=pickle.HIGHEST_PROTOCOL)


def encode_deep(task: Task) -> bytes:
    """A task's deep record (executable, args, kwargs, env, result), as `Task.pack` used to."""
    d = task.__dict__
    if task.serialize_executable_by_value:
        return cloudpickle.dumps({f: d[f] for f in _DEEP_FIELDS})
    exec_blob = pickle.dumps(d["executable"])
    rest_blob = cloudpickle.dumps({f: d[f] for f in _DEEP_FIELDS_NO_EXEC})
    return struct.pack("!I", len(exec_blob)) + exec_blob + rest_blob


def decode_deep(buf, by_value: bool) -> Dict[str, Any]:
    if by_value:
        return cloudpickle.loads(buf)
    (exec_len,) = struct.unpack_from("!I", buf, 0)
    executable = pickle.loads(buf[4 : 4 + exec_len])
    deep = cloudpickle.loads(buf[4 + exec_len :])
    deep["executable"] = executable
    return deep


def build_task(uid: bytes, meta_values: tuple, deep: Dict[str, Any]) -> Task:
    """A full Task from its decoded records, skipping validation (it was validated when created)."""
    fields = dict(zip(META_FIELDS, meta_values))
    fields.update(deep)
    fields["uid"] = uid
    return Task.model_construct(**fields)


class TaskMeta:
    """Read-only view of one task's non-deep fields, decoded from its metadata record.

    Built on demand (``store[uid]``) and meant to be dropped right after use: holding many of
    these alive would bring back the per-task objects the blob format exists to avoid.
    """

    __slots__ = ("uid", "_values")

    def __init__(self, uid: bytes, values: tuple):
        self.uid = uid
        self._values = values  # in META_FIELDS order; each field is a property below

    def get_resource_requirements(self) -> "JobResource":
        return resource_requirements(
            self.nnodes, self.ppn, self.ngpus_per_process, self.cpu_affinity, self.gpu_affinity
        )

    def __repr__(self) -> str:
        return f"TaskMeta(uid={self.uid.hex()}, task_id={self.task_id!r}, nnodes={self.nnodes}, ppn={self.ppn})"


def _meta_property(i: int, name: str) -> property:
    return property(lambda self: self._values[i], doc=f"Task.{name}")


for _i, _name in enumerate(META_FIELDS):
    setattr(TaskMeta, _name, _meta_property(_i, _name))
del _i, _name


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
