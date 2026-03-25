import asyncio
import os
from asyncio import Future as AsyncFuture
from logging import Logger
from typing import Any, Callable, Dict, Optional, Tuple, Union

from ensemble_launcher.profiling import EventRegistry, get_registry
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)

from ._targeted_pool import TargetedMPIPoolExecutor
from .utils import executor_registry, run_callable_with_env


@executor_registry.register("async_mpi_processpool", type="async")
class AsyncMPIPoolExecutor(TargetedMPIPoolExecutor):
    def __init__(
        self,
        logger: Logger,
        mpi_info: Dict[str, str],
        gpu_selector: str = "ZE_AFFINITY_MASK",
        **kwargs,
    ):
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = False
        if "return_stdout" in kwargs:
            self._return_stdout = kwargs["return_stdout"]
            del kwargs["return_stdout"]

        self._mpi_info: Dict[str, str] = mpi_info

        hosts = self._mpi_info["--host"].split(",")
        ppn = int(self._mpi_info.get("-ppn", "0"))
        np = int(self._mpi_info.get("-np", len(hosts) * ppn))

        if ppn == 0 and np == 0:
            raise ValueError("mpi info needs either ppn or np")

        if ppn == 0:
            ppn = np // len(hosts)

        if np == 0:
            np = ppn * len(hosts)
            self._mpi_info["-np"] = str(np)

        cpu_binding = self._mpi_info.get("--cpu-bind", None)
        cpu_binding = (
            map(int, cpu_binding.split("list:")[-1].split(":"))
            if cpu_binding is not None
            else list(range(ppn))
        )

        # get cpu to pid mapping
        self._cpu_to_pid = {
            (hname, cpu_binding[local_rank]): hid * ppn + local_rank
            for hid, hname in enumerate(hosts)
            for local_rank in range(ppn)
        }

        super().__init__(max_workers=np, mpi_info=self._mpi_info, **kwargs)

        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry: EventRegistry = get_registry()
        self.logger.info("Initialized AsyncProcessPool Executor!")

    def submit(
        self,
        job_resource: JobResource,
        fn: Union[Callable, str],
        task_args: Tuple = (),
        task_kwargs: Dict = {},
        env: Dict[str, Any] = {},
        **kwargs,
    ) -> AsyncFuture:

        np = sum([res.cpu_count for res in job_resource.resources])
        if np > 1:
            raise ValueError("AsyncMPIExecutor can only execute serial tasks")

        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus[0]

        if cpu_id is None:
            raise ValueError(
                "Not setting cpu affinity could cause deadlocks. Raising error....."
            )

        if req.gpu_count > 0:
            if isinstance(req, NodeResourceCount):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpu_count])
                self.logger.warning(
                    "Received non-zero gpu request using NodeResourceCount. Oversubscribing"
                )
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env.update({self._gpu_selector: gpu_ids})

        cpu = (job_resource.nodes[0], cpu_id)
        if callable(fn):
            future = super().submit_to(
                self._cpu_to_pid[cpu],
                run_callable_with_env,
                *(fn, task_args, task_kwargs, env),
            )
        # elif isinstance(fn, str):
        #     future = super().submit(
        #         run_cmd, *(fn, task_args, task_kwargs, cpu_id, env, self._return_stdout)
        #     )
        else:
            self.logger.warning("Can only excute either a callable")
            return None

        return asyncio.wrap_future(future)
