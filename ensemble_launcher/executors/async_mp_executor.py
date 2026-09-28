import asyncio
import multiprocessing as mp
import os
from asyncio import Future as AsyncFuture
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import loky

from ensemble_launcher.profiling import EventRegistry, get_registry
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceCount,
    NodeResourceList,
)
from ensemble_launcher.scheduler.resource.node import GpuId

from .utils import executor_registry, run_callable_with_affinity, run_cmd


def dummy_task():
    return


def dummy_task():
    return

def init_worker():
    import sys
    worker_id = os.getpid()
    fname = os.path.join(os.getcwd(),"logs","worker_logs",f"worker_{worker_id}.log")
    os.makedirs(os.path.dirname(fname),exist_ok=True)
    sys.stdout = open(fname, "w")
    sys.stderr = sys.stdout

    

@executor_registry.register("async_processpool", type="async")
class AsyncProcessPoolExecutor(ProcessPoolExecutor):
    def __init__(
        self,
        logger: Logger,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        worker_method: str = "spawn",
        **kwargs,
    ):
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = False
        if "return_stdout" in kwargs:
            self._return_stdout = kwargs["return_stdout"]

        if worker_method == "spawn":
            mp_context = kwargs.pop("mp_context", mp.get_context("spawn"))
            super().__init__(
                mp_context=mp_context, max_workers=kwargs.get("max_workers", None), initializer=init_worker
            )
        else:
            super().__init__(max_workers=kwargs.get("max_workers", None),initializer=init_worker)
        # super().__init__()

        super().submit(dummy_task)

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
        driver_only: bool = False,
        **kwargs,
    ) -> AsyncFuture:
        if len(job_resource.nodes) > 1 and not driver_only:
            raise ValueError(
                "MultiProcessingExecutor can only execute single node tasks"
            )

        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus

        if req.gpu_count > 0:
            if isinstance(req, NodeResourceCount):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
                self.logger.warning(
                    "Received non-zero gpu request using NodeResourceCount. Oversubscribing"
                )
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env.update({self._gpu_selector: gpu_ids})

        if driver_only:
            env = dict(env)  # do not mutate the caller's dict
            env["EL_TASK_NODES"] = ",".join(str(n) for n in job_resource.nodes)
            env["EL_TASK_NNODES"] = str(len(job_resource.nodes))
            for i, r in enumerate(job_resource.resources):
                if isinstance(r, NodeResourceList):
                    env[f"EL_TASK_CPUS_{i}"] = ",".join(str(c) for c in r.cpus)
                    env[f"EL_TASK_GPUS_{i}"] = ",".join(str(g) for g in r.gpus)

        if callable(fn):
            future = super().submit(
                run_callable_with_affinity, *(fn, task_args, task_kwargs, cpu_id, env)
            )
        elif isinstance(fn, str):
            future = super().submit(
                run_cmd, *(fn, task_args, task_kwargs, cpu_id, env, self._return_stdout)
            )
        else:
            self.logger.warning("Can only excute either a str or a callable")
            return None

        return asyncio.wrap_future(future)

    def shutdown(self, wait=True, **kwargs):
        processes = dict(self._processes) if self._processes else {}
        kwargs.setdefault("cancel_futures", True)
        super().shutdown(wait=wait, **kwargs)
        for p in processes.values():
            if p.is_alive():
                p.kill()
                p.join(timeout=5.0)
            elif p.exitcode is None:
                p.join(timeout=5.0)


@executor_registry.register("async_threadpool", type="async")
class AsyncThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(
        self, logger: Logger, gpu_selector: str = "ZE_AFFINITY_MASK", **kwargs
    ):
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = False
        if "return_stdout" in kwargs:
            self._return_stdout = kwargs["return_stdout"]
        super().__init__(max_workers=kwargs.get("max_workers", None))
        self.logger.info("Initialized threadpool executor")

    def submit(
        self,
        job_resource: JobResource,
        fn: Union[Callable, str],
        task_args: Tuple = (),
        task_kwargs: Dict = {},
        env: Dict[str, Any] = None,
        **kwargs,
    ) -> AsyncFuture:
        if env is None:
            env = {}

        if len(job_resource.nodes) > 1 or job_resource.resources[0].cpu_count > 1:
            raise ValueError(
                "AsyncThreadPool can only execute serial tasks. Use MPI/ProcessPool for parallel tasks."
            )

        req = job_resource.resources[0]
        if isinstance(req, NodeResourceList):
            cpu_req = req.cpus
            gpu_req = req.gpus
        else:
            cpu_req = None
            gpu_req = None

        if callable(fn):
            self.logger.info
            # STRICT RULE: Threads cannot have private envs or affinity
            if env or (gpu_req is not None and len(gpu_req) > 0):
                self.logger.error(
                    "Safety Violation: Cannot set 'env' or 'gpu' constraints for a Python function"
                    "running in a ThreadPool. \n"
                    "Reason: Threads share the same process environment and affinity.\n"
                    "Solution: Use 'ProcessPool'."
                )
                raise ValueError(
                    "Safety Violation: Cannot set 'env' or 'gpu' constraints for a Python function"
                    "running in a ThreadPool. \n"
                    "Reason: Threads share the same process environment and affinity.\n"
                    "Solution: Use 'ProcessPool'."
                )

            # Warn about CPU affinity if they asked for a specific core
            if cpu_req is not None:
                self.logger.warning(
                    "Ignoring CPU pinning for threaded Python task. "
                    "Threads cannot be pinned individually without affecting the whole process."
                )

            future = super().submit(fn, *task_args, **task_kwargs)

        elif isinstance(fn, str):
            # Prepare the environment (COPY it to avoid race conditions)
            task_env = os.environ.copy()
            task_env.update(env)

            # Inject GPU IDs if present (Safe here because it's a new process)
            if gpu_req is not None:
                gpu_ids = ",".join([str(g) for g in gpu_req])
                task_env[self._gpu_selector] = gpu_ids

            # We can also respect CPU affinity using `taskset` inside run_cmd if you implemented that support
            future = super().submit(
                run_cmd,
                fn,
                task_args,
                task_kwargs,
                cpu_req,
                task_env,
                self._return_stdout,
            )

        else:
            raise TypeError(f"Task must be str or callable, got {type(fn)}")

        return asyncio.wrap_future(future)


@executor_registry.register("async_loky", type="async")
class AsyncLokyExecutor:
    def __init__(
        self, logger: Logger, gpu_selector: str = "ZE_AFFINITY_MASK", **kwargs
    ):
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = kwargs.pop("return_stdout", False)
        max_workers = kwargs.pop("max_workers", None)
        self._timeout = kwargs.pop("timeout", 300)
        self._executor = loky.get_reusable_executor(
            max_workers=max_workers, timeout=self._timeout
        )
        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry = get_registry()
        self.logger.info("Initialized AsyncLoky Executor!")

    def submit(
        self,
        job_resource: JobResource,
        fn: Union[Callable, str],
        task_args: Tuple = (),
        task_kwargs: Dict = {},
        env: Dict[str, Any] = {},
        driver_only: bool = False,
        **kwargs,
    ) -> AsyncFuture:
        if len(job_resource.nodes) > 1 and not driver_only:
            raise ValueError("AsyncLokyExecutor can only execute single node tasks")

        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus

        if req.gpu_count > 0:
            if isinstance(req, NodeResourceCount):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
                self.logger.warning(
                    "Received non-zero gpu request using NodeResourceCount. Oversubscribing"
                )
            elif isinstance(req, NodeResourceList):
                gpu_ids = ",".join([str(gpu) for gpu in req.gpus])
            env.update({self._gpu_selector: gpu_ids})

        if driver_only:
            env = dict(env)  # do not mutate the caller's dict
            env["EL_TASK_NODES"] = ",".join(str(n) for n in job_resource.nodes)
            env["EL_TASK_NNODES"] = str(len(job_resource.nodes))
            for i, r in enumerate(job_resource.resources):
                if isinstance(r, NodeResourceList):
                    env[f"EL_TASK_CPUS_{i}"] = ",".join(str(c) for c in r.cpus)
                    env[f"EL_TASK_GPUS_{i}"] = ",".join(str(g) for g in r.gpus)

        if callable(fn):
            future = self._executor.submit(
                run_callable_with_affinity, *(fn, task_args, task_kwargs, cpu_id, env)
            )
        elif isinstance(fn, str):
            future = self._executor.submit(
                run_cmd, *(fn, task_args, task_kwargs, cpu_id, env, self._return_stdout)
            )
        else:
            self.logger.warning("Can only execute either a str or a callable")
            return None

        return asyncio.wrap_future(future)

    def shutdown(self, wait: bool = True, kill_workers: bool = False):
        self._executor.shutdown(wait=wait, kill_workers=kill_workers)


@executor_registry.register("async_pinned", type="async")
class AsyncPinnedExecutor:
    """A loky pool per GPU, each pinned to its device at worker creation.

    ``AsyncLokyExecutor`` sets the GPU selector per task, inside the worker, via
    :func:`run_callable_with_affinity`. That works exactly once: CUDA (and Level Zero) read
    device enumeration when the context is initialised and never look again, so every task
    after the first on a given worker keeps the device the first one bound. With a reusing
    pool, every task on the node ends up on whichever GPU happened to be granted first --
    measured on a 128-node MOFA run as 8 processes on GPU 0, memory climbing, the other three
    GPUs idle a third of the time, and not one of 1225 task pids ever on a second device.

    The fix is to bind the device before the worker exists.  ``loky.ProcessPoolExecutor``
    takes an ``env`` dict which it applies in the child "before any module is loaded", so a
    pool built with ``env={gpu_selector: "2"}`` has workers that can only ever see GPU 2 --
    including workers loky respawns later, after reaping them for idleness. Submitting to a
    pool is then choosing a GPU, and the scheduler's grant picks the pool.

    CPU affinity stays per task, applied by :func:`run_callable_with_affinity` as before.
    Cores are re-settable at any time, which GPUs are not, and pinning them here as well
    would fight the scheduler: on the run above only 34% of the granted (core-block, GPU)
    pairs matched any fixed block-to-GPU mapping. Pin what is immutable, honour the
    allocator for what is not.
    """

    def __init__(
        self,
        logger: Logger,
        gpu_selector: str = "ZE_AFFINITY_MASK",
        gpus: Optional[Sequence[GpuId]] = None,
        **kwargs,
    ):
        """
        Args:
            logger: Logger for this executor
            gpu_selector: Environment variable a task's device is pinned through,
                ``CUDA_VISIBLE_DEVICES`` or ``ZE_AFFINITY_MASK``
            gpus: Devices to build pools for, spelled as the system spells them -- ints on
                polaris, numeric strings on aurora -- since a grant is matched against these
                exactly. Defaults to the ids of this worker's own node allocation, which the
                orchestrator passes in, so it matches by construction.
            max_workers: Workers across all pools, distributed round-robin over them.
                Defaults to one per GPU. Note this is a very different default from the other
                pool executors, which size themselves by core count: a pool per GPU sized by
                cores is the pile-up this executor exists to prevent.
            timeout: Seconds an idle worker survives before loky reaps it
            return_stdout: Whether a command task's stdout is captured
        """
        self.logger = logger
        self._gpu_selector = gpu_selector
        self._return_stdout = kwargs.pop("return_stdout", False)
        self._timeout = kwargs.pop("timeout", 300)
        max_workers = kwargs.pop("max_workers", None)

        # Ids stay in whatever type the system describes them with -- polaris ints, aurora
        # numeric strings -- because that is what a grant's `req.gpus` will hold and what
        # `_pool_of_gpu` is looked up by. `str()` goes on the env var value alone, below.
        self._gpus: List[GpuId] = list(gpus or [])
        if not self._gpus:
            # No GPUs in the allocation: one unpinned pool, so a CPU-only worker configured
            # with this executor still runs rather than failing to construct.
            self.logger.warning(
                "AsyncPinnedExecutor built with no GPUs. Falling back to a single "
                "unpinned pool; tasks requesting a GPU will be rejected."
            )

        npools = max(1, len(self._gpus))
        if max_workers is None:
            max_workers = npools
        max_workers = max(npools, int(max_workers))

        # Spread the workers over the pools, remainder to the low-numbered ones, so
        # max_workers > ngpus round-robins rather than piling onto one device.
        self._pools: List[loky.ProcessPoolExecutor] = []
        for i in range(npools):
            per_pool = max_workers // npools + (1 if i < max_workers % npools else 0)
            env = {gpu_selector: str(self._gpus[i])} if self._gpus else {}
            self._pools.append(
                loky.ProcessPoolExecutor(
                    max_workers=per_pool, timeout=self._timeout, env=env
                )
            )

        self._pool_of_gpu: Dict[GpuId, int] = {
            gpu: i for i, gpu in enumerate(self._gpus)
        }
        self._next_pool = 0  # round-robin cursor for tasks that ask for no GPU

        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self._event_registry = get_registry()
        self.logger.info(
            f"Initialized AsyncPinned Executor! {npools} pool(s), {max_workers} worker(s), "
            f"{gpu_selector} pinned to {self._gpus or 'nothing'}"
        )

    def _select_pool(self, req) -> loky.ProcessPoolExecutor:
        """The pool whose GPU this request was granted.

        Args:
            req: The single node's resources from the task's ``JobResource``
        Returns:
            The pool pinned to the granted device, or the next pool round-robin when the
            request asked for no GPU
        Raises:
            ValueError: The grant spans more than one device, or a device this executor has
                no pool for
        """
        gpus = list(req.gpus)

        if not gpus:
            # No device wanted. Any pool will do, so spread these over all of them rather
            # than stacking them on pool 0 behind the GPU work.
            pool = self._pools[self._next_pool]
            self._next_pool = (self._next_pool + 1) % len(self._pools)
            return pool

        # A worker sees exactly one device, so a multi-GPU task cannot run here. Fail rather
        # than silently dropping the extra devices -- silently dropping them is the bug.
        # Test the id list, not gpu_count: a fractional grant is one device with count < 1.
        if len(gpus) > 1:
            raise ValueError(
                f"AsyncPinnedExecutor can only execute single GPU tasks, got {gpus}. "
                "Use async_mpi for multi-GPU tasks."
            )

        if gpus[0] not in self._pool_of_gpu:
            raise ValueError(
                f"AsyncPinnedExecutor has no pool for GPU {gpus[0]!r}; it was built for "
                f"{self._gpus}. Note ids are matched as spelled: this system describes its "
                f"devices as {type(self._gpus[0]).__name__ if self._gpus else 'n/a'}."
            )
        return self._pools[self._pool_of_gpu[gpus[0]]]

    def submit(
        self,
        job_resource: JobResource,
        fn: Union[Callable, str],
        task_args: Tuple = (),
        task_kwargs: Dict = {},
        env: Dict[str, Any] = {},
        driver_only: bool = False,
        **kwargs,
    ) -> AsyncFuture:
        if len(job_resource.nodes) > 1 and not driver_only:
            raise ValueError("AsyncPinnedExecutor can only execute single node tasks")

        req = job_resource.resources[0]
        if isinstance(req, NodeResourceCount):
            cpu_id = None
        elif isinstance(req, NodeResourceList):
            cpu_id = req.cpus

        # Note what is *not* here: the other pool executors set the gpu selector in `env` at
        # this point. The pool's own env already holds it, set before the worker imported
        # anything, and re-setting it per task is precisely the write that does nothing.
        pool = self._select_pool(req)

        if driver_only:
            env = dict(env)  # do not mutate the caller's dict
            env["EL_TASK_NODES"] = ",".join(str(n) for n in job_resource.nodes)
            env["EL_TASK_NNODES"] = str(len(job_resource.nodes))
            for i, r in enumerate(job_resource.resources):
                if isinstance(r, NodeResourceList):
                    env[f"EL_TASK_CPUS_{i}"] = ",".join(str(c) for c in r.cpus)
                    env[f"EL_TASK_GPUS_{i}"] = ",".join(str(g) for g in r.gpus)

        if callable(fn):
            future = pool.submit(
                run_callable_with_affinity, *(fn, task_args, task_kwargs, cpu_id, env)
            )
        elif isinstance(fn, str):
            future = pool.submit(
                run_cmd, *(fn, task_args, task_kwargs, cpu_id, env, self._return_stdout)
            )
        else:
            self.logger.warning("Can only execute either a str or a callable")
            return None

        return asyncio.wrap_future(future)

    def shutdown(self, wait: bool = True, kill_workers: bool = False):
        for pool in self._pools:
            pool.shutdown(wait=wait, kill_workers=kill_workers)
