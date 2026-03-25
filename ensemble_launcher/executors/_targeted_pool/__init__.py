"""mpi4py.futures with targeted worker dispatch."""

import collections
import sys

from mpi4py import MPI
from mpi4py.futures import (
    ALL_COMPLETED,
    FIRST_COMPLETED,
    FIRST_EXCEPTION,
    BrokenExecutor,
    CancelledError,
    Executor,
    Future,
    InvalidStateError,
    MPICommExecutor,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    TimeoutError,  # noqa: A004
    _core,
    as_completed,
    collect,
    compose,
    get_comm_workers,
    wait,
)
from mpi4py.futures._core import (
    Backoff,
    Pool,
    _get_request,
    _getopt_backoff,
    client_connect,
    client_init,
    client_spawn,
    client_stop,
    client_sync,
    serialized,
    sys_exception,
)
from mpi4py.futures.pool import MPIPoolExecutor

__all__ = [
    "Future",
    "Executor",
    "wait",
    "FIRST_COMPLETED",
    "FIRST_EXCEPTION",
    "ALL_COMPLETED",
    "as_completed",
    "CancelledError",
    "TimeoutError",
    "InvalidStateError",
    "BrokenExecutor",
    "TargetedMPIPoolExecutor",
    "MPIPoolExecutor",
    "MPICommExecutor",
    "ThreadPoolExecutor",
    "ProcessPoolExecutor",
    "get_comm_workers",
    "collect",
    "compose",
]


class WorkerSet(collections.deque):
    add = collections.deque.append
    pop = collections.deque.popleft

    def pop_specific(self, pid):
        self.remove(pid)
        return pid

    def __contains__(self, pid):
        return collections.deque.__contains__(self, pid)


def client_exec(comm, options, tag, worker_set, task_queue):
    backoff = Backoff(_getopt_backoff(options))

    status = MPI.Status()
    comm_recv = serialized(comm.recv)
    comm_isend = serialized(comm.issend)
    comm_iprobe = serialized(comm.iprobe)
    request_free = serialized(_get_request(comm).Free)

    pending = {}

    def iprobe():
        return comm_iprobe(MPI.ANY_SOURCE, tag, status)

    def probe():
        backoff.reset()
        while not comm_iprobe(MPI.ANY_SOURCE, tag, status):
            backoff.sleep()

    def recv():
        try:
            task = comm_recv(None, MPI.ANY_SOURCE, tag, status)
        except BaseException:
            task = (None, sys_exception())
        pid = status.source
        worker_set.add(pid)

        future, request = pending.pop(pid)
        request_free(request)
        result, exception = task
        if exception is None:
            future.set_result(result)
        else:
            future.set_exception(exception)

        del result, exception, future, task

    def send():
        try:
            item = task_queue.pop()
        except LookupError:
            return False

        if item is None:
            return True  # stop signal; no worker was consumed

        future, task, target_pid = item

        if target_pid is not None:
            if target_pid not in worker_set:
                task_queue.add(item)  # put back at front of queue
                return False
            worker_set.pop_specific(target_pid)
            pid = target_pid
        else:
            try:
                pid = worker_set.pop()
            except LookupError:
                task_queue.add(item)  # put back at front of queue
                return False

        if not future.set_running_or_notify_cancel():
            worker_set.add(pid)
            return False

        try:
            request = comm_isend(task, pid, tag)
            pending[pid] = (future, request)
        except BaseException:
            worker_set.add(pid)
            future.set_exception(sys_exception())

        del future, task, item
        return None

    while True:
        if task_queue and worker_set:
            backoff.reset()
            stop = send()
            if stop:
                break
        if pending and iprobe():
            backoff.reset()
            recv()
        backoff.sleep()

    while pending:
        probe()
        recv()


def _manager_comm(pool, options, comm, sync=True):
    assert comm != MPI.COMM_NULL  # noqa: S101
    assert comm.Is_inter()  # noqa: S101
    assert comm.Get_size() == 1  # noqa: S101
    comm = client_sync(comm, options, sync)
    if not client_init(comm, options):
        pool.broken("initializer failed")
        client_stop(comm)
        return
    size = comm.Get_remote_size()
    queue = pool.setup(size)
    workers = WorkerSet(range(size))
    client_exec(comm, options, 0, workers, queue)
    client_stop(comm)


def _manager_spawn(pool, options):
    pyexe = options.pop("python_exe", None)
    pyargs = options.pop("python_args", None)
    nprocs = options.pop("max_workers", None)
    info = options.pop("mpi_info", None)
    comm = serialized(client_spawn)(pyexe, pyargs, nprocs, info)
    _manager_comm(pool, options, comm)


def _manager_service(pool, options):
    service = options.pop("service", None)
    info = options.pop("mpi_info", None)
    comm = serialized(client_connect)(service, info)
    _manager_comm(pool, options, comm)


def _TargetedSpawnPool(executor):
    return Pool(executor, _manager_spawn)


def _TargetedServicePool(executor):
    return Pool(executor, _manager_service)


def _TargetedWorkerPool(executor):
    if _core.SharedPool is not None:
        # Shared pool mode (python -m mpi4py.futures): targeted dispatch
        # is not supported; fall back to standard pool.
        return _core.SharedPool(executor)
    if "service" in executor._options:
        return _TargetedServicePool(executor)
    return _TargetedSpawnPool(executor)


class TargetedMPIPoolExecutor(MPIPoolExecutor):
    """MPIPoolExecutor that supports pinning tasks to specific worker ranks.

    Use ``submit_to(worker_id, fn, ...)`` to send a task to a specific worker.
    Use ``submit(fn, ...)`` as usual for any-worker dispatch.

    Worker IDs are 0-based MPI ranks within the spawned worker pool.

    Note:
        Targeted tasks at the head of the queue will block untargeted tasks
        behind them if the target worker is busy.  If you need finer-grained
        scheduling, consider maintaining per-worker queues externally.
    """

    _make_pool = staticmethod(_TargetedWorkerPool)

    def submit(self, fn, /, *args, **kwargs):
        """Submit to any available worker."""
        with self._lock:
            if self._broken:
                raise _core.BrokenExecutor(self._broken)
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            self._bootstrap()
            future = self.Future()
            task = (fn, args, kwargs)
            self._pool.push((future, task, None))
            return future

    def submit_to(self, worker_id: int, fn, /, *args, **kwargs):
        """Submit a callable to a specific worker by MPI rank.

        Args:
            worker_id: Target worker rank (0-based, must be < num_workers).
            fn: Callable to execute.
            *args, **kwargs: Arguments forwarded to fn.

        Returns:
            A Future representing the result of fn(*args, **kwargs).
        """
        with self._lock:
            if self._broken:
                raise _core.BrokenExecutor(self._broken)
            if self._shutdown:
                raise RuntimeError("cannot submit after shutdown")
            self._bootstrap()
            future = self.Future()
            task = (fn, args, kwargs)
            self._pool.push((future, task, worker_id))
            return future
