import asyncio
import socket

import pytest
from utils import echo_mpi

from ensemble_launcher.config import MPIConfig
from ensemble_launcher.executors import AsyncMPIPoolExecutor
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)

pytestmark = pytest.mark.core


@pytest.mark.asyncio
async def test_async_mpi_pool():

    mpi_config = MPIConfig(
        flavor="test",
        extra_launcher_flags=["--oversubscribe", "--allow-run-as-root"],
    )
    mpi_info = {}
    mpi_info["np"] = 12
    cpu_to_pid = {(socket.gethostname(), i): i for i in range(12)}

    job_resource = JobResource(
        resources=[NodeResourceList(cpus=(1,))], nodes=[socket.gethostname()]
    )

    logger = setup_logger(name="test_mpi_pool")
    exec = AsyncMPIPoolExecutor(
        logger, cpu_to_pid=cpu_to_pid, mpi_info=mpi_info, mpi_config=mpi_config
    )
    future = exec.submit(job_resource, echo_mpi)
    try:
        result = await future
    finally:
        await exec.ashutdown()

    assert result == 1


if __name__ == "__main__":
    asyncio.run(test_async_mpi_pool())


def _touch_after(path: str, delay: float) -> None:
    import time

    time.sleep(delay)
    with open(path, "w") as f:
        f.write("done")


def _pool_owner(q, path):
    """Own an AsyncProcessPoolExecutor running one slow job; report its workers, then wait
    to be SIGKILLed."""
    import time

    from ensemble_launcher.executors.async_mp_executor import AsyncProcessPoolExecutor

    async def main():
        ex = AsyncProcessPoolExecutor(setup_logger("pool-owner", "pool-owner"), max_workers=2)
        jr = JobResource(resources=[NodeResourceList(cpus=(1,))], nodes=[socket.gethostname()])
        ex.submit(jr, _touch_after, task_args=(path, 3.0))
        await asyncio.sleep(1.0)  # the job is running
        q.put(list(ex._processes))
        time.sleep(60)

    asyncio.run(main())


def test_pool_workers_exit_with_owner(tmp_path):
    """SIGKILL a pool's owner: its workers exit on their own once their running job is done,
    instead of living forever as orphans."""
    import multiprocessing as mp
    import os
    import signal
    import time

    path = str(tmp_path / "job_done")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    owner = ctx.Process(target=_pool_owner, args=(q, path))
    owner.start()
    workers = q.get(timeout=60)
    assert workers
    os.kill(owner.pid, signal.SIGKILL)
    owner.join(timeout=10)

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False

    deadline = time.monotonic() + 20
    while any(alive(w) for w in workers) and time.monotonic() < deadline:
        time.sleep(0.2)
    left = [w for w in workers if alive(w)]
    for w in left:
        os.kill(w, signal.SIGKILL)
    assert not left, f"pool workers outlived their owner: {left}"
    assert os.path.exists(path), "the running job was cut off instead of finishing"
