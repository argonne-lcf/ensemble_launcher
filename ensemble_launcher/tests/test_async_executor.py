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


@pytest.mark.asyncio
async def test_async_mpi_pool():

    mpi_config = MPIConfig(
        processes_per_node_flag=None, hosts_flag=None, cpu_bind_method="none"
    )
    mpi_info = {}
    mpi_info["np"] = 12
    cpu_to_pid = {(socket.gethostname(), i): i for i in range(12)}

    job_resource = JobResource(
        resources=[NodeResourceList(cpus=(1,))], nodes=[socket.gethostname()]
    )

    logger = setup_logger(name="test_mpi_pool", log_dir="logs")
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
