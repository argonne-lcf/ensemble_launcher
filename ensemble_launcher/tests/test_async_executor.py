import asyncio
import socket

from utils import echo_mpi

from ensemble_launcher.executors import AsyncMPIPoolExecutor
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)


async def test_mpi_pool():
    logger = setup_logger("mpi_executor", log_dir="logs")
    mpi_info = {}
    mpi_info["-np"] = "12"
    mpi_info["--host"] = f"{socket.gethostname()}"
    executor = AsyncMPIPoolExecutor(logger, mpi_info)

    res = JobResource(
        resources=[NodeResourceList(cpus=(4,))], nodes=[socket.gethostname()]
    )
    future = executor.submit(res, echo_mpi)

    result = await future
    assert result == 4


if __name__ == "__main__":
    asyncio.run(test_mpi_pool())
