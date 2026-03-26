import asyncio
import os
import socket
import sys

import pytest
from ensemble_launcher.executors import AsyncMPIPoolExecutor
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)
from test_helpers import echo_mpi


@pytest.mark.asyncio
async def test_mpi_pool():
    logger = setup_logger("mpi_executor", log_dir="logs")
    mpi_info = {}
    mpi_info["-np"] = "12"
    mpi_info["--cpu-bind"] = "list:1:2:3:4:5:6:7:8:9:10:11:12"
    mpi_info["--ppn"] = "12"
    mpi_info["--hosts"] = f"{socket.gethostname()}"
    executor = AsyncMPIPoolExecutor(logger, mpi_info)

    cpu_id = 4
    expected_rank = 3
    res = JobResource(
        resources=[NodeResourceList(cpus=(cpu_id,))], nodes=[socket.gethostname()]
    )

    future = executor.submit(res, echo_mpi)

    result = await future
    assert result == expected_rank, f"expected 4 got {result}"


if __name__ == "__main__":
    os.environ.update({"PYTHONPATH": str(os.getcwd())})
    asyncio.run(test_mpi_pool())
