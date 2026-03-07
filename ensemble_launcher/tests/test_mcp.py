import asyncio
import math
import os

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from utils import async_compute_density, compute_density


async def call_tools():
    server_params = StdioServerParameters(
        command="python3", args=["start_mcp.py"], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "compute_density", arguments={"Temperature": 1.0, "Pressure": 1.0}
            )
            return result


async def call_ensemble_tools():
    server_params = StdioServerParameters(
        command="python3", args=["start_mcp.py"], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "ensemble_compute_density",
                arguments={
                    "Temperature": [1.0, 1.0, 1.0],
                    "Pressure": [1.0, 1.0, 1.0],
                },
            )
            return result


async def call_async_tools():
    server_params = StdioServerParameters(
        command="python3", args=["start_mcp.py"], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "async_compute_density",
                arguments={"Temperature": 1.0, "Pressure": 1.0},
            )
            return result


async def call_async_ensemble_tools():
    server_params = StdioServerParameters(
        command="python3", args=["start_mcp.py"], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "ensemble_async_compute_density",
                arguments={
                    "Temperature": [1.0, 1.0, 1.0],
                    "Pressure": [1.0, 1.0, 1.0],
                },
            )
            return result


def test_mcp():
    try:
        result = asyncio.run(call_tools())
        ensemble_result = asyncio.run(call_ensemble_tools())
        direct_result = compute_density(1.0, 1.0)

        value = float(result.content[0].text)
        assert math.isclose(value, direct_result), f"{value} != {direct_result}"
        assert all(
            [
                math.isclose(result, direct_result)
                for result in ensemble_result.structuredContent["result"]
            ]
        )
        print("All tests passed")

    except Exception as e:
        print(f"Test failed: {e}")
        raise


def test_mcp_async():
    try:
        direct_result = asyncio.run(async_compute_density(1.0, 1.0))

        result = asyncio.run(call_async_tools())
        value = float(result.content[0].text)
        assert math.isclose(value, direct_result), f"{value} != {direct_result}"

        ensemble_result = asyncio.run(call_async_ensemble_tools())
        assert all(
            [
                math.isclose(r, direct_result)
                for r in ensemble_result.structuredContent["result"]
            ]
        )
        print("All async tests passed")

    except Exception as e:
        print(f"Async test failed: {e}")
        raise


if __name__ == "__main__":
    test_mcp()
