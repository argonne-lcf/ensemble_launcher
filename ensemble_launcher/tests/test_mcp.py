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


async def call_string_tool(task_id: str):
    server_params = StdioServerParameters(
        command="python3", args=["start_mcp.py"], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("py_echo", arguments={"task_id": task_id})
            return result


async def call_string_ensemble_tool(task_ids: list):
    server_params = StdioServerParameters(
        command="python3", args=["start_mcp.py"], env=os.environ.copy()
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "ensemble_py_echo", arguments={"task_id": task_ids}
            )
            return result


def test_mcp_string():
    try:
        task_id = "hello"
        result = asyncio.run(call_string_tool(task_id))
        stdout = result.content[0].text
        assert f"Hello from task task-{task_id}" in stdout, (
            f"Unexpected output: {stdout!r}"
        )

        task_ids = ["a", "b", "c"]
        ensemble_result = asyncio.run(call_string_ensemble_tool(task_ids))
        outputs = ensemble_result.structuredContent["result"]
        assert len(outputs) == len(task_ids)
        for tid, out in zip(task_ids, outputs):
            assert f"Hello from task task-{tid}" in out, f"Unexpected output: {out!r}"

        print("All string tool tests passed")

    except Exception as e:
        print(f"String tool test failed: {e}")
        raise


if __name__ == "__main__":
    # test_mcp()
    test_mcp_string()
