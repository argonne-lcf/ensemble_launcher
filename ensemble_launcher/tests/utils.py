import asyncio
import time


def echo(task_id: str):
    return f"Hello from task {task_id}"


def echo_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    return comm.rank


def echo_stdout(task_id: str):
    print(f"Hello from task {task_id}")


def echo_sleep(task_id: str, sleep_time: float = 0.0):
    time.sleep(sleep_time)
    return f"Hello from task {task_id}"


def compute_density(Temperature: float, Pressure: float) -> float:
    """
    Computes density of Air from the temperature (K) and pressure (Pa) from ideal gas law
    """
    R_specific = 8.314 / 28.96e-3
    return Pressure / R_specific / Temperature


async def async_compute_density(Temperature: float, Pressure: float) -> float:
    """
    Async version of compute_density.
    Computes density of Air from the temperature (K) and pressure (Pa) from ideal gas law
    """
    await asyncio.sleep(0)
    R_specific = 8.314 / 28.96e-3
    return Pressure / R_specific / Temperature
