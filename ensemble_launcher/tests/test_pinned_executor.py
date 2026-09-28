"""Tests for ``async_pinned`` (:class:`AsyncPinnedExecutor`).

The point of this executor is that a worker's GPU is fixed before the worker imports anything,
and stays fixed for the worker's whole life. That is a property of real child processes, so the
tests here actually spawn pools and read the environment variable back from inside them --
``CUDA_VISIBLE_DEVICES`` rather than a real device, so they need no GPU.
"""

import asyncio
import logging
import os

import pytest

from ensemble_launcher.executors import AsyncPinnedExecutor
from ensemble_launcher.scheduler.resource import (
    JobResource,
    NodeResourceList,
)

pytestmark = pytest.mark.core

SELECTOR = "CUDA_VISIBLE_DEVICES"


def _probe():
    """Report which worker ran this and what device it can see."""
    return os.getpid(), os.environ.get(SELECTOR)


def _job(cpus, gpus, host="node0"):
    return JobResource(
        resources=[NodeResourceList(cpus=tuple(cpus), gpus=tuple(gpus))],
        nodes=[host],
    )


def _run(executor, jobs):
    async def _gather():
        return await asyncio.gather(
            *[executor.submit(job, _probe) for job in jobs]
        )

    return asyncio.run(_gather())


@pytest.fixture
def logger():
    return logging.getLogger("test_pinned")


@pytest.fixture
def executor(logger):
    ex = AsyncPinnedExecutor(
        logger=logger, gpu_selector=SELECTOR, gpus=[0, 1, 2, 3], timeout=30
    )
    yield ex
    ex.shutdown(wait=True, kill_workers=True)


# ------------------------------------------------------------------ #
#  The pinning itself                                                  #
# ------------------------------------------------------------------ #


def test_task_runs_on_the_gpu_it_was_granted(executor):
    """Each grant must land in the pool pinned to that device."""
    jobs = [_job(range(i * 8, i * 8 + 8), [i]) for i in range(4)]
    for job, expected in zip(jobs, "0123"):
        (_, seen), = _run(executor, [job])
        assert seen == expected


def test_pinning_survives_worker_reuse(executor):
    """The regression this executor exists for.

    ``AsyncLokyExecutor`` passes the device per task into a reused worker, where CUDA has
    already read its enumeration and will not read it again. Here the device belongs to the
    worker, so a second round on the same pids still sees the right one.
    """
    jobs = [_job(range(i * 8, i * 8 + 8), [i]) for i in range(4)]
    first = _run(executor, jobs)
    second = _run(executor, jobs)

    assert [gpu for _, gpu in first] == ["0", "1", "2", "3"]
    assert [gpu for _, gpu in second] == ["0", "1", "2", "3"]
    # Same workers the second time round -- this is reuse, not fresh processes.
    assert {pid for pid, _ in first} == {pid for pid, _ in second}


def test_every_gpu_gets_its_own_workers(executor):
    """No worker may serve two devices, which is what the 8-on-GPU-0 pile-up looked like."""
    jobs = [_job(range(i * 8, i * 8 + 8), [i]) for i in range(4)] * 3
    results = _run(executor, jobs)

    by_pid = {}
    for pid, gpu in results:
        by_pid.setdefault(pid, set()).add(gpu)
    assert all(len(gpus) == 1 for gpus in by_pid.values())
    assert len({gpu for _, gpu in results}) == 4


def test_fractional_gpu_grant_is_accepted(executor):
    """A shared device is still one device: route on the id list, not on ``gpu_count``."""
    req = NodeResourceList.request(cpus=(0, 1), gpus=(2,), gpu_fraction=0.25)
    job = JobResource(resources=[req], nodes=["node0"])
    assert req.gpu_count == 0.25
    (_, seen), = _run(executor, [job])
    assert seen == "2"


# ------------------------------------------------------------------ #
#  Worker distribution                                                 #
# ------------------------------------------------------------------ #


def test_one_worker_per_gpu_by_default(logger):
    """`max_workers` must not default to the node's core count.

    Loky spawns every worker on first use, so a pool-per-GPU sized by cores would recreate
    the pile-up exactly.
    """
    ex = AsyncPinnedExecutor(logger=logger, gpu_selector=SELECTOR, gpus=[0, 1, 2, 3])
    try:
        assert len(ex._pools) == 4
        assert [p._max_workers for p in ex._pools] == [1, 1, 1, 1]
    finally:
        ex.shutdown(wait=True, kill_workers=True)


def test_extra_workers_round_robin_over_the_pools(logger):
    """`max_workers > ngpus` spreads, remainder to the low-numbered pools."""
    ex = AsyncPinnedExecutor(
        logger=logger, gpu_selector=SELECTOR, gpus=[0, 1, 2, 3], max_workers=6
    )
    try:
        assert [p._max_workers for p in ex._pools] == [2, 2, 1, 1]
    finally:
        ex.shutdown(wait=True, kill_workers=True)


def test_max_workers_below_gpu_count_is_raised(logger):
    """Every GPU keeps at least one worker rather than an unusable empty pool."""
    ex = AsyncPinnedExecutor(
        logger=logger, gpu_selector=SELECTOR, gpus=[0, 1, 2, 3], max_workers=2
    )
    try:
        assert [p._max_workers for p in ex._pools] == [1, 1, 1, 1]
    finally:
        ex.shutdown(wait=True, kill_workers=True)


# ------------------------------------------------------------------ #
#  Tasks that ask for no GPU                                           #
# ------------------------------------------------------------------ #


def test_zero_gpu_tasks_spread_and_are_not_pinned(logger):
    """CPU-only work round-robins, and does not override the pool's own device.

    The pool's ``env`` still applies -- a worker cannot un-see its device -- but nothing per
    task is added on top, so these tasks are free to run wherever there is a free worker.
    """
    ex = AsyncPinnedExecutor(logger=logger, gpu_selector=SELECTOR, gpus=[0, 1, 2, 3])
    try:
        jobs = [_job([i], []) for i in range(4)]
        results = _run(ex, jobs)
        assert len({pid for pid, _ in results}) == 4
    finally:
        ex.shutdown(wait=True, kill_workers=True)


# ------------------------------------------------------------------ #
#  Rejections                                                          #
# ------------------------------------------------------------------ #


def test_multi_gpu_task_is_rejected(executor):
    """A worker sees one device, so a two-GPU task cannot run -- say so, do not drop one."""
    with pytest.raises(ValueError, match="single GPU"):
        _run(executor, [_job(range(16), [0, 1])])


def test_multi_node_task_is_rejected(executor):
    job = JobResource(
        resources=[
            NodeResourceList(cpus=(0,), gpus=(0,)),
            NodeResourceList(cpus=(0,), gpus=(0,)),
        ],
        nodes=["node0", "node1"],
    )
    with pytest.raises(ValueError, match="single node"):
        _run(executor, [job])


def test_unknown_gpu_is_rejected(executor):
    """A grant for a device this executor has no pool for is an error, not a silent remap."""
    with pytest.raises(ValueError, match="no pool for GPU"):
        _run(executor, [_job([0], [7])])


def test_string_gpu_ids_are_kept_and_matched(logger):
    """Aurora spells its devices as numeric strings, and a grant carries them back that way.

    `GpuId` is `Union[int, str]` throughout the scheduler, so an executor that normalised ids
    to one of the two would match no grant on the system that uses the other.
    """
    ex = AsyncPinnedExecutor(
        logger=logger, gpu_selector=SELECTOR, gpus=["0", "1", "2", "3"]
    )
    try:
        assert ex._gpus == ["0", "1", "2", "3"]
        (_, seen), = _run(ex, [_job([0], ["2"])])
        assert seen == "2"
    finally:
        ex.shutdown(wait=True, kill_workers=True)


def test_no_gpus_falls_back_to_one_unpinned_pool(logger):
    """A CPU-only allocation must still construct, and still run CPU work."""
    ex = AsyncPinnedExecutor(logger=logger, gpu_selector=SELECTOR, gpus=[])
    try:
        assert len(ex._pools) == 1
        (_, seen), = _run(ex, [_job([0], [])])
        assert seen is None
        with pytest.raises(ValueError, match="no pool for GPU"):
            _run(ex, [_job([0], [0])])
    finally:
        ex.shutdown(wait=True, kill_workers=True)


# ------------------------------------------------------------------ #
#  Registration                                                        #
# ------------------------------------------------------------------ #


def test_registered_as_an_async_executor():
    from ensemble_launcher.executors import executor_registry

    assert "async_pinned" in executor_registry.async_executors


def test_built_from_a_launcher_config(logger):
    """The config path an orchestrator actually takes, including the wire hop."""
    import json

    from ensemble_launcher.config import LauncherConfig
    from ensemble_launcher.executors import executor_registry

    config = LauncherConfig(
        gpu_selector="ZE_AFFINITY_MASK",
        executor_configs={
            "async_pinned": {"gpus": [0, 1], "gpu_selector": SELECTOR}
        },
    )
    config = LauncherConfig.model_validate_json(
        json.loads(json.dumps({"c": config.model_dump_json()}, default=str))["c"]
    )

    # What AsyncWorker._initialize builds for every executor.
    runtime = dict(
        logger=logger,
        gpu_selector=config.gpu_selector,
        max_workers=32,
        return_stdout=False,
        gpus=(0, 1, 2, 3),
    )
    ex = executor_registry.create_executor(
        "async_pinned", kwargs=config.executor_kwargs("async_pinned", **runtime)
    )
    try:
        # The config's gpus and selector win over the node's, and the ids survive the wire
        # as the ints they were written as -- polaris' spelling, not a stringified copy.
        assert ex._gpus == [0, 1]
        assert ex._gpu_selector == SELECTOR
        (_, seen), = _run(ex, [_job([0], [1])])
        assert seen == "1"
    finally:
        ex.shutdown(wait=True, kill_workers=True)
