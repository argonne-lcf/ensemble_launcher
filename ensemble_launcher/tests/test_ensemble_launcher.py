import os
import socket

import pytest
from utils import echo

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient

pytestmark = pytest.mark.core


def _make_tasks(n: int):
    return {
        f"task-{i}": Task(
            task_id=f"task-{i}", nnodes=1, ppn=1, executable=echo, args=(f"task-{i}",)
        )
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Non-cluster mode — blocking run()
# ---------------------------------------------------------------------------


def test_el_run():
    tasks = _make_tasks(8)
    el = EnsembleLauncher(
        ensemble_file=tasks,
        system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=0),
            return_stdout=False,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        Nodes=[socket.gethostname()],
    )
    result_batch = el.run()
    results = {r.task_id: r.data for r in result_batch.data}

    assert len(results) == len(tasks)
    for task_id in tasks:
        assert results[task_id] == f"Hello from task {task_id}"


# ---------------------------------------------------------------------------
# Cluster mode — start() + ClusterClient
# ---------------------------------------------------------------------------

import uuid


def test_el_cluster_mode():
    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())}")
    tasks = _make_tasks(8)

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=0),
            return_stdout=True,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        Nodes=[socket.gethostname()],
    )

    el.start()

    results = {}
    with ClusterClient(checkpoint_dir=ckpt_dir, node_id="global") as client:
        futures = {task_id: client.submit(task) for task_id, task in tasks.items()}
        for task_id, fut in futures.items():
            r = fut.result(timeout=30.0)
            results[task_id] = r.split(",")[0].strip() if isinstance(r, str) else r

    el.stop()

    assert len(results) == len(tasks)


if __name__ == "__main__":
    # print("test_el_run")
    # test_el_run()
    print("test_el_cluster_mode")
    test_el_cluster_mode()
    print("All tests passed")


# ---------------------------------------------------------------------------
# Task identity: EL routes by Task.uid, so user task_ids need not be unique
# ---------------------------------------------------------------------------


def test_duplicate_task_ids_both_run():
    tasks = {
        f"k{i}": Task(task_id="same", nnodes=1, ppn=1, executable=echo, args=(f"run-{i}",))
        for i in range(2)
    }
    el = EnsembleLauncher(
        ensemble_file=tasks,
        system_config=SystemConfig(name="local", ncpus=4, cpus=list(range(4))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=1, nchildren=1),
            return_stdout=False,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        Nodes=[socket.gethostname()],
    )
    results = el.run().data
    assert [r.task_id for r in results] == ["same", "same"]
    assert sorted(r.data for r in results) == ["Hello from task run-0", "Hello from task run-1"]


def test_worker_builds_tasks_only_while_running(monkeypatch):
    """A worker keeps tasks as bytes and builds a Task object only to run it."""
    import gc
    import weakref

    from ensemble_launcher.taskstore import SegmentTaskStore

    built = []  # weak references to every Task the worker built (Tasks are unhashable)
    most_alive = [0]
    real_task = SegmentTaskStore.task

    def counting_task(self, uid):
        gc.collect()
        alive = sum(1 for ref in built if ref() is not None)
        most_alive[0] = max(most_alive[0], alive + 1)
        t = real_task(self, uid)
        built.append(weakref.ref(t))
        return t

    monkeypatch.setattr(SegmentTaskStore, "task", counting_task)
    ntasks, ncpus = 40, 2
    el = EnsembleLauncher(
        ensemble_file=_make_tasks(ntasks),
        system_config=SystemConfig(name="local", ncpus=ncpus, cpus=list(range(ncpus))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=0),
            return_stdout=False,
            task_flush_interval=0.5,
            result_flush_interval=0.5,
        ),
        Nodes=[socket.gethostname()],
    )
    results = el.run().data  # nlevels=0: the worker runs in this process
    assert len(results) == ntasks
    assert most_alive[0] <= ncpus + 1, most_alive[0]
