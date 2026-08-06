import asyncio
import logging
import multiprocessing as mp
import os
import socket
import uuid

import pytest

from ensemble_launcher.comm.pipe import transport_registry
from ensemble_launcher.config import LauncherConfig, MPIConfig, SystemConfig
from ensemble_launcher.ensemble.actor import PrivateActor, action
from ensemble_launcher.ensemble.actor_pool import ActorPool
from ensemble_launcher.orchestrator import AsyncWorker, ClusterClient
from ensemble_launcher.scheduler.resource import JobResource, NodeResourceList

pytestmark = pytest.mark.core


class AddPrivateActor(PrivateActor):
    @action
    def add(self, a, b):
        return a + b



# ---------------------------------------------------------------------------
# Unit tests (no cluster needed)
# ---------------------------------------------------------------------------


def test_actor_pool_construction():
    transport = transport_registry.get("zmq")["transport"]()
    _, client = transport.create_child_pipe("pool-parent", "secret", "pool", "secret")

    pool = ActorPool(
        name="pool",
        client_conn=client,
        actor_class=AddPrivateActor,
        n_actors=3,
        actor_kwargs={},
        task_kwargs={"nnodes": 1, "ppn": 1},
        checkpoint_dir="/tmp/test_ckpt",
    )

    assert pool._n_children == 3
    assert pool._actor_class is AddPrivateActor
    assert len(pool._actor_kwargs_list) == 3
    assert len(pool._task_kwargs_list) == 3
    assert pool._child_handle is None
    assert pool._child_names == []


def test_actor_pool_per_actor_kwargs():
    transport = transport_registry.get("zmq")["transport"]()
    _, client = transport.create_child_pipe("pool-parent2", "secret", "pool2", "secret")

    per_actor_kwargs = [
        {"name": "actor-a"},
        {"name": "actor-b"},
    ]
    per_task_kwargs = [
        {"nnodes": 1, "ppn": 1},
        {"nnodes": 2, "ppn": 4},
    ]

    pool = ActorPool(
        name="pool2",
        client_conn=client,
        actor_class=AddPrivateActor,
        n_actors=2,
        actor_kwargs=per_actor_kwargs,
        task_kwargs=per_task_kwargs,
        checkpoint_dir="/tmp/test_ckpt",
    )

    assert pool._actor_kwargs_list[0]["name"] == "actor-a"
    assert pool._actor_kwargs_list[1]["name"] == "actor-b"
    assert pool._task_kwargs_list[0]["ppn"] == 1
    assert pool._task_kwargs_list[1]["ppn"] == 4


def test_actor_pool_has_actions():
    assert "invoke_children" in ActorPool.__actions__
    assert "invoke_all_children" in ActorPool.__actions__
    assert "get_n_actors" in ActorPool.__actions__
    assert "get_actor_ids" in ActorPool.__actions__
    assert "stop" in ActorPool.__actions__


def test_actor_pool_create_task():
    transport = transport_registry.get("zmq")["transport"]()
    _, client = transport.create_child_pipe("pool-parent3", "secret", "pool3", "secret")

    pool = ActorPool(
        name="pool3",
        client_conn=client,
        actor_class=AddPrivateActor,
        n_actors=2,
        actor_kwargs={},
        task_kwargs={"nnodes": 1, "ppn": 1},
        checkpoint_dir="/tmp/test_ckpt",
    )

    task = pool.create_task(task_id="pool-task", nnodes=1, ppn=1)
    assert task.task_id == "pool-task"
    assert task.nnodes == 1
    assert task.ppn == 1
    assert task.executable is pool


# ---------------------------------------------------------------------------
# Helpers for cluster integration tests
# ---------------------------------------------------------------------------


def _start_cluster(ckpt_dir, ncpus=12):
    nodes = [socket.gethostname()]
    sys_info = NodeResourceList.from_config(
        SystemConfig(name="local", ncpus=ncpus, cpus=list(range(1, ncpus + 1)))
    )
    job_resource = JobResource(resources=[sys_info], nodes=nodes)
    w = AsyncWorker(
        "test",
        LauncherConfig(
            task_executor_name="async_processpool",
            comm_name="async_zmq",
            report_interval=100.0,
            log_level=logging.INFO,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            return_stdout=True,
            worker_logs=True,
            master_logs=True,
            mpi_config=MPIConfig(flavor="test"),
        ),
        job_resource,
    )
    process = mp.Process(target=w.create_an_event_loop)
    process.start()
    return process


# ---------------------------------------------------------------------------
# Cluster integration tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_actor_pool_invoke():
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    process = _start_cluster(ckpt_dir)

    transport = transport_registry.get("zmq")["transport"]()
    pool_secret = "pool_secret"
    server, client = transport.create_child_pipe(
        "pool_parent", pool_secret, "pool", pool_secret
    )

    pool = ActorPool(
        name="pool",
        client_conn=client,
        actor_class=AddPrivateActor,
        n_actors=2,
        actor_kwargs={},
        task_kwargs={"nnodes": 1, "ppn": 1},
        checkpoint_dir=ckpt_dir,
    )
    pool_task = pool.create_task(task_id="pool-task", nnodes=1, ppn=1)

    cluster_client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
    cluster_client.start()
    cluster_client.submit(pool_task)

    handle = ActorPool.create_handle(
        server, default_target_id=f"pool:{pool_secret}"
    )
    await handle.open()
    await handle.wait_for_ready(expected=1)

    result = await asyncio.wait_for(
        handle.invoke_children(0, "add", (3, 4)), timeout=30.0
    )
    assert result == 7

    result = await asyncio.wait_for(
        handle.invoke_children(1, "add", (10, 20)), timeout=30.0
    )
    assert result == 30

    await handle.stop(timeout=1.0)
    await handle.close()
    cluster_client.teardown()
    process.terminate()
    process.join(timeout=10.0)


@pytest.mark.asyncio
async def test_actor_pool_invoke_all():
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    process = _start_cluster(ckpt_dir)

    transport = transport_registry.get("zmq")["transport"]()
    pool_secret = "pool_secret_bc"
    server, client = transport.create_child_pipe(
        "pool_parent_bc", pool_secret, "pool_bc", pool_secret
    )

    pool = ActorPool(
        name="pool_bc",
        client_conn=client,
        actor_class=AddPrivateActor,
        n_actors=3,
        actor_kwargs={},
        task_kwargs={"nnodes": 1, "ppn": 1},
        checkpoint_dir=ckpt_dir,
    )
    pool_task = pool.create_task(task_id="pool-bc-task", nnodes=1, ppn=1)

    cluster_client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
    cluster_client.start()
    cluster_client.submit(pool_task)

    handle = ActorPool.create_handle(
        server, default_target_id=f"pool_bc:{pool_secret}"
    )
    await handle.open()
    await handle.wait_for_ready(expected=1)

    # Same args to all
    results = await asyncio.wait_for(
        handle.invoke_all_children("add", (5, 6)),
        timeout=30.0,
    )
    assert len(results) == 3
    assert all(r == 11 for r in results)

    # Different args per actor
    per_actor_args = [
        (1, 10),
        (2, 20),
        (3, 30),
    ]
    results = await asyncio.wait_for(
        handle.invoke_all_children("add", per_actor_args),
        timeout=30.0,
    )
    assert len(results) == 3
    assert sorted(results) == [11, 22, 33]

    await handle.stop(timeout=1.0)
    await handle.close()
    cluster_client.teardown()
    process.terminate()
    process.join(timeout=10.0)


# @pytest.mark.asyncio
# async def test_actor_pool_invoke_all_stream():
#     ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
#     process = _start_cluster(ckpt_dir)
#
#     transport = transport_registry.get("zmq")["transport"]()
#     pool_secret = "pool_secret_stream"
#     server, client = transport.create_child_pipe(
#         "pool_parent_stream", pool_secret, "pool_stream", pool_secret
#     )
#
#     pool = ActorPool(
#         name="pool_stream",
#         client_conn=client,
#         actor_class=AddPrivateActor,
#         n_actors=2,
#         actor_kwargs={},
#         task_kwargs={"nnodes": 1, "ppn": 1},
#         checkpoint_dir=ckpt_dir,
#     )
#     pool_task = pool.create_task(task_id="pool-stream-task", nnodes=1, ppn=1)
#
#     cluster_client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
#     cluster_client.start()
#     cluster_client.submit(pool_task)
#
#     handle = ActorPool.create_handle(
#         server, default_target_id=f"pool_stream:{pool_secret}"
#     )
#     await handle.open()
#     await handle.wait_for_ready(expected=1)
#
#     # invoke_all_stream yields one result per child
#     await handle.send(
#         ("invoke_all_stream", (("add", (100, 200), None),), None),
#         target_id=f"pool_stream:{pool_secret}",
#     )
#     streamed_results = []
#     for _ in range(2):
#         result = await asyncio.wait_for(handle.recv("invoke_all_stream"), timeout=30.0)
#         streamed_results.append(result)
#     assert len(streamed_results) == 2
#     assert all(r == 300 for r in streamed_results)
#
#     await handle.stop(timeout=1.0)
#     await handle.close()
#     cluster_client.teardown()
#     process.terminate()
#     process.join(timeout=10.0)


@pytest.mark.asyncio
async def test_actor_pool_get_n_actors_and_ids():
    ckpt_dir = os.path.join("/tmp", f"ckpt_{uuid.uuid4()}")
    process = _start_cluster(ckpt_dir)

    transport = transport_registry.get("zmq")["transport"]()
    pool_secret = "pool_secret_info"
    server, client = transport.create_child_pipe(
        "pool_parent_info", pool_secret, "pool_info", pool_secret
    )

    pool = ActorPool(
        name="pool_info",
        client_conn=client,
        actor_class=AddPrivateActor,
        n_actors=2,
        actor_kwargs=[{"name": "alice"}, {"name": "bob"}],
        task_kwargs={"nnodes": 1, "ppn": 1},
        checkpoint_dir=ckpt_dir,
    )
    pool_task = pool.create_task(task_id="pool-info-task", nnodes=1, ppn=1)

    cluster_client = ClusterClient(node_id="test", checkpoint_dir=ckpt_dir)
    cluster_client.start()
    cluster_client.submit(pool_task)

    handle = ActorPool.create_handle(
        server, default_target_id=f"pool_info:{pool_secret}"
    )
    await handle.open()
    await handle.wait_for_ready(expected=1)

    n = await asyncio.wait_for(handle.get_n_actors(), timeout=30.0)
    assert n == 2

    ids = await asyncio.wait_for(handle.get_actor_ids(), timeout=30.0)
    assert ids == ["alice", "bob"]

    await handle.stop(timeout=1.0)
    await handle.close()
    cluster_client.teardown()
    process.terminate()
    process.join(timeout=10.0)


