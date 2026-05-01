import asyncio
import multiprocessing as mp
import os
import secrets
import socket
import socket as _socket
import uuid

import cloudpickle
import pytest

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.comm.pipe import ClientConnection
from ensemble_launcher.comm.pipe.async_connection import (
    AsyncZMQDealerConnection,
    AsyncZMQRouterConnection,
)
from ensemble_launcher.comm.pipe.async_transport import AsyncZMQTransport
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble.ensemble import Actor
from ensemble_launcher.orchestrator import ClusterClient

Actor.model_rebuild(_types_namespace={"ClientConnection": ClientConnection})


def _free_port() -> int:
    with _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def add(a, b):
    return a + b


def square(x):
    return x * x


@pytest.mark.asyncio
async def test_actor_single_call():
    """Actor receives a single (tuple) invocation and returns the result."""
    server_id = "server"
    server_secret = secrets.token_hex(16)
    actor_id = "actor-0"
    actor_secret = secrets.token_hex(16)
    port = _free_port()

    server = AsyncZMQRouterConnection(
        identity=server_id,
        secret_id=server_secret,
        address=f"127.0.0.1:{port}",
        expected_remotes={actor_id: actor_secret},
    )
    await server.open()

    client = AsyncZMQDealerConnection(
        identity=actor_id,
        secret_id=actor_secret,
        remote_address=server.address,
        remote_identity=server_id,
        remote_secret_id=server_secret,
    )

    actor = Actor(
        task_id="actor-task-0",
        nnodes=1,
        ppn=1,
        executable=add,
        connection=client,
    )

    proc = mp.Process(target=actor.executable)
    proc.start()

    try:
        await asyncio.sleep(0.5)

        target_identity = f"{actor_id}:{actor_secret}"
        payload = cloudpickle.dumps((3, 4))
        await server._socket.send_multipart(
            [target_identity.encode(), f"{server_id}:{server_secret}".encode(), payload]
        )

        frames = await asyncio.wait_for(server.recv(), timeout=5.0)
        result = cloudpickle.loads(frames[1])
        assert result == 7

        stop_payload = cloudpickle.dumps("stop")
        await server._socket.send_multipart(
            [target_identity.encode(), f"{server_id}:{server_secret}".encode(), stop_payload]
        )

        proc.join(timeout=5.0)
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        await server.close()


@pytest.mark.asyncio
async def test_actor_batch_call():
    """Actor receives a batch (list) of invocations and returns all results."""
    server_id = "server"
    server_secret = secrets.token_hex(16)
    actor_id = "actor-1"
    actor_secret = secrets.token_hex(16)
    port = _free_port()

    server = AsyncZMQRouterConnection(
        identity=server_id,
        secret_id=server_secret,
        address=f"127.0.0.1:{port}",
        expected_remotes={actor_id: actor_secret},
    )
    await server.open()

    client = AsyncZMQDealerConnection(
        identity=actor_id,
        secret_id=actor_secret,
        remote_address=server.address,
        remote_identity=server_id,
        remote_secret_id=server_secret,
    )

    actor = Actor(
        task_id="actor-task-1",
        nnodes=1,
        ppn=1,
        executable=square,
        connection=client,
    )

    proc = mp.Process(target=actor.executable)
    proc.start()

    try:
        await asyncio.sleep(0.5)

        target_identity = f"{actor_id}:{actor_secret}"
        batch_args = [(2,), (3,), (5,)]
        payload = cloudpickle.dumps(batch_args)
        await server._socket.send_multipart(
            [target_identity.encode(), f"{server_id}:{server_secret}".encode(), payload]
        )

        frames = await asyncio.wait_for(server.recv(), timeout=5.0)
        results = cloudpickle.loads(frames[1])
        assert results == [4, 9, 25]

        stop_payload = cloudpickle.dumps("stop")
        await server._socket.send_multipart(
            [target_identity.encode(), f"{server_id}:{server_secret}".encode(), stop_payload]
        )

        proc.join(timeout=5.0)
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        await server.close()


@pytest.mark.asyncio
async def test_actor_multiple_calls():
    """Actor handles multiple sequential calls before stopping."""
    server_id = "server"
    server_secret = secrets.token_hex(16)
    actor_id = "actor-2"
    actor_secret = secrets.token_hex(16)
    port = _free_port()

    server = AsyncZMQRouterConnection(
        identity=server_id,
        secret_id=server_secret,
        address=f"127.0.0.1:{port}",
        expected_remotes={actor_id: actor_secret},
    )
    await server.open()

    client = AsyncZMQDealerConnection(
        identity=actor_id,
        secret_id=actor_secret,
        remote_address=server.address,
        remote_identity=server_id,
        remote_secret_id=server_secret,
    )

    actor = Actor(
        task_id="actor-task-2",
        nnodes=1,
        ppn=1,
        executable=add,
        connection=client,
    )

    proc = mp.Process(target=actor.executable)
    proc.start()

    try:
        await asyncio.sleep(0.5)
        target_identity = f"{actor_id}:{actor_secret}"

        for a, b, expected in [(1, 2, 3), (10, 20, 30), (-1, 1, 0)]:
            payload = cloudpickle.dumps((a, b))
            await server._socket.send_multipart(
                [target_identity.encode(), f"{server_id}:{server_secret}".encode(), payload]
            )
            frames = await asyncio.wait_for(server.recv(), timeout=5.0)
            result = cloudpickle.loads(frames[1])
            assert result == expected, f"add({a}, {b}) expected {expected}, got {result}"

        stop_payload = cloudpickle.dumps("stop")
        await server._socket.send_multipart(
            [target_identity.encode(), f"{server_id}:{server_secret}".encode(), stop_payload]
        )

        proc.join(timeout=5.0)
        assert not proc.is_alive()
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=5.0)
        await server.close()


@pytest.mark.asyncio
async def test_actor_via_ensemble_launcher():
    """Submit an Actor task through EnsembleLauncher in cluster mode."""
    controller_id = "controller"
    controller_secret = secrets.token_hex(16)
    actor_id = "actor-cluster"
    actor_secret = secrets.token_hex(16)

    transport = AsyncZMQTransport()
    server, client_conn = transport.create_child_pipe(
        parent_id=controller_id,
        parent_secret=controller_secret,
        child_id=actor_id,
        child_secret=actor_secret,
    )
    await server.open()

    actor = Actor(
        task_id="actor-cluster-0",
        nnodes=1,
        ppn=1,
        executable=add,
        connection=client_conn,
    )

    ckpt_dir = os.path.join("/tmp", f"ckpt_{str(uuid.uuid4())}")
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

    try:
        with ClusterClient(
            checkpoint_dir=ckpt_dir,
            node_id="global",
            task_buffer_size=1,
            task_flush_interval=0.5,
        ) as client:
            future = client.submit(actor)

            await asyncio.sleep(3.0)

            target_identity = f"{actor_id}:{actor_secret}"

            # Single call: add(10, 20)
            payload = cloudpickle.dumps((10, 20))
            await server.send(payload, target_identity)
            frames = await asyncio.wait_for(server.recv(), timeout=10.0)
            result = cloudpickle.loads(frames[1])
            assert result == 30

            # Batch call: add(1,2), add(3,4)
            batch_payload = cloudpickle.dumps([(1, 2), (3, 4)])
            await server.send(batch_payload, target_identity)
            frames = await asyncio.wait_for(server.recv(), timeout=10.0)
            batch_results = cloudpickle.loads(frames[1])
            assert batch_results == [3, 7]

            # Stop the actor
            stop_payload = cloudpickle.dumps("stop")
            await server.send(stop_payload, target_identity)

            future.result(timeout=30.0)
    finally:
        el.stop()
        await server.close()
