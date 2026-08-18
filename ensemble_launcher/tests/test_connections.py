import asyncio
import multiprocessing as mp
import uuid

import cloudpickle
import pytest

from ensemble_launcher.comm.pipe import (
    AsyncZMQDealerConnection,
    AsyncZMQRouterConnection,
)
from ensemble_launcher.comm.pipe.mp_connection import AsyncMPConnection

pytestmark = pytest.mark.core

_counter = 0


def _unique_ids():
    """Generate unique identity/secret pairs to avoid ZMQ routing table bleed."""
    global _counter
    _counter += 1
    tag = f"{_counter}-{uuid.uuid4().hex[:6]}"
    return f"server-{tag}", f"ss-{tag}", f"client-{tag}", f"cs-{tag}"


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


async def _open_zmq_pair(req_res=False):
    """Create and open a connected ROUTER/DEALER pair."""
    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id,
        secret_id=srv_secret,
        address="127.0.0.1:0",
        expected_remotes={cli_id: cli_secret},
        req_res=req_res,
    )
    await server.open()
    client = AsyncZMQDealerConnection(
        identity=cli_id,
        secret_id=cli_secret,
        remote_address=server.address,
        remote_identity=srv_id,
        remote_secret_id=srv_secret,
        req_res=req_res,
    )
    await client.open()
    await asyncio.sleep(0.1)
    return server, client


async def _close_pair(server, client):
    await client.close()
    await server.close()


def _make_mp_pair(req_res=False):
    """Create a connected pair of MP connections over a multiprocessing.Pipe."""
    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    left, right = mp.Pipe()
    server = AsyncMPConnection(
        identity=srv_id,
        secret_id=srv_secret,
        pipe_conn=left,
        expected_remotes={cli_id: cli_secret},
        req_res=req_res,
    )
    client = AsyncMPConnection(
        identity=cli_id,
        secret_id=cli_secret,
        pipe_conn=right,
        expected_remotes={srv_id: srv_secret},
        req_res=req_res,
    )
    return server, client


# ---------------------------------------------------------------------------
#  ZMQ — basic send/recv (no req_res)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio

async def test_zmq_basic_send_recv():
    server, client = await _open_zmq_pair(req_res=False)

    payload = b"hello world"
    await client.send(payload)
    frames = await server.recv(timeout=5.0)
    assert frames[-1] == payload

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_basic_roundtrip():
    server, client = await _open_zmq_pair(req_res=False)

    await client.send(b"request")
    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"request"

    sender_id = frames[0].decode()
    await server.send(b"response", target_id=sender_id)
    reply = await client.recv(timeout=5.0)
    assert reply[-1] == b"response"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_multiple_messages():
    server, client = await _open_zmq_pair(req_res=False)

    for i in range(10):
        await client.send(f"msg-{i}".encode())

    for i in range(10):
        frames = await server.recv(timeout=5.0)
        assert frames[-1] == f"msg-{i}".encode()

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  ZMQ — req_res mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio

async def test_zmq_req_res_send_recv():
    server, client = await _open_zmq_pair(req_res=True)

    success = await client.send(b"hello-rr", timeout=5.0)
    assert success is True

    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"hello-rr"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_req_res_roundtrip():
    server, client = await _open_zmq_pair(req_res=True)

    success = await client.send(b"ping", timeout=5.0)
    assert success is True
    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"ping"

    sender_id = frames[0].decode()
    success = await server.send(b"pong", target_id=sender_id, timeout=5.0)
    assert success is True
    reply = await client.recv(timeout=5.0)
    assert reply[-1] == b"pong"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_req_res_multiple():
    server, client = await _open_zmq_pair(req_res=True)

    for i in range(5):
        success = await client.send(f"rr-{i}".encode(), timeout=5.0)
        assert success is True

    for i in range(5):
        frames = await server.recv(timeout=5.0)
        assert frames[-1] == f"rr-{i}".encode()

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_req_res_pickled_data():
    """Verify cloudpickled payloads work through req_res."""
    server, client = await _open_zmq_pair(req_res=True)

    obj = {"key": "value", "nums": [1, 2, 3]}
    payload = cloudpickle.dumps(obj)

    success = await client.send(payload, timeout=5.0)
    assert success is True

    frames = await server.recv(timeout=5.0)
    result = cloudpickle.loads(frames[-1])
    assert result == obj

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  ZMQ — state serialization
# ---------------------------------------------------------------------------


@pytest.mark.asyncio

async def test_zmq_router_state_roundtrip():
    server, client = await _open_zmq_pair(req_res=True)

    state = server.get_state()
    assert state.req_res is True
    assert state.transport_type == "zmq"

    serialized = state.serialize()
    from ensemble_launcher.comm.pipe import AsyncZMQRouterConnectionState

    restored = AsyncZMQRouterConnectionState.deserialize(serialized)
    assert restored.req_res is True
    assert restored.identity == state.identity
    assert restored.address == server.address

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_dealer_state_roundtrip():
    server, client = await _open_zmq_pair(req_res=True)

    state = client.get_state()
    assert state.req_res is True

    serialized = state.serialize()
    from ensemble_launcher.comm.pipe import AsyncZMQDealerConnectionState

    restored = AsyncZMQDealerConnectionState.deserialize(serialized)
    assert restored.req_res is True
    assert restored.identity == state.identity

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  MP — basic send/recv (no req_res)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio

async def test_mp_basic_send_recv():
    server, client = _make_mp_pair(req_res=False)
    await server.open()
    await client.open()

    await client.send(b"hello-mp")
    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"hello-mp"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_mp_basic_roundtrip():
    server, client = _make_mp_pair(req_res=False)
    await server.open()
    await client.open()

    await client.send(b"request-mp")
    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"request-mp"

    sender_id = frames[0].decode()
    await server.send(b"response-mp", target_id=sender_id)
    reply = await client.recv(timeout=5.0)
    assert reply[-1] == b"response-mp"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_mp_multiple_messages():
    server, client = _make_mp_pair(req_res=False)
    await server.open()
    await client.open()

    for i in range(10):
        await client.send(f"mp-{i}".encode())

    for i in range(10):
        frames = await server.recv(timeout=5.0)
        assert frames[-1] == f"mp-{i}".encode()

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  MP — req_res mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio

async def test_mp_req_res_send_recv():
    server, client = _make_mp_pair(req_res=True)
    await server.open()
    await client.open()

    success = await client.send(b"hello-mp-rr", timeout=5.0)
    assert success is True

    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"hello-mp-rr"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_mp_req_res_roundtrip():
    server, client = _make_mp_pair(req_res=True)
    await server.open()
    await client.open()

    success = await client.send(b"ping-mp", timeout=5.0)
    assert success is True
    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"ping-mp"

    sender_id = frames[0].decode()
    success = await server.send(b"pong-mp", target_id=sender_id, timeout=5.0)
    assert success is True
    reply = await client.recv(timeout=5.0)
    assert reply[-1] == b"pong-mp"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_mp_req_res_multiple():
    server, client = _make_mp_pair(req_res=True)
    await server.open()
    await client.open()

    for i in range(5):
        success = await client.send(f"mp-rr-{i}".encode(), timeout=5.0)
        assert success is True

    for i in range(5):
        frames = await server.recv(timeout=5.0)
        assert frames[-1] == f"mp-rr-{i}".encode()

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  ZMQ — batched send (List[bytes])
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_zmq_batched_send_client_to_server():
    """Client sends List[bytes]; server receives all payloads in frames[1:]."""
    server, client = await _open_zmq_pair(req_res=False)

    payloads = [f"msg-{i}".encode() for i in range(5)]
    await client.send(payloads)

    frames = await server.recv(timeout=5.0)
    assert frames[1:] == payloads

    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_zmq_batched_send_server_to_client():
    """Server sends List[bytes]; client receives all payloads in frames[1:]."""
    server, client = await _open_zmq_pair(req_res=False)

    # Need a client→server message first to learn the routing id
    await client.send(b"hello")
    frames = await server.recv(timeout=5.0)
    sender_id = frames[0].decode()

    payloads = [f"reply-{i}".encode() for i in range(5)]
    await server.send(payloads, target_id=sender_id)

    reply = await client.recv(timeout=5.0)
    assert reply[1:] == payloads

    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_zmq_batched_send_req_res():
    """Batched send works with req_res ACK mode."""
    server, client = await _open_zmq_pair(req_res=True)

    payloads = [f"rr-batch-{i}".encode() for i in range(5)]
    success = await client.send(payloads, timeout=5.0)
    assert success is True

    frames = await server.recv(timeout=5.0)
    assert frames[1:] == payloads

    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_zmq_batched_send_pickled_roundtrip():
    """Batched pickled payloads survive a full roundtrip."""
    server, client = await _open_zmq_pair(req_res=False)

    objects = [{"i": i, "data": list(range(i))} for i in range(4)]
    payloads = [cloudpickle.dumps(obj) for obj in objects]
    await client.send(payloads)

    frames = await server.recv(timeout=5.0)
    received = [cloudpickle.loads(f) for f in frames[1:]]
    assert received == objects

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  MP — batched send (List[bytes])
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mp_batched_send():
    """MP connection sends List[bytes]; receiver gets all payloads in frames[1:]."""
    server, client = _make_mp_pair(req_res=False)
    await server.open()
    await client.open()

    payloads = [f"mp-batch-{i}".encode() for i in range(5)]
    await client.send(payloads)

    frames = await server.recv(timeout=5.0)
    assert frames[1:] == payloads

    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_mp_batched_send_req_res():
    """MP batched send works with req_res ACK mode."""
    server, client = _make_mp_pair(req_res=True)
    await server.open()
    await client.open()

    payloads = [f"mp-rr-batch-{i}".encode() for i in range(5)]
    success = await client.send(payloads, timeout=5.0)
    assert success is True

    frames = await server.recv(timeout=5.0)
    assert frames[1:] == payloads

    await _close_pair(server, client)


# ---------------------------------------------------------------------------
#  Sync wrappers
# ---------------------------------------------------------------------------


def test_zmq_sync_wrappers():
    import time

    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id,
        secret_id=srv_secret,
        address="127.0.0.1:0",
        expected_remotes={cli_id: cli_secret},
        req_res=False,
    )
    server.sopen()

    client = AsyncZMQDealerConnection(
        identity=cli_id,
        secret_id=cli_secret,
        remote_address=server.address,
        remote_identity=srv_id,
        remote_secret_id=srv_secret,
        req_res=False,
    )
    client.sopen()
    time.sleep(0.1)

    client.ssend(b"sync-hello")
    frames = server.srecv()
    assert frames[-1] == b"sync-hello"

    client.sclose()
    server.sclose()
