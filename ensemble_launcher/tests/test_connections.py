import asyncio
import multiprocessing as mp
import uuid

import cloudpickle
import pytest

from ensemble_launcher.comm.pipe import (
    AsyncZMQDealerConnection,
    AsyncZMQRouterConnection,
)
from ensemble_launcher.comm.pipe.mp import AsyncMPConnection

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


@pytest.fixture(params=[False, True], ids=["inproc", "ioproc"])
def io_process(request):
    """req_res ZMQ tests run with the IOProcess in process and in its own process."""
    return request.param


async def _open_zmq_pair(req_res=False, io_process=False):
    """Create and open a connected ROUTER/DEALER pair."""
    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id,
        secret_id=srv_secret,
        address="127.0.0.1:0",
        expected_remotes={cli_id: cli_secret},
        req_res=req_res,
        io_process=io_process,
    )
    await server.open()
    client = AsyncZMQDealerConnection(
        identity=cli_id,
        secret_id=cli_secret,
        remote_address=server.address,
        remote_identity=srv_id,
        remote_secret_id=srv_secret,
        req_res=req_res,
        io_process=io_process,
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

async def test_zmq_req_res_send_recv(io_process):
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

    success = await client.send(b"hello-rr", timeout=5.0)
    assert success is True

    frames = await server.recv(timeout=5.0)
    assert frames[-1] == b"hello-rr"

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_req_res_roundtrip(io_process):
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

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

async def test_zmq_req_res_multiple(io_process):
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

    for i in range(5):
        success = await client.send(f"rr-{i}".encode(), timeout=5.0)
        assert success is True

    for i in range(5):
        frames = await server.recv(timeout=5.0)
        assert frames[-1] == f"rr-{i}".encode()

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_req_res_pickled_data(io_process):
    """Verify cloudpickled payloads work through req_res."""
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

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

async def test_zmq_router_state_roundtrip(io_process):
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

    state = server.get_state()
    assert state.req_res is True
    assert state.transport_type == "zmq"

    serialized = state.serialize()
    from ensemble_launcher.comm.pipe import AsyncZMQRouterConnectionState

    restored = AsyncZMQRouterConnectionState.deserialize(serialized)
    assert restored.req_res is True
    assert restored.io_process is io_process
    assert AsyncZMQRouterConnection.set_state(restored).out_of_process is io_process
    assert restored.identity == state.identity
    assert restored.address == server.address

    await _close_pair(server, client)


@pytest.mark.asyncio

async def test_zmq_dealer_state_roundtrip(io_process):
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

    state = client.get_state()
    assert state.req_res is True

    serialized = state.serialize()
    from ensemble_launcher.comm.pipe import AsyncZMQDealerConnectionState

    restored = AsyncZMQDealerConnectionState.deserialize(serialized)
    assert restored.req_res is True
    assert restored.io_process is io_process
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
async def test_zmq_batched_send_req_res(io_process):
    """Batched send works with req_res ACK mode."""
    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)

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


# ---------------------------------------------------------------------------
#  IOProcess: socket, ACKs and sends off the owner's event loop
# ---------------------------------------------------------------------------


def _sync_dealer(server, cli_id, cli_secret, srv_id, srv_secret):
    """A DEALER on its own loop thread (sync wrappers), so it keeps running while the test's
    loop is blocked."""
    client = AsyncZMQDealerConnection(
        identity=cli_id,
        secret_id=cli_secret,
        remote_address=server.address,
        remote_identity=srv_id,
        remote_secret_id=srv_secret,
        req_res=True,
    )
    client.sopen()
    return client


@pytest.mark.asyncio
async def test_busy_receiver_still_acks_out_of_process(io_process):
    """The receiver's event loop is blocked for 3 s while a peer sends 50 messages with a
    0.5 s ACK timeout. Out of process, its IOProcess ACKs them all on time and they arrive in
    order afterwards; in process, sends time out (what this change fixes)."""
    import threading
    import time

    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id, secret_id=srv_secret, address="127.0.0.1:0",
        expected_remotes={cli_id: cli_secret}, req_res=True, io_process=io_process,
    )  # fmt: skip
    await server.open()
    client = _sync_dealer(server, cli_id, cli_secret, srv_id, srv_secret)
    time.sleep(0.2)

    results = []

    def send_all():
        for i in range(50):
            results.append(client.ssend(f"m{i}".encode(), timeout=0.5))

    sender = threading.Thread(target=send_all)
    sender.start()
    time.sleep(3.0)  # the receiver's loop is blocked
    sender.join(timeout=60)

    if io_process:
        assert results == [True] * 50
        got = [(await server.recv(timeout=5.0))[-1] for _ in range(50)]
        assert got == [f"m{i}".encode() for i in range(50)]
    else:
        assert not all(results)
    client.sclose()
    await server.close()


@pytest.mark.asyncio
async def test_busy_sender_still_gets_acked_out_of_process():
    """A send issued just before the sender's loop blocks is still sent and ACKed in time:
    the ACK wait runs in the IOProcess."""
    import time

    server, client = await _open_zmq_pair(req_res=True, io_process=True)
    task = asyncio.create_task(client.send(b"while-busy", timeout=0.5))
    await asyncio.sleep(0)  # the request is on the IOProcess's input queue
    time.sleep(2.0)  # the sender's loop is blocked well past the timeout
    assert await task is True
    assert (await server.recv(timeout=5.0))[-1] == b"while-busy"
    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_send_to_unknown_target_fails_fast(io_process):
    """ROUTER_MANDATORY: a send to an identity that isn't connected fails at once, not after
    the ACK timeout."""
    import time

    server, client = await _open_zmq_pair(req_res=True, io_process=io_process)
    t0 = time.monotonic()
    ok = await server.send(b"x", target_id="nobody:nothing", timeout=5.0)
    assert ok is False and time.monotonic() - t0 < 2.0
    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_expected_remotes_change_after_open(io_process):
    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id, secret_id=srv_secret, address="127.0.0.1:0",
        expected_remotes={"someone-else": "x"}, req_res=True, io_process=io_process,
    )  # fmt: skip
    await server.open()
    client = AsyncZMQDealerConnection(
        identity=cli_id, secret_id=cli_secret, remote_address=server.address,
        remote_identity=srv_id, remote_secret_id=srv_secret, req_res=True,
    )  # fmt: skip
    await client.open()
    await asyncio.sleep(0.1)

    assert await client.send(b"before", timeout=0.3) is False  # unknown: dropped, no ACK
    server.add_expected_remote(cli_id, cli_secret)
    await asyncio.sleep(0.2)
    assert await client.send(b"known", timeout=2.0) is True
    assert (await server.recv(timeout=2.0))[-1] == b"known"
    server.remove_expected_remote(cli_id)
    await asyncio.sleep(0.2)
    assert await client.send(b"after", timeout=0.3) is False
    await _close_pair(server, client)


@pytest.mark.asyncio
async def test_validator_set_after_open_applies(io_process):
    """AsyncComm sets its client rule on an already-open data server: it must take effect."""
    from ensemble_launcher.comm.pipe import SenderRule

    srv_id, srv_secret, cli_id, cli_secret = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id, secret_id=srv_secret, address="127.0.0.1:0",
        expected_remotes={"someone-else": "x"}, req_res=True, io_process=io_process,
    )  # fmt: skip
    await server.open()
    server.set_unknown_sender_validator(SenderRule(prefix="client-", secret=cli_secret))
    client = AsyncZMQDealerConnection(
        identity=cli_id, secret_id=cli_secret, remote_address=server.address,
        remote_identity=srv_id, remote_secret_id=srv_secret, req_res=True,
    )  # fmt: skip
    await client.open()
    await asyncio.sleep(0.2)
    assert await client.send(b"by-rule", timeout=2.0) is True
    assert (await server.recv(timeout=2.0))[-1] == b"by-rule"
    await _close_pair(server, client)


def test_sender_rule():
    from ensemble_launcher.comm.pipe import SenderRule

    rule = SenderRule(prefix="client-", secret="s")
    assert rule("client-1", "s")
    assert not rule("client-1", "wrong") and not rule("main.w0", "s")
    assert SenderRule(prefix="client-")("client-1", None)


@pytest.mark.asyncio
async def test_closure_validator_rejected_out_of_process():
    """An out-of-process IOProcess can't receive a closure; a SenderRule is required."""
    srv_id, srv_secret, _, _ = _unique_ids()
    server = AsyncZMQRouterConnection(
        identity=srv_id, secret_id=srv_secret, address="127.0.0.1:0",
        req_res=True, io_process=True,
    )  # fmt: skip
    server.set_unknown_sender_validator(lambda sid, sec: True)
    with pytest.raises(TypeError):
        await server.open()


def _owner_with_io_process(pid_q):
    async def main():
        srv_id, srv_secret, _, _ = _unique_ids()
        server = AsyncZMQRouterConnection(
            identity=srv_id, secret_id=srv_secret, address="127.0.0.1:0",
            req_res=True, io_process=True,
        )  # fmt: skip
        await server.open()
        pid_q.put(server._proc.pid)
        await asyncio.sleep(60)

    asyncio.run(main())


def test_io_process_exits_when_owner_is_killed():
    import os
    import signal
    import time

    ctx = mp.get_context("spawn")
    pid_q = ctx.Queue()
    owner = ctx.Process(target=_owner_with_io_process, args=(pid_q,))
    owner.start()
    helper_pid = pid_q.get(timeout=60)
    os.kill(owner.pid, signal.SIGKILL)
    owner.join(5)

    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    deadline = time.monotonic() + 10
    while alive(helper_pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    assert not alive(helper_pid), "IOProcess outlived its owner"
