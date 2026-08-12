import asyncio
import logging
import multiprocessing as mp
import secrets

import pytest

from ensemble_launcher.comm.async_base import AsyncComm
from ensemble_launcher.comm.messages import Result
from ensemble_launcher.comm.nodeinfo import NodeInfo

pytestmark = pytest.mark.core

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def _node_worker(
    depth: int,
    max_depth: int,
    parent_conn,  # Type hints omitted for brevity, but keep them if imported
    parent_hb_conn,
    transport: str,
):
    logger.info(f"Entering Depth:{depth}, max: {max_depth}")

    async def _run_node(
        depth: int,
        max_depth: int,
        parent_conn,
        parent_hb_conn,
        transport: str,
    ):
        parent_id = str(depth - 1) if depth > 0 else None
        my_nodeinfo = NodeInfo(
            node_id=str(depth),
            secret_id=secrets.token_hex(16),
            parent_id=parent_id,
            children_ids=[str(depth + 1)] if depth < max_depth else [],
            children_secret_ids={str(depth + 1): secrets.token_hex(16)}
            if depth < max_depth
            else {},
        )
        comm = AsyncComm(
            logger,  # Ensure logger is passed correctly to child processes
            node_info=my_nodeinfo,
            parent_conn=parent_conn,
            hb_parent_conn=parent_hb_conn,
            child_transport=transport,
            heartbeat_interval=0.1,
        )

        logger.info(f"After Async comm creation depth:{depth}")
        await comm.start_monitors()
        logger.info(f"Done starting monittors:{depth}")

        if parent_id is not None:
            logger.info(f"waiting for parent {depth}")
            await comm.sync_heartbeat_with_parent(timeout=5.0)
        logger.info(f"Done waiting for parent {depth}")
        send_result = Result(data=[])

        if depth < max_depth:
            child_id = my_nodeinfo.children_ids[0]
            child_secret_id = my_nodeinfo.children_secret_ids[child_id]
            child_conn, child_hb_conn = await comm.create_child_pipe(
                child_id=child_id, child_secret_id=child_secret_id
            )

            p = mp.Process(
                target=_node_worker,
                args=(depth + 1, max_depth, child_conn, child_hb_conn, transport),
            )
            p.start()
            logger.info(f"Waiting for child {depth}")
            await comm.sync_heartbeat_with_child(child_id)
            logger.info(f"Done waiting for child {depth}")

            recv_result = await comm.recv_message_from_child(
                Result, child_id=child_id, block=True, unpack=True
            )

            send_result.data.extend(recv_result.data)

            p.join(timeout=1.0)
            if p.is_alive:
                p.kill()

        if parent_id is not None:
            send_result.data.append(f"Message from {depth} to {depth - 1}")
            await comm.send_message_to_parent(send_result)

        await comm.close()
        return send_result

    # This blocks the process synchronously until the async node is finished
    result = asyncio.run(
        _run_node(
            depth,
            max_depth,
            parent_conn,
            parent_hb_conn,
            transport,
        )
    )
    return result



def test_mp_comm():
    max_depth = 3
    results = _node_worker(0, max_depth, None, None, "mp")

    data = results.data
    for i, result in enumerate(reversed(data)):
        assert result == f"Message from {i + 1} to {i}"



def test_zmq_comm():
    max_depth = 2
    results = _node_worker(0, max_depth, None, None, "zmq")

    data = results.data
    for i, result in enumerate(reversed(data)):
        assert result == f"Message from {i + 1} to {i}"



@pytest.mark.asyncio
async def test_comm_state_roundtrip_zmq():
    from ensemble_launcher.comm.async_base import AsyncCommState
    from ensemble_launcher.comm.pipe import (
        AsyncZMQDealerConnectionState,
        AsyncZMQTransportState,
    )

    secret_ids = [secrets.token_hex(16) for _ in range(2)]
    parent_info = NodeInfo(
        node_id="parent",
        secret_id=secret_ids[0],
        parent_id=None,
        parent_secret_id=None,
        children_ids=["child"],
        children_secret_ids={"child": secret_ids[1]},
    )
    child_info = NodeInfo(
        node_id="child",
        secret_id=secret_ids[1],
        parent_id="parent",
        parent_secret_id=secret_ids[0],
        children_ids=[],
        children_secret_ids={},
    )

    parent_comm = AsyncComm(logger, node_info=parent_info)
    data_client, hb_client = await parent_comm.create_child_pipe(
        child_id="child", child_secret_id=secret_ids[1]
    )
    child_comm = AsyncComm(
        logger, node_info=child_info, parent_conn=data_client, hb_parent_conn=hb_client
    )

    child_state = child_comm.get_state()
    serialized = child_state.serialize()
    restored = AsyncCommState.deserialize(serialized)

    assert isinstance(restored.parent_conn_state, AsyncZMQDealerConnectionState)
    assert isinstance(restored.hb_parent_conn_state, AsyncZMQDealerConnectionState)
    assert isinstance(restored.data_transport_state, AsyncZMQTransportState)
    assert isinstance(restored.hb_transport_state, AsyncZMQTransportState)

    assert (
        restored.parent_conn_state.remote_address
        == child_state.parent_conn_state.remote_address
    )
    assert (
        restored.data_transport_state.hostname
        == child_state.data_transport_state.hostname
    )

    rebuilt = AsyncComm.set_state(restored)
    assert rebuilt._node_info.node_id == "child"
    assert rebuilt._parent_conn is not None


def _node_worker_req_res(
    depth: int,
    max_depth: int,
    parent_conn,
    parent_hb_conn,
    transport: str,
):
    async def _run_node(depth, max_depth, parent_conn, parent_hb_conn, transport):
        parent_id = str(depth - 1) if depth > 0 else None
        my_nodeinfo = NodeInfo(
            node_id=str(depth),
            secret_id=secrets.token_hex(16),
            parent_id=parent_id,
            children_ids=[str(depth + 1)] if depth < max_depth else [],
            children_secret_ids={str(depth + 1): secrets.token_hex(16)}
            if depth < max_depth
            else {},
        )
        comm = AsyncComm(
            logger,
            node_info=my_nodeinfo,
            parent_conn=parent_conn,
            hb_parent_conn=parent_hb_conn,
            child_transport=transport,
            heartbeat_interval=0.1,
            req_res=True,
            send_retries=3,
            send_timeout=5.0,
        )

        await comm.start_monitors()

        if parent_id is not None:
            await comm.sync_heartbeat_with_parent(timeout=5.0)

        send_result = Result(data=[])

        if depth < max_depth:
            child_id = my_nodeinfo.children_ids[0]
            child_secret_id = my_nodeinfo.children_secret_ids[child_id]
            child_conn, child_hb_conn = await comm.create_child_pipe(
                child_id=child_id, child_secret_id=child_secret_id
            )

            p = mp.Process(
                target=_node_worker_req_res,
                args=(depth + 1, max_depth, child_conn, child_hb_conn, transport),
            )
            p.start()
            await comm.sync_heartbeat_with_child(child_id)

            recv_result = await comm.recv_message_from_child(
                Result, child_id=child_id, block=True, unpack=True
            )

            assert recv_result.message_id is not None, "req_res message should have a UUID"
            send_result.data.extend(recv_result.data)

            p.join(timeout=1.0)
            if p.is_alive:
                p.kill()

        if parent_id is not None:
            send_result.data.append(f"Message from {depth} to {depth - 1}")
            success = await comm.send_message_to_parent(send_result)
            assert success, "send_message_to_parent should return True with req_res"

        await comm.close()
        return send_result

    return asyncio.run(
        _run_node(depth, max_depth, parent_conn, parent_hb_conn, transport)
    )


def test_zmq_comm_req_res():
    max_depth = 2
    results = _node_worker_req_res(0, max_depth, None, None, "zmq")

    data = results.data
    for i, result in enumerate(reversed(data)):
        assert result == f"Message from {i + 1} to {i}"


@pytest.mark.asyncio
async def test_req_res_dedup():
    from ensemble_launcher.comm.async_base import AsyncComm

    parent_secret = secrets.token_hex(16)
    child_secret = secrets.token_hex(16)

    parent_info = NodeInfo(
        node_id="parent",
        secret_id=parent_secret,
        parent_id=None,
        children_ids=["child"],
        children_secret_ids={"child": child_secret},
    )
    child_info = NodeInfo(
        node_id="child",
        secret_id=child_secret,
        parent_id="parent",
        parent_secret_id=parent_secret,
        children_ids=[],
        children_secret_ids={},
    )

    parent_comm = AsyncComm(
        logger, node_info=parent_info, req_res=True, skip_hb=True,
    )
    data_client, hb_client = await parent_comm.create_child_pipe(
        child_id="child", child_secret_id=child_secret
    )
    child_comm = AsyncComm(
        logger, node_info=child_info,
        parent_conn=data_client, hb_parent_conn=hb_client,
        req_res=True, skip_hb=True,
    )

    await parent_comm.start_monitors()
    await child_comm.start_monitors()

    msg1 = Result(data="hello", task_id="t1")
    await child_comm.send_message_to_parent(msg1)
    await asyncio.sleep(0.2)

    recv1 = await parent_comm.recv_message_from_child(
        Result, child_id="child", block=True, timeout=5.0, unpack=True
    )
    assert recv1 is not None
    assert recv1.message_id is not None
    assert recv1.data == "hello"

    # Simulate a duplicate by directly dispatching the same raw data again
    saved_id = recv1.message_id
    dup_msg = Result(data="hello", task_id="t1")
    dup_msg.message_id = saved_id
    assert parent_comm._is_duplicate(dup_msg) is True

    # A new message should not be a duplicate
    msg2 = Result(data="world", task_id="t2")
    await child_comm.send_message_to_parent(msg2)
    await asyncio.sleep(0.2)

    recv2 = await parent_comm.recv_message_from_child(
        Result, child_id="child", block=True, timeout=5.0, unpack=True
    )
    assert recv2 is not None
    assert recv2.message_id != saved_id
    assert recv2.data == "world"

    await child_comm.close()
    await parent_comm.close()


@pytest.mark.asyncio
async def test_req_res_state_roundtrip():
    from ensemble_launcher.comm.async_base import AsyncCommState

    parent_info = NodeInfo(
        node_id="parent",
        secret_id=secrets.token_hex(16),
        parent_id=None,
        children_ids=[],
        children_secret_ids={},
    )

    comm = AsyncComm(
        logger, node_info=parent_info,
        req_res=True, send_retries=5, send_timeout=10.0,
    )

    state = comm.get_state()
    assert state.req_res is True
    assert state.send_retries == 5
    assert state.send_timeout == 10.0

    serialized = state.serialize()
    restored = AsyncCommState.deserialize(serialized)
    assert restored.req_res is True
    assert restored.send_retries == 5
    assert restored.send_timeout == 10.0

    rebuilt = AsyncComm.set_state(restored)
    assert rebuilt._req_res is True
    assert rebuilt._send_retries == 5
    assert rebuilt._send_timeout == 10.0


if __name__ == "__main__":
    # msgs = test_zmq_comm()
    # print("zmq done")
    # msgs = asyncio.run(test_comm_state_roundtrip_zmq())
    # print("roundtrip zmq done")
    msgs = test_mp_comm()
    print("mp done")
