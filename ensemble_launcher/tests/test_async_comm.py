import asyncio
import logging
import multiprocessing as mp
import secrets

import pytest
from ensemble_launcher.comm.async_base import AsyncComm
from ensemble_launcher.comm.messages import Result
from ensemble_launcher.comm.nodeinfo import NodeInfo

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()


def _node_worker(
    depth: int,
    max_depth: int,
    parent_conn,  # Type hints omitted for brevity, but keep them if imported
    parent_hb_conn,
    transport: str,
):
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
            heartbeat_interval=1.0,
        )

        await comm.start_monitors(parent_only=True)

        if parent_id is not None:
            await comm.sync_heartbeat_with_parent(timeout=5.0)

        send_result = Result(data=[])

        if depth < max_depth:
            child_id = my_nodeinfo.children_ids[0]
            child_secret_id = my_nodeinfo.children_secret_ids[child_id]
            child_conn, child_hb_conn = comm.create_child_pipe(
                child_id=child_id, child_secret_id=child_secret_id
            )

            await comm.start_monitors(children_only=True)

            p = mp.Process(
                target=_node_worker,
                args=(depth + 1, max_depth, child_conn, child_hb_conn, transport),
            )
            p.start()

            await comm.sync_heartbeat_with_child(child_id)

            recv_result = await comm.recv_message_from_child(
                Result, child_id=child_id, block=True
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


@pytest.mark.timeout(60)
def test_mp_comm():
    max_depth = 3
    results = _node_worker(0, max_depth, None, None, "mp")

    data = results.data
    for i, result in enumerate(reversed(data)):
        assert result == f"Message from {i + 1} to {i}"


@pytest.mark.timeout(60)
def test_zmq_comm():
    max_depth = 3
    results = _node_worker(0, max_depth, None, None, "zmq")

    data = results.data
    for i, result in enumerate(reversed(data)):
        assert result == f"Message from {i + 1} to {i}"


@pytest.mark.timeout(60)
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
    data_client, hb_client = parent_comm.create_child_pipe(
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


if __name__ == "__main__":
    msgs = test_zmq_comm()
    print("zmq done")
    # msgs = asyncio.run(test_comm_state_roundtrip_zmq())
    # print("roundtrip zmq done")
    msgs = test_mp_comm()
    print("mp done")
