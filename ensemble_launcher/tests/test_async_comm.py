import asyncio
import logging
import secrets

import pytest
from ensemble_launcher.comm.async_base import AsyncComm
from ensemble_launcher.comm.messages import Result
from ensemble_launcher.comm.nodeinfo import NodeInfo

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()


@pytest.mark.asyncio
async def test_zmq_comm():

    comms = []
    futures = []
    try:
        secret_ids = [secrets.token_hex(16) for _ in range(3)]
        for i in range(3):
            my_nodeinfo = NodeInfo(
                node_id=str(i),
                secret_id=secret_ids[i],
                parent_id=str(i - 1) if i > 0 else None,
                parent_secret_id=secret_ids[i - 1] if i > 0 else None,
                children_ids=[str(i + 1)] if i < 2 else [],
                children_secret_ids={str(i + 1): secret_ids[i + 1]} if i < 2 else {},
            )
            parent_conn = None
            hb_parent_conn = None
            if i > 0:
                parent_conn, hb_parent_conn = comms[i - 1].create_child_pipe(
                    child_id=str(i),
                    child_secret_id=secret_ids[i],
                )
            comm = AsyncComm(
                logger,
                node_info=my_nodeinfo,
                parent_conn=parent_conn,
                hb_parent_conn=hb_parent_conn,
            )
            comms.append(comm)

        for comm in comms:
            await comm.start_monitors()

        for i in range(3):
            comm = comms[i]
            futures.append(
                asyncio.create_task(comm.sync_heartbeat_with_parent(timeout=5.0))
            )
            if i < 2:
                futures.append(
                    asyncio.create_task(
                        comm.sync_heartbeat_with_child(child_id=str(i + 1), timeout=5.0)
                    )
                )

        await asyncio.gather(*futures)

        msgs = []
        for i in range(2):
            comm: AsyncComm = comms[i]
            await comm.send_message_to_child(
                child_id=str(i + 1),
                msg=Result(data=f"Message from parent {i} to child {i + 1}"),
            )

        for i in range(1, 3):
            comm: AsyncComm = comms[i]
            msg = await comm.recv_message_from_parent(
                cls=Result, block=True, timeout=5.0
            )
            msgs.append(msg.data)

        print("Messages received by children:", msgs)
        assert all(
            msg == f"Message from parent {i} to child {i + 1}"
            for i, msg in enumerate(msgs)
        )
        return msgs
    finally:
        for comm in reversed(comms):
            try:
                await comm.close()
            except Exception as e:
                logger.warning(f"Error closing comm: {e}")


@pytest.mark.asyncio
async def test_mp_comm():

    nnodes = 3
    comms = []
    futures = []
    try:
        secret_ids = [secrets.token_hex(16) for _ in range(nnodes)]
        for i in range(nnodes):
            my_nodeinfo = NodeInfo(
                node_id=str(i),
                secret_id=secret_ids[i],
                parent_id=str(i - 1) if i > 0 else None,
                parent_secret_id=secret_ids[i - 1] if i > 0 else None,
                children_ids=[str(i + 1)] if i < 2 else [],
                children_secret_ids={str(i + 1): secret_ids[i + 1]} if i < 2 else {},
            )
            parent_conn = None
            hb_parent_conn = None
            if i > 0:
                parent_conn, hb_parent_conn = comms[i - 1].create_child_pipe(
                    child_id=str(i),
                    child_secret_id=secret_ids[i],
                )
            comm = AsyncComm(
                logger,
                node_info=my_nodeinfo,
                parent_conn=parent_conn,
                hb_parent_conn=hb_parent_conn,
                child_transport="mp",
                heartbeat_interval=1.0,
            )
            comms.append(comm)
        for comm in comms:
            await comm.start_monitors()
        for i in range(nnodes):
            comm = comms[i]
            futures.append(
                asyncio.create_task(comm.sync_heartbeat_with_parent(timeout=5.0))
            )
            if i < nnodes - 1:
                futures.append(
                    asyncio.create_task(
                        comm.sync_heartbeat_with_child(child_id=str(i + 1), timeout=5.0)
                    )
                )
        await asyncio.gather(*futures)
        msgs = []
        for i in range(nnodes - 1):
            comm: AsyncComm = comms[i]
            await comm.send_message_to_child(
                child_id=str(i + 1),
                msg=Result(data=f"Message from parent {i} to child {i + 1}"),
            )
        for i in range(1, nnodes):
            comm: AsyncComm = comms[i]
            msg = await comm.recv_message_from_parent(
                cls=Result, block=True, timeout=5.0
            )
            msgs.append(msg.data)
        print("Messages received by children:", msgs)
        assert all(
            msg == f"Message from parent {i} to child {i + 1}"
            for i, msg in enumerate(msgs)
        )
    finally:
        for comm in comms:
            try:
                await comm.close()
            except Exception as e:
                logger.warning(f"Error closing comm: {e}")

    return msgs


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
    # msgs = asyncio.run(test_zmq_comm())
    # print("zmq done")
    # msgs = asyncio.run(test_comm_state_roundtrip_zmq())
    # print("roundtrip zmq done")
    msgs = asyncio.run(test_mp_comm())
    print("mp done")
