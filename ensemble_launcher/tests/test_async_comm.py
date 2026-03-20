from ensemble_launcher.orchestrator import Node
from ensemble_launcher.comm.nodeinfo import NodeInfo
from ensemble_launcher.comm.async_zmq import AsyncZMQComm
from ensemble_launcher.comm.async_base import AsyncComm
from ensemble_launcher.comm.messages import Message, Result
import time
import asyncio
import pytest
import uuid

import logging

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()

@pytest.mark.asyncio
async def test_zmq_comm():

    comms = []
    futures = []
    try:
        secret_ids = [ str(uuid.uuid4()) for _ in range(3)]
        for i in range(3):
            my_nodeinfo = NodeInfo(
                node_id=str(i),
                secret_id=secret_ids[i],
                parent_id=str(i - 1) if i > 0 else None,
                parent_secret_id=secret_ids[i-1] if i > 0 else None,
                children_ids=[str(i + 1)] if i < 2 else [],
                children_secret_ids={str(i+1):secret_ids[i+1]} if i < 2 else {},
            )
            comm = AsyncZMQComm(logger, node_info=my_nodeinfo, parent_comm=comms[i - 1] if i > 0 else None)
            await comm.setup_zmq_sockets()
            await comm.start_monitors()
            comms.append(comm)
        
        for i in range(3):
            comm = comms[i]
            futures.append(asyncio.create_task(comm.sync_heartbeat_with_parent(timeout=5.0)))
            if i < 2:   
                futures.append(asyncio.create_task(comm.sync_heartbeat_with_child(child_id=str(i + 1), timeout=5.0)))

        await asyncio.gather(*futures)

        msgs = []
        for i in range(2):
            comm: AsyncComm = comms[i]
            await comm.send_message_to_child(child_id=str(i + 1), msg=Result(data=f"Message from parent {i} to child {i+1}"))
        
        for i in range(1,3):
            comm: AsyncComm = comms[i]
            msg = await comm.recv_message_from_parent(cls=Result, block=True, timeout=5.0)
            msgs.append(msg.data)
        
        print("Messages received by children:", msgs)
        assert all(msg == f"Message from parent {i} to child {i+1}" for i, msg in enumerate(msgs))
        return msgs
    finally:
        # Cleanup: close all comm objects in reverse order
        for comm in reversed(comms):
            try:
                await comm.close()
            except Exception as e:
                logger.warning(f"Error closing comm: {e}")
    

if __name__ == "__main__":
    msgs = asyncio.run(test_zmq_comm())