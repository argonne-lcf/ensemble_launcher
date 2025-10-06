from ensemble_launcher.orchestrator import Node
from ensemble_launcher.comm import MPComm, NodeInfo, ZMQComm
from ensemble_launcher.comm.messages import Message, Result
import multiprocessing as mp
import time

def launch_recursive_node(nodes, parent_comm=None, comm_type: str = "mp", parent_address: str = None):
    if len(nodes) == 0:
        return
    
    node_info: NodeInfo = nodes[0].info()
    if comm_type == "mp":
        comm = MPComm(node_info=nodes[0].info(), parent_comm=parent_comm)
    elif comm_type == "zmq":
        comm = ZMQComm(node_info=nodes[0].info(), parent_address=parent_address)
        comm.setup_zmq_sockets()
    else:
        return 
    
    if comm_type == "mp":
        p = mp.Process(target=launch_recursive_node, args=(nodes[1:] if len(nodes) > 1 else [], comm))
    elif comm_type == "zmq":
        p = mp.Process(target=launch_recursive_node, 
                      args=(nodes[1:] if len(nodes) > 1 else [],), 
                      kwargs={"parent_address": comm.my_address, "comm_type": comm_type})
    p.start()

    if node_info.parent_id is not None:
        comm.sync_heartbeat_with_parent(timeout=10.0)
    
    if node_info.children_ids:
        comm.sync_heartbeat_with_children(timeout=10.0)


    if node_info.parent_id:
        msg = comm.recv_message_from_parent(Message, timeout=10.0)
    else:
        msg = None
    if node_info.children_ids:
        comm.send_message_to_child(node_info.children_ids[0],msg=Message(sender=node_info.node_id))
    
    p.join()
    res = None
    if node_info.children_ids:
         res = comm.recv_message_from_child(Result,node_info.children_ids[0], timeout=10.0)
    if node_info.parent_id:
        if res is None:
            new_result = Result(data=[msg])
            comm.send_message_to_parent(new_result)
        else:
            updated_data = res.data.copy() if res.data else []  # Make a copy to avoid modifying original
            updated_data.append(msg)
            new_result = Result(data=updated_data)
            comm.send_message_to_parent(new_result)
    return res.data if res is not None else None

def test_mp_comm():
    nodes = [Node(f"0.")]

    for i in range(3):
        nodes.append(Node(f"{".".join([str(j) for j in  range(i+2)])}",parent=nodes[i],children={}))
    
    for i in range(3):
        nodes[i].children[nodes[i+1].node_id] = nodes[i+1]
    
    res = launch_recursive_node(nodes)
    res = res[::-1] if res is not None else None

    for i,r in enumerate(res):
        assert r.sender == nodes[i].node_id

def test_zmq_comm():
    nodes = [Node(f"0.")]

    for i in range(3):
        nodes.append(Node(f"{".".join([str(j) for j in  range(i+2)])}",parent=nodes[i],children={}))
    
    for i in range(3):
        nodes[i].children[nodes[i+1].node_id] = nodes[i+1]
    
    res = launch_recursive_node(nodes, comm_type="zmq")  # Fixed: use comm_type instead of type
    res = res[::-1] if res is not None else None

    for i,r in enumerate(res):
        assert r.sender == nodes[i].node_id
    

if __name__ == "__main__":
    # test_mp_comm()
    test_zmq_comm()