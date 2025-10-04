from ensemble_launcher.orchestrator import Node
from ensemble_launcher.comm import MPComm, NodeInfo
from ensemble_launcher.comm.messages import Message, Result
import multiprocessing as mp


def launch_recursive_node(nodes,parent_comm = None):
        if len(nodes) == 0:
            return
        
        node_info: NodeInfo = nodes[0].info()
        comm = MPComm(node_info=nodes[0].info(),parent_comm=parent_comm)
        p = mp.Process(target=launch_recursive_node, args=(nodes[1:] if len(nodes) > 1 else [],comm))
        p.start()
        if node_info.parent_id:
            msg = comm.recv_message_from_parent(Message)
        else:
            msg = None

        if node_info.children_ids:
            comm.send_message_to_child(node_info.children_ids[0],msg=Message(sender=node_info.node_id))
        
        p.join()

        res = None
        if node_info.children_ids:
             res = comm.recv_message_from_child(Result,node_info.children_ids[0])

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
    

if __name__ == "__main__":
    test_mp_comm()