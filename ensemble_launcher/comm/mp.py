from typing import Union, Any
import multiprocessing as mp
import logging
import time
from .base import Comm, NodeInfo
from queue import Empty, Full


logger = logging.getLogger(__name__)

class MPComm(Comm):
    def __init__(self, 
                 node_info: NodeInfo,
                 parent_comm: "MPComm",              
                 heartbeat_interval:int=1):
        super().__init__()
        self.node_info = node_info
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_comm = parent_comm

        ###This only to send the results back to the parent
        self.result_queue = mp.Queue()

        
        self._my_conn_to_child = {}
        self._child_conn_to_me = {}
        for child_id in self.node_info.children_ids:
            self._my_conn_to_child[child_id], self._child_conn_to_me[child_id]= mp.Pipe()
        self._parent_result_queue = None
        self._my_conn_to_parent = None
        if self._parent_comm:
            self._parent_result_queue = self._parent_comm.result_queue() if self._parent_comm is not None else None        
            self._my_conn_to_parent = self._parent_comm._child_conn_to_me[self.node_info.node_id]

    def send_to_parent(self, parent_id: Union[int, str], data) -> int:
        logger.debug(f"send_to_parent: node {self.node_id}")
        self._my_conn_to_parent.send(data)
        logger.debug(f"Sent message to parent {parent_id}: {data}")
        return 0

    def recv_from_parent(self, parent_id: Union[int, str], timeout: int = 60):
        if self._my_conn.poll(timeout):
            msg = self._my_conn_to_parent.recv()
            logger.debug(f"Received message {msg} from parent {parent_id}.")
            return msg
        logger.debug(f"No message received from parent {parent_id} within timeout {timeout} seconds.")
        return None

    def send_to_child(self, child_id: Union[int, str], message) -> int:
        if child_id in self.node_info.children_ids:
            self._my_conn_to_child[child_id].send(message)
            logger.debug(f"Sent message to child {child_id}")
            return 0
        else:
            return 1

    def recv_from_child(self, child_id: Union[int, str], timeout: int = 60):
        if child_id in self.node_info.children_ids:
            logger.debug(f"recv_from_child pipe {child_id}")
            if self._my_conn_to_child[child_id].poll(timeout):
                msg = self._my_conn_to_child[child_id].recv()
                logger.debug(f"Received message {msg} from child {child_id}.")
                return msg
        else:
            logger.debug(f"Cannot receive from child {child_id}: child does not exist.")
            raise ValueError(f"Child {child_id} does not exist.")
        logger.debug(f"No message received from child {child_id} within timeout {timeout} seconds.")
        return None

    def blocking_recv_from_parent(self, parent_id: Union[int, str]):
        """
        Blocking receive from a specific parent. Waits indefinitely until a message is available.
        """
        logger.debug(f"Waiting for message from parent {parent_id}......")
        msg = self._my_conn_to_parent.recv()  # Blocking call
        logger.debug(f"Received message from parent {parent_id} (blocking)")
        return msg

    def blocking_recv_from_child(self, child_id: Union[int, str]):
        """
        Blocking receive from a specific child. Waits indefinitely until a message is available.
        """
        if child_id in self.node_info.children_ids:
            logger.debug(f"Waiting for message from child {child_id}......")
            msg = self._my_conn_to_child[child_id].recv()
        else:
            logger.debug(f"Cannot receive from child {child_id}: child does not exist.")
            raise ValueError(f"Child {child_id} does not exist.")
        logger.debug(f"Received message {msg} from child {child_id}")
        return msg
    
    def return_result(self, result: Any):
        try:
            # Non‐blocking put; if the queue is full, Full is raised immediately
            self._parent_result_queue.put(result, block=False)
            logger.debug(f"Returned result to parent (non‐blocking): {result}")
        except Full:
            logger.warning(f"Parent result queue is full; dropping result: {result}")
    
    def get_results(self, timeout: float = None) -> list[Any]:
        """
        Retrieve all results from this node's result queue until it's empty.
        If timeout is provided, waits for the first result up to timeout seconds, then
        drains the queue immediately.
        Returns a list of results (empty if no result is available).
        """
        results = []

        # Try to get the first item (blocking or non-blocking depending on timeout)
        try:
            if timeout is not None:
                first = self.result_queue.get(timeout=timeout)
            else:
                first = self.result_queue.get_nowait()
            results.append(first)
        except Empty:
            logger.debug(f"No result available within {timeout} seconds.")
            return results

        # Drain remaining items without blocking
        while True:
            try:
                results.append(self.result_queue.get_nowait())
            except Empty:
                break

        return results