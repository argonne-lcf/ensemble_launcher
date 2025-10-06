from typing import Any, Optional
import multiprocessing as mp
import logging
from .base import Comm, NodeInfo

logger = logging.getLogger(__name__)

class MPComm(Comm):
    def __init__(self, 
                 node_info: NodeInfo,
                 parent_comm: Optional["MPComm"] = None,  # Fixed: Should be Optional, not required
                 heartbeat_interval: int = 1):
        super().__init__(node_info,
                         parent_comm=parent_comm,
                         heartbeat_interval=heartbeat_interval)
        
        self._my_conn_to_child = {}
        self._child_conn_to_me = {}
        for child_id in self.node_info.children_ids:
            self._my_conn_to_child[child_id], self._child_conn_to_me[child_id] = mp.Pipe()
        
        self._my_conn_to_parent = None
        if self._parent_comm:
            self._my_conn_to_parent = self._parent_comm._child_conn_to_me[self.node_info.node_id]

    def _send_to_parent(self, data: Any) -> bool:
        if self._my_conn_to_parent is None:
            logger.warning(f"{self.node_info.node_id}: No parent connection available")
            return False
            
        try:
            self._my_conn_to_parent.send(data)
            logger.debug(f"{self.node_info.node_id}: Sent message to parent: {data}")
            return True 
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Sending message to parent failed with {e}")
            return False

    def _recv_from_parent(self, timeout: Optional[float] = None) -> Any:
        # Fixed: Added None check for parent connection
        if self._my_conn_to_parent is None:
            logger.debug(f"{self.node_info.node_id}: No parent connection available")
            return None
            
        try:
            if self._my_conn_to_parent.poll(timeout):
                msg = self._my_conn_to_parent.recv()
                logger.debug(f"{self.node_info.node_id}: Received message {msg} from parent.")
                return msg
            logger.debug(f"{self.node_info.node_id}: No message received from parent within timeout {timeout} seconds.")
            return None
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Receiving message failed with exception {e}!")  # Fixed typo
            return None

    def _send_to_child(self, child_id: str, data: Any) -> bool:
        # Fixed: Added validation for child_id existence
        if child_id not in self._my_conn_to_child:
            logger.warning(f"{self.node_info.node_id}: No connection to child {child_id}")
            return False
            
        try:
            self._my_conn_to_child[child_id].send(data)
            logger.debug(f"{self.node_info.node_id}: Sent message to child {child_id}")
            return True
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Sending message to child {child_id} failed with {e}")
            return False

    def _recv_from_child(self, child_id: str, timeout: Optional[float] = None) -> Any:
        # Fixed: Added validation for child_id existence
        if child_id not in self._my_conn_to_child:
            logger.warning(f"{self.node_info.node_id}: No connection to child {child_id}")
            return None
            
        try:
            logger.debug(f"{self.node_info.node_id}: recv_from_child pipe {child_id}")
            if self._my_conn_to_child[child_id].poll(timeout):
                msg = self._my_conn_to_child[child_id].recv()
                logger.debug(f"{self.node_info.node_id}: Received message {msg} from child {child_id}.")
                return msg
            logger.debug(f"{self.node_info.node_id}: No message received from child {child_id} within timeout {timeout} seconds.")
            return None
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Receiving message from child {child_id} failed with exception {e}!")
            return None

    def close(self):
        """Clean up connections."""
        try:
            # Close child connections
            for child_id in list(self._my_conn_to_child.keys()):
                self._my_conn_to_child[child_id].close()
            if self._my_conn_to_parent:
                self._my_conn_to_parent.close()
            
            # Note: Parent connection is owned by parent, so we don't close it
            
        except Exception as e:
            logger.warning(f"{self.node_info.node_id}: Error during cleanup: {e}")
