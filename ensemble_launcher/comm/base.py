from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
from .messages import Message, HeartBeat
from dataclasses import dataclass, field
import time


@dataclass
class NodeInfo:
    node_id:str
    parent_id: str =  None
    children_ids: List[str] =  field(default_factory=list)

class Comm(ABC):
    def __init__(self, 
                 node_info: NodeInfo, 
                 parent_comm: "Comm"= None, 
                 heartbeat_interval: int = 1):
        
        self.node_info = node_info
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_comm = parent_comm
        self._cache: Dict[str, List[Any]] = {}
        for child_id in self.node_info.children_ids:
            self._cache[child_id] = []
        
        self._cache[self.node_info.parent_id] = []

    @abstractmethod
    def _send_to_parent(self, data: Any, **kwargs) -> bool:
        pass

    @abstractmethod
    def _recv_from_parent(self, timeout: Optional[float] = None, **kwargs) -> Any:
        pass

    @abstractmethod
    def _send_to_child(self, child_id: str, data: Any, **kwargs) -> bool:
        pass

    @abstractmethod
    def _recv_from_child(self, child_id: str, timeout: Optional[float] = None, **kwargs) -> Any:
        pass

    def _send_to_children(self, data: Any) -> Dict[str, bool]:
        ret = {}
        for child_id in self.node_info.children_ids:
            ret[child_id] = self._send_to_child(child_id, data)
        return ret

    def _recv_from_children(self, timeout: Optional[float] = None) -> Dict[str, Any]:  # Missing str key type
        messages = {}
        for child_id in self.node_info.children_ids:
            msg = self._recv_from_child(child_id, timeout)
            messages[child_id] = msg
        return messages

    def recv_message_from_child(self,cls: Type[Message], child_id: str, timeout: Optional[float] = None) -> Message | None:
        # Check cache first for existing status message
        for i, msg in enumerate(self._cache[child_id]):
            if isinstance(msg, cls):
                # Remove and return the status message
                return self._cache[child_id].pop(i)
            
        if timeout:
            end_time = time.time() + timeout
            while time.time() < end_time:
                msg = self._recv_from_child(child_id,timeout=0.1)
                if msg is not None:
                    if isinstance(msg,cls):
                        return msg
                    else:
                        self._cache[child_id].append(msg)
            return None
        else:
            while True:
                msg = self._recv_from_child(child_id, timeout=0.1)
                if msg is not None:
                    if isinstance(msg, cls):
                        return msg
                    else:
                        self._cache[child_id].append(msg)
    
    def recv_messages_from_children(self, cls: Type[Message], timeout: Optional[float] = None) -> Dict[str, Message | None]:
        status = {}
        for child_id in self.node_info.children_ids:
            status[child_id] = self.recv_message_from_child(cls,child_id,timeout=timeout)
        return status
    
    def send_message_to_child(self, child_id: str, msg: Message) -> bool:
        return self._send_to_child(child_id=child_id, data=msg)
    
    def send_messages_to_children(self, msg: Message) -> Dict[str, bool]:
        return self._send_to_children(msg)

    def send_message_to_parent(self, msg: Message) -> bool:
        """Send a message to the parent node."""
        return self._send_to_parent(data=msg)

    def recv_message_from_parent(self, cls: Type[Message], timeout: Optional[float] = None) -> Message | None:
        """Receive a specific message type from parent node with caching."""
        parent_id = self.node_info.parent_id
        if parent_id is None:
            return None
        
        # Check cache first for existing message of specified type
        for i, msg in enumerate(self._cache[parent_id]):
            if isinstance(msg, cls):
                # Remove and return the message
                return self._cache[parent_id].pop(i)
        
        if timeout:
            end_time = time.time() + timeout
            while time.time() < end_time:
                msg = self._recv_from_parent(timeout=0.1)
                if msg is not None:
                    if isinstance(msg, cls):
                        return msg
                    else:
                        self._cache[parent_id].append(msg)
            return None
        else:
            msg = self._recv_from_parent()
            if msg is not None:
                if isinstance(msg, cls):
                    return msg
                else:
                    self._cache[parent_id].append(msg)
    

    def wait_for_children(self, timeout: float = None) -> bool:
        return self.recv_messages_from_children(HeartBeat, timeout=timeout)

    def send_heartbeat(self) -> bool:
        return self.send_message_to_parent(msg=HeartBeat())
        




