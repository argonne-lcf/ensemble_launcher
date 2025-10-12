from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union, overload
from .messages import Message, HeartBeat, Status
from dataclasses import dataclass, field
import time
from logging import Logger
import threading
import queue


@dataclass
class NodeInfo:
    node_id:str
    parent_id: str =  None
    children_ids: List[str] =  field(default_factory=list)
    level: int = 0


class MessageRoutingQueue:
    """A routing queue that organizes messages by type using separate LifoQueues"""
    
    def __init__(self):
        self._queues: Dict[Type[Message], queue.LifoQueue] = {}
        self._lock = threading.RLock()
    
    def put(self, message: Message):
        """Put a message into the appropriate type-specific queue"""
        msg_type = type(message)
        with self._lock:
            if msg_type not in self._queues:
                if msg_type == Status:
                    self._queues[msg_type] = queue.LifoQueue()
                else:
                    self._queues[msg_type] = queue.Queue()
            self._queues[msg_type].put(message)
    
    @overload
    def get(self, msg_type: Type[Message], timeout: Optional[float] = None) -> Optional[Message]:
        """Get the latest message of a specific type"""
        ...
    
    @overload
    def get(self, timeout: Optional[float] = None) -> Optional[Message]:
        """Get the latest message of any type"""
        ...
    
    def get(self, msg_type_or_timeout: Union[Type[Message], Optional[float]] = None, timeout: Optional[float] = None) -> Optional[Message]:
        """Get the latest message of a specific type or any type if msg_type is None"""
        # Handle overloaded arguments
        if isinstance(msg_type_or_timeout, type) and issubclass(msg_type_or_timeout, Message):
            msg_type = msg_type_or_timeout
        else:
            msg_type = None
            timeout = msg_type_or_timeout
        
        with self._lock:
            if msg_type is not None:
                # Get specific message type
                if msg_type not in self._queues:
                    return None
                
                try:
                    if timeout is None:
                        return self._queues[msg_type].get_nowait()
                    else:
                        return self._queues[msg_type].get(timeout=timeout)
                except queue.Empty:
                    return None
            else:
                # Get any message type (check all queues, return most recent)
                if timeout is None:
                    # Get immediately from any non-empty queue
                    for queue_obj in self._queues.values():
                        try:
                            return queue_obj.get_nowait()
                        except queue.Empty:
                            continue
                    return None
                else:
                    # Wait for any message with timeout
                    start_time = time.time()
                    while time.time() - start_time < timeout:
                        for queue_obj in self._queues.values():
                            try:
                                return queue_obj.get_nowait()
                            except queue.Empty:
                                continue
                        time.sleep(0.001)  # Small sleep to avoid busy waiting
                    return None
    
    @overload
    def get_nowait(self, msg_type: Type[Message]) -> Optional[Message]:
        """Get the latest message of a specific type without blocking"""
        ...
    
    @overload
    def get_nowait(self) -> Optional[Message]:
        """Get the latest message of any type without blocking"""
        ...
    
    def get_nowait(self, msg_type: Optional[Type[Message]] = None) -> Optional[Message]:
        """Get the latest message of a specific type or any type without blocking"""
        return self.get(msg_type, timeout=None)
    
    def get_all_of_type(self, msg_type: Type[Message]) -> List[Message]:
        """Get all messages of a specific type (newest first due to LIFO)"""
        messages = []
        with self._lock:
            if msg_type in self._queues:
                try:
                    while True:
                        messages.append(self._queues[msg_type].get_nowait())
                except queue.Empty:
                    pass
        return messages
    
    def get_all(self) -> List[Message]:
        """Get all messages of all types"""
        messages = []
        with self._lock:
            for queue_obj in self._queues.values():
                try:
                    while True:
                        messages.append(queue_obj.get_nowait())
                except queue.Empty:
                    continue
        return messages
    
    def peek(self, msg_type: Optional[Type[Message]] = None) -> Optional[Message]:
        """Peek at the latest message without removing it"""
        with self._lock:
            if msg_type is not None:
                if msg_type not in self._queues:
                    return None
                try:
                    msg = self._queues[msg_type].get_nowait()
                    self._queues[msg_type].put(msg)  # Put it back
                    return msg
                except queue.Empty:
                    return None
            else:
                # Peek at any message type
                for queue_obj in self._queues.values():
                    try:
                        msg = queue_obj.get_nowait()
                        queue_obj.put(msg)  # Put it back
                        return msg
                    except queue.Empty:
                        continue
                return None
    
    def has_message(self, msg_type: Optional[Type[Message]] = None) -> bool:
        """Check if there are any messages of a specific type or any type"""
        with self._lock:
            if msg_type is not None:
                if msg_type not in self._queues:
                    return False
                return not self._queues[msg_type].empty()
            else:
                # Check if any queue has messages
                return any(not queue_obj.empty() for queue_obj in self._queues.values())
    
    def get_message_types(self) -> List[Type[Message]]:
        """Get all message types currently in the routing queue"""
        with self._lock:
            return [msg_type for msg_type, queue_obj in self._queues.items() if not queue_obj.empty()]
    
    def count_messages(self, msg_type: Optional[Type[Message]] = None) -> int:
        """Count messages of a specific type or all types"""
        with self._lock:
            if msg_type is not None:
                if msg_type not in self._queues:
                    return 0
                return self._queues[msg_type].qsize()
            else:
                return sum(queue_obj.qsize() for queue_obj in self._queues.values())
    
    def clear(self, msg_type: Optional[Type[Message]] = None):
        """Clear messages of a specific type or all types"""
        with self._lock:
            if msg_type is not None:
                if msg_type in self._queues:
                    try:
                        while True:
                            self._queues[msg_type].get_nowait()
                    except queue.Empty:
                        pass
            else:
                # Clear all queues
                for queue_obj in self._queues.values():
                    try:
                        while True:
                            queue_obj.get_nowait()
                    except queue.Empty:
                        pass
                self._queues.clear()
    
    def empty(self, msg_type: Optional[Type[Message]] = None) -> bool:
        """Check if a specific message type queue or all queues are empty"""
        with self._lock:
            if msg_type is not None:
                if msg_type not in self._queues:
                    return True
                return self._queues[msg_type].empty()
            else:
                return all(queue_obj.empty() for queue_obj in self._queues.values())


class Comm(ABC):
    def __init__(self, 
                 logger: Logger,
                 node_info: NodeInfo, 
                 parent_comm: "Comm"= None, 
                 heartbeat_interval: int = 1):
        
        self.logger = logger
        self.node_info = node_info
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_comm = parent_comm
        self._cache: Dict[str, MessageRoutingQueue] = {}

    def init_cache(self):
        for child_id in self.node_info.children_ids:
            self._cache[child_id] = MessageRoutingQueue()
        
        if self.node_info.parent_id:
            self._cache[self.node_info.parent_id] = MessageRoutingQueue()

    def update_node_info(self,node_info: NodeInfo):
        self.node_info = node_info
        for child_id in self.node_info.children_ids:
            if child_id not in self._cache:
                self._cache[child_id] = MessageRoutingQueue()

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

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def pickable_copy(self):
        pass

    def _send_to_children(self, data: Any) -> Dict[str, bool]:
        ret = {}
        for child_id in self.node_info.children_ids:
            ret[child_id] = self._send_to_child(child_id, data)
        return ret

    def _recv_from_children(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        messages = {}
        for child_id in self.node_info.children_ids:
            msg = self._recv_from_child(child_id, timeout)
            messages[child_id] = msg
        return messages

    def recv_message_from_child(self,cls: Type[Message], child_id: str, timeout: Optional[float] = None) -> Message | None:
        """Receive a specific message type from child node with efficient type-based caching."""
        if child_id not in self._cache:
            return None
        
        # First check cache for existing message of this type
        routing_queue = self._cache[child_id]
        msg = routing_queue.get_nowait(cls)
        if msg is not None:
            return msg
        
        # If not in cache and timeout specified, wait for new messages
        if timeout is not None:
            start = time.time()
            while time.time() - start < timeout:
                # Check cache again (async monitoring might have added messages)
                msg = routing_queue.get_nowait(cls)
                if msg is not None:
                    return msg
                time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        return None

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
        """Receive a specific message type from parent node with efficient type-based caching."""
        parent_id = self.node_info.parent_id
        if parent_id is None or parent_id not in self._cache:
            return None
        
        # First check cache for existing message of this type
        routing_queue = self._cache[parent_id]
        msg = routing_queue.get_nowait(cls)
        if msg is not None:
            return msg
        
        # If not in cache and timeout specified, wait for new messages
        if timeout is not None:
            start = time.time()
            while time.time() - start < timeout:
                # Check cache again (async monitoring might have added messages)
                msg = routing_queue.get_nowait(cls)
                if msg is not None:
                    return msg
                time.sleep(0.01)  # Small sleep to avoid busy waiting
        
        return None

    def sync_heartbeat_with_parent(self, timeout: Optional[float] = None) -> bool:
        #heart beat sync with parent
        if self.node_info.parent_id is None:
            return True
        
        self.send_message_to_parent(HeartBeat())
        msg = self.recv_message_from_parent(HeartBeat,timeout=timeout)
        if msg is not None:
            return True
        return False
        
    def sync_heartbeat_with_child(self, child_id: str, timeout: Optional[float] = None) -> bool:
        if len(self.node_info.children_ids) == 0:
            return True
    
        msg = self.recv_message_from_child(HeartBeat,child_id, timeout=timeout)
        self.send_message_to_child(child_id, HeartBeat())
        if msg is not None:
            return True
        return False
    
    def sync_heartbeat_with_children(self, timeout: Optional[float] = None) -> bool:
        status = []
        for child_id in self.node_info.children_ids:
            status.append(self.sync_heartbeat_with_child(child_id, timeout=timeout))
        return all(status)
    
    def async_recv(self):
        """This method starts a thread that continuously monitor the endpoints and push to self._cache"""
        
        def _monitor_parent():
            """Monitor messages from parent and cache them"""
            while getattr(self, '_stop_monitoring', False) is False:
                try:
                    msg = self._recv_from_parent(timeout=0.1)
                    if msg is not None and self.node_info.parent_id is not None:
                        if isinstance(msg, Message):
                            self._cache[self.node_info.parent_id].put(msg)
                    else:
                        # Add small sleep when no messages to reduce CPU usage
                        time.sleep(0.01)
                except Exception as e:
                    self.logger.error(f"Error monitoring parent: {e}")
                    time.sleep(0.1)  # Longer sleep on error
        
        def _monitor_children():
            """Monitor messages from all children and cache them"""
            while getattr(self, '_stop_monitoring', False) is False:
                try:
                    messages_received = 0
                    for child_id in self.node_info.children_ids:
                        msg = self._recv_from_child(child_id, timeout=0.01)  # Shorter timeout
                        if msg is not None:
                            if isinstance(msg, Message):
                                self._cache[child_id].put(msg)
                                messages_received += 1
                    
                    # Only sleep if no messages were received to reduce CPU usage
                    if messages_received == 0:
                        time.sleep(0.01)
                        
                except Exception as e:
                    self.logger.error(f"Error monitoring children: {e}")
                    time.sleep(0.1)  # Longer sleep on error
        
        # Initialize cache if not already done
        if not self._cache:
            self.init_cache()
        
        # Initialize stop flag
        self._stop_monitoring = False
        
        # Start monitoring threads
        if self.node_info.parent_id is not None:
            self._parent_thread = threading.Thread(target=_monitor_parent, daemon=True)
            self._parent_thread.start()
        
        if self.node_info.children_ids:
            self._children_thread = threading.Thread(target=_monitor_children, daemon=True)
            self._children_thread.start()
    
    def stop_async_recv(self):
        """Stop the async monitoring threads"""
        self._stop_monitoring = True
        if hasattr(self, '_parent_thread') and self._parent_thread.is_alive():
            self._parent_thread.join(timeout=1.0)
        if hasattr(self, '_children_thread') and self._children_thread.is_alive():
            self._children_thread.join(timeout=1.0)
    
    def clear_cache(self):
        """Close all cache queues and clear remaining messages"""
        for routing_queue in self._cache.values():
            routing_queue.clear()
        self._cache.clear()


