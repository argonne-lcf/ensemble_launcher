from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type, Union, overload
from .messages import Message, Status, all_messages
from dataclasses import dataclass, field
import time
from logging import Logger
import threading
import queue
from datetime import datetime
import asyncio
from .nodeinfo import NodeInfo

class MessageRoutingQueue:
    """A routing queue that organizes messages by type using separate LifoQueues"""
    
    def __init__(self,logger: Logger, message_types: Optional[List[Type[Message]]] = None):
        self.logger = logger
        self._queues: Dict[Type[Message], queue.Queue] = {}
        if message_types is not None:
            for msg_type in message_types:
                self._queues[msg_type] = queue.Queue()
    
    def put(self, message: Message):
        """Put a message into the appropriate type-specific queue"""
        msg_type = type(message)
        if msg_type not in self._queues:
            self._queues[msg_type] = queue.Queue()
            self.logger.debug(f"Created new queue for message type: {msg_type.__name__}")
        self._queues[msg_type].put(message)
    
    def get(self, msg_type: Type[Message], timeout: Optional[float] = None) -> Optional[Message]:
        """Get the latest message of a specific type or any type if msg_type is None"""
        
        # Get specific message type
        if msg_type not in self._queues:
            self.logger.debug(f"No messages of type {msg_type.__name__} available")
            return None
        
        try:
            return self._queues[msg_type].get(timeout=timeout)
        except queue.Empty:
            self.logger.debug(f"No messages of type {msg_type.__name__} available within timeout {timeout}s")
            return None
    
    def get_nowait(self, msg_type: Type[Message]) -> Optional[Message]:
        """Get the latest message of a specific type or any type without blocking"""
    
        # Get specific message type
        if msg_type not in self._queues:
            self.logger.debug(f"No messages of type {msg_type.__name__} available")
            return None
        
        try:
            return self._queues[msg_type].get_nowait()
        except queue.Empty:
            self.logger.debug(f"No messages of type {msg_type.__name__} available in queue")
            return None
    
    def clear(self, msg_type: Optional[Type[Message]] = None):
        """Clear messages of a specific type or all types"""

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
    
        if msg_type is not None:
            if msg_type not in self._queues:
                self.logger.debug(f"No messages of type {msg_type.__name__} available")
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
        self._node_info = node_info
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_comm = parent_comm
        self._cache: Dict[str, MessageRoutingQueue] = {}

    def init_cache(self):
        for child_id in self._node_info.children_ids:
            if child_id in self._cache:
                continue
            self.logger.info(f"Initializing cache for child_id: {child_id}")
            self._cache[child_id] = MessageRoutingQueue(self.logger, message_types=all_messages)
        
        if self._node_info.parent_id and self._node_info.parent_id not in self._cache:
            self.logger.info(f"Initializing cache for parent_id: {self._node_info.parent_id}")
            self._cache[self._node_info.parent_id] = MessageRoutingQueue(self.logger, message_types=all_messages)

    def update_node_info(self,node_info: NodeInfo):
        self._node_info = node_info
        self.init_cache()

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

    def recv_message_from_child(self,cls: Type[Message], child_id: str, block: bool = False, timeout: Optional[float] = None) -> Message | None:
        """Receive a specific message type from child node with efficient type-based caching.
            If block is False: 
                timeout is None, it will return immediately if no message is available.
                timeout is specified, it will wait up to timeout seconds for a message of the specified type.
            If block is True:
                timeout is None, it will wait indefinitely for a message of the specified type.
                timeout is specified, it will wait up to timeout seconds for a message of the specified type.
        """
        if child_id not in self._cache:
            return None
        
        routing_queue = self._cache[child_id]
        if block is False and timeout is None:
            msg = routing_queue.get_nowait(cls)
        else:
            msg = routing_queue.get(cls, timeout=timeout)
        
        return msg
    
    def send_message_to_child(self, child_id: str, msg: Message) -> bool:
        return self._send_to_child(child_id=child_id, data=msg)

    def send_message_to_parent(self, msg: Message) -> bool:
        """Send a message to the parent node."""
        return self._send_to_parent(data=msg)

    def recv_message_from_parent(self, cls: Type[Message], block: bool = False, timeout: Optional[float] = None) -> Message | None:
        """Receive a specific message type from parent node with efficient type-based caching.
            If block is False: 
                timeout is None, it will return immediately if no message is available.
                timeout is specified, it will wait up to timeout seconds for a message of the specified type.
            If block is True:
                timeout is None, it will wait indefinitely for a message of the specified type.
                timeout is specified, it will wait up to timeout seconds for a message of the specified type.
        """
        parent_id = self._node_info.parent_id
        if parent_id is None or parent_id not in self._cache:
            return None
        
        if block is False and timeout is None:
            msg = self._cache[parent_id].get_nowait(cls)
        else:
            msg = self._cache[parent_id].get(cls, timeout=timeout)
        
        return msg

    def sync_heartbeat_with_parent(self, timeout: Optional[float] = None) -> bool:
        #heart beat sync with parent
        if self._node_info.parent_id is None:
            return True
        
        self.send_message_to_parent(HeartBeat())
        msg = self.recv_message_from_parent(HeartBeat,timeout=timeout)
        if msg is not None:
            return True
        return False
        
    def sync_heartbeat_with_child(self, child_id: str, timeout: Optional[float] = None) -> bool:
        if len(self._node_info.children_ids) == 0:
            return True
    
        msg = self.recv_message_from_child(HeartBeat,child_id, timeout=timeout)
        self.send_message_to_child(child_id, HeartBeat())
        if msg is not None:
            return True
        return False
    
    def sync_heartbeat_with_children(self, timeout: Optional[float] = None) -> bool:
        status = []
        for child_id in self._node_info.children_ids:
            status.append(self.sync_heartbeat_with_child(child_id, timeout=timeout))
        return all(status)
    
    def async_recv(self):
        """Start async monitoring threads for parent and children"""
        self.async_recv_parent()
        self.async_recv_children()
        
    def async_recv_parent(self):
        """Start a thread that continuously monitors the parent endpoint and pushes to self._cache"""
        
        def _monitor_parent():
            """Monitor messages from parent and cache them"""
            while getattr(self, '_stop_monitoring_parent', False) is False:
                try:
                    msg = self._recv_from_parent(timeout=0.1)
                    if msg is not None and self._node_info.parent_id is not None:
                        if isinstance(msg, Message):
                            self._cache[self._node_info.parent_id].put(msg)
                except Exception as e:
                    self.logger.error(f"Error monitoring parent: {e}")
                    time.sleep(0.1)  # Longer sleep on error
        
        # Initialize cache if not already done
        if not self._cache:
            self.init_cache()
        
        # Initialize stop flag
        self._stop_monitoring_parent = False
        
        # Start monitoring thread
        if self._node_info.parent_id is not None:
            self._parent_thread = threading.Thread(target=_monitor_parent, daemon=True)
            self._parent_thread.start()
    
    def async_recv_children(self):
        """Start a thread that continuously monitors the children endpoints and pushes to self._cache"""
        
        def _monitor_children():
            """Monitor messages from all children and cache them"""
            while getattr(self, '_stop_monitoring_children', False) is False:
                try:
                    for child_id in self._node_info.children_ids:
                        msg = self._recv_from_child(child_id, timeout=0.1)
                        if msg is not None:
                            if isinstance(msg, Message):
                                self._cache[child_id].put(msg)
                # No sleep needed - _recv_from_child handles blocking
                        
                except Exception as e:
                    self.logger.error(f"Error monitoring children: {e}")
                    time.sleep(0.1)  # Only sleep on error
        
        # Initialize cache if not already done
        if not self._cache:
            self.init_cache()
        
        # Initialize stop flag
        self._stop_monitoring_children = False
        
        # Start monitoring thread
        if self._node_info.children_ids:
            self._children_thread = threading.Thread(target=_monitor_children, daemon=True)
            self._children_thread.start()
    
    def stop_async_recv(self):
        """Stop the async monitoring threads"""
        self._stop_monitoring_parent = True
        self._stop_monitoring_children = True
        if hasattr(self, '_parent_thread') and self._parent_thread.is_alive():
            self._parent_thread.join(timeout=1.0)
        if hasattr(self, '_children_thread') and self._children_thread.is_alive():
            self._children_thread.join(timeout=1.0)
    
    def clear_cache(self):
        """Close all cache queues and clear remaining messages"""
        for routing_queue in self._cache.values():
            routing_queue.clear()
        self._cache.clear()