import asyncio
import os
import time
import uuid
from abc import ABC, abstractmethod
from asyncio import Queue
from logging import Logger
from typing import Any, Dict, List, Optional, Type, TypeVar

from pydantic import BaseModel

from ensemble_launcher.profiling import EventRegistry, get_registry

from .messages import Message, all_messages
from .nodeinfo import NodeInfo

T = TypeVar("T", bound="AsyncCommState")


class AsyncCommState(BaseModel):
    def serialize(self, *args, **kwargs) -> str:
        return self.model_dump_json(*args, **kwargs)

    @classmethod
    def deserialize(cls: Type[T], data: str) -> T:
        return cls.model_validate_json(data)


class AsyncMessageRoutingQueue:
    """An async routing queue that organizes messages by type using separate LifoQueues. Not thread-safe."""

    def __init__(
        self, logger: Logger, message_types: Optional[List[Type[Message]]] = None
    ):
        self.logger = logger
        self._queues: Dict[Type[Message], Queue] = {}
        self._message_types = message_types
        if message_types is not None:
            for msg_type in message_types:
                self._queues[msg_type] = Queue()

    async def put(self, message: Message):
        """Put a message into the appropriate type-specific queue"""
        msg_type = type(message)
        if msg_type not in self._queues:
            self._queues[msg_type] = Queue()
            self.logger.debug(
                f"Created new queue for message type: {msg_type.__name__}"
            )
        await self._queues[msg_type].put(message)

    def put_nowait(self, message: Message):
        """Put a message into the appropriate type-specific queue"""
        msg_type = type(message)
        if msg_type not in self._queues:
            self._queues[msg_type] = Queue()
            self.logger.debug(
                f"Created new queue for message type: {msg_type.__name__}"
            )
        self._queues[msg_type].put_nowait(message)

    async def get(
        self, msg_type: Type[Message], timeout: Optional[float] = None
    ) -> Optional[Message]:
        """Get the latest message of a specific type or any type if msg_type is None"""

        # Get specific message type
        if msg_type not in self._queues:
            self.logger.warning(f"No messages of type {msg_type.__name__} available")
            return None

        try:
            self.logger.debug(
                f"Waiting for message of type {msg_type.__name__} with timeout {timeout}"
            )
            msg = await asyncio.wait_for(self._queues[msg_type].get(), timeout=timeout)
            self.logger.debug(
                f"Retrieved message of type {msg_type.__name__} with timeout {timeout}"
            )
            return msg
        except asyncio.TimeoutError:
            self.logger.debug(
                f"No messages of type {msg_type.__name__} available within timeout {timeout}s"
            )
            return None
        except asyncio.QueueEmpty:
            self.logger.debug(f"Queue of type {msg_type.__name__} is empty")
            return None

    def get_nowait(self, msg_type: Type[Message]) -> Optional[Message]:
        """Get the latest message of a specific type or any type without blocking"""
        # Get specific message type
        if msg_type not in self._queues:
            self.logger.warning(f"No messages of type {msg_type.__name__} available")
            return None

        try:
            msg = self._queues[msg_type].get_nowait()
            self.logger.debug(
                f"Retrieved message of type {msg_type.__name__} without blocking"
            )
            return msg
        except asyncio.QueueEmpty:
            self.logger.debug(
                f"No messages of type {msg_type.__name__} available in queue"
            )
            return None

    def clear(self, msg_type: Optional[Type[Message]] = None):
        """Clear messages of a specific type or all types"""
        if msg_type is not None:
            if msg_type in self._queues:
                try:
                    while True:
                        self._queues[msg_type].get_nowait()
                except asyncio.QueueEmpty:
                    pass
        else:
            # Clear all queues
            for queue_obj in self._queues.values():
                try:
                    while True:
                        queue_obj.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            self._queues.clear()

    def empty(self, msg_type: Optional[Type[Message]] = None) -> bool:
        """Check if a specific message type queue or all queues are empty"""
        if msg_type is not None:
            if msg_type not in self._queues:
                self.logger.warning(
                    f"No messages of type {msg_type.__name__} available"
                )
                return True
            return self._queues[msg_type].empty()
        else:
            return all(queue_obj.empty() for queue_obj in self._queues.values())


class AsyncComm(ABC):
    def __init__(
        self,
        logger: Logger,
        node_info: NodeInfo,
        parent_comm: "AsyncComm" = None,
        heartbeat_interval: int = 1,
    ):

        self.logger = logger
        self._node_info = node_info
        self.last_update_time = time.time()
        self.last_heartbeat_time = None
        self.heartbeat_interval = heartbeat_interval
        self._parent_comm = parent_comm
        self._cache: Dict[str, AsyncMessageRoutingQueue] = {}
        self._stop_event = None

        self.parent_dead_event: Optional[asyncio.Event] = None
        self._child_dead_events: Dict[str, asyncio.Event] = {}

        self._event_registry: Optional[EventRegistry] = None
        if os.getenv("EL_ENABLE_PROFILING", "0") == "1":
            self.event_registry: EventRegistry = get_registry()

    async def init_cache(self):
        for child_id in self._node_info.children_ids:
            if child_id in self._cache:
                continue
            self.logger.info(f"Initializing cache for child_id: {child_id}")
            self._cache[child_id] = AsyncMessageRoutingQueue(
                logger=self.logger, message_types=all_messages
            )

        if self._node_info.parent_id and self._node_info.parent_id not in self._cache:
            self.logger.info(
                f"Initializing cache for parent_id: {self._node_info.parent_id}"
            )
            self._cache[self._node_info.parent_id] = AsyncMessageRoutingQueue(
                logger=self.logger, message_types=all_messages
            )

    async def update_node_info(self, node_info: NodeInfo):
        removed_children = set(self._node_info.children_ids) - set(
            node_info.children_ids
        )
        for child_id in removed_children:
            self._cache.pop(child_id, None)
        self._node_info = node_info
        await self.init_cache()

    @abstractmethod
    async def _send_to_parent(self, data: Any, **kwargs) -> bool:
        pass

    @abstractmethod
    async def _recv_from_parent(self, timeout: Optional[float] = None, **kwargs) -> Any:
        pass

    @abstractmethod
    async def _send_to_child(self, child_id: str, data: Any, **kwargs) -> bool:
        pass

    @abstractmethod
    async def _recv_from_child(
        self, child_id: str, timeout: Optional[float] = None, **kwargs
    ) -> Any:
        pass

    async def close(self):
        """Base cleanup - signal stop and clear cache"""
        if self._stop_event:
            self._stop_event.set()

        await self.clear_cache()

    @abstractmethod
    def pickable_copy(self):
        pass

    @classmethod
    @abstractmethod
    def set_state(self, state: AsyncCommState) -> "AsyncComm":
        pass

    @abstractmethod
    def get_state(self) -> AsyncCommState:
        pass

    async def start_monitors(
        self, parent_only: bool = False, children_only: bool = False
    ):
        """Start background tasks to monitor communication endpoints.
        Base implementation just initializes cache and stop event.
        Subclasses should override to add their specific monitoring tasks."""
        await self.init_cache()
        if self._stop_event is None:
            self._stop_event = asyncio.Event()

    async def recv_message_from_child(
        self,
        cls: Type[Message],
        child_id: str,
        block: bool = False,
        timeout: Optional[float] = None,
    ) -> Message | None:
        """
        Receive a specific message type from child node with efficient type-based caching.
        If block is False:
            timeout is None, it will return immediately if no message is available.
            timeout is specified, it will wait up to timeout seconds for a message of the specified type.
        If block is True:
            timeout is None, it will wait indefinitely for a message of the specified type.
            timeout is specified, it will wait up to timeout seconds for a message of the specified type.
        """
        if child_id not in self._cache:
            self.logger.warning(
                f"{child_id} not in cache. Current keys {self._cache.keys()}"
            )
            return None

        if block is False and timeout is None:
            # First check cache for existing message of this type
            routing_queue = self._cache[child_id]
            msg = routing_queue.get_nowait(cls)
        else:
            routing_queue = self._cache[child_id]
            msg = await routing_queue.get(cls, timeout=timeout)

        return msg

    async def send_message_to_child(self, child_id: str, msg: Message) -> bool:
        return await self._send_to_child(child_id=child_id, data=msg)

    async def send_message_to_parent(self, msg: Message) -> bool:
        return await self._send_to_parent(data=msg)

    async def recv_message_from_parent(
        self, cls: Type[Message], block: bool = False, timeout: Optional[float] = None
    ) -> Message | None:
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
            self.logger.warning("No parent available to receive message from.")
            return None

        if block is False and timeout is None:
            # First check cache for existing message of this type
            routing_queue = self._cache[parent_id]
            msg = routing_queue.get_nowait(cls)
        else:
            routing_queue = self._cache[parent_id]
            msg = await routing_queue.get(cls, timeout=timeout)
        return msg

    async def sync_heartbeat_with_parent(self, timeout: Optional[float] = None) -> bool:
        # Default implementation: subclasses override with HB process logic.
        return True

    async def sync_heartbeat_with_child(
        self, child_id: str, timeout: Optional[float] = None
    ) -> bool:
        # Default implementation: subclasses override with HB process logic.
        return True

    async def clear_cache(self):
        """Close all cache queues and clear remaining messages"""
        for routing_queue in self._cache.values():
            routing_queue.clear()
        self._cache.clear()
