from typing import Protocol, Any, Optional

class QueueProtocol(Protocol):
    """Defines the structural interface for all queue wrappers in the project."""

    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> None:
        """Add an item to the queue."""
        ...

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Any:
        """Remove and return an item from the queue."""
        ...

    def empty(self) -> bool:
        """Return True if the queue is empty, False otherwise."""
        ...