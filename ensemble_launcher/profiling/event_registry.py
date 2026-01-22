"""Event registry for profiling and timeline visualization."""

import time
import json
import threading
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Literal
from collections import defaultdict
from contextlib import contextmanager


@dataclass
class Event:
    """Represents a single profiling event for Perfetto visualization.
    
    Perfetto Trace Format Mapping:
    ==============================
    
    Attributes:
        name (str): 
            - Event name/label displayed in Perfetto UI
            - Perfetto field: "name"
            - Appears as: The text label on timeline bars/markers
            - Example: "process_task", "http_request", "computation"
            - Best practice: Use descriptive, hierarchical names like "worker.process_task"
        
        category (str): 
            - Category for grouping and filtering events
            - Perfetto field: "cat" 
            - Appears as: Filter dropdown in Perfetto UI, color coding
            - Example: "task", "comm", "scheduler", "io", "compute"
            - Best practice: Use consistent categories across your app
            - Multiple categories: Can use comma-separated like "task,compute"
        
        timestamp (float): 
            - Event start time in seconds (from time.perf_counter())
            - Perfetto field: "ts" (converted to microseconds)
            - Appears as: Horizontal position on timeline (when event occurs)
            - Note: Relative to first event's timestamp for display
            - Example: 1234.567890 seconds → 1234567890 microseconds in Perfetto
        
        event_type (Literal["B", "E", "X", "i", "C", "M"]): 
            - Type of event determines how Perfetto renders it
            - Perfetto field: "ph" (phase)
            - Values:
              * "X" (Complete/Duration): Solid bar with duration
                - Shows: Colored bar spanning the duration
                - Use: Context managers, operations with known duration
                - Single event contains both start time and duration
              
              * "B" (Begin): Start marker of async operation
                - Shows: Left edge of a bar/span
                - Renders: Paired with "E" to create a solid bar
                - Use: Async operation start, must pair with "E"
                - Matching: B/E paired by same name + pid + tid
              
              * "E" (End): End marker of async operation  
                - Shows: Right edge of a bar/span
                - Renders: Paired with "B" to create a solid bar
                - Use: Async operation end, must match "B" event
                - Matching: Must have same name, pid, tid as the B event
                - Result: B/E pair renders as solid bar (like X event)
              
              * "i" (Instant): Point-in-time marker
                - Shows: Vertical line or triangle marker
                - Use: Milestones, events, state changes
              
              * "C" (Counter): Numeric value over time
                - Shows: Line graph in separate track
                - Use: Metrics, gauges (queue depth, memory usage)
              
              * "M" (Metadata): Process/thread naming
                - Shows: Track labels in UI
                - Use: Auto-generated for process/thread names
        
        node_id (str): 
            - ID of node/component that generated this event
            - Perfetto field: Added to "args" metadata
            - Appears as: In event details tooltip when clicked
            - Example: "master_0", "worker_3", "scheduler"
            - Best practice: Use hierarchical IDs for distributed systems
            - Visualization: Can help identify which component did what
        
        tid (Optional[int]): 
            - Thread ID within the process
            - Perfetto field: "tid"
            - Appears as: Separate horizontal track for each thread
            - Example: 12345 (OS thread ID)
            - Effect: Events with same tid show on same track
            - Use: Multi-threaded applications to see thread activity
            - Layout: Threads grouped under their parent process
        
        pid (Optional[int]): 
            - Process ID 
            - Perfetto field: "pid"
            - Appears as: Separate section for each process in UI
            - Example: 98765 (OS process ID)
            - Effect: Primary grouping in Perfetto timeline
            - Use: Multi-process applications, distributed systems
            - Layout: Top-level grouping, contains thread tracks
            - Tip: Can use virtual PIDs to group logical components
        
        duration (Optional[float]): 
            - Duration in seconds (only for "X" Complete events)
            - Perfetto field: "dur" (converted to microseconds)
            - Appears as: Width of the bar on timeline
            - Example: 0.523 seconds → 523000 microseconds
            - Note: Only used for event_type="X", ignored for B/E/i/C
            - Calculation: End time - start time
        
        task_id (Optional[str]): 
            - ID of associated task/job/request
            - Perfetto field: Added to "args" metadata
            - Appears as: In event details tooltip, can filter by this
            - Example: "task_42", "request_abc123"
            - Use: Track individual tasks through system
            - Best practice: Use same task_id across all events for one task
            - Benefit: Can search/filter for specific task in UI
        
        metadata (Dict[str, Any]): 
            - Additional key-value pairs for custom data
            - Perfetto field: Added to "args" object
            - Appears as: In event details panel when event is selected
            - Example: {"bytes": 1024, "status": "success", "retries": 2}
            - Use: Any contextual information about the event
            - Types: Supports strings, numbers, booleans
            - Visualization: Shows in right panel as key: value list
    
    Perfetto UI Layout:
    ===================
    
    Timeline View (horizontal):
    
        Process 1 (pid=100) "master (pid:100)"
        ├─ Thread 1 (tid=101)
        │  ├─ [Complete Event Bar]─────────────┤  ← name="process_task", dur=0.5s
        │  └─ △ Instant                             ← name="checkpoint", event_type="i"
        └─ Thread 2 (tid=102)
           └─ [Another Event]──────┤
        
        Process 2 (pid=200) "worker_1 (pid:200)"
        └─ Thread 1 (tid=201)
           ├─ B Begin                    E End ┤    ← name="async_op", paired B/E
           └─ [Nested Event]────┤
        
        Counters
        └─ queue_depth ────────────/────\─────     ← Line graph, event_type="C"
    
    Interaction:
    - Click event → Shows all metadata in right panel
    - Hover → Shows quick tooltip with name, duration, category
    - Select time range → Statistics for selected events
    - Search → Can filter by name, category, metadata values
    """
    name: str
    category: str
    timestamp: float
    event_type: Literal["B", "E", "X", "i", "C", "M"] = "X"  # B=begin, E=end, X=complete, i=instant, C=counter, M=metadata
    node_id: str = ""
    tid: Optional[int] = None
    pid: Optional[int] = None
    duration: Optional[float] = None
    task_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_perfetto_event(self, base_timestamp: float = 0) -> Dict[str, Any]:
        """Convert to Perfetto/Chrome Trace Format.
        
        Args:
            base_timestamp: Base timestamp to subtract for relative timing
            
        Returns:
            Dictionary in Perfetto format
        """
        # Convert timestamp to microseconds relative to base
        ts_us = int((self.timestamp - base_timestamp) * 1_000_000)
        
        event = {
            "name": self.name,
            "cat": self.category,
            "ph": self.event_type,
            "ts": ts_us,
            "pid": self.pid or 0,
            "tid": self.tid or 0,
        }
        
        # Add duration for complete events (in microseconds)
        # Note: B/E (Begin/End) events don't have duration, only X (Complete) events do
        if self.event_type == "X" and self.duration is not None:
            event["dur"] = int(self.duration * 1_000_000)
        
        # Add arguments (metadata + task_id + node_id)
        args = dict(self.metadata)
        if self.task_id:
            args["task_id"] = self.task_id
        if self.node_id:
            args["node_id"] = self.node_id
        if args:
            event["args"] = args
            
        return event


class EventRegistry:
    """Registry for collecting and managing profiling events.
    
    This class is thread-safe and can be used across multiple threads/processes.
    Events are stored in memory and can be exported to various formats including
    Perfetto/Chrome Trace Format for visualization.
    """
    
    def __init__(self, enabled: bool = True):
        """Initialize the event registry.
        
        Args:
            enabled: Whether event recording is enabled
        """
        self._enabled = enabled
        self._events: List[Event] = []
        self._lock = threading.Lock()
        self._base_timestamp: Optional[float] = None
        self._counters: Dict[str, List[tuple]] = defaultdict(list)  # For counter events
        
    def enable(self):
        """Enable event recording."""
        self._enabled = True
        
    def disable(self):
        """Disable event recording."""
        self._enabled = False
        
    @property
    def enabled(self) -> bool:
        """Check if recording is enabled."""
        return self._enabled
    
    def record(self, event: Event):
        """Record a single event.
        
        Args:
            event: Event to record
        """
        if not self._enabled:
            return
            
        with self._lock:
            # Set base timestamp on first event
            if self._base_timestamp is None:
                self._base_timestamp = event.timestamp
            self._events.append(event)
    
    def record_instant(
        self,
        name: str,
        category: str,
        node_id: str = "",
        tid: Optional[int] = None,
        pid: Optional[int] = None,
        task_id: Optional[str] = None,
        **metadata
    ):
        """Record an instant event (point in time).
        
        Args:
            name: Event name
            category: Event category
            node_id: Node ID
            tid: Thread ID
            pid: Process ID
            task_id: Task ID
            **metadata: Additional metadata
        """
        event = Event(
            name=name,
            category=category,
            timestamp=time.perf_counter(),
            event_type="i",
            node_id=node_id,
            tid=tid,
            pid=pid,
            task_id=task_id,
            metadata=metadata
        )
        self.record(event)
    
    def record_complete(
        self,
        name: str,
        category: str,
        duration: float,
        timestamp: float,
        node_id: str = "",
        tid: Optional[int] = None,
        pid: Optional[int] = None,
        task_id: Optional[str] = None,
        **metadata
    ):
        """Record a complete event (with duration).
        
        Args:
            name: Event name
            category: Event category
            duration: Event duration in seconds
            timestamp: Event start timestamp
            node_id: Node ID
            tid: Thread ID
            pid: Process ID
            task_id: Task ID
            **metadata: Additional metadata
        """
        event = Event(
            name=name,
            category=category,
            timestamp=timestamp,
            event_type="X",
            duration=duration,
            node_id=node_id,
            tid=tid,
            pid=pid,
            task_id=task_id,
            metadata=metadata
        )
        self.record(event)
    
    def record_begin(
        self,
        name: str,
        category: str,
        node_id: str = "",
        tid: Optional[int] = None,
        pid: Optional[int] = None,
        task_id: Optional[str] = None,
        **metadata
    ) -> float:
        """Record a begin event (start of an operation).
        
        Use with record_end() for async operations where begin and end
        happen in different contexts. Returns timestamp for use with record_end().
        
        Args:
            name: Event name
            category: Event category
            node_id: Node ID
            tid: Thread ID
            pid: Process ID
            task_id: Task ID
            **metadata: Additional metadata
            
        Returns:
            Timestamp of the begin event
        """
        timestamp = time.perf_counter()
        event = Event(
            name=name,
            category=category,
            timestamp=timestamp,
            event_type="B",
            node_id=node_id,
            tid=tid,
            pid=pid,
            task_id=task_id,
            metadata=metadata
        )
        self.record(event)
        return timestamp
    
    def record_end(
        self,
        name: str,
        category: str,
        node_id: str = "",
        tid: Optional[int] = None,
        pid: Optional[int] = None,
        task_id: Optional[str] = None,
        **metadata
    ):
        """Record an end event (end of an operation).
        
        Must be paired with a record_begin() call with the same name, pid, and tid.
        Perfetto matches B/E events by: name + pid + tid (all must match).
        
        Args:
            name: Event name (MUST match the begin event exactly)
            category: Event category
            node_id: Node ID
            tid: Thread ID (MUST match begin event if specified)
            pid: Process ID (MUST match begin event if specified)
            task_id: Task ID
            **metadata: Additional metadata
        """
        event = Event(
            name=name,
            category=category,
            timestamp=time.perf_counter(),
            event_type="E",
            node_id=node_id,
            tid=tid,
            pid=pid,
            task_id=task_id,
            metadata=metadata
        )
        self.record(event)
    
    def record_counter(
        self,
        name: str,
        value: float,
        category: str = "counter",
        node_id: str = "",
        pid: Optional[int] = None,
    ):
        """Record a counter value (for metrics over time).
        
        Args:
            name: Counter name
            value: Counter value
            category: Event category
            node_id: Node ID
            pid: Process ID
        """
        timestamp = time.perf_counter()
        with self._lock:
            if self._base_timestamp is None:
                self._base_timestamp = timestamp
            self._counters[name].append((timestamp, value, node_id, pid or 0))
    
    @contextmanager
    def measure(
        self,
        name: str,
        category: str,
        node_id: str = "",
        tid: Optional[int] = None,
        pid: Optional[int] = None,
        task_id: Optional[str] = None,
        **metadata
    ):
        """Context manager for measuring event duration.
        
        Usage:
            with registry.measure("process_task", "task", task_id="123"):
                # Do work
                pass
        
        Args:
            name: Event name
            category: Event category
            node_id: Node ID
            tid: Thread ID
            pid: Process ID
            task_id: Task ID
            **metadata: Additional metadata
        """
        if not self._enabled:
            yield
            return
            
        start_time = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start_time
            self.record_complete(
                name=name,
                category=category,
                duration=duration,
                timestamp=start_time,
                node_id=node_id,
                tid=tid,
                pid=pid,
                task_id=task_id,
                **metadata
            )
    
    def get_events(
        self,
        category: Optional[str] = None,
        node_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> List[Event]:
        """Query events with optional filters.
        
        Args:
            category: Filter by category
            node_id: Filter by node ID
            task_id: Filter by task ID
            
        Returns:
            List of matching events
        """
        with self._lock:
            events = self._events.copy()
        
        if category:
            events = [e for e in events if e.category == category]
        if node_id:
            events = [e for e in events if e.node_id == node_id]
        if task_id:
            events = [e for e in events if e.task_id == task_id]
            
        return events
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate statistics for all event categories.
        
        Returns:
            Dictionary with statistics per event name
        """
        stats = defaultdict(lambda: {"durations": [], "count": 0})
        
        with self._lock:
            for event in self._events:
                if event.duration is not None:
                    stats[event.name]["durations"].append(event.duration)
                stats[event.name]["count"] += 1
        
        # Compute summary statistics
        result = {}
        for name, data in stats.items():
            durations = data["durations"]
            result[name] = {
                "count": data["count"],
            }
            if durations:
                result[name].update({
                    "mean": sum(durations) / len(durations),
                    "sum": sum(durations),
                    "min": min(durations),
                    "max": max(durations),
                    "std": (sum((x - sum(durations) / len(durations)) ** 2 for x in durations) / len(durations)) ** 0.5 if len(durations) > 1 else 0.0,
                })
        
        return result
    
    def export_perfetto(self, filepath: str, include_metadata: bool = True):
        """Export events to Perfetto/Chrome Trace Format.
        
        This creates a JSON file that can be opened in:
        - Chrome/Edge: chrome://tracing
        - Perfetto UI: https://ui.perfetto.dev
        
        Args:
            filepath: Output file path (should end in .json)
            include_metadata: Include process/thread metadata
        """
        with self._lock:
            base_ts = self._base_timestamp or 0
            trace_events = []
            
            # Convert all events to Perfetto format
            for event in self._events:
                trace_events.append(event.to_perfetto_event(base_ts))
            
            # Add counter events
            for counter_name, values in self._counters.items():
                for timestamp, value, node_id, pid in values:
                    ts_us = int((timestamp - base_ts) * 1_000_000)
                    trace_events.append({
                        "name": counter_name,
                        "cat": "counter",
                        "ph": "C",
                        "ts": ts_us,
                        "pid": pid,
                        "args": {counter_name: value, "node_id": node_id}
                    })
            
            # Add process/thread name metadata
            if include_metadata:
                processes = set()
                threads = set()
                for event in self._events:
                    if event.pid:
                        processes.add((event.pid, event.node_id))
                    if event.tid and event.pid:
                        threads.add((event.pid, event.tid, event.node_id))
                
                for pid, node_id in processes:
                    trace_events.append({
                        "name": "process_name",
                        "ph": "M",
                        "pid": pid,
                        "args": {"name": f"{node_id} (pid:{pid})"}
                    })
                
                for pid, tid, node_id in threads:
                    trace_events.append({
                        "name": "thread_name",
                        "ph": "M",
                        "pid": pid,
                        "tid": tid,
                        "args": {"name": f"Thread {tid}"}
                    })
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump({"traceEvents": trace_events}, f, indent=2)
    
    def export_json(self, filepath: str):
        """Export events to simple JSON format.
        
        Args:
            filepath: Output file path
        """
        with self._lock:
            events_data = [asdict(event) for event in self._events]
        
        with open(filepath, 'w') as f:
            json.dump(events_data, f, indent=2)
    
    def clear(self):
        """Clear all recorded events."""
        with self._lock:
            self._events.clear()
            self._counters.clear()
            self._base_timestamp = None
    
    def __len__(self) -> int:
        """Return number of recorded events."""
        with self._lock:
            return len(self._events)


# Global registry instance
_global_registry: Optional[EventRegistry] = None
_registry_lock = threading.Lock()


def get_registry() -> EventRegistry:
    """Get the global event registry instance.
    
    Returns:
        Global EventRegistry instance
    """
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = EventRegistry(enabled=False)
    
    return _global_registry


def set_registry(registry: EventRegistry):
    """Set the global event registry instance.
    
    Args:
        registry: EventRegistry instance to set as global
    """
    global _global_registry
    with _registry_lock:
        _global_registry = registry
