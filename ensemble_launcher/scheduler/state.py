from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from typing_extensions import TypedDict

from .resource import JobResource


class ChildrenAssignment(TypedDict):
    """Per-child assignment stored in AsyncChildrenScheduler._child_assignments."""

    job_resource: JobResource
    task_ids: List[str]
    wid: int
    task_executor_name: str
    child_class: str


class SchedulerState(BaseModel):
    """
    Snapshot of scheduler state for fault tolerance and checkpointing.

    Covers both AsyncChildrenScheduler (worker_* fields) and
    AsyncTaskScheduler (task status sets).  Either set of fields may be
    left at its default (empty) value when only one scheduler type needs
    to be captured.

    The ``nodes`` and ``children_resources`` fields hold ``JobResource``
    dataclass instances.  Custom serialisers/validators ensure they
    round-trip cleanly through ``model_dump_json`` / ``model_validate_json``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str
    level: Optional[int] = None
    nodes: Optional[JobResource] = None

    # ------------------------------------------------------------------ #
    # Task status sets (AsyncTaskScheduler)                               #
    # ------------------------------------------------------------------ #

    pending_tasks: Set[str] = Field(default_factory=set)
    running_tasks: Set[str] = Field(default_factory=set)
    completed_tasks: Set[str] = Field(default_factory=set)
    failed_tasks: Set[str] = Field(default_factory=set)

    # ------------------------------------------------------------------ #
    # Children bookkeeping (AsyncChildrenScheduler)                           #
    # ------------------------------------------------------------------ #

    # child_id -> task ids assigned to that child
    children_task_ids: Dict[str, List[str]] = Field(default_factory=dict)

    # child_id -> cluster resources allocated to that child
    children_resources: Dict[str, JobResource] = Field(default_factory=dict)

    # child_id -> extra keyword args forwarded from policy
    # (e.g. {"task_executor_name": "mpi"})
    children_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Bidirectional wid <-> child_id maps (AsyncChildrenScheduler only)
    child_id_to_wid: Dict[str, int] = Field(default_factory=dict)
    wid_to_child_id: Dict[int, str] = Field(default_factory=dict)

    # A map of task_id -> child_id
    task_to_child: Dict[str, str] = Field(default_factory=dict)

    # A map of task_id -> client_id (cluster mode only)
    task_to_client: Dict[str, str] = Field(default_factory=dict)

    # Informational snapshot of child states at checkpoint time.
    # Not used to restore state: children always restart as NOTREADY.
    child_states: Dict[str, str] = Field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Custom serialisers / validators for JobResource fields              #
    # ------------------------------------------------------------------ #

    @field_serializer("nodes")
    def _serialize_nodes(self, v: Optional[JobResource]) -> Optional[Dict[str, Any]]:
        return v.serialize() if v is not None else None

    @field_validator("nodes", mode="before")
    @classmethod
    def _validate_nodes(cls, v: Any) -> Optional[JobResource]:
        if isinstance(v, dict):
            return JobResource.deserialize(v)
        return v

    @field_serializer("children_resources")
    def _serialize_children_resources(
        self, v: Dict[str, JobResource]
    ) -> Dict[str, Any]:
        return {k: jr.serialize() for k, jr in v.items()}

    @field_validator("children_resources", mode="before")
    @classmethod
    def _validate_children_resources(cls, v: Any) -> Dict[str, JobResource]:
        if isinstance(v, dict):
            return {
                k: JobResource.deserialize(val) if isinstance(val, dict) else val
                for k, val in v.items()
            }
        return v
