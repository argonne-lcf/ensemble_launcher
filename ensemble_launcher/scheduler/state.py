from __future__ import annotations

from typing import Any, Dict, List, Set

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from .resource import JobResource


class WorkerAssignment(TypedDict):
    """Per-child assignment stored in AsyncWorkerScheduler._child_assignments."""

    job_resource: JobResource
    task_ids: List[str]
    wid: int


class SchedulerState(BaseModel):
    """
    Snapshot of scheduler state for fault tolerance and checkpointing.

    Covers both AsyncWorkerScheduler (worker_* fields) and
    AsyncTaskScheduler (task status sets).  Either set of fields may be
    left at its default (empty) value when only one scheduler type needs
    to be captured.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    node_id: str
    nodes: JobResource

    # ------------------------------------------------------------------ #
    # Task status sets (AsyncTaskScheduler)                               #
    # ------------------------------------------------------------------ #

    pending_tasks: Set[str] = Field(default_factory=set)
    running_tasks: Set[str] = Field(default_factory=set)
    completed_tasks: Set[str] = Field(default_factory=set)
    failed_tasks: Set[str] = Field(default_factory=set)

    # ------------------------------------------------------------------ #
    # Children bookkeeping (AsyncWorkerScheduler)                           #
    # ------------------------------------------------------------------ #

    # child_id -> task ids assigned to that child
    children_task_ids: Dict[str, List[str]] = Field(default_factory=dict)

    # child_id -> cluster resources allocated to that child
    children_resources: Dict[str, JobResource] = Field(default_factory=dict)

    # child_id -> extra keyword args forwarded from policy
    # (e.g. {"task_executor_name": "mpi"})
    children_kwargs: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
