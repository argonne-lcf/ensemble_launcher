from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from ensemble_launcher.config import SystemConfig


@dataclass(frozen=True, eq=True)
class NodeResource(ABC):
    """Base class for node resources"""

    @property
    @abstractmethod
    def cpu_count(self) -> int:
        """Total number of CPUs."""
        pass

    @property
    @abstractmethod
    def gpu_count(self) -> int:
        """Total number of GPUs."""
        pass

    @property
    def counts(self) -> dict:
        """Counts of all resources"""
        return {"cpus": self.cpu_count, "gpus": self.gpu_count}

    def is_empty(self) -> bool:
        """Check if resource has no CPUs or GPUs."""
        return self.cpu_count == 0 and self.gpu_count == 0

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(cpus={self.cpu_count}, gpus={self.gpu_count})"
        )

    def __add__(self, other):
        if isinstance(other, NodeResource):
            return self._add_impl(other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, NodeResource):
            return self._sub_impl(other)
        return NotImplemented

    def __radd__(self, other):
        if other == 0:  # Support sum() with start=0
            return self
        return self.__add__(other)

    def __eq__(self, other) -> bool:
        """Check equality based on resource counts."""
        if not isinstance(other, NodeResource):
            return False
        return self.cpu_count == other.cpu_count and self.gpu_count == other.gpu_count

    def __hash__(self) -> int:
        """Hash based on resource counts for use in sets/dicts."""
        return hash((self.cpu_count, self.gpu_count))

    @abstractmethod
    def _add_impl(self, other: "NodeResource") -> "NodeResource":
        """Implementation-specific addition"""
        pass

    @abstractmethod
    def _sub_impl(self, other: "NodeResource") -> "NodeResource":
        """Implementation-specific subtraction"""
        pass

    @abstractmethod
    def __contains__(self, other) -> bool:
        """Check if another resource is contained within this one."""
        pass

    @abstractmethod
    def divide(self, n: int) -> List["NodeResource"]:
        """Divide this resource into n approximately equal parts."""
        pass

    @abstractmethod
    def serialize(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict that includes a ``type`` discriminator."""
        pass

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "NodeResource":
        """Reconstruct a concrete ``NodeResource`` from a dict produced by ``serialize``."""
        type_map: Dict[str, type] = {
            "count": NodeResourceCount,
            "list": NodeResourceList,
        }
        kind = d.get("type")
        if kind not in type_map:
            raise ValueError(f"Unknown NodeResource type '{kind}'")
        return type_map[kind].deserialize(d)


@dataclass(frozen=True, eq=True)
class NodeResourceCount(NodeResource):
    """Count-based node resource representation"""

    ncpus: int = 0
    ngpus: int = 0
    # Future: memory: int = 0

    @property
    def cpu_count(self) -> int:
        return self.ncpus

    @property
    def gpu_count(self) -> int:
        return self.ngpus

    @property
    def cpus(self) -> Tuple[int]:
        return tuple(range(self.ncpus))

    @property
    def gpus(self) -> Tuple[int]:
        return tuple(range(self.ngpus))

    def _add_impl(self, other: NodeResource) -> "NodeResourceCount":
        return NodeResourceCount(
            ncpus=self.ncpus + other.cpu_count, ngpus=self.ngpus + other.gpu_count
        )

    def _sub_impl(self, other: NodeResource) -> "NodeResourceCount":
        return NodeResourceCount(
            ncpus=max(0, self.ncpus - other.cpu_count),
            ngpus=max(0, self.ngpus - other.gpu_count),
        )

    def __contains__(self, other) -> bool:
        """Check if another resource can be satisfied by this count-based resource."""
        if isinstance(other, NodeResource):
            return other.cpu_count <= self.ncpus and other.gpu_count <= self.ngpus
        return False

    def divide(self, n: int) -> List["NodeResourceCount"]:
        """Divide this resource into n approximately equal parts."""
        if n <= 0:
            raise ValueError("Division count must be positive")

        base_cpus = self.ncpus // n
        cpu_remainder = self.ncpus % n

        base_gpus = self.ngpus // n
        gpu_remainder = self.ngpus % n

        result = []
        for i in range(n):
            # First 'remainder' parts get one extra resource
            cpus = base_cpus + (1 if i < cpu_remainder else 0)
            gpus = base_gpus + (1 if i < gpu_remainder else 0)
            result.append(NodeResourceCount(ncpus=cpus, ngpus=gpus))

        return result

    def to_dict(self):
        return {"ncpus": self.ncpus, "ngpus": self.ngpus}

    def serialize(self) -> Dict[str, Any]:
        return {"type": "count", "ncpus": self.ncpus, "ngpus": self.ngpus}

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "NodeResourceCount":
        return cls(ncpus=d["ncpus"], ngpus=d["ngpus"])

    @classmethod
    def from_config(self, info: SystemConfig):
        """creates a node resource list from a dict"""
        return NodeResourceCount(
            ncpus=info.ncpus if len(info.cpus) == 0 else len(info.cpus),
            ngpus=info.ngpus if len(info.gpus) == 0 else len(info.gpus),
        )


@dataclass(frozen=True, eq=True)
class NodeResourceList(NodeResource):
    """List-based (specific IDs) node resource representation"""

    cpus: tuple[int, ...] = field(default_factory=tuple)
    gpus: tuple[int, ...] = field(default_factory=tuple)
    # Future: memory: int = 0

    @property
    def cpu_count(self) -> int:
        return len(self.cpus)

    @property
    def gpu_count(self) -> int:
        return len(self.gpus)

    def _add_impl(self, other: NodeResource) -> "NodeResourceList":
        if isinstance(other, NodeResourceList):
            return NodeResourceList(
                cpus=tuple((Counter(self.cpus) + Counter(other.cpus)).elements()),
                gpus=tuple((Counter(self.gpus) + Counter(other.gpus)).elements()),
            )
        elif isinstance(other, NodeResourceCount):
            # Convert count to consecutive IDs and add
            next_cpu_id = max(self.cpus) + 1 if self.cpus else 0
            next_gpu_id = max(self.gpus) + 1 if self.gpus else 0
            new_cpus = tuple(range(next_cpu_id, next_cpu_id + other.ncpus))
            new_gpus = tuple(range(next_gpu_id, next_gpu_id + other.ngpus))
            return NodeResourceList(
                cpus=self.cpus + new_cpus, gpus=self.gpus + new_gpus
            )
        return NotImplemented

    def _sub_impl(self, other: NodeResource) -> "NodeResourceList":
        if isinstance(other, NodeResourceList):
            remaining_cpus = tuple(
                (Counter(self.cpus) - Counter(other.cpus)).elements()
            )
            remaining_gpus = tuple(
                (Counter(self.gpus) - Counter(other.gpus)).elements()
            )
            return NodeResourceList(cpus=remaining_cpus, gpus=remaining_gpus)
        elif isinstance(other, NodeResourceCount):
            # Remove first N CPUs and GPUs
            remaining_cpus = self.cpus[other.ncpus :]
            remaining_gpus = self.gpus[other.ngpus :]
            return NodeResourceList(cpus=remaining_cpus, gpus=remaining_gpus)
        return NotImplemented

    def __contains__(self, other) -> bool:
        """Check if another resource is contained within this list-based resource."""
        if isinstance(other, NodeResourceList):
            # Check if all CPUs and GPUs in 'other' are available in 'self'
            return (Counter(other.cpus) <= Counter(self.cpus)) and (
                Counter(other.gpus) <= Counter(self.gpus)
            )
        elif isinstance(other, NodeResourceCount):
            # Check if we have enough resources
            return other.ncpus <= self.cpu_count and other.ngpus <= self.gpu_count
        return False

    def divide(self, n: int) -> List["NodeResourceList"]:
        """Divide this resource into n approximately equal parts."""
        if n <= 0:
            raise ValueError("Division count must be positive")

        # Divide CPUs
        cpu_list = list(self.cpus)
        base_cpus_per_part = len(cpu_list) // n
        cpu_remainder = len(cpu_list) % n

        cpu_parts = []
        start_idx = 0
        for i in range(n):
            count = base_cpus_per_part + (1 if i < cpu_remainder else 0)
            cpu_parts.append(tuple(cpu_list[start_idx : start_idx + count]))
            start_idx += count

        # Divide GPUs
        gpu_list = list(self.gpus)
        base_gpus_per_part = len(gpu_list) // n
        gpu_remainder = len(gpu_list) % n

        gpu_parts = []
        start_idx = 0
        for i in range(n):
            count = base_gpus_per_part + (1 if i < gpu_remainder else 0)
            gpu_parts.append(tuple(gpu_list[start_idx : start_idx + count]))
            start_idx += count

        return [
            NodeResourceList(cpus=cpu_parts[i], gpus=gpu_parts[i]) for i in range(n)
        ]

    def __eq__(self, other) -> bool:
        """Check equality based on CPU and GPU lists."""
        if not isinstance(other, NodeResourceList):
            # Fall back to parent class equality for cross-type comparison
            return super().__eq__(other)
        return Counter(self.cpus) == Counter(other.cpus) and Counter(
            self.gpus
        ) == Counter(other.gpus)

    def __hash__(self) -> int:
        """Hash based on sorted CPU and GPU tuples for use in sets/dicts."""
        return hash((tuple(sorted(self.cpus)), tuple(sorted(self.gpus))))

    @classmethod
    def from_config(self, info: SystemConfig):
        """creates a node resource list from a dict"""
        return NodeResourceList(
            cpus=tuple(range(info.ncpus)) if len(info.cpus) == 0 else tuple(info.cpus),
            gpus=tuple(range(info.ngpus)) if len(info.gpus) == 0 else tuple(info.gpus),
        )

    def to_dict(self):
        return {"cpus": self.cpus, "gpus": self.gpus}

    def serialize(self) -> Dict[str, Any]:
        return {"type": "list", "cpus": list(self.cpus), "gpus": list(self.gpus)}

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "NodeResourceList":
        return cls(cpus=tuple(d["cpus"]), gpus=tuple(d["gpus"]))


@dataclass(eq=True)
class JobResource:
    """
    Represents the computational resources required for a job.

    This immutable dataclass encapsulates a collection of node resources
    that define the computational requirements for executing a job in a
    distributed computing environment.

    Attributes:
        resources (List[NodeResource]): A list of NodeResource objects defining
            the computational requirements for the job.
        nodes (List): A list of node identifiers where the job resources
            will be allocated. Defaults to an empty list.
    """

    resources: List[NodeResource]
    nodes: List = field(default_factory=list)

    def __post_init__(self):
        if self.nodes:
            assert len(self.nodes) == len(self.resources), (
                "number of nodes != number of job resources"
            )

        # Validate that resources is not empty
        if not self.resources:
            raise ValueError("JobResource must have at least one resource")

        # Validate that all resources are NodeResource instances
        for i, resource in enumerate(self.resources):
            if not isinstance(resource, NodeResource):
                raise TypeError(
                    f"Resource at index {i} must be a NodeResource instance"
                )

    def __repr__(self) -> str:
        total_cpus = sum(r.cpu_count for r in self.resources)
        total_gpus = sum(r.gpu_count for r in self.resources)
        nodes_info = f", nodes={self.nodes}" if self.nodes else ""
        return f"JobResource({len(self.resources)} nodes, total_cpus={total_cpus}, total_gpus={total_gpus}{nodes_info})"

    def __eq__(self, other) -> bool:
        """Check equality based on resources and nodes."""
        if not isinstance(other, JobResource):
            return False
        return tuple(self.resources) == tuple(other.resources) and tuple(
            self.nodes
        ) == tuple(other.nodes)

    def __hash__(self) -> int:
        """Hash based on resources and nodes for use in sets/dicts."""
        return hash((tuple(self.resources), tuple(self.nodes)))

    def to_dict(self) -> Dict[str, NodeResource]:
        """Convert JobResource to a dictionary mapping node identifiers to resources."""
        if not self.nodes:
            raise ValueError("Cannot convert to dict without node identifiers")
        return {node: resource for node, resource in zip(self.nodes, self.resources)}

    @classmethod
    def from_dict(cls, resource_dict: Dict[str, NodeResource]) -> "JobResource":
        """Create a JobResource from a dictionary mapping node identifiers to resources."""
        nodes = list(resource_dict.keys())
        resources = list(resource_dict.values())
        return cls(resources=resources, nodes=nodes)

    def __contains__(self, other) -> bool:
        """Check if another JobResource can be satisfied by this JobResource.

        Args:
            other: Another JobResource to check

        Returns:
            True if this JobResource can satisfy the other's requirements
        """
        if not isinstance(other, JobResource):
            return False

        # If other requires more nodes than we have, it can't be contained
        if len(other.resources) > len(self.resources):
            return False

        # Try to match each required resource with an available resource
        available = list(self.resources)
        for required in other.resources:
            # Find a resource that can satisfy this requirement
            found = False
            for i, avail in enumerate(available):
                if required in avail:
                    available.pop(i)
                    found = True
                    break
            if not found:
                return False

        return True

    def serialize(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict including type-discriminated resources."""
        return {
            "resources": [r.serialize() for r in self.resources],
            "nodes": list(self.nodes),
        }

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "JobResource":
        """Reconstruct a ``JobResource`` from a dict produced by ``serialize``."""
        resources = [NodeResource.deserialize(r) for r in d["resources"]]
        return cls(resources=resources, nodes=d.get("nodes", []))
