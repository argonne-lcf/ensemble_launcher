from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ensemble_launcher.config import SystemConfig

# A GPU device id. Polaris and most CUDA/ROCm systems use ints (0, 1, 2, 3);
# aurora uses numeric strings ("0" .. "11"), which is why ordering goes
# through _amount_sort_key rather than a plain sorted(). Mirrors
# SystemConfig.gpus (List[Union[str, int]]) and Task.gpu_affinity.
GpuId = Union[int, str]

# Amounts are rounded to this many decimals everywhere so that repeated
# allocate/deallocate cycles of fractional GPUs (e.g. 0.1 ten times) don't
# drift away from a clean 1.0/0.0 due to floating point error.
_AMOUNT_DECIMALS = 6
_EPS = 1e-6


def _round_amount(x: float) -> float:
    return round(float(x), _AMOUNT_DECIMALS)


def _amount_sort_key(gpu_id: GpuId):
    """
    Canonical ordering for GPU ids. ``SystemConfig.gpus`` mixes ``int`` ids
    (e.g. polaris: ``[0, 1, 2, 3]``) and numeric ``str`` ids (e.g. aurora:
    ``["0", ..., "11"]``), and a plain ``sorted()`` raises on a mixed list.
    Sort numerically whenever the id looks numeric (covers both cases) and
    fall back to lexicographic string order only for genuinely non-numeric
    ids, so aurora's string ids don't end up in "0, 1, 10, 11, 2, ..." order.
    """
    try:
        return (0, int(gpu_id))
    except (TypeError, ValueError):
        return (1, str(gpu_id))


def _normalize_gpu_amounts(gpus: Iterable[GpuId]) -> Tuple[Tuple[GpuId, float], ...]:
    """
    Normalize the ``gpus=`` constructor argument into a canonical, sorted
    tuple of ``(id, amount)`` pairs.

    The only accepted form is a flat list/tuple of ids -- "these devices
    exist" -- and every one of them starts fully free, at 1.0. A GPU is
    never worth more than one whole GPU, so there is nothing else to say at
    construction time. Sub-1.0 amounts arise only from subtracting an
    allocation, and reach an instance only through
    :meth:`NodeResourceList._from_amounts`.

    A repeated id is an error. It used to mean "N copies" (the retired
    oversubscription trick); share a device between tasks with
    ``Task.ngpus_per_process`` instead.
    """
    if isinstance(gpus, dict):
        raise TypeError(
            "gpus= takes a flat list of device ids, not a mapping. Every GPU "
            "starts fully available; to share one between tasks, use "
            "Task.ngpus_per_process (e.g. 0.25) rather than a custom amount."
        )

    amounts: Dict[GpuId, float] = {}
    for gid in gpus:
        if isinstance(gid, (tuple, list)):
            raise TypeError(
                f"gpus= takes a flat list of device ids, not (id, amount) pairs "
                f"(got {gid!r}). Every GPU starts fully available; to share one "
                f"between tasks, use Task.ngpus_per_process."
            )
        if gid in amounts:
            raise ValueError(
                f"duplicate GPU id {gid!r} in {gpus!r}. A device can only be "
                f"listed once; to let several tasks share it, use "
                f"Task.ngpus_per_process (e.g. 0.25)."
            )
        amounts[gid] = 1.0

    return tuple(sorted(amounts.items(), key=lambda kv: _amount_sort_key(kv[0])))


def _plan_gpu_take(
    amount_pairs: Tuple[Tuple[GpuId, float], ...], demand: float
) -> Optional[Dict[GpuId, float]]:
    """
    Decide which devices satisfy a ``NodeResourceCount`` demand for
    ``demand`` GPUs, or return ``None`` if it cannot be satisfied.

    **A fraction of a GPU never spans two physical devices.** 0.3 of a GPU
    means 0.3 of *one* device; stitching it together from 0.1 of GPU 0 and
    0.2 of GPU 1 would hand the task ``ZE_AFFINITY_MASK=0,1`` -- two devices
    it can only partly use -- which is not a thing a caller can ask for.
    So a sub-1.0 remainder must land on a single device with room for all
    of it, and a whole-GPU request takes fully-free devices.

    The remainder uses **best fit**: the smallest device that still has room.
    That fills partly-used devices before breaking into an untouched one, so
    whole-GPU requests keep finding whole GPUs. With four 0.3 requests on a
    4-GPU node the first three share GPU 0 (leaving 0.1) and the fourth
    moves to GPU 1, rather than the fourth straddling both.

    Why a shared helper rather than the check inlined in each caller:
    ``__contains__`` and ``_sub_impl`` have to agree exactly. When they did
    not, ``__contains__`` approved a request on total free capacity that
    ``_sub_impl`` then satisfied by straddling two devices. Routing both
    through one planner makes the feasibility test *be* the subtraction:
    ``__contains__`` asks whether a plan exists, ``_sub_impl`` applies it.
    Note a per-device test alone ("does any one GPU have room") cannot
    replace this -- it answers False for ``ngpus=2`` on an idle 4-GPU node,
    since no single device holds 2.0. Counting whole devices and placing a
    remainder are two different questions, and this function is where they
    are answered together.

    ``_add_impl`` deliberately does not use this: ``deallocate`` returns the
    granted resource, which ``allocate`` built by subtraction, so it is
    always a ``NodeResourceList`` carrying explicit ids.

    Returns ``{gpu_id: amount_to_take}``, which the caller subtracts.
    """
    demand = _round_amount(demand)
    if demand <= _EPS:
        return {}

    n_whole = int(demand + _EPS)
    remainder = _round_amount(demand - n_whole)

    take: Dict[GpuId, float] = {}
    free = [(gid, amt) for gid, amt in amount_pairs if amt > _EPS]

    # Whole devices first, in canonical order, so integer requests are
    # unchanged: N GPUs means N untouched devices.
    if n_whole:
        whole = [gid for gid, amt in free if amt >= 1.0 - _EPS]
        if len(whole) < n_whole:
            return None
        for gid in whole[:n_whole]:
            take[gid] = 1.0

    if remainder > _EPS:
        candidates = [
            (amt, gid) for gid, amt in free
            if gid not in take and amt + _EPS >= remainder
        ]
        if not candidates:
            return None
        # Best fit: tightest device that fits, ties broken canonically.
        _, gid = min(candidates, key=lambda c: (c[0], _amount_sort_key(c[1])))
        take[gid] = remainder

    return take


def _expand_gpu_ids(amount_pairs: Tuple[Tuple[GpuId, float], ...]) -> Tuple[GpuId, ...]:
    """
    The ids held by an (id, amount) mapping, for callers that only care
    "which devices may I use" -- env var construction, ``set()`` comparisons
    in the MPI executors, ``len()`` for a device count.

    Each id appears exactly once whatever its amount: the amount is how much
    of that one device is free, not a number of devices. A partly-allocated
    ``{0: 0.25}`` is still the single GPU 0, so it yields ``(0,)``.
    """
    return tuple(gid for gid, _ in amount_pairs)


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
    def gpu_count(self) -> Union[int, float]:
        """Total number of GPUs (may be fractional)."""
        pass

    @property
    def counts(self) -> dict:
        """Counts of all resources"""
        return {"cpus": self.cpu_count, "gpus": self.gpu_count}

    def is_empty(self) -> bool:
        """Check if resource has no CPUs or GPUs."""
        return self.cpu_count == 0 and _round_amount(self.gpu_count) <= _EPS

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
        return self.cpu_count == other.cpu_count and _round_amount(
            self.gpu_count
        ) == _round_amount(other.gpu_count)

    def __hash__(self) -> int:
        """Hash based on resource counts for use in sets/dicts."""
        return hash((self.cpu_count, _round_amount(self.gpu_count)))

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
    ngpus: Union[int, float] = 0
    # Future: memory: int = 0

    @property
    def cpu_count(self) -> int:
        return self.ncpus

    @property
    def gpu_count(self) -> Union[int, float]:
        return self.ngpus

    @property
    def cpus(self) -> Tuple[int]:
        return tuple(range(self.ncpus))

    @property
    def gpus(self) -> Tuple[int, ...]:
        n = int(self.ngpus)
        if n == 0 and self.ngpus > 0:
            n = 1
        return tuple(range(n))

    def _add_impl(self, other: NodeResource) -> "NodeResourceCount":
        return NodeResourceCount(
            ncpus=self.ncpus + other.cpu_count,
            ngpus=_round_amount(self.ngpus + other.gpu_count),
        )

    def _sub_impl(self, other: NodeResource) -> "NodeResourceCount":
        return NodeResourceCount(
            ncpus=max(0, self.ncpus - other.cpu_count),
            ngpus=max(0.0, _round_amount(self.ngpus - other.gpu_count)),
        )

    def __contains__(self, other) -> bool:
        """Check if another resource can be satisfied by this count-based resource."""
        if isinstance(other, NodeResource):
            return other.cpu_count <= self.ncpus and other.gpu_count <= self.ngpus + _EPS
        return False

    def divide(self, n: int) -> List["NodeResourceCount"]:
        """Divide this resource into n approximately equal parts."""
        if n <= 0:
            raise ValueError("Division count must be positive")

        base_cpus = self.ncpus // n
        cpu_remainder = self.ncpus % n

        if float(self.ngpus).is_integer():
            ngpus_int = int(self.ngpus)
            base_gpus = ngpus_int // n
            gpu_remainder = ngpus_int % n
            gpus_per_part = [
                base_gpus + (1 if i < gpu_remainder else 0) for i in range(n)
            ]
        else:
            share = _round_amount(self.ngpus / n)
            gpus_per_part = [share for _ in range(n)]

        result = []
        for i in range(n):
            # First 'remainder' parts get one extra resource
            cpus = base_cpus + (1 if i < cpu_remainder else 0)
            result.append(NodeResourceCount(ncpus=cpus, ngpus=gpus_per_part[i]))

        return result

    def to_dict(self):
        return {"ncpus": self.ncpus, "ngpus": self.ngpus}

    def serialize(self) -> Dict[str, Any]:
        return {"type": "count", "ncpus": self.ncpus, "ngpus": self.ngpus}

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "NodeResourceCount":
        return cls(ncpus=d["ncpus"], ngpus=d["ngpus"])

    @classmethod
    def from_config(cls, info: SystemConfig):
        """creates a node resource list from a dict"""
        return NodeResourceCount(
            ncpus=info.ncpus if len(info.cpus) == 0 else len(info.cpus),
            ngpus=info.ngpus if len(info.gpus) == 0 else len(info.gpus),
        )


@dataclass(frozen=True, eq=True)
class NodeResourceList(NodeResource):
    """List-based (specific IDs) node resource representation.

    ``gpus`` is a flat list/tuple of device ids, and every listed device
    starts fully available (amount 1.0). Internally each id carries how much
    of it is still free, so that a fraction of a GPU can be handed out:
    subtracting an allocation leaves e.g. ``{0: 0.75}``. Those sub-1.0
    amounts are produced by the arithmetic operators, never by a caller --
    the one exception is :meth:`request`, which builds a *demand* for a
    fraction of specific ids.

    ``gpus`` itself always reads back as the flat tuple of ids (each listed
    once, whatever its amount), which is what env-var construction and the
    MPI executors want. The amounts live in ``gpu_amounts``.
    """

    cpus: tuple[int, ...] = field(default_factory=tuple)
    gpus: tuple[GpuId, ...] = field(default_factory=tuple)
    # Future: memory: int = 0

    def __post_init__(self):
        object.__setattr__(self, "cpus", tuple(self.cpus))
        amount_pairs = _normalize_gpu_amounts(self.gpus)
        object.__setattr__(self, "_gpu_amounts", amount_pairs)
        object.__setattr__(self, "gpus", _expand_gpu_ids(amount_pairs))

    @classmethod
    def _from_amounts(
        cls, cpus: Iterable[int], amounts: Dict[GpuId, float]
    ) -> "NodeResourceList":
        """
        Build directly from ``{id: amount}``, bypassing the ids-only
        constructor. This is the *only* way a non-1.0 amount enters an
        instance, and it is internal: the arithmetic operators, ``divide``
        and ``deserialize`` use it to carry partial availability.

        Devices drained to nothing are dropped. An amount above 1.0 is
        impossible by construction -- pools start at 1.0 and the only
        addition in the codebase is ``deallocate`` returning what was
        subtracted -- so it means a resource was deallocated twice. Raise
        rather than let the phantom capacity oversubscribe a real GPU.
        """
        cleaned: Dict[GpuId, float] = {}
        for gid, amt in amounts.items():
            rounded = _round_amount(amt)
            if rounded > 1.0 + _EPS:
                raise ValueError(
                    f"GPU {gid!r} has amount {rounded} > 1.0, which cannot happen "
                    f"for a real device. This usually means the same allocation "
                    f"was deallocated twice."
                )
            if rounded > _EPS:
                cleaned[gid] = rounded

        obj = cls(cpus=cpus, gpus=())
        pairs = tuple(sorted(cleaned.items(), key=lambda kv: _amount_sort_key(kv[0])))
        object.__setattr__(obj, "_gpu_amounts", pairs)
        object.__setattr__(obj, "gpus", _expand_gpu_ids(pairs))
        return obj

    @classmethod
    def request(
        cls,
        cpus: Iterable[int] = (),
        gpus: Iterable[GpuId] = (),
        gpu_fraction: float = 1.0,
    ) -> "NodeResourceList":
        """
        Build a *demand* for ``gpu_fraction`` of each of the given device
        ids -- what a task with an explicit ``gpu_affinity`` and a fractional
        ``ngpus_per_process`` is asking for.

        Unlike the constructor (which describes a pool of whole devices),
        this describes how much of each named device is wanted.
        """
        if not 0.0 < gpu_fraction <= 1.0:
            raise ValueError(
                f"gpu_fraction must be in (0.0, 1.0], got {gpu_fraction}. A task "
                f"cannot request more than a whole GPU from a single device."
            )
        # Reuse the constructor's id validation (duplicates, wrong forms).
        ids = _normalize_gpu_amounts(gpus)
        return cls._from_amounts(
            cpus, {gid: gpu_fraction for gid, _ in ids}
        )

    def with_cpus(self, cpus: Iterable[int]) -> "NodeResourceList":
        """This node's GPUs (amounts intact) with a different set of CPUs."""
        return self._from_amounts(cpus, dict(self._gpu_amounts))

    @property
    def gpu_amounts(self) -> Dict[GpuId, float]:
        """Mapping of gpu id -> available fraction (1.0 == a whole GPU)."""
        return dict(self._gpu_amounts)

    @property
    def cpu_count(self) -> int:
        return len(self.cpus)

    @property
    def gpu_count(self) -> Union[int, float]:
        return _round_amount(sum(amt for _, amt in self._gpu_amounts))

    def _add_impl(self, other: NodeResource) -> "NodeResourceList":
        if isinstance(other, NodeResourceList):
            merged_cpus = tuple((Counter(self.cpus) + Counter(other.cpus)).elements())
            merged_gpus: Dict[GpuId, float] = dict(self._gpu_amounts)
            for gid, amt in other._gpu_amounts:
                merged_gpus[gid] = _round_amount(merged_gpus.get(gid, 0.0) + amt)
            return NodeResourceList._from_amounts(merged_cpus, merged_gpus)
        elif isinstance(other, NodeResourceCount):
            # Convert count to consecutive IDs and add
            next_cpu_id = max(self.cpus) + 1 if self.cpus else 0
            new_cpus = tuple(range(next_cpu_id, next_cpu_id + other.ncpus))

            merged_gpus: Dict[GpuId, float] = dict(self._gpu_amounts)
            remaining = float(other.ngpus)
            if remaining > _EPS:
                int_ids = [gid for gid in merged_gpus if isinstance(gid, int)]
                next_gpu_id = (max(int_ids) + 1) if int_ids else 0
                if float(remaining).is_integer():
                    for _ in range(int(remaining)):
                        merged_gpus[next_gpu_id] = merged_gpus.get(next_gpu_id, 0.0) + 1.0
                        next_gpu_id += 1
                else:
                    merged_gpus[next_gpu_id] = merged_gpus.get(next_gpu_id, 0.0) + remaining
            return NodeResourceList._from_amounts(self.cpus + new_cpus, merged_gpus)
        return NotImplemented

    def _sub_impl(self, other: NodeResource) -> "NodeResourceList":
        if isinstance(other, NodeResourceList):
            remaining_cpus = tuple(
                (Counter(self.cpus) - Counter(other.cpus)).elements()
            )
            other_map = dict(other._gpu_amounts)
            remaining_gpus: Dict[GpuId, float] = {}
            for gid, amt in self._gpu_amounts:
                new_amt = _round_amount(amt - other_map.get(gid, 0.0))
                if new_amt > _EPS:
                    remaining_gpus[gid] = new_amt
            return NodeResourceList._from_amounts(remaining_cpus, remaining_gpus)
        elif isinstance(other, NodeResourceCount):
            # Remove first N CPUs, and take other.ngpus via the shared
            # planner so a fraction never straddles two physical devices
            # (see _plan_gpu_take). __contains__ uses the same planner, so
            # the feasibility check and the subtraction cannot disagree.
            remaining_cpus = self.cpus[other.ncpus :]
            plan = _plan_gpu_take(self._gpu_amounts, other.ngpus)
            if plan is None:
                raise ValueError(
                    f"cannot take {other.ngpus} GPUs from {self.gpu_amounts!r}: "
                    f"no single device has room for the fractional part. "
                    f"A fraction of a GPU cannot span two devices."
                )
            remaining_gpus: Dict[GpuId, float] = {}
            for gid, amt in self._gpu_amounts:
                left = _round_amount(amt - plan.get(gid, 0.0))
                if left > _EPS:
                    remaining_gpus[gid] = left
            return NodeResourceList._from_amounts(remaining_cpus, remaining_gpus)
        return NotImplemented

    def __contains__(self, other) -> bool:
        """Check if another resource is contained within this list-based resource."""
        if isinstance(other, NodeResourceList):
            if not (Counter(other.cpus) <= Counter(self.cpus)):
                return False
            self_map = dict(self._gpu_amounts)
            for gid, amt in other._gpu_amounts:
                if self_map.get(gid, 0.0) + _EPS < amt:
                    return False
            return True
        elif isinstance(other, NodeResourceCount):
            # Not a bare total-capacity check: a node with 0.1 free on each
            # of three devices has 0.3 GPUs free but cannot satisfy a single
            # 0.3 request, because that fraction has to live on one device.
            # Defer to the same planner _sub_impl uses.
            if other.ncpus > self.cpu_count:
                return False
            return _plan_gpu_take(self._gpu_amounts, other.ngpus) is not None
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

        # Divide GPUs a whole device at a time, carrying each device's amount
        # with it -- a child gets all of a device's capacity or none of it.
        gpu_list = list(self._gpu_amounts)
        base_gpus_per_part = len(gpu_list) // n
        gpu_remainder = len(gpu_list) % n

        gpu_parts = []
        start_idx = 0
        for i in range(n):
            count = base_gpus_per_part + (1 if i < gpu_remainder else 0)
            gpu_parts.append(tuple(gpu_list[start_idx : start_idx + count]))
            start_idx += count

        return [
            NodeResourceList._from_amounts(cpu_parts[i], dict(gpu_parts[i]))
            for i in range(n)
        ]

    def __eq__(self, other) -> bool:
        """Check equality based on CPU multiset and GPU id->amount mapping."""
        if not isinstance(other, NodeResourceList):
            # Fall back to parent class equality for cross-type comparison
            return super().__eq__(other)
        if Counter(self.cpus) != Counter(other.cpus):
            return False
        self_amounts = dict(self._gpu_amounts)
        other_amounts = dict(other._gpu_amounts)
        if set(self_amounts) != set(other_amounts):
            return False
        return all(
            abs(self_amounts[gid] - other_amounts[gid]) < _EPS for gid in self_amounts
        )

    def __hash__(self) -> int:
        """Hash based on sorted CPU tuple and canonical GPU amounts."""
        return hash((tuple(sorted(self.cpus)), self._gpu_amounts))

    @classmethod
    def from_config(cls, info: SystemConfig):
        """creates a node resource list from a dict"""
        return NodeResourceList(
            cpus=tuple(range(info.ncpus)) if len(info.cpus) == 0 else tuple(info.cpus),
            gpus=tuple(range(info.ngpus)) if len(info.gpus) == 0 else tuple(info.gpus),
        )

    def to_dict(self):
        # Bare ids -- "which devices". Partial amounts are not represented
        # here; serialize()/deserialize() is the lossless pair used for
        # checkpoints of a partially-allocated node.
        return {"cpus": self.cpus, "gpus": self.gpus}

    def serialize(self) -> Dict[str, Any]:
        return {
            "type": "list",
            "cpus": list(self.cpus),
            "gpus": [[gid, amt] for gid, amt in self._gpu_amounts],
        }

    @classmethod
    def deserialize(cls, d: Dict[str, Any]) -> "NodeResourceList":
        raw_gpus = d.get("gpus", [])
        cpus = tuple(d["cpus"])
        # Backward compatible: old checkpoints stored a flat id list (whole
        # GPUs); the current format stores [[id, amount], ...] pairs so a
        # partially-allocated node round-trips.
        if raw_gpus and isinstance(raw_gpus[0], (list, tuple)):
            return cls._from_amounts(
                cpus, {gid: float(amt) for gid, amt in raw_gpus}
            )
        return cls(cpus=cpus, gpus=list(raw_gpus))


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
