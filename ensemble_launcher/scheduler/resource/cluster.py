from abc import ABC, abstractmethod
from bisect import bisect_left, insort
from logging import Logger
from typing import List, Optional, Tuple
import asyncio

from .node import NodeResource, JobResource, NodeResourceCount, NodeResourceList, _round_amount

# GPU amounts are kept as integers: _UNITS per whole GPU (matches the 6-decimal rounding
# in node.py), so fractional allocations never drift.
_UNITS = 1_000_000


def _units(amount) -> int:
    return int(round(float(amount) * _UNITS))


def _gpu_split(units: int) -> Tuple[int, int]:
    """A node's GPU request as (whole GPUs, fraction of one GPU in _UNITS): one of the two
    is 0. A request is whole GPUs or a fraction of a single GPU, never both."""
    whole, rem = divmod(units, _UNITS)
    if whole and rem:
        raise ValueError(
            f"a node can ask for whole GPUs or a fraction of one GPU, not both: got "
            f"{_round_amount(units / _UNITS)} (e.g. from ppn * ngpus_per_process)"
        )
    return whole, rem


def _lowest(mask: int, n: int) -> int:
    """The ``n`` lowest set bits of ``mask`` (fewer if it has fewer)."""
    take = 0
    for _ in range(n):
        if not mask:
            break
        low = mask & -mask
        take |= low
        mask ^= low
    return take


def _bit_indexes(mask: int) -> List[int]:
    """Indexes of the set bits of ``mask``, lowest first."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


class _Node:
    """One node's free resources.

    cpu_mask    bit c set = CPU ``cpu_ids[c]`` is free (a plain Python int)
    gpu_free    free amount of each GPU, in _UNITS (``gpu_ids[j]`` is GPU j)
    full_mask   bit j set = GPU j is fully free
    partial     sorted [(free amount, j)] of partly used GPUs, so the best fit for a
                fraction (least room that still fits) is one bisect away
    """

    __slots__ = ("name", "cpu_ids", "cpu_bit", "gpu_ids", "gpu_col",
                 "cpu_mask", "ncpu_free", "gpu_free", "full_mask", "partial")

    def __init__(self, name: str, res: NodeResourceList):
        self.name = name
        self.cpu_ids = tuple(res.cpus)
        self.cpu_bit = {c: 1 << b for b, c in enumerate(self.cpu_ids)}
        self.gpu_ids = tuple(res.gpus)
        self.gpu_col = {g: j for j, g in enumerate(self.gpu_ids)}
        self.cpu_mask = (1 << len(self.cpu_ids)) - 1
        self.ncpu_free = len(self.cpu_ids)
        self.gpu_free = [_units(a) for _, a in res._gpu_amounts]
        self.full_mask = 0
        self.partial = []
        for j, u in enumerate(self.gpu_free):
            if u == _UNITS:
                self.full_mask |= 1 << j
            elif u > 0:
                self.partial.append((u, j))
        self.partial.sort()

    def _set_gpu(self, j: int, new: int) -> None:
        """Change GPU j's free amount, keeping full_mask and partial in step."""
        old = self.gpu_free[j]
        if old == _UNITS:
            self.full_mask &= ~(1 << j)
        elif old > 0:
            self.partial.pop(bisect_left(self.partial, (old, j)))
        self.gpu_free[j] = new
        if new == _UNITS:
            self.full_mask |= 1 << j
        elif new > 0:
            insort(self.partial, (new, j))

    def plan(self, req: NodeResource):
        """What ``req`` would take: (cpu bits, {gpu column: units}), or None if it doesn't fit."""
        if isinstance(req, NodeResourceList):
            try:
                want = 0
                for c in req.cpus:
                    want |= self.cpu_bit[c]
                take = {self.gpu_col[g]: _units(a) for g, a in req._gpu_amounts}
            except KeyError:
                return None
            if self.cpu_mask & want != want:
                return None
            if any(self.gpu_free[j] < u for j, u in take.items()):
                return None
            return want, take

        # Count request: the lowest free CPUs, then whole GPUs in id order, or the
        # fraction on the partly used GPU with the least room that still fits it, else
        # on the lowest fully free GPU.
        if self.ncpu_free < req.ncpus:
            return None
        take = {}
        units = _units(req.ngpus)
        if units:
            whole, rem = _gpu_split(units)
            if whole:
                gpus = _lowest(self.full_mask, whole)
                if gpus.bit_count() < whole:
                    return None
                take = {j: _UNITS for j in _bit_indexes(gpus)}
            else:
                k = bisect_left(self.partial, (rem, -1))
                if k < len(self.partial):
                    take = {self.partial[k][1]: rem}
                elif self.full_mask:
                    take = {(self.full_mask & -self.full_mask).bit_length() - 1: rem}
                else:
                    return None
        return _lowest(self.cpu_mask, req.ncpus), take

    def take(self, cpus: int, gpus: dict) -> NodeResourceList:
        """Take what ``plan`` returned; returns it as a NodeResourceList."""
        self.cpu_mask ^= cpus
        self.ncpu_free -= cpus.bit_count()
        for j, u in gpus.items():
            self._set_gpu(j, self.gpu_free[j] - u)
        cpu_ids = self.cpu_ids
        pairs = tuple(
            (self.gpu_ids[j], _round_amount(u / _UNITS)) for j, u in sorted(gpus.items())
        )
        return NodeResourceList._trusted(tuple(cpu_ids[b] for b in _bit_indexes(cpus)), pairs)

    def freed(self, res: NodeResource):
        """What deallocating ``res`` gives back: (cpu bits, {gpu column: units}).
        Raises on unknown ids or a double free; changes nothing."""
        if not isinstance(res, NodeResourceList):
            raise ValueError(
                f"cannot deallocate {res} on {self.name}: deallocate takes what allocate "
                f"returned, which names CPU and GPU ids"
            )
        try:
            cpus = 0
            for c in res.cpus:
                cpus |= self.cpu_bit[c]
            gpus = {self.gpu_col[g]: _units(a) for g, a in res._gpu_amounts}
        except KeyError:
            raise ValueError(f"{res} names ids that node {self.name} does not have")
        if self.cpu_mask & cpus:
            raise ValueError(
                f"CPUs on {self.name} are already free: the same allocation was "
                f"deallocated twice."
            )
        if any(self.gpu_free[j] + u > _UNITS for j, u in gpus.items()):
            raise ValueError(
                f"A GPU on {self.name} would have more than 1.0 free: the same allocation "
                f"was deallocated twice."
            )
        return cpus, gpus

    def give(self, cpus: int, gpus: dict) -> None:
        """Give back what ``freed`` returned."""
        self.cpu_mask |= cpus
        self.ncpu_free += cpus.bit_count()
        for j, u in gpus.items():
            self._set_gpu(j, self.gpu_free[j] + u)

    def resource(self) -> NodeResourceList:
        """What is free on this node."""
        pairs = tuple(
            (g, _round_amount(u / _UNITS)) for g, u in zip(self.gpu_ids, self.gpu_free) if u > 0
        )
        cpus = tuple(self.cpu_ids[b] for b in _bit_indexes(self.cpu_mask))
        return NodeResourceList._trusted(cpus, pairs)


def _with_ids(node: NodeResource) -> NodeResourceList:
    """A node as ids. A NodeResourceCount (pin_resources=False) gets CPUs 0..ncpus-1 and
    GPUs 0..: one whole GPU per whole unit of ngpus, plus one partly free GPU for any
    fraction (``divide`` can leave e.g. 1.333)."""
    if isinstance(node, NodeResourceList):
        return node
    whole, rem = divmod(_units(node.ngpus), _UNITS)
    amounts = {g: 1.0 for g in range(whole)}
    if rem:
        amounts[whole] = rem / _UNITS
    return NodeResourceList._from_amounts(range(node.ncpus), amounts)


class ClusterResource(ABC):
    """
    Abstract base class for managing cluster resources and job allocation.

    Each node's free resources are a ``_Node``: a bitmask of free CPUs, the free amount
    of each GPU, a bitmask of fully free GPUs and a sorted list of partly used ones.
    Nodes are kept in a list and searched in order (first fit), which is meant for
    clusters of up to ~128 nodes. Count nodes (NodeResourceCount, pin_resources=False)
    are given ids (``_with_ids``), so allocations always carry ids.

    Note:
        This is an abstract base class and cannot be instantiated directly. Concrete
        implementations must provide the allocate() and deallocate() methods.
    """

    def __init__(self, logger: Logger, nodes: JobResource):
        self.logger = logger
        self.update_nodes(nodes)

    def update_nodes(self, nodes: JobResource):
        """Replace the cluster's nodes; everything in ``nodes`` is free."""
        self.logger.info(f"Updating cluster nodes to {len(nodes.nodes)} nodes")
        self._nodes = [_Node(name, _with_ids(r)) for name, r in zip(nodes.nodes, nodes.resources)]
        self._row = {node.name: node for node in self._nodes}
        self._free_cpus = sum(node.ncpu_free for node in self._nodes)
        self._free_gpu_units = sum(sum(node.gpu_free) for node in self._nodes)

    @property
    def free_cpus(self) -> int:
        return self._free_cpus

    @property
    def free_gpus(self):
        return _round_amount(self._free_gpu_units / _UNITS)

    @property
    def nodes(self) -> Optional[JobResource]:
        return JobResource(
            resources=[node.resource() for node in self._nodes],
            nodes=[node.name for node in self._nodes],
        )

    @abstractmethod
    def allocate(self, job_resource: JobResource):
        pass

    @abstractmethod
    def deallocate(self, job_resource: JobResource):
        pass

    def _place(self, job_resource: JobResource):
        """Where the job would go: [(node, cpu bits, {gpu column: units})] per node, or
        None if it doesn't fit. Explicit nodes are used as given; otherwise each request
        takes the first node after the previous one that fits (first fit, in order)."""
        if job_resource.nodes:
            out = []
            for name, req in zip(job_resource.nodes, job_resource.resources):
                node = self._row.get(name)
                if node is None:
                    self.logger.error(f"Node {name} not found in cluster")
                    return None
                plan = node.plan(req)
                if plan is None:
                    return None
                out.append((node, *plan))
            return out

        out = []
        nodes = self._nodes
        start = 0
        for req in job_resource.resources:
            # Cheap counters first, so full nodes are skipped without calling plan().
            ncpus = req.cpu_count
            whole, rem = _gpu_split(_units(req.gpu_count)) if isinstance(req, NodeResourceCount) else (0, 0)
            for k in range(start, len(nodes)):
                node = nodes[k]
                if node.ncpu_free < ncpus:
                    continue
                if whole and node.full_mask.bit_count() < whole:
                    continue
                if rem and not node.full_mask and (not node.partial or node.partial[-1][0] < rem):
                    continue
                plan = node.plan(req)
                if plan is not None:
                    out.append((nodes[k], *plan))
                    start = k + 1
                    break
            else:
                return None
        return out

    def _can_allocate(self, job_resource: JobResource) -> List[str]:
        """Names of the nodes the allocation would use, or [] if it doesn't fit."""
        placed = self._place(job_resource)
        return [node.name for node, _, _ in placed] if placed else []

    def get_status(self):
        """Returns current free resources: (free_cpus, free_gpus)"""
        return (self.free_cpus, self.free_gpus)

    def __eq__(self, other) -> bool:
        """Check equality between two ClusterResource instances."""
        if not isinstance(other, ClusterResource):
            return False
        return self.nodes.to_dict() == other.nodes.to_dict()

    def __repr__(self) -> str:
        """Return string representation of the cluster."""
        nodes_str = "\n  ".join(f"{name}: {r}" for name, r in self.nodes.to_dict().items())
        return f"{self.__class__.__name__}(\n  {nodes_str}\n)"

class LocalClusterResource(ClusterResource):
    """
    Manages resource allocation and deallocation for a cluster of nodes.
    Args:
        nodes (JobResource): Job resource containing node names and their resource configurations.
    """

    def allocate(self, job_resource: JobResource) -> Tuple[bool, JobResource]:
        """Allocate specific resource IDs."""
        placed = self._place(job_resource)
        if placed is None:
            return False, job_resource
        resources = []
        for node, cpus, gpus in placed:
            resources.append(node.take(cpus, gpus))
            self._free_cpus -= cpus.bit_count()
            self._free_gpu_units -= sum(gpus.values())
        return True, JobResource(resources=resources, nodes=[node.name for node, _, _ in placed])

    def deallocate(self, job_resource: JobResource) -> bool:
        """Deallocate the resources. Checks every node first (unknown ids, CPUs already
        free, GPUs going past 1.0) and changes nothing if a check fails."""
        if not job_resource.nodes:
            self.logger.error("Deallocation failed: JobResource must have nodes specified")
            raise ValueError("JobResource must have nodes specified for deallocation")
        freed = [
            (self._row[name], *self._row[name].freed(res))
            for name, res in zip(job_resource.nodes, job_resource.resources)
        ]
        for node, cpus, gpus in freed:
            node.give(cpus, gpus)
            self._free_cpus += cpus.bit_count()
            self._free_gpu_units += sum(gpus.values())
        return True


class AsyncLocalClusterResource(LocalClusterResource):
    def __init__(self, logger, nodes):
        super().__init__(logger, nodes)
        self._resource_available = asyncio.Event()
        self._resource_available.set()
        self._loop = None
        self._min_resources: Optional[JobResource] = None

    def set_event_loop(self, loop):
        self._loop = loop

    async def wait_for_free(self, min_resources: Optional[JobResource]=None):
        """This will return only when thereare free resources"""
        self._min_resources = min_resources
        await self._resource_available.wait()

    async def signal_resource_available(self):
        """Signal that resources might be available. Used for waking up blocked waiters."""
        self._resource_available.set()

    def allocate(self, job_resource):
        allocated,allocated_resource = super().allocate(job_resource)

        if allocated:
            # Clear the event when the cluster is full, or can no longer fit the
            # smallest pending task.
            if self._free_cpus == 0 and self._free_gpu_units == 0:
                self._resource_available.clear()
            elif self._min_resources is not None and self._place(self._min_resources) is None:
                self._resource_available.clear()

        return allocated, allocated_resource

    def deallocate(self, job_resource):
        """Deallocate resources and return them to the queue.

        Must be called on the event loop thread (both schedulers' ``free`` are).
        The event is set synchronously, in the same step that frees the
        resources, so it always describes the cluster as it is now. Deferring the
        set with ``call_soon_threadsafe`` let it fire after the freed slot had
        already been refilled, leaving the event set on a full cluster.
        """
        result = super().deallocate(job_resource)

        if result:
            if self._min_resources is None or self._place(self._min_resources) is not None:
                self._resource_available.set()
        return result

    def set_resource_available(self):
        """Set the resource available event."""
        self.logger.debug("Setting resource available event")
        self._resource_available.set()

    def clear_resource_available(self):
        """Clear the resource available event."""
        self.logger.debug("Clearing resource available event")
        self._resource_available.clear()
