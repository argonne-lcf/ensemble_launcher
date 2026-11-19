from abc import ABC, abstractmethod
import heapq
from logging import Logger
from typing import List, Optional, Tuple
import asyncio

from .node import NodeResource, JobResource, NodeResourceCount, NodeResourceList

# GPU amounts are integers counted in quarters of a GPU: a GPU is 1.0 (4 quarters)
# and the smallest share is 0.25 (1 quarter).
_UNITS = 4


def _units(amount) -> int:
    """An amount of GPUs in quarters. Raises unless it is a whole number of quarters
    (0.25, 0.5, 0.75, 1, 1.25, ...); quarters are exact in floating point."""
    q = float(amount) * _UNITS
    if q != int(q):
        raise ValueError(f"GPU amounts must be multiples of 0.25: got {amount}")
    return int(q)


def _gpu_split(units: int) -> Tuple[int, int]:
    """A node's GPU request in quarters, as (whole GPUs, quarters of one GPU): one of the
    two is 0. A node asks for 1, 2, 3, ... whole GPUs (4, 8, 12, ... quarters) or for 1,
    2 or 3 quarters (0.25 / 0.5 / 0.75) of one GPU; anything else raises ValueError."""
    if units < _UNITS:
        return 0, units
    if units % _UNITS:
        raise ValueError(
            f"a node's GPU request must be a whole number of GPUs (1, 2, 3, ...) or 0.25, "
            f"0.5 or 0.75 of one GPU: got {units / _UNITS} (per node, ppn * ngpus_per_process)"
        )
    return units // _UNITS, 0


def _request_units(amount) -> int:
    """A requested GPU amount in quarters, with the request error message."""
    try:
        return _units(amount)
    except ValueError:
        raise ValueError(
            f"a node's GPU request must be a whole number of GPUs (1, 2, 3, ...) or 0.25, "
            f"0.5 or 0.75 of one GPU: got {amount} (per node, ppn * ngpus_per_process)"
        ) from None


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
    gpu_free    free quarters of each GPU, 0..4 (``gpu_ids[j]`` is GPU j)
    free_mask   free_mask[k], k = 1..4: bit j set = GPU j has exactly k quarters free.
                free_mask[4] is the fully free GPUs. The best fit for a fraction of q
                quarters is the lowest GPU in the first non-empty mask of q, q+1, ..., 4.
    """

    __slots__ = ("name", "index", "cpu_ids", "cpu_bit", "gpu_ids", "gpu_col",
                 "cpu_mask", "ncpu_free", "gpu_free", "free_mask")

    def __init__(self, name: str, index: int, res: NodeResourceList):
        self.name = name
        self.index = index  # position in the cluster's node order (first fit = lowest)
        self.cpu_ids = tuple(res.cpus)
        self.cpu_bit = {c: 1 << b for b, c in enumerate(self.cpu_ids)}
        self.gpu_ids = tuple(res.gpus)
        self.gpu_col = {g: j for j, g in enumerate(self.gpu_ids)}
        self.cpu_mask = (1 << len(self.cpu_ids)) - 1
        self.ncpu_free = len(self.cpu_ids)
        self.gpu_free = [_units(a) for _, a in res._gpu_amounts]
        self.free_mask = [0] * (_UNITS + 1)
        for j, q in enumerate(self.gpu_free):
            if q:
                self.free_mask[q] |= 1 << j

    @property
    def full_mask(self) -> int:
        """Bit j set = GPU j is fully free."""
        return self.free_mask[_UNITS]

    def has_room(self, q: int) -> bool:
        """Whether some GPU has at least ``q`` quarters free."""
        fm = self.free_mask
        return any(fm[k] for k in range(q, _UNITS + 1))

    def _set_gpu(self, j: int, new: int) -> None:
        """Change GPU j's free quarters, moving its bit to the matching free_mask."""
        old = self.gpu_free[j]
        bit = 1 << j
        if old:
            self.free_mask[old] &= ~bit
        self.gpu_free[j] = new
        if new:
            self.free_mask[new] |= bit

    def plan(self, req: NodeResource):
        """What ``req`` would take: (cpu bits, {gpu column: units}), or None if it doesn't fit."""
        if isinstance(req, NodeResourceList):
            try:
                want = 0
                for c in req.cpus:
                    want |= self.cpu_bit[c]
                take = {self.gpu_col[g]: _request_units(a) for g, a in req._gpu_amounts}
            except KeyError:
                return None
            if self.cpu_mask & want != want:
                return None
            if any(self.gpu_free[j] < u for j, u in take.items()):
                return None
            return want, take

        # Count request: the lowest free CPUs, then whole GPUs in id order, or the
        # fraction on the GPU with the least room that still fits it (lowest id on a tie).
        if self.ncpu_free < req.ncpus:
            return None
        take = {}
        units = _request_units(req.ngpus)
        if units:
            whole, q = _gpu_split(units)
            if whole:
                gpus = _lowest(self.free_mask[_UNITS], whole)
                if gpus.bit_count() < whole:
                    return None
                take = {j: _UNITS for j in _bit_indexes(gpus)}
            else:
                for k in range(q, _UNITS + 1):
                    m = self.free_mask[k]
                    if m:
                        take = {(m & -m).bit_length() - 1: q}
                        break
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
        pairs = tuple((self.gpu_ids[j], u / _UNITS) for j, u in sorted(gpus.items()))
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
        pairs = tuple((g, u / _UNITS) for g, u in zip(self.gpu_ids, self.gpu_free) if u > 0)
        cpus = tuple(self.cpu_ids[b] for b in _bit_indexes(self.cpu_mask))
        return NodeResourceList._trusted(cpus, pairs)


def _with_ids(node: NodeResource) -> NodeResourceList:
    """A node as ids. A NodeResourceCount (pin_resources=False) gets CPUs 0..ncpus-1 and
    GPUs 0..: one whole GPU per whole unit of ngpus, plus one partly free GPU for any
    quarters left (ngpus must be a multiple of 0.25)."""
    if isinstance(node, NodeResourceList):
        return node
    q = _units(node.ngpus)
    amounts = {g: 1.0 for g in range(q // _UNITS)}
    if q % _UNITS:
        amounts[q // _UNITS] = (q % _UNITS) / _UNITS
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
        self._nodes = [
            _Node(name, i, _with_ids(r)) for i, (name, r) in enumerate(zip(nodes.nodes, nodes.resources))
        ]
        # Heaps of node indexes by what a request needs, made the first time it's needed:
        #   ("cpu", n)   nodes with at least n free CPUs           (CPU-only requests)
        #   ("full", k)  nodes with at least k fully free GPUs     (k whole GPUs)
        #   ("frac", u)  nodes with a GPU with at least u free     (0.25 / 0.5 / 0.75 of one)
        # Each holds every node that meets its threshold, and maybe some that no longer
        # do (allocate leaves them; they are dropped when they reach the top).
        # _in_heap[key] is the set of nodes in heap ``key``, so none is pushed twice.
        self._heaps = {}
        self._in_heap = {}
        self._row = {node.name: node for node in self._nodes}
        self._free_cpus = sum(node.ncpu_free for node in self._nodes)
        self._free_gpu_units = sum(sum(node.gpu_free) for node in self._nodes)

    @property
    def free_cpus(self) -> int:
        return self._free_cpus

    @property
    def free_gpus(self):
        return self._free_gpu_units / _UNITS

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

        reqs = job_resource.resources
        first = reqs[0]
        if type(first) is NodeResourceCount and all(
            r is first or (type(r) is NodeResourceCount and r == first) for r in reqs
        ):
            found = self._lowest_fitting(first, len(reqs))
            return None if found is None else [(node, *plan) for node, plan in found]

        # Different requests per node, or requests naming ids: scan the nodes in order.
        out = []
        nodes = self._nodes
        start = 0
        for req in reqs:
            # Cheap counters first, so full nodes are skipped without calling plan().
            ncpus = req.cpu_count
            whole, rem = _gpu_split(_request_units(req.gpu_count)) if isinstance(req, NodeResourceCount) else (0, 0)
            for k in range(start, len(nodes)):
                node = nodes[k]
                if node.ncpu_free < ncpus:
                    continue
                if whole and node.full_mask.bit_count() < whole:
                    continue
                if rem and not node.has_room(rem):
                    continue
                plan = node.plan(req)
                if plan is not None:
                    out.append((nodes[k], *plan))
                    start = k + 1
                    break
            else:
                return None
        return out

    @staticmethod
    def _heap_key(req: NodeResourceCount):
        """Which heap answers ``req``: its GPU heap if it asks for GPUs, else its CPU
        heap. None if it asks for nothing."""
        units = _request_units(req.ngpus)
        if units:
            whole, rem = _gpu_split(units)
            return ("full", whole) if whole else ("frac", rem)
        return ("cpu", req.ncpus) if req.ncpus else None

    @staticmethod
    def _meets(key, node: _Node) -> bool:
        """Whether ``node`` meets heap ``key``'s threshold."""
        kind, amount = key
        if kind == "cpu":
            return node.ncpu_free >= amount
        if kind == "full":
            return node.full_mask.bit_count() >= amount
        return node.has_room(amount)

    def _heap(self, key):
        if key not in self._heaps:
            heap = [node.index for node in self._nodes if self._meets(key, node)]
            self._heaps[key] = heap  # ascending, so already a heap
            self._in_heap[key] = set(heap)
        return self._heaps[key], self._in_heap[key]

    def _lowest_fitting(self, req: NodeResourceCount, k: int):
        """The ``k`` lowest nodes that fit ``req``, as [(node, plan)], or None.

        Walks the request's heap from the top: nodes that no longer meet the threshold
        are dropped; nodes that meet it but don't fit the rest of the request (a GPU
        node short of CPUs) are set aside and pushed back afterwards. Every node that
        fits ``req`` is in the heap, so the first k that fit are the k lowest.
        """
        key = self._heap_key(req)
        nodes = self._nodes
        if key is None:  # asks for nothing: the first k nodes
            return [(node, node.plan(req)) for node in nodes[:k]] if len(nodes) >= k else None
        heap, in_heap = self._heap(key)
        found, set_aside = [], []
        while heap:
            node = nodes[heap[0]]
            if not self._meets(key, node):
                in_heap.discard(heapq.heappop(heap))
                continue
            plan = node.plan(req)
            if plan is not None:
                found.append((node, plan))
                if len(found) == k:
                    break  # leave it on top: it may well fit the next request too
            set_aside.append(heapq.heappop(heap))
        for i in set_aside:  # they still meet the threshold
            heapq.heappush(heap, i)
        return found if len(found) == k else None

    def _back_in_heaps(self, node: _Node) -> None:
        """After a deallocate: put ``node`` in every heap whose threshold it now meets."""
        i = node.index
        for key, heap in self._heaps.items():
            in_heap = self._in_heap[key]
            if i not in in_heap and self._meets(key, node):
                in_heap.add(i)
                heapq.heappush(heap, i)

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
            self._back_in_heaps(node)
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
