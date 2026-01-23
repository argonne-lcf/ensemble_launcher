from typing import List, Dict, Optional
from abc import ABC, abstractmethod
# from SimAIBench.datastore import DataStore, ServerManager
import os
import contextlib
import uuid
import time
import copy
from logging import Logger
from .node import NodeResource, JobResource, NodeResourceCount, NodeResourceList
# from SimAIBench.config import server_registry
import asyncio

SUPPORTED_BACKENDS = ["redis","filesystem"]
# Configure logging
# self.logger = logging.getself.logger(__name__)

class ClusterResource(ABC):
    """
    Abstract base class for managing cluster resources and job allocation.
    Note:
        This is an abstract base class and cannot be instantiated directly. Concrete
        implementations must provide the allocate() and deallocate() methods.
    """

    def __init__(self, logger: Logger, nodes: JobResource):
        self.logger = logger
        self.update_nodes(nodes)
        if self._nodes is not None:
            self.logger.debug(f"Node configuration: {list(self._nodes.keys())}")

    def update_nodes(self, nodes: JobResource):
        """Update the cluster nodes and their system information."""
        try:
            self.logger.info(f"Updating cluster nodes to {len(nodes.nodes)} nodes")
            self._nodes = nodes.to_dict()
            self.logger.debug(f"Updated node configuration: {list(self._nodes.keys())}")
        except Exception as e:
            self._nodes = None
            self.logger.warning(f"Failed to update cluster nodes: {e}")
            
    
    @property
    def free_cpus(self) -> int:
        return sum([node.cpu_count for node in self._nodes.values()])

    @property
    def free_gpus(self) -> int:
        return sum([node.gpu_count for node in self._nodes.values()])
    
    @property
    def nodes(self) -> Optional[JobResource]:
        try:
            return JobResource.from_dict(self._nodes)
        except Exception as e:
            self.logger.error(f"Failed to retrieve nodes: {e}")
            return None
    
    @abstractmethod
    def allocate(self, job_resource: JobResource):
        pass

    @abstractmethod
    def deallocate(self, job_resource: JobResource):
        pass

    def _can_allocate(self, job_resource: JobResource) -> List[str]:
        """Check if the job resource can be allocated.
        
        Returns:
            List[str]: List of node names where allocation can happen. 
                      Empty list if allocation is not possible.
        """
        self.logger.debug(f"Checking allocation feasibility for job with {len(job_resource.resources)} resources")
        
        if not job_resource.nodes:
            # Need to find at least len(resources) nodes to allocate
            job_counter = 0
            cluster_counter = 0
            allocated_nodes = []
            node_names = list(self._nodes.keys())
            
            self.logger.debug("Auto-selecting nodes for allocation")
            
            while True:
                if job_counter >= len(job_resource.resources):
                    self.logger.debug(f"Successfully found {len(allocated_nodes)} suitable nodes: {allocated_nodes}")
                    return allocated_nodes
                
                if cluster_counter >= len(self._nodes):
                    self.logger.debug(f"Insufficient resources: only found {len(allocated_nodes)} nodes, need {len(job_resource.resources)}")
                    return []  
                
                resource_req = job_resource.resources[job_counter]
                node_name = node_names[cluster_counter]
                
                if resource_req in self._nodes[node_name]:
                    allocated_nodes.append(node_name)
                    job_counter += 1
                    self.logger.debug(f"Node {node_name} can satisfy resource requirement {job_counter}")
                
                cluster_counter += 1
        else:
            self.logger.debug(f"Checking specific nodes: {job_resource.nodes}")
            for node_id, node_name in enumerate(job_resource.nodes):
                if node_name not in self._nodes:
                    self.logger.error(f"Node {node_name} not found in cluster")
                    return []
                
                available = self._nodes[node_name]
                resource_req = job_resource.resources[node_id]
                
                if resource_req not in available:
                    self.logger.warning(f"Node {node_name} cannot satisfy resource requirement: need {resource_req}, available {available}")
                    return []
            
            self.logger.debug("All specified nodes can satisfy requirements")
            return job_resource.nodes
    
    def get_status(self):
        """Returns current free resources i.e self._nodes dict"""
        free_cpus = sum([node.cpu_count for node in self._nodes.values()])
        free_gpus = sum([node.gpu_count for node in self._nodes.values()])
        return (free_cpus, free_gpus)
    
    def __eq__(self, other) -> bool:
        """Check equality between two ClusterResource instances."""
        if not isinstance(other, ClusterResource):
            return False
        
        # Check if nodes dictionaries have same keys
        if set(self._nodes.keys()) != set(other._nodes.keys()):
            return False
        
        # Check if each node's NodeResource is equal
        for node_name in self._nodes:
            if self._nodes[node_name] != other._nodes[node_name]:
                return False
        
        return True
    
    def __repr__(self) -> str:
        """Return string representation of the cluster."""
        node_info = []
        for node_name, resource in self._nodes.items():
            node_info.append(f"{node_name}: {resource}")
        
        nodes_str = "\n  ".join(node_info)
        return f"{self.__class__.__name__}(\n  {nodes_str}\n)"

class LocalClusterResource(ClusterResource):
    """
    Manages resource allocation and deallocation for a cluster of nodes.
    Attributes:
        _nodes (Dict[str, NodeResource]): Mapping of node names to their available resources.
    Args:
        nodes (JobResource): Job resource containing node names and their resource configurations.
    """

    def allocate(self, job_resource: JobResource) -> tuple[bool, JobResource]:
        """Allocate specific resource IDs."""
        self.logger.debug(f"Starting allocation for job with {len(job_resource.resources)} resource requirements")
        allocation_result = self._can_allocate(job_resource)
        if not allocation_result:
            self.logger.debug("Allocation failed: insufficient resources")
            return False, job_resource

        # Track original state before allocation
        original_state = {}
        allocated_resources = []

        if not job_resource.nodes:
            allocated_nodes = allocation_result
            self.logger.debug(f"Allocating resources on auto-selected nodes: {allocated_nodes}")

            # Capture original state and perform allocation
            for node_id, node_name in enumerate(allocated_nodes):
                resource_req = job_resource.resources[node_id]
                original_state[node_name] = self._nodes[node_name]
                self.logger.debug(f"Requesting {resource_req} from node {node_name}")
                self._nodes[node_name] = self._nodes[node_name] - resource_req

                # Calculate what was actually allocated
                allocated_resource = original_state[node_name] - self._nodes[node_name]
                allocated_resources.append(allocated_resource)
                self.logger.debug(f"Allocated {allocated_resource} on node {node_name}")
                self.logger.debug(f"Remaining resources on node {node_name} {self._nodes[node_name]}")

            # Return JobResource with actual allocated resources
            self.logger.debug(f"Allocation successful.")
            return True, JobResource(resources=allocated_resources, nodes=allocated_nodes)
        else:
            self.logger.debug(f"Allocating resources on specified nodes: {job_resource.nodes}")

            # Handle specified nodes case
            for node_id, node_name in enumerate(job_resource.nodes):
                resource_req = job_resource.resources[node_id]
                original_state[node_name] = self._nodes[node_name]
                self._nodes[node_name] = self._nodes[node_name] - resource_req

                # Calculate what was actually allocated
                allocated_resource = original_state[node_name] - self._nodes[node_name]
                allocated_resources.append(allocated_resource)
                self.logger.debug(f"Allocated {allocated_resource} on node {node_name}")

            self.logger.debug("Allocation successful")
            return True, JobResource(resources=allocated_resources, nodes=job_resource.nodes)
    
    def deallocate(self, job_resource: JobResource) -> bool:
        """Deallocate the resources"""
        if not job_resource.nodes:
            self.logger.error("Deallocation failed: JobResource must have nodes specified")
            raise ValueError("JobResource must have nodes specified for deallocation")
        
        self.logger.debug(f"Starting deallocation for {len(job_resource.nodes)} nodes: {job_resource.nodes}")
        
        for node_id, node_name in enumerate(job_resource.nodes):
            resource_req = job_resource.resources[node_id]
            self._nodes[node_name] += resource_req
            self.logger.debug(f"Deallocated {resource_req} from node {node_name}")

        self.logger.debug("Deallocation successful")
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
        try:
            self._loop.call_soon_threadsafe(self._resource_available.set)
        except RuntimeError:
            self._resource_available.set()

    def allocate(self, job_resource):
        allocated,allocated_resource = super().allocate(job_resource)
        
        if allocated:
            # Check if we should clear the event
            should_clear = False
            
            if all([node_resource.is_empty() for node_resource in self._nodes.values()]):
                # Cluster completely empty
                should_clear = True
            elif self._min_resources is not None:
                # Check if we can still satisfy minimum requirement
                can_allocate_min = self._can_allocate(self._min_resources)
                if not can_allocate_min:
                    should_clear = True
            
            if should_clear:
                self._resource_available.clear()

        return allocated, allocated_resource

    def deallocate(self, job_resource):
        """Deallocate resources and return them to the queue."""
        if not job_resource.nodes:
            self.logger.error("Deallocation failed: JobResource must have nodes specified")
            raise ValueError("JobResource must have nodes specified for deallocation")
        
        # Call parent's deallocate to update internal state
        result = super().deallocate(job_resource)
        
        if result:
            # Signal that resources are now available - instant notification
            if self._min_resources is None:
                self.logger.debug("No minimum resource requirement set, setting resource available event")
                try:
                    self._loop.call_soon_threadsafe(self._resource_available.set)
                except RuntimeError:
                    self._resource_available.set()
            else:
                can_allocate = self._can_allocate(self._min_resources)
                if can_allocate:
                    self.logger.debug(f"Minimum resource requirement met, setting resource available event. Requirement: {self._min_resources}, Available nodes: {can_allocate}")
                    try:
                        self._loop.call_soon_threadsafe(self._resource_available.set)
                    except RuntimeError:
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