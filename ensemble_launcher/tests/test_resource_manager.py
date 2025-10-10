from ensemble_launcher.scheduler.resource import NodeResourceList, LocalClusterResource, NodeResourceCount, JobResource
import logging

logger = logging.getLogger()

def test_resource():
    import copy
    
    sys_info = NodeResourceList(cpus=list(range(10)),gpus=[])
    cluster = LocalClusterResource(logger, nodes=[f"node:{str(i)}" for i in range(2)],system_info=sys_info)

    cluster_copy = copy.deepcopy(cluster)

    resources = []
    resources.append(NodeResourceCount(ncpus=5,ngpus=0))
    resources.append(NodeResourceList(cpus=[1,3,5,7,9]))

    job = JobResource(resources=resources)    
    allocated,allocated_job = cluster.allocate(job)

    for req,alloc in zip(job.resources,allocated_job.resources):
        assert req == alloc, "request is not same as allocation"
    
    cluster.deallocate(allocated_job)
    
    assert cluster == cluster_copy, "Cluster is not the same"

if __name__ == "__main__":
    test_resource()