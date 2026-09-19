import logging

import pytest
from pydantic import ValidationError

from ensemble_launcher.config import SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.scheduler.resource import (
    NodeResourceList,
    LocalClusterResource,
    NodeResourceCount,
    JobResource,
)
from ensemble_launcher.scheduler.resource.node import NodeResource

pytestmark = pytest.mark.core

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def test_resource():
    import copy

    sys_info = NodeResourceList(cpus=list(range(10)),gpus=[])
    cluster_nodes = JobResource(
        resources=[sys_info, sys_info],
        nodes=[f"node:{str(i)}" for i in range(2)]
    )
    cluster = LocalClusterResource(logger, nodes=cluster_nodes)

    # cluster_copy = copy.deepcopy(cluster.nodes)

    resources = []
    resources.append(NodeResourceCount(ncpus=5,ngpus=0))
    resources.append(NodeResourceList(cpus=[1,3,5,7,9]))

    job = JobResource(resources=resources)
    allocated,allocated_job = cluster.allocate(job)

    for req,alloc in zip(job.resources,allocated_job.resources):
        assert req == alloc, "request is not same as allocation"

    cluster.deallocate(allocated_job)

    # assert cluster == cluster_copy, "Cluster is not the same"


def test_whole_gpu_unchanged():
    """Regression: whole-GPU (integer) requests still hand out distinct ids."""
    sys_info = NodeResourceList(cpus=list(range(10)), gpus=[0, 1, 2, 3])
    cluster_nodes = JobResource(
        resources=[sys_info, sys_info],
        nodes=[f"node:{str(i)}" for i in range(2)],
    )
    cluster = LocalClusterResource(logger, nodes=cluster_nodes)

    job = JobResource(resources=[NodeResourceCount(ncpus=2, ngpus=2)])
    allocated, allocated_job = cluster.allocate(job)
    assert allocated is True
    assert allocated_job.resources[0].gpu_count == 2
    assert set(allocated_job.resources[0].gpus).issubset({0, 1, 2, 3})

    cluster.deallocate(allocated_job)
    assert cluster.free_gpus == 8


def test_resource_overload():
    """
    Historically this test expressed "5 tasks sharing one GPU" by repeating
    id 0 ten times in SystemConfig-style construction. That mechanism is
    retired (see test_duplicate_gpu_ids_rejected) in favor of fractional
    ngpus_per_process requests against a single real id, which is the
    scenario rewritten here: five 0.2-GPU requests against gpus=[0] all fit,
    a sixth does not.
    """
    # A single node so a 6th request has nowhere else to go -- with two
    # nodes the scheduler would just satisfy it from the second node's
    # independent GPU 0, which isn't the oversubscription check this test
    # wants.
    sys_info = NodeResourceList(cpus=list(range(10)), gpus=[0])
    cluster_nodes = JobResource(
        resources=[sys_info],
        nodes=["node:0"],
    )
    cluster = LocalClusterResource(logger, nodes=cluster_nodes)

    allocated_jobs = []
    for _ in range(5):
        job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.2)])
        allocated, allocated_job = cluster.allocate(job)
        assert allocated is True, "expected fractional GPU request to be satisfied"
        assert allocated_job.resources[0].gpus == (0,)
        allocated_jobs.append(allocated_job)

    # A sixth 0.2 request would push the node's GPU 0 past 1.0 -- rejected.
    sixth_job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.2)])
    allocated, _ = cluster.allocate(sixth_job)
    assert allocated is False, "sixth fractional request should not fit"

    for allocated_job in allocated_jobs:
        cluster.deallocate(allocated_job)
    print(cluster)


def test_fractional_gpu_allocation():
    sys_info = NodeResourceList(cpus=list(range(4)), gpus=[0, 1])
    cluster_nodes = JobResource(resources=[sys_info], nodes=["node:0"])
    cluster = LocalClusterResource(logger, nodes=cluster_nodes)

    allocated_jobs = []
    for _ in range(4):
        job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.25)])
        allocated, allocated_job = cluster.allocate(job)
        assert allocated is True
        assert allocated_job.resources[0].gpus == (0,)
        allocated_jobs.append(allocated_job)

    assert cluster.free_gpus == 1.0

    for allocated_job in allocated_jobs:
        cluster.deallocate(allocated_job)
    assert cluster.free_gpus == 2.0


def test_fractional_no_oversubscription():
    sys_info = NodeResourceList(cpus=list(range(4)), gpus=[0])
    cluster_nodes = JobResource(resources=[sys_info], nodes=["node:0"])
    cluster = LocalClusterResource(logger, nodes=cluster_nodes)

    before = cluster.nodes.resources[0]
    allocated_jobs = []
    for _ in range(4):
        job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.25)])
        allocated, allocated_job = cluster.allocate(job)
        assert allocated is True
        allocated_jobs.append(allocated_job)

    fifth_job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.25)])
    allocated, unchanged = cluster.allocate(fifth_job)
    assert allocated is False
    assert unchanged == fifth_job

    for allocated_job in allocated_jobs:
        cluster.deallocate(allocated_job)


def test_fractional_deallocate_restores():
    """Float-drift guard: many allocate/deallocate cycles return exactly to 1.0."""
    sys_info = NodeResourceList(cpus=list(range(4)), gpus=[0])
    cluster_nodes = JobResource(resources=[sys_info], nodes=["node:0"])
    cluster = LocalClusterResource(logger, nodes=cluster_nodes)

    for _ in range(25):
        job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.1)])
        allocated, allocated_job = cluster.allocate(job)
        assert allocated is True
        cluster.deallocate(allocated_job)
        assert cluster.free_gpus == 1.0


def test_duplicate_gpu_ids_rejected():
    with pytest.raises(ValidationError):
        SystemConfig(name="x", ngpus=8, gpus=[0, 1] * 4)

    with pytest.raises(ValidationError):
        SystemConfig(name="x", ncpus=8, cpus=[0, 1] * 4)

    unique = SystemConfig(name="x", ngpus=2, gpus=[0, 1])
    assert NodeResourceList.from_config(unique).gpu_count == 2.0


def test_fractional_serialization_roundtrip():
    # A partially-allocated node can only be built by subtracting -- the
    # constructor takes ids only, all fully free.
    original = NodeResourceList(cpus=(0, 1), gpus=[0, 1]) - NodeResourceCount(
        ncpus=0, ngpus=0.75
    )
    assert original.gpu_amounts == {0: 0.25, 1: 1.0}

    payload = original.serialize()
    restored = NodeResource.deserialize(payload)
    assert restored == original
    assert restored.gpu_amounts == {0: 0.25, 1: 1.0}

    # Old checkpoints stored a flat id list; must still deserialize as
    # whole GPUs.
    old_style = {"type": "list", "cpus": [0, 1], "gpus": [0, 1]}
    restored_old = NodeResource.deserialize(old_style)
    assert restored_old.gpu_count == 2
    assert restored_old.gpu_amounts == {0: 1.0, 1: 1.0}


def test_duplicate_gpu_ids_raise_in_node_resource():
    """Repeating an id is the retired oversubscription trick -- reject it
    rather than quietly folding the duplicates into one device."""
    with pytest.raises(ValueError, match="duplicate GPU id"):
        NodeResourceList(cpus=(0, 1), gpus=[0, 0, 1])

    # Amount-carrying forms are no longer constructor input at all: every
    # GPU starts at 1.0 and fractions come from allocation.
    with pytest.raises(TypeError, match="flat list of device ids"):
        NodeResourceList(cpus=(0, 1), gpus={0: 5.0})

    with pytest.raises(TypeError, match="flat list of device ids"):
        NodeResourceList(cpus=(0, 1), gpus=[(0, 1.0), (1, 2.0)])


def test_with_cpus_preserves_gpu_amounts():
    """The head-node trimming path in async_master/async_worker: dropping a
    CPU must not silently restore a partly-used GPU to fully free."""
    node = NodeResourceList(cpus=(0, 1, 2, 3), gpus=[0, 1])
    partial = node - NodeResourceCount(ncpus=0, ngpus=0.25)
    assert partial.gpu_amounts == {0: 0.75, 1: 1.0}

    trimmed = partial.with_cpus(partial.cpus[1:])
    assert trimmed.cpus == (1, 2, 3)
    assert trimmed.gpu_amounts == {0: 0.75, 1: 1.0}


def test_divide_preserves_partial_amounts():
    """divide() used to rebuild each part from bare ids, rounding a
    partly-allocated device back up to a whole free GPU."""
    node = NodeResourceList(cpus=(0, 1, 2, 3), gpus=[0, 1])
    partial = node - NodeResourceCount(ncpus=0, ngpus=0.25)

    parts = partial.divide(2)
    assert [p.gpu_amounts for p in parts] == [{0: 0.75}, {1: 1.0}]


def test_gpu_amount_never_exceeds_one():
    """A device is never worth more than one whole GPU. Exceeding that can
    only come from deallocating the same allocation twice, which would let
    the scheduler oversubscribe a real GPU -- so it must raise."""
    sys_info = NodeResourceList(cpus=list(range(4)), gpus=[0])
    cluster = LocalClusterResource(
        logger, nodes=JobResource(resources=[sys_info], nodes=["node:0"])
    )

    job = JobResource(resources=[NodeResourceCount(ncpus=1, ngpus=0.5)])
    allocated, allocated_job = cluster.allocate(job)
    assert allocated is True

    cluster.deallocate(allocated_job)
    with pytest.raises(ValueError, match="deallocated twice"):
        cluster.deallocate(allocated_job)

    with pytest.raises(ValueError, match="gpu_fraction"):
        NodeResourceList.request(cpus=(0,), gpus=[0], gpu_fraction=1.5)


def test_request_builds_fractional_demand():
    """The one caller that legitimately wants a fraction of specific ids:
    a task with gpu_affinity set and a fractional ngpus_per_process."""
    demand = NodeResourceList.request(cpus=(0, 1), gpus=[0, 1], gpu_fraction=0.25)
    assert demand.gpu_amounts == {0: 0.25, 1: 0.25}
    assert demand.gpus == (0, 1)

    pool = NodeResourceList(cpus=(0, 1, 2, 3), gpus=[0, 1])
    assert demand in pool
    assert (pool - demand).gpu_amounts == {0: 0.75, 1: 0.75}


def test_task_fractional_requirements():
    task_multi = Task(
        task_id="t1", nnodes=1, ppn=4, ngpus_per_process=0.25, executable="true"
    )
    req_multi = task_multi.get_resource_requirements()
    assert req_multi.resources[0].ngpus == 1.0

    task_single = Task(
        task_id="t2", nnodes=1, ppn=1, ngpus_per_process=0.25, executable="true"
    )
    req_single = task_single.get_resource_requirements()
    assert req_single.resources[0].ngpus == 0.25


if __name__ == "__main__":
    test_resource()
    test_whole_gpu_unchanged()
    test_resource_overload()
    test_fractional_gpu_allocation()
    test_fractional_no_oversubscription()
    test_fractional_deallocate_restores()
    test_duplicate_gpu_ids_rejected()
    test_fractional_serialization_roundtrip()
    test_duplicate_gpu_ids_raise_in_node_resource()
    test_with_cpus_preserves_gpu_amounts()
    test_divide_preserves_partial_amounts()
    test_gpu_amount_never_exceeds_one()
    test_request_builds_fractional_demand()
    test_task_fractional_requirements()
