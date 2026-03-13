import logging
import os
import signal
import socket
import time
import uuid

import pytest
from utils import echo_sleep

from ensemble_launcher import EnsembleLauncher
from ensemble_launcher.config import LauncherConfig, PolicyConfig, SystemConfig
from ensemble_launcher.ensemble import Task
from ensemble_launcher.orchestrator import ClusterClient


def _make_tasks(n: int):
    return {
        f"task-{i}": Task(
            task_id=f"task-{i}",
            nnodes=1,
            ppn=1,
            executable=echo_sleep,
            args=(f"task-{i}", 5.0),
        )
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# Cluster mode — start() + ClusterClient
# ---------------------------------------------------------------------------


def test_el_fault_tolerance():
    ckpt_dir = os.path.join(os.getcwd(), f"ckpt_{str(uuid.uuid4())}")
    os.makedirs(ckpt_dir, exist_ok=True)
    tasks = _make_tasks(120)

    el = EnsembleLauncher(
        ensemble_file={},
        system_config=SystemConfig(name="local", ncpus=12, cpus=list(range(12))),
        launcher_config=LauncherConfig(
            task_executor_name="async_processpool",
            child_executor_name="async_mpi",
            comm_name="async_zmq",
            policy_config=PolicyConfig(nlevels=2, nchildren=2),
            return_stdout=True,
            worker_logs=True,
            master_logs=True,
            cpu_binding_option="",
            use_mpi_ppn=False,
            cluster=True,
            checkpoint_dir=ckpt_dir,
            report_interval=1.0,
            children_scheduler_policy="simple_split_children_policy",
            log_level=logging.INFO,
        ),
        Nodes=[socket.gethostname()],
    )

    el.start()
    time.sleep(10.0)

    results = {}
    with ClusterClient(checkpoint_dir=ckpt_dir, node_id="global") as client:
        futures = {task_id: client.submit(task) for task_id, task in tasks.items()}

        # Kill main.m0 node
        import json

        fname = os.path.join(ckpt_dir, "main", "m0", "main.m0_meta.json")
        with open(fname, "r") as f:
            meta_data = json.load(f)
            pid = meta_data.get("pid", None)
            hostname = meta_data.get("hostname", None)

            if pid is not None and hostname in socket.gethostname():
                os.kill(pid, signal.SIGKILL)

        for task_id, fut in futures.items():
            r = fut.result(timeout=300.0)
            results[task_id] = r.split(",")[0].strip() if isinstance(r, str) else r

    el.stop()

    assert len(results) == len(tasks)


if __name__ == "__main__":
    test_el_fault_tolerance()
    print("All tests passed")
