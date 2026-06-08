from ensemble_launcher.config import LauncherConfig, MPIConfig, PolicyConfig


def default_inference_launcher_config(
    nnodes: int,
    checkpoint_dir: str,
    **overrides,
) -> LauncherConfig:
    defaults = dict(
        child_executor_name="async_mpi",
        task_executor_name=["async_processpool", "async_mpi"],
        comm_name="async_zmq",
        children_scheduler_policy="fixed_leafs_children_policy",
        policy_config=PolicyConfig(
            nlevels=1 if nnodes <= 256 else 2, leaf_nodes=nnodes
        ),
        mpi_config=MPIConfig(flavor="mpich", cpu_bind_method="none"),
        cluster=True,
        worker_logs=False,
        master_logs=True,
        return_stdout=True,
        checkpoint_dir=checkpoint_dir,
        report_interval=10.0,
        task_flush_interval=0.5,
        result_flush_interval=0.5,
    )
    defaults.update(overrides)
    return LauncherConfig(**defaults)
