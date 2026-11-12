import multiprocessing
import os


def pytest_configure(config):
    multiprocessing.set_start_method("spawn", force=True)
    config.addinivalue_line("markers", "core: core functionality tests")
    config.addinivalue_line("markers", "extensions: extension tests (vllm, mcp, etc.)")

    # EL_TEST_COMM_IO_PROCESS=1 runs the whole suite with every node's data connections
    # out of process (LauncherConfig.comm_io_process defaults to True). Spawned nodes get the
    # setting through their serialized config.
    if os.environ.get("EL_TEST_COMM_IO_PROCESS") == "1":
        from ensemble_launcher.config import LauncherConfig

        LauncherConfig.model_fields["comm_io_process"].default = True
        LauncherConfig.model_rebuild(force=True)
