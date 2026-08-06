import multiprocessing


def pytest_configure(config):
    multiprocessing.set_start_method("forkserver", force=True)
    config.addinivalue_line("markers", "core: core functionality tests")
    config.addinivalue_line("markers", "extensions: extension tests (vllm, mcp, etc.)")
