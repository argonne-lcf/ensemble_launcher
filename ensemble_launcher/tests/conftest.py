def pytest_configure(config):
    config.addinivalue_line("markers", "core: core functionality tests")
    config.addinivalue_line("markers", "extensions: extension tests (vllm, mcp, etc.)")
