# Contributing

We welcome contributions!

## Development Setup

```bash
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
pip install -e ".[dev-core]"
```

This installs the core dev dependencies (`pytest`, `pytest-timeout`, `pytest-asyncio`) plus `mpi4py`.

For the full dev environment including MCP and inference dependencies:

```bash
pip install -e ".[dev-extensions]"
```

### Building Docs

```bash
pip install -e ".[docs]"
cd docs
mkdocs serve
```

## Running Tests

Tests live in `ensemble_launcher/tests/` and must be run from that directory:

```bash
cd ensemble_launcher/tests
pytest                                    # all tests
pytest test_ensemble_launcher.py          # end-to-end launcher tests
pytest test_async_master.py               # async master/worker tests
pytest test_cluster.py                    # cluster-mode tests
pytest test_mcp.py                        # MCP interface tests
pytest test_ensemble_launcher.py::test_el_run  # single test
```

Tests require `pytest-asyncio`. Async tests are decorated with `@pytest.mark.asyncio`.

## Pull Request Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- **Issues**: [GitHub Issues](https://github.com/argonne-lcf/ensemble_launcher/issues)
- **Discussions**: [GitHub Discussions](https://github.com/argonne-lcf/ensemble_launcher/discussions)
