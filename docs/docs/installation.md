# Installation

## Quick Install

```bash
pip install ensemble_launcher
```

Or from source:

```bash
git clone https://github.com/argonne-lcf/ensemble_launcher.git
cd ensemble_launcher
pip install .
```

## Editable Install (Development)

```bash
pip install -e ".[dev-core]"
```

This includes `pytest`, `pytest-timeout`, `pytest-asyncio`, and `mpi4py`.

For the full dev environment (adds MCP and inference dependencies):

```bash
pip install -e ".[dev-extensions]"
```

## Optional Extras

### HPC (MPI)

For distributed multi-node execution:

```bash
pip install "ensemble_launcher[hpc]"
```

Installs [mpi4py](https://mpi4py.readthedocs.io/). Requires an MPI implementation (e.g. MPICH, OpenMPI) on the system.

### MCP Support

For hosting MCP servers on HPC compute nodes:

```bash
pip install "ensemble_launcher[mcp]"
```

Installs [MCP](https://github.com/modelcontextprotocol/python-sdk) and [paramiko](https://www.paramiko.org/) for SSH tunneling.

### Inference

For vLLM-based inference serving:

```bash
pip install "ensemble_launcher[inference]"
```

Installs [vLLM](https://docs.vllm.ai/), uvloop, uvicorn, and FastAPI.

### Everything

```bash
pip install "ensemble_launcher[all]"
```

Installs HPC, MCP, and inference dependencies together.
