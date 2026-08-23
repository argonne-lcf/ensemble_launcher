# Ensemble Launcher

A lightweight, scalable tool for launching and orchestrating task ensembles across HPC clusters with intelligent resource management and hierarchical execution.

[Documentation](https://argonne-lcf.github.io/ensemble_launcher/) | [GitHub](https://github.com/argonne-lcf/ensemble_launcher)

## Features

- **Scalability** -- Hierarchical master-worker architecture tested from 1 to 2048+ nodes
- **Flexible Execution** -- Support for serial, MPI, and mixed workloads with Python callables or shell commands
- **Co-Scheduling** -- Run heterogeneous tasks (different node counts, GPU requirements) in a single ensemble
- **Custom Scheduling Policies** -- Pluggable policy system with built-in bin-packing, split, and FIFO strategies, or write your own
- **Actors** -- Distributed actor model with async/await communication over ZMQ for long-lived stateful services
- **Inference** -- Actor-based vLLM wrappers for offline, online, and multi-node LLM serving on HPC clusters

## Quick Example

```python
from ensemble_launcher import EnsembleLauncher

el = EnsembleLauncher("config.json")
results = el.run()
```

```bash
# Or use the CLI
el start my_ensemble.json
```

## Acknowledgments

This work was supported by the U.S. Department of Energy, Office of Science, under contract DE-AC02-06CH11357.

## Citation

```bibtex
@article{tummalapalli2026overcoming,
  title={Overcoming Orchestration Bottlenecks at Exascale: A Decentralized, Policy-Driven Approach for Sim-AI Ensembles},
  author={Tummalapalli, Harikrishna and Simpson, Christine M and Balin, Riccardo and Morozov, Vitali A and Pham, Thang D and Keceli, Murat and Uram, Thomas D},
  journal={arXiv preprint arXiv:2607.12211},
  year={2026}
}
```
