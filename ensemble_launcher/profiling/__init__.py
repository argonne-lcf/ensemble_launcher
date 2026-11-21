"""Profile orchestrator nodes with py-spy, located through the checkpoint directory.

Merging per-node traces lives in :mod:`ensemble_launcher.profiling.merge_traces`, which is
also the command ``python -m ensemble_launcher.profiling.merge_traces``. It is deliberately
not imported here: importing it from the package makes ``python -m`` warn about the module
being loaded twice.
"""

from .pyspy import PySpyProfiler

__all__ = ["PySpyProfiler"]
