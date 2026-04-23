import logging
import os

# Load external policies
import sys

from .async_master import AsyncMaster
from .async_worker import AsyncWorker
from .async_workstealing_master import AsyncWorkStealingMaster
from .async_workstealing_worker import AsyncWorkStealingWorker
from .cluster_client import ClusterClient
from .node import Node

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def load_external_policies():
    """
    Load external policy modules for use in distributed settings.

    Set EL_EXTERNAL_POLICY_MODULE environment variable to specify the module to import.
    Example: EL_EXTERNAL_POLICY_MODULE=my_custom_policies

    Alternatively, set EL_EXTERNAL_POLICY_PATH to add a directory to sys.path,
    then the module will be loaded from there.
    Example: EL_EXTERNAL_POLICY_PATH=/path/to/policies EL_EXTERNAL_POLICY_MODULE=my_policies
    """
    policy_path = os.environ.get("EL_EXTERNAL_POLICY_PATH")
    policy_module = os.environ.get("EL_EXTERNAL_POLICY_MODULE")

    if policy_path and policy_path not in sys.path:
        logger.info(f"Adding {policy_path} to sys.path for external policies")
        sys.path.insert(0, policy_path)

    if policy_module:
        try:
            logger.info(f"Attempting to load external policy module: {policy_module}")
            imported_module = __import__(policy_module)

            # Log what policies are now available
            from ensemble_launcher.scheduler.policy import policy_registry

            all_policies = list(policy_registry.available_policies.keys()) + list(
                policy_registry.available_children_policies.keys()
            )
            logger.info(f"External policies loaded. Available policies: {all_policies}")

        except ImportError as e:
            logger.warning(
                f"Could not import external policy module '{policy_module}': {e}"
            )
        except Exception as e:
            logger.error(
                f"Error loading external policy module '{policy_module}': {e}",
                exc_info=True,
            )
    elif policy_path:
        logger.info(
            f"EL_EXTERNAL_POLICY_PATH set but EL_EXTERNAL_POLICY_MODULE not specified. "
            f"Policies from {policy_path} must be imported explicitly."
        )


# Load policies when module is imported (both on master and workers)
load_external_policies()
