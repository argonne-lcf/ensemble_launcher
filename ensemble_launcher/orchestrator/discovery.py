"""File-based discovery of running orchestrator nodes.

Both ``ClusterClient`` and ``PolicyClient`` find a live node the same way: poll the
checkpoint directory until the node has written its address there. These helpers are
the shared half of that, so the two clients cannot drift on path layout.

A node's files live at ``<checkpoint_dir>/<node_id split on ".">/<node_id>_*``, e.g.
``main.w0`` -> ``<ckpt>/main/w0/main.w0_policy.ckpt``.
"""

import os
import time
from dataclasses import dataclass
from typing import List


def _wait_for_path(path: str, timeout: float, poll_interval: float = 1.0) -> None:
    """Block until *path* exists (file or non-empty directory), or raise TimeoutError."""
    deadline = time.monotonic() + timeout
    while True:
        if os.path.isfile(path):
            return
        if os.path.isdir(path) and os.listdir(path):
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out after {timeout:.0f}s waiting for checkpoint path: {path}"
            )
        time.sleep(min(poll_interval, remaining))


def _resolve_node_id(checkpoint_dir: str, node_id: str) -> str:
    """Resolve a symbolic node_id to a concrete node_id.

    ``"global"`` resolves to the shortest node_id found in *checkpoint_dir*
    (the global master always has the shortest name, e.g. ``"main"``).
    Any other value is returned unchanged.
    """
    if node_id not in ["global", "local"]:
        return node_id
    else:
        if node_id == "global":
            return os.listdir(checkpoint_dir)[0]
        elif node_id == "local":
            raise NotImplementedError("local connection is not implemented")
        else:
            raise ValueError(f"Unknown node_id {node_id}")


def node_dir(checkpoint_dir: str, node_id: str) -> str:
    """Directory holding *node_id*'s checkpoint files."""
    return os.path.join(checkpoint_dir, *node_id.split("."))


@dataclass
class PolicyEndpointInfo:
    """Everything a ``PolicyClient`` needs to connect to one node's policy endpoint."""

    node_id: str
    address: str
    secret: str


def read_policy_endpoint(
    checkpoint_dir: str, node_id: str = "global", timeout: float = 60.0
) -> PolicyEndpointInfo:
    """Wait for *node_id*'s policy endpoint files and return its address and secret.

    Raises:
        TimeoutError: the node never published an endpoint. The usual cause is that
            it was launched with neither ``cluster=True`` nor
            ``enable_policy_client=True``, so no endpoint was ever bound.
    """
    from ensemble_launcher.comm.pipe import AsyncZMQRouterConnectionState

    _wait_for_path(checkpoint_dir, timeout=timeout)
    resolved = _resolve_node_id(checkpoint_dir, node_id)
    base = node_dir(checkpoint_dir, resolved)

    endpoint_path = os.path.join(base, f"{resolved}_policy.ckpt")
    _wait_for_path(endpoint_path, timeout=timeout)
    with open(endpoint_path, "r") as f:
        state = AsyncZMQRouterConnectionState.deserialize(f.read())

    # The endpoint writes the secret before the address, so by the time the address
    # file exists the secret is already there -- no second wait needed.
    secret_path = os.path.join(base, f"{resolved}_policy_secret")
    with open(secret_path, "r") as f:
        secret = f.read().strip()

    return PolicyEndpointInfo(
        node_id=resolved, address=state.address, secret=secret
    )


def discover_policy_nodes(checkpoint_dir: str) -> List[str]:
    """Return every node id under *checkpoint_dir* that is serving a policy endpoint.

    Sorted, so a caller fanning out over the result gets a stable order.
    """
    found: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(checkpoint_dir):
        for name in filenames:
            if name.endswith("_policy.ckpt"):
                found.append(name[: -len("_policy.ckpt")])
    return sorted(found)
