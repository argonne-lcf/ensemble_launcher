"""Read and retune a running node's scheduling-policy state from outside the run.

A policy can only see ``SchedulerState`` -- queue depth, resource occupancy, what is
running where. It cannot see *workflow* state: which round of an optimization loop you
are in, how the last batch scored, which configurations turned out to matter. This
client is how that outside knowledge gets in.

``PolicyClient`` is synchronous with a background thread, matching ``ClusterClient``
rather than ``ActorHandle``: the intended caller is a workflow script that already holds
a ``ClusterClient`` in the same scope, and an async-only handle would push
``asyncio.run`` into every tuning callback.

    with ClusterClient(ckpt) as cc, PolicyClient(ckpt, node_id="main.w0") as pc:
        pc.set_state({"gpu_weight": 8.0}, rescore=True)
        futs = cc.map(train, configs)

The node must have been launched with ``cluster=True`` or ``enable_policy_client=True``,
and with a ``checkpoint_dir`` -- discovery is file-based.
"""

import asyncio
import logging
import secrets
import threading
import uuid
from concurrent.futures import Future as ConcurrentFuture
from typing import Any, Dict, List, Optional

import cloudpickle

from ensemble_launcher.comm.pipe import AsyncZMQDealerConnection
from ensemble_launcher.logging import setup_logger
from ensemble_launcher.orchestrator.discovery import (
    PolicyEndpointInfo,
    discover_policy_nodes,
    read_policy_endpoint,
)
from ensemble_launcher.orchestrator.policy_endpoint import POLICY_CLIENT_PREFIX


class PolicyStateError(RuntimeError):
    """The endpoint rejected a request -- bad policy kind, bad payload, or a policy
    that raised while applying the new state (in which case the state was rolled back
    and the node is unchanged)."""


class PolicyClient:
    """Get and set one node's policy state.

    Args:
        checkpoint_dir: Directory the orchestrator writes its checkpoints to.
        node_id: Which node to tune. ``"global"`` (default) resolves to the root
            master, whose policy is the *children* policy -- task scoring lives on
            workers, so pass e.g. ``"main.w0"`` to tune dispatch order.
        policy_kind: ``"auto"`` picks the node's only policy and errors if ambiguous.
            Pass ``"task"`` or ``"children"`` to assert which one you mean; a mismatch
            raises rather than silently tuning the wrong object.
        request_timeout: Seconds to wait for a reply before raising ``TimeoutError``.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        node_id: str = "global",
        policy_kind: str = "auto",
        client_id: Optional[str] = None,
        log_level: int = logging.INFO,
        checkpoint_timeout: float = 60.0,
        request_timeout: float = 30.0,
    ):
        # The prefix is not cosmetic: PolicyEndpoint rejects identities without it.
        self._client_id = client_id or f"{POLICY_CLIENT_PREFIX}{secrets.token_hex(4)}"
        self._policy_kind = policy_kind
        self._request_timeout = request_timeout
        self.logger = setup_logger(__name__, self._client_id, level=log_level)

        self._info: PolicyEndpointInfo = read_policy_endpoint(
            checkpoint_dir, node_id, timeout=checkpoint_timeout
        )
        self.logger.info(
            f"Resolved policy endpoint for {self._info.node_id} at {self._info.address}"
        )

        self._conn = AsyncZMQDealerConnection(
            identity=self._client_id,
            # The endpoint authenticates on this, so it is the policy secret, not the
            # cluster secret -- tuning a policy must not require the ability to submit
            # arbitrary tasks.
            secret_id=self._info.secret,
            remote_address=self._info.address,
            req_res=True,
        )

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_ready = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._pending: Dict[str, ConcurrentFuture] = {}
        self._lock = threading.Lock()
        self._policy_name: Optional[str] = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        if self._started:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._loop_ready.wait()
        self._started = True

    def teardown(self) -> None:
        if not self._started:
            return
        # Fail every in-flight call rather than leaving a caller blocked forever on a
        # reply that can no longer arrive.
        with self._lock:
            pending, self._pending = self._pending, {}
        for fut in pending.values():
            if not fut.done():
                fut.set_exception(PolicyStateError("client torn down"))

        if self._loop is not None:
            asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop).result(
                timeout=10.0
            )
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._started = False

    def __enter__(self) -> "PolicyClient":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.teardown()

    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._conn.open())
        self._recv_task = self._loop.create_task(self._recv_loop())
        self._loop_ready.set()
        self._loop.run_forever()

    async def _shutdown(self) -> None:
        if self._recv_task is not None:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        await self._conn.close()

    # ------------------------------------------------------------------
    # Transport
    # ------------------------------------------------------------------

    async def _recv_loop(self) -> None:
        while True:
            try:
                frames = await self._conn.recv(timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"PolicyClient recv failed: {e}")
                continue

            try:
                request_id, ok, payload = cloudpickle.loads(frames[-1])
            except Exception as e:
                self.logger.warning(f"PolicyClient could not decode a reply: {e}")
                continue

            with self._lock:
                fut = self._pending.pop(request_id, None)
            if fut is None or fut.done():
                # A duplicate delivery of a reply we already resolved, or one for a
                # request that timed out on this side. Both are expected.
                continue
            if ok:
                self._policy_name = payload.get("policy_name")
                fut.set_result(payload)
            else:
                fut.set_exception(PolicyStateError(str(payload)))

    def _request(self, op: str, params: Dict[str, Any]) -> ConcurrentFuture:
        if not self._started:
            raise RuntimeError(
                "PolicyClient is not started; call start() or use it as a "
                "context manager"
            )
        params = dict(params)
        params.setdefault("policy_kind", self._policy_kind)
        request_id = str(uuid.uuid4())
        fut: ConcurrentFuture = ConcurrentFuture()
        with self._lock:
            self._pending[request_id] = fut

        payload = cloudpickle.dumps((request_id, op, params))

        async def _send() -> None:
            ok = await self._conn.send(payload, timeout=self._request_timeout)
            if not ok:
                with self._lock:
                    f = self._pending.pop(request_id, None)
                if f is not None and not f.done():
                    f.set_exception(
                        PolicyStateError(
                            f"{op} to {self._info.node_id} was not acknowledged; "
                            "the node may be down or restarting"
                        )
                    )

        asyncio.run_coroutine_threadsafe(_send(), self._loop)
        return fut

    def _resolve(self, fut: ConcurrentFuture, timeout: Optional[float]) -> Dict[str, Any]:
        return fut.result(timeout=timeout if timeout is not None else self._request_timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(
        self, keys: Optional[List[str]] = None, timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """Return the node's current policy state, or just *keys* if given.

        Keys the policy does not hold are omitted rather than returned as ``None``.
        """
        return self._resolve(self.get_state_async(keys), timeout)["state"]

    def set_state(
        self,
        state: Dict[str, Any],
        merge: bool = True,
        rescore: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Update the node's policy state and return the full state afterwards.

        Args:
            state: Keys to write. Values are cloudpickled, so arbitrary objects work.
            merge: ``True`` (default) updates the given keys and leaves the rest alone;
                ``False`` replaces the state wholesale.
            rescore: Re-prioritize the tasks already queued on this node. **Off by
                default**, because without it a tune only affects tasks submitted
                afterwards -- priorities are computed once, when a task enters the
                pending heap, and the heap is thereafter a cache of those scores.
                Reordering a live queue is a real scheduling change that costs one
                ``get_score`` per pending task, so it should never happen as a side
                effect of a tune the caller thought was passive. Only tasks still
                *pending* are reordered: already-dispatched tasks are not recalled, and
                masters ignore the flag entirely (see ``PolicyEndpoint.register``).

        Raises:
            PolicyStateError: the endpoint rejected the update, or the policy raised
                while applying it. In the latter case the node rolled back to its
                previous state, so a failed call leaves nothing half-applied.
        """
        return self._resolve(
            self.set_state_async(state, merge=merge, rescore=rescore), timeout
        )["state"]

    def get_state_async(self, keys: Optional[List[str]] = None) -> ConcurrentFuture:
        """Non-blocking :meth:`get_state`. The future resolves to the full reply dict
        (``state``, ``version``, ``policy_name``, ...), not just the state."""
        return self._request("get", {"keys": keys})

    def set_state_async(
        self, state: Dict[str, Any], merge: bool = True, rescore: bool = False
    ) -> ConcurrentFuture:
        """Non-blocking :meth:`set_state`. Resolves to the full reply dict, whose
        ``rescored`` field reports how many tasks were re-prioritized."""
        return self._request(
            "set", {"state": state, "merge": merge, "rescore": rescore}
        )

    @property
    def node_id(self) -> str:
        return self._info.node_id

    @property
    def policy_name(self) -> Optional[str]:
        """Class name of the policy being tuned; ``None`` until the first call."""
        return self._policy_name

    @staticmethod
    def discover_nodes(checkpoint_dir: str) -> List[str]:
        """Node ids under *checkpoint_dir* that are serving a policy endpoint."""
        return discover_policy_nodes(checkpoint_dir)


class PolicyGroupClient:
    """Fan one ``get_state``/``set_state`` out over several nodes.

    Fan-out is client-side: there is no broadcast on the server, so this is N
    independent requests. They are issued together and collected afterwards, so the
    cost is one round trip rather than N.

        with PolicyGroupClient(ckpt) as g:      # every node with an endpoint
            g.set_state({"gpu_weight": 8.0}, rescore=True)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        node_ids: Optional[List[str]] = None,
        **kwargs,
    ):
        if node_ids is None:
            node_ids = discover_policy_nodes(checkpoint_dir)
            if not node_ids:
                raise ValueError(
                    f"No policy endpoints found under {checkpoint_dir}. Launch with "
                    "cluster=True or enable_policy_client=True."
                )
        self._clients = [
            PolicyClient(checkpoint_dir, node_id=n, **kwargs) for n in node_ids
        ]

    def start(self) -> None:
        for c in self._clients:
            c.start()

    def teardown(self) -> None:
        for c in self._clients:
            c.teardown()

    def __enter__(self) -> "PolicyGroupClient":
        self.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.teardown()

    @property
    def node_ids(self) -> List[str]:
        return [c.node_id for c in self._clients]

    def get_state(
        self, keys: Optional[List[str]] = None, timeout: Optional[float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """``{node_id: state}`` across the group."""
        futs = [(c, c.get_state_async(keys)) for c in self._clients]
        return {c.node_id: c._resolve(f, timeout)["state"] for c, f in futs}

    def set_state(
        self,
        state: Dict[str, Any],
        merge: bool = True,
        rescore: bool = False,
        timeout: Optional[float] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """``{node_id: state}`` after the update.

        Not atomic across nodes: if one node rejects the update the others keep theirs,
        and the ``PolicyStateError`` names only the first failure. Each node is
        individually all-or-nothing, so there are no half-applied nodes -- but a partial
        group is possible, and a caller that needs consistency should re-read with
        :meth:`get_state`.
        """
        futs = [
            (c, c.set_state_async(state, merge=merge, rescore=rescore))
            for c in self._clients
        ]
        return {c.node_id: c._resolve(f, timeout)["state"] for c, f in futs}
