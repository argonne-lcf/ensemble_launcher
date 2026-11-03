"""Serve a node's scheduling-policy state to external ``PolicyClient``s.

Architecturally this mirrors ``PublicActor`` (``ensemble_launcher/ensemble/actor.py``):
a dedicated ZMQ ROUTER whose address is published to a checkpoint file, discovered
from disk by a DEALER-based client. The one deliberate difference is that a policy
has no event loop of its own -- the serve coroutine is scheduled onto the
orchestrator's existing loop, so policy mutations happen on the same loop that makes
scheduling decisions. That is what makes the atomicity argument in :meth:`apply` hold.

Wire format -- a single cloudpickled frame each way:

    request:  (request_id, op, params)      op in {"get", "set"}
    reply:    (request_id, ok, payload)

Replies are correlated by the explicit ``request_id``, not by operation name. This is
the other deliberate departure from the actor layer: ``ActorHandle`` demuxes into
per-action FIFO queues and so mismatches replies under concurrent same-action calls.

Unlike ``PublicActor`` there is no ``msg_id`` in the envelope -- the endpoint runs with
``req_res=True``, and in that mode the connection's recv loop strips the id and ACKs
before handing the payload up.
"""

import asyncio
import secrets
from collections import OrderedDict
from logging import Logger
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple

import cloudpickle

from ensemble_launcher.comm.pipe import decode_identity, transport_registry

if TYPE_CHECKING:
    from ensemble_launcher.checkpointing import Checkpointer
    from ensemble_launcher.scheduler.policy import PolicyStateMixin

# Client identities must carry this prefix to be accepted by the endpoint.
POLICY_CLIENT_PREFIX = "policy-client-"

# Registered policy kinds.
KIND_TASK = "task"
KIND_CHILDREN = "children"

# How many request_id -> reply pairs to remember for retry suppression.
_REPLY_CACHE_MAX = 256


class PolicyEndpoint:
    """ROUTER endpoint exposing ``get``/``set`` over one node's policy state.

    Register the node's policy (or policies) with :meth:`register`, then
    :meth:`start` to bind and publish. The serve loop runs as a task on the
    caller's event loop.
    """

    def __init__(
        self,
        node_id: str,
        checkpointer: "Checkpointer",
        logger: Logger,
        transport: str = "zmq",
        send_timeout: float = 5.0,
        send_retries: int = 3,
    ) -> None:
        self._node_id = node_id
        self._checkpointer = checkpointer
        self.logger = logger
        self._transport_classes = transport_registry.get(transport)
        self._transport = None
        self._conn = None
        self._secret = secrets.token_hex(16)
        self._send_timeout = send_timeout
        self._send_retries = send_retries

        self._policies: Dict[
            str, Tuple["PolicyStateMixin", Optional[Callable[[], int]]]
        ] = {}
        self._version: int = 0
        self._serve_task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()
        self._started = False
        # request_id -> (ok, payload). The transport is at-least-once: a client whose
        # ACK is lost re-sends the same request_id, and replaying the original reply
        # keeps a retried `set` from bumping `version` a second time.
        self._replies: "OrderedDict[str, Tuple[bool, Any]]" = OrderedDict()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        kind: str,
        policy: "PolicyStateMixin",
        rescore_cb: Optional[Callable[[], int]] = None,
    ) -> None:
        """Expose *policy* under *kind* (``"task"`` or ``"children"``).

        Args:
            kind: Namespace clients address this policy by.
            policy: Any object mixing in ``PolicyStateMixin``.
            rescore_cb: Invoked when a client passes ``rescore=True``; returns the
                number of tasks re-scored. Workers pass
                ``AsyncTaskScheduler.reprioritize_pending``. Masters pass ``None``:
                the children-policy analogue would tear down and recreate child
                processes mid-flight, which is not something to trigger from a
                one-line client call. Must be synchronous -- see :meth:`apply`.
        """
        self._policies[kind] = (policy, rescore_cb)

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def version(self) -> int:
        """Incremented on every successful ``set``; lets callers skip no-op writes."""
        return self._version

    @property
    def address(self) -> Optional[str]:
        return self._conn.address if self._conn is not None else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Bind the ROUTER, publish its address, and spawn the serve task."""
        if self._started:
            return
        if not self._policies:
            self.logger.warning(
                f"PolicyEndpoint on {self._node_id} started with no policies registered"
            )

        self._transport = self._transport_classes["transport"]()
        # req_res=True, unlike PublicActor: the connection's recv loop (and with it
        # all sender bookkeeping) only runs in req_res mode, and it buys ACK-based
        # retry for replies.
        self._conn = self._transport.get_server_connection(
            f"{self._node_id}-policy",
            secrets.token_hex(16),
            req_res=True,
            address=None,
        )
        self._conn.set_unknown_sender_validator(self._validate_sender)
        await self._conn.open()

        self._checkpointer.write_policy_endpoint(
            self._conn.get_state().serialize(), self._secret
        )
        self._stop.clear()
        self._serve_task = asyncio.create_task(self._serve())
        self._started = True
        self.logger.info(
            f"PolicyEndpoint for {self._node_id} listening on {self._conn.address} "
            f"(policies: {sorted(self._policies)})"
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._stop.set()
        if self._serve_task is not None:
            self._serve_task.cancel()
            try:
                await self._serve_task
            except asyncio.CancelledError:
                pass
            self._serve_task = None
        if self._conn is not None:
            await self._conn.close()
        self._started = False
        self.logger.info(f"PolicyEndpoint for {self._node_id} stopped")

    def _validate_sender(self, sender_id: str, sender_secret: Optional[str]) -> bool:
        return (
            sender_id.startswith(POLICY_CLIENT_PREFIX)
            and sender_secret == self._secret
        )

    # ------------------------------------------------------------------
    # Serve loop
    # ------------------------------------------------------------------

    async def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                frames = await self._conn.recv(timeout=1.0)
            except (TimeoutError, asyncio.TimeoutError):
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger.warning(f"PolicyEndpoint recv failed: {e}")
                continue

            try:
                await self._handle(frames)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # A malformed request must never kill the serve task.
                self.logger.warning(f"PolicyEndpoint dropped a request: {e}")

    async def _handle(self, frames) -> None:
        sender_full_id, sender_id, sender_secret = decode_identity(frames[0])

        # ServerConnection.verify_sender short-circuits to True when no expected
        # remotes are configured, which is exactly this endpoint's situation, so the
        # validator above never runs. Enforce the secret here instead.
        if not self._validate_sender(sender_id, sender_secret):
            self.logger.warning(
                f"PolicyEndpoint on {self._node_id}: rejecting request from "
                f"{sender_id} (bad or missing policy secret)"
            )
            return

        request_id, op, params = cloudpickle.loads(frames[-1])

        cached = self._replies.get(request_id)
        if cached is not None:
            ok, payload = cached
        else:
            try:
                payload = self.apply(op, params)
                ok = True
            except Exception as e:
                payload = f"{type(e).__name__}: {e}"
                ok = False
            self._replies[request_id] = (ok, payload)
            while len(self._replies) > _REPLY_CACHE_MAX:
                self._replies.popitem(last=False)

        reply = cloudpickle.dumps((request_id, ok, payload))
        for attempt in range(self._send_retries):
            # send() does not retry internally -- it returns False on ACK timeout.
            if await self._conn.send(
                reply, target_id=sender_full_id, timeout=self._send_timeout
            ):
                return
            self.logger.warning(
                f"PolicyEndpoint on {self._node_id}: reply to {sender_id} "
                f"not acked (attempt {attempt + 1}/{self._send_retries})"
            )
        self.logger.warning(
            f"PolicyEndpoint on {self._node_id}: giving up on reply to {sender_id}; "
            "the client will see a request timeout"
        )

    # ------------------------------------------------------------------
    # Request handling
    # ------------------------------------------------------------------

    def apply(self, op: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute one ``get``/``set`` against a registered policy.

        Pure in-memory and **fully synchronous** -- it contains no ``await``, so the
        event loop cannot interleave anything between the rollback snapshot and the
        final write. That is why no lock is needed: every policy call site in the
        schedulers runs synchronously through its consuming bookkeeping, so a state
        change can never land in the middle of a scheduling decision. Keep it that
        way; if ``on_state_update`` or ``rescore_cb`` ever became coroutines this
        reasoning would silently collapse.

        Returns:
            ``{node_id, policy_kind, policy_name, state, version, rescored}``.
            ``rescored`` is the number of tasks re-prioritized, or ``None`` when not
            requested or unsupported on this node.

        Raises:
            ValueError/TypeError: on an unknown op, unresolvable policy kind, or a
                malformed payload. ``_handle`` turns these into an error reply.
        """
        kind, policy, rescore_cb = self._resolve(params.get("policy_kind", "auto"))

        if op == "get":
            state = policy.get_policy_state(params.get("keys"))
            rescored = None
        elif op == "set":
            new_state = params.get("state")
            if not isinstance(new_state, dict):
                raise TypeError(
                    f"'state' must be a dict, got {type(new_state).__name__}. "
                    "Pass {'key': value}, not a bare value."
                )
            # Snapshot for rollback. set_policy_state and rescore_cb are two
            # mutations of one logical operation: poison state only surfaces inside
            # rescore_cb -> get_score, *after* the write. Leaving it in place would
            # wedge the node, since every later add_task scores against it too.
            prev = policy.get_policy_state()
            try:
                state = policy.set_policy_state(
                    new_state, merge=params.get("merge", True)
                )
                rescored = (
                    rescore_cb()
                    if params.get("rescore", False) and rescore_cb is not None
                    else None
                )
            except Exception:
                policy.set_policy_state(prev, merge=False)
                self.logger.warning(
                    f"PolicyEndpoint on {self._node_id}: rolled back a failed "
                    f"set on the '{kind}' policy"
                )
                raise
            self._version += 1
        else:
            raise ValueError(f"unknown op {op!r}; expected 'get' or 'set'")

        return {
            "node_id": self._node_id,
            "policy_kind": kind,
            "policy_name": type(policy).__name__,
            "state": state,
            "version": self._version,
            "rescored": rescored,
        }

    def _resolve(
        self, kind: str
    ) -> Tuple[str, "PolicyStateMixin", Optional[Callable[[], int]]]:
        """Pick the target policy. A mismatch is an error, never a silent redirect."""
        if kind == "auto":
            if not self._policies:
                raise ValueError(f"no policy registered on {self._node_id}")
            if len(self._policies) > 1:
                raise ValueError(
                    f"{self._node_id} hosts multiple policies "
                    f"({sorted(self._policies)}); pass policy_kind explicitly"
                )
            kind = next(iter(self._policies))
        if kind not in self._policies:
            raise ValueError(
                f"{self._node_id} hosts no '{kind}' policy "
                f"(has: {sorted(self._policies)})"
            )
        policy, rescore_cb = self._policies[kind]
        return kind, policy, rescore_cb

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        """``{kind: state}`` for the checkpointer."""
        return {
            kind: policy.get_policy_state()
            for kind, (policy, _cb) in self._policies.items()
        }

    def restore(self, states: Optional[Dict[str, Dict[str, Any]]]) -> None:
        """Re-apply checkpointed state after a restart.

        Replaces rather than merges: the checkpoint is a full snapshot, and merging
        would resurrect keys a client had explicitly removed before the crash.
        """
        if not states:
            return
        for kind, state in states.items():
            entry = self._policies.get(kind)
            if entry is None or not isinstance(state, dict):
                continue
            try:
                entry[0].set_policy_state(state, merge=False)
            except Exception as e:
                self.logger.warning(
                    f"PolicyEndpoint on {self._node_id}: could not restore "
                    f"'{kind}' policy state: {e}"
                )
                continue
            self.logger.info(
                f"PolicyEndpoint on {self._node_id}: restored '{kind}' policy state"
            )
