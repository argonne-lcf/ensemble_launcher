"""Tunable policies used by ``test_policy_client.py``.

These live in their own module rather than in the test file because orchestrator
nodes are spawned (``conftest.py`` forces the ``spawn`` start method), so the child
process never imports the test module. The child picks these up through the
documented external-policy hook instead: ``EL_EXTERNAL_POLICY_PATH`` +
``EL_EXTERNAL_POLICY_MODULE``, which ``ensemble_launcher.orchestrator`` honours at
import time. The test module imports this file directly so the parent has them too.
"""

from ensemble_launcher.scheduler.policy import (
    Policy,
    SimpleSplitChildrenPolicy,
    policy_registry,
)


@policy_registry.register("tunable_policy")
class TunableScorePolicy(Policy):
    """Scores by ``nnodes`` scaled by a tunable weight.

    ``state["boom"]`` makes ``get_score`` raise, which is how the rollback and
    fail-atomicity tests inject a poison tune.
    """

    def get_score(self, task, scheduler_state=None) -> float:
        if self.state.get("boom"):
            raise RuntimeError("poison state: get_score refuses to score")
        return float(task.nnodes) * float(self.state.get("node_weight", 1.0))


@policy_registry.register("tunable_index_policy")
class TunableIndexPolicy(Policy):
    """Scores from a tunable per-task table, so a test can dictate exact ordering.

    ``state["scores"]`` maps task_id -> score; unlisted tasks all score
    ``state["default"]`` (0.0), which makes them tie and so exposes whether the
    FIFO tiebreak survived a rescore.
    """

    def get_score(self, task, scheduler_state=None) -> float:
        scores = self.state.get("scores") or {}
        return float(scores.get(task.task_id, self.state.get("default", 0.0)))


@policy_registry.register("tunable_children_policy", type="children_policy")
class TunableSplitPolicy(SimpleSplitChildrenPolicy):
    """A children policy whose only purpose is to carry externally-set state."""

    def on_state_update(self, changed) -> None:
        # Records that the hook fired, so a test can assert it ran on the node.
        self.state["_updates"] = self.state.get("_updates", 0) + 1
