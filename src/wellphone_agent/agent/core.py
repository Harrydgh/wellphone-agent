from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from ..actions import ActionController
from ..perception.state import PageState
from ..runlog import RunLogger

ActionKind = Literal["tap", "scroll_down", "back", "finish", "abort"]
RiskLevel = Literal["low", "medium", "high"]


class AgentError(RuntimeError):
    """The agent could not safely complete its goal."""


@dataclass(frozen=True)
class AgentGoal:
    description: str
    allowed_package: str
    target_label: str
    success_activity_contains: tuple[str, ...]
    success_text_contains: tuple[str, ...] = ()
    additional_allowed_packages: tuple[str, ...] = ()
    allowed_tap_targets: tuple[str, ...] = ()
    risk_level: RiskLevel = "low"
    requires_confirmation: bool = False

    @property
    def tap_targets(self) -> tuple[str, ...]:
        return self.allowed_tap_targets or (self.target_label,)

    @property
    def package_scope(self) -> tuple[str, ...]:
        return (self.allowed_package, *self.additional_allowed_packages)

    def clickable_candidates(self, state: PageState) -> tuple[str, ...]:
        return tuple(
            target
            for target in self.tap_targets
            if state.clickable_center(target) is not None
        )

    def is_satisfied(self, state: PageState) -> bool:
        activity = state.current_activity or ""
        activity_matches = any(
            marker in activity for marker in self.success_activity_contains
        )
        text_matches = not self.success_text_contains or any(
            marker in state.visible_text for marker in self.success_text_contains
        )
        return (
            state.current_app in self.package_scope
            and activity_matches
            and text_matches
        )


@dataclass(frozen=True)
class AgentAction:
    kind: ActionKind
    reason: str
    target: str | None = None


@dataclass(frozen=True)
class AgentTransition:
    step: int
    action: AgentAction
    before_activity: str | None
    after_activity: str | None
    page_changed: bool


@dataclass(frozen=True)
class AgentResult:
    success: bool
    reason: str
    steps: int
    final_state: PageState
    history: tuple[AgentTransition, ...]


class AgentPlanner(Protocol):
    def plan(
        self,
        goal: AgentGoal,
        state: PageState,
        history: Sequence[AgentTransition],
    ) -> AgentAction: ...


class PageObserver(Protocol):
    def observe(self, label: str = "frame") -> PageState: ...


class AgentSafetyPolicy:
    """Validate planner output against the current screen and task scope."""

    BLOCKED_TARGET_TERMS = (
        "支付",
        "付款",
        "删除",
        "卸载",
        "恢复出厂",
        "账号",
        "密码",
        "权限",
    )

    def validate(
        self, goal: AgentGoal, state: PageState, action: AgentAction
    ) -> None:
        if state.display_id <= 0:
            raise AgentError("Agent refused to act on the primary display.")
        if state.current_app not in goal.package_scope:
            raise AgentError(
                f"Agent left its allowed app: {state.current_app or 'unknown'}."
            )
        if any(
            term in target
            for target in goal.tap_targets
            for term in self.BLOCKED_TARGET_TERMS
        ):
            raise AgentError("The task contains a target blocked by the safety policy.")
        if action.kind == "tap":
            if action.target not in goal.tap_targets:
                raise AgentError("Planner attempted to tap outside the requested target.")
            if state.clickable_center(action.target) is None:
                raise AgentError(f"Target is not safely clickable: {action.target}")
        elif action.kind not in {"scroll_down", "back", "finish", "abort"}:
            raise AgentError(f"Unsupported agent action: {action.kind}")
        if action.kind == "finish" and not goal.is_satisfied(state):
            raise AgentError("Planner claimed completion before the goal was verified.")


class AgentLoop:
    """Observe, plan, validate, execute and verify until a goal terminates."""

    def __init__(
        self,
        observer: PageObserver,
        actions: ActionController,
        planner: AgentPlanner,
        *,
        policy: AgentSafetyPolicy | None = None,
        logger: RunLogger | None = None,
        max_steps: int = 6,
        max_unchanged_actions: int = 2,
        settle_seconds: float = 1.5,
    ) -> None:
        if max_steps <= 0 or max_unchanged_actions <= 0:
            raise ValueError("Agent limits must be positive.")
        self.observer = observer
        self.actions = actions
        self.planner = planner
        self.policy = policy or AgentSafetyPolicy()
        self.logger = logger
        self.max_steps = max_steps
        self.max_unchanged_actions = max_unchanged_actions
        self.settle_seconds = settle_seconds

    @staticmethod
    def _page_changed(before: PageState, after: PageState) -> bool:
        return (
            before.current_activity != after.current_activity
            or before.visible_text != after.visible_text
            or before.perceptual_hash != after.perceptual_hash
        )

    def _log(self, event: str, **data: object) -> None:
        if self.logger is not None:
            self.logger.write(event, **data)

    def _execute(self, state: PageState, action: AgentAction) -> None:
        if action.kind == "tap":
            assert action.target is not None
            center = state.clickable_center(action.target)
            if center is None:
                raise AgentError(f"Target disappeared before execution: {action.target}")
            self.actions.tap(*center)
        elif action.kind == "scroll_down":
            x = state.width // 2
            self.actions.swipe(x, state.height * 3 // 4, x, state.height // 3, 500)
        elif action.kind == "back":
            self.actions.keyevent("KEYCODE_BACK")

    def run(self, goal: AgentGoal) -> AgentResult:
        history: list[AgentTransition] = []
        unchanged_actions = 0
        state = self.observer.observe("agent-step-0")
        self._log("agent_started", goal=goal.description, max_steps=self.max_steps)
        self._log("agent_observed", step=0, **state.to_dict())

        for step in range(1, self.max_steps + 1):
            if state.display_id != self.actions.display_id:
                raise AgentError(
                    "Observed display does not match the controller's virtual display."
                )
            if goal.is_satisfied(state):
                action = AgentAction("finish", "目标页面已经由本地成功条件验证。")
            else:
                action = self.planner.plan(goal, state, tuple(history))
            self.policy.validate(goal, state, action)
            self._log(
                "agent_decision",
                step=step,
                action=action.kind,
                target=action.target,
                reason=action.reason,
            )

            if action.kind in {"finish", "abort"}:
                success = action.kind == "finish"
                self._log(
                    "agent_completed" if success else "agent_aborted",
                    success=success,
                    step=step - 1,
                    reason=action.reason,
                )
                return AgentResult(
                    success,
                    action.reason,
                    step - 1,
                    state,
                    tuple(history),
                )

            self._execute(state, action)
            if self.settle_seconds > 0:
                time.sleep(self.settle_seconds)
            next_state = self.observer.observe(f"agent-step-{step}")
            changed = self._page_changed(state, next_state)
            transition = AgentTransition(
                step,
                action,
                state.current_activity,
                next_state.current_activity,
                changed,
            )
            history.append(transition)
            self._log(
                "agent_action_verified",
                step=step,
                action=action.kind,
                target=action.target,
                page_changed=changed,
                before_activity=state.current_activity,
                after_activity=next_state.current_activity,
            )
            self._log("agent_observed", step=step, **next_state.to_dict())
            unchanged_actions = 0 if changed else unchanged_actions + 1
            if unchanged_actions >= self.max_unchanged_actions:
                reason = "Agent stopped after repeated actions produced no page change."
                self._log("agent_failed", step=step, reason=reason)
                return AgentResult(False, reason, step, next_state, tuple(history))
            state = next_state

        reason = f"Agent reached the {self.max_steps}-step safety limit."
        self._log("agent_failed", step=self.max_steps, reason=reason)
        return AgentResult(False, reason, self.max_steps, state, tuple(history))
