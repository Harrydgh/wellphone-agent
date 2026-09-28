from __future__ import annotations

import time
import uuid
from typing import Callable, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..actions import ActionController
from ..perception.state import PageState
from ..runlog import RunLogger
from .core import (
    AgentAction,
    AgentError,
    AgentGoal,
    AgentPlanner,
    AgentResult,
    AgentSafetyPolicy,
    AgentTransition,
    PageObserver,
)


class PhoneAgentState(TypedDict):
    goal: AgentGoal
    page: PageState | None
    action: AgentAction | None
    history: tuple[AgentTransition, ...]
    step: int
    unchanged_actions: int
    status: Literal["running", "succeeded", "failed", "aborted"]
    reason: str
    result: AgentResult | None


ApprovalHandler = Callable[[dict[str, object]], bool]


class AgentApprovalRequired(AgentError):
    """The graph paused before a task-scoped action awaiting user approval."""

    def __init__(self, thread_id: str, request: dict[str, object]) -> None:
        super().__init__("Agent paused and requires user approval before continuing.")
        self.thread_id = thread_id
        self.request = request


class LangGraphAgentLoop:
    """Stateful LangGraph orchestration around the existing safe device kernel."""

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
        observation_retries: int = 2,
        observation_retry_seconds: float = 0.5,
        checkpointer: InMemorySaver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        if max_steps <= 0 or max_unchanged_actions <= 0 or observation_retries < 0:
            raise ValueError("Agent limits must be positive.")
        self.observer = observer
        self.actions = actions
        self.planner = planner
        self.policy = policy or AgentSafetyPolicy()
        self.logger = logger
        self.max_steps = max_steps
        self.max_unchanged_actions = max_unchanged_actions
        self.settle_seconds = settle_seconds
        self.observation_retries = observation_retries
        self.observation_retry_seconds = observation_retry_seconds
        self.checkpointer = checkpointer or InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("wellphone_agent.agent.core", "AgentAction"),
                    ("wellphone_agent.agent.core", "AgentGoal"),
                    ("wellphone_agent.agent.core", "AgentResult"),
                    ("wellphone_agent.agent.core", "AgentTransition"),
                    ("wellphone_agent.perception.state", "PageState"),
                    ("wellphone_agent.perception.state", "VisibleElement"),
                ]
            )
        )
        self.approval_handler = approval_handler
        self.graph = self._build_graph()

    def _log(self, event: str, **data: object) -> None:
        if self.logger is not None:
            self.logger.write(event, engine="langgraph", **data)

    @staticmethod
    def _page_changed(before: PageState, after: PageState) -> bool:
        return (
            before.current_activity != after.current_activity
            or before.visible_text != after.visible_text
            or before.perceptual_hash != after.perceptual_hash
        )

    def _observe(self, state: PhoneAgentState) -> dict[str, object]:
        page = self._observe_stable(state["goal"], "langgraph-step-0")
        self._log("agent_observed", step=0, **page.to_dict())
        return {"page": page}

    def _observe_stable(self, goal: AgentGoal, label: str) -> PageState:
        transient_sources = {
            "uiautomator_unavailable",
            "uiautomator_invalid_xml",
            "uiautomator_package_mismatch",
        }
        page = self.observer.observe(label)
        for attempt in range(1, self.observation_retries + 1):
            if not (
                page.current_app in goal.package_scope
                and page.text_source in transient_sources
            ):
                return page
            self._log(
                "agent_observation_retry",
                attempt=attempt,
                text_source=page.text_source,
            )
            if self.observation_retry_seconds > 0:
                time.sleep(self.observation_retry_seconds)
            page = self.observer.observe(f"{label}-retry-{attempt}")
        if (
            page.current_app in goal.package_scope
            and page.text_source in transient_sources
        ):
            raise AgentError(
                "页面结构暂时无法可靠读取，Agent 已在执行动作前安全停止。"
            )
        return page

    def _plan(self, state: PhoneAgentState) -> dict[str, object]:
        page = state["page"]
        assert page is not None
        if page.display_id != self.actions.display_id:
            raise AgentError(
                "Observed display does not match the controller's virtual display."
            )
        goal = state["goal"]
        if goal.is_satisfied(page):
            action = AgentAction("finish", "目标页面已经由本地成功条件验证。")
        elif state["step"] >= self.max_steps:
            reason = f"Agent reached the {self.max_steps}-step safety limit."
            result = AgentResult(
                False, reason, state["step"], page, state["history"]
            )
            return {"status": "failed", "reason": reason, "result": result}
        else:
            action = self.planner.plan(goal, page, state["history"])
        self._log(
            "agent_decision",
            step=state["step"] + 1,
            action=action.kind,
            target=action.target,
            reason=action.reason,
        )
        return {"action": action}

    @staticmethod
    def _after_plan(state: PhoneAgentState) -> str:
        return "finalize" if state["status"] == "failed" else "validate"

    def _validate(self, state: PhoneAgentState) -> dict[str, object]:
        page = state["page"]
        action = state["action"]
        assert page is not None and action is not None
        self.policy.validate(state["goal"], page, action)
        return {}

    @staticmethod
    def _after_validate(state: PhoneAgentState) -> str:
        action = state["action"]
        assert action is not None
        if action.kind in {"finish", "abort"}:
            return "finalize"
        if state["goal"].requires_confirmation and action.kind == "tap":
            return "approve"
        return "execute"

    def _approve(self, state: PhoneAgentState) -> dict[str, object]:
        action = state["action"]
        assert action is not None
        request = {
            "type": "action_approval",
            "task": state["goal"].description,
            "risk_level": state["goal"].risk_level,
            "action": action.kind,
            "target": action.target,
            "reason": action.reason,
        }
        response = interrupt(request)
        approved = (
            bool(response.get("approved"))
            if isinstance(response, dict)
            else bool(response)
        )
        self._log(
            "agent_approval_resolved",
            approved=approved,
            action=action.kind,
            target=action.target,
        )
        if approved:
            return {}
        return {"action": AgentAction("abort", "用户拒绝了需要确认的动作。")}

    @staticmethod
    def _after_approve(state: PhoneAgentState) -> str:
        action = state["action"]
        assert action is not None
        return "finalize" if action.kind == "abort" else "execute"

    def _execute(self, state: PhoneAgentState) -> dict[str, object]:
        page = state["page"]
        action = state["action"]
        assert page is not None and action is not None
        if action.kind == "tap":
            assert action.target is not None
            center = page.clickable_center(action.target)
            if center is None:
                raise AgentError(f"Target disappeared before execution: {action.target}")
            self.actions.tap(*center)
        elif action.kind == "scroll_down":
            x = page.width // 2
            self.actions.swipe(x, page.height * 3 // 4, x, page.height // 3, 500)
        elif action.kind == "back":
            self.actions.keyevent("KEYCODE_BACK")
        if self.settle_seconds > 0:
            time.sleep(self.settle_seconds)
        return {}

    def _verify(self, state: PhoneAgentState) -> dict[str, object]:
        before = state["page"]
        action = state["action"]
        assert before is not None and action is not None
        step = state["step"] + 1
        after = self._observe_stable(
            state["goal"], f"langgraph-step-{step}"
        )
        changed = self._page_changed(before, after)
        transition = AgentTransition(
            step,
            action,
            before.current_activity,
            after.current_activity,
            changed,
        )
        history = (*state["history"], transition)
        unchanged = 0 if changed else state["unchanged_actions"] + 1
        self._log(
            "agent_action_verified",
            step=step,
            action=action.kind,
            target=action.target,
            page_changed=changed,
            before_activity=before.current_activity,
            after_activity=after.current_activity,
        )
        self._log("agent_observed", step=step, **after.to_dict())
        update: dict[str, object] = {
            "page": after,
            "history": history,
            "step": step,
            "unchanged_actions": unchanged,
            "action": None,
        }
        if unchanged >= self.max_unchanged_actions:
            reason = "Agent stopped after repeated actions produced no page change."
            update.update(
                status="failed",
                reason=reason,
                result=AgentResult(False, reason, step, after, history),
            )
        return update

    @staticmethod
    def _after_verify(state: PhoneAgentState) -> str:
        return "finalize" if state["status"] == "failed" else "plan"

    def _finalize(self, state: PhoneAgentState) -> dict[str, object]:
        if state["result"] is not None:
            result = state["result"]
            self._log("agent_failed", step=result.steps, reason=result.reason)
            return {}
        page = state["page"]
        action = state["action"]
        assert page is not None and action is not None
        success = action.kind == "finish"
        status: Literal["succeeded", "aborted"] = (
            "succeeded" if success else "aborted"
        )
        result = AgentResult(
            success,
            action.reason,
            state["step"],
            page,
            state["history"],
        )
        self._log(
            "agent_completed" if success else "agent_aborted",
            success=success,
            step=state["step"],
            reason=action.reason,
        )
        return {
            "status": status,
            "reason": action.reason,
            "result": result,
        }

    def _build_graph(self):
        workflow = StateGraph(PhoneAgentState)
        workflow.add_node("observe", self._observe)
        workflow.add_node("plan", self._plan)
        workflow.add_node("validate", self._validate)
        workflow.add_node("execute", self._execute)
        workflow.add_node("approve", self._approve)
        workflow.add_node("verify", self._verify)
        workflow.add_node("finalize", self._finalize)
        workflow.add_edge(START, "observe")
        workflow.add_edge("observe", "plan")
        workflow.add_conditional_edges(
            "plan", self._after_plan, {"validate": "validate", "finalize": "finalize"}
        )
        workflow.add_conditional_edges(
            "validate",
            self._after_validate,
            {"approve": "approve", "execute": "execute", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "approve",
            self._after_approve,
            {"execute": "execute", "finalize": "finalize"},
        )
        workflow.add_edge("execute", "verify")
        workflow.add_conditional_edges(
            "verify", self._after_verify, {"plan": "plan", "finalize": "finalize"}
        )
        workflow.add_edge("finalize", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def _invoke_until_result(
        self,
        graph_input: PhoneAgentState | Command,
        *,
        thread_id: str,
    ) -> AgentResult:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.max_steps * 6 + 24,
        }
        final = self.graph.invoke(graph_input, config=config)
        while "__interrupt__" in final:
            item = final["__interrupt__"][0]
            request = item.value
            if not isinstance(request, dict):
                raise AgentError("LangGraph returned an invalid approval request.")
            self._log("agent_approval_required", **request)
            if self.approval_handler is None:
                raise AgentApprovalRequired(thread_id, request)
            approved = self.approval_handler(request)
            final = self.graph.invoke(
                Command(resume={"approved": approved}), config=config
            )
        result = final.get("result")
        if not isinstance(result, AgentResult):
            raise AgentError("LangGraph Agent ended without a valid result.")
        return result

    def run(self, goal: AgentGoal, *, thread_id: str | None = None) -> AgentResult:
        run_id = thread_id or str(uuid.uuid4())
        self._log("agent_started", goal=goal.description, max_steps=self.max_steps)
        initial: PhoneAgentState = {
            "goal": goal,
            "page": None,
            "action": None,
            "history": (),
            "step": 0,
            "unchanged_actions": 0,
            "status": "running",
            "reason": "",
            "result": None,
        }
        return self._invoke_until_result(initial, thread_id=run_id)

    def resume(self, thread_id: str, *, approved: bool) -> AgentResult:
        return self._invoke_until_result(
            Command(resume={"approved": approved}), thread_id=thread_id
        )
