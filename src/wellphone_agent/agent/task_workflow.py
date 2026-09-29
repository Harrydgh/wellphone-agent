from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Callable, Literal, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from ..actions import ActionController
from ..perception.candidates import CandidateAction
from ..perception.state import PageState
from ..perception.understanding import PageUnderstanding, PageUnderstandingEngine
from ..runlog import RunLogger
from .core import AgentError, PageObserver
from .task_planner import TaskAction, TaskPlanner, TaskTransition
from .tasks import AppTask


@dataclass(frozen=True)
class TaskRunResult:
    success: bool
    reason: str
    steps: int
    final_understanding: PageUnderstanding
    history: tuple[TaskTransition, ...]


class AppTaskState(TypedDict):
    task: AppTask
    page: PageState | None
    understanding: PageUnderstanding | None
    action: TaskAction | None
    history: tuple[TaskTransition, ...]
    step: int
    unchanged_actions: int
    status: Literal["running", "succeeded", "failed", "aborted"]
    reason: str
    result: TaskRunResult | None


ApprovalHandler = Callable[[dict[str, object]], bool]
PreActionGuard = Callable[[], None]


class TaskApprovalRequired(AgentError):
    def __init__(self, thread_id: str, request: dict[str, object]) -> None:
        super().__init__("任务在执行受控动作前等待用户确认。")
        self.thread_id = thread_id
        self.request = request


class AppTaskSafetyPolicy:
    def candidate_for(
        self,
        task: AppTask,
        understanding: PageUnderstanding,
        action: TaskAction,
    ) -> CandidateAction | None:
        if action.kind != "tap_candidate":
            return None
        return next(
            (
                item
                for item in task.candidates(understanding)
                if item.candidate_id == action.candidate_id
            ),
            None,
        )

    def validate(
        self,
        task: AppTask,
        page: PageState,
        understanding: PageUnderstanding,
        history: tuple[TaskTransition, ...],
        action: TaskAction,
    ) -> CandidateAction | None:
        if page.display_id <= 0:
            raise AgentError("任务拒绝控制手机主显示。")
        if understanding.current_app != task.allowed_package:
            raise AgentError("任务已离开允许的美团 App。")
        if understanding.screen.kind in {
            "checkout",
            "payment",
            "verification",
            "login",
            "permission",
        }:
            if action.kind not in {"abort", "back"}:
                raise AgentError("敏感页面只允许停止或返回。")

        candidate = self.candidate_for(task, understanding, action)
        if action.kind == "tap_candidate":
            if candidate is None:
                raise AgentError("候选控件已经消失或不属于当前任务。")
            if candidate.requires_user_takeover:
                raise AgentError("该动作必须由用户接管，Agent 不会执行。")
            if candidate.source == "ocr" and candidate.confidence < 0.9:
                raise AgentError("OCR 候选置信度不足，拒绝使用其坐标。")
            if not candidate.executable:
                raise AgentError("候选仅用于页面理解，缺少可验证的可点击控件。")
        elif action.kind == "input_query":
            activity = understanding.current_activity or ""
            focused_by_workflow = bool(
                history and history[-1].target_label == "搜索框"
            )
            if (
                not focused_by_workflow
                and understanding.screen.kind != "search"
                and "Search" not in activity
            ):
                raise AgentError("当前页面未被验证为搜索页面，拒绝输入文字。")
        elif action.kind == "finish":
            completed = tuple(item.target_label or "" for item in history)
            if not task.is_satisfied(understanding, completed):
                raise AgentError("任务尚未到达购物车停止条件。")
        elif action.kind not in {"scroll_down", "wait", "back", "abort"}:
            raise AgentError(f"不支持的任务动作：{action.kind}")
        return candidate


class LangGraphAppTaskLoop:
    """Grounded observe-plan-confirm-execute-verify loop for complex apps."""

    def __init__(
        self,
        observer: PageObserver,
        understanding_engine: PageUnderstandingEngine,
        actions: ActionController,
        planner: TaskPlanner,
        *,
        policy: AppTaskSafetyPolicy | None = None,
        logger: RunLogger | None = None,
        settle_seconds: float = 1.5,
        max_unchanged_actions: int = 2,
        approval_handler: ApprovalHandler | None = None,
        pre_action_guard: PreActionGuard | None = None,
        checkpointer: InMemorySaver | None = None,
    ) -> None:
        if max_unchanged_actions <= 0 or settle_seconds < 0:
            raise ValueError("Task loop limits are invalid.")
        self.observer = observer
        self.understanding_engine = understanding_engine
        self.actions = actions
        self.planner = planner
        self.policy = policy or AppTaskSafetyPolicy()
        self.logger = logger
        self.settle_seconds = settle_seconds
        self.max_unchanged_actions = max_unchanged_actions
        self.approval_handler = approval_handler
        self.pre_action_guard = pre_action_guard
        self.checkpointer = checkpointer or InMemorySaver(
            serde=JsonPlusSerializer(
                allowed_msgpack_modules=[
                    ("wellphone_agent.agent.tasks", "AppTask"),
                    ("wellphone_agent.agent.task_planner", "TaskAction"),
                    ("wellphone_agent.agent.task_planner", "TaskTransition"),
                    ("wellphone_agent.agent.task_workflow", "TaskRunResult"),
                    ("wellphone_agent.perception.state", "PageState"),
                    ("wellphone_agent.perception.state", "VisibleElement"),
                    ("wellphone_agent.perception.understanding", "PageUnderstanding"),
                    ("wellphone_agent.perception.fusion", "PerceivedElement"),
                    ("wellphone_agent.perception.screen_classifier", "ScreenClassification"),
                    ("wellphone_agent.perception.candidates", "CandidateAction"),
                ]
            )
        )
        self.graph = self._build_graph()

    def _log(self, event: str, **data: object) -> None:
        if self.logger is not None:
            self.logger.write(event, engine="langgraph_app_task", **data)

    def _observe_page(
        self, task: AppTask, label: str
    ) -> tuple[PageState, PageUnderstanding]:
        page = self.observer.observe(label)
        understanding = self.understanding_engine.analyze(page)
        self._log(
            "task_observed",
            screenshot=understanding.screenshot,
            text_source=understanding.text_source,
            screen=understanding.screen.to_dict(),
            task_candidates=[item.to_dict() for item in task.candidates(understanding)],
        )
        return page, understanding

    def _observe(self, state: AppTaskState) -> dict[str, object]:
        page, understanding = self._observe_page(state["task"], "task-step-0")
        return {"page": page, "understanding": understanding}

    def _plan(self, state: AppTaskState) -> dict[str, object]:
        understanding = state["understanding"]
        assert understanding is not None
        task = state["task"]
        completed = tuple(item.target_label or "" for item in state["history"])
        if task.is_satisfied(understanding, completed):
            reason = (
                "检测到该商品已在购物车，未重复添加。"
                if not any("加入购物车" in label for label in completed)
                else "商品已加入购物车并验证到购物车状态。"
            )
            action = TaskAction(kind="finish", reason=reason)
        elif state["step"] >= task.max_steps:
            reason = f"任务达到 {task.max_steps} 步安全上限。"
            result = TaskRunResult(
                False, reason, state["step"], understanding, state["history"]
            )
            return {"status": "failed", "reason": reason, "result": result}
        else:
            action = self.planner.plan(task, understanding, state["history"])
        self._log(
            "task_decision",
            step=state["step"] + 1,
            action=action.kind,
            candidate_id=action.candidate_id,
            reason=action.reason,
        )
        return {"action": action}

    @staticmethod
    def _after_plan(state: AppTaskState) -> str:
        return "finalize" if state["status"] == "failed" else "validate"

    def _validate(self, state: AppTaskState) -> dict[str, object]:
        page = state["page"]
        understanding = state["understanding"]
        action = state["action"]
        assert page is not None and understanding is not None and action is not None
        self.policy.validate(
            state["task"], page, understanding, state["history"], action
        )
        return {}

    def _needs_approval(self, state: AppTaskState) -> str:
        action = state["action"]
        understanding = state["understanding"]
        assert action is not None and understanding is not None
        if action.kind in {"finish", "abort"}:
            return "finalize"
        candidate = self.policy.candidate_for(state["task"], understanding, action)
        if candidate and (candidate.requires_confirmation or not candidate.executable):
            return "approve"
        return "execute"

    def _approve(self, state: AppTaskState) -> dict[str, object]:
        action = state["action"]
        understanding = state["understanding"]
        assert action is not None and understanding is not None
        candidate = self.policy.candidate_for(state["task"], understanding, action)
        assert candidate is not None
        request = {
            "type": "grounded_action_approval",
            "task": state["task"].description,
            "action": action.kind,
            "target": candidate.label,
            "source": candidate.source,
            "risk_level": candidate.risk_level,
            "reason": action.reason,
        }
        response = interrupt(request)
        approved = bool(response.get("approved")) if isinstance(response, dict) else bool(response)
        self._log("task_approval_resolved", approved=approved, target=candidate.label)
        if approved:
            return {}
        return {"action": TaskAction(kind="abort", reason="用户拒绝了候选动作。")}

    @staticmethod
    def _after_approve(state: AppTaskState) -> str:
        action = state["action"]
        assert action is not None
        return "finalize" if action.kind == "abort" else "execute"

    def _execute(self, state: AppTaskState) -> dict[str, object]:
        action = state["action"]
        understanding = state["understanding"]
        page = state["page"]
        assert action is not None and understanding is not None and page is not None
        if self.pre_action_guard is not None:
            # Approval may take time. Recheck display isolation at the last possible
            # moment so no action is sent after the protected App reaches display 0.
            self.pre_action_guard()
        candidate = self.policy.candidate_for(state["task"], understanding, action)
        if candidate is not None and candidate.requires_confirmation:
            # A user may take minutes to approve. Never execute coordinates from
            # the pre-approval frame: re-observe and require the exact grounded
            # candidate to still exist on the same safe page.
            page, understanding = self._observe_page(
                state["task"], f"task-pre-action-{state['step'] + 1}"
            )
            candidate = self.policy.validate(
                state["task"],
                page,
                understanding,
                state["history"],
                action,
            )
            self._log(
                "task_pre_action_revalidated",
                step=state["step"] + 1,
                candidate_id=action.candidate_id,
                screen=understanding.screen.kind,
            )
        if action.kind == "tap_candidate":
            assert candidate is not None
            left, top, right, bottom = candidate.bounds
            self.actions.tap((left + right) // 2, (top + bottom) // 2)
        elif action.kind == "input_query":
            self.actions.replace_text(state["task"].query)
            self.actions.keyevent("KEYCODE_ENTER")
        elif action.kind == "scroll_down":
            self.actions.swipe(
                page.width // 2,
                page.height * 3 // 4,
                page.width // 2,
                page.height // 3,
                500,
            )
        elif action.kind == "back":
            self.actions.keyevent("KEYCODE_BACK")
        # "wait" intentionally sends no control message; the normal settle and
        # verify steps below only re-observe the virtual display.
        if self.settle_seconds:
            time.sleep(self.settle_seconds)
        return {}

    def _verify(self, state: AppTaskState) -> dict[str, object]:
        before_page = state["page"]
        before = state["understanding"]
        action = state["action"]
        assert before_page is not None and before is not None and action is not None
        candidate = self.policy.candidate_for(state["task"], before, action)
        step = state["step"] + 1
        page, after = self._observe_page(state["task"], f"task-step-{step}")
        changed = (
            before_page.current_activity != page.current_activity
            or before_page.perceptual_hash != page.perceptual_hash
            or before.screen.kind != after.screen.kind
        )
        transition = TaskTransition(
            step=step,
            action=action,
            target_label=candidate.label if candidate else None,
            before_screen=before.screen.kind,
            after_screen=after.screen.kind,
            page_changed=changed,
        )
        history = (*state["history"], transition)
        unchanged = 0 if changed else state["unchanged_actions"] + 1
        update: dict[str, object] = {
            "page": page,
            "understanding": after,
            "history": history,
            "step": step,
            "unchanged_actions": unchanged,
            "action": None,
        }
        if unchanged >= self.max_unchanged_actions:
            reason = "连续动作未产生页面变化，任务已安全停止。"
            update.update(
                status="failed",
                reason=reason,
                result=TaskRunResult(False, reason, step, after, history),
            )
        return update

    @staticmethod
    def _after_verify(state: AppTaskState) -> str:
        return "finalize" if state["status"] == "failed" else "plan"

    def _finalize(self, state: AppTaskState) -> dict[str, object]:
        if state["result"] is not None:
            return {}
        action = state["action"]
        understanding = state["understanding"]
        assert action is not None and understanding is not None
        success = action.kind == "finish"
        result = TaskRunResult(
            success,
            action.reason,
            state["step"],
            understanding,
            state["history"],
        )
        self._log(
            "task_completed" if success else "task_aborted",
            success=success,
            steps=state["step"],
            reason=action.reason,
        )
        return {
            "status": "succeeded" if success else "aborted",
            "reason": action.reason,
            "result": result,
        }

    def _build_graph(self):
        workflow = StateGraph(AppTaskState)
        workflow.add_node("observe", self._observe)
        workflow.add_node("plan", self._plan)
        workflow.add_node("validate", self._validate)
        workflow.add_node("approve", self._approve)
        workflow.add_node("execute", self._execute)
        workflow.add_node("verify", self._verify)
        workflow.add_node("finalize", self._finalize)
        workflow.add_edge(START, "observe")
        workflow.add_edge("observe", "plan")
        workflow.add_conditional_edges(
            "plan", self._after_plan, {"validate": "validate", "finalize": "finalize"}
        )
        workflow.add_conditional_edges(
            "validate",
            self._needs_approval,
            {"approve": "approve", "execute": "execute", "finalize": "finalize"},
        )
        workflow.add_conditional_edges(
            "approve", self._after_approve, {"execute": "execute", "finalize": "finalize"}
        )
        workflow.add_edge("execute", "verify")
        workflow.add_conditional_edges(
            "verify", self._after_verify, {"plan": "plan", "finalize": "finalize"}
        )
        workflow.add_edge("finalize", END)
        return workflow.compile(checkpointer=self.checkpointer)

    def _invoke(
        self, graph_input: AppTaskState | Command, *, thread_id: str
    ) -> TaskRunResult:
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 120,
        }
        final = self.graph.invoke(graph_input, config=config)
        while "__interrupt__" in final:
            request = final["__interrupt__"][0].value
            if not isinstance(request, dict):
                raise AgentError("LangGraph 返回了无效的确认请求。")
            self._log("task_approval_required", **request)
            if self.approval_handler is None:
                raise TaskApprovalRequired(thread_id, request)
            approved = self.approval_handler(request)
            final = self.graph.invoke(
                Command(resume={"approved": approved}), config=config
            )
        result = final.get("result")
        if not isinstance(result, TaskRunResult):
            raise AgentError("任务工作流未返回有效结果。")
        return result

    def run(self, task: AppTask, *, thread_id: str | None = None) -> TaskRunResult:
        run_id = thread_id or str(uuid.uuid4())
        initial: AppTaskState = {
            "task": task,
            "page": None,
            "understanding": None,
            "action": None,
            "history": (),
            "step": 0,
            "unchanged_actions": 0,
            "status": "running",
            "reason": "",
            "result": None,
        }
        self._log("task_started", task=task.description, package=task.allowed_package)
        return self._invoke(initial, thread_id=run_id)

    def resume(self, thread_id: str, *, approved: bool) -> TaskRunResult:
        return self._invoke(
            Command(resume={"approved": approved}), thread_id=thread_id
        )
