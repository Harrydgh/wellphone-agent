from __future__ import annotations

import json
from typing import Any, Sequence

from ..perception.state import PageState
from .core import AgentAction, AgentError, AgentGoal, AgentTransition


class OpenAIPlanner:
    """Use Structured Outputs to select one locally validated agent action."""

    ACTION_SCHEMA = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tap", "scroll_down", "abort"],
            },
            "target": {
                "anyOf": [{"type": "string"}, {"type": "null"}],
            },
            "reason": {"type": "string"},
        },
        "required": ["action", "target", "reason"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        model: str,
        client: Any | None = None,
        max_scrolls: int = 3,
    ) -> None:
        if not model.strip():
            raise ValueError("An OpenAI model name is required.")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise AgentError(
                    "OpenAI SDK is not installed. Run scripts/setup.ps1 first."
                ) from exc
            client = OpenAI(timeout=30.0, max_retries=1)
        self.model = model
        self.client = client
        self.max_scrolls = max_scrolls

    def plan(
        self,
        goal: AgentGoal,
        state: PageState,
        history: Sequence[AgentTransition],
    ) -> AgentAction:
        if state.current_app not in goal.package_scope:
            return AgentAction("abort", "当前页面已离开任务允许的应用。")
        scroll_count = sum(item.action.kind == "scroll_down" for item in history)
        candidates = goal.clickable_candidates(state)
        if not candidates and scroll_count >= self.max_scrolls:
            return AgentAction("abort", "达到安全浏览上限后仍未发现目标。")

        # Deliberately omit screenshots and unrelated visible text. The model
        # receives only the minimum state needed to choose an allowed action.
        state_payload = {
            "task": goal.description,
            "current_app": state.current_app,
            "current_activity": state.current_activity,
            "target_label": goal.target_label,
            "clickable_candidates": candidates,
            "scroll_count": scroll_count,
            "max_scrolls": self.max_scrolls,
            "allowed_actions": ["tap", "scroll_down", "abort"],
        }
        try:
            response = self.client.responses.create(
                model=self.model,
                store=False,
                instructions=(
                    "你是安卓虚拟屏的受限动作规划器。只选择一个动作。"
                    "clickable_candidates 非空时选择 tap，且 target 必须来自该列表；"
                    "列表为空时可以选择 scroll_down；无法安全继续时选择 abort。"
                    "不要声称已执行动作，不要生成坐标。"
                ),
                input=json.dumps(state_payload, ensure_ascii=False),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "wellphone_agent_action",
                        "strict": True,
                        "schema": self.ACTION_SCHEMA,
                    }
                },
            )
            output = json.loads(response.output_text)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            if status_code == 401:
                detail = "OpenAI API 凭据无效，请更新 OPENAI_API_KEY。"
            elif status_code == 429:
                detail = "OpenAI API 当前无可用额度或已达到调用限制。"
            else:
                detail = "OpenAI API 调用失败，请检查网络、模型权限和账户状态。"
            raise AgentError(detail) from exc

        action = output["action"]
        target = output["target"]
        reason = str(output["reason"])[:300]
        if action == "tap" and target not in candidates:
            raise AgentError("AI planner returned a tap target outside the approved goal.")
        if action != "tap":
            target = None
        return AgentAction(action, reason, target)
