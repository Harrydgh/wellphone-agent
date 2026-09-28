from __future__ import annotations

import json
import os
from typing import Any, Literal, Sequence

from pydantic import BaseModel, Field

from ..perception.state import PageState
from .core import AgentAction, AgentError, AgentGoal, AgentTransition


class DeepSeekActionDecision(BaseModel):
    """The only action shape the remote planner is allowed to return."""

    action: Literal["tap", "scroll_down", "abort"]
    target: str | None = None
    reason: str = Field(
        default="DeepSeek 根据当前受限页面状态选择了该动作。",
        min_length=1,
        max_length=300,
    )


class DeepSeekPlanner:
    """Select one constrained action through LangChain's DeepSeek integration."""

    def __init__(
        self,
        *,
        model: str,
        structured_model: Any | None = None,
        max_scrolls: int = 3,
        max_attempts: int = 2,
        prefer_single_candidate: bool = True,
        base_url: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("A DeepSeek model name is required.")
        if max_scrolls <= 0 or max_attempts <= 0:
            raise ValueError("DeepSeek planner limits must be positive.")
        if structured_model is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise AgentError("未设置 DEEPSEEK_API_KEY，无法启动 DeepSeek Agent。")
            try:
                from langchain_deepseek import ChatDeepSeek
            except ImportError as exc:
                raise AgentError(
                    "DeepSeek LangChain 适配器未安装，请先运行 scripts/setup.ps1。"
                ) from exc

            resolved_base_url = (
                base_url or os.environ.get("WELLPHONE_DEEPSEEK_BASE_URL") or ""
            ).strip()
            model_options: dict[str, Any] = {}
            if resolved_base_url:
                model_options["base_url"] = resolved_base_url
            chat_model = ChatDeepSeek(
                model=model,
                api_key=api_key,
                temperature=0,
                timeout=30,
                max_retries=1,
                **model_options,
            )
            structured_model = chat_model.with_structured_output(
                DeepSeekActionDecision,
                method="json_mode",
                include_raw=True,
            )
        self.model = model
        self.structured_model = structured_model
        self.max_scrolls = max_scrolls
        self.max_attempts = max_attempts
        self.prefer_single_candidate = prefer_single_candidate

    @staticmethod
    def _validate_decision(raw_decision: Any) -> DeepSeekActionDecision:
        if isinstance(raw_decision, DeepSeekActionDecision):
            return raw_decision
        if isinstance(raw_decision, dict) and "parsed" in raw_decision:
            parsed = raw_decision.get("parsed")
            if parsed is not None:
                return (
                    parsed
                    if isinstance(parsed, DeepSeekActionDecision)
                    else DeepSeekActionDecision.model_validate(parsed)
                )
            raw_message = raw_decision.get("raw")
            content = getattr(raw_message, "content", "")
            if not isinstance(content, str):
                raise ValueError("DeepSeek returned non-text JSON content.")
            cleaned = content.strip()
            if cleaned.startswith("```json") and cleaned.endswith("```"):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
            return DeepSeekActionDecision.model_validate(json.loads(cleaned))
        return DeepSeekActionDecision.model_validate(raw_decision)

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
        if self.prefer_single_candidate and len(candidates) == 1:
            return AgentAction(
                "tap",
                "当前页面只有一个经过任务白名单批准的可点击候选。",
                candidates[0],
            )
        if not candidates and scroll_count >= self.max_scrolls:
            return AgentAction("abort", "达到安全浏览上限后仍未发现目标。")

        # Screenshots, unrelated page text and device identifiers are intentionally
        # excluded. The model receives only the minimum state required to decide.
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
        messages = [
            (
                "system",
                "你是安卓虚拟屏的受限动作规划器。只返回一个符合指定结构的 JSON 动作。"
                "clickable_candidates 非空时选择 tap，且 target 必须严格来自该列表；"
                "列表为空时可以选择 scroll_down；无法安全继续时选择 abort。"
                "不要生成坐标，不要声称动作已经执行，不要扩大任务范围。",
            ),
            ("human", json.dumps(state_payload, ensure_ascii=False)),
        ]
        last_error: Exception | None = None
        decision: DeepSeekActionDecision | None = None
        for _ in range(self.max_attempts):
            try:
                raw_decision = self.structured_model.invoke(messages)
                decision = self._validate_decision(raw_decision)
                break
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) in {401, 429}:
                    break
        if decision is None:
            assert last_error is not None
            exc = last_error
            status_code = getattr(exc, "status_code", None)
            if status_code == 401:
                detail = "DeepSeek API 凭据无效，请更新 DEEPSEEK_API_KEY。"
            elif status_code == 429:
                detail = "DeepSeek API 当前达到调用限制或无可用额度。"
            else:
                detail = "DeepSeek API 调用失败，请检查网络、模型名称和账户状态。"
            raise AgentError(detail) from exc

        target = decision.target
        if decision.action == "tap" and target not in candidates:
            raise AgentError("DeepSeek planner returned a target outside the approved goal.")
        if decision.action != "tap":
            target = None
        return AgentAction(decision.action, decision.reason, target)
