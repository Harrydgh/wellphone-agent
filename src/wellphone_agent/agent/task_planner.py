from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol, Sequence

from pydantic import BaseModel, Field

from ..perception.understanding import PageUnderstanding
from .core import AgentError
from .tasks import AppTask


TaskActionKind = Literal[
    "tap_candidate",
    "input_query",
    "scroll_down",
    "back",
    "finish",
    "abort",
]


class TaskAction(BaseModel):
    kind: TaskActionKind
    reason: str = Field(min_length=1, max_length=300)
    candidate_id: str | None = None


class TaskTransition(BaseModel):
    step: int
    action: TaskAction
    target_label: str | None
    before_screen: str
    after_screen: str
    page_changed: bool


class TaskPlanner(Protocol):
    def plan(
        self,
        task: AppTask,
        understanding: PageUnderstanding,
        history: Sequence[TaskTransition],
    ) -> TaskAction: ...


class DeepSeekTaskDecision(BaseModel):
    action: Literal["tap_candidate", "input_query", "scroll_down", "back", "abort"]
    candidate_id: str | None = None
    reason: str = Field(default="选择任务范围内的下一步。", min_length=1, max_length=300)


class DeepSeekTaskPlanner:
    """Choose only task-scoped candidate IDs from redacted page understanding."""

    def __init__(
        self,
        *,
        model: str,
        structured_model: Any | None = None,
        max_attempts: int = 2,
        base_url: str | None = None,
    ) -> None:
        if not model.strip() or max_attempts <= 0:
            raise ValueError("A model name and positive attempt limit are required.")
        if structured_model is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise AgentError("未设置 DEEPSEEK_API_KEY，无法启动美团任务 Agent。")
            try:
                from langchain_deepseek import ChatDeepSeek
            except ImportError as exc:
                raise AgentError("DeepSeek LangChain 适配器尚未安装。") from exc
            options: dict[str, Any] = {}
            resolved_base_url = (
                base_url or os.environ.get("WELLPHONE_DEEPSEEK_BASE_URL") or ""
            ).strip()
            if resolved_base_url:
                options["base_url"] = resolved_base_url
            chat_model = ChatDeepSeek(
                model=model,
                api_key=api_key,
                temperature=0,
                timeout=30,
                max_retries=1,
                **options,
            )
            structured_model = chat_model.with_structured_output(
                DeepSeekTaskDecision,
                method="json_mode",
                include_raw=True,
            )
        self.model = model
        self.structured_model = structured_model
        self.max_attempts = max_attempts

    @staticmethod
    def _normalize_payload(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        aliases = {
            "tap": "tap_candidate",
            "click": "tap_candidate",
            "scroll": "scroll_down",
        }
        action = normalized.get("action")
        if isinstance(action, str) and action in aliases:
            normalized["action"] = aliases[action]
        if not normalized.get("candidate_id"):
            normalized["candidate_id"] = normalized.get("target") or normalized.get(
                "candidate"
            )
        reason = normalized.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            normalized["reason"] = "选择任务范围内的下一步。"
        return normalized

    @classmethod
    def _parse(cls, raw: Any) -> DeepSeekTaskDecision:
        if isinstance(raw, DeepSeekTaskDecision):
            return raw
        if isinstance(raw, dict) and "parsed" in raw:
            parsed = raw.get("parsed")
            if parsed is not None:
                if isinstance(parsed, DeepSeekTaskDecision):
                    return parsed
                return DeepSeekTaskDecision.model_validate(
                    cls._normalize_payload(parsed)
                )
            content = getattr(raw.get("raw"), "content", "")
            if not isinstance(content, str):
                raise ValueError("DeepSeek returned invalid JSON content.")
            cleaned = content.strip()
            if cleaned.startswith("```json") and cleaned.endswith("```"):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
            return DeepSeekTaskDecision.model_validate(
                cls._normalize_payload(json.loads(cleaned))
            )
        return DeepSeekTaskDecision.model_validate(cls._normalize_payload(raw))

    def plan(
        self,
        task: AppTask,
        understanding: PageUnderstanding,
        history: Sequence[TaskTransition],
    ) -> TaskAction:
        if understanding.current_app != task.allowed_package:
            return TaskAction(kind="abort", reason="页面已离开美团安全范围。")
        if understanding.screen.kind in {"checkout", "payment", "login", "permission"}:
            return TaskAction(kind="abort", reason="到达禁止自动操作的敏感页面。")

        candidates = task.candidates(understanding)
        added_to_cart = any(
            item.target_label and "加入购物车" in item.target_label
            for item in history
        )
        if not added_to_cart:
            candidates = tuple(
                item for item in candidates if item.label != "购物车"
            )
        action_candidates = tuple(item for item in candidates if item.executable)
        if understanding.screen.kind == "cart" and not added_to_cart:
            home = next(
                (item for item in action_candidates if item.label == "首页"), None
            )
            if home is not None:
                return TaskAction(
                    kind="tap_candidate",
                    candidate_id=home.candidate_id,
                    reason="当前是已有购物车，先返回首页开始新搜索。",
                )
        if understanding.screen.kind == "home":
            search_box = next(
                (item for item in action_candidates if item.label == "搜索框"), None
            )
            if search_box is not None:
                return TaskAction(
                    kind="tap_candidate",
                    candidate_id=search_box.candidate_id,
                    reason="先聚焦搜索框，避免提交其中残留的旧搜索词。",
                )
        if understanding.screen.kind == "modal":
            dismiss = next(
                (item for item in action_candidates if item.label == "暂不升级"),
                None,
            )
            if dismiss is not None:
                return TaskAction(
                    kind="tap_candidate",
                    candidate_id=dismiss.candidate_id,
                    reason="关闭不属于外卖任务的升级弹窗。",
                )

        last_target = history[-1].target_label if history else None
        if last_target == "搜索框":
            return TaskAction(kind="input_query", reason="输入用户提供的商品搜索词。")
        if history and history[-1].action.kind == "input_query":
            food_tab = next(
                (item for item in action_candidates if item.label == "外卖"), None
            )
            if food_tab is not None:
                return TaskAction(
                    kind="tap_candidate",
                    candidate_id=food_tab.candidate_id,
                    reason="将搜索结果限定到外卖分类。",
                )

        matched = tuple(
            item
            for item in action_candidates
            if any(term.casefold() in item.label.casefold() for term in task.query_terms)
        )
        product_term = task.query_terms[-1].casefold()
        executable_product_matches = tuple(
            item
            for item in matched
            if item.executable
            and product_term in item.label.casefold()
            and item.bounds[1] >= 300
        )
        if len(executable_product_matches) == 1:
            return TaskAction(
                kind="tap_candidate",
                candidate_id=executable_product_matches[0].candidate_id,
                reason="当前只有一个可执行的具体商品候选与用户搜索词明确匹配。",
            )
        if len(matched) == 1:
            return TaskAction(
                kind="tap_candidate",
                candidate_id=matched[0].candidate_id,
                reason="当前只有一个与用户搜索词明确匹配的候选。",
            )
        next_button = next(
            (
                item
                for item in action_candidates
                if item.label in {"选规格", "加入购物车"}
            ),
            None,
        )
        if next_button is not None:
            return TaskAction(
                kind="tap_candidate",
                candidate_id=next_button.candidate_id,
                reason=f"在目标商品页面定位到“{next_button.label}”按钮。",
            )
        if last_target == "外卖" and not matched:
            return TaskAction(
                kind="abort",
                reason="外卖结果中没有发现与用户搜索词明确匹配的候选。",
            )
        if understanding.screen.kind in {"store", "product"} and not action_candidates:
            return TaskAction(
                kind="scroll_down",
                reason="店铺页暂未暴露可执行控件，向下滚动以定位商品与加购按钮。",
            )

        payload = {
            "task": task.description,
            "query": task.query,
            "current_app": understanding.current_app,
            "current_activity": understanding.current_activity,
            "screen": understanding.screen.to_dict(),
            "history": [item.model_dump() for item in history[-6:]],
            "candidates": [item.to_dict() for item in action_candidates],
            "allowed_actions": [
                "tap_candidate",
                "input_query",
                "scroll_down",
                "back",
                "abort",
            ],
            "stop_boundary": "加入购物车并验证购物车状态后停止；禁止提交订单和支付",
        }
        messages = [
            (
                "system",
                "你是美团外卖任务的受限规划器。只能选择给出的 candidate_id，"
                "不能生成坐标、商品、店铺、地址或支付动作。先进入首页/外卖，再搜索；"
                "选择与 query 明确匹配的结果；加入购物车后进入购物车。"
                "遇到登录、权限、确认订单、支付或不确定状态时 abort。"
                "只返回一个符合给定结构的 JSON 对象，不要返回 Markdown。",
            ),
            ("human", json.dumps(payload, ensure_ascii=False)),
        ]
        decision: DeepSeekTaskDecision | None = None
        last_error: Exception | None = None
        for _ in range(self.max_attempts):
            try:
                decision = self._parse(self.structured_model.invoke(messages))
                break
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) in {401, 429}:
                    break
        if decision is None:
            status = getattr(last_error, "status_code", None)
            if status == 401:
                detail = "DeepSeek API 凭据无效。"
            elif status == 429:
                detail = "DeepSeek API 当前达到调用限制或无可用额度。"
            else:
                error_type = type(last_error).__name__ if last_error else "UnknownError"
                detail = (
                    "DeepSeek 任务规划失败，已在本轮手机动作前停止"
                    f"（{error_type}）。"
                )
            raise AgentError(detail) from last_error

        by_id = {item.candidate_id: item for item in action_candidates}
        if decision.action == "tap_candidate":
            if decision.candidate_id not in by_id:
                raise AgentError("DeepSeek 选择了任务候选列表之外的控件。")
            return TaskAction(
                kind="tap_candidate",
                candidate_id=decision.candidate_id,
                reason=decision.reason,
            )
        if decision.candidate_id is not None:
            raise AgentError("非点击动作不能携带候选控件编号。")
        return TaskAction(kind=decision.action, reason=decision.reason)
