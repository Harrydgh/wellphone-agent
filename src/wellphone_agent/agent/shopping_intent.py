from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field

from .core import AgentError


class ShoppingIntent(BaseModel):
    """A locally enforced shopping goal that always stops before checkout."""

    query: str = Field(min_length=1, max_length=80)
    selection_strategy: Literal["exact_match", "first_match"] = "exact_match"
    quantity: Literal[1] = 1
    specification_policy: Literal["default", "confirm"] = "confirm"
    stop_at: Literal["cart"] = "cart"


class DeepSeekShoppingIntentParser:
    BLOCKED_TERMS = (
        "下单",
        "结算",
        "提交订单",
        "确认订单",
        "立即支付",
        "确认支付",
        "付款",
        "立即购买",
        "支付密码",
        "修改地址",
    )

    def __init__(
        self,
        *,
        model: str,
        structured_model: Any | None = None,
        base_url: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("A model name is required.")
        if structured_model is None:
            api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not api_key:
                raise AgentError("未设置 DEEPSEEK_API_KEY，无法理解自然语言购物任务。")
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
                ShoppingIntent,
                method="json_mode",
                include_raw=True,
            )
        self.model = model
        self.structured_model = structured_model

    @staticmethod
    def _normalize(payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        normalized = dict(payload)
        strategy_aliases = {
            "first": "first_match",
            "first_product": "first_match",
            "exact": "exact_match",
        }
        specification_aliases = {
            "defaults": "default",
            "default_specs": "default",
            "ask": "confirm",
        }
        strategy = normalized.get("selection_strategy")
        specification = normalized.get("specification_policy")
        if strategy in strategy_aliases:
            normalized["selection_strategy"] = strategy_aliases[strategy]
        if specification in specification_aliases:
            normalized["specification_policy"] = specification_aliases[specification]
        normalized["quantity"] = normalized.get("quantity", 1)
        normalized["stop_at"] = "cart"
        return normalized

    @classmethod
    def _parse(cls, raw: Any) -> ShoppingIntent:
        if isinstance(raw, ShoppingIntent):
            return raw
        if isinstance(raw, dict) and "parsed" in raw:
            parsed = raw.get("parsed")
            if parsed is not None:
                return ShoppingIntent.model_validate(cls._normalize(parsed))
            content = getattr(raw.get("raw"), "content", "")
            if not isinstance(content, str):
                raise ValueError("DeepSeek returned invalid shopping intent JSON.")
            cleaned = content.strip()
            if cleaned.startswith("```json") and cleaned.endswith("```"):
                cleaned = cleaned[7:-3].strip()
            elif cleaned.startswith("```") and cleaned.endswith("```"):
                cleaned = cleaned[3:-3].strip()
            return ShoppingIntent.model_validate(cls._normalize(json.loads(cleaned)))
        return ShoppingIntent.model_validate(cls._normalize(raw))

    def parse(self, instruction: str) -> ShoppingIntent:
        normalized = instruction.strip()
        if (
            not normalized
            or len(normalized.encode("utf-8")) > 600
            or any(ord(character) < 32 for character in normalized)
        ):
            raise AgentError("购物任务必须为 1 到 600 字节且不能包含控制字符。")
        if any(term in normalized for term in self.BLOCKED_TERMS):
            raise AgentError("任务包含结算、支付或地址修改，超出当前安全边界。")
        messages = [
            (
                "system",
                "你只负责将美团商品加购需求转为结构化意图。"
                "selection_strategy 只能是 exact_match 或 first_match；"
                "quantity 当前只允许 1；specification_policy 只能是 default 或 confirm；"
                "stop_at 必须是 cart。不得生成结算、下单、地址或支付动作。"
                "只返回符合给定结构的 JSON。",
            ),
            ("human", normalized),
        ]
        try:
            intent = self._parse(self.structured_model.invoke(messages))
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status == 401:
                detail = "DeepSeek API 凭据无效。"
            elif status == 429:
                detail = "DeepSeek API 当前达到调用限制或无可用额度。"
            else:
                detail = f"购物任务理解失败（{type(exc).__name__}）。"
            raise AgentError(detail) from exc
        if any(term in intent.query for term in self.BLOCKED_TERMS):
            raise AgentError("模型返回了超出安全边界的搜索词。")
        return intent
