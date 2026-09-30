from __future__ import annotations

import json
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from .core import AgentError


class ShoppingIntent(BaseModel):
    """A locally enforced shopping goal that always stops before checkout."""

    app: Literal["meituan"] = "meituan"
    merchant_query: str | None = Field(default=None, min_length=1, max_length=80)
    product_query: str = Field(min_length=1, max_length=80)
    selection_strategy: Literal["exact_match", "first_match"] = "exact_match"
    quantity: Literal[1] = 1
    specification_policy: Literal["default", "confirm"] = "confirm"
    stop_at: Literal["cart"] = "cart"

    @property
    def query(self) -> str:
        """Backward-compatible product query used by the execution adapter."""

        return self.product_query


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
        max_attempts: int = 2,
    ) -> None:
        if not model.strip() or max_attempts <= 0:
            raise ValueError("A model name and positive attempt limit are required.")
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
        self.max_attempts = max_attempts

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

        def first_text(*names: str) -> str | None:
            for name in names:
                value = normalized.get(name)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return None

        product = first_text(
            "product_query",
            "product_name",
            "product",
            "item_name",
            "item",
            "query",
        )
        merchant = first_text(
            "merchant_query",
            "merchant_name",
            "merchant",
            "store_query",
            "store_name",
            "brand",
        )
        search_keyword = first_text("search_keyword", "keyword")
        if product is None:
            product = search_keyword
        elif merchant is None and search_keyword and search_keyword != product:
            merchant = search_keyword
        normalized["app"] = "meituan"
        normalized["product_query"] = product
        normalized["merchant_query"] = merchant
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
        messages: list[tuple[str, str]] = [
            (
                "system",
                "你只负责将用户自由表达的美团商品加购需求转为结构化意图。"
                "app 必须是 meituan；product_query 是必填的商品名称；"
                "merchant_query 是可选的店铺、品牌或商家名称。"
                "如果用户说‘搜索 X，选择 Y’，通常 X 是 merchant_query，"
                "Y 是 product_query；如果只提到一个商品，则 merchant_query 为 null。"
                "selection_strategy 只能是 exact_match 或 first_match；"
                "quantity 当前只允许 1；specification_policy 只能是 default 或 confirm；"
                "stop_at 必须是 cart。不得生成结算、下单、地址或支付动作。"
                "只返回符合给定结构的 JSON。",
            ),
            ("human", normalized),
        ]
        intent: ShoppingIntent | None = None
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                intent = self._parse(self.structured_model.invoke(messages))
                break
            except Exception as exc:
                last_error = exc
                if getattr(exc, "status_code", None) in {401, 429}:
                    break
                if attempt + 1 < self.max_attempts:
                    messages.append(
                        (
                            "human",
                            "上一次返回没有通过本地校验。请重新返回 JSON："
                            "product_query 必填；merchant_query 可为 null；"
                            "quantity 只能为 1；stop_at 只能为 cart。",
                        )
                    )
        if intent is None:
            status = getattr(last_error, "status_code", None)
            if status == 401:
                detail = "DeepSeek API 凭据无效。"
            elif status == 429:
                detail = "DeepSeek API 当前达到调用限制或无可用额度。"
            elif isinstance(last_error, ValidationError) and any(
                error.get("loc") == ("quantity",)
                for error in last_error.errors()
            ):
                detail = "当前安全版本一次只支持加购 1 份商品。"
            elif isinstance(last_error, ValidationError) and any(
                error.get("loc") == ("product_query",)
                for error in last_error.errors()
            ):
                detail = "无法确定商品名称，请说明想要购买的具体商品。"
            else:
                detail = "购物需求暂时无法稳定解析，请换一种自然说法后重试。"
            raise AgentError(detail) from last_error
        if any(term in intent.query for term in self.BLOCKED_TERMS):
            raise AgentError("模型返回了超出安全边界的搜索词。")
        if intent.merchant_query and any(
            term in intent.merchant_query for term in self.BLOCKED_TERMS
        ):
            raise AgentError("模型返回了超出安全边界的店铺名称。")
        return intent
