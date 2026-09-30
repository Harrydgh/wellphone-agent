from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from typing import Literal

from ..perception.candidates import CandidateAction
from ..perception.understanding import PageUnderstanding
from .core import AgentError


MEITUAN_PACKAGE = "com.sankuai.meituan"


@dataclass(frozen=True)
class AppTask:
    task_id: str
    description: str
    allowed_package: str
    query: str
    navigation_labels: tuple[str, ...]
    merchant_query: str | None = None
    selection_strategy: Literal["exact_match", "first_match"] = "exact_match"
    quantity: int = 1
    specification_policy: Literal["default", "confirm"] = "confirm"
    max_steps: int = 12

    BLOCKED_TERMS = (
        "提交订单",
        "确认订单",
        "立即购买",
        "支付",
        "付款",
        "密码",
        "验证码",
        "收货地址",
        "删除",
        "借钱",
        "月付",
        "账单",
        "实名认证",
        "立即升级",
        "下载并安装",
        "清空购物车",
        "去结算",
    )

    def __post_init__(self) -> None:
        query = self.query.strip()
        if (
            not query
            or len(query.encode("utf-8")) > 120
            or any(ord(character) < 32 for character in query)
        ):
            raise AgentError("商品搜索词必须为 1 到 120 字节且不能包含控制字符。")
        if self.max_steps <= 0:
            raise AgentError("任务最大步骤数必须为正数。")
        if self.quantity != 1:
            raise AgentError("当前安全版本只支持每次加购 1 份商品。")
        object.__setattr__(self, "query", query)
        if self.merchant_query is not None:
            merchant = self.merchant_query.strip()
            if (
                not merchant
                or len(merchant.encode("utf-8")) > 120
                or any(ord(character) < 32 for character in merchant)
            ):
                raise AgentError("店铺搜索词必须为 1 到 120 字节且不能包含控制字符。")
            object.__setattr__(self, "merchant_query", merchant)

    @staticmethod
    def _blocked(label: str) -> bool:
        return any(term in label for term in AppTask.BLOCKED_TERMS)

    @staticmethod
    def _trusted_ocr_button(label: str, source: str, confidence: float) -> bool:
        if source != "ocr":
            return False
        if label == "选规格":
            return confidence >= 0.98
        return "加入购物车" in label and confidence >= 0.9

    def _trusted_ocr_merchant(
        self,
        label: str,
        source: str,
        confidence: float,
        bounds: tuple[int, int, int, int],
        understanding: PageUnderstanding,
    ) -> bool:
        """Allow only an exact, user-requested merchant OCR target.

        Meituan's delivery results sometimes expose merchant cards only as pixels.
        The target remains medium risk, requires approval, and is re-observed before
        execution by the workflow. Product OCR stays non-executable because one OCR
        region can span several horizontally adjacent products.
        """

        return bool(
            source == "ocr"
            and confidence >= 0.90
            and bounds[1] >= 300
            and understanding.screen.kind in {"results", "store"}
            and self.merchant_query
            and self.merchant_query.casefold() in label.casefold()
            and not self._blocked(label)
        )

    @staticmethod
    def _canonical_label(label: str) -> str:
        normalized = label.strip()
        if "加入购物车" in normalized:
            return "加入购物车"
        return normalized

    def _label_is_task_related(
        self, label: str, understanding: PageUnderstanding
    ) -> bool:
        if self._blocked(label):
            return False
        if label == "暂不升级":
            return True
        if any(
            label == marker or (marker == "搜索" and marker in label)
            for marker in self.navigation_labels
        ):
            return True
        return (
            understanding.screen.kind in {"results", "store", "product"}
            and (
                self.label_matches_query(label)
                or self.label_matches_merchant(label)
            )
        )

    def label_matches_query(self, label: str) -> bool:
        normalized_label = label.casefold()
        if any(term.casefold() in normalized_label for term in self.query_terms):
            return True
        if self.selection_strategy != "first_match":
            return False
        ignored = set("的一个份只装款新鲜经典招牌")
        query_characters = {
            character
            for character in self.query
            if "\u4e00" <= character <= "\u9fff" and character not in ignored
        }
        label_characters = {
            character
            for character in label
            if "\u4e00" <= character <= "\u9fff" and character not in ignored
        }
        return len(query_characters & label_characters) >= 2

    def label_matches_merchant(self, label: str) -> bool:
        if self.merchant_query is None:
            return False
        return self.merchant_query.casefold() in label.casefold()

    def label_matches_quantity(self, label: str) -> bool:
        if self.quantity != 1:
            return False
        normalized = label.casefold().replace(" ", "")
        return not any(
            marker in normalized
            for marker in ("双杯", "两杯", "2杯", "*2", "×2", "x2")
        )

    def merchant_was_selected(self, completed_targets: tuple[str, ...]) -> bool:
        return self.merchant_query is None or any(
            self.label_matches_merchant(label) for label in completed_targets
        )

    def input_query(self, completed_targets: tuple[str, ...]) -> str:
        if not self.merchant_was_selected(completed_targets):
            assert self.merchant_query is not None
            return self.merchant_query
        return self.query

    @property
    def initial_search_query(self) -> str:
        return self.merchant_query or self.query

    def _is_search_header(self, label: str, top: int) -> bool:
        return top < 300 and label.casefold() in {
            self.query.casefold(),
            self.initial_search_query.casefold(),
        }

    @property
    def query_terms(self) -> tuple[str, ...]:
        terms = [self.query]
        for marker in ("咖啡", "奶茶", "汉堡", "披萨"):
            end = self.query.find(marker)
            if end < 0:
                continue
            end += len(marker)
            for part in (self.query[:end], self.query[end:]):
                if len(part) >= 2 and part not in terms:
                    terms.append(part)
        return tuple(terms)

    def candidates(
        self, understanding: PageUnderstanding
    ) -> tuple[CandidateAction, ...]:
        """Return only candidates grounded in this task and its stop boundary."""

        selected: list[CandidateAction] = []
        identities: set[tuple[str, tuple[int, int, int, int]]] = set()
        has_spec_modal = any(
            "已选规格" in element.text or "加入购物车" in element.text
            for element in understanding.elements
        )
        is_embedded_marketplace = any(
            marker in element.text
            for element in understanding.elements
            for marker in ("神抢手", "神枪手", "我的券")
        )
        if is_embedded_marketplace:
            embedded_search = next(
                (
                    element
                    for element in understanding.elements
                    if element.source == "ocr"
                    and element.confidence >= 0.98
                    and element.bounds[1] < 300
                    and (
                        (
                            element.bounds[0] >= 500
                            and element.text.lstrip().startswith("Q")
                        )
                        or (
                            element.bounds[0] >= 800
                            and element.text.strip() == "搜索"
                        )
                    )
                ),
                None,
            )
            if embedded_search is not None:
                bounds = tuple(embedded_search.bounds)
                digest = hashlib.sha256(
                    f"搜索框|{bounds}|embedded-marketplace|task".encode("utf-8")
                ).hexdigest()[:12]
                selected.append(
                    CandidateAction(
                        candidate_id=digest,
                        action="tap_visible_target",
                        label="搜索框",
                        bounds=bounds,  # type: ignore[arg-type]
                        source="ocr",
                        confidence=embedded_search.confidence,
                        risk_level="low",
                        requires_confirmation=False,
                        requires_user_takeover=False,
                        executable=True,
                    )
                )
                identities.add((embedded_search.text.strip(), bounds))
        for element in understanding.elements:
            if "search_layout_area" not in element.resource_id:
                continue
            bounds = tuple(element.bounds)
            digest = hashlib.sha256(
                f"搜索框|{bounds}|{element.source}|task".encode("utf-8")
            ).hexdigest()[:12]
            selected.append(
                CandidateAction(
                    candidate_id=digest,
                    action="tap_visible_target",
                    label="搜索框",
                    bounds=bounds,  # type: ignore[arg-type]
                    source=element.source,
                    confidence=element.confidence,
                    risk_level="low",
                    requires_confirmation=False,
                    requires_user_takeover=False,
                    executable=element.source == "uiautomator" and element.clickable,
                )
            )
            identities.add((element.text.strip(), bounds))
        for candidate in understanding.candidates:
            label = self._canonical_label(candidate.label)
            candidate_bounds = tuple(candidate.bounds)
            identity = (label, candidate_bounds)
            target_match = self.label_matches_query(label) or self.label_matches_merchant(
                label
            )
            is_search_header = (
                understanding.screen.kind in {"results", "store", "product"}
                and self._is_search_header(label, candidate_bounds[1])
            )
            if (
                candidate.requires_user_takeover
                or not self._label_is_task_related(label, understanding)
                or identity in identities
                or is_search_header
                or (has_spec_modal and label == "选规格")
            ):
                continue
            if label != candidate.label:
                candidate = replace(candidate, label=label)
            if target_match and not candidate.requires_confirmation:
                candidate = replace(
                    candidate,
                    risk_level="medium",
                    requires_confirmation=True,
                )
            if self._trusted_ocr_button(
                label, candidate.source, candidate.confidence
            ) or self._trusted_ocr_merchant(
                label,
                candidate.source,
                candidate.confidence,
                candidate_bounds,
                understanding,
            ):
                candidate = replace(
                    candidate,
                    risk_level="medium",
                    requires_confirmation=True,
                    executable=True,
                )
            selected.append(candidate)
            identities.add(identity)

        for element in understanding.elements:
            label = self._canonical_label(element.text)
            element_bounds = tuple(element.bounds)
            identity = (label, element_bounds)
            if not label or identity in identities:
                continue
            if has_spec_modal and label == "选规格":
                continue
            if not self._label_is_task_related(label, understanding):
                continue
            if (
                understanding.screen.kind in {"results", "store", "product"}
                and self._is_search_header(label, element_bounds[1])
            ):
                continue
            digest = hashlib.sha256(
                f"{label}|{element_bounds}|{element.source}|task".encode("utf-8")
            ).hexdigest()[:12]
            target_match = self.label_matches_query(label) or self.label_matches_merchant(
                label
            )
            trusted_ocr_button = self._trusted_ocr_button(
                label, element.source, element.confidence
            )
            trusted_ocr_merchant = self._trusted_ocr_merchant(
                label,
                element.source,
                element.confidence,
                element_bounds,
                understanding,
            )
            selected.append(
                CandidateAction(
                    candidate_id=digest,
                    action="tap_visible_target",
                    label=label,
                    bounds=element_bounds,  # type: ignore[arg-type]
                    source=element.source,
                    confidence=element.confidence,
                    risk_level=(
                        "medium"
                        if target_match
                        or trusted_ocr_button
                        or trusted_ocr_merchant
                        else "low"
                    ),
                    requires_confirmation=target_match or element.source == "ocr",
                    requires_user_takeover=False,
                    executable=(
                        element.source == "uiautomator" and element.clickable
                    )
                    or trusted_ocr_button
                    or trusted_ocr_merchant,
                )
            )
            identities.add(identity)

        # Meituan can expose one large clickable merchant card together with
        # individually clickable products nested inside that card.  Preserve
        # that visual relationship in the product label so the planner can
        # select the requested product directly without guessing that tapping
        # the whole card will open the merchant menu.
        if self.merchant_query:
            merchant_cards = tuple(
                item
                for item in selected
                if item.executable
                and self.label_matches_merchant(item.label)
                and not self.label_matches_query(item.label)
            )
            associated: list[CandidateAction] = []
            for item in selected:
                if not item.executable or not self.label_matches_query(item.label):
                    associated.append(item)
                    continue
                containers = tuple(
                    card
                    for card in merchant_cards
                    if card.bounds != item.bounds
                    and card.bounds[0] <= item.bounds[0]
                    and card.bounds[1] <= item.bounds[1]
                    and card.bounds[2] >= item.bounds[2]
                    and card.bounds[3] >= item.bounds[3]
                )
                if not containers:
                    associated.append(item)
                    continue
                merchant = min(
                    containers,
                    key=lambda card: (
                        (card.bounds[2] - card.bounds[0])
                        * (card.bounds[3] - card.bounds[1]),
                        card.bounds[1],
                    ),
                )
                associated.append(
                    replace(item, label=f"{merchant.label} | {item.label}")
                )
            selected = associated
        return tuple(selected)

    def is_satisfied(
        self,
        understanding: PageUnderstanding,
        completed_targets: tuple[str, ...],
    ) -> bool:
        added = any("加入购物车" in label for label in completed_targets)
        if added and understanding.screen.kind == "cart":
            return True
        if added and self._has_inline_cart_state(understanding, require_query=False):
            return True
        return self.has_verified_existing_cart_item(understanding)

    def has_verified_existing_cart_item(
        self, understanding: PageUnderstanding
    ) -> bool:
        """Recognize this exact product's post-add state without adding it again."""

        return self._has_inline_cart_state(understanding, require_query=True)

    def _has_inline_cart_state(
        self,
        understanding: PageUnderstanding,
        *,
        require_query: bool,
    ) -> bool:

        texts = tuple(element.text.strip() for element in understanding.elements)
        has_query_match = any(self.label_matches_query(text) for text in texts)
        has_selected_specs = any("已选规格" in text for text in texts)
        has_cart_total = any(
            "去结算" in text
            or ("差" in text and "起送" in text)
            for text in texts
        )
        add_button_gone = not any("加入购物车" in text for text in texts)
        if understanding.screen.kind == "cart":
            return has_query_match and has_cart_total and add_button_gone
        return (
            understanding.screen.kind in {"product", "store"}
            and (has_query_match or not require_query)
            and (has_selected_specs or has_query_match)
            and has_cart_total
            and add_button_gone
        )


def meituan_food_task(
    query: str,
    *,
    merchant_query: str | None = None,
    selection_strategy: Literal["exact_match", "first_match"] = "exact_match",
    quantity: int = 1,
    specification_policy: Literal["default", "confirm"] = "confirm",
) -> AppTask:
    strategy_text = (
        "选择第一个真实匹配商品"
        if selection_strategy == "first_match"
        else "选择明确匹配商品"
    )
    specification_text = (
        "保持默认规格"
        if specification_policy == "default"
        else "规格需要确认"
    )
    location_text = (
        f"先进入店铺“{merchant_query.strip()}”，再查找"
        if merchant_query
        else "搜索"
    )
    return AppTask(
        task_id="meituan_food_to_cart",
        description=(
            f"在美团{location_text}商品“{query.strip()}”，{strategy_text}，"
            f"{specification_text}，加购 1 份并在购物车停止"
        ),
        allowed_package=MEITUAN_PACKAGE,
        query=query,
        merchant_query=merchant_query,
        selection_strategy=selection_strategy,
        quantity=quantity,
        specification_policy=specification_policy,
        max_steps=16 if merchant_query else 12,
        navigation_labels=(
            "暂不升级",
            "首页",
            "外卖",
            "搜索",
            "搜索框",
            "加入购物车",
            "选规格",
            "购物车",
        ),
    )
