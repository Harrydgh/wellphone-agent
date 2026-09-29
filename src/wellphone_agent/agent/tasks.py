from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

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
        object.__setattr__(self, "query", query)

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
        return understanding.screen.kind in {"results", "store", "product"} and any(
            term.casefold() in label.casefold() for term in self.query_terms
        )

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
            query_match = any(
                term.casefold() in label.casefold()
                for term in self.query_terms
            )
            is_search_header = (
                understanding.screen.kind in {"results", "store", "product"}
                and label.casefold() == self.query.casefold()
                and candidate_bounds[1] < 300
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
            if query_match and not candidate.requires_confirmation:
                candidate = replace(
                    candidate,
                    risk_level="medium",
                    requires_confirmation=True,
                )
            if self._trusted_ocr_button(
                label, candidate.source, candidate.confidence
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
                and label.casefold() == self.query.casefold()
                and element_bounds[1] < 300
            ):
                continue
            digest = hashlib.sha256(
                f"{label}|{element_bounds}|{element.source}|task".encode("utf-8")
            ).hexdigest()[:12]
            query_match = any(
                term.casefold() in label.casefold() for term in self.query_terms
            )
            trusted_ocr_button = self._trusted_ocr_button(
                label, element.source, element.confidence
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
                        "medium" if query_match or trusted_ocr_button else "low"
                    ),
                    requires_confirmation=query_match or element.source == "ocr",
                    requires_user_takeover=False,
                    executable=(
                        element.source == "uiautomator" and element.clickable
                    )
                    or trusted_ocr_button,
                )
            )
            identities.add(identity)
        return tuple(selected)

    def is_satisfied(
        self,
        understanding: PageUnderstanding,
        completed_targets: tuple[str, ...],
    ) -> bool:
        added = any("加入购物车" in label for label in completed_targets)
        if not added:
            return False
        if understanding.screen.kind == "cart":
            return True
        texts = tuple(element.text.strip() for element in understanding.elements)
        has_selected_specs = any("已选规格" in text for text in texts)
        has_cart_total = any("去结算" in text for text in texts)
        add_button_gone = not any("加入购物车" in text for text in texts)
        return (
            understanding.screen.kind in {"product", "store"}
            and has_selected_specs
            and has_cart_total
            and add_button_gone
        )


def meituan_food_task(query: str) -> AppTask:
    return AppTask(
        task_id="meituan_food_to_cart",
        description=f"在美团搜索“{query.strip()}”并加入购物车，验证购物车状态后停止",
        allowed_package=MEITUAN_PACKAGE,
        query=query,
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
