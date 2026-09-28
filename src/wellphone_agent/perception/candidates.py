from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

from .fusion import PerceivedElement
from .privacy import PrivacyRedactor

CandidateRisk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class CandidateAction:
    candidate_id: str
    action: Literal["tap_visible_target"]
    label: str
    bounds: tuple[int, int, int, int]
    source: str
    confidence: float
    risk_level: CandidateRisk
    requires_confirmation: bool
    requires_user_takeover: bool
    executable: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class CandidateGenerator:
    BLOCKED_TERMS = (
        "删除",
        "卸载",
        "恢复出厂",
        "密码",
        "开启权限",
        "授权全部",
    )
    HIGH_RISK_TERMS = (
        "提交订单",
        "确认订单",
        "立即支付",
        "确认支付",
        "付款",
        "立即购买",
    )
    MEDIUM_RISK_TERMS = (
        "加入购物车",
        "去结算",
        "选规格",
        "选择规格",
        "收货地址",
        "优惠券",
    )
    OCR_ACTION_TERMS = (
        "搜索",
        "加入购物车",
        "去结算",
        "选规格",
        "提交订单",
        "支付",
        "返回",
    )

    @staticmethod
    def _risk(text: str) -> tuple[CandidateRisk, bool, bool]:
        if any(term in text for term in CandidateGenerator.HIGH_RISK_TERMS):
            return "high", True, True
        if any(term in text for term in CandidateGenerator.MEDIUM_RISK_TERMS):
            return "medium", True, False
        return "low", False, False

    def generate(
        self, elements: tuple[PerceivedElement, ...], max_candidates: int = 40
    ) -> tuple[CandidateAction, ...]:
        candidates: list[CandidateAction] = []
        for item in elements:
            text = item.text.strip()
            if not text or PrivacyRedactor.is_redacted(text):
                continue
            if any(term in text for term in self.BLOCKED_TERMS):
                continue
            is_ocr_action = item.source == "ocr" and any(
                term in text for term in self.OCR_ACTION_TERMS
            )
            if not (item.clickable and item.enabled) and not is_ocr_action:
                continue
            risk, confirmation, takeover = self._risk(text)
            identity = f"{text}|{item.bounds}|{item.source}".encode("utf-8")
            candidates.append(
                CandidateAction(
                    candidate_id=hashlib.sha256(identity).hexdigest()[:12],
                    action="tap_visible_target",
                    label=text,
                    bounds=item.bounds,
                    source=item.source,
                    confidence=item.confidence,
                    risk_level=risk,
                    requires_confirmation=confirmation,
                    requires_user_takeover=takeover,
                    executable=item.source == "uiautomator" and item.clickable,
                )
            )
            if len(candidates) >= max_candidates:
                break
        return tuple(candidates)
