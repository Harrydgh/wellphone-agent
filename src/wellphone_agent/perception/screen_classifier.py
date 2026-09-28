from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .fusion import PerceivedElement

ScreenKind = Literal[
    "home",
    "search",
    "results",
    "store",
    "product",
    "cart",
    "checkout",
    "payment",
    "login",
    "permission",
    "unknown",
]


@dataclass(frozen=True)
class ScreenClassification:
    kind: ScreenKind
    confidence: float
    evidence: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class ScreenClassifier:
    KEYWORDS: tuple[tuple[ScreenKind, tuple[str, ...]], ...] = (
        ("payment", ("支付密码", "确认支付", "立即支付", "付款")),
        ("checkout", ("提交订单", "确认订单", "配送地址", "应付")),
        ("permission", ("允许", "仅在使用中", "权限")),
        ("login", ("登录", "验证码", "手机号登录")),
        ("product", ("选规格", "加入购物车", "商品详情")),
        ("cart", ("购物车", "去结算", "清空购物车")),
        ("store", ("配送费", "起送", "店铺", "商家")),
        ("results", ("综合排序", "筛选", "搜索结果")),
        ("search", ("搜索历史", "大家都在搜", "热门搜索")),
        ("home", ("首页", "推荐", "附近")),
    )

    def classify(
        self,
        elements: tuple[PerceivedElement, ...],
        current_activity: str | None = None,
    ) -> ScreenClassification:
        texts = tuple(item.text for item in elements)
        for kind, keywords in self.KEYWORDS:
            evidence = tuple(
                keyword
                for keyword in keywords
                if any(keyword in text for text in texts)
            )
            if evidence:
                confidence = min(0.98, 0.62 + 0.12 * len(evidence))
                return ScreenClassification(kind, confidence, evidence)
        activity = current_activity or ""
        if "Search" in activity:
            return ScreenClassification("search", 0.6, ("activity:Search",))
        return ScreenClassification("unknown", 0.25, ())
