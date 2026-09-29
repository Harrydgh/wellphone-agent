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
    "modal",
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
        ("modal", ("暂不升级", "新版本抢先体验", "以后再说")),
        ("payment", ("支付密码", "确认支付", "立即支付", "付款")),
        ("checkout", ("提交订单", "确认订单", "配送地址", "应付")),
        ("permission", ("允许", "仅在使用中", "权限")),
        ("login", ("登录", "验证码", "手机号登录")),
        ("product", ("选规格", "加入购物车", "商品详情")),
        ("cart", ("购物车", "去结算", "清空购物车", "结算", "全选")),
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
        page_bottom = max((item.bounds[3] for item in elements), default=1)
        activity = current_activity or ""
        sensitive_kinds = {"modal", "payment", "checkout", "permission", "login"}
        for kind, keywords in self.KEYWORDS:
            if kind not in sensitive_kinds:
                continue
            evidence = tuple(
                keyword
                for keyword in keywords
                if any(keyword in text for text in texts)
            )
            if evidence:
                confidence = min(0.98, 0.62 + 0.12 * len(evidence))
                return ScreenClassification(kind, confidence, evidence)
        cart_strong = tuple(
            keyword
            for keyword in ("结算", "全选", "清空购物车")
            if any(keyword in text for text in texts)
        )
        if cart_strong and any("购物车" in text for text in texts):
            return ScreenClassification("cart", 0.94, cart_strong)
        if "homepage.activity.MainActivity" in activity and any(
            any(marker in text for marker in ("首页", "推荐", "外卖", "团购"))
            for text in texts
        ):
            return ScreenClassification(
                "home", 0.9, ("activity:homepage.MainActivity",)
            )
        for kind, keywords in self.KEYWORDS:
            if kind in sensitive_kinds:
                continue
            evidence = tuple(
                keyword
                for keyword in keywords
                if any(keyword in text for text in texts)
            )
            if kind == "cart" and evidence == ("购物车",):
                cart_regions = tuple(
                    item for item in elements if "购物车" in item.text
                )
                if cart_regions and all(
                    item.bounds[1] >= page_bottom * 0.82 for item in cart_regions
                ):
                    continue
            if evidence:
                confidence = min(0.98, 0.62 + 0.12 * len(evidence))
                return ScreenClassification(kind, confidence, evidence)
        if "Search" in activity:
            return ScreenClassification("search", 0.6, ("activity:Search",))
        return ScreenClassification("unknown", 0.25, ())
