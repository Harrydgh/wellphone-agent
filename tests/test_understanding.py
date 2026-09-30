from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from wellphone_agent.perception.candidates import CandidateGenerator
from wellphone_agent.perception.fusion import PerceivedElement, fuse_page_elements
from wellphone_agent.perception.ocr import OCRText, RapidOCREngine
from wellphone_agent.perception.privacy import PrivacyRedactor
from wellphone_agent.perception.screen_classifier import ScreenClassifier
from wellphone_agent.perception.state import PageState, VisibleElement
from wellphone_agent.perception.understanding import PageUnderstandingEngine


def element(
    text: str,
    *,
    source: str = "uiautomator",
    clickable: bool = True,
    bounds: tuple[int, int, int, int] = (10, 10, 100, 50),
) -> PerceivedElement:
    return PerceivedElement(
        text=text,
        bounds=bounds,
        source=source,  # type: ignore[arg-type]
        confidence=0.95,
        clickable=clickable,
        enabled=True,
    )


def page_state(screenshot: str, elements: tuple[VisibleElement, ...]) -> PageState:
    return PageState(
        captured_at="2026-09-28T00:00:00+00:00",
        serial="private-device-id",
        display_id=7,
        current_app="com.example.food",
        current_activity=".ProductActivity",
        screenshot=screenshot,
        width=1080,
        height=1920,
        sha256="0" * 64,
        perceptual_hash="0" * 16,
        changed_from_previous=None,
        visual_change_ratio=None,
        consecutive_static_frames=0,
        is_stale=False,
        ui_elements=elements,
        text_source="uiautomator",
    )


class FakeOCREngine:
    def __init__(self, results: tuple[OCRText, ...]) -> None:
        self.results = results

    def recognize(self, image_path: Path) -> tuple[OCRText, ...]:
        return self.results


class UnderstandingTests(unittest.TestCase):
    def test_rapidocr_adapter_filters_low_confidence(self) -> None:
        output = SimpleNamespace(
            boxes=[[[1, 2], [10, 2], [10, 12], [1, 12]]] * 2,
            txts=["咖啡", "噪声"],
            scores=[0.98, 0.2],
        )

        class Engine:
            def __call__(self, image_path: Path) -> object:
                return output

        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (20, 20), "white").save(image_path)
            results = RapidOCREngine(engine=Engine()).recognize(image_path)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].text, "咖啡")
        self.assertEqual(results[0].bounds, (1, 2, 10, 12))

    def test_fusion_prefers_uiautomator_and_keeps_ocr_only_price(self) -> None:
        state = page_state(
            "unused.png",
            (
                VisibleElement(
                    text="加入购物车",
                    content_description="",
                    resource_id="add_cart",
                    class_name="Button",
                    bounds="[10,10][200,80]",
                    clickable=True,
                    enabled=True,
                ),
            ),
        )
        fused = fuse_page_elements(
            state,
            (
                OCRText("加入购物车", (12, 12, 198, 78), 0.99),
                OCRText("¥29.90", (20, 100, 130, 150), 0.96),
            ),
        )
        self.assertEqual([item.text for item in fused], ["加入购物车", "¥29.90"])
        self.assertEqual(fused[0].source, "uiautomator")
        self.assertEqual(fused[1].source, "ocr")

    def test_privacy_redacts_personal_order_data(self) -> None:
        redactor = PrivacyRedactor()
        self.assertEqual(redactor.redact("手机 13812345678"), "手机 [手机号已隐藏]")
        self.assertEqual(redactor.redact("订单 123456789012"), "订单 [编号已隐藏]")
        self.assertEqual(redactor.redact("收货地址 上海市某路"), "收货地址：[地址已隐藏]")
        self.assertEqual(redactor.redact("联系人 张三"), "联系人：[姓名已隐藏]")
        self.assertEqual(redactor.redact("家庭网络-5G>"), "[网络名称已隐藏]")

    def test_classifier_uses_highest_risk_page_signal_first(self) -> None:
        classifier = ScreenClassifier()
        self.assertEqual(classifier.classify((element("加入购物车"),)).kind, "product")
        self.assertEqual(classifier.classify((element("购物车"),)).kind, "cart")
        self.assertEqual(classifier.classify((element("提交订单"),)).kind, "checkout")
        self.assertEqual(
            classifier.classify((element("拖动滑块刚露出完整的1个喜条就松开"),)).kind,
            "verification",
        )
        self.assertEqual(
            classifier.classify((element("请向右滑动滑块验证"),)).kind,
            "verification",
        )
        mixed = (element("提交订单"), element("立即支付"))
        self.assertEqual(classifier.classify(mixed).kind, "payment")
        self.assertEqual(classifier.classify((element("搜索系统设置项"),)).kind, "unknown")

    def test_bottom_navigation_cart_label_does_not_classify_home_as_cart(self) -> None:
        classifier = ScreenClassifier()
        home = element("推荐", bounds=(10, 200, 200, 260))
        cart_tab = element("购物车", bounds=(700, 1800, 850, 1900))
        self.assertEqual(classifier.classify((home, cart_tab)).kind, "home")

    def test_meituan_home_activity_outweighs_delivery_fee_promotion(self) -> None:
        classifier = ScreenClassifier()
        items = (
            element("外卖", bounds=(10, 200, 200, 260)),
            element("0配送费", bounds=(10, 500, 200, 560)),
            element("购物车", bounds=(700, 1800, 850, 1900)),
        )
        result = classifier.classify(
            items, "com.meituan.android.pt.homepage.activity.MainActivity"
        )
        self.assertEqual(result.kind, "home")

    def test_kfc_embedded_page_is_classified_as_store(self) -> None:
        classifier = ScreenClassifier()
        items = (
            element("神抢手", source="ocr", clickable=False),
            element("我的券", source="ocr", clickable=False),
        )
        self.assertEqual(classifier.classify(items).kind, "store")

    def test_compact_embedded_search_page_is_classified_as_search(self) -> None:
        classifier = ScreenClassifier()
        items = (
            element(
                "米线",
                source="ocr",
                clickable=False,
                bounds=(140, 170, 300, 230),
            ),
            element(
                "搜索",
                source="uiautomator",
                clickable=False,
                bounds=(892, 165, 1044, 249),
            ),
        )
        self.assertEqual(classifier.classify(items).kind, "search")

    def test_embedded_search_history_labels_are_classified_as_search(self) -> None:
        classifier = ScreenClassifier()
        items = (
            element("历史搜索", clickable=False, bounds=(31, 288, 196, 346)),
            element("搜索发现", clickable=False, bounds=(31, 498, 196, 556)),
        )
        self.assertEqual(classifier.classify(items).kind, "search")

    def test_sensitive_signal_outweighs_meituan_home_activity(self) -> None:
        classifier = ScreenClassifier()
        items = (
            element("外卖", bounds=(10, 200, 200, 260)),
            element("立即支付", bounds=(700, 1200, 1000, 1300)),
        )
        result = classifier.classify(
            items, "com.meituan.android.pt.homepage.activity.MainActivity"
        )
        self.assertEqual(result.kind, "payment")

    def test_cart_body_outweighs_meituan_home_activity(self) -> None:
        classifier = ScreenClassifier()
        items = (
            element("首页", bounds=(10, 1800, 150, 1900)),
            element("购物车", bounds=(650, 1800, 850, 1900)),
            element("全选", bounds=(10, 1650, 150, 1720)),
            element("结算", bounds=(800, 1650, 1030, 1750)),
        )
        result = classifier.classify(
            items, "com.meituan.android.pt.homepage.activity.MainActivity"
        )
        self.assertEqual(result.kind, "cart")

    def test_candidates_apply_risk_and_source_boundaries(self) -> None:
        candidates = CandidateGenerator().generate(
            (
                element("经典拿铁"),
                element("加入购物车", bounds=(10, 60, 200, 120)),
                element("提交订单", bounds=(10, 130, 200, 190)),
                element("搜索", source="ocr", clickable=False, bounds=(10, 200, 100, 250)),
                element("删除订单", bounds=(10, 260, 160, 310)),
                element("支付密码", bounds=(10, 320, 160, 370)),
            )
        )
        by_label = {item.label: item for item in candidates}
        self.assertEqual(by_label["经典拿铁"].risk_level, "low")
        self.assertTrue(by_label["加入购物车"].requires_confirmation)
        self.assertTrue(by_label["提交订单"].requires_user_takeover)
        self.assertFalse(by_label["搜索"].executable)
        self.assertNotIn("删除订单", by_label)
        self.assertNotIn("支付密码", by_label)

    def test_understanding_payload_is_redacted_and_excludes_raw_image_and_device(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (100, 100), "white").save(image_path)
            state = page_state(
                str(image_path),
                (
                    VisibleElement(
                        text="联系人 张三 13812345678",
                        content_description="",
                        resource_id="recipient",
                        class_name="TextView",
                        bounds="[1,1][90,30]",
                        clickable=False,
                        enabled=True,
                    ),
                ),
            )
            engine = PageUnderstandingEngine(
                ocr_engine=FakeOCREngine(
                    (OCRText("加入购物车", (1, 40, 90, 70), 0.99),)
                )
            )
            result = engine.analyze(state)
        payload = result.to_model_payload()
        serialized = str(payload)
        self.assertNotIn("13812345678", serialized)
        self.assertNotIn("private-device-id", serialized)
        self.assertNotIn("screenshot", payload)
        self.assertNotIn(str(image_path), serialized)
        self.assertEqual(result.screen.kind, "product")
        self.assertFalse(result.candidates[0].executable)


if __name__ == "__main__":
    unittest.main()
