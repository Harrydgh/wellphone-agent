from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from wellphone_agent.agent.core import AgentError
from wellphone_agent.agent.task_planner import (
    DeepSeekTaskPlanner,
    TaskAction,
    TaskTransition,
)
from wellphone_agent.agent.task_workflow import (
    AppTaskSafetyPolicy,
    LangGraphAppTaskLoop,
)
from wellphone_agent.agent.tasks import MEITUAN_PACKAGE, meituan_food_task
from wellphone_agent.perception.candidates import CandidateAction
from wellphone_agent.perception.fusion import PerceivedElement
from wellphone_agent.perception.ocr import OCRText
from wellphone_agent.perception.screen_classifier import ScreenClassification
from wellphone_agent.perception.state import PageState, VisibleElement
from wellphone_agent.perception.understanding import (
    PageUnderstanding,
    PageUnderstandingEngine,
)


def page(*, screenshot: str = "frame.png", frame_hash: str = "a") -> PageState:
    return PageState(
        captured_at="now",
        serial="private-device",
        display_id=7,
        current_app=MEITUAN_PACKAGE,
        current_activity=".FoodActivity",
        screenshot=screenshot,
        width=1080,
        height=1920,
        sha256=frame_hash,
        perceptual_hash=frame_hash,
        changed_from_previous=None,
        visual_change_ratio=None,
        consecutive_static_frames=0,
        is_stale=False,
    )


def candidate(
    label: str,
    *,
    candidate_id: str = "candidate-1",
    source: str = "uiautomator",
    confidence: float = 1.0,
    confirmation: bool = False,
    bounds: tuple[int, int, int, int] = (100, 200, 300, 300),
) -> CandidateAction:
    return CandidateAction(
        candidate_id=candidate_id,
        action="tap_visible_target",
        label=label,
        bounds=bounds,
        source=source,
        confidence=confidence,
        risk_level="medium" if confirmation else "low",
        requires_confirmation=confirmation,
        requires_user_takeover=False,
        executable=source == "uiautomator",
    )


def understanding(
    kind: str,
    *,
    candidates: tuple[CandidateAction, ...] = (),
    elements: tuple[PerceivedElement, ...] = (),
) -> PageUnderstanding:
    return PageUnderstanding(
        current_app=MEITUAN_PACKAGE,
        current_activity=".FoodActivity",
        screenshot="frame.png",
        text_source="uiautomator+ocr",
        ocr_count=0,
        elements=elements,
        screen=ScreenClassification(kind, 0.95, (kind,)),  # type: ignore[arg-type]
        candidates=candidates,
    )


class FakeModel:
    def __init__(self, decision: object) -> None:
        self.decision = decision
        self.calls: list[object] = []

    def invoke(self, messages: object) -> object:
        self.calls.append(messages)
        return self.decision


class FakeObserver:
    def __init__(self, pages: list[PageState]) -> None:
        self.pages = pages

    def observe(self, label: str = "frame") -> PageState:
        return self.pages.pop(0)


class FakeUnderstandingEngine:
    def __init__(self, values: list[PageUnderstanding]) -> None:
        self.values = values

    def analyze(self, state: PageState) -> PageUnderstanding:
        return self.values.pop(0)


class FakeActions:
    display_id = 7

    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def tap(self, x: int, y: int) -> None:
        self.calls.append(("tap", x, y))

    def swipe(self, *args: object) -> None:
        self.calls.append(("swipe", *args))

    def keyevent(self, keycode: str) -> None:
        self.calls.append(("keyevent", keycode))

    def type_text(self, text: str) -> None:
        self.calls.append(("type_text", text))

    def replace_text(self, text: str) -> None:
        self.calls.append(("replace_text", text))


class SequencePlanner:
    def __init__(self, actions: list[TaskAction]) -> None:
        self.actions = actions

    def plan(self, task: object, state: object, history: object) -> TaskAction:
        return self.actions.pop(0)


class AppTaskTests(unittest.TestCase):
    def test_task_rejects_control_characters(self) -> None:
        with self.assertRaises(AgentError):
            meituan_food_task("拿铁\n支付")

    def test_task_candidates_exclude_checkout_and_unrelated_account_controls(self) -> None:
        task = meituan_food_task("经典拿铁")
        state = understanding(
            "product",
            candidates=(
                candidate("加入购物车", candidate_id="add", confirmation=True),
                candidate("去结算", candidate_id="checkout"),
                candidate("美团月付", candidate_id="credit"),
            ),
        )
        self.assertEqual(
            [item.candidate_id for item in task.candidates(state)], ["add"]
        )

    def test_query_matching_ocr_candidate_requires_confirmation(self) -> None:
        task = meituan_food_task("经典拿铁")
        match = PerceivedElement(
            text="经典拿铁",
            bounds=(20, 430, 300, 500),
            source="ocr",
            confidence=0.98,
            clickable=False,
            enabled=True,
        )
        item = task.candidates(understanding("results", elements=(match,)))[0]
        self.assertEqual(item.source, "ocr")
        self.assertTrue(item.requires_confirmation)
        self.assertFalse(item.executable)

    def test_exact_high_confidence_ocr_merchant_is_confirmed_and_executable(
        self,
    ) -> None:
        task = meituan_food_task("生椰拿铁", merchant_query="瑞幸")
        merchant = PerceivedElement(
            text="瑞幸咖啡(亳州一中大师店)",
            bounds=(200, 432, 659, 478),
            source="ocr",
            confidence=0.98783,
            clickable=False,
            enabled=True,
        )
        item = task.candidates(
            understanding("store", elements=(merchant,))
        )[0]
        self.assertEqual(item.source, "ocr")
        self.assertTrue(item.executable)
        self.assertTrue(item.requires_confirmation)
        self.assertEqual(item.risk_level, "medium")

    def test_low_confidence_or_product_ocr_stays_non_executable(self) -> None:
        task = meituan_food_task("生椰拿铁", merchant_query="瑞幸")
        merchant = PerceivedElement(
            text="瑞幸咖啡(亳州一中大师店)",
            bounds=(200, 432, 659, 478),
            source="ocr",
            confidence=0.85,
            clickable=False,
            enabled=True,
        )
        product = PerceivedElement(
            text="生椰拿铁（首...小黄油拿铁",
            bounds=(659, 905, 1055, 944),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        items = task.candidates(
            understanding("store", elements=(merchant, product))
        )
        self.assertEqual(len(items), 2)
        self.assertTrue(all(not item.executable for item in items))

    def test_high_confidence_whitelisted_ocr_button_is_confirmed_and_executable(
        self,
    ) -> None:
        task = meituan_food_task("经典拿铁")
        select_specs = PerceivedElement(
            text="选规格",
            bounds=(926, 1794, 1031, 1839),
            source="ocr",
            confidence=0.9999,
            clickable=False,
            enabled=True,
        )
        item = task.candidates(
            understanding("product", elements=(select_specs,))
        )[0]
        self.assertTrue(item.executable)
        self.assertTrue(item.requires_confirmation)
        self.assertEqual(item.risk_level, "medium")

    def test_low_confidence_whitelisted_ocr_button_remains_non_executable(self) -> None:
        task = meituan_food_task("经典拿铁")
        select_specs = PerceivedElement(
            text="选规格",
            bounds=(926, 1794, 1031, 1839),
            source="ocr",
            confidence=0.9,
            clickable=False,
            enabled=True,
        )
        item = task.candidates(
            understanding("product", elements=(select_specs,))
        )[0]
        self.assertFalse(item.executable)

    def test_specs_modal_normalizes_add_button_and_hides_background_specs(self) -> None:
        task = meituan_food_task("经典拿铁")
        selected = PerceivedElement(
            text="已选规格：超大杯(570ml)、不额外加糖、标准、冰",
            bounds=(82, 1235, 954, 1271),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        noisy_add = PerceivedElement(
            text="十加入购物车",
            bounds=(770, 1396, 980, 1436),
            source="ocr",
            confidence=0.92,
            clickable=False,
            enabled=True,
        )
        background_specs = PerceivedElement(
            text="选规格",
            bounds=(928, 1796, 1030, 1839),
            source="ocr",
            confidence=0.999,
            clickable=False,
            enabled=True,
        )
        items = task.candidates(
            understanding(
                "product",
                elements=(selected, noisy_add, background_specs),
            )
        )
        self.assertEqual([item.label for item in items], ["加入购物车"])
        self.assertTrue(items[0].executable)
        self.assertTrue(items[0].requires_confirmation)

    def test_results_exclude_top_search_header_and_confirm_product(self) -> None:
        task = meituan_food_task("库迪咖啡经典拿铁")
        header = candidate(
            "库迪咖啡经典拿铁",
            candidate_id="header",
            bounds=(95, 103, 1029, 192),
        )
        product = candidate(
            "经典拿铁(超大杯)",
            candidate_id="product",
            bounds=(744, 728, 980, 1052),
        )
        items = task.candidates(
            understanding("results", candidates=(header, product))
        )
        self.assertEqual([item.candidate_id for item in items], ["product"])
        self.assertTrue(items[0].requires_confirmation)
        self.assertEqual(items[0].risk_level, "medium")

    def test_task_extracts_brand_and_product_query_terms(self) -> None:
        task = meituan_food_task("库迪咖啡经典拿铁")
        self.assertEqual(
            task.query_terms,
            ("库迪咖啡经典拿铁", "库迪咖啡", "经典拿铁"),
        )

    def test_first_match_strategy_selects_top_real_product_not_ad_or_store(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        advertisement = candidate(
            "广告 鲜肉包子",
            candidate_id="advertisement",
            confirmation=True,
            bounds=(20, 350, 1000, 520),
        )
        store = candidate(
            "老台门鲜肉包子店",
            candidate_id="store",
            confirmation=True,
            bounds=(20, 530, 1000, 680),
        )
        first_product = candidate(
            "鲜肉包子",
            candidate_id="first-product",
            confirmation=True,
            bounds=(200, 700, 900, 850),
        )
        second_product = candidate(
            "鲜肉包子 6个装",
            candidate_id="second-product",
            confirmation=True,
            bounds=(200, 1000, 900, 1150),
        )
        task = meituan_food_task(
            "鲜肉包子",
            selection_strategy="first_match",
            specification_policy="default",
        )
        action = planner.plan(
            task,
            understanding(
                "results",
                candidates=(
                    second_product,
                    store,
                    advertisement,
                    first_product,
                ),
            ),
            (),
        )
        self.assertEqual(action.candidate_id, "first-product")
        self.assertIn("第一个", action.reason)
        self.assertEqual(model.calls, [])

    def test_first_match_strategy_accepts_semantic_character_overlap(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        store = candidate(
            "爱已成粥(河北店)",
            candidate_id="store",
            confirmation=True,
            bounds=(20, 430, 1000, 620),
        )
        meat_bun = candidate(
            "猪肉小包(5个)",
            candidate_id="meat-bun",
            confirmation=True,
            bounds=(650, 700, 900, 900),
        )
        unrelated = candidate(
            "老面馒头",
            candidate_id="unrelated",
            confirmation=True,
            bounds=(200, 650, 500, 900),
        )
        task = meituan_food_task("鲜肉包子", selection_strategy="first_match")
        state = understanding(
            "results", candidates=(store, unrelated, meat_bun)
        )
        scoped = task.candidates(state)
        self.assertEqual([item.candidate_id for item in scoped], ["meat-bun"])
        action = planner.plan(task, state, ())
        self.assertEqual(action.candidate_id, "meat-bun")
        self.assertEqual(model.calls, [])

    def test_task_rejects_quantity_greater_than_one(self) -> None:
        with self.assertRaisesRegex(AgentError, "只支持.*1 份"):
            meituan_food_task("鲜肉包子", quantity=2)

    def test_store_and_product_are_separate_task_targets(self) -> None:
        task = meituan_food_task(
            "生椰拿铁",
            merchant_query="瑞幸咖啡",
            specification_policy="default",
        )
        store = candidate(
            "瑞幸咖啡（河北店）",
            candidate_id="store",
            confirmation=True,
            bounds=(20, 500, 1000, 700),
        )
        product = candidate(
            "生椰拿铁",
            candidate_id="product",
            confirmation=True,
            bounds=(200, 800, 900, 1000),
        )
        scoped = task.candidates(
            understanding("results", candidates=(store, product))
        )
        self.assertEqual(
            [item.candidate_id for item in scoped], ["store", "product"]
        )
        self.assertEqual(task.input_query(()), "瑞幸咖啡")
        self.assertEqual(
            task.input_query(("瑞幸咖啡（河北店）",)), "生椰拿铁"
        )
        self.assertFalse(task.label_matches_quantity("双杯生椰拿铁"))
        self.assertFalse(task.label_matches_quantity("生椰拿铁×2"))
        self.assertTrue(task.label_matches_quantity("生椰拿铁"))

    def test_kfc_embedded_store_exposes_only_top_search_hotword(self) -> None:
        task = meituan_food_task("薯条", merchant_query="肯德基")
        brand = PerceivedElement(
            text="神抢手",
            bounds=(80, 125, 276, 199),
            source="ocr",
            confidence=0.999,
            clickable=False,
            enabled=True,
        )
        hotword = PerceivedElement(
            text="Q烤肉拌饭",
            bounds=(603, 143, 778, 183),
            source="ocr",
            confidence=0.99966,
            clickable=False,
            enabled=True,
        )
        unrelated = PerceivedElement(
            text="每日秒杀",
            bounds=(691, 1337, 957, 1416),
            source="ocr",
            confidence=0.999,
            clickable=False,
            enabled=True,
        )
        items = task.candidates(
            understanding("store", elements=(brand, hotword, unrelated))
        )
        self.assertEqual([item.label for item in items], ["搜索框"])
        self.assertTrue(items[0].executable)
        self.assertEqual(items[0].source, "ocr")

    def test_embedded_marketplace_exposes_top_search_button_for_any_merchant(
        self,
    ) -> None:
        task = meituan_food_task("生椰拿铁", merchant_query="瑞幸")
        marketplace = PerceivedElement(
            text="我的券",
            bounds=(856, 1809, 943, 1845),
            source="ocr",
            confidence=0.999,
            clickable=False,
            enabled=True,
        )
        search = PerceivedElement(
            text="搜索",
            bounds=(908, 138, 1021, 188),
            source="ocr",
            confidence=0.997,
            clickable=False,
            enabled=True,
        )
        items = task.candidates(
            understanding("store", elements=(marketplace, search))
        )
        self.assertEqual([item.label for item in items], ["搜索框"])
        self.assertTrue(items[0].executable)

    def test_planner_enters_requested_store_before_selecting_product(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        task = meituan_food_task("生椰拿铁", merchant_query="瑞幸咖啡")
        store = candidate(
            "瑞幸咖啡（河北店）",
            candidate_id="store",
            confirmation=True,
            bounds=(20, 500, 1000, 700),
        )
        product = candidate(
            "生椰拿铁",
            candidate_id="product",
            confirmation=True,
            bounds=(200, 800, 900, 1000),
        )
        after_food_tab = TaskTransition(
            step=3,
            action=TaskAction(
                kind="tap_candidate",
                candidate_id="food",
                reason="进入外卖",
            ),
            target_label="外卖",
            before_screen="results",
            after_screen="results",
            page_changed=True,
        )
        action = planner.plan(
            task,
            understanding("results", candidates=(product, store)),
            (after_food_tab,),
        )
        self.assertEqual(action.candidate_id, "store")
        self.assertIn("指定的店铺", action.reason)

        after_store = TaskTransition(
            step=4,
            action=action,
            target_label="瑞幸咖啡（河北店）",
            before_screen="results",
            after_screen="store",
            page_changed=True,
        )
        action = planner.plan(
            task,
            understanding("store", candidates=(store, product)),
            (after_food_tab, after_store),
        )
        self.assertEqual(action.candidate_id, "product")
        self.assertEqual(model.calls, [])

    def test_planner_selects_product_nested_in_requested_store_card(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        task = meituan_food_task("生椰拿铁", merchant_query="瑞幸")
        store = candidate(
            "瑞幸咖啡(亳州杉杉国际城店)",
            candidate_id="store",
            confirmation=True,
            bounds=(21, 408, 1059, 1011),
        )
        product = candidate(
            "生椰拿铁（首创）",
            candidate_id="product",
            confirmation=True,
            bounds=(665, 674, 896, 990),
        )
        state = understanding("results", candidates=(store, product))

        scoped = task.candidates(state)
        nested_product = next(
            item for item in scoped if item.candidate_id == "product"
        )
        self.assertIn("瑞幸咖啡", nested_product.label)
        self.assertIn("生椰拿铁", nested_product.label)

        after_food_tab = TaskTransition(
            step=3,
            action=TaskAction(
                kind="tap_candidate",
                candidate_id="food",
                reason="进入外卖",
            ),
            target_label="外卖",
            before_screen="results",
            after_screen="results",
            page_changed=True,
        )
        action = planner.plan(task, state, (after_food_tab,))
        self.assertEqual(action.candidate_id, "product")
        self.assertIn("直接选择该商品", action.reason)
        self.assertEqual(model.calls, [])

    def test_search_layout_is_exposed_as_search_box_without_old_query(self) -> None:
        task = meituan_food_task("库迪咖啡经典拿铁")
        old_query = PerceivedElement(
            text="鲜花店",
            bounds=(30, 200, 900, 330),
            source="uiautomator",
            confidence=1.0,
            clickable=True,
            enabled=True,
            resource_id="com.sankuai.meituan:id/search_layout_area",
        )
        items = task.candidates(understanding("home", elements=(old_query,)))
        self.assertEqual([item.label for item in items], ["搜索框"])
        self.assertNotIn("鲜花店", str(items))

    def test_planner_closes_upgrade_modal_without_model_call(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        close = candidate("暂不升级", candidate_id="dismiss")
        action = planner.plan(
            meituan_food_task("经典拿铁"),
            understanding("modal", candidates=(close,)),
            (),
        )
        self.assertEqual(action.candidate_id, "dismiss")
        self.assertEqual(model.calls, [])

    def test_planner_focuses_search_box_locally_on_home(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        action = planner.plan(
            meituan_food_task("经典拿铁"),
            understanding(
                "home", candidates=(candidate("搜索框", candidate_id="search-box"),)
            ),
            (),
        )
        self.assertEqual(action.candidate_id, "search-box")
        self.assertEqual(model.calls, [])

    def test_planner_waits_for_search_box_on_partially_loaded_home(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        action = planner.plan(
            meituan_food_task("薯条", merchant_query="肯德基"),
            understanding(
                "home", candidates=(candidate("搜索", source="ocr"),)
            ),
            (),
        )
        self.assertEqual(action.kind, "wait")
        self.assertIn("搜索框", action.reason)
        self.assertEqual(model.calls, [])

    def test_planner_leaves_existing_cart_before_starting_search(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        action = planner.plan(
            meituan_food_task("经典拿铁"),
            understanding(
                "cart", candidates=(candidate("首页", candidate_id="home"),)
            ),
            (),
        )
        self.assertEqual(action.candidate_id, "home")
        self.assertEqual(model.calls, [])

    def test_planner_normalizes_safe_deepseek_field_aliases(self) -> None:
        model = FakeModel(
            {
                "action": "tap",
                "target": "food",
                "reason": "",
            }
        )
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        action = planner.plan(
            meituan_food_task("经典拿铁"),
            understanding("home", candidates=(candidate("外卖", candidate_id="food"),)),
            (),
        )
        self.assertEqual(action.kind, "tap_candidate")
        self.assertEqual(action.candidate_id, "food")
        self.assertTrue(action.reason)

    def test_planner_prefers_only_executable_product_over_ocr_duplicates(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        product = candidate(
            "经典拿铁(超大杯)",
            candidate_id="product",
            confirmation=True,
            bounds=(744, 728, 980, 1052),
        )
        store_ocr = candidate(
            "库迪咖啡(亳州花戏楼店)",
            candidate_id="store-ocr",
            source="ocr",
            confirmation=True,
            bounds=(271, 438, 689, 481),
        )
        product_ocr = candidate(
            "经典拿铁(超大杯)",
            candidate_id="product-ocr",
            source="ocr",
            confirmation=True,
            bounds=(740, 965, 943, 1002),
        )
        action = planner.plan(
            meituan_food_task("库迪咖啡经典拿铁"),
            understanding(
                "results", candidates=(product, store_ocr, product_ocr)
            ),
            (),
        )
        self.assertEqual(action.candidate_id, "product")
        self.assertEqual(model.calls, [])

    def test_planner_rejects_candidate_id_outside_task_scope(self) -> None:
        model = FakeModel(
            {"action": "tap_candidate", "candidate_id": "outside", "reason": "test"}
        )
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        with self.assertRaisesRegex(AgentError, "候选列表之外"):
            planner.plan(
                meituan_food_task("经典拿铁"),
                understanding("home", candidates=(candidate("外卖"),)),
                (),
            )

    def test_planner_does_not_offer_cart_before_an_item_was_added(self) -> None:
        model = FakeModel({"action": "abort", "reason": "inspect"})
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        planner.plan(
            meituan_food_task("经典拿铁"),
            understanding(
                "home",
                candidates=(
                    candidate("外卖", candidate_id="food"),
                    candidate("购物车", candidate_id="cart"),
                ),
            ),
            (),
        )
        self.assertIn("food", str(model.calls[0]))
        self.assertNotIn('"candidate_id": "cart"', str(model.calls[0]))

    def test_policy_rejects_low_confidence_ocr_coordinates(self) -> None:
        task = meituan_food_task("经典拿铁")
        low = candidate(
            "经典拿铁",
            source="ocr",
            confidence=0.6,
            confirmation=True,
            bounds=(100, 400, 300, 500),
        )
        state = understanding("results", candidates=(low,))
        action = TaskAction(
            kind="tap_candidate", reason="test", candidate_id=low.candidate_id
        )
        with self.assertRaisesRegex(AgentError, "置信度不足"):
            AppTaskSafetyPolicy().validate(task, page(), state, (), action)

    def test_policy_rejects_high_confidence_non_executable_ocr(self) -> None:
        task = meituan_food_task("经典拿铁")
        text_only = candidate(
            "经典拿铁",
            source="ocr",
            confidence=0.99,
            confirmation=True,
            bounds=(100, 400, 300, 500),
        )
        state = understanding("store", candidates=(text_only,))
        action = TaskAction(
            kind="tap_candidate",
            reason="test",
            candidate_id=text_only.candidate_id,
        )
        with self.assertRaisesRegex(AgentError, "缺少可验证"):
            AppTaskSafetyPolicy().validate(task, page(), state, (), action)

    def test_policy_rejects_double_cup_for_single_item_task(self) -> None:
        task = meituan_food_task("生椰拿铁", merchant_query="瑞幸")
        double = candidate(
            "瑞幸咖啡【双杯】人气拿铁（含生椰拿铁）",
            candidate_id="double",
            confirmation=True,
        )
        state = understanding("results", candidates=(double,))
        action = TaskAction(
            kind="tap_candidate", reason="test", candidate_id="double"
        )
        with self.assertRaisesRegex(AgentError, "单份任务"):
            AppTaskSafetyPolicy().validate(task, page(), state, (), action)

    def test_planner_scrolls_store_when_only_ocr_evidence_is_available(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        store_name = candidate(
            "库迪咖啡(亳州花戏楼店)",
            source="ocr",
            confidence=0.99,
            confirmation=True,
            bounds=(228, 831, 788, 891),
        )
        action = planner.plan(
            meituan_food_task("库迪咖啡经典拿铁"),
            understanding("store", candidates=(store_name,)),
            (),
        )
        self.assertEqual(action.kind, "scroll_down")
        self.assertEqual(model.calls, [])

    def test_planner_scrolls_past_store_promotions_after_merchant_selected(
        self,
    ) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        selected_store = TaskTransition(
            step=4,
            action=TaskAction(
                kind="tap_candidate", candidate_id="store", reason="进入肯德基"
            ),
            target_label="肯德基（亳州和平店）",
            before_screen="store",
            after_screen="store",
            page_changed=True,
        )
        promotion = candidate(
            "肯德基 | 【国庆节】经典汉堡 9 件套随心选",
            candidate_id="promotion",
        )
        action = planner.plan(
            meituan_food_task("薯条", merchant_query="肯德基"),
            understanding("store", candidates=(promotion,)),
            (selected_store,),
        )
        self.assertEqual(action.kind, "scroll_down")
        self.assertIn("薯条", action.reason)
        self.assertEqual(model.calls, [])

    def test_planner_skips_double_cup_results_for_single_item_task(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        selected_store = TaskTransition(
            step=4,
            action=TaskAction(
                kind="tap_candidate", candidate_id="store", reason="进入瑞幸"
            ),
            target_label="瑞幸咖啡(杉杉国际城店)",
            before_screen="store",
            after_screen="store",
            page_changed=True,
        )
        double = candidate(
            "瑞幸咖啡【双杯】人气拿铁（含生椰拿铁）",
            candidate_id="double",
            confirmation=True,
            bounds=(386, 746, 1012, 795),
        )
        action = planner.plan(
            meituan_food_task("生椰拿铁", merchant_query="瑞幸"),
            understanding("results", candidates=(double,)),
            (selected_store,),
        )
        self.assertEqual(action.kind, "scroll_down")
        self.assertIn("单份", action.reason)
        self.assertEqual(model.calls, [])

    def test_planner_selects_confirmed_specs_button_without_model(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        select_specs = candidate(
            "选规格",
            source="uiautomator",
            confirmation=True,
            bounds=(926, 1794, 1031, 1839),
        )
        action = planner.plan(
            meituan_food_task("库迪咖啡经典拿铁"),
            understanding("product", candidates=(select_specs,)),
            (),
        )
        self.assertEqual(action.kind, "tap_candidate")
        self.assertEqual(action.candidate_id, select_specs.candidate_id)
        self.assertEqual(model.calls, [])

    def test_planner_waits_locally_while_product_page_is_loading(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        loading = PerceivedElement(
            text="加载中",
            bounds=(450, 850, 650, 950),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        previous = TaskTransition(
            step=1,
            action=TaskAction(
                kind="tap_candidate", reason="open product", candidate_id="product"
            ),
            target_label="经典拿铁(超大杯)",
            before_screen="store",
            after_screen="unknown",
            page_changed=True,
        )
        action = planner.plan(
            meituan_food_task("库迪咖啡经典拿铁"),
            understanding("unknown", elements=(loading,)),
            (previous,),
        )
        self.assertEqual(action.kind, "wait")
        self.assertEqual(model.calls, [])

    def test_planner_stops_when_requested_store_is_closed(self) -> None:
        model = FakeModel(AssertionError("must not be used"))
        planner = DeepSeekTaskPlanner(model="test", structured_model=model)
        closed = PerceivedElement(
            text="门店已打烊",
            bounds=(430, 360, 670, 420),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        action = planner.plan(
            meituan_food_task("生椰拿铁", merchant_query="瑞幸"),
            understanding("store", elements=(closed,)),
            (),
        )
        self.assertEqual(action.kind, "abort")
        self.assertIn("打烊", action.reason)
        self.assertEqual(model.calls, [])

    def test_workflow_confirms_add_to_cart_and_stops_at_cart(self) -> None:
        add = candidate("加入购物车", candidate_id="add", confirmation=True)
        product = understanding("product", candidates=(add,))
        cart = understanding("cart")
        actions = FakeActions()
        loop = LangGraphAppTaskLoop(
            FakeObserver(
                [
                    page(frame_hash="a"),
                    page(frame_hash="a-confirmed"),
                    page(frame_hash="b"),
                ]
            ),  # type: ignore[arg-type]
            FakeUnderstandingEngine([product, product, cart]),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SequencePlanner(
                [TaskAction(kind="tap_candidate", reason="add", candidate_id="add")]
            ),
            settle_seconds=0,
            approval_handler=lambda request: True,
        )
        result = loop.run(meituan_food_task("经典拿铁"))
        self.assertTrue(result.success)
        self.assertEqual(actions.calls, [("tap", 200, 250)])

    def test_workflow_rejects_stale_candidate_after_slow_approval(self) -> None:
        add = candidate("加入购物车", candidate_id="add", confirmation=True)
        product = understanding("product", candidates=(add,))
        changed_page = understanding("checkout")
        actions = FakeActions()
        loop = LangGraphAppTaskLoop(
            FakeObserver([page(frame_hash="a"), page(frame_hash="changed")]),  # type: ignore[arg-type]
            FakeUnderstandingEngine([product, changed_page]),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SequencePlanner(
                [TaskAction(kind="tap_candidate", reason="add", candidate_id="add")]
            ),
            settle_seconds=0,
            approval_handler=lambda request: True,
        )
        with self.assertRaisesRegex(AgentError, "敏感页面"):
            loop.run(meituan_food_task("经典拿铁"))
        self.assertEqual(actions.calls, [])

    def test_task_accepts_verified_inline_cart_state_after_add(self) -> None:
        selected = PerceivedElement(
            text="已选规格：超大杯(570ml)、不额外加糖、标准、冰",
            bounds=(82, 1235, 954, 1271),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        checkout = PerceivedElement(
            text="去结算",
            bounds=(840, 1749, 982, 1805),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        task = meituan_food_task("库迪咖啡经典拿铁")
        self.assertTrue(
            task.is_satisfied(
                understanding("product", elements=(selected, checkout)),
                ("加入购物车",),
            )
        )

    def test_task_does_not_accept_specs_before_add_button_disappears(self) -> None:
        selected = PerceivedElement(
            text="已选规格：超大杯(570ml)",
            bounds=(82, 1235, 954, 1271),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        checkout = PerceivedElement(
            text="去结算",
            bounds=(840, 1749, 982, 1805),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        add = PerceivedElement(
            text="加入购物车",
            bounds=(770, 1396, 980, 1436),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        task = meituan_food_task("库迪咖啡经典拿铁")
        self.assertFalse(
            task.is_satisfied(
                understanding("product", elements=(selected, checkout, add)),
                ("加入购物车",),
            )
        )

    def test_task_accepts_matching_existing_cart_state_without_adding_again(self) -> None:
        product = PerceivedElement(
            text="经典拿铁(超大杯)",
            bounds=(80, 900, 600, 950),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        selected = PerceivedElement(
            text="已选规格：超大杯(570ml)",
            bounds=(80, 1200, 900, 1250),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        checkout = PerceivedElement(
            text="去结算",
            bounds=(840, 1749, 982, 1805),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        task = meituan_food_task("库迪咖啡经典拿铁")
        self.assertTrue(
            task.is_satisfied(
                understanding("product", elements=(product, selected, checkout)),
                (),
            )
        )

    def test_task_rejects_unrelated_existing_cart_state(self) -> None:
        other = PerceivedElement(
            text="珍珠奶茶",
            bounds=(80, 900, 600, 950),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        selected = PerceivedElement(
            text="已选规格：大杯",
            bounds=(80, 1200, 900, 1250),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        checkout = PerceivedElement(
            text="去结算",
            bounds=(840, 1749, 982, 1805),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        task = meituan_food_task("库迪咖啡经典拿铁")
        self.assertFalse(
            task.is_satisfied(
                understanding("product", elements=(other, selected, checkout)),
                (),
            )
        )

    def test_task_accepts_matching_item_in_cart_without_duplicate_add(self) -> None:
        product = PerceivedElement(
            text="经典拿铁(超大杯)",
            bounds=(550, 1550, 980, 1620),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        checkout = PerceivedElement(
            text="去结算",
            bounds=(780, 1720, 1040, 1840),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        task = meituan_food_task("库迪咖啡经典拿铁")
        self.assertTrue(
            task.is_satisfied(
                understanding("cart", elements=(product, checkout)),
                (),
            )
        )

    def test_task_accepts_item_below_store_minimum_order_without_duplicate(self) -> None:
        product = PerceivedElement(
            text="酱香猪肉包(1个)(力推新品)",
            bounds=(60, 1080, 720, 1160),
            source="uiautomator",
            confidence=1.0,
            clickable=False,
            enabled=True,
        )
        minimum = PerceivedElement(
            text="差¥16.2起送",
            bounds=(760, 1740, 1040, 1840),
            source="uiautomator",
            confidence=1.0,
            clickable=False,
            enabled=True,
        )
        task = meituan_food_task(
            "鲜肉包子",
            selection_strategy="first_match",
            specification_policy="default",
        )
        self.assertTrue(
            task.is_satisfied(
                understanding("product", elements=(product, minimum)),
                ("加入购物车",),
            )
        )
        self.assertTrue(
            task.is_satisfied(
                understanding("product", elements=(product, minimum)),
                (),
            )
        )

        other = PerceivedElement(
            text="八宝粥",
            bounds=(60, 1080, 720, 1160),
            source="uiautomator",
            confidence=1.0,
            clickable=False,
            enabled=True,
        )
        self.assertFalse(
            task.is_satisfied(
                understanding("product", elements=(other, minimum)),
                (),
            )
        )

    def test_pre_action_guard_blocks_action_after_approval(self) -> None:
        add = candidate("加入购物车", candidate_id="add", confirmation=True)
        actions = FakeActions()

        def reject_main_display_collision() -> None:
            raise AgentError("任务 App 已出现在主屏")

        loop = LangGraphAppTaskLoop(
            FakeObserver([page(frame_hash="a")]),  # type: ignore[arg-type]
            FakeUnderstandingEngine(
                [understanding("product", candidates=(add,))]
            ),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SequencePlanner(
                [TaskAction(kind="tap_candidate", reason="add", candidate_id="add")]
            ),
            settle_seconds=0,
            approval_handler=lambda request: True,
            pre_action_guard=reject_main_display_collision,
        )
        with self.assertRaisesRegex(AgentError, "主屏"):
            loop.run(meituan_food_task("经典拿铁"))
        self.assertEqual(actions.calls, [])

    def test_workflow_stops_without_action_for_matching_existing_cart_item(self) -> None:
        product = PerceivedElement(
            text="经典拿铁(超大杯)",
            bounds=(80, 900, 600, 950),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        selected = PerceivedElement(
            text="已选规格：超大杯(570ml)",
            bounds=(80, 1200, 900, 1250),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        checkout = PerceivedElement(
            text="去结算",
            bounds=(840, 1749, 982, 1805),
            source="ocr",
            confidence=0.99,
            clickable=False,
            enabled=True,
        )
        actions = FakeActions()
        loop = LangGraphAppTaskLoop(
            FakeObserver([page(frame_hash="a")]),  # type: ignore[arg-type]
            FakeUnderstandingEngine(
                [understanding("product", elements=(product, selected, checkout))]
            ),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SequencePlanner([]),
            settle_seconds=0,
        )
        result = loop.run(meituan_food_task("库迪咖啡经典拿铁"))
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 0)
        self.assertIn("未重复添加", result.reason)
        self.assertEqual(actions.calls, [])

    def test_modal_understanding_hides_background_ocr(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "page.png"
            Image.new("RGB", (100, 100), "white").save(image_path)

            class OCR:
                def recognize(self, path: Path) -> tuple[OCRText, ...]:
                    return (OCRText("美团借钱 183元", (1, 1, 90, 20), 0.99),)

            raw = page(screenshot=str(image_path))
            raw = PageState(
                **{
                    **raw.to_dict(),
                    "visible_text": (),
                    "ui_elements": (
                        VisibleElement(
                            "暂不升级",
                            "",
                            "btn_cancel",
                            "Button",
                            "[10,40][80,80]",
                            True,
                            True,
                        ),
                    ),
                }
            )
            result = PageUnderstandingEngine(ocr_engine=OCR()).analyze(raw)
        self.assertEqual(result.screen.kind, "modal")
        self.assertEqual([item.text for item in result.elements], ["暂不升级"])


if __name__ == "__main__":
    unittest.main()
