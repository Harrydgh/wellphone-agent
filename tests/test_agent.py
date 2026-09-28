from __future__ import annotations

import unittest

from wellphone_agent.agent import (
    AgentAction,
    AgentGoal,
    AgentLoop,
    AgentSafetyPolicy,
    OpenAIPlanner,
    SettingsPlanner,
    parse_safe_goal,
)
from wellphone_agent.agent.core import AgentError
from wellphone_agent.perception.state import PageState, VisibleElement


GOAL = AgentGoal(
    "打开 WLAN 设置",
    "com.android.settings",
    "WLAN",
    ("WifiSettings",),
)


def page(
    *,
    activity: str = ".MainSettings",
    app: str = "com.android.settings",
    label: str | None = "WLAN",
    frame_hash: str = "a",
) -> PageState:
    elements = ()
    texts = ()
    if label:
        texts = (label,)
        elements = (
            VisibleElement(
                label,
                "",
                "android:id/title",
                "android.widget.LinearLayout",
                "[0,900][1080,1060]",
                True,
                True,
            ),
        )
    return PageState(
        "now",
        "device",
        7,
        app,
        activity,
        "frame.png",
        1080,
        1920,
        frame_hash,
        frame_hash,
        None,
        None,
        0,
        False,
        texts,
        elements,
        "uiautomator",
    )


class FakeObserver:
    def __init__(self, states: list[PageState]) -> None:
        self.states = states
        self.labels: list[str] = []

    def observe(self, label: str = "frame") -> PageState:
        self.labels.append(label)
        return self.states.pop(0)


class FakeActions:
    def __init__(self) -> None:
        self.display_id = 7
        self.calls: list[tuple[object, ...]] = []

    def tap(self, x: int, y: int) -> None:
        self.calls.append(("tap", x, y))

    def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration_ms: int = 400
    ) -> None:
        self.calls.append(("swipe", x1, y1, x2, y2, duration_ms))

    def keyevent(self, keycode: str) -> None:
        self.calls.append(("keyevent", keycode))


class FakeModelResponse:
    def __init__(self, output_text: str) -> None:
        self.output_text = output_text


class FakeResponses:
    def __init__(self, output_text: str, error: Exception | None = None) -> None:
        self.output_text = output_text
        self.error = error
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> FakeModelResponse:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return FakeModelResponse(self.output_text)


class FakeOpenAI:
    def __init__(
        self, output_text: str = "", error: Exception | None = None
    ) -> None:
        self.responses = FakeResponses(output_text, error)


class FakeAPIError(RuntimeError):
    def __init__(self, status_code: int, secret_detail: str) -> None:
        super().__init__(secret_detail)
        self.status_code = status_code


class AgentTests(unittest.TestCase):
    def test_natural_language_goal_parser_accepts_wifi_synonym(self) -> None:
        goal = parse_safe_goal("请帮我打开 Wi-Fi 设置")
        self.assertEqual(goal.target_label, "WLAN")
        self.assertEqual(goal.allowed_package, "com.android.settings")

    def test_natural_language_goal_parser_supports_safe_settings_tasks(self) -> None:
        cases = {
            "打开蓝牙设置": "蓝牙",
            "打开移动网络": "移动网络",
            "查看我的设备": "我的设备",
            "进入更多连接": "更多连接",
        }
        for task, expected_target in cases.items():
            with self.subTest(task=task):
                self.assertEqual(parse_safe_goal(task).target_label, expected_target)

    def test_subsettings_goal_requires_matching_page_title(self) -> None:
        goal = parse_safe_goal("打开蓝牙设置")
        self.assertFalse(goal.is_satisfied(page(label="蓝牙")))
        self.assertTrue(goal.is_satisfied(page(activity=".SubSettings", label="蓝牙")))
        self.assertFalse(goal.is_satisfied(page(activity=".SubSettings", label="移动网络")))

    def test_goal_accepts_only_explicit_additional_system_package(self) -> None:
        goal = parse_safe_goal("打开移动网络")
        phone_page = page(activity=".settings.MobileNetworkSettings", label="移动网络")
        phone_page = PageState(
            **{**phone_page.__dict__, "current_app": "com.android.phone"}
        )
        self.assertTrue(goal.is_satisfied(phone_page))
        unsafe_page = PageState(
            **{**phone_page.__dict__, "current_app": "com.example.unapproved"}
        )
        self.assertFalse(goal.is_satisfied(unsafe_page))

    def test_natural_language_goal_parser_rejects_unapproved_task(self) -> None:
        with self.assertRaisesRegex(AgentError, "只开放"):
            parse_safe_goal("删除账号")

    def test_planner_taps_visible_goal(self) -> None:
        action = SettingsPlanner().plan(GOAL, page(), ())
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "WLAN")

    def test_planner_scrolls_when_goal_is_not_visible(self) -> None:
        action = SettingsPlanner().plan(GOAL, page(label=None), ())
        self.assertEqual(action.kind, "scroll_down")

    def test_planner_uses_only_goal_scoped_navigation_candidates(self) -> None:
        multi_step_goal = AgentGoal(
            "打开 VPN 设置",
            "com.android.settings",
            "VPN",
            ("VpnSettings",),
            allowed_tap_targets=("更多连接", "VPN"),
        )
        action = SettingsPlanner().plan(
            multi_step_goal, page(label="更多连接"), ()
        )
        self.assertEqual(action, AgentAction("tap", "当前页面存在允许点击的“更多连接”。", "更多连接"))

    def test_policy_rejects_planner_tapping_another_target(self) -> None:
        with self.assertRaises(AgentError):
            AgentSafetyPolicy().validate(
                GOAL, page(), AgentAction("tap", "bad", "蓝牙")
            )

    def test_policy_rejects_false_completion(self) -> None:
        with self.assertRaises(AgentError):
            AgentSafetyPolicy().validate(
                GOAL, page(), AgentAction("finish", "not really complete")
            )

    def test_openai_planner_uses_structured_minimal_state(self) -> None:
        client = FakeOpenAI(
            '{"action":"tap","target":"WLAN","reason":"目标可点击"}'
        )
        planner = OpenAIPlanner(model="test-model", client=client)
        action = planner.plan(GOAL, page(), ())
        self.assertEqual(action, AgentAction("tap", "目标可点击", "WLAN"))
        call = client.responses.calls[0]
        self.assertEqual(call["model"], "test-model")
        self.assertFalse(call["store"])
        self.assertIn("json_schema", str(call["text"]))
        self.assertNotIn("3109319912", str(call["input"]))

    def test_openai_planner_rejects_unapproved_tap_target(self) -> None:
        client = FakeOpenAI(
            '{"action":"tap","target":"蓝牙","reason":"点击另一个目标"}'
        )
        with self.assertRaisesRegex(AgentError, "outside the approved goal"):
            OpenAIPlanner(model="test-model", client=client).plan(GOAL, page(), ())

    def test_openai_planner_sanitizes_authentication_error(self) -> None:
        client = FakeOpenAI(error=FakeAPIError(401, "secret-key-detail"))
        with self.assertRaisesRegex(AgentError, "凭据无效") as caught:
            OpenAIPlanner(model="test-model", client=client).plan(GOAL, page(), ())
        self.assertNotIn("secret-key-detail", str(caught.exception))

    def test_openai_planner_reports_exhausted_quota(self) -> None:
        client = FakeOpenAI(error=FakeAPIError(429, "billing-detail"))
        with self.assertRaisesRegex(AgentError, "无可用额度"):
            OpenAIPlanner(model="test-model", client=client).plan(GOAL, page(), ())

    def test_agent_loop_observes_taps_and_verifies_goal(self) -> None:
        observer = FakeObserver(
            [
                page(frame_hash="a"),
                page(activity=".Settings$WifiSettingsActivity", frame_hash="b"),
            ]
        )
        actions = FakeActions()
        result = AgentLoop(
            observer,  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            settle_seconds=0,
        ).run(GOAL)
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 1)
        self.assertEqual(actions.calls, [("tap", 540, 980)])
        self.assertTrue(result.history[0].page_changed)

    def test_agent_stops_after_repeated_unchanged_actions(self) -> None:
        same = page(label=None, frame_hash="same")
        observer = FakeObserver([same, same, same])
        actions = FakeActions()
        result = AgentLoop(
            observer,  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            max_unchanged_actions=2,
            settle_seconds=0,
        ).run(GOAL)
        self.assertFalse(result.success)
        self.assertIn("no page change", result.reason)
        self.assertEqual(len(actions.calls), 2)

    def test_agent_rejects_observer_from_another_display(self) -> None:
        state = page()
        wrong_display = PageState(
            **{**state.__dict__, "display_id": 8}
        )
        with self.assertRaisesRegex(AgentError, "does not match"):
            AgentLoop(
                FakeObserver([wrong_display]),  # type: ignore[arg-type]
                FakeActions(),  # type: ignore[arg-type]
                SettingsPlanner(),
                settle_seconds=0,
            ).run(GOAL)


if __name__ == "__main__":
    unittest.main()
