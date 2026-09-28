from __future__ import annotations

import unittest

from langchain_core.messages import AIMessage

from wellphone_agent.agent import (
    AgentAction,
    AgentApprovalRequired,
    AgentGoal,
    DeepSeekActionDecision,
    DeepSeekPlanner,
    LangGraphAgentLoop,
    SettingsPlanner,
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
    label: str | None = "WLAN",
    frame_hash: str = "a",
    display_id: int = 7,
    text_source: str = "uiautomator",
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
        display_id,
        "com.android.settings",
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
        text_source,
    )


class FakeObserver:
    def __init__(self, states: list[PageState]) -> None:
        self.states = states

    def observe(self, label: str = "frame") -> PageState:
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


class FakeStructuredModel:
    def __init__(self, decision: object = None, error: Exception | None = None) -> None:
        self.decision = decision
        self.error = error
        self.calls: list[object] = []

    def invoke(self, messages: object) -> object:
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return self.decision


class FlakyStructuredModel:
    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, messages: object) -> object:
        self.calls += 1
        if self.calls == 1:
            raise ValueError("malformed first response")
        return {"action": "tap", "target": "WLAN"}


class FakeAPIError(RuntimeError):
    def __init__(self, status_code: int, secret: str) -> None:
        super().__init__(secret)
        self.status_code = status_code


class LangGraphAgentTests(unittest.TestCase):
    def test_langgraph_runs_observe_plan_validate_execute_verify(self) -> None:
        observer = FakeObserver(
            [
                page(frame_hash="a"),
                page(activity=".Settings$WifiSettingsActivity", frame_hash="b"),
            ]
        )
        actions = FakeActions()
        result = LangGraphAgentLoop(
            observer,  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            settle_seconds=0,
        ).run(GOAL)
        self.assertTrue(result.success)
        self.assertEqual(result.steps, 1)
        self.assertEqual(actions.calls, [("tap", 540, 980)])

    def test_langgraph_retries_transient_ui_tree_before_planning(self) -> None:
        observer = FakeObserver(
            [
                page(label=None, text_source="uiautomator_package_mismatch"),
                page(frame_hash="a"),
                page(activity=".Settings$WifiSettingsActivity", frame_hash="b"),
            ]
        )
        actions = FakeActions()
        result = LangGraphAgentLoop(
            observer,  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            settle_seconds=0,
            observation_retry_seconds=0,
        ).run(GOAL)
        self.assertTrue(result.success)
        self.assertEqual(len(actions.calls), 1)

    def test_langgraph_stops_after_repeated_unchanged_actions(self) -> None:
        same = page(label=None, frame_hash="same")
        result = LangGraphAgentLoop(
            FakeObserver([same, same, same]),  # type: ignore[arg-type]
            FakeActions(),  # type: ignore[arg-type]
            SettingsPlanner(),
            max_unchanged_actions=2,
            settle_seconds=0,
        ).run(GOAL)
        self.assertFalse(result.success)
        self.assertIn("no page change", result.reason)

    def test_langgraph_rejects_primary_or_wrong_display(self) -> None:
        with self.assertRaisesRegex(AgentError, "does not match"):
            LangGraphAgentLoop(
                FakeObserver([page(display_id=0)]),  # type: ignore[arg-type]
                FakeActions(),  # type: ignore[arg-type]
                SettingsPlanner(),
                settle_seconds=0,
            ).run(GOAL)

    def test_deepseek_planner_returns_constrained_action(self) -> None:
        model = FakeStructuredModel(
            DeepSeekActionDecision(action="tap", target="WLAN", reason="目标可点击")
        )
        action = DeepSeekPlanner(
            model="test-model",
            structured_model=model,
            prefer_single_candidate=False,
        ).plan(GOAL, page(), ())
        self.assertEqual(action, AgentAction("tap", "目标可点击", "WLAN"))
        self.assertNotIn("device", str(model.calls[0]))

    def test_deepseek_planner_uses_local_choice_for_single_safe_candidate(self) -> None:
        model = FakeStructuredModel(error=AssertionError("model must not be called"))
        action = DeepSeekPlanner(
            model="test-model", structured_model=model
        ).plan(GOAL, page(), ())
        self.assertEqual(action.target, "WLAN")
        self.assertEqual(model.calls, [])

    def test_deepseek_planner_rejects_unapproved_target(self) -> None:
        model = FakeStructuredModel(
            {"action": "tap", "target": "蓝牙", "reason": "越权目标"}
        )
        with self.assertRaisesRegex(AgentError, "outside the approved goal"):
            DeepSeekPlanner(
                model="test-model",
                structured_model=model,
                prefer_single_candidate=False,
            ).plan(
                GOAL, page(), ()
            )

    def test_deepseek_planner_receives_only_clickable_goal_candidates(self) -> None:
        goal = AgentGoal(
            "打开 VPN 设置",
            "com.android.settings",
            "VPN",
            ("VpnSettings",),
            allowed_tap_targets=("更多连接", "VPN"),
        )
        model = FakeStructuredModel(
            {"action": "tap", "target": "更多连接", "reason": "安全路径"}
        )
        action = DeepSeekPlanner(
            model="test-model",
            structured_model=model,
            prefer_single_candidate=False,
        ).plan(goal, page(label="更多连接"), ())
        self.assertEqual(action.target, "更多连接")
        prompt = str(model.calls[0])
        self.assertIn("clickable_candidates", prompt)
        self.assertIn("更多连接", prompt)
        self.assertNotIn("蓝牙", prompt)

    def test_deepseek_planner_retries_malformed_response_before_action(self) -> None:
        model = FlakyStructuredModel()
        action = DeepSeekPlanner(
            model="test-model",
            structured_model=model,
            prefer_single_candidate=False,
        ).plan(GOAL, page(), ())
        self.assertEqual(action.kind, "tap")
        self.assertEqual(model.calls, 2)

    def test_deepseek_planner_validates_raw_json_when_adapter_parser_fails(self) -> None:
        model = FakeStructuredModel(
            {
                "raw": AIMessage(content='{"action":"tap","target":"WLAN"}'),
                "parsed": None,
                "parsing_error": ValueError("adapter parser failed"),
            }
        )
        action = DeepSeekPlanner(
            model="test-model",
            structured_model=model,
            prefer_single_candidate=False,
        ).plan(GOAL, page(), ())
        self.assertEqual(action, AgentAction(
            "tap", "DeepSeek 根据当前受限页面状态选择了该动作。", "WLAN"
        ))

    def test_deepseek_error_does_not_expose_secret_detail(self) -> None:
        model = FakeStructuredModel(error=FakeAPIError(401, "secret-key-value"))
        with self.assertRaisesRegex(AgentError, "凭据无效") as caught:
            DeepSeekPlanner(
                model="test-model",
                structured_model=model,
                prefer_single_candidate=False,
            ).plan(
                GOAL, page(), ()
            )
        self.assertNotIn("secret-key-value", str(caught.exception))

    def test_approval_node_executes_only_after_user_approval(self) -> None:
        approval_goal = AgentGoal(
            "需要确认的 WLAN 导航",
            "com.android.settings",
            "WLAN",
            ("WifiSettings",),
            risk_level="medium",
            requires_confirmation=True,
        )
        actions = FakeActions()
        requests: list[dict[str, object]] = []
        result = LangGraphAgentLoop(
            FakeObserver(
                [
                    page(frame_hash="a"),
                    page(activity=".Settings$WifiSettingsActivity", frame_hash="b"),
                ]
            ),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            approval_handler=lambda request: requests.append(request) or True,
            settle_seconds=0,
        ).run(approval_goal)
        self.assertTrue(result.success)
        self.assertEqual(actions.calls, [("tap", 540, 980)])
        self.assertEqual(requests[0]["risk_level"], "medium")
        self.assertEqual(requests[0]["target"], "WLAN")

    def test_approval_node_rejection_never_executes_action(self) -> None:
        approval_goal = AgentGoal(
            "需要确认的 WLAN 导航",
            "com.android.settings",
            "WLAN",
            ("WifiSettings",),
            risk_level="medium",
            requires_confirmation=True,
        )
        actions = FakeActions()
        result = LangGraphAgentLoop(
            FakeObserver([page()]),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            approval_handler=lambda request: False,
            settle_seconds=0,
        ).run(approval_goal)
        self.assertFalse(result.success)
        self.assertEqual(actions.calls, [])
        self.assertIn("用户拒绝", result.reason)

    def test_approval_node_can_pause_and_resume_same_thread(self) -> None:
        approval_goal = AgentGoal(
            "需要确认的 WLAN 导航",
            "com.android.settings",
            "WLAN",
            ("WifiSettings",),
            risk_level="medium",
            requires_confirmation=True,
        )
        actions = FakeActions()
        loop = LangGraphAgentLoop(
            FakeObserver(
                [
                    page(frame_hash="a"),
                    page(activity=".Settings$WifiSettingsActivity", frame_hash="b"),
                ]
            ),  # type: ignore[arg-type]
            actions,  # type: ignore[arg-type]
            SettingsPlanner(),
            settle_seconds=0,
        )
        with self.assertRaises(AgentApprovalRequired) as paused:
            loop.run(approval_goal, thread_id="approval-test")
        self.assertEqual(actions.calls, [])
        result = loop.resume(paused.exception.thread_id, approved=True)
        self.assertTrue(result.success)
        self.assertEqual(len(actions.calls), 1)


if __name__ == "__main__":
    unittest.main()
