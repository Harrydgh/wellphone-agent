from __future__ import annotations

import unittest

from wellphone_agent.agent import (
    AgentAction,
    AgentGoal,
    AgentLoop,
    AgentSafetyPolicy,
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


class AgentTests(unittest.TestCase):
    def test_planner_taps_visible_goal(self) -> None:
        action = SettingsPlanner().plan(GOAL, page(), ())
        self.assertEqual(action.kind, "tap")
        self.assertEqual(action.target, "WLAN")

    def test_planner_scrolls_when_goal_is_not_visible(self) -> None:
        action = SettingsPlanner().plan(GOAL, page(label=None), ())
        self.assertEqual(action.kind, "scroll_down")

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
