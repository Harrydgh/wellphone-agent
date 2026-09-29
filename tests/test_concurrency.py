from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from wellphone_agent.agent.core import AgentError
from wellphone_agent.concurrency import MainDisplayMonitor, write_acceptance_report


class SequenceInspector:
    def __init__(self, states: list[object]) -> None:
        self.states = states

    def current_app_activity(self, display_id: int) -> tuple[str | None, str | None]:
        assert display_id == 0
        state = self.states.pop(0)
        if isinstance(state, Exception):
            raise state
        return state  # type: ignore[return-value]


class MainDisplayMonitorTests(unittest.TestCase):
    def test_allows_user_to_switch_between_unrelated_main_screen_apps(self) -> None:
        inspector = SequenceInspector(
            [
                ("com.tencent.mm", ".LauncherUI"),
                ("com.android.browser", ".BrowserActivity"),
            ]
        )
        monitor = MainDisplayMonitor(
            inspector,
            "com.sankuai.meituan",
            interval_seconds=60,
        )
        monitor.sample_once()
        monitor.assert_safe()
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot.app_transitions, 1)
        self.assertEqual(snapshot.protected_app_seen, 0)

    def test_rejects_task_app_on_main_display(self) -> None:
        inspector = SequenceInspector(
            [("com.sankuai.meituan", ".MainActivity")]
        )
        monitor = MainDisplayMonitor(
            inspector,
            "com.sankuai.meituan",
            interval_seconds=60,
        )
        with self.assertRaisesRegex(AgentError, "主屏"):
            monitor.assert_safe()
        self.assertEqual(monitor.snapshot().protected_app_seen, 1)

    def test_fails_closed_after_repeated_monitor_errors(self) -> None:
        inspector = SequenceInspector(
            [RuntimeError("offline"), RuntimeError("offline"), RuntimeError("offline")]
        )
        monitor = MainDisplayMonitor(
            inspector,
            "com.sankuai.meituan",
            interval_seconds=60,
            max_consecutive_errors=3,
        )
        monitor.sample_once()
        monitor.sample_once()
        with self.assertRaisesRegex(AgentError, "连续失败"):
            monitor.assert_safe()

    def test_writes_machine_readable_acceptance_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = write_acceptance_report(
                Path(directory), {"status": "passed", "steps": 4}
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload, {"status": "passed", "steps": 4})
        self.assertTrue(path.name.startswith("concurrency-acceptance-"))


if __name__ == "__main__":
    unittest.main()
