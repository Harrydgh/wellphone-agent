from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path

from wellphone_agent.live import LiveDisplayViewer, LiveProgressReporter
from wellphone_agent.runlog import RunLogger


class LiveDisplayTests(unittest.TestCase):
    def test_viewer_is_bound_to_secondary_display_and_read_only(self) -> None:
        viewer = LiveDisplayViewer(
            Path("scrcpy.exe"),
            "192.168.1.22:12345",
            13,
        )
        self.assertIn("--serial=192.168.1.22:12345", viewer.command)
        self.assertIn("--display-id=13", viewer.command)
        self.assertIn("--no-control", viewer.command)
        self.assertIn("--no-audio", viewer.command)
        self.assertNotIn("--new-display", viewer.command)

    def test_viewer_rejects_primary_display(self) -> None:
        with self.assertRaisesRegex(ValueError, "secondary"):
            LiveDisplayViewer(Path("scrcpy.exe"), "device", 0)

    def test_progress_reports_safe_structured_events(self) -> None:
        output = io.StringIO()
        reporter = LiveProgressReporter(output=output)
        reporter.handle(
            {
                "event": "task_observed",
                "screen": {"kind": "results"},
                "task_candidates": [{"candidate_id": "safe"}],
            }
        )
        reporter.handle(
            {
                "event": "task_decision",
                "step": 2,
                "reason": "选择第一个真实商品。",
            }
        )
        reporter.handle(
            {
                "event": "task_action_executing",
                "action": "input_query",
                "target": None,
            }
        )
        text = output.getvalue()
        self.assertIn("[观察] 当前页面：results；候选控件：1。", text)
        self.assertIn("[计划] 第 2 步：选择第一个真实商品。", text)
        self.assertIn("[执行] 输入并提交搜索词", text)

    def test_heartbeat_distinguishes_processing_and_confirmation(self) -> None:
        now = [0.0]
        output = io.StringIO()
        reporter = LiveProgressReporter(
            output=output,
            heartbeat_seconds=5,
            clock=lambda: now[0],
        )
        now[0] = 6
        reporter.heartbeat_once()
        reporter.handle({"event": "task_approval_required", "target": "商品"})
        now[0] = 12
        reporter.heartbeat_once()
        text = output.getvalue()
        self.assertIn("[处理中]", text)
        self.assertIn("[等待]", text)

    def test_run_logger_forwards_record_to_listener(self) -> None:
        records: list[dict[str, object]] = []
        with tempfile.TemporaryDirectory() as directory:
            logger = RunLogger(Path(directory), listener=records.append)
            logger.write("task_decision", step=3, reason="test")
            saved = json.loads(logger.path.read_text(encoding="utf-8"))
        self.assertEqual(records[0]["event"], "task_decision")
        self.assertEqual(records[0]["step"], 3)
        self.assertEqual(saved["reason"], "test")


if __name__ == "__main__":
    unittest.main()
