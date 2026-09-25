from __future__ import annotations

import unittest
from pathlib import Path

from wellphone_agent.display import VirtualDisplaySession


class VirtualDisplaySessionTests(unittest.TestCase):
    def test_command_contains_device_and_isolated_display_options(self) -> None:
        session = VirtualDisplaySession(
            Path("scrcpy.exe"),
            "192.168.1.22:12345",
            size="1080x1920",
            dpi=420,
            start_app="com.android.browser",
        )
        self.assertIn("--serial=192.168.1.22:12345", session.command)
        self.assertIn("--new-display=1080x1920/420", session.command)
        self.assertIn("--display-ime-policy=local", session.command)
        self.assertIn("--start-app=com.android.browser", session.command)

    def test_parses_dynamic_display_id(self) -> None:
        line = "[server] INFO: New display: 1080x1920/420 (id=12)"
        match = VirtualDisplaySession.DISPLAY_PATTERN.search(line)
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(match.group(1), "12")


if __name__ == "__main__":
    unittest.main()

