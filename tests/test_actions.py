from __future__ import annotations

import subprocess
import unittest

from wellphone_agent.actions import ActionController, UnsafeActionError


class FakeAdb:
    def __init__(self) -> None:
        self.shell_calls: list[tuple[object, ...]] = []
        self.run_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.activity_output = ""

    def shell(self, *args: object, **kwargs: object) -> str:
        self.shell_calls.append(args)
        if "dumpsys" in args:
            return self.activity_output
        return ""

    def run(self, *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        self.run_calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, "Starting: Intent", "")


class ActionControllerTests(unittest.TestCase):
    def test_rejects_primary_display(self) -> None:
        with self.assertRaises(UnsafeActionError):
            ActionController(FakeAdb(), "device", 0)  # type: ignore[arg-type]

    def test_tap_is_always_targeted_to_virtual_display(self) -> None:
        adb = FakeAdb()
        controller = ActionController(adb, "device", 7)  # type: ignore[arg-type]
        controller.tap(100, 200)
        self.assertEqual(
            adb.shell_calls[0],
            (
                "device",
                "input",
                "touchscreen",
                "-d",
                "7",
                "tap",
                "100",
                "200",
            ),
        )

    def test_rejects_unsafe_url_scheme(self) -> None:
        controller = ActionController(FakeAdb(), "device", 7)  # type: ignore[arg-type]
        with self.assertRaises(UnsafeActionError):
            controller.open_url("file:///data/local/tmp/private", "com.android.browser")

    def test_verifies_package_inside_requested_display_section(self) -> None:
        adb = FakeAdb()
        adb.activity_output = """
Display: mDisplayId=7 rootTasks=1
  ResumedActivity: com.android.browser/.BrowserActivity
Display: mDisplayId=0 rootTasks=2
  ResumedActivity: com.example.user/.MainActivity
"""
        controller = ActionController(adb, "device", 7)  # type: ignore[arg-type]
        self.assertTrue(controller.app_is_on_display("com.android.browser"))
        self.assertFalse(controller.app_is_on_display("com.example.user"))


if __name__ == "__main__":
    unittest.main()
