from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from wellphone_agent.perception.frame_capture import (
    FrameCaptureError,
    FrameSnapshot,
    VirtualDisplayCapture,
    image_perceptual_hash,
    perceptual_distance,
)
from wellphone_agent.perception.state import (
    AndroidPageInspector,
    PageState,
    PageStateTracker,
    UIHierarchyInspector,
    VisibleElement,
)


class FakeAdb:
    def __init__(self, output: str) -> None:
        self.output = output

    def shell(self, *args: object, **kwargs: object) -> str:
        return self.output


class FakeCapture:
    def __init__(self, hashes: list[str], directory: Path) -> None:
        self.hashes = iter(hashes)
        self.directory = directory

    def capture(self, label: str = "frame") -> FrameSnapshot:
        value = next(self.hashes)
        return FrameSnapshot(
            path=self.directory / f"{label}.png",
            captured_at="2026-09-25T00:00:00+00:00",
            width=1080,
            height=1920,
            sha256=value,
            perceptual_hash=value,
            display_id=7,
        )


class PerceptionTests(unittest.TestCase):
    def test_perceptual_hash_detects_visual_change(self) -> None:
        first = Image.new("RGB", (100, 100), "white")
        second = first.copy()
        draw = ImageDraw.Draw(second)
        draw.rectangle((50, 0, 99, 99), fill="black")
        first_hash = image_perceptual_hash(first)
        second_hash = image_perceptual_hash(second)
        self.assertEqual(perceptual_distance(first_hash, first_hash), 0.0)
        self.assertGreater(perceptual_distance(first_hash, second_hash), 0.0)

    def test_capture_rejects_primary_display(self) -> None:
        with self.assertRaises(FrameCaptureError):
            VirtualDisplayCapture(
                Path("scrcpy.exe"), "device", 0, Path("screenshots")
            )

    def test_inspector_reads_only_requested_display(self) -> None:
        output = """
Display #7 (activities from top to bottom):
  topResumedActivity=ActivityRecord{123 u0 com.android.browser/.BrowserActivity t1}
Display #0 (activities from top to bottom):
  topResumedActivity=ActivityRecord{456 u0 com.user.chat/.MainActivity t2}
"""
        inspector = AndroidPageInspector(FakeAdb(output), "device")  # type: ignore[arg-type]
        package, activity = inspector.current_app_activity(7)
        self.assertEqual(package, "com.android.browser")
        self.assertEqual(activity, ".BrowserActivity")

    def test_ui_hierarchy_extracts_text_and_actionable_elements(self) -> None:
        xml = """<?xml version='1.0' encoding='UTF-8'?>
<hierarchy rotation="0">
  <node text="Example Domain" resource-id="title" class="android.widget.TextView"
        package="com.android.browser" content-desc="" clickable="false"
        enabled="true" scrollable="false" bounds="[0,0][500,100]" />
  <node text="" resource-id="learn" class="android.widget.Button"
        package="com.android.browser" content-desc="Learn more" clickable="true"
        enabled="true" scrollable="false" bounds="[0,100][500,200]" />
</hierarchy>"""

        class HierarchyAdb(FakeAdb):
            def shell(self, *args: object, **kwargs: object) -> str:
                return xml if "cat" in args else ""

        inspector = UIHierarchyInspector(  # type: ignore[arg-type]
            HierarchyAdb(""), "device"
        )
        texts, elements, source = inspector.inspect("com.android.browser")
        self.assertEqual(source, "uiautomator")
        self.assertIn("Example Domain", texts)
        self.assertIn("Learn more", texts)
        self.assertEqual(len(elements), 2)
        self.assertTrue(elements[1].clickable)

    def test_page_state_returns_clickable_element_center(self) -> None:
        state = PageState(
            captured_at="2026-09-25T00:00:00+00:00",
            serial="device",
            display_id=7,
            current_app="com.android.browser",
            current_activity=".BrowserActivity",
            screenshot="frame.png",
            width=1080,
            height=1920,
            sha256="0" * 64,
            perceptual_hash="0" * 16,
            changed_from_previous=None,
            visual_change_ratio=None,
            consecutive_static_frames=0,
            is_stale=False,
            ui_elements=(
                VisibleElement(
                    text="拒绝",
                    content_description="",
                    resource_id="left_button",
                    class_name="android.widget.Button",
                    bounds="[80,1530][522,1668]",
                    clickable=True,
                    enabled=True,
                ),
            ),
        )
        self.assertEqual(state.clickable_center("拒绝"), (301, 1599))
        self.assertIsNone(state.resource_center("missing"))

    def test_clickable_parent_inherits_descendant_label(self) -> None:
        xml = """<?xml version='1.0' encoding='UTF-8'?>
<hierarchy rotation="0">
  <node text="" resource-id="row" class="android.view.ViewGroup"
        package="com.android.settings" content-desc="" clickable="true"
        enabled="true" scrollable="false" bounds="[0,500][1080,650]">
    <node text="WLAN" resource-id="title" class="android.widget.TextView"
          package="com.android.settings" content-desc="" clickable="false"
          enabled="true" scrollable="false" bounds="[100,520][400,620]" />
  </node>
</hierarchy>"""

        class HierarchyAdb(FakeAdb):
            def shell(self, *args: object, **kwargs: object) -> str:
                return xml if "cat" in args else ""

        inspector = UIHierarchyInspector(  # type: ignore[arg-type]
            HierarchyAdb(""), "device"
        )
        _, elements, _ = inspector.inspect("com.android.settings")
        self.assertEqual(elements[0].text, "WLAN")
        self.assertTrue(elements[0].clickable)

    def test_tracker_marks_repeated_frames_as_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            capture = FakeCapture(["0" * 16] * 4, Path(directory))
            inspector = AndroidPageInspector(FakeAdb(""), "device")  # type: ignore[arg-type]
            tracker = PageStateTracker(  # type: ignore[arg-type]
                capture, inspector, stale_after=3
            )
            states = [tracker.observe() for _ in range(4)]
        self.assertIsNone(states[0].changed_from_previous)
        self.assertFalse(states[2].is_stale)
        self.assertTrue(states[3].is_stale)


if __name__ == "__main__":
    unittest.main()
