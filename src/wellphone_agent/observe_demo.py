from __future__ import annotations

import json
import time
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .display import VirtualDisplaySession
from .perception.frame_capture import VirtualDisplayCapture
from .perception.state import (
    AndroidPageInspector,
    PageStateTracker,
    UIHierarchyInspector,
)
from .runlog import RunLogger
from .tools import find_adb, find_scrcpy


def run_observation_demo(
    *,
    project_root: Path,
    preferred_serial: str | None,
    package: str,
    url: str,
) -> tuple[dict[str, object], dict[str, object], Path]:
    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    scrcpy = find_scrcpy()
    log = RunLogger(project_root / "logs")
    display = VirtualDisplaySession(scrcpy, device.serial, start_app=package)

    try:
        display_id = display.start()
        log.write("display_ready", display_id=display_id, mode="observation_demo")
        actions = ActionController(adb, device.serial, display_id)
        actions.open_url(url, package)
        time.sleep(4)

        tracker = PageStateTracker(
            VirtualDisplayCapture(
                scrcpy,
                device.serial,
                display_id,
                project_root / "screenshots",
            ),
            AndroidPageInspector(adb, device.serial),
            UIHierarchyInspector(adb, device.serial),
        )
        initial = tracker.observe("initial")
        log.write("page_observed", **initial.to_dict())

        if initial.current_app != package:
            raise RuntimeError(
                f"Expected {package} on display {display_id}, got "
                f"{initial.current_app or 'no foreground app'}."
            )
        if initial.width <= 0 or initial.height <= 0:
            raise RuntimeError("The captured virtual-screen frame has an invalid size.")

        # Capture a second observation without injecting input. On MIUI 14, ADB's
        # `input -d` command may return success while not controlling a scrcpy-created
        # virtual display. Input will be moved to scrcpy's control channel in the next
        # stage; this command deliberately verifies perception only.
        time.sleep(2)
        before = initial
        after = tracker.observe("verification")
        log.write("page_observed", **after.to_dict())
        log.write("run_completed", success=True)
        return before.to_dict(), after.to_dict(), log.path
    except Exception as exc:
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        display.stop()
        log.write("display_stopped")


def format_observation_result(
    before: dict[str, object], after: dict[str, object], log_path: Path
) -> str:
    def summarize(state: dict[str, object]) -> dict[str, object]:
        elements = state.get("ui_elements")
        texts = state.get("visible_text")
        summary = dict(state)
        summary["visible_text"] = list(texts[:20]) if isinstance(texts, list) else []
        summary["visible_text_count"] = len(texts) if isinstance(texts, list) else 0
        summary["ui_elements"] = list(elements[:10]) if isinstance(elements, list) else []
        summary["ui_element_count"] = len(elements) if isinstance(elements, list) else 0
        return summary

    return json.dumps(
        {
            "before": summarize(before),
            "after": summarize(after),
            "log": str(log_path),
        },
        ensure_ascii=False,
        indent=2,
    )
