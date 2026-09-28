from __future__ import annotations

import json
import time
from pathlib import Path

from .adb import AdbClient, ensure_connected
from .display import VirtualDisplaySession
from .perception.frame_capture import VirtualDisplayCapture
from .perception.state import AndroidPageInspector, PageStateTracker, UIHierarchyInspector
from .perception.understanding import PageUnderstanding, PageUnderstandingEngine
from .runlog import RunLogger
from .tools import find_adb, find_scrcpy


def run_understanding_demo(
    *,
    project_root: Path,
    preferred_serial: str | None,
    package: str,
    settle_seconds: float = 4.0,
) -> tuple[PageUnderstanding, Path]:
    """Observe and understand one virtual-display frame without injecting input."""

    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    scrcpy = find_scrcpy()
    log = RunLogger(project_root / "logs")
    display = VirtualDisplaySession(scrcpy, device.serial, start_app=package)

    try:
        display_id = display.start()
        log.write("display_ready", display_id=display_id, mode="understand")
        time.sleep(settle_seconds)
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
        page = tracker.observe("understand")
        if page.current_app != package:
            raise RuntimeError(
                f"Expected {package} on display {display_id}, got "
                f"{page.current_app or 'no foreground app'}."
            )

        understanding = PageUnderstandingEngine().analyze(page)
        # Only redacted text is logged. The model-safe payload deliberately excludes
        # screenshots and device identifiers.
        log.write(
            "page_understood",
            display_id=display_id,
            screenshot=understanding.screenshot,
            text_source=understanding.text_source,
            ocr_count=understanding.ocr_count,
            model_payload=understanding.to_model_payload(),
        )
        log.write("run_completed", success=True, actions_executed=0)
        return understanding, log.path
    except Exception as exc:
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        display.stop()
        log.write("display_stopped")


def format_understanding_result(
    understanding: PageUnderstanding, log_path: Path
) -> str:
    result = understanding.to_dict()
    result["element_count"] = len(understanding.elements)
    result["elements"] = [item.to_dict() for item in understanding.elements[:30]]
    result["candidate_count"] = len(understanding.candidates)
    result["safety"] = {
        "actions_executed": 0,
        "model_called": False,
        "ocr_only_candidates_executable": False,
    }
    result["model_payload_preview"] = understanding.to_model_payload(max_elements=30)
    result["log"] = str(log_path)
    return json.dumps(result, ensure_ascii=False, indent=2)
