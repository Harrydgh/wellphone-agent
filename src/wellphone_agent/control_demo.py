from __future__ import annotations

import time
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .observe_demo import format_observation_result
from .perception.frame_capture import VirtualDisplayCapture
from .perception.state import AndroidPageInspector, PageStateTracker, UIHierarchyInspector
from .runlog import RunLogger
from .scrcpy_control import ScrcpyControlSession
from .tools import find_adb, find_scrcpy


def run_control_demo(
    *,
    project_root: Path,
    preferred_serial: str | None,
    package: str,
) -> tuple[dict[str, object], dict[str, object], Path]:
    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    scrcpy = find_scrcpy()
    log = RunLogger(project_root / "logs")
    session = ScrcpyControlSession(adb, scrcpy, device.serial)

    try:
        display_id = session.start()
        log.write("display_ready", display_id=display_id, mode="scrcpy_control")
        actions = ActionController(adb, device.serial, display_id, session)
        session.start_app(package)
        log.write("action", action="start_app", package=package)
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
        state = tracker.observe("control-initial")
        log.write("page_observed", **state.to_dict())
        if state.current_app != package:
            raise RuntimeError(
                f"Expected {package} on display {display_id}, got "
                f"{state.current_app or 'no foreground app'}."
            )

        tap_target: tuple[str, tuple[int, int]] | None = None
        for label in ("WLAN", "蓝牙", "我的设备"):
            center = state.clickable_center(label)
            if center is not None:
                tap_target = (label, center)
                break
        if tap_target is None:
            raise RuntimeError("Could not find a safe Settings navigation row to tap.")

        label, center = tap_target
        actions.tap(*center)
        log.write(
            "action",
            action="tap",
            display_id=display_id,
            target=label,
            coordinates=list(center),
        )
        time.sleep(2)
        state_after_tap = tracker.observe("control-after-tap")
        log.write("page_observed", **state_after_tap.to_dict())
        if (
            state_after_tap.changed_from_previous is not True
            and state.current_activity == state_after_tap.current_activity
            and state.visible_text == state_after_tap.visible_text
        ):
            raise RuntimeError(
                "The scrcpy tap produced no observable page change. Server output: "
                + " | ".join(session.recent_output())
            )

        actions.keyevent("KEYCODE_BACK")
        log.write("action", action="back", display_id=display_id)
        time.sleep(2)
        before = tracker.observe("control-after-back")
        log.write("page_observed", **before.to_dict())
        if (
            before.current_activity == state_after_tap.current_activity
            and before.visible_text == state_after_tap.visible_text
        ):
            raise RuntimeError("The scrcpy Back key produced no page-state change.")

        actions.swipe(540, 1500, 540, 650, 600)
        log.write(
            "action",
            action="swipe",
            display_id=display_id,
            start=[540, 1500],
            end=[540, 650],
            duration_ms=600,
        )
        time.sleep(2)
        after = tracker.observe("control-after-swipe")
        log.write("page_observed", **after.to_dict())

        changed_text = before.visible_text != after.visible_text
        if after.changed_from_previous is not True and not changed_text:
            raise RuntimeError(
                "The scrcpy swipe was sent, but no page-state change was observed."
            )
        log.write(
            "control_verified",
            success=True,
            tap_target=label,
            back_verified=True,
            visual_change_ratio=after.visual_change_ratio,
            visible_text_changed=changed_text,
        )
        log.write("run_completed", success=True)
        return before.to_dict(), after.to_dict(), log.path
    except Exception as exc:
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        session.stop()
        log.write("display_stopped")


def format_control_result(
    before: dict[str, object], after: dict[str, object], log_path: Path
) -> str:
    return format_observation_result(before, after, log_path)
