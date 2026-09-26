from __future__ import annotations

import time
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .agent import AgentGoal, AgentLoop, AgentResult, SettingsPlanner
from .perception.frame_capture import VirtualDisplayCapture
from .perception.state import AndroidPageInspector, PageStateTracker, UIHierarchyInspector
from .runlog import RunLogger
from .scrcpy_control import ScrcpyControlSession
from .tools import find_adb, find_scrcpy


SAFE_SETTINGS_GOALS = {
    "WLAN": AgentGoal(
        description="打开系统 WLAN 设置页面",
        allowed_package="com.android.settings",
        target_label="WLAN",
        success_activity_contains=("WifiSettings",),
    ),
}


def run_agent_demo(
    *, project_root: Path, preferred_serial: str | None, target: str
) -> tuple[AgentResult, Path]:
    if target not in SAFE_SETTINGS_GOALS:
        raise ValueError(f"当前安全演示不支持目标：{target}")
    goal = SAFE_SETTINGS_GOALS[target]
    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    scrcpy = find_scrcpy()
    log = RunLogger(project_root / "logs")
    session = ScrcpyControlSession(adb, scrcpy, device.serial)
    try:
        display_id = session.start()
        log.write("display_ready", display_id=display_id, mode="agent_loop")
        session.start_app(goal.allowed_package)
        log.write("action", action="start_app", package=goal.allowed_package)
        time.sleep(4)
        tracker = PageStateTracker(
            VirtualDisplayCapture(
                scrcpy, device.serial, display_id, project_root / "screenshots"
            ),
            AndroidPageInspector(adb, device.serial),
            UIHierarchyInspector(adb, device.serial),
        )
        loop = AgentLoop(
            tracker,
            ActionController(adb, device.serial, display_id, session),
            SettingsPlanner(),
            logger=log,
        )
        result = loop.run(goal)
        if not result.success:
            raise RuntimeError(result.reason)
        return result, log.path
    except Exception as exc:
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        session.stop()
        log.write("display_stopped")
