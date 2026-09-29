from __future__ import annotations

import time
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .agent import (
    DeepSeekTaskPlanner,
    LangGraphAppTaskLoop,
    TaskRunResult,
    meituan_food_task,
)
from .perception.frame_capture import VirtualDisplayCapture
from .perception.state import AndroidPageInspector, PageStateTracker, UIHierarchyInspector
from .perception.understanding import PageUnderstandingEngine
from .runlog import RunLogger
from .scrcpy_control import ScrcpyControlSession
from .tools import find_adb, find_scrcpy


def run_meituan_agent(
    *,
    project_root: Path,
    preferred_serial: str | None,
    query: str,
    model: str,
) -> tuple[TaskRunResult, Path]:
    task = meituan_food_task(query)
    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    scrcpy = find_scrcpy()
    log = RunLogger(project_root / "logs")
    session = ScrcpyControlSession(adb, scrcpy, device.serial)

    def request_approval(request: dict[str, object]) -> bool:
        print("\nAgent 请求执行经过页面定位的动作：")
        print(f"任务: {request.get('task')}")
        print(f"目标: {request.get('target')}")
        print(f"来源: {request.get('source')}")
        print(f"风险: {request.get('risk_level')}")
        print(f"原因: {request.get('reason')}")
        try:
            answer = input("请输入 yes 确认，其他内容拒绝: ")
        except EOFError:
            return False
        return answer.strip().lower() in {"yes", "y"}

    try:
        display_id = session.start()
        log.write("display_ready", display_id=display_id, mode="meituan_agent")
        session.start_app(task.allowed_package)
        log.write("action", action="start_app", package=task.allowed_package)
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
        loop = LangGraphAppTaskLoop(
            tracker,
            PageUnderstandingEngine(),
            ActionController(adb, device.serial, display_id, session),
            DeepSeekTaskPlanner(model=model),
            logger=log,
            approval_handler=request_approval,
        )
        result = loop.run(task)
        if not result.success:
            raise RuntimeError(result.reason)
        return result, log.path
    except Exception as exc:
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
        raise
    finally:
        session.stop()
        log.write("display_stopped")
