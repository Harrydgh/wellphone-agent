from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .agent import (
    DeepSeekTaskPlanner,
    LangGraphAppTaskLoop,
    TaskRunResult,
    meituan_food_task,
)
from .concurrency import MainDisplayMonitor, write_acceptance_report
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
    monitor_main_display: bool = False,
) -> tuple[TaskRunResult, Path, Path | None]:
    task = meituan_food_task(query)
    adb = AdbClient(find_adb())
    device = ensure_connected(adb, preferred_serial)
    scrcpy = find_scrcpy()
    log = RunLogger(project_root / "logs")
    session = ScrcpyControlSession(adb, scrcpy, device.serial)
    monitor: MainDisplayMonitor | None = None
    report_path: Path | None = None
    result: TaskRunResult | None = None
    failure: Exception | None = None
    display_id: int | None = None
    started_at = datetime.now(timezone.utc)
    started_clock = time.monotonic()

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
        inspector = AndroidPageInspector(adb, device.serial)
        if monitor_main_display:
            monitor = MainDisplayMonitor(
                inspector,
                task.allowed_package,
                logger=log,
            )
            monitor.start()
            monitor.assert_safe()
        # Restart only the task package (without clearing account or cart data)
        # so a previous failed run cannot resume inside a stale loading dialog.
        adb.shell(
            device.serial,
            "am",
            "force-stop",
            task.allowed_package,
        )
        log.write("action", action="force_stop_app", package=task.allowed_package)
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
            inspector,
            UIHierarchyInspector(adb, device.serial),
        )
        loop = LangGraphAppTaskLoop(
            tracker,
            PageUnderstandingEngine(),
            ActionController(adb, device.serial, display_id, session),
            DeepSeekTaskPlanner(model=model),
            logger=log,
            approval_handler=request_approval,
            pre_action_guard=monitor.assert_safe if monitor is not None else None,
        )
        result = loop.run(task)
        if not result.success:
            raise RuntimeError(result.reason)
        if monitor is not None:
            monitor.assert_safe()
    except Exception as exc:
        failure = exc
        log.write("run_failed", error=type(exc).__name__, detail=str(exc))
    finally:
        if monitor is not None:
            monitor.stop()
        try:
            session.stop()
            log.write("display_stopped")
        except Exception as exc:
            log.write("display_stop_failed", error=type(exc).__name__)
            if failure is None:
                failure = exc
        if monitor_main_display:
            finished_at = datetime.now(timezone.utc)
            snapshot = monitor.snapshot() if monitor is not None else None
            if (
                failure is None
                and snapshot is not None
                and snapshot.violation_reason is not None
            ):
                failure = RuntimeError(snapshot.violation_reason)
            final_understanding = result.final_understanding if result else None
            payload: dict[str, object] = {
                "schema_version": 1,
                "status": (
                    "passed"
                    if failure is None
                    and result is not None
                    and result.success
                    and snapshot is not None
                    and snapshot.violation_reason is None
                    else "failed"
                ),
                "reason": (
                    str(failure)
                    if failure is not None
                    else result.reason if result is not None else "任务未启动。"
                ),
                "query": task.query,
                "task_package": task.allowed_package,
                "started_at": started_at.isoformat(),
                "finished_at": finished_at.isoformat(),
                "duration_seconds": round(time.monotonic() - started_clock, 3),
                "virtual_display_id": display_id,
                "steps": result.steps if result is not None else 0,
                "final_screen": (
                    final_understanding.screen.kind if final_understanding else None
                ),
                "final_screenshot": (
                    final_understanding.screenshot if final_understanding else None
                ),
                "run_log": str(log.path),
                "history": (
                    [item.model_dump(mode="json") for item in result.history]
                    if result
                    else []
                ),
                "main_display": snapshot.to_dict() if snapshot else None,
            }
            report_path = write_acceptance_report(
                project_root / "reports", payload
            )
            log.write("acceptance_report_written", path=str(report_path))

    if failure is not None:
        detail = str(failure)
        if report_path is not None:
            detail = f"{detail} 验收报告：{report_path}"
        raise RuntimeError(detail) from failure
    if result is None:
        raise RuntimeError("美团任务未返回结果。")
    return result, log.path, report_path
