from __future__ import annotations

import time
from pathlib import Path

from .actions import ActionController
from .adb import AdbClient, ensure_connected
from .agent import (
    AgentGoal,
    AgentLoop,
    AgentResult,
    DeepSeekPlanner,
    LangGraphAgentLoop,
    OpenAIPlanner,
    SAFE_SETTINGS_GOALS,
    SettingsPlanner,
    parse_safe_goal,
)
from .agent.core import AgentPlanner
from .agent.langgraph_workflow import ApprovalHandler
from .perception.frame_capture import VirtualDisplayCapture
from .perception.state import AndroidPageInspector, PageStateTracker, UIHierarchyInspector
from .runlog import RunLogger
from .scrcpy_control import ScrcpyControlSession
from .tools import find_adb, find_scrcpy


def _run_agent(
    *,
    project_root: Path,
    preferred_serial: str | None,
    goal: AgentGoal,
    planner: AgentPlanner,
    engine: str = "legacy",
    approval_handler: ApprovalHandler | None = None,
) -> tuple[AgentResult, Path]:
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
        loop_class = LangGraphAgentLoop if engine == "langgraph" else AgentLoop
        loop_options: dict[str, object] = {"logger": log}
        if engine == "langgraph":
            loop_options["approval_handler"] = approval_handler
        loop = loop_class(
            tracker,
            ActionController(adb, device.serial, display_id, session),
            planner,
            **loop_options,
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


def run_agent_demo(
    *, project_root: Path, preferred_serial: str | None, target: str
) -> tuple[AgentResult, Path]:
    if target not in SAFE_SETTINGS_GOALS:
        raise ValueError(f"当前安全演示不支持目标：{target}")
    return _run_agent(
        project_root=project_root,
        preferred_serial=preferred_serial,
        goal=SAFE_SETTINGS_GOALS[target],
        planner=SettingsPlanner(),
    )


def run_ai_agent_demo(
    *,
    project_root: Path,
    preferred_serial: str | None,
    task: str,
    model: str,
) -> tuple[AgentResult, Path]:
    return _run_agent(
        project_root=project_root,
        preferred_serial=preferred_serial,
        goal=parse_safe_goal(task),
        planner=OpenAIPlanner(model=model),
    )


def run_langgraph_agent_demo(
    *,
    project_root: Path,
    preferred_serial: str | None,
    task: str,
    model: str,
) -> tuple[AgentResult, Path]:
    def request_approval(request: dict[str, object]) -> bool:
        print("\nAgent 请求执行需要确认的动作：")
        print(f"任务: {request.get('task')}")
        print(f"风险: {request.get('risk_level')}")
        print(f"动作: {request.get('action')} {request.get('target') or ''}")
        print(f"原因: {request.get('reason')}")
        try:
            answer = input("是否允许？请输入 yes 确认，其他内容拒绝: ")
        except EOFError:
            return False
        return answer.strip().lower() in {"yes", "y"}

    return _run_agent(
        project_root=project_root,
        preferred_serial=preferred_serial,
        goal=parse_safe_goal(task),
        planner=DeepSeekPlanner(model=model),
        engine="langgraph",
        approval_handler=request_approval,
    )
