from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .actions import UnsafeActionError
from .adb import AdbClient, AdbError, ensure_connected
from .agent.core import AgentError
from .agent import SAFE_SETTINGS_GOALS
from .agent_demo import run_agent_demo, run_ai_agent_demo, run_langgraph_agent_demo
from .control_demo import format_control_result, run_control_demo
from .demo import run_browser_demo
from .display import VirtualDisplayError, VirtualDisplaySession
from .observe_demo import format_observation_result, run_observation_demo
from .perception.frame_capture import FrameCaptureError
from .scrcpy_control import ScrcpyControlError
from .tools import ToolNotFoundError, find_adb, find_scrcpy


def command_doctor() -> int:
    adb_path = find_adb()
    scrcpy_path = find_scrcpy()
    preferred = os.environ.get("WELLPHONE_SERIAL") or None

    print(f"ADB:     {adb_path}")
    print(f"scrcpy:  {scrcpy_path}")
    client = AdbClient(adb_path)
    device = ensure_connected(client, preferred)
    info = client.device_info(device.serial)

    print(f"Serial:  {info.serial}")
    print(f"Model:   {info.model}")
    print(f"Android: {info.android_version} (SDK {info.sdk})")
    print(f"MIUI:    {info.miui_version or 'not detected'}")
    print("Browsers:")
    if info.browser_packages:
        for package in info.browser_packages:
            print(f"  - {package}")
    else:
        print("  - none detected")
    print("Status:  READY")
    return 0


def command_display_test(duration: float, app: str) -> int:
    adb_path = find_adb()
    scrcpy_path = find_scrcpy()
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    client = AdbClient(adb_path)
    device = ensure_connected(client, preferred)

    size = os.environ.get("WELLPHONE_DISPLAY_SIZE", "1080x1920")
    dpi = int(os.environ.get("WELLPHONE_DISPLAY_DPI", "420"))
    session = VirtualDisplaySession(
        scrcpy_path,
        device.serial,
        size=size,
        dpi=dpi,
        start_app=app,
    )
    try:
        display_id = session.start()
        print(f"Virtual display ready: display {display_id}")
        print(f"App: {app}")
        print(f"Keeping it open for {duration:g} seconds...")
        time.sleep(duration)
        print("Display test completed.")
        return 0
    finally:
        session.stop()


def command_demo(url: str, hold: float, app: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    log_path = run_browser_demo(
        project_root=project_root,
        preferred_serial=preferred,
        package=app,
        url=url,
        hold_seconds=hold,
    )
    print("Browser demo completed successfully.")
    print(f"Log: {log_path}")
    return 0


def command_observe(url: str, app: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    before, after, log_path = run_observation_demo(
        project_root=project_root,
        preferred_serial=preferred,
        package=app,
        url=url,
    )
    print(format_observation_result(before, after, log_path))
    return 0


def command_control_test(app: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    before, after, log_path = run_control_demo(
        project_root=project_root,
        preferred_serial=preferred,
        package=app,
    )
    print(format_control_result(before, after, log_path))
    return 0


def command_agent_test(target: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    result, log_path = run_agent_demo(
        project_root=project_root,
        preferred_serial=preferred,
        target=target,
    )
    print("Agent goal completed successfully.")
    print(f"Goal:  {target}")
    print(f"Steps: {result.steps}")
    print(f"Final activity: {result.final_state.current_activity}")
    print(f"Screenshot: {result.final_state.screenshot}")
    print(f"Log: {log_path}")
    return 0


def command_ai_agent(task: str, model: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    result, log_path = run_ai_agent_demo(
        project_root=project_root,
        preferred_serial=preferred,
        task=task,
        model=model,
    )
    print("AI Agent goal completed successfully.")
    print(f"Task:  {task}")
    print(f"Model: {model}")
    print(f"Steps: {result.steps}")
    print(f"Final activity: {result.final_state.current_activity}")
    print(f"Screenshot: {result.final_state.screenshot}")
    print(f"Log: {log_path}")
    return 0


def command_langgraph_agent(task: str, model: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    result, log_path = run_langgraph_agent_demo(
        project_root=project_root,
        preferred_serial=preferred,
        task=task,
        model=model,
    )
    print("LangGraph DeepSeek Agent goal completed successfully.")
    print(f"Task:  {task}")
    print(f"Model: {model}")
    print(f"Steps: {result.steps}")
    print(f"Final activity: {result.final_state.current_activity}")
    print(f"Screenshot: {result.final_state.screenshot}")
    print(f"Log: {log_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="wellphone")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="discover and inspect an Android device")
    display = subparsers.add_parser(
        "display-test", help="create a temporary scrcpy virtual display"
    )
    display.add_argument("--duration", type=float, default=10.0)
    display.add_argument(
        "--app",
        default=os.environ.get("WELLPHONE_BROWSER_PACKAGE", "com.android.browser"),
    )
    demo = subparsers.add_parser(
        "demo", help="run the fixed browser automation on a virtual display"
    )
    demo.add_argument("--url", default="https://example.com")
    demo.add_argument("--hold", type=float, default=5.0)
    demo.add_argument(
        "--app",
        default=os.environ.get("WELLPHONE_BROWSER_PACKAGE", "com.android.browser"),
    )
    observe = subparsers.add_parser(
        "observe", help="capture and compare two virtual-screen page states"
    )
    observe.add_argument("--url", default="https://www.baidu.com/s?wd=Android")
    observe.add_argument(
        "--app",
        default=os.environ.get("WELLPHONE_BROWSER_PACKAGE", "com.android.browser"),
    )
    control_test = subparsers.add_parser(
        "control-test", help="verify scrcpy input on an isolated virtual display"
    )
    control_test.add_argument(
        "--app",
        default="com.android.settings",
    )
    agent_test = subparsers.add_parser(
        "agent-test", help="run the safe goal-driven Settings agent"
    )
    agent_test.add_argument(
        "--target", choices=tuple(SAFE_SETTINGS_GOALS), default="WLAN"
    )
    ai_agent = subparsers.add_parser(
        "ai-agent", help="run the safe natural-language OpenAI planner"
    )
    ai_agent.add_argument("--task", default="打开WLAN设置")
    ai_agent.add_argument(
        "--model",
        default=os.environ.get("WELLPHONE_OPENAI_MODEL", "gpt-6-astra"),
    )
    langgraph_agent = subparsers.add_parser(
        "langgraph-agent", help="run the safe LangGraph agent with DeepSeek"
    )
    langgraph_agent.add_argument("--task", default="打开WLAN设置")
    langgraph_agent.add_argument(
        "--model",
        default=os.environ.get("WELLPHONE_DEEPSEEK_MODEL", "deepseek-flash"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            return command_doctor()
        if args.command == "display-test":
            return command_display_test(args.duration, args.app)
        if args.command == "demo":
            return command_demo(args.url, args.hold, args.app)
        if args.command == "observe":
            return command_observe(args.url, args.app)
        if args.command == "control-test":
            return command_control_test(args.app)
        if args.command == "agent-test":
            return command_agent_test(args.target)
        if args.command == "ai-agent":
            return command_ai_agent(args.task, args.model)
        if args.command == "langgraph-agent":
            return command_langgraph_agent(args.task, args.model)
    except (
        AdbError,
        AgentError,
        ToolNotFoundError,
        VirtualDisplayError,
        UnsafeActionError,
        FrameCaptureError,
        ScrcpyControlError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2
