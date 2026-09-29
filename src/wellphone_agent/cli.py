from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from .actions import UnsafeActionError
from .adb import AdbClient, AdbError, ensure_connected
from .agent.core import AgentError
from .agent import DeepSeekShoppingIntentParser, SAFE_SETTINGS_GOALS
from .agent_demo import run_agent_demo, run_ai_agent_demo, run_langgraph_agent_demo
from .control_demo import format_control_result, run_control_demo
from .demo import run_browser_demo
from .display import VirtualDisplayError, VirtualDisplaySession
from .observe_demo import format_observation_result, run_observation_demo
from .meituan_demo import run_meituan_agent
from .perception.frame_capture import FrameCaptureError
from .perception.ocr import OCRError
from .scrcpy_control import ScrcpyControlError
from .tools import ToolNotFoundError, find_adb, find_scrcpy
from .understanding_demo import format_understanding_result, run_understanding_demo


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


def command_understand(app: str) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    understanding, log_path = run_understanding_demo(
        project_root=project_root,
        preferred_serial=preferred,
        package=app,
    )
    print(format_understanding_result(understanding, log_path))
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


def command_meituan_agent(query: str, model: str, concurrent: bool) -> int:
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    result, log_path, report_path = run_meituan_agent(
        project_root=project_root,
        preferred_serial=preferred,
        query=query,
        model=model,
        monitor_main_display=concurrent,
    )
    print("美团外卖任务已到达购物车并安全停止。")
    print(f"搜索词: {query}")
    print(f"步骤数: {result.steps}")
    print(f"最终页面: {result.final_understanding.screen.kind}")
    print(f"截图: {result.final_understanding.screenshot}")
    print(f"日志: {log_path}")
    if report_path is not None:
        print(f"并发验收报告: {report_path}")
    return 0


def command_shopping_agent(
    instruction: str,
    model: str,
    concurrent: bool,
    dry_run: bool,
) -> int:
    intent = DeepSeekShoppingIntentParser(model=model).parse(instruction)
    print("已将自然语言任务解析为受控购物目标：")
    print(f"搜索词: {intent.query}")
    print(f"选择策略: {intent.selection_strategy}")
    print(f"数量: {intent.quantity}")
    print(f"规格策略: {intent.specification_policy}")
    print("停止位置: 购物车（不结算、不支付）")
    if dry_run:
        print("预览完成：未连接手机，未执行任何动作。")
        return 0
    preferred = os.environ.get("WELLPHONE_SERIAL") or None
    project_root = Path(__file__).resolve().parents[2]
    result, log_path, report_path = run_meituan_agent(
        project_root=project_root,
        preferred_serial=preferred,
        query=intent.query,
        model=model,
        monitor_main_display=concurrent,
        selection_strategy=intent.selection_strategy,
        quantity=intent.quantity,
        specification_policy=intent.specification_policy,
    )
    print("通用购物任务已在购物车边界安全停止。")
    print(f"步骤数: {result.steps}")
    print(f"最终页面: {result.final_understanding.screen.kind}")
    print(f"截图: {result.final_understanding.screenshot}")
    print(f"日志: {log_path}")
    if report_path is not None:
        print(f"并发验收报告: {report_path}")
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
    understand = subparsers.add_parser(
        "understand",
        help="locally understand one virtual-screen frame without taking action",
    )
    understand.add_argument("--app", default="com.android.settings")
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
    meituan_agent = subparsers.add_parser(
        "meituan-agent",
        help="search a product in Meituan and stop after reaching the cart",
    )
    meituan_agent.add_argument("--query", required=True)
    meituan_agent.add_argument(
        "--model",
        default=os.environ.get("WELLPHONE_DEEPSEEK_MODEL", "deepseek-flash"),
    )
    meituan_agent.add_argument(
        "--concurrent",
        action="store_true",
        help="monitor display 0 and write a concurrency acceptance report",
    )
    shopping_agent = subparsers.add_parser(
        "shopping-agent",
        help="understand a natural-language Meituan shopping task",
    )
    shopping_agent.add_argument("--task", required=True)
    shopping_agent.add_argument(
        "--model",
        default=os.environ.get("WELLPHONE_DEEPSEEK_MODEL", "deepseek-flash"),
    )
    shopping_agent.add_argument(
        "--concurrent",
        action="store_true",
        help="monitor display 0 and write a concurrency acceptance report",
    )
    shopping_agent.add_argument(
        "--dry-run",
        action="store_true",
        help="parse and display the shopping intent without connecting a phone",
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
        if args.command == "understand":
            return command_understand(args.app)
        if args.command == "agent-test":
            return command_agent_test(args.target)
        if args.command == "ai-agent":
            return command_ai_agent(args.task, args.model)
        if args.command == "langgraph-agent":
            return command_langgraph_agent(args.task, args.model)
        if args.command == "meituan-agent":
            return command_meituan_agent(args.query, args.model, args.concurrent)
        if args.command == "shopping-agent":
            return command_shopping_agent(
                args.task,
                args.model,
                args.concurrent,
                args.dry_run,
            )
    except (
        AdbError,
        AgentError,
        ToolNotFoundError,
        VirtualDisplayError,
        UnsafeActionError,
        FrameCaptureError,
        OCRError,
        ScrcpyControlError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 2
