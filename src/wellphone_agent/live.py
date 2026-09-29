from __future__ import annotations

import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable, TextIO


class LiveDisplayError(RuntimeError):
    """The read-only live display viewer could not be started."""


class LiveDisplayViewer:
    """Mirror an existing Android display in a visible, read-only scrcpy window."""

    def __init__(
        self,
        scrcpy: Path,
        serial: str,
        display_id: int,
        *,
        title: str = "Wellphone Agent · 只读虚拟屏",
    ) -> None:
        if display_id <= 0:
            raise ValueError("Live viewer requires a secondary display id.")
        self.scrcpy = scrcpy
        self.serial = serial
        self.display_id = display_id
        self.title = title
        self.process: subprocess.Popen[str] | None = None
        self._lines: deque[str] = deque(maxlen=100)
        self._reader: threading.Thread | None = None

    @property
    def command(self) -> list[str]:
        return [
            str(self.scrcpy),
            f"--serial={self.serial}",
            f"--display-id={self.display_id}",
            "--no-control",
            "--no-audio",
            "--always-on-top",
            "--window-width=432",
            "--window-height=768",
            f"--window-title={self.title}",
        ]

    def _read_output(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for raw_line in self.process.stdout:
            self._lines.append(raw_line.rstrip())

    def start(self, startup_seconds: float = 1.0) -> None:
        if self.process is not None:
            raise LiveDisplayError("实时虚拟屏窗口已经启动。")
        creation_flags = 0
        if hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=creation_flags,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()
        if startup_seconds:
            time.sleep(startup_seconds)
        if self.process.poll() is not None:
            detail = "\n".join(self._lines) or "scrcpy 未提供错误详情。"
            raise LiveDisplayError(f"无法打开实时虚拟屏窗口：{detail}")

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        process = self.process
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=timeout)


class LiveProgressReporter:
    """Render safe workflow events and periodic activity heartbeats."""

    def __init__(
        self,
        *,
        output: TextIO | None = None,
        heartbeat_seconds: float = 12.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if heartbeat_seconds <= 0:
            raise ValueError("Heartbeat interval must be positive.")
        self.output = output or sys.stdout
        self.heartbeat_seconds = heartbeat_seconds
        self.clock = clock
        self._last_output = clock()
        self._waiting_confirmation = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _emit(self, message: str) -> None:
        with self._lock:
            print(message, file=self.output, flush=True)
            self._last_output = self.clock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._emit("[实时] 已启用只读虚拟屏和终端步骤进度。")
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(min(self.heartbeat_seconds, 1.0)):
            self.heartbeat_once()

    def heartbeat_once(self) -> None:
        with self._lock:
            due = self.clock() - self._last_output >= self.heartbeat_seconds
            waiting = self._waiting_confirmation
        if due:
            self._emit(
                "[等待] 请在 Windows 确认窗口中选择“是”或“否”。"
                if waiting
                else "[处理中] 正在截图、识别页面或等待界面稳定……"
            )

    @staticmethod
    def _screen(record: dict[str, object]) -> str:
        screen = record.get("screen")
        if isinstance(screen, dict):
            return str(screen.get("kind") or "unknown")
        return str(screen or "unknown")

    def handle(self, record: dict[str, object]) -> None:
        event = record.get("event")
        if event == "display_ready":
            self._emit(f"[准备] 已创建虚拟显示 {record.get('display_id')}。")
        elif event == "live_viewer_started":
            self._emit("[画面] 只读虚拟屏窗口已打开。")
        elif event == "main_display_monitor_started":
            self._emit("[安全] 已开始监测手机主屏，防止任务 App 串屏。")
        elif event == "task_started":
            self._emit(f"[任务] {record.get('task')}")
        elif event == "task_observed":
            candidates = record.get("task_candidates")
            count = len(candidates) if isinstance(candidates, list) else 0
            self._emit(f"[观察] 当前页面：{self._screen(record)}；候选控件：{count}。")
        elif event == "task_decision":
            self._emit(
                f"[计划] 第 {record.get('step')} 步：{record.get('reason')}"
            )
        elif event == "task_approval_required":
            self._waiting_confirmation = True
            self._emit(f"[确认] 等待批准目标：{record.get('target')}。")
        elif event == "task_approval_resolved":
            self._waiting_confirmation = False
            status = "已批准" if record.get("approved") else "已拒绝"
            self._emit(f"[确认] {status}：{record.get('target')}。")
        elif event == "task_pre_action_revalidated":
            self._emit("[复核] 页面和目标仍然有效，可以安全执行。")
        elif event == "task_action_executing":
            action = str(record.get("action") or "unknown")
            target = record.get("target")
            if action == "tap_candidate":
                description = f"点击：{target}"
            else:
                description = {
                    "input_query": "输入并提交搜索词",
                    "scroll_down": "向下滚动页面",
                    "back": "返回上一页",
                    "wait": "等待页面加载",
                }.get(action, action)
            self._emit(f"[执行] {description}")
        elif event == "task_action_verified":
            changed = "页面已变化" if record.get("page_changed") else "页面未变化"
            self._emit(
                f"[验证] {record.get('before_screen')} → "
                f"{record.get('after_screen')}；{changed}。"
            )
        elif event == "task_completed":
            self._emit(f"[完成] {record.get('reason')}")
        elif event == "task_aborted":
            self._emit(f"[停止] {record.get('reason')}")
        elif event == "run_failed":
            self._emit(f"[错误] {record.get('detail')}")
        elif event == "live_viewer_stopped":
            self._emit("[画面] 只读虚拟屏窗口已关闭。")

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
