from __future__ import annotations

import os
import sys
from collections.abc import Callable
from typing import TextIO


KeyReader = Callable[[], str]
Dialog = Callable[[str, str], bool]


def _windows_confirmation_dialog(prompt: str, title: str) -> bool:
    import ctypes

    yes_no = 0x00000004
    icon_question = 0x00000020
    set_foreground = 0x00010000
    topmost = 0x00040000
    result = ctypes.windll.user32.MessageBoxW(
        None,
        prompt,
        title,
        yes_no | icon_question | set_foreground | topmost,
    )
    return result == 6  # IDYES


def request_terminal_confirmation(
    prompt: str,
    *,
    title: str = "Wellphone Agent 操作确认",
    dialog: Dialog | None = None,
    key_reader: KeyReader | None = None,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
) -> bool:
    """Request approval through a native Windows dialog or safe fallback."""

    input_stream = input_stream or sys.stdin
    output_stream = output_stream or sys.stdout
    if dialog is not None:
        return dialog(prompt, title)
    if key_reader is None and os.name == "nt" and input_stream.isatty():
        output_stream.write("已弹出 Windows 操作确认窗口，请选择“是”或“否”。\n")
        output_stream.flush()
        return _windows_confirmation_dialog(prompt, title)

    if key_reader is not None:
        output_stream.write(f"{prompt} [Y/N] ")
        output_stream.flush()
        while True:
            key = key_reader()
            if key in {"\x00", "\xe0"}:
                # Consume the scan code that follows a Windows special key.
                key_reader()
                continue
            if key == "\x03":
                raise KeyboardInterrupt
            if key.casefold() == "y":
                output_stream.write("Y\n")
                output_stream.flush()
                return True
            if key.casefold() == "n" or key == "\x1b":
                output_stream.write("N\n")
                output_stream.flush()
                return False
            # Enter and unrelated keys are deliberately ignored. Approval must
            # be an explicit Y, while rejection must be N or Escape.

    output_stream.write(f"{prompt}（输入 y 或 yes 后回车）: ")
    output_stream.flush()
    answer = input_stream.readline()
    if not answer:
        return False
    return answer.strip().casefold() in {"y", "yes"}
