from __future__ import annotations

import io
import unittest

from wellphone_agent.confirmation import request_terminal_confirmation


class TerminalConfirmationTests(unittest.TestCase):
    @staticmethod
    def reader(*keys: str):
        values = iter(keys)
        return lambda: next(values)

    def test_single_y_approves_without_enter(self) -> None:
        output = io.StringIO()
        approved = request_terminal_confirmation(
            "确认",
            key_reader=self.reader("y"),
            output_stream=output,
        )
        self.assertTrue(approved)
        self.assertEqual(output.getvalue(), "确认 [Y/N] Y\n")

    def test_dialog_result_is_used_without_reading_terminal(self) -> None:
        calls: list[tuple[str, str]] = []

        def dialog(prompt: str, title: str) -> bool:
            calls.append((prompt, title))
            return True

        self.assertTrue(
            request_terminal_confirmation(
                "允许这个动作？",
                title="确认测试",
                dialog=dialog,
                key_reader=lambda: (_ for _ in ()).throw(
                    AssertionError("terminal must not be read")
                ),
            )
        )
        self.assertEqual(calls, [("允许这个动作？", "确认测试")])

    def test_enter_is_ignored_until_explicit_y(self) -> None:
        self.assertTrue(
            request_terminal_confirmation(
                "确认",
                key_reader=self.reader("\r", "\n", "Y"),
                output_stream=io.StringIO(),
            )
        )

    def test_n_and_escape_reject(self) -> None:
        for key in ("n", "N", "\x1b"):
            with self.subTest(key=repr(key)):
                self.assertFalse(
                    request_terminal_confirmation(
                        "确认",
                        key_reader=self.reader(key),
                        output_stream=io.StringIO(),
                    )
                )

    def test_ctrl_c_interrupts(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            request_terminal_confirmation(
                "确认",
                key_reader=self.reader("\x03"),
                output_stream=io.StringIO(),
            )

    def test_redirected_input_uses_line_mode(self) -> None:
        self.assertTrue(
            request_terminal_confirmation(
                "确认",
                input_stream=io.StringIO("yes\n"),
                output_stream=io.StringIO(),
            )
        )

    def test_redirected_eof_rejects(self) -> None:
        self.assertFalse(
            request_terminal_confirmation(
                "确认",
                input_stream=io.StringIO(""),
                output_stream=io.StringIO(),
            )
        )


if __name__ == "__main__":
    unittest.main()
