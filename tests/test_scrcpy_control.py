from __future__ import annotations

import struct
import unittest
from pathlib import Path

from wellphone_agent.scrcpy_control import ScrcpyControlError, ScrcpyControlSession


class FakeAdb:
    executable = Path("adb.exe")

    def __init__(self, power: str = "mWakefulness=Awake", policy: str = "") -> None:
        self.power = power
        self.policy = policy

    def shell(self, serial: str, *args: str, check: bool = True) -> str:
        return self.power if args == ("dumpsys", "power") else self.policy


class ScrcpyControlProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = ScrcpyControlSession(
            FakeAdb(),  # type: ignore[arg-type]
            Path("scrcpy.exe"),
            "device",
            size="1080x1920",
        )

    def test_touch_message_matches_scrcpy_32_byte_protocol(self) -> None:
        message = self.session._touch_message(0, 100, 200, pressure=0xFFFF)
        self.assertEqual(len(message), 32)
        self.assertEqual(message[:2], bytes([2, 0]))
        self.assertEqual(struct.unpack(">ii", message[10:18]), (100, 200))
        self.assertEqual(struct.unpack(">HH", message[18:22]), (1080, 1920))
        self.assertEqual(message[22:24], b"\xff\xff")

    def test_touch_rejects_coordinates_outside_virtual_display(self) -> None:
        with self.assertRaises(ScrcpyControlError):
            self.session._touch_message(0, 1080, 100, pressure=0xFFFF)

    def test_server_command_uses_matching_server_and_control_socket(self) -> None:
        command = self.session._server_command("4.1", 0x1234ABCD)
        self.assertIn("4.1", command)
        self.assertIn("scid=1234abcd", command)
        self.assertIn("control=true", command)
        self.assertIn("new_display=1080x1920/420", command)
        self.assertIn("send_dummy_byte=true", command)

    def test_device_readiness_accepts_awake_unlocked_phone(self) -> None:
        self.session.require_interactive_unlocked_device()

    def test_device_readiness_rejects_dozing_phone(self) -> None:
        session = ScrcpyControlSession(
            FakeAdb(power="mWakefulness=Dozing"),  # type: ignore[arg-type]
            Path("scrcpy.exe"),
            "device",
        )
        with self.assertRaisesRegex(ScrcpyControlError, "点亮并解锁"):
            session.require_interactive_unlocked_device()

    def test_device_readiness_rejects_keyguard(self) -> None:
        session = ScrcpyControlSession(
            FakeAdb(policy="mIsShowing=true"),  # type: ignore[arg-type]
            Path("scrcpy.exe"),
            "device",
        )
        with self.assertRaisesRegex(ScrcpyControlError, "点亮并解锁"):
            session.require_interactive_unlocked_device()

    def test_miui_current_keyguard_state_overrides_stale_field(self) -> None:
        session = ScrcpyControlSession(
            FakeAdb(
                policy="showingAndNotOccluded=true\nKeyguardStateMonitor\n"
                "mIsShowing=false"
            ),  # type: ignore[arg-type]
            Path("scrcpy.exe"),
            "device",
        )
        session.require_interactive_unlocked_device()


if __name__ == "__main__":
    unittest.main()
