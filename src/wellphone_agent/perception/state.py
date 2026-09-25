from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any

from ..adb import AdbClient
from .frame_capture import FrameSnapshot, VirtualDisplayCapture, perceptual_distance


@dataclass(frozen=True)
class VisibleElement:
    text: str
    content_description: str
    resource_id: str
    class_name: str
    bounds: str
    clickable: bool
    enabled: bool


@dataclass(frozen=True)
class PageState:
    captured_at: str
    serial: str
    display_id: int
    current_app: str | None
    current_activity: str | None
    screenshot: str
    width: int
    height: int
    sha256: str
    perceptual_hash: str
    changed_from_previous: bool | None
    visual_change_ratio: float | None
    consecutive_static_frames: int
    is_stale: bool
    visible_text: tuple[str, ...] = ()
    ui_elements: tuple[VisibleElement, ...] = ()
    text_source: str = "not_available"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["visible_text"] = list(self.visible_text)
        data["ui_elements"] = [asdict(element) for element in self.ui_elements]
        return data

    def clickable_center(self, label: str) -> tuple[int, int] | None:
        pattern = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
        for element in self.ui_elements:
            if not element.clickable or not element.enabled:
                continue
            if label not in {element.text, element.content_description}:
                continue
            match = pattern.fullmatch(element.bounds)
            if not match:
                continue
            left, top, right, bottom = (int(value) for value in match.groups())
            return (left + right) // 2, (top + bottom) // 2
        return None

    def resource_center(self, resource_id_fragment: str) -> tuple[int, int] | None:
        pattern = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")
        for element in self.ui_elements:
            if resource_id_fragment not in element.resource_id or not element.enabled:
                continue
            match = pattern.fullmatch(element.bounds)
            if not match:
                continue
            left, top, right, bottom = (int(value) for value in match.groups())
            return (left + right) // 2, (top + bottom) // 2
        return None


class AndroidPageInspector:
    ACTIVITY_PATTERN = re.compile(
        r"(?:topResumedActivity|ResumedActivity).*?\s"
        r"(?P<package>[A-Za-z0-9_.]+)/(?P<activity>[A-Za-z0-9_.$]+)"
    )

    def __init__(self, adb: AdbClient, serial: str) -> None:
        self.adb = adb
        self.serial = serial

    def current_app_activity(self, display_id: int) -> tuple[str | None, str | None]:
        output = self.adb.shell(self.serial, "dumpsys", "activity", "activities")
        marker = f"Display #{display_id} (activities from top to bottom):"
        start = output.find(marker)
        if start < 0:
            marker = f"Display: mDisplayId={display_id}"
            start = output.find(marker)
        if start < 0:
            return None, None
        next_display = output.find("Display #", start + len(marker))
        if next_display < 0:
            next_display = output.find("Display: mDisplayId=", start + len(marker))
        section = output[start:] if next_display < 0 else output[start:next_display]
        match = self.ACTIVITY_PATTERN.search(section)
        if not match:
            return None, None
        return match.group("package"), match.group("activity")


class UIHierarchyInspector:
    REMOTE_PATH = "/sdcard/wellphone-window.xml"

    def __init__(self, adb: AdbClient, serial: str, max_elements: int = 120) -> None:
        self.adb = adb
        self.serial = serial
        self.max_elements = max_elements

    def inspect(
        self, expected_package: str | None
    ) -> tuple[tuple[str, ...], tuple[VisibleElement, ...], str]:
        self.adb.shell(
            self.serial,
            "uiautomator",
            "dump",
            "--compressed",
            self.REMOTE_PATH,
            check=False,
        )
        xml_text = self.adb.shell(self.serial, "cat", self.REMOTE_PATH, check=False)
        self.adb.shell(self.serial, "rm", "-f", self.REMOTE_PATH, check=False)
        if not xml_text.lstrip().startswith("<?xml"):
            return (), (), "uiautomator_unavailable"

        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return (), (), "uiautomator_invalid_xml"

        packages = {
            node.attrib.get("package", "") for node in root.iter("node")
        } - {""}
        if expected_package and expected_package not in packages:
            return (), (), "uiautomator_package_mismatch"

        texts: list[str] = []
        seen_texts: set[str] = set()
        elements: list[VisibleElement] = []
        for node in root.iter("node"):
            attributes = node.attrib
            text = attributes.get("text", "").strip()
            description = attributes.get("content-desc", "").strip()
            for value in (text, description):
                if value and value not in seen_texts:
                    seen_texts.add(value)
                    texts.append(value)

            actionable = (
                attributes.get("clickable") == "true"
                or attributes.get("scrollable") == "true"
                or bool(text)
                or bool(description)
            )
            if actionable and len(elements) < self.max_elements:
                effective_text = text
                effective_description = description
                if attributes.get("clickable") == "true" and not (
                    effective_text or effective_description
                ):
                    for descendant in node.iter("node"):
                        if descendant is node:
                            continue
                        descendant_text = descendant.attrib.get("text", "").strip()
                        descendant_description = descendant.attrib.get(
                            "content-desc", ""
                        ).strip()
                        if descendant_text or descendant_description:
                            effective_text = descendant_text
                            effective_description = descendant_description
                            break
                elements.append(
                    VisibleElement(
                        text=effective_text,
                        content_description=effective_description,
                        resource_id=attributes.get("resource-id", ""),
                        class_name=attributes.get("class", ""),
                        bounds=attributes.get("bounds", ""),
                        clickable=attributes.get("clickable") == "true",
                        enabled=attributes.get("enabled") != "false",
                    )
                )
        return tuple(texts), tuple(elements), "uiautomator"


class PageStateTracker:
    def __init__(
        self,
        capture: VirtualDisplayCapture,
        inspector: AndroidPageInspector,
        ui_inspector: UIHierarchyInspector | None = None,
        *,
        change_threshold: float = 0.02,
        stale_after: int = 3,
    ) -> None:
        self.capture = capture
        self.inspector = inspector
        self.ui_inspector = ui_inspector
        self.change_threshold = change_threshold
        self.stale_after = stale_after
        self._previous: FrameSnapshot | None = None
        self._static_frames = 0

    def observe(self, label: str = "frame") -> PageState:
        snapshot = self.capture.capture(label)
        current_app, current_activity = self.inspector.current_app_activity(
            snapshot.display_id
        )
        visible_text: tuple[str, ...] = ()
        ui_elements: tuple[VisibleElement, ...] = ()
        text_source = "not_available"
        if self.ui_inspector is not None:
            visible_text, ui_elements, text_source = self.ui_inspector.inspect(
                current_app
            )

        changed: bool | None = None
        change_ratio: float | None = None
        if self._previous is not None:
            change_ratio = perceptual_distance(
                self._previous.perceptual_hash,
                snapshot.perceptual_hash,
            )
            changed = change_ratio > self.change_threshold
            self._static_frames = 0 if changed else self._static_frames + 1

        self._previous = snapshot
        return PageState(
            captured_at=snapshot.captured_at,
            serial=self.inspector.serial,
            display_id=snapshot.display_id,
            current_app=current_app,
            current_activity=current_activity,
            screenshot=str(snapshot.path),
            width=snapshot.width,
            height=snapshot.height,
            sha256=snapshot.sha256,
            perceptual_hash=snapshot.perceptual_hash,
            changed_from_previous=changed,
            visual_change_ratio=change_ratio,
            consecutive_static_frames=self._static_frames,
            is_stale=self._static_frames >= self.stale_after,
            visible_text=visible_text,
            ui_elements=ui_elements,
            text_source=text_source,
        )
