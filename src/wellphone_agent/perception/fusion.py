from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Literal

from .ocr import OCRText
from .state import PageState

ElementSource = Literal["uiautomator", "ocr"]


@dataclass(frozen=True)
class PerceivedElement:
    text: str
    bounds: tuple[int, int, int, int]
    source: ElementSource
    confidence: float
    clickable: bool
    enabled: bool
    resource_id: str = ""

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_BOUNDS_PATTERN = re.compile(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]")


def _parse_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = _BOUNDS_PATTERN.fullmatch(value)
    if not match:
        return None
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", "", value).casefold()


def _intersection_ratio(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max(1, (first[2] - first[0]) * (first[3] - first[1]))
    second_area = max(1, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / min(first_area, second_area)


def fuse_page_elements(
    page: PageState, ocr_items: tuple[OCRText, ...]
) -> tuple[PerceivedElement, ...]:
    """Prefer UI Automator and add non-duplicate OCR-only text regions."""

    fused: list[PerceivedElement] = []
    for item in page.ui_elements:
        label = (item.text or item.content_description).strip()
        bounds = _parse_bounds(item.bounds)
        if not label or bounds is None:
            continue
        fused.append(
            PerceivedElement(
                text=label,
                bounds=bounds,
                source="uiautomator",
                confidence=1.0,
                clickable=item.clickable,
                enabled=item.enabled,
                resource_id=item.resource_id,
            )
        )

    for item in ocr_items:
        duplicate = any(
            _normalized_text(existing.text) == _normalized_text(item.text)
            and _intersection_ratio(existing.bounds, item.bounds) >= 0.35
            for existing in fused
        )
        if duplicate:
            continue
        fused.append(
            PerceivedElement(
                text=item.text,
                bounds=item.bounds,
                source="ocr",
                confidence=item.confidence,
                clickable=False,
                enabled=True,
            )
        )

    return tuple(sorted(fused, key=lambda item: (item.bounds[1], item.bounds[0])))
