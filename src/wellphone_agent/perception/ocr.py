from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


class OCRError(RuntimeError):
    """Local OCR could not safely produce a result."""


@dataclass(frozen=True)
class OCRText:
    text: str
    bounds: tuple[int, int, int, int]
    confidence: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class OCREngine(Protocol):
    def recognize(self, image_path: Path) -> tuple[OCRText, ...]: ...


class RapidOCREngine:
    """Lazy, fully local OCR backed by RapidOCR and ONNX Runtime."""

    def __init__(
        self,
        *,
        engine: Any | None = None,
        min_confidence: float = 0.55,
        max_results: int = 200,
    ) -> None:
        if not 0 <= min_confidence <= 1:
            raise ValueError("OCR confidence must be between zero and one.")
        if max_results <= 0:
            raise ValueError("OCR result limit must be positive.")
        self._engine = engine
        self.min_confidence = min_confidence
        self.max_results = max_results

    def _load_engine(self) -> Any:
        if self._engine is None:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:
                raise OCRError(
                    "RapidOCR is not installed. Run scripts/setup.ps1 first."
                ) from exc
            self._engine = RapidOCR()
        return self._engine

    @staticmethod
    def _box_bounds(box: Any) -> tuple[int, int, int, int]:
        points = box.tolist() if hasattr(box, "tolist") else box
        xs = [int(round(float(point[0]))) for point in points]
        ys = [int(round(float(point[1]))) for point in points]
        return min(xs), min(ys), max(xs), max(ys)

    def recognize(self, image_path: Path) -> tuple[OCRText, ...]:
        if not image_path.is_file():
            raise OCRError(f"OCR image does not exist: {image_path}")
        try:
            output = self._load_engine()(image_path)
            boxes = output.boxes if output.boxes is not None else ()
            texts = output.txts if output.txts is not None else ()
            scores = output.scores if output.scores is not None else ()
            results: list[OCRText] = []
            for box, text, score in zip(boxes, texts, scores):
                normalized = str(text).strip()
                confidence = float(score)
                if not normalized or confidence < self.min_confidence:
                    continue
                bounds = self._box_bounds(box)
                if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
                    continue
                results.append(OCRText(normalized, bounds, confidence))
                if len(results) >= self.max_results:
                    break
            return tuple(results)
        except OCRError:
            raise
        except Exception as exc:
            raise OCRError("Local OCR failed to analyze the screenshot.") from exc
