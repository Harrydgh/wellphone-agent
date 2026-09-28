from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .candidates import CandidateAction, CandidateGenerator
from .fusion import PerceivedElement, fuse_page_elements
from .ocr import OCREngine, RapidOCREngine
from .privacy import PrivacyRedactor
from .screen_classifier import ScreenClassification, ScreenClassifier
from .state import PageState


@dataclass(frozen=True)
class PageUnderstanding:
    current_app: str | None
    current_activity: str | None
    screenshot: str
    text_source: str
    ocr_count: int
    elements: tuple[PerceivedElement, ...]
    screen: ScreenClassification
    candidates: tuple[CandidateAction, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "current_app": self.current_app,
            "current_activity": self.current_activity,
            "screenshot": self.screenshot,
            "text_source": self.text_source,
            "ocr_count": self.ocr_count,
            "elements": [item.to_dict() for item in self.elements],
            "screen": self.screen.to_dict(),
            "candidates": [item.to_dict() for item in self.candidates],
        }

    def to_model_payload(self, *, max_elements: int = 80) -> dict[str, object]:
        """Return only redacted, task-relevant state; never include the screenshot."""
        return {
            "current_app": self.current_app,
            "current_activity": self.current_activity,
            "screen": self.screen.to_dict(),
            "elements": [
                asdict(item) for item in self.elements[:max_elements]
            ],
            "candidates": [item.to_dict() for item in self.candidates],
        }


class PageUnderstandingEngine:
    def __init__(
        self,
        ocr_engine: OCREngine | None = None,
        *,
        run_ocr: bool = True,
        redactor: PrivacyRedactor | None = None,
        classifier: ScreenClassifier | None = None,
        candidate_generator: CandidateGenerator | None = None,
    ) -> None:
        self.ocr_engine = ocr_engine or RapidOCREngine()
        self.run_ocr = run_ocr
        self.redactor = redactor or PrivacyRedactor()
        self.classifier = classifier or ScreenClassifier()
        self.candidate_generator = candidate_generator or CandidateGenerator()

    def analyze(self, page: PageState) -> PageUnderstanding:
        ocr_items = (
            self.ocr_engine.recognize(Path(page.screenshot)) if self.run_ocr else ()
        )
        fused = fuse_page_elements(page, ocr_items)
        redacted = self.redactor.redact_elements(fused)
        screen = self.classifier.classify(redacted, page.current_activity)
        candidates = self.candidate_generator.generate(redacted)
        sources = {item.source for item in redacted}
        if sources == {"uiautomator", "ocr"}:
            source = "uiautomator+ocr"
        elif sources == {"ocr"}:
            source = "ocr"
        else:
            source = page.text_source
        return PageUnderstanding(
            current_app=page.current_app,
            current_activity=page.current_activity,
            screenshot=page.screenshot,
            text_source=source,
            ocr_count=len(ocr_items),
            elements=redacted,
            screen=screen,
            candidates=candidates,
        )
