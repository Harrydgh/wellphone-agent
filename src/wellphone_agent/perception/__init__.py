"""Virtual-display observation and page-state extraction."""

from .frame_capture import FrameSnapshot, VirtualDisplayCapture
from .candidates import CandidateAction, CandidateGenerator
from .fusion import PerceivedElement, fuse_page_elements
from .ocr import OCRError, OCRText, RapidOCREngine
from .privacy import PrivacyRedactor
from .screen_classifier import ScreenClassification, ScreenClassifier
from .state import PageState, PageStateTracker
from .understanding import PageUnderstanding, PageUnderstandingEngine

__all__ = [
    "FrameSnapshot",
    "CandidateAction",
    "CandidateGenerator",
    "OCRError",
    "OCRText",
    "PageState",
    "PageStateTracker",
    "PageUnderstanding",
    "PageUnderstandingEngine",
    "PerceivedElement",
    "PrivacyRedactor",
    "RapidOCREngine",
    "ScreenClassification",
    "ScreenClassifier",
    "VirtualDisplayCapture",
    "fuse_page_elements",
]
