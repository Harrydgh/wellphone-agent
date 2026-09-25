"""Virtual-display observation and page-state extraction."""

from .frame_capture import FrameSnapshot, VirtualDisplayCapture
from .state import PageState, PageStateTracker

__all__ = [
    "FrameSnapshot",
    "PageState",
    "PageStateTracker",
    "VirtualDisplayCapture",
]

