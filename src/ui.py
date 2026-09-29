"""
Shared on-screen UI primitives for the OpenCV preview windows.

Every script that opens a webcam window (collect_data.py,
collect_dynamic_data.py, infer_realtime.py) draws through these same few
primitives -- a translucent panel, consistent typography, a status dot, a
minimal progress bar -- instead of each script inventing its own ad hoc
cv2.putText calls. That's what keeps the app looking like one coherent
piece of software instead of three different debug overlays.

Design language: dark translucent panels, a single teal accent color for
anything "selected" or "active", muted gray for secondary text, and
anti-aliased drawing (cv2.LINE_AA) everywhere -- the single biggest lever
for making an OpenCV overlay look deliberate rather than like a debug
print statement rendered onto a frame.
"""

from __future__ import annotations

import time
from collections import deque

import cv2
import numpy as np

# --------------------------------------------------------------------------
# Palette (BGR - OpenCV's channel order, not RGB)
# --------------------------------------------------------------------------
BG_PANEL = (26, 26, 26)
BORDER = (66, 66, 66)

TEXT_PRIMARY = (240, 240, 240)
TEXT_MUTED = (150, 150, 150)

ACCENT = (210, 175, 40)    # teal-blue - selection / headings / in-progress
SUCCESS = (110, 190, 90)   # muted green - "target met" / "detected"
WARNING = (70, 165, 230)   # amber - "below target"
DANGER = (80, 80, 225)     # soft red - "recording" / "not detected"
IDLE_DOT = (95, 95, 95)

SKELETON_LINE = (200, 165, 80)
SKELETON_JOINT = (225, 225, 225)
SKELETON_TIP = (210, 175, 40)

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_HEADING = cv2.FONT_HERSHEY_DUPLEX

PAD = 12
LINE_H = 20


def panel(frame: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.6) -> None:
    """Translucent dark backdrop with a thin border, so text stays legible
    over any background without fully blocking the camera feed."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), BG_PANEL, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BORDER, 1, cv2.LINE_AA)


def text(
    frame: np.ndarray,
    s: str,
    pos: tuple[int, int],
    scale: float = 0.5,
    color=TEXT_PRIMARY,
    heading: bool = False,
    thickness: int = 1,
) -> None:
    cv2.putText(
        frame, s, pos, FONT_HEADING if heading else FONT, scale, color, thickness, cv2.LINE_AA
    )


def status_dot(frame: np.ndarray, center: tuple[int, int], ok: bool, radius: int = 5) -> None:
    cv2.circle(frame, center, radius, SUCCESS if ok else IDLE_DOT, -1, cv2.LINE_AA)


def progress_bar(
    frame: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    fraction: float,
    color=ACCENT,
) -> None:
    fraction = max(0.0, min(1.0, fraction))
    cv2.rectangle(frame, (x, y), (x + w, y + h), (48, 48, 48), -1, cv2.LINE_AA)
    filled = int(w * fraction)
    if filled > 0:
        cv2.rectangle(frame, (x, y), (x + filled, y + h), color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x, y), (x + w, y + h), BORDER, 1, cv2.LINE_AA)


class FPSCounter:
    """Achieved processing rate (camera read + detection + drawing) over a
    short rolling window, so it reflects current conditions rather than a
    long-run average.

    Low fps and a laggy-looking skeleton are the same symptom. It also
    matters for recognition: an 18-frame swipe window takes over 2 seconds
    to fill at 8fps, far longer than a swipe lasts.
    """

    def __init__(self, window: int = 30) -> None:
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Call once per loop iteration. Returns the current fps estimate
        (0.0 until enough samples have accumulated)."""
        now = time.time()
        self._times.append(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = self._times[-1] - self._times[0]
        if elapsed <= 1e-6:
            return 0.0
        return (len(self._times) - 1) / elapsed


def fps_color(fps: float):
    """Green when comfortably fast enough for an 18-frame swipe window,
    amber when marginal, red when too slow."""
    if fps >= 15:
        return SUCCESS
    if fps >= 8:
        return WARNING
    return DANGER


def short_description(description: str) -> str:
    """First sentence only -- panels show the full multi-sentence
    descriptions (e.g. NEUTRAL's) as a single clipped line."""
    return description.split(". ")[0].split(".")[0]
