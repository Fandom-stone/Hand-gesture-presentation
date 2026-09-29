"""
Shared hand-landmark detection and feature-extraction module.

Wraps MediaPipe's *current* Tasks API (HandLandmarker) -- not the legacy
``mp.solutions.hands`` interface used in many older tutorials, which is
deprecated (see Section 8.2 of the project report). This module is imported
by the data collection, training-time sanity checks, and real-time inference
scripts so all three phases build feature vectors the exact same way.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

if TYPE_CHECKING:
    # For type checkers only -- never imported at runtime, so this cannot
    # affect the program whatever MediaPipe's internal layout happens to be.
    # Imported from the module that DEFINES the class: vision/__init__
    # re-exports it by assignment, which a type checker sees as a variable
    # rather than a class, and variables aren't valid in type annotations.
    from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker

from . import config


@dataclass
class HandDetection:
    """One detected hand: raw landmarks plus the derived feature vector."""

    landmarks: Sequence  # list of NormalizedLandmark (21 entries), image space
    handedness: str
    feature_vector: np.ndarray  # normalised, fixed-length (Section 7)


def create_hand_landmarker(
    model_path=config.HAND_LANDMARKER_MODEL_PATH,
    num_hands: int = config.MAX_NUM_HANDS,
) -> HandLandmarker:
    """Create a HandLandmarker in synchronous VIDEO mode: one call per frame,
    one result back, no callbacks needed."""
    model_path = str(model_path)
    try:
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
    except Exception as exc:  # noqa: BLE001
        raise FileNotFoundError(
            f"Could not load HandLandmarker model at '{model_path}'. "
            "Run `python scripts/download_model.py` first (Section 8.2)."
        ) from exc

    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=config.MIN_HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=config.MIN_HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


class MonotonicTimestamp:
    """VIDEO-mode HandLandmarker requires strictly increasing millisecond
    timestamps. A tiny helper avoids duplicate/out-of-order timestamps when
    frame capture is faster than the system clock's resolution."""

    def __init__(self) -> None:
        self._last_ms = -1
        self._start = time.time()

    def next(self) -> int:
        ms = int((time.time() - self._start) * 1000)
        if ms <= self._last_ms:
            ms = self._last_ms + 1
        self._last_ms = ms
        return ms


def detect_hands(
    landmarker: HandLandmarker,
    frame_bgr: np.ndarray,
    timestamp_ms: int,
) -> list[HandDetection]:
    """Run detection on one BGR frame (as read by cv2.VideoCapture).

    Returns a list of HandDetection (usually 0 or 1 entries, since
    num_hands=1 by default per config.MAX_NUM_HANDS).
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    detections: list[HandDetection] = []
    if not result.hand_landmarks:
        return detections

    for hand_landmarks, handedness in zip(
        result.hand_landmarks, result.handedness
    ):
        feature_vector = landmarks_to_feature_vector(hand_landmarks)
        label = handedness[0].category_name if handedness else "Unknown"
        if config.SWAP_HANDEDNESS_LABEL:
            label = {"Left": "Right", "Right": "Left"}.get(label, label)
        detections.append(
            HandDetection(
                landmarks=hand_landmarks,
                handedness=label,
                feature_vector=feature_vector,
            )
        )
    return detections


def landmarks_to_feature_vector(
    hand_landmarks: Sequence,
    use_z: bool = config.USE_Z_COORDINATE,
) -> np.ndarray:
    """Flatten and normalise 21 landmarks into a fixed-length feature vector.

    Translates so the wrist is the origin, then scales so the furthest
    landmark sits at distance 1.0. This makes the features invariant to hand
    position and distance from the camera, so the classifier learns the
    gesture rather than where the hand happened to be.

    Rotation is deliberately NOT normalised away: some gestures differ only
    by orientation (thumbs up vs pointing).
    """
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    if not use_z:
        pts = pts[:, :2]

    origin = pts[0].copy()
    pts = pts - origin

    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 1e-6:
        pts = pts / scale

    return pts.flatten()


_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]
_FINGERTIPS = {4, 8, 12, 16, 20}


def draw_landmarks(
    frame_bgr: np.ndarray, detections: Sequence[HandDetection]
) -> np.ndarray:
    """Draw a minimal hand skeleton on a copy of the frame. Drawn manually
    because the Tasks API does not ship the old drawing_utils helpers."""
    from . import ui  # local import: keeps hand_tracker usable without ui.py present

    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        pts_px = [(int(lm.x * w), int(lm.y * h)) for lm in det.landmarks]

        for a, b in _HAND_CONNECTIONS:
            cv2.line(annotated, pts_px[a], pts_px[b], ui.SKELETON_LINE, 1, cv2.LINE_AA)

        for idx, (x, y) in enumerate(pts_px):
            if idx in _FINGERTIPS:
                cv2.circle(annotated, (x, y), 4, ui.SKELETON_TIP, -1, cv2.LINE_AA)
            else:
                cv2.circle(annotated, (x, y), 2, ui.SKELETON_JOINT, -1, cv2.LINE_AA)

        if pts_px:
            label_pos = (pts_px[0][0] - 18, pts_px[0][1] + 34)
            # dark outline keeps the label legible on any background
            cv2.putText(annotated, det.handedness, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(annotated, det.handedness, label_pos, cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, ui.TEXT_MUTED, 1, cv2.LINE_AA)
    return annotated


class _BufferlessVideoCapture:
    """cv2.VideoCapture with a background thread that keeps only the newest
    frame, discarding any backlog.

    CAP_PROP_BUFFERSIZE is only a hint and several Windows webcam backends
    ignore it, so the driver keeps buffering and cap.read() returns stale
    frames -- visible as the preview lagging behind the hand even while
    processing keeps up. Draining the camera continuously and always serving
    the latest frame avoids that regardless of the backend.
    """

    # Camera init can take up to a couple of seconds, so the first read()
    # waits rather than reporting failure the caller would retry against.
    _STARTUP_TIMEOUT_SECONDS = 8.0
    # A single failed grab is tolerated by continuing to serve the last good
    # frame; only a sustained gap counts as a real failure.
    _STALE_FRAME_TIMEOUT_SECONDS = 2.0

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self._cap = cap
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._last_good_time: Optional[float] = None
        self._stopped = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        while not self._stopped:
            ok, frame = self._cap.read()
            if ok:
                with self._lock:
                    self._frame = frame
                    self._last_good_time = time.time()
            else:
                time.sleep(0.01)  # avoid a hot spin if the camera drops out

    def read(self):
        """Returns (ok, frame) for the newest frame, matching
        cv2.VideoCapture.read(). Blocks only while waiting for the very
        first frame."""
        deadline = time.time() + self._STARTUP_TIMEOUT_SECONDS
        while self._last_good_time is None and not self._stopped and time.time() < deadline:
            time.sleep(0.01)

        with self._lock:
            frame = self._frame
            last_good_time = self._last_good_time

        if frame is None:
            time.sleep(0.01)  # throttle, so a caller cannot busy-spin
            return False, None

        if time.time() - last_good_time > self._STALE_FRAME_TIMEOUT_SECONDS:
            time.sleep(0.01)
            return False, None

        return True, frame.copy()

    def release(self) -> None:
        self._stopped = True
        self._thread.join(timeout=1.0)
        self._cap.release()


def _open_capture(camera_index: int) -> cv2.VideoCapture:
    """Open the camera, naming a capture backend when config asks for one.

    Falls back to letting OpenCV choose whenever the named backend doesn't
    exist on this platform or fails to open, so a wrong config.CAMERA_BACKEND
    can only cost startup time -- never stop the program from capturing.
    """
    backend_name = getattr(config, "CAMERA_BACKEND", None)
    if backend_name:
        flag = getattr(cv2, backend_name, None)  # absent off Windows
        if flag is not None:
            cap = cv2.VideoCapture(camera_index, flag)
            if cap.isOpened():
                return cap
            cap.release()
    return cv2.VideoCapture(camera_index)


def open_webcam(camera_index: int = 0) -> _BufferlessVideoCapture:
    cap = _open_capture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {camera_index}. "
            "Check that it is connected and not in use by another app."
        )
    # Best-effort hints; _BufferlessVideoCapture is what actually keeps the
    # feed fresh on backends that ignore CAP_PROP_BUFFERSIZE.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, config.CAMERA_BUFFER_SIZE)
    return _BufferlessVideoCapture(cap)
