"""
Motion-trajectory tracking and feature extraction for the dynamic swipe
classifier.

Where the static classifier turns ONE frame into a feature vector, this
module does the equivalent for MOTION: it keeps a rolling window of the
hand's (x, y) position and reduces that window to one fixed-length vector
describing how the hand moved -- displacement, path length, straightness,
speed. A Random Forest then classifies it (see train_dynamic_model.py).

Hand-crafted features rather than a sequence model (LSTM/GRU): it keeps the
project on one scikit-learn stack, needs far less data, and trains in a
fraction of a second.
"""

from __future__ import annotations

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from . import config


def hand_center(landmarks) -> tuple[float, float]:
    """A stable 2D reference point for the hand. Averaging the wrist with the
    middle-finger MCP resists finger-pose changes better than the wrist
    alone."""
    x = (landmarks[0].x + landmarks[9].x) / 2.0
    y = (landmarks[0].y + landmarks[9].y) / 2.0
    return (x, y)


class TrajectoryTracker:
    """Maintains a sliding window of recent hand-centre positions.

    Call update(position_or_None) once per frame, passing None when no hand
    was detected. A single missed frame does not reset the window -- brief
    dropouts are normal mid-swipe, where motion blur defeats detection. The
    window clears only once the hand has been missing for longer than
    max_miss_seconds, and update() returns True when that happens so callers
    can discard in-progress work.

    The tolerance is time-based rather than a frame count, which would only
    be correct at the frame rate it was tuned for.
    """

    def __init__(
        self,
        window_frames: int = config.SWIPE_WINDOW_FRAMES,
        max_miss_seconds: float = config.SWIPE_MAX_MISS_SECONDS,
    ) -> None:
        self.window_frames = window_frames
        self.max_miss_seconds = max_miss_seconds
        self._positions: deque[tuple[float, float]] = deque(maxlen=window_frames)
        # Starts at creation time so a hand that never appears still times out.
        self._last_valid_time: float = time.time()

    def update(self, position: Optional[tuple[float, float]]) -> bool:
        """Returns True only if this update reset the window (hand missing
        longer than max_miss_seconds)."""
        now = time.time()
        if position is None:
            if now - self._last_valid_time > self.max_miss_seconds:
                self.reset()
                return True
            return False
        self._last_valid_time = now
        self._positions.append(position)
        return False

    def reset(self) -> None:
        self._positions.clear()
        self._last_valid_time = time.time()

    @property
    def is_full(self) -> bool:
        return len(self._positions) == self.window_frames

    @property
    def frame_count(self) -> int:
        """How many positions are currently buffered (for progress UI)."""
        return len(self._positions)

    def feature_vector(self) -> Optional[np.ndarray]:
        """Returns the fixed-length motion feature vector for the current
        window, or None if the window isn't full yet."""
        if not self.is_full:
            return None
        return extract_features(list(self._positions))


def extract_features(positions: list[tuple[float, float]]) -> np.ndarray:
    """Turn time-ordered (x, y) positions into the fixed-length feature
    vector listed in config.DYNAMIC_FEATURE_NAMES."""
    pts = np.array(positions, dtype=np.float32)
    n = len(pts)
    if n < 2:
        return np.zeros(config.DYNAMIC_FEATURE_VECTOR_LENGTH, dtype=np.float32)

    deltas = np.diff(pts, axis=0)  # (n-1, 2): per-step (dx_i, dy_i)
    step_distances = np.linalg.norm(deltas, axis=1)  # (n-1,)

    dx = float(pts[-1, 0] - pts[0, 0])
    dy = float(pts[-1, 1] - pts[0, 1])
    path_length = float(step_distances.sum())
    net_displacement = float(math.hypot(dx, dy))
    straightness = net_displacement / path_length if path_length > 1e-6 else 0.0
    mean_speed = path_length / (n - 1)
    max_speed = float(step_distances.max())

    if abs(dx) > 1e-6:
        overall_sign = np.sign(dx)
        step_signs = np.sign(deltas[:, 0])
        # ignore near-zero steps (sign ~0) when scoring consistency
        moving_steps = step_signs[step_signs != 0]
        if len(moving_steps) > 0:
            direction_consistency = float(np.mean(moving_steps == overall_sign))
        else:
            direction_consistency = 0.0
    else:
        direction_consistency = 0.0

    return np.array(
        [
            dx,
            dy,
            path_length,
            net_displacement,
            straightness,
            mean_speed,
            max_speed,
            direction_consistency,
        ],
        dtype=np.float32,
    )
