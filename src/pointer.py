"""
Fingertip cursor control.

Moves the real mouse pointer to follow the index fingertip while the POINTING
pose is held, turning the hand into a laser pointer that works in any
application -- including PowerPoint for the web, where the built-in laser
shortcut (Ctrl+L) is unusable because the browser steals it for the address
bar.

No classifier of its own. The static pose model already recognises POINTING;
this module only decides where on the screen the fingertip is aiming, which is
geometry. That makes it a different kind of control from the rest of the
system: swipes and poses are DISCRETE classification, this is CONTINUOUS
positioning derived straight from the landmarks.

Three problems have to be solved to make it usable, and each maps to one class
or function below:

  1. Raw landmarks jitter by a few pixels even with a still hand, and a cursor
     that visibly shakes is unusable       -> AdaptiveSmoother
  2. A hand cannot comfortably reach the edges of the camera frame, so mapping
     the whole frame to the whole screen makes the corners unreachable
                                            -> map_to_screen's active region
  3. A single misclassified frame must not yank the cursor away or strand it
                                            -> FingertipPointer's hysteresis
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Sequence

from . import config


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def map_to_screen(
    nx: float,
    ny: float,
    screen_width: int,
    screen_height: int,
    region: Sequence[float] = config.POINTER_ACTIVE_REGION,
) -> tuple[int, int]:
    """Map a normalised landmark position to absolute screen pixels.

    `region` is the (x0, y0, x1, y1) sub-rectangle of the camera frame that
    gets stretched across the whole screen. Using a sub-rectangle rather than
    the full frame matters: reaching the true edge of frame means fully
    extending your arm, so without it the screen corners are effectively
    unreachable and the usable range is cramped.

    Positions outside the region clamp to the screen edge, so the cursor stops
    at the border instead of wrapping or disappearing.

    No mirroring is applied. The frame is already flipped horizontally before
    detection, so moving your hand right increases nx, which is what the screen
    expects.
    """
    x0, y0, x1, y1 = region
    span_x = x1 - x0
    span_y = y1 - y0
    fx = _clamp01((nx - x0) / span_x) if span_x > 1e-9 else 0.5
    fy = _clamp01((ny - y0) / span_y) if span_y > 1e-9 else 0.5
    return (
        int(round(fx * (screen_width - 1))),
        int(round(fy * (screen_height - 1))),
    )


class AdaptiveSmoother:
    """Exponential moving average whose responsiveness rises with speed.

    A fixed smoothing factor forces a bad choice: smooth enough to kill the
    jitter of a resting hand, and the cursor lags noticeably behind a fast
    one; responsive enough to keep up, and it shakes when you hold still.

    Blending the factor with the measured speed avoids the choice. Slow
    movement gets heavy smoothing, where jitter is what you notice and lag is
    not; fast movement gets light smoothing, where the reverse is true. This
    is a simplified One Euro filter -- the same idea, with the cutoff
    frequency replaced by a straight linear blend so the behaviour stays easy
    to reason about.
    """

    def __init__(
        self,
        min_alpha: float = config.POINTER_SMOOTHING_MIN_ALPHA,
        max_alpha: float = config.POINTER_SMOOTHING_MAX_ALPHA,
        speed_scale: float = config.POINTER_SPEED_SCALE_PX,
    ) -> None:
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.speed_scale = speed_scale
        self._previous: Optional[tuple[float, float]] = None

    def update(self, x: float, y: float) -> tuple[float, float]:
        """Feed one raw position, get the smoothed one back. The first call
        after a reset passes straight through, so the cursor starts where the
        finger actually is rather than easing in from a stale position."""
        if self._previous is None:
            self._previous = (float(x), float(y))
            return self._previous

        prev_x, prev_y = self._previous
        speed = math.hypot(x - prev_x, y - prev_y)
        blend = min(1.0, speed / self.speed_scale) if self.speed_scale > 0 else 1.0
        alpha = self.min_alpha + (self.max_alpha - self.min_alpha) * blend

        self._previous = (
            alpha * x + (1.0 - alpha) * prev_x,
            alpha * y + (1.0 - alpha) * prev_y,
        )
        return self._previous

    def reset(self) -> None:
        """Forget the previous position, so the next update starts fresh."""
        self._previous = None


def _default_mover() -> Callable[[int, int], None]:
    """pyautogui's moveTo, configured for once-per-frame use.

    Two of its defaults are actively harmful here:

      PAUSE = 0.1   sleeps a tenth of a second after every call, which would
                    cap the whole capture loop at 10fps.
      FAILSAFE      raises if the cursor reaches a screen corner. That is a
                    sensible panic-stop for a script driving the mouse blind,
                    but this cursor is under continuous human control and
                    reaching a corner is ordinary use -- so it would crash the
                    program during normal pointing. Releasing the pose stops
                    the cursor, which is the better escape hatch.
    """
    import pyautogui

    pyautogui.PAUSE = 0
    pyautogui.FAILSAFE = False
    return lambda x, y: pyautogui.moveTo(x, y)


def screen_size() -> tuple[int, int]:
    import pyautogui

    width, height = pyautogui.size()
    return int(width), int(height)


class FingertipPointer:
    """Drives the mouse from the index fingertip while POINTING is held.

    Call update() once per frame with the current hand landmarks (or None) and
    the pose label the static classifier produced for this frame. It returns
    the screen position it moved to, or None when the pointer is inactive.

    It deliberately uses the RAW per-frame pose label rather than the
    debounced one from StableGestureTracker. That tracker is built to fire an
    action once per hold and then lock; a cursor needs to track continuously
    for as long as the pose lasts. Flicker is handled here instead, by
    requiring several frames to activate and several consecutive misses to
    release -- so one misclassified frame neither starts nor stops the cursor.
    """

    def __init__(
        self,
        screen: Optional[tuple[int, int]] = None,
        mover: Optional[Callable[[int, int], None]] = None,
        pose: str = config.POINTER_POSE,
        landmark_index: int = config.POINTER_LANDMARK,
        activate_frames: int = config.POINTER_ACTIVATE_FRAMES,
        release_frames: int = config.POINTER_RELEASE_FRAMES,
        region: Sequence[float] = config.POINTER_ACTIVE_REGION,
        min_move_px: int = config.POINTER_MIN_MOVE_PX,
    ) -> None:
        self.screen_width, self.screen_height = screen or screen_size()
        self._move = mover if mover is not None else _default_mover()
        self.pose = pose
        self.landmark_index = landmark_index
        self.activate_frames = activate_frames
        self.release_frames = release_frames
        self.region = region
        self.min_move_px = min_move_px

        self.active = False
        self._smoother = AdaptiveSmoother()
        self._holding = 0
        self._missing = 0
        self._last_sent: Optional[tuple[int, int]] = None

    def update(self, landmarks, pose_label: str) -> Optional[tuple[int, int]]:
        pointing = landmarks is not None and pose_label == self.pose

        if pointing:
            self._holding += 1
            self._missing = 0
        else:
            self._missing += 1
            self._holding = 0

        if not self.active:
            if self._holding < self.activate_frames:
                return None
            # Starting fresh: drop any stale smoothing state so the cursor
            # jumps straight to the finger instead of sliding in from where it
            # was last time.
            self.active = True
            self._smoother.reset()
            self._last_sent = None
        elif self._missing >= self.release_frames:
            self.active = False
            self._smoother.reset()
            return None

        if not pointing:
            # Active, but this particular frame lost the hand or read another
            # pose. Hold position rather than jumping somewhere wrong.
            return None

        tip = landmarks[self.landmark_index]
        raw_x, raw_y = map_to_screen(
            tip.x, tip.y, self.screen_width, self.screen_height, self.region
        )
        smooth_x, smooth_y = self._smoother.update(raw_x, raw_y)
        target = (int(round(smooth_x)), int(round(smooth_y)))

        # Skip sub-threshold moves: they are invisible but still cost a system
        # call on every frame.
        if self._last_sent is not None:
            dx = target[0] - self._last_sent[0]
            dy = target[1] - self._last_sent[1]
            if math.hypot(dx, dy) < self.min_move_px:
                return self._last_sent

        try:
            self._move(*target)
        except Exception:  # noqa: BLE001 - a failed cursor move must never end the session
            return None
        self._last_sent = target
        return target
