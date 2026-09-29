"""
Real-time inference and action mapping.

Loads both trained models once and feeds every webcam frame to both
recognition tracks in parallel:

  - STATIC:  single-frame landmarks -> pose label, debounced so a held pose
             fires its action exactly once per hold.
  - DYNAMIC: a rolling window of hand positions -> motion label, fired once
             per swipe and then cleared so its tail can't re-trigger.

Both dispatch through the same pyautogui action system.

Usage:
    python -m src.infer_realtime
    python -m src.infer_realtime --dry-run      # predict only, send no keys
    python -m src.infer_realtime --quiet        # no per-gesture explanations
    python -m src.infer_realtime --no-preview   # no camera window
    python -m src.infer_realtime --pointer      # fingertip moves the mouse
    python -m src.infer_realtime --target desktop
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from typing import Optional

import cv2
import joblib

from . import config, ui
from .hand_tracker import (
    MonotonicTimestamp,
    create_hand_landmarker,
    detect_hands,
    draw_landmarks,
    open_webcam,
)
from .trajectory_features import TrajectoryTracker, hand_center


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def _prepare_for_single_frame_use(model):
    """Force single-threaded prediction on a loaded model.

    Models are trained with n_jobs=-1, which suits fitting thousands of rows.
    At inference we predict one row per frame, where joblib's worker overhead
    costs far more than the work itself (measured ~3.8x slower) and emits a
    warning on every prediction.
    """
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    return model


def load_static_model():
    if not config.TRAINED_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained static model found at {config.TRAINED_MODEL_PATH}. "
            "Run `python -m src.train_model` first."
        )
    return _prepare_for_single_frame_use(joblib.load(config.TRAINED_MODEL_PATH))


def load_dynamic_model():
    if not config.DYNAMIC_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No trained dynamic (swipe) model found at {config.DYNAMIC_MODEL_PATH}. "
            "Run `python -m src.train_dynamic_model` first."
        )
    return _prepare_for_single_frame_use(joblib.load(config.DYNAMIC_MODEL_PATH))


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def predict_static_gesture(model, feature_vector) -> tuple[str, float]:
    """Returns (label, confidence). label is config.GESTURE_NONE when the
    top prediction's confidence is below config.MIN_PREDICTION_CONFIDENCE
    (a belt-and-suspenders fallback alongside the explicitly-trained
    NEUTRAL class)."""
    proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
    best_idx = proba.argmax()
    label = model.classes_[best_idx]
    confidence = float(proba[best_idx])
    if confidence < config.MIN_PREDICTION_CONFIDENCE:
        return config.GESTURE_NONE, confidence
    return label, confidence


def predict_dynamic_gesture(model, feature_vector) -> tuple[str, float]:
    """Returns (label, confidence). Falls back to config.NO_SWIPE (i.e. "do
    nothing") when confidence is below config.MIN_DYNAMIC_PREDICTION_CONFIDENCE,
    since a false-positive swipe is more disruptive than a missed one."""
    proba = model.predict_proba(feature_vector.reshape(1, -1))[0]
    best_idx = proba.argmax()
    label = model.classes_[best_idx]
    confidence = float(proba[best_idx])
    if confidence < config.MIN_DYNAMIC_PREDICTION_CONFIDENCE:
        return config.NO_SWIPE, confidence
    return label, confidence


# --------------------------------------------------------------------------
# Static pose debounce (majority-vote filter)
# --------------------------------------------------------------------------
class StableGestureTracker:
    """Fires a static-pose action at most once per hold.

    Labels in _NO_ACTION_LABELS release the lock, letting the same pose fire
    again next time it is held. OPEN_PALM is among them because it is the
    swiping hand shape: giving it an action would fire a command on every
    swipe.
    """

    _NO_ACTION_LABELS = {
        config.GESTURE_NONE,
        config.GESTURE_NEUTRAL,
        config.GESTURE_OPEN_PALM,
    }

    def __init__(self, verbose: bool = config.VERBOSE_DIAGNOSTICS) -> None:
        self.history: deque[str] = deque(maxlen=config.PREDICTION_HISTORY_LENGTH)
        self.locked_gesture: str | None = None
        self.last_action_time: float = 0.0
        self.verbose = verbose
        self._last_log_time: float = 0.0
        self.last_block_reason: str | None = None

    def _log_block(self, message: str) -> None:
        if not self.verbose:
            return
        now = time.time()
        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            print(f"  [pose not fired] {message}")

    def update(self, single_frame_label: str) -> str | None:
        self.history.append(single_frame_label)
        self.last_block_reason = None

        if len(self.history) < config.STABLE_FRAMES_REQUIRED:
            return None

        # Count only actionable poses. Idle frames left in the buffer from
        # before the pose was raised must not win the vote and hide it; a
        # pose still has to reach STABLE_FRAMES_REQUIRED on its own.
        counts = Counter(self.history)
        actionable = [
            (lbl, n) for lbl, n in counts.items() if lbl not in self._NO_ACTION_LABELS
        ]

        if not actionable:
            self.locked_gesture = None
            return None

        label, count = max(actionable, key=lambda item: item[1])

        if count < config.STABLE_FRAMES_REQUIRED:
            self.locked_gesture = None
            self.last_block_reason = "not steady"
            self._log_block(
                f"{label}: held for only {count} of the last {len(self.history)} frames, "
                f"needs {config.STABLE_FRAMES_REQUIRED} "
                f"(config.STABLE_FRAMES_REQUIRED) -- hold it a moment longer"
            )
            return None

        if label == self.locked_gesture:
            self.last_block_reason = "already fired"
            return None  # already actioned this hold

        if time.time() - self.last_action_time < config.ACTION_COOLDOWN_SECONDS:
            self.last_block_reason = "cooldown"
            self._log_block(
                f"{label}: recognised, but another action fired "
                f"{time.time() - self.last_action_time:.1f}s ago and the cooldown is "
                f"{config.ACTION_COOLDOWN_SECONDS}s (config.ACTION_COOLDOWN_SECONDS)"
            )
            return None

        self.locked_gesture = label
        self.last_action_time = time.time()
        return label


# --------------------------------------------------------------------------
# Command-pose memory
# --------------------------------------------------------------------------
class CommandPoseMemory:
    """Remembers that a command pose was deliberately held recently.

    Exists because counting poses in the frame history does not work for the
    case it most needs to: moving your hand while pointing blurs the frame,
    the static classifier correctly falls back to NEUTRAL/NONE, and the pose
    vanishes from the history exactly as the swipe window fills. Both
    frame-counting gates then see nothing and let the swipe through.

    A timestamp does not care whether the current frame is readable, so it
    keeps blocking straight through the blur. Arming requires several
    sightings in a short window, so one misread frame during a genuine swipe
    cannot lock swiping out.
    """

    def __init__(self) -> None:
        self._recent: deque[bool] = deque(
            maxlen=config.SWIPE_POSE_MEMORY_ARM_FRAMES
        )
        self._last_confirmed: float = 0.0

    def update(self, label: str) -> None:
        """Call once per frame with the raw static label."""
        self._recent.append(label in config.GESTURE_ACTION_MAP)
        if sum(self._recent) >= config.SWIPE_POSE_MEMORY_ARM_SIGHTINGS:
            self._last_confirmed = time.time()

    @property
    def seconds_since_confirmed(self) -> float:
        if self._last_confirmed == 0.0:
            return float("inf")
        return time.time() - self._last_confirmed

    @property
    def is_blocking(self) -> bool:
        return self.seconds_since_confirmed < config.SWIPE_POSE_MEMORY_SECONDS


# --------------------------------------------------------------------------
# Dynamic swipe firing (cooldown + reset-after-fire)
# --------------------------------------------------------------------------
class SwipeFirer:
    """Decides whether a recognised swipe is allowed to fire.

    Beyond the cooldown, two gates guard against over-eager classification:
    a geometric floor on how far the hand actually travelled, and a check
    that a command pose isn't simply being moved around.
    """

    def __init__(self, verbose: bool = config.VERBOSE_DIAGNOSTICS) -> None:
        self.last_action_time: float = 0.0
        self.last_block_reason: str | None = None
        self.verbose = verbose
        self._last_log_time: float = 0.0
        self.best_blocked_dx: float = 0.0

    def _log_block(self, message: str) -> None:
        """Rate-limited to once a second. Printed to the terminal because the
        camera window is hidden behind a fullscreen presentation."""
        if not self.verbose:
            return
        now = time.time()
        if now - self._last_log_time >= 1.0:
            self._last_log_time = now
            print(f"  [swipe blocked] {message}")

    def maybe_fire(
        self,
        label: str,
        travelled_dx: Optional[float] = None,
        command_pose_fraction: Optional[float] = None,
        recent_command_poses: Optional[int] = None,
        pose_memory: Optional["CommandPoseMemory"] = None,
    ) -> str | None:
        self.last_block_reason = None

        if label == config.NO_SWIPE:
            return None

        if travelled_dx is not None and abs(travelled_dx) < config.MIN_SWIPE_DX:
            self.last_block_reason = "too small"
            self.best_blocked_dx = max(self.best_blocked_dx, abs(travelled_dx))
            self._log_block(
                f"{label}: hand travelled {abs(travelled_dx):.3f} of the frame width, "
                f"needs {config.MIN_SWIPE_DX:.2f} (config.MIN_SWIPE_DX)"
            )
            return None

        if (
            config.SWIPE_BLOCKED_BY_COMMAND_POSE
            and command_pose_fraction is not None
            and command_pose_fraction >= config.SWIPE_BLOCKING_POSE_FRACTION
        ):
            self.last_block_reason = "command pose held"
            self._log_block(
                f"{label}: a command pose was held for "
                f"{command_pose_fraction:.0%} of the window -- moving a pose, not a swipe"
            )
            return None

        # The hand is in a command pose right now, so the travel that filled
        # this window was the move INTO that pose.
        if (
            config.SWIPE_BLOCKED_BY_COMMAND_POSE
            and recent_command_poses is not None
            and recent_command_poses >= config.SWIPE_BLOCKING_RECENT_POSES
        ):
            self.last_block_reason = "moving into a pose"
            self._log_block(
                f"{label}: a command pose filled {recent_command_poses} of the last "
                f"{config.SWIPE_BLOCKING_RECENT_FRAMES} frames -- that's a hand moving "
                f"into a pose, not a swipe"
            )
            return None

        # Blur-proof gate: the pose may be unreadable in THIS frame, but it was
        # confirmed moments ago, so this motion belongs to it.
        if (
            config.SWIPE_BLOCKED_BY_COMMAND_POSE
            and pose_memory is not None
            and pose_memory.is_blocking
        ):
            self.last_block_reason = "pose held moments ago"
            self._log_block(
                f"{label}: a command pose was confirmed "
                f"{pose_memory.seconds_since_confirmed:.1f}s ago -- still that gesture's "
                f"motion, not a swipe (config.SWIPE_POSE_MEMORY_SECONDS)"
            )
            return None

        if time.time() - self.last_action_time < config.SWIPE_ACTION_COOLDOWN_SECONDS:
            return None

        self.last_action_time = time.time()
        return label


# --------------------------------------------------------------------------
# Action dispatch
# --------------------------------------------------------------------------
def dispatch_action(
    gesture: str,
    action_map: dict,
    dry_run: bool,
    key_map: Optional[dict] = None,
    target: Optional[str] = None,
) -> str:
    """Turn a recognised gesture into a keypress and describe what happened.

    key_map/target are passed in rather than read from config so that the
    --target flag never has to write back into the config module: a module
    global mutated at startup is invisible at the point it's read, and makes
    the function impossible to test at two targets in one process.
    """
    key_map = config.ACTION_KEY_MAP if key_map is None else key_map
    target = config.PRESENTATION_TARGET if target is None else target

    action = action_map.get(gesture)
    if action is None:
        return f"(no action mapped for {gesture})"

    key_spec = key_map.get(action)
    if key_spec is None:
        # Unavailable on this target (e.g. laser pointer in PowerPoint web).
        return f"{gesture} -> {action} (not available on '{target}')"

    # The first element of key_spec ("press" / "hotkey") is documentation only:
    # keysend.press_keys already distinguishes a single key from a chord by the
    # type of `keys` itself.
    _, keys = key_spec
    description = f"{gesture} -> {action} -> {keys}"

    if dry_run:
        return f"[DRY RUN] {description}"

    from . import keysend

    # Keys go to whichever window has focus, so report where this one landed.
    target_window = keysend.foreground_window_title()
    keysend.press_keys(keys)
    drain_console_input()

    if target_window:
        if config.CAMERA_WINDOW_TITLE.lower() in target_window.lower():
            return (
                f"{description}  [SENT TO THE CAMERA WINDOW, NOT YOUR SLIDES! "
                f"click your presentation to focus it, or use --no-preview]"
            )
        return f"{description}  [-> {target_window[:45]}]"
    return description


def drain_console_input() -> None:
    """Throw away anything waiting in this console's keyboard buffer.

    Called right after the program sends a keystroke. pyautogui injects into
    the system input queue, so if this console happens to be the focused
    window -- which it is whenever you run from an editor's built-in terminal
    -- the program's own keypress lands in its own input buffer.
    """
    try:
        import msvcrt
    except ImportError:
        return
    while msvcrt.kbhit():
        msvcrt.getch()


def quit_key_pressed() -> bool:
    """True when 'q' has been typed in the terminal. Never blocks.

    Without a preview window there is no OpenCV window to read a key from,
    which used to leave Ctrl+C as the only way to stop -- and Ctrl+C inside a
    .bat makes cmd.exe ask "Terminate batch job (Y/N)?" on top of the
    program's own shutdown, so quitting took two confirmations.

    ONLY 'q', deliberately: Esc is a key this program SENDS (FIST ends the
    presentation with it). Run from an editor's built-in terminal and that Esc
    arrives right back in this console, which read as the user asking to quit
    -- so ending a slideshow killed the program. 'q' is never sent by any
    action, so it cannot collide. The preview window still accepts both,
    because OpenCV only sees keys delivered to the window itself.

    Windows-only (msvcrt); returns False elsewhere, where Ctrl+C is clean.
    """
    try:
        import msvcrt
    except ImportError:
        return False

    pressed = False
    while msvcrt.kbhit():          # drain, so a held key can't queue up
        if msvcrt.getch() in (b"q", b"Q"):
            pressed = True
    return pressed


def send_key_spec(key_spec, dry_run: bool) -> None:
    """Send a raw ("press"|"hotkey", keys) spec.

    Used for pointer-mode switching, which isn't a gesture action and so has
    no place in dispatch_action's gesture -> action -> key chain.
    """
    if key_spec is None or dry_run:
        return
    from . import keysend

    keysend.press_keys(key_spec[1])
    drain_console_input()


def _confidence_row(frame, label_text, value_label, confidence, active, x, y, w):
    color = ui.ACCENT if active else ui.TEXT_MUTED
    ui.text(frame, label_text, (x, y), scale=0.42, color=ui.TEXT_MUTED)
    ui.text(frame, value_label, (x + 62, y), scale=0.48, color=color, thickness=2 if active else 1)
    bar_x = x + 210
    bar_w = w - (bar_x - x) - ui.PAD
    if bar_w > 20:
        ui.progress_bar(frame, bar_x, y - 9, bar_w, 8, confidence, color=color)


def _draw_hud(frame, static_label, static_conf, dynamic_label, dynamic_conf, dynamic_ready, fired_action_text, hand_found, fps, swipe_blocked=None, pointer_state=None):
    x = ui.PAD
    w = 400

    header_h = ui.PAD + 4 * ui.LINE_H + ui.PAD
    ui.panel(frame, x, ui.PAD, w, header_h)
    ty = ui.PAD + ui.PAD
    ui.text(frame, "GESTURE CONTROL", (x + ui.PAD, ty), scale=0.42, color=ui.TEXT_MUTED)
    ui.text(frame, f"{fps:4.1f} fps", (x + w - ui.PAD - 70, ty), scale=0.42, color=ui.fps_color(fps))
    ty += ui.LINE_H
    ui.status_dot(frame, (x + ui.PAD + 4, ty - 4), hand_found)
    ui.text(frame, "hand detected" if hand_found else "no hand", (x + ui.PAD + 16, ty), scale=0.45,
            color=ui.TEXT_PRIMARY if hand_found else ui.TEXT_MUTED)
    ty += ui.LINE_H

    static_active = static_label not in (config.GESTURE_NONE, config.GESTURE_NEUTRAL)
    _confidence_row(frame, "static", f"{static_label} {static_conf:.0%}", static_conf,
                     static_active, x + ui.PAD, ty, w)
    ty += ui.LINE_H

    if dynamic_ready:
        swipe_active = dynamic_label != config.NO_SWIPE
        shown = f"{dynamic_label} {dynamic_conf:.0%}"
        if swipe_blocked:
            shown = f"{dynamic_label} ({swipe_blocked})"
        _confidence_row(frame, "swipe", shown, dynamic_conf,
                         swipe_active and not swipe_blocked, x + ui.PAD, ty, w)
    else:
        ui.text(frame, "swipe", (x + ui.PAD, ty), scale=0.42, color=ui.TEXT_MUTED)
        ui.text(frame, "warming up...", (x + ui.PAD + 62, ty), scale=0.45, color=ui.TEXT_MUTED)

    action_h = ui.PAD + ui.LINE_H + ui.PAD
    ay = ui.PAD + header_h + ui.PAD
    ui.panel(frame, x, ay, w, action_h)
    ui.text(frame, "last action", (x + ui.PAD, ay + ui.PAD + 6), scale=0.42, color=ui.TEXT_MUTED)
    ui.text(frame, fired_action_text or "-", (x + ui.PAD + 90, ay + ui.PAD + 6), scale=0.46,
            color=ui.SUCCESS if fired_action_text else ui.TEXT_MUTED)

    footer_y = ay + action_h + ui.PAD + 8
    if pointer_state is not None:
        tracking, position = pointer_state
        label = f"cursor  {position[0]},{position[1]}" if tracking and position else "cursor  idle"
        ui.text(frame, label, (x, footer_y), scale=0.44,
                color=ui.SUCCESS if tracking else ui.TEXT_MUTED)
        footer_y += ui.LINE_H

    ui.text(frame, "q/Esc quit", (x, footer_y), scale=0.42, color=ui.TEXT_MUTED)
    return frame


def run(
    camera_index: int,
    dry_run: bool,
    show_preview: bool = True,
    key_map: Optional[dict] = None,
    target: Optional[str] = None,
    verbose: bool = config.VERBOSE_DIAGNOSTICS,
    use_pointer: bool = config.POINTER_ENABLED,
) -> None:
    key_map = config.ACTION_KEY_MAP if key_map is None else key_map
    target = config.PRESENTATION_TARGET if target is None else target

    # With the fingertip cursor running, POINTING drives the mouse instead of
    # firing its laser-pointer key -- otherwise holding the pose would do both.
    static_action_map = dict(config.GESTURE_ACTION_MAP)
    fingertip_pointer = None
    pointer_mode_keys = None
    pointer_was_active = False
    if use_pointer and not dry_run:
        from .pointer import FingertipPointer

        fingertip_pointer = FingertipPointer()
        static_action_map.pop(config.POINTER_POSE, None)
        pointer_mode_keys = config.POINTER_MODE_KEYS.get(target)
        if pointer_mode_keys:
            print(
                f"Laser pointer ON: hold {config.POINTER_POSE} and the laser dot "
                f"follows your fingertip.\n"
                f"  Raising the pose switches PowerPoint to its laser "
                f"({pointer_mode_keys['enter'][1]}); lowering it restores the arrow "
                f"({pointer_mode_keys['exit'][1]})."
            )
        else:
            print(
                f"Fingertip cursor ON: hold {config.POINTER_POSE} to move the mouse "
                f"({fingertip_pointer.screen_width}x{fingertip_pointer.screen_height} screen).\n"
                f"  '{target}' has no laser mode, so this moves the ordinary cursor. "
                f"Use --target desktop for a real laser dot."
            )
    elif use_pointer and dry_run:
        print("Fingertip cursor requested but --dry-run is on, so the mouse won't be moved.")

    static_model = load_static_model()
    dynamic_model = load_dynamic_model()
    landmarker = create_hand_landmarker()
    cap = open_webcam(camera_index)
    ts = MonotonicTimestamp()

    static_tracker = StableGestureTracker(verbose=verbose)
    swipe_firer = SwipeFirer(verbose=verbose)
    pose_memory = CommandPoseMemory()
    trajectory = TrajectoryTracker()
    static_history: deque[str] = deque(maxlen=config.SWIPE_WINDOW_FRAMES)
    # Diagnostics, summarised on exit -- so a session where nothing fired
    # still tells you WHY, instead of leaving you guessing.
    stats = {"windows": 0, "max_dx": 0.0, "recognised": 0, "fired": 0, "best_conf": 0.0}

    last_fired_text = ""
    consecutive_read_failures = 0
    fps_counter = ui.FPSCounter()
    print("Real-time inference started." + (" (DRY RUN - no keypresses will be sent)" if dry_run else ""))
    print("Press 'q' to quit (Esc also works in the video window, but not here --\n"
          "  the FIST gesture sends Esc, so it can't double as the quit key).")
    print("Watch the fps reading in the top-right of the window -- if it's consistently")
    print("below ~15, the pipeline can't keep up and fast swipes will be missed/misread.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # A single failed grab is usually a transient driver hiccup,
                # not a real disconnect -- skip it and keep going instead of
                # ending the whole session.
                consecutive_read_failures += 1
                if consecutive_read_failures > config.MAX_CONSECUTIVE_FRAME_READ_FAILURES:
                    print("Warning: webcam stopped responding (too many failed frame reads).")
                    break
                continue
            consecutive_read_failures = 0
            frame = cv2.flip(frame, 1)

            detections = detect_hands(landmarker, frame, ts.next())

            if detections:
                static_label, static_conf = predict_static_gesture(
                    static_model, detections[0].feature_vector
                )
                position = hand_center(detections[0].landmarks)
            else:
                static_label, static_conf = config.GESTURE_NONE, 0.0
                position = None

            trajectory.update(position)

            # Continuous control, so it reads the raw per-frame label rather
            # than the debounced one -- see pointer.FingertipPointer.
            pointer_at = None
            if fingertip_pointer is not None:
                pointer_at = fingertip_pointer.update(
                    detections[0].landmarks if detections else None, static_label
                )
                # Switching PowerPoint's pointer mode is an edge, not a state:
                # send the key once when the pose is raised and once when it is
                # released, never on the frames in between.
                if fingertip_pointer.active != pointer_was_active:
                    pointer_was_active = fingertip_pointer.active
                    if pointer_mode_keys:
                        send_key_spec(
                            pointer_mode_keys["enter" if pointer_was_active else "exit"],
                            dry_run,
                        )
                        last_fired_text = (
                            "laser ON" if pointer_was_active else "laser off"
                        )

            fired_static = static_tracker.update(static_label)
            if fired_static is not None:
                last_fired_text = dispatch_action(
                    fired_static, static_action_map, dry_run, key_map, target
                )
                print(last_fired_text)

            # The swipe track is blind to hand shape, so track what the STATIC
            # classifier saw across the same window. We look for COMMAND poses
            # (ones mapped to an action) -- if you're holding one of those and
            # moving it around, that's not a swipe.
            #
            # Deliberately the full config map, not static_action_map: with the
            # fingertip cursor on, POINTING no longer fires a key but you still
            # sweep your hand across the frame to aim, and that must not be read
            # as a swipe.
            static_history.append(static_label)
            pose_memory.update(static_label)
            command_pose_fraction = (
                sum(1 for s in static_history if s in config.GESTURE_ACTION_MAP)
                / len(static_history)
                if static_history
                else 0.0
            )

            dynamic_label, dynamic_conf = config.NO_SWIPE, 0.0
            dynamic_ready = trajectory.is_full
            if dynamic_ready:
                dynamic_feature_vector = trajectory.feature_vector()
                travelled_dx = float(dynamic_feature_vector[0])
                stats["windows"] += 1
                stats["max_dx"] = max(stats["max_dx"], abs(travelled_dx))

                # Skip the classifier when the hand plainly hasn't gone
                # anywhere: SwipeFirer's MIN_SWIPE_DX floor would veto the
                # result whatever the model said, so the prediction can only
                # cost time. That is most frames of a normal session.
                #
                # Only while --quiet, though. Without it the blocked-swipe
                # diagnostics are the point of the run, and they need the
                # label the model would have given.
                if verbose or abs(travelled_dx) >= config.MIN_SWIPE_DX:
                    dynamic_label, dynamic_conf = predict_dynamic_gesture(
                        dynamic_model, dynamic_feature_vector
                    )
                    stats["best_conf"] = max(stats["best_conf"], dynamic_conf)
                    if dynamic_label != config.NO_SWIPE:
                        stats["recognised"] += 1

                    tail = list(static_history)[-config.SWIPE_BLOCKING_RECENT_FRAMES:]
                    fired_dynamic = swipe_firer.maybe_fire(
                        dynamic_label,
                        travelled_dx=travelled_dx,
                        command_pose_fraction=command_pose_fraction,
                        recent_command_poses=sum(
                            1 for s in tail if s in config.GESTURE_ACTION_MAP
                        ),
                        pose_memory=pose_memory,
                    )
                else:
                    fired_dynamic = None

                if fired_dynamic is not None:
                    stats["fired"] += 1
                    last_fired_text = dispatch_action(
                        fired_dynamic, config.DYNAMIC_ACTION_MAP, dry_run, key_map, target
                    )
                    print(last_fired_text)
                    # Require a fresh window of motion before the next swipe
                    # can fire, so the tail of this same swipe doesn't
                    # immediately re-trigger it as the window keeps sliding.
                    trajectory.reset()

            fps = fps_counter.tick()
            annotated = draw_landmarks(frame, detections)
            annotated = _draw_hud(
                annotated,
                static_label,
                static_conf,
                dynamic_label,
                dynamic_conf,
                dynamic_ready,
                last_fired_text,
                bool(detections),
                fps,
                swipe_firer.last_block_reason,
                None
                if fingertip_pointer is None
                else (fingertip_pointer.active, pointer_at),
            )
            if show_preview:
                cv2.imshow(config.CAMERA_WINDOW_TITLE, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                # No window exists, so there's no OpenCV event loop to pump --
                # read the console directly instead.
                if quit_key_pressed():
                    break
                time.sleep(0.001)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        # Quitting mid-point would otherwise leave PowerPoint stuck in laser
        # mode with no gesture left to turn it off.
        if pointer_was_active and pointer_mode_keys:
            send_key_spec(pointer_mode_keys["exit"], dry_run)

        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

        print("\n=== swipe diagnostics for this session ===")
        print(f"  motion windows analysed      : {stats['windows']}")
        print(f"  largest sideways travel seen : {stats['max_dx']:.3f} of frame width "
              f"(must exceed {config.MIN_SWIPE_DX:.2f} to fire)")
        print(f"  windows called a swipe       : {stats['recognised']}")
        print(f"  swipes actually fired        : {stats['fired']}")
        if stats["recognised"] == 0 and stats["windows"] > 0:
            print("  -> the model never called any motion a swipe. Either the hand wasn't")
            print("     tracked during the motion, or the swipe was too small/slow to")
            print(f"     resemble the training clips (best confidence {stats['best_conf']:.0%}).")
        elif stats["fired"] == 0 and stats["recognised"] > 0:
            print("  -> swipes WERE recognised but every one was blocked; the reasons")
            print("     were printed above as they happened.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print recognised actions instead of sending real keypresses",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help=(
            "Suppress the per-gesture 'blocked / not fired' explanations. "
            "Use when presenting for real; leave off when tuning. The "
            "end-of-session summary still prints."
        ),
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help=(
            "Don't open the camera window at all. Use this when actually "
            "presenting: the preview window can take keyboard focus, and "
            "keystrokes go to whatever window is focused -- so your slides "
            "never see them. Press q in this terminal to stop."
        ),
    )
    parser.add_argument(
        "--present",
        action="store_true",
        help=(
            "Presenting mode -- shorthand for --no-preview --quiet --pointer. "
            "The combination you actually want in front of an audience: no "
            "window to steal keyboard focus from your slides, no diagnostic "
            "chatter, and the fingertip cursor ready. Press q to stop."
        ),
    )
    parser.add_argument(
        "--pointer",
        action="store_true",
        help=(
            "Move the real mouse to follow your index fingertip while the "
            "POINTING pose is held -- a laser pointer that works in any app, "
            "including PowerPoint web where Ctrl+L is unusable. While this is "
            "on, POINTING stops sending its laser-pointer key. Release the "
            "pose to stop the cursor."
        ),
    )
    parser.add_argument(
        "--target",
        choices=sorted(config.ACTION_KEY_MAPS),
        default=config.PRESENTATION_TARGET,
        help=(
            "Which PowerPoint the keypresses are aimed at. 'web' avoids the "
            "shortcuts a browser steals (F5 reloads the page, Ctrl+L focuses "
            "the address bar). Default: %(default)s"
        ),
    )
    args = parser.parse_args()

    # --present is pure shorthand: it turns on the three flags that belong
    # together when presenting for real, and never turns any of them off.
    no_preview = args.no_preview or args.present
    quiet = args.quiet or args.present
    use_pointer = args.pointer or args.present

    print(f"Target: PowerPoint ({args.target})")
    if no_preview:
        print("No preview window (so it can't steal focus). Press q here to stop.")

    run(
        camera_index=args.camera,
        dry_run=args.dry_run,
        show_preview=not no_preview,
        key_map=config.ACTION_KEY_MAPS[args.target],
        target=args.target,
        verbose=not quiet,
        use_pointer=use_pointer,
    )


if __name__ == "__main__":
    main()
