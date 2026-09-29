"""
Central configuration for the Gesture-Controlled Presentation System.

Keeping the gesture vocabulary, action mapping, feature-vector format and
file paths in one place means the data collection, training and inference
scripts all agree on the same contract.

Two independent classifiers run side by side (see infer_realtime.py):

  1. STATIC pose classifier  - one frame in, one held-pose label out.
  2. DYNAMIC swipe classifier - a rolling window of hand-position history
     in, a motion label out.

See README.md for the reasoning behind the tuned values below.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

HAND_LANDMARKER_MODEL_PATH = ASSETS_DIR / "hand_landmarker.task"

# Static pose classifier artifacts
GESTURE_DATA_CSV = DATA_DIR / "gesture_data.csv"
TRAINED_MODEL_PATH = MODELS_DIR / "model.pkl"
TRAINING_REPORT_PATH = MODELS_DIR / "training_report.txt"
CONFUSION_MATRIX_PATH = MODELS_DIR / "confusion_matrix.png"

# Dynamic swipe classifier artifacts
DYNAMIC_DATA_CSV = DATA_DIR / "dynamic_gesture_data.csv"
DYNAMIC_MODEL_PATH = MODELS_DIR / "dynamic_model.pkl"
DYNAMIC_TRAINING_REPORT_PATH = MODELS_DIR / "dynamic_training_report.txt"
DYNAMIC_CONFUSION_MATRIX_PATH = MODELS_DIR / "dynamic_confusion_matrix.png"

# --------------------------------------------------------------------------
# Hand landmark / feature vector settings
# --------------------------------------------------------------------------
NUM_LANDMARKS = 21          # MediaPipe HandLandmarker output per hand
USE_Z_COORDINATE = True     # True -> 63 features (x,y,z * 21), False -> 42
MAX_NUM_HANDS = 1           # single-hand control only

FEATURE_VECTOR_LENGTH = NUM_LANDMARKS * (3 if USE_Z_COORDINATE else 2)

# Lowered from the usual 0.5: fast hand motion blurs the frame, which lowers
# MediaPipe's confidence in an otherwise-correct detection.
MIN_HAND_DETECTION_CONFIDENCE = 0.4
MIN_HAND_PRESENCE_CONFIDENCE = 0.4
MIN_TRACKING_CONFIDENCE = 0.4

# Webcam capture (see hand_tracker.open_webcam). Higher resolution does not
# improve tracking accuracy here, it only adds latency.
CAMERA_CAPTURE_WIDTH = 640
CAMERA_CAPTURE_HEIGHT = 480
CAMERA_BUFFER_SIZE = 2

# Which OpenCV capture backend to open the webcam with, as the name of a
# cv2.CAP_* constant. Letting OpenCV choose makes it probe the available
# backends first, which costs over a second of startup on Windows; naming one
# skips the probe. Measured with scripts/profile_startup.py:
#
#   CAP_MSMF   0.88s     <- Media Foundation, the current choice
#   CAP_DSHOW  1.73s     <- DirectShow
#   default    1.99s     (probe + open)
#
# Re-run that script on a different machine before trusting these numbers --
# the fastest backend varies by webcam driver. Set to None to let OpenCV
# choose, which is the right setting off Windows. An unavailable or failing
# backend falls back to the default automatically, so a wrong value here can
# only cost speed, never break capture.
CAMERA_BACKEND = "CAP_MSMF"

# A failed cap.read() is usually a one-frame hiccup, not a disconnect. Give
# up only after this many consecutive failures.
MAX_CONSECUTIVE_FRAME_READ_FAILURES = 30

# The webcam frame is mirror-flipped for a natural preview, which swaps
# MediaPipe's handedness label. This corrects it back for display only; it
# does not affect recognition.
SWAP_HANDEDNESS_LABEL = True

# --------------------------------------------------------------------------
# Static pose vocabulary
# --------------------------------------------------------------------------
GESTURE_NEUTRAL = "NEUTRAL"
GESTURE_OPEN_PALM = "OPEN_PALM"
GESTURE_FIST = "FIST"
GESTURE_POINTING = "POINTING"
GESTURE_THUMBS_UP = "THUMBS_UP"

STATIC_GESTURES = [
    GESTURE_NEUTRAL,
    GESTURE_OPEN_PALM,
    GESTURE_FIST,
    GESTURE_POINTING,
    GESTURE_THUMBS_UP,
]

GESTURE_DESCRIPTIONS = {
    GESTURE_NEUTRAL: (
        "Relaxed/resting hand - NOT any of the other poses. Collect plenty of "
        "variety: loosely open, half-curled, resting, entering/leaving frame."
    ),
    GESTURE_OPEN_PALM: (
        "Palm facing camera, fingers extended. The swiping hand shape, so it "
        "triggers no action - trained only so it can be recognised and ignored."
    ),
    GESTURE_FIST: "All fingers curled into a closed fist",
    GESTURE_POINTING: "Only the index finger extended",
    GESTURE_THUMBS_UP: "Thumb extended upward, all other fingers curled in",
}

# Minimum recommended samples per gesture, per person. NEUTRAL is the
# "everything else" class, so it needs noticeably more.
MIN_SAMPLES_PER_GESTURE_PER_PERSON = 150   # aim for ~200
MIN_NEUTRAL_SAMPLES_PER_PERSON = 250       # aim for ~350


def min_samples_for(gesture: str) -> int:
    return (
        MIN_NEUTRAL_SAMPLES_PER_PERSON
        if gesture == GESTURE_NEUTRAL
        else MIN_SAMPLES_PER_GESTURE_PER_PERSON
    )


# --------------------------------------------------------------------------
# Dynamic swipe vocabulary
# --------------------------------------------------------------------------
SWIPE_LEFT = "SWIPE_LEFT"
SWIPE_RIGHT = "SWIPE_RIGHT"
NO_SWIPE = "NO_SWIPE"

DYNAMIC_GESTURES = [SWIPE_LEFT, SWIPE_RIGHT, NO_SWIPE]

DYNAMIC_GESTURE_DESCRIPTIONS = {
    SWIPE_LEFT: "Open hand, swipe steadily from right to left across the frame",
    SWIPE_RIGHT: "Open hand, swipe steadily from left to right across the frame",
    NO_SWIPE: (
        "Everything that ISN'T a swipe: standing still, holding a static pose, "
        "small random movements, slow drift, fidgeting. Needs the most variety."
    ),
}

# Number of hand positions that make up one swipe window.
SWIPE_WINDOW_FRAMES = 18

# How long the hand may go undetected mid-swipe before the window resets.
# Expressed in seconds, not frames, so it stays correct at any frame rate.
SWIPE_MAX_MISS_SECONDS = 0.5

# --------------------------------------------------------------------------
# Dynamic training data: cleaning + augmentation
# --------------------------------------------------------------------------
# The recorder captures a fixed-length window from the moment 'c' is pressed,
# so a hand returning to rest can land in the same window and cancel or
# reverse the swipe. Both filters below remove the resulting bad labels.
DROP_DIRECTION_CONTRADICTING_CLIPS = True
MIN_TRAINING_SWIPE_DX = 0.10

# Runtime floor: a swipe never fires unless the hand actually travelled this
# far sideways, whatever the classifier says.
MIN_SWIPE_DX = 0.05

# The swipe classifier sees only WHERE the hand is, never its shape, so
# moving a fist sideways would otherwise change slides. Block a swipe when a
# command pose is held for at least this fraction of the window.
SWIPE_BLOCKED_BY_COMMAND_POSE = True
SWIPE_BLOCKING_POSE_FRACTION = 0.5

# Second, sharper version of the same gate, aimed at pose TRANSITIONS.
#
# Raising your hand into the pointing pose sweeps it sideways, and that sweep
# is real travel -- easily past MIN_SWIPE_DX. The fraction gate above misses it
# because the pose only occupies the last few frames of the window, well under
# 50%, so the window reads as a swipe that happens to end in a pose.
#
# Looking at just the END of the window catches it: if the hand is in a command
# pose NOW, the motion that got it there was a transition, not a swipe. Checked
# over a short tail so one blurred frame mid-swipe can't trigger it.
SWIPE_BLOCKING_RECENT_FRAMES = 5
SWIPE_BLOCKING_RECENT_POSES = 3

# Third gate, and the one that actually holds. Both gates above count frames,
# and both fail in the same place: sweeping your hand BLURS it, so the static
# classifier honestly reports NEUTRAL/NONE, and the pose disappears from the
# history at exactly the moment the swipe window fills. Pointing at the screen
# and moving your arm therefore reads as a clean swipe.
#
# Remembering that a pose was deliberately held survives the blur, because
# memory doesn't depend on the current frame being readable. After a command
# pose is confirmed, swipes stay blocked for this long past its last sighting
# -- long enough to cover the sweep and the hand being lowered afterwards.
SWIPE_POSE_MEMORY_SECONDS = 1.0

# Arming needs several sightings inside a short window, not one frame, so a
# single misread during a genuine swipe can't lock swiping out for a second.
SWIPE_POSE_MEMORY_ARM_FRAMES = 8
SWIPE_POSE_MEMORY_ARM_SIGHTINGS = 3

# Recorded clips cluster around small, hesitant motions, and a Random Forest
# cannot extrapolate past its training range. Mirrored and speed-scaled
# variants extend that range. Applied to the TRAINING SPLIT ONLY, so the
# reported accuracy stays measured against real recorded clips.
AUGMENT_DYNAMIC_TRAINING_DATA = True
DYNAMIC_AUGMENT_SPEED_FACTORS = (1.5, 2.0, 2.5)

# Recommended clip counts per class, per person.
MIN_SWIPE_CLIPS_PER_CLASS_PER_PERSON = 40  # aim for ~60
MIN_NO_SWIPE_CLIPS_PER_PERSON = 60         # background class: collect a bit more


def min_clips_for(gesture: str) -> int:
    return (
        MIN_NO_SWIPE_CLIPS_PER_PERSON
        if gesture == NO_SWIPE
        else MIN_SWIPE_CLIPS_PER_CLASS_PER_PERSON
    )

# Motion features extracted from a window of (x, y) positions.
# Order must match trajectory_features.extract_features().
DYNAMIC_FEATURE_NAMES = [
    "dx",                     # net horizontal displacement (end - start)
    "dy",                     # net vertical displacement (end - start)
    "path_length",            # total distance travelled along the path
    "net_displacement",       # straight-line distance from start to end
    "straightness",           # net_displacement / path_length
    "mean_speed",             # path_length / number of steps
    "max_speed",              # largest frame-to-frame distance
    "direction_consistency",  # fraction of steps agreeing with overall dx
]
DYNAMIC_FEATURE_VECTOR_LENGTH = len(DYNAMIC_FEATURE_NAMES)

# --------------------------------------------------------------------------
# Model training
# --------------------------------------------------------------------------
# Random Forest size. Both were originally set high "to be safe", but a
# Random Forest stops improving once the trees agree, and every extra tree
# costs prediction time on every single frame. Measured on the collected
# datasets (time-based split, averaged over 3 seeds):
#
#   static   50 trees 0.909 label / 1.000 action, 3.4ms   <- accuracy is flat
#            75 trees 0.909 label / 1.000 action, 4.5ms      from 50 to 200;
#           200 trees 0.911 label / 1.000 action, 11.9ms     the spread is
#                                                            smaller than the
#                                                            seed-to-seed noise
#   dynamic  25 trees 1.000, 1.6ms   |  150 trees 1.000, 8.5ms
#
# 75/50 keeps the accuracy and roughly halves the per-frame cost, which is
# budget handed straight back to the frame rate.
STATIC_FOREST_TREES = 75
DYNAMIC_FOREST_TREES = 50

# How the train/test split is made. "temporal" splits each class by capture
# time (honest); "random" is the conventional stratified shuffle, which is
# optimistic on this data because held-key capture produces near-duplicate
# consecutive frames. See evaluation.temporal_split_indices. Both figures are
# always reported; this only chooses which model is saved for deployment.
EVALUATION_SPLIT = "temporal"

# --------------------------------------------------------------------------
# Gesture -> action mapping
# --------------------------------------------------------------------------
ACTION_NEXT_SLIDE = "next_slide"
ACTION_PREVIOUS_SLIDE = "previous_slide"
ACTION_START_PRESENTATION = "start_presentation"
ACTION_END_PRESENTATION = "end_presentation"
ACTION_LASER_POINTER = "laser_pointer"

# NEUTRAL and OPEN_PALM are intentionally absent: OPEN_PALM is the swiping
# hand shape, so giving it an action would fire a command on every swipe.
GESTURE_ACTION_MAP = {
    GESTURE_THUMBS_UP: ACTION_START_PRESENTATION,
    GESTURE_FIST: ACTION_END_PRESENTATION,
    GESTURE_POINTING: ACTION_LASER_POINTER,
}

# NO_SWIPE is intentionally absent (no action).
DYNAMIC_ACTION_MAP = {
    SWIPE_RIGHT: ACTION_NEXT_SLIDE,
    SWIPE_LEFT: ACTION_PREVIOUS_SLIDE,
}

# Which PowerPoint the keypresses are aimed at.
#
#   "desktop" -- the installed PowerPoint app. The full shortcut set works,
#                including the laser pointer. This is the default.
#   "web"     -- powerpoint.com in a browser. A browser steals two of the
#                desktop shortcuts: F5 reloads the page and Ctrl+L focuses the
#                address bar, so those are remapped or dropped.
#
# Switch per run with --target web / --target desktop (or `.\go web`).
PRESENTATION_TARGET = "desktop"

# action -> ("press", key) | ("hotkey", (key, ...)) | None when that action
# has no usable shortcut on that target.
ACTION_KEY_MAPS = {
    "desktop": {
        ACTION_NEXT_SLIDE: ("press", "right"),
        ACTION_PREVIOUS_SLIDE: ("press", "left"),
        ACTION_START_PRESENTATION: ("press", "f5"),
        ACTION_END_PRESENTATION: ("press", "esc"),
        ACTION_LASER_POINTER: ("hotkey", ("ctrl", "l")),
    },
    "web": {
        ACTION_NEXT_SLIDE: ("press", "right"),
        ACTION_PREVIOUS_SLIDE: ("press", "left"),
        ACTION_START_PRESENTATION: ("hotkey", ("ctrl", "f5")),
        ACTION_END_PRESENTATION: ("press", "esc"),
        # No laser pointer in the web player, and Ctrl+L would focus the
        # browser address bar.
        ACTION_LASER_POINTER: None,
    },
}

ACTION_KEY_MAP = ACTION_KEY_MAPS[PRESENTATION_TARGET]

# Pointer-mode keys used while the FINGERTIP CURSOR is active (--pointer).
#
# Desktop PowerPoint has a laser mode: Ctrl+L turns the mouse cursor into a
# red laser dot for the rest of the slideshow, and Ctrl+A puts the ordinary
# arrow back. Pairing that with the fingertip cursor is what actually makes a
# laser pointer: Ctrl+L on the way in, the finger moves the dot, Ctrl+A on the
# way out. Raising and lowering the pose is the whole interaction.
#
# The web player has no such mode, so there is nothing to switch -- the
# fingertip cursor still moves the ordinary mouse pointer there, which is a
# usable pointer, just not a red dot.
POINTER_MODE_KEYS = {
    "desktop": {
        "enter": ("hotkey", ("ctrl", "l")),   # laser dot
        "exit": ("hotkey", ("ctrl", "a")),    # back to the arrow
    },
    "web": None,
}

# Title of the camera preview window, used to warn when a keystroke was sent
# while the preview had focus (so the presentation never received it).
CAMERA_WINDOW_TITLE = "Gesture Presentation Control"

# Explain on the terminal why a recognised gesture was suppressed. Useful
# when tuning, noisy when presenting; turn off per run with --quiet.
VERBOSE_DIAGNOSTICS = True

# --------------------------------------------------------------------------
# Real-time inference: static pose debounce
# --------------------------------------------------------------------------
# A pose must be seen in this many of the last PREDICTION_HISTORY_LENGTH
# frames before it may fire.
STABLE_FRAMES_REQUIRED = 6
PREDICTION_HISTORY_LENGTH = 12

# After a pose fires, ignore further pose triggers for this long.
ACTION_COOLDOWN_SECONDS = 1.2

# Minimum classifier confidence (max predict_proba) to accept a prediction.
MIN_PREDICTION_CONFIDENCE = 0.70

# Internal "nothing confidently recognised" label. Distinct from
# GESTURE_NEUTRAL, which is an explicitly trained class.
GESTURE_NONE = "NONE"

# --------------------------------------------------------------------------
# Fingertip cursor (see pointer.py) -- opt in per run with --pointer
# --------------------------------------------------------------------------
# Moves the real mouse to follow the index fingertip while POINTING is held.
# Off by default: it takes over the cursor, which should never be a surprise.
POINTER_ENABLED = False

# Index fingertip in MediaPipe's 21-landmark hand model.
POINTER_LANDMARK = 8

# Which pose switches the cursor on. When the pointer is running this pose
# stops firing its usual laser-pointer key, so the two can't fight.
POINTER_POSE = GESTURE_POINTING

# (x0, y0, x1, y1) sub-rectangle of the camera frame stretched across the full
# screen. Reaching the true frame edge means fully extending your arm, so
# mapping the whole frame would leave the screen corners unreachable. Widen
# toward (0, 0, 1, 1) for finer control over a smaller screen area.
POINTER_ACTIVE_REGION = (0.15, 0.15, 0.85, 0.85)

# Frames the pose must be seen before the cursor takes over, and consecutive
# frames it must be absent before releasing. Release is the larger of the two
# so a single dropped detection mid-point doesn't strand the cursor.
POINTER_ACTIVATE_FRAMES = 3
POINTER_RELEASE_FRAMES = 5

# Speed-adaptive smoothing (see pointer.AdaptiveSmoother). Lower min_alpha =
# steadier when still; higher max_alpha = more responsive when moving fast.
# POINTER_SPEED_SCALE_PX is the per-frame movement, in pixels, at which
# smoothing reaches its most responsive setting.
POINTER_SMOOTHING_MIN_ALPHA = 0.15
POINTER_SMOOTHING_MAX_ALPHA = 0.75
POINTER_SPEED_SCALE_PX = 60.0

# Don't issue a cursor move smaller than this; invisible, but still a system
# call on every frame.
POINTER_MIN_MOVE_PX = 2

# --------------------------------------------------------------------------
# Real-time inference: dynamic swipe firing
# --------------------------------------------------------------------------
# Higher than the static threshold: a false-positive swipe skips a slide
# unexpectedly, which is more disruptive than a missed pose.
MIN_DYNAMIC_PREDICTION_CONFIDENCE = 0.80

# After a swipe fires, ignore further swipe triggers for this long.
SWIPE_ACTION_COOLDOWN_SECONDS = 0.8
