"""
Where does startup time actually go?

Times each stage of starting the system separately, then compares the three
webcam backends OpenCV offers on Windows. Run it when the program feels slow
to start:

    python scripts/profile_startup.py

Prints a breakdown, names the slowest stage, and says whether switching camera
backend would help. It opens and closes the camera a few times, so close any
other app using the webcam first.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

stages: list[tuple[str, float]] = []


def timed(label: str, fn):
    t0 = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        stages.append((f"{label}  (FAILED: {type(exc).__name__})", elapsed))
        return None
    elapsed = time.perf_counter() - t0
    stages.append((label, elapsed))
    return result


def bar(seconds: float, longest: float, width: int = 28) -> str:
    filled = 0 if longest <= 0 else int(round(width * seconds / longest))
    return "#" * filled + "." * (width - filled)


print("Profiling startup. This takes under a minute.\n")

# --- imports --------------------------------------------------------------
overall = time.perf_counter()


def _imp(name):
    def go():
        __import__(name)
    return go


timed("import numpy", _imp("numpy"))
timed("import cv2 (OpenCV)", _imp("cv2"))
timed("import mediapipe", _imp("mediapipe"))
timed("import joblib + sklearn", _imp("sklearn.ensemble"))
timed("import src.config", _imp("src.config"))
timed("import src.hand_tracker", _imp("src.hand_tracker"))
timed("import src.infer_realtime", _imp("src.infer_realtime"))

import_total = time.perf_counter() - overall

# --- is matplotlib being dragged in? -------------------------------------
matplotlib_loaded = "matplotlib" in sys.modules

# --- model loading --------------------------------------------------------
import joblib  # noqa: E402

from src import config  # noqa: E402

timed(
    "load model.pkl",
    lambda: joblib.load(config.TRAINED_MODEL_PATH)
    if config.TRAINED_MODEL_PATH.exists()
    else None,
)
timed(
    "load dynamic_model.pkl",
    lambda: joblib.load(config.DYNAMIC_MODEL_PATH)
    if config.DYNAMIC_MODEL_PATH.exists()
    else None,
)

from src.hand_tracker import create_hand_landmarker  # noqa: E402

landmarker = timed("create HandLandmarker", create_hand_landmarker)
if landmarker is not None:
    landmarker.close()

# --- camera backends ------------------------------------------------------
import cv2  # noqa: E402

backends = [("default (whatever OpenCV picks)", None)]
for attr in ("CAP_DSHOW", "CAP_MSMF"):
    if hasattr(cv2, attr):
        backends.append((attr, getattr(cv2, attr)))

print("Testing camera backends...\n")
camera_results: list[tuple[str, float, bool]] = []
for label, flag in backends:
    t0 = time.perf_counter()
    try:
        cap = cv2.VideoCapture(0) if flag is None else cv2.VideoCapture(0, flag)
        opened = cap.isOpened()
        ok = False
        if opened:
            ok, _ = cap.read()  # first frame is where the real cost usually is
        cap.release()
    except Exception:  # noqa: BLE001
        opened = ok = False
    camera_results.append((label, time.perf_counter() - t0, bool(ok)))
    time.sleep(0.3)  # let the driver settle between attempts

# --- report ---------------------------------------------------------------
print("=" * 62)
print("STARTUP BREAKDOWN")
print("=" * 62)
longest = max((s for _, s in stages), default=0.0)
for label, seconds in stages:
    print(f"  {label:34s} {seconds:6.2f}s  {bar(seconds, longest)}")

print(f"\n  {'imports subtotal':34s} {import_total:6.2f}s")

print("\n" + "=" * 62)
print("CAMERA BACKENDS (open + first frame)")
print("=" * 62)
cam_longest = max((s for _, s, _ in camera_results), default=0.0)
for label, seconds, ok in camera_results:
    mark = "got a frame" if ok else "NO FRAME"
    print(f"  {label:34s} {seconds:6.2f}s  {bar(seconds, cam_longest)}  {mark}")

print("\n" + "=" * 62)
print("VERDICT")
print("=" * 62)

if stages:
    worst_label, worst_seconds = max(stages, key=lambda s: s[1])
    print(f"  Slowest single stage: {worst_label.strip()} at {worst_seconds:.2f}s")

working = [(l, s) for l, s, ok in camera_results if ok]
if len(working) > 1:
    fastest_label, fastest = min(working, key=lambda c: c[1])
    default = next((s for l, s in working if l.startswith("default")), None)
    if default is not None and default - fastest > 0.5:
        print(
            f"  Camera: '{fastest_label}' opens {default - fastest:.2f}s faster than the\n"
            f"    default backend. Worth switching -- see the note below."
        )
    else:
        print("  Camera: no backend is meaningfully faster. Not the problem.")
elif working:
    print("  Camera: only one backend works on this machine, so no choice to make.")
else:
    print("  Camera: no backend returned a frame. Is another app using the webcam?")

if matplotlib_loaded:
    print(
        "  Note: matplotlib got imported even though inference never plots --\n"
        "    mediapipe pulls it in internally. Unavoidable, but it explains a\n"
        "    chunk of the import time."
    )

print(
    "\n  Run this twice. The FIRST run after any file changes is slower because\n"
    "  Python recompiles bytecode and Windows Defender rescans changed files.\n"
    "  The second run is the number that reflects normal use."
)
