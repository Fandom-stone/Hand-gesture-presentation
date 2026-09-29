"""
Static pose data collection.

Interactive tool that opens the webcam, overlays the detected hand
skeleton, and lets anyone record labelled feature-vector samples into
data/gesture_data.csv -- one row per sample, appended so everyone who runs
this builds one shared dataset (no per-person tagging; samples aren't
attributed to whoever recorded them).

This records the STATIC (held-pose) vocabulary only: NEUTRAL, OPEN_PALM,
FIST, POINTING. Next/previous slide are handled by the separate dynamic
swipe classifier -- see collect_dynamic_data.py.

Usage:
    python -m src.collect_data
    python -m src.collect_data --camera 1

Controls (shown on-screen too):
    1-N   select the gesture to record (matches config.STATIC_GESTURES order,
          shown on-screen; currently NEUTRAL/OPEN_PALM/FIST/POINTING/THUMBS_UP)
    c     capture one sample of the currently selected gesture
          (hold the key down -- OS key-repeat lets you rack up samples
          fast; aim for the on-screen counter turning green)
    r     reset the on-screen "captured this session" counters
    q/Esc quit

Don't forget NEUTRAL: it needs MORE samples than the other poses (it's the
"everything else" class) and more variety -- relaxed hand, half-curled,
resting position, hand entering/leaving frame, etc.
"""

from __future__ import annotations

import argparse
import csv
import time

import cv2

from . import config, dataset_io, ui
from .hand_tracker import (
    MonotonicTimestamp,
    create_hand_landmarker,
    detect_hands,
    draw_landmarks,
    open_webcam,
)

FEATURE_COLUMNS = [f"f{i}" for i in range(config.FEATURE_VECTOR_LENGTH)]
CSV_HEADER = ["gesture", "timestamp"] + FEATURE_COLUMNS


def append_sample(writer, gesture: str, feature_vector) -> None:
    """Write one row through an already-open csv.writer. Holding 'c' fires
    many captures a second via key-repeat, and reopening the file each time
    is slow enough to stall the preview."""
    writer.writerow(dataset_io.timestamped_row(gesture, feature_vector))


def _draw_hud(frame, selected_gesture, counts, session_counts, hand_found, warning_text=None, fps=0.0):
    x = ui.PAD
    w = 380

    header_h = ui.PAD + 3 * ui.LINE_H + ui.PAD
    ui.panel(frame, x, ui.PAD, w, header_h)
    ty = ui.PAD + ui.PAD
    ui.text(frame, "STATIC POSE COLLECTION", (x + ui.PAD, ty), scale=0.42, color=ui.TEXT_MUTED)
    ui.text(frame, f"{fps:4.1f} fps", (x + w - ui.PAD - 70, ty), scale=0.42, color=ui.fps_color(fps))
    ty += ui.LINE_H
    ui.status_dot(frame, (x + ui.PAD + 4, ty - 4), hand_found)
    ui.text(frame, "hand detected" if hand_found else "no hand", (x + ui.PAD + 16, ty), scale=0.45,
            color=ui.TEXT_PRIMARY if hand_found else ui.TEXT_MUTED)
    ty += ui.LINE_H
    idx = config.STATIC_GESTURES.index(selected_gesture) + 1
    ui.text(frame, f"[{idx}] {selected_gesture}", (x + ui.PAD, ty), scale=0.6, color=ui.ACCENT, heading=True)
    ty += ui.LINE_H
    ui.text(frame, ui.short_description(config.GESTURE_DESCRIPTIONS[selected_gesture]),
            (x + ui.PAD, ty), scale=0.42, color=ui.TEXT_MUTED)

    counts_y = ui.PAD + header_h + ui.PAD
    counts_h = ui.PAD + ui.LINE_H * (len(config.STATIC_GESTURES) + 1) + ui.PAD
    ui.panel(frame, x, counts_y, w, counts_h)
    cy = counts_y + ui.PAD + 6
    ui.text(frame, "SAMPLES  (total / session)", (x + ui.PAD, cy), scale=0.42, color=ui.TEXT_MUTED)
    cy += ui.LINE_H
    for gesture in config.STATIC_GESTURES:
        total = counts[gesture]
        this_session = session_counts[gesture]
        met = total >= config.min_samples_for(gesture)
        color = ui.SUCCESS if met else ui.WARNING
        is_selected = gesture == selected_gesture
        if is_selected:
            cv2.circle(frame, (x + ui.PAD + 3, cy - 4), 3, ui.ACCENT, -1, cv2.LINE_AA)
        ui.text(
            frame,
            f"{gesture:<10s} {total:4d} / {this_session:<3d}",
            (x + ui.PAD + 14, cy),
            scale=0.48,
            color=color,
            thickness=1 if not is_selected else 2,
        )
        cy += ui.LINE_H

    footer = f"1-{len(config.STATIC_GESTURES)} select   c capture   r reset session   q/Esc quit"
    ui.text(frame, footer, (x, counts_y + counts_h + ui.PAD + 8), scale=0.42, color=ui.TEXT_MUTED)

    if warning_text:
        ui.text(frame, warning_text, (x, counts_y + counts_h + ui.PAD + 8 + ui.LINE_H),
                 scale=0.46, color=ui.DANGER, thickness=2)
    return frame


def run(camera_index: int) -> None:
    dataset_io.ensure_csv_exists(config.GESTURE_DATA_CSV, CSV_HEADER)
    counts = dataset_io.count_existing_rows(config.GESTURE_DATA_CSV, config.STATIC_GESTURES)
    session_counts = {g: 0 for g in config.STATIC_GESTURES}

    landmarker = create_hand_landmarker()
    cap = open_webcam(camera_index)
    ts = MonotonicTimestamp()

    selected_gesture = config.STATIC_GESTURES[0]
    key_to_gesture = {ord(str(i + 1)): g for i, g in enumerate(config.STATIC_GESTURES)}

    # Brief on-screen warning after pressing 'c' with no hand detected, so
    # the no-op keypress is not silently ignored.
    WARNING_DISPLAY_SECONDS = 1.0
    warning_text: str | None = None
    warning_until: float = 0.0
    consecutive_read_failures = 0
    fps_counter = ui.FPSCounter()

    # Opened once for the whole session (see append_sample).
    csv_file = open(config.GESTURE_DATA_CSV, "a", newline="")
    csv_writer = csv.writer(csv_file)

    print("Data collection started. Press 'q' or Esc in the video window to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # A single failed grab is a transient hiccup, not a reason to
                # end the session.
                consecutive_read_failures += 1
                if consecutive_read_failures > config.MAX_CONSECUTIVE_FRAME_READ_FAILURES:
                    print("Warning: webcam stopped responding (too many failed frame reads).")
                    break
                continue
            consecutive_read_failures = 0
            frame = cv2.flip(frame, 1)  # mirror for a natural "looking in a mirror" feel

            detections = detect_hands(landmarker, frame, ts.next())

            if warning_text and time.time() > warning_until:
                warning_text = None

            fps = fps_counter.tick()
            annotated = draw_landmarks(frame, detections)
            annotated = _draw_hud(
                annotated, selected_gesture, counts, session_counts, bool(detections), warning_text, fps
            )
            cv2.imshow("Gesture Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 27 == Esc
                break
            if key in key_to_gesture:
                selected_gesture = key_to_gesture[key]
            elif key == ord("r"):
                session_counts = {g: 0 for g in config.STATIC_GESTURES}
            elif key == ord("c"):
                if detections:
                    append_sample(csv_writer, selected_gesture, detections[0].feature_vector)
                    csv_file.flush()
                    counts[selected_gesture] += 1
                    session_counts[selected_gesture] += 1
                else:
                    # Capture needs a hand detected on this frame; motion blur
                    # or being out of frame are the usual causes.
                    print("  Capture failed: no hand detected. Hold the pose steady and try again.")
                    warning_text = "No hand detected -- hold the pose steady"
                    warning_until = time.time() + WARNING_DISPLAY_SECONDS
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        csv_file.close()

    print("\nFinal per-gesture totals in", config.GESTURE_DATA_CSV)
    for g in config.STATIC_GESTURES:
        print(f"  {g:<10s} {counts[g]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    args = parser.parse_args()
    run(camera_index=args.camera)


if __name__ == "__main__":
    main()
