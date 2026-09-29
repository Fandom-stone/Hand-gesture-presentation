"""
Dynamic swipe-gesture data collection.

Companion to collect_data.py, but records short MOTION CLIPS instead of
single-frame poses: each clip is config.SWIPE_WINDOW_FRAMES consecutive
hand-position samples, reduced by trajectory_features.extract_features()
into one feature vector, and appended as one row to
data/dynamic_gesture_data.csv (no per-person tagging -- anyone who runs
this adds to the same shared dataset).

Usage:
    python -m src.collect_dynamic_data
    python -m src.collect_dynamic_data --camera 1

Controls (shown on-screen too):
    1-3   select which class you're about to record:
          SWIPE_LEFT, SWIPE_RIGHT, NO_SWIPE (matches config.DYNAMIC_GESTURES)
    c     start recording one clip of the selected class -- perform the
          motion right after pressing it; the on-screen bar shows capture
          progress. A brief dropped-detection frame (common during a fast
          swipe, which blurs the frame) is tolerated and does not discard
          the clip; only losing the hand for several frames in a row does
          (no partial/broken clips get saved).
    r     reset the on-screen "captured this session" counters
    q/Esc quit

Recording tips:
  - SWIPE_LEFT / SWIPE_RIGHT: hold your open hand up, then swipe it
    steadily across the frame in ~0.5-1 second. Vary speed and how much of
    the frame you cross across different clips.
  - NO_SWIPE: this is the background/negative class -- record clips of
    standing still, holding a static pose (thumbs up, peace, etc.), small
    random movements, slow drifting, adjusting your hand position. This
    class needs the MOST variety of all of them, since "everything that
    isn't a clean swipe" is a big space.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import cv2

from . import config, dataset_io, ui
from .hand_tracker import (
    MonotonicTimestamp,
    create_hand_landmarker,
    detect_hands,
    draw_landmarks,
    open_webcam,
)
from .trajectory_features import TrajectoryTracker, hand_center

FEATURE_COLUMNS = list(config.DYNAMIC_FEATURE_NAMES)
CSV_HEADER = ["gesture", "timestamp"] + FEATURE_COLUMNS


def append_clip(csv_path: Path, gesture: str, feature_vector) -> None:
    """One clip per 'c' press, so reopening the file each time costs nothing
    (unlike the static collector, where key-repeat fires many a second)."""
    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(dataset_io.timestamped_row(gesture, feature_vector))


def _draw_hud(frame, selected_gesture, counts, session_counts, recording, progress, hand_found, fps):
    x = ui.PAD
    w = 380

    header_h = ui.PAD + 3 * ui.LINE_H + ui.PAD
    ui.panel(frame, x, ui.PAD, w, header_h)
    ty = ui.PAD + ui.PAD
    ui.text(frame, "SWIPE COLLECTION", (x + ui.PAD, ty), scale=0.42, color=ui.TEXT_MUTED)
    ui.text(frame, f"{fps:4.1f} fps", (x + w - ui.PAD - 70, ty), scale=0.42, color=ui.fps_color(fps))
    ty += ui.LINE_H
    ui.status_dot(frame, (x + ui.PAD + 4, ty - 4), hand_found)
    ui.text(frame, "hand detected" if hand_found else "no hand", (x + ui.PAD + 16, ty), scale=0.45,
            color=ui.TEXT_PRIMARY if hand_found else ui.TEXT_MUTED)
    ty += ui.LINE_H
    idx = config.DYNAMIC_GESTURES.index(selected_gesture) + 1
    ui.text(frame, f"[{idx}] {selected_gesture}", (x + ui.PAD, ty), scale=0.6, color=ui.ACCENT, heading=True)
    ty += ui.LINE_H
    ui.text(frame, ui.short_description(config.DYNAMIC_GESTURE_DESCRIPTIONS[selected_gesture]),
            (x + ui.PAD, ty), scale=0.42, color=ui.TEXT_MUTED)

    y = ui.PAD + header_h + ui.PAD

    if recording:
        rec_h = ui.PAD + ui.LINE_H + 16 + ui.PAD
        ui.panel(frame, x, y, w, rec_h)
        ui.text(frame, f"RECORDING {selected_gesture}  {progress}/{config.SWIPE_WINDOW_FRAMES}",
                (x + ui.PAD, y + ui.PAD + 12), scale=0.5, color=ui.DANGER, heading=True)
        ui.progress_bar(frame, x + ui.PAD, y + ui.PAD + 20, w - 2 * ui.PAD, 10,
                         progress / config.SWIPE_WINDOW_FRAMES, color=ui.DANGER)
        y += rec_h + ui.PAD

    counts_h = ui.PAD + ui.LINE_H * (len(config.DYNAMIC_GESTURES) + 1) + ui.PAD
    ui.panel(frame, x, y, w, counts_h)
    cy = y + ui.PAD + 6
    ui.text(frame, "CLIPS  (total / session)", (x + ui.PAD, cy), scale=0.42, color=ui.TEXT_MUTED)
    cy += ui.LINE_H
    for gesture in config.DYNAMIC_GESTURES:
        total = counts[gesture]
        this_session = session_counts[gesture]
        met = total >= config.min_clips_for(gesture)
        color = ui.SUCCESS if met else ui.WARNING
        is_selected = gesture == selected_gesture
        if is_selected:
            cv2.circle(frame, (x + ui.PAD + 3, cy - 4), 3, ui.ACCENT, -1, cv2.LINE_AA)
        ui.text(
            frame,
            f"{gesture:<11s} {total:4d} / {this_session:<3d}",
            (x + ui.PAD + 14, cy),
            scale=0.48,
            color=color,
            thickness=1 if not is_selected else 2,
        )
        cy += ui.LINE_H

    footer = "1-3 select   c record clip   r reset session   q/Esc quit"
    ui.text(frame, footer, (x, y + counts_h + ui.PAD + 8), scale=0.42, color=ui.TEXT_MUTED)
    return frame


def run(camera_index: int) -> None:
    dataset_io.ensure_csv_exists(config.DYNAMIC_DATA_CSV, CSV_HEADER)
    counts = dataset_io.count_existing_rows(config.DYNAMIC_DATA_CSV, config.DYNAMIC_GESTURES)
    session_counts = {g: 0 for g in config.DYNAMIC_GESTURES}

    landmarker = create_hand_landmarker()
    cap = open_webcam(camera_index)
    ts = MonotonicTimestamp()

    selected_gesture = config.DYNAMIC_GESTURES[0]
    key_to_gesture = {ord(str(i + 1)): g for i, g in enumerate(config.DYNAMIC_GESTURES)}

    recording = False
    tracker: Optional[TrajectoryTracker] = None
    consecutive_read_failures = 0
    fps_counter = ui.FPSCounter()

    print("Dynamic swipe data collection started. Press 'q' or Esc in the video window to quit.")
    print("Watch the fps reading in the top-right of the window -- if it's consistently")
    print("below ~15, the pipeline can't keep up and fast swipes will be missed/misread.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                # Transient driver hiccup, not a disconnect. Skip the frame
                # and leave any in-progress recording untouched.
                consecutive_read_failures += 1
                if consecutive_read_failures > config.MAX_CONSECUTIVE_FRAME_READ_FAILURES:
                    print("Warning: webcam stopped responding (too many failed frame reads).")
                    break
                continue
            consecutive_read_failures = 0
            frame = cv2.flip(frame, 1)

            detections = detect_hands(landmarker, frame, ts.next())
            hand_found = bool(detections)

            if recording:
                position = hand_center(detections[0].landmarks) if hand_found else None
                was_reset = tracker.update(position)
                if was_reset:
                    # Only a sustained loss of tracking discards the clip; a
                    # brief mid-swipe dropout is tolerated.
                    print(f"  Clip discarded (hand lost too long mid-recording): {selected_gesture}")
                    recording = False
                    tracker = None
                elif tracker.is_full:
                    feature_vector = tracker.feature_vector()
                    append_clip(config.DYNAMIC_DATA_CSV, selected_gesture, feature_vector)
                    counts[selected_gesture] += 1
                    session_counts[selected_gesture] += 1
                    print(f"  Captured clip: {selected_gesture} (total {counts[selected_gesture]})")
                    recording = False
                    tracker = None

            progress = tracker.frame_count if tracker else 0
            fps = fps_counter.tick()
            annotated = draw_landmarks(frame, detections)
            annotated = _draw_hud(
                annotated, selected_gesture, counts, session_counts, recording, progress, hand_found, fps
            )
            cv2.imshow("Dynamic Swipe Data Collection", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if not recording:
                if key in key_to_gesture:
                    selected_gesture = key_to_gesture[key]
                elif key == ord("r"):
                    session_counts = {g: 0 for g in config.DYNAMIC_GESTURES}
                elif key == ord("c"):
                    recording = True
                    tracker = TrajectoryTracker()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()

    print("\nFinal per-class clip totals in", config.DYNAMIC_DATA_CSV)
    for g in config.DYNAMIC_GESTURES:
        print(f"  {g:<10s} {counts[g]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    args = parser.parse_args()
    run(camera_index=args.camera)


if __name__ == "__main__":
    main()
