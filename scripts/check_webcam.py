"""
Phase 1 - Environment Setup sanity check (Section 8.1).

Opens the webcam and shows a live preview window so you can confirm OpenCV
can see your camera before building anything else. No MediaPipe/ML code
involved -- if this doesn't work, nothing downstream will either.

Usage:
    python scripts/check_webcam.py
    python scripts/check_webcam.py --camera 1
Press 'q' or Esc to quit.
"""

import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(
            f"Could not open camera index {args.camera}. Things to check:\n"
            "  - Is a webcam physically connected?\n"
            "  - Is another app (Zoom, Teams, browser tab) already using it?\n"
            "  - On Windows, check Settings > Privacy > Camera access for apps.\n"
            "  - Try --camera 1 (or 2) if you have multiple capture devices."
        )
        return

    print("Webcam opened OK. Press 'q' or Esc in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read a frame.")
            break
        frame = cv2.flip(frame, 1)
        cv2.putText(
            frame, "Webcam OK - press q/Esc to quit", (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )
        cv2.imshow("Webcam Check", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
