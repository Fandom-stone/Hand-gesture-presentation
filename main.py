import time

import cv2

from src.hand_tracker import HandTracker
from src.gesture_engine import GestureEngine
from src.presentation_controller import PresentationController


MODEL_PATH = "models/hand_landmarker.task"
GESTURE_MODEL_PATH = "models/gesture_model.joblib"
CAMERA_INDEX = 0
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
WINDOW_WIDTH, WINDOW_HEIGHT = 960, 540


def draw_hand(frame, landmarks):
    if landmarks is None:
        return
    height, width, _ = frame.shape
    connections = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6),
                   (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                   (0, 13), (13, 14), (14, 15), (15, 16), (0, 17),
                   (17, 18), (18, 19), (19, 20), (5, 9), (9, 13), (13, 17)]
    points = [(int(point.x * width), int(point.y * height)) for point in landmarks]
    for point in points:
        cv2.circle(frame, point, 4, (86, 220, 137), -1)
    for start, end in connections:
        cv2.line(frame, points[start], points[end], (97, 218, 251), 2)


def draw_dashboard(display, gesture, confidence, command, fps):
    """Draw a restrained status bar that keeps the camera feed readable."""
    height, width, _ = display.shape
    overlay = display.copy()
    cv2.rectangle(overlay, (14, 14), (width - 14, 64), (18, 24, 38), -1)
    cv2.rectangle(overlay, (14, height - 40), (width - 14, height - 14), (18, 24, 38), -1)
    cv2.addWeighted(overlay, 0.78, display, 0.22, 0, display)
    cv2.circle(display, (34, 39), 6, (73, 214, 134), -1)
    cv2.putText(display, "GESTURE CONTROL", (50, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (245, 247, 250), 1)
    cv2.putText(display, f"{fps:.0f} FPS", (width - 88, 45), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (170, 181, 199), 1)
    cv2.putText(display, gesture.replace("_", " "), (22, height - 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 1)
    cv2.putText(display, f"{confidence * 100:.0f}%", (170, height - 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 206, 255), 1)
    if command:
        cv2.rectangle(display, (width // 2 - 95, 86), (width // 2 + 95, 125), (31, 133, 89), -1)
        cv2.putText(display, command, (width // 2 - 35, 112), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (255, 255, 255), 2)


def main():
    tracker = HandTracker(MODEL_PATH, CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, max_hands=1)
    gesture_engine = GestureEngine(GESTURE_MODEL_PATH, confidence_threshold=0.50,
                                   smoothing_window=5)
    presentation_controller = PresentationController(cooldown=0.8)
    previous_time, fps = time.perf_counter(), 0.0
    print("Gesture Presentation Assistant\nOpen palm: next | Fist: previous | Q: quit")
    try:
        while True:
            frame, landmarks = tracker.get_frame()
            if frame is None:
                print("Could not read camera frame.")
                break
            draw_hand(frame, landmarks)
            gesture, confidence, command = gesture_engine.predict(landmarks)
            executed_command = None
            if command is not None and presentation_controller.execute(command):
                executed_command = command
                print(f"Command executed: {executed_command}")
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 / max(now - previous_time, 0.001)
            previous_time = now
            display = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
            draw_dashboard(display, gesture, confidence, executed_command, fps)
            cv2.imshow("Gesture Presentation Assistant", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
