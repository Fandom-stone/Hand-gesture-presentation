import csv
import os
import sys
import time

import cv2

from src.hand_tracker import HandTracker


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/hand_landmarker.task"
DATASET_FOLDER = "dataset"

CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

WINDOW_WIDTH = 640
WINDOW_HEIGHT = 360

# Number of NEW samples collected in each run
SAMPLES_PER_RUN = 100

CAPTURE_INTERVAL = 0.15
COUNTDOWN_SECONDS = 2.0


# ============================================================
# GESTURES
# ============================================================

GESTURES = {
    "open_palm": "OPEN_PALM",
    "fist": "FIST",
    "thumbs_up": "THUMBS_UP",
    "peace": "PEACE",
    "no_gesture": "NO_GESTURE"
}


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_landmarks(landmarks):
    if landmarks is None:
        return None

    features = []

    wrist_x = landmarks[0].x
    wrist_y = landmarks[0].y

    for landmark in landmarks:
        x = landmark.x - wrist_x
        y = landmark.y - wrist_y

        features.append(x)
        features.append(y)

    return features


# ============================================================
# DATASET FUNCTIONS
# ============================================================

def get_dataset_path(gesture_name):
    filename = f"{gesture_name.lower()}.csv"
    return os.path.join(DATASET_FOLDER, filename)


def get_existing_count(path):
    if not os.path.exists(path):
        return 0

    with open(path, "r", newline="") as file:
        reader = csv.reader(file)

        # Skip header
        next(reader, None)

        count = 0

        for row in reader:
            if len(row) == 43:
                count += 1

        return count


def create_dataset_file(gesture_name, existing_count):
    os.makedirs(DATASET_FOLDER, exist_ok=True)

    path = get_dataset_path(gesture_name)

    if existing_count == 0:
        file = open(path, "w", newline="")

        writer = csv.writer(file)

        header = []

        for landmark_number in range(21):
            header.append(f"x{landmark_number}")
            header.append(f"y{landmark_number}")

        header.append("label")

        writer.writerow(header)

    else:
        file = open(path, "a", newline="")
        writer = csv.writer(file)

    return file, writer


# ============================================================
# DRAW HAND
# ============================================================

def draw_hand(frame, landmarks):
    if landmarks is None:
        return

    height, width, _ = frame.shape

    connections = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17)
    ]

    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

    for start, end in connections:
        cv2.line(
            frame,
            points[start],
            points[end],
            (0, 255, 0),
            2
        )


# ============================================================
# DISPLAY
# ============================================================

def show_frame(frame, gesture_name, current_run, saved):
    frame = cv2.resize(
        frame,
        (WINDOW_WIDTH, WINDOW_HEIGHT)
    )

    cv2.putText(
        frame,
        f"COLLECTING: {gesture_name}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"This run: {saved}/{current_run}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        "Q - to quit",
        (15, WINDOW_HEIGHT - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Hand Gesture Data Collection",
        frame
    )


# ============================================================
# COUNTDOWN
# ============================================================

def countdown(tracker):
    start_time = time.perf_counter()

    while True:
        frame, landmarks = tracker.get_frame()

        if frame is None:
            return False

        draw_hand(frame, landmarks)

        elapsed = time.perf_counter() - start_time
        remaining = COUNTDOWN_SECONDS - elapsed

        if remaining <= 0:
            break

        number = int(remaining) + 1

        frame = cv2.resize(
            frame,
            (WINDOW_WIDTH, WINDOW_HEIGHT)
        )

        cv2.putText(
            frame,
            f"Starting in {number}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Q - to quit",
            (15, WINDOW_HEIGHT - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Hand Gesture Data Collection",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

    return True


# ============================================================
# COLLECT SAMPLES
# ============================================================

def collect_samples(
    tracker,
    gesture_name,
    writer,
    file,
    samples_to_collect
):
    saved = 0
    last_capture_time = 0

    while saved < samples_to_collect:

        frame, landmarks = tracker.get_frame()

        if frame is None:
            return False

        draw_hand(frame, landmarks)

        current_time = time.perf_counter()

        if (
            current_time - last_capture_time
            >= CAPTURE_INTERVAL
        ):

            # ------------------------------------------------
            # NO GESTURE
            # ------------------------------------------------

            if gesture_name == "NO_GESTURE":

                if landmarks is None:
                    features = [0.0] * 42
                else:
                    features = None

            # ------------------------------------------------
            # NORMAL GESTURES
            # ------------------------------------------------

            else:
                features = extract_landmarks(landmarks)

            # ------------------------------------------------
            # SAVE SAMPLE
            # ------------------------------------------------

            if features is not None:

                row = features + [gesture_name]

                writer.writerow(row)
                file.flush()

                saved += 1

                last_capture_time = current_time

                print(
                    f"{gesture_name}: "
                    f"{saved}/{samples_to_collect}"
                )

        show_frame(
            frame,
            gesture_name,
            samples_to_collect,
            saved
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK COMMAND
    # --------------------------------------------------------

    if len(sys.argv) != 2:

        print(
            "Usage: python collect_data.py <gesture>"
        )

        print("\nAvailable gestures:")

        for key in GESTURES:
            print(f"  {key}")

        return


    gesture_key = sys.argv[1].lower()


    # --------------------------------------------------------
    # CHECK GESTURE
    # --------------------------------------------------------

    if gesture_key not in GESTURES:

        print(
            f"Unknown gesture: {gesture_key}"
        )

        print("\nAvailable gestures:")

        for key in GESTURES:
            print(f"  {key}")

        return


    gesture_name = GESTURES[gesture_key]


    # --------------------------------------------------------
    # EXISTING DATA
    # --------------------------------------------------------

    dataset_path = get_dataset_path(gesture_name)

    existing_count = get_existing_count(dataset_path)


    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    print("\n==============================")
    print("HAND GESTURE DATA COLLECTION")
    print("==============================")

    print(f"Gesture          : {gesture_name}")
    print(f"Existing samples : {existing_count}")
    print(f"New samples      : {SAMPLES_PER_RUN}")
    print(
        f"After this run   : "
        f"{existing_count + SAMPLES_PER_RUN}"
    )


    # --------------------------------------------------------
    # INSTRUCTIONS
    # --------------------------------------------------------

    print("\nInstructions:")

    if gesture_name == "NO_GESTURE":

        print(
            "- Keep both hands outside the camera."
        )

    else:

        print(
            f"- Hold the {gesture_name} gesture."
        )

    print("- Move your hand slightly between samples.")
    print("- Vary distance and position a little.")
    print("- Press SPACE to start.")
    print("- Press Q to quit.")


    # --------------------------------------------------------
    # CAMERA
    # --------------------------------------------------------

    tracker = HandTracker(
        model_path=MODEL_PATH,
        camera_index=CAMERA_INDEX,
        frame_width=FRAME_WIDTH,
        frame_height=FRAME_HEIGHT,
        max_hands=1
    )


    file = None


    try:

        # ----------------------------------------------------
        # READY SCREEN
        # ----------------------------------------------------

        while True:

            frame, landmarks = tracker.get_frame()

            if frame is None:
                break

            draw_hand(frame, landmarks)

            frame = cv2.resize(
                frame,
                (WINDOW_WIDTH, WINDOW_HEIGHT)
            )

            cv2.putText(
                frame,
                f"Ready: {gesture_name}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Existing: {existing_count}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"This run: +{SAMPLES_PER_RUN}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Press SPACE to start",
                (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "Q - to quit",
                (15, WINDOW_HEIGHT - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "Hand Gesture Data Collection",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                break

            if key == ord("q"):
                return


        # ----------------------------------------------------
        # COUNTDOWN
        # ----------------------------------------------------

        if not countdown(tracker):
            return


        # ----------------------------------------------------
        # OPEN DATASET
        # ----------------------------------------------------

        file, writer = create_dataset_file(
            gesture_name,
            existing_count
        )


        # ----------------------------------------------------
        # COLLECT
        # ----------------------------------------------------

        success = collect_samples(
            tracker,
            gesture_name,
            writer,
            file,
            SAMPLES_PER_RUN
        )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        if success:

            final_count = (
                existing_count
                + SAMPLES_PER_RUN
            )

            print(
                f"\nDONE! {gesture_name} now has "
                f"{final_count} samples."
            )

        else:

            print("\nCollection stopped.")


    finally:

        if file is not None:
            file.close()

        tracker.release()

        cv2.destroyAllWindows()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()