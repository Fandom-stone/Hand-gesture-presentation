import cv2
import mediapipe as mp


class HandTracker:
    def __init__(
        self,
        model_path,
        camera_index=0,
        frame_width=1280,
        frame_height=720,
        max_hands=1
    ):
        self.model_path = model_path
        self.camera_index = camera_index
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_hands = max_hands

        self.latest_result = None
        self.frame_timestamp = 0

        self.BaseOptions = mp.tasks.BaseOptions
        self.HandLandmarker = mp.tasks.vision.HandLandmarker
        self.HandLandmarkerOptions = (
            mp.tasks.vision.HandLandmarkerOptions
        )
        self.RunningMode = mp.tasks.vision.RunningMode

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            self.frame_width
        )

        self.cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            self.frame_height
        )

        options = self.HandLandmarkerOptions(
            base_options=self.BaseOptions(
                model_asset_path=self.model_path
            ),
            running_mode=self.RunningMode.LIVE_STREAM,
            num_hands=self.max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._result_callback
        )

        self.landmarker = (
            self.HandLandmarker.create_from_options(options)
        )

    def _result_callback(
        self,
        result,
        output_image,
        timestamp_ms
    ):
        self.latest_result = result

    def get_frame(self):
        success, frame = self.cap.read()

        if not success:
            return None, None

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        self.frame_timestamp += 1

        self.landmarker.detect_async(
            mp_image,
            self.frame_timestamp
        )

        landmarks = None

        if self.latest_result is not None:

            if self.latest_result.hand_landmarks:
                landmarks = (
                    self.latest_result.hand_landmarks[0]
                )

        return frame, landmarks

    def release(self):
        self.cap.release()
        self.landmarker.close()