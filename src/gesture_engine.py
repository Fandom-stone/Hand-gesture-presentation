import os
from collections import Counter, deque

import joblib


class GestureEngine:
    """Classifies stable static gestures and emits one command per held pose."""

    def __init__(self, model_path="models/gesture_model.joblib",
                 confidence_threshold=0.50, smoothing_window=3):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Gesture model not found: {model_path}")
        self.model = joblib.load(model_path)
        self.confidence_threshold = confidence_threshold
        self.prediction_history = deque(maxlen=smoothing_window)
        self.last_triggered_gesture = None

    @staticmethod
    def extract_landmarks(landmarks):
        if landmarks is None:
            return None
        wrist_x, wrist_y = landmarks[0].x, landmarks[0].y
        return [value for point in landmarks for value in
                (point.x - wrist_x, point.y - wrist_y)]

    def predict(self, landmarks):
        if landmarks is None:
            self.reset()
            return "NO_GESTURE", 1.0, None
        features = self.extract_landmarks(landmarks)
        if len(features) != 42:
            return "NO_GESTURE", 0.0, None
        prediction = self.model.predict([features])[0]
        confidence = max(self.model.predict_proba([features])[0])
        if confidence < self.confidence_threshold:
            self.prediction_history.clear()
            return prediction, confidence, None
        self.prediction_history.append(prediction)
        counts = Counter(self.prediction_history)
        highest_count = max(counts.values())
        # Prefer the newest prediction when two classes are tied. This keeps
        # the UI responsive and avoids relying on unordered set iteration.
        gesture = next(item for item in reversed(self.prediction_history)
                       if counts[item] == highest_count)
        if gesture != self.last_triggered_gesture:
            self.last_triggered_gesture = None
        command = self.get_command(gesture)
        if command is not None and gesture == self.last_triggered_gesture:
            command = None
        elif command is not None:
            # Every static pose, including OPEN_PALM, fires just once while
            # it is held.  The presenter must change pose before using it again.
            self.last_triggered_gesture = gesture
        return gesture, confidence, command

    @staticmethod
    def get_command(gesture):
        return {
            "OPEN_PALM": "NEXT",
            "FIST": "PREVIOUS",
            "THUMBS_UP": "START",
            "PEACE": "END",
        }.get(gesture)

    def reset(self):
        self.prediction_history.clear()
        self.last_triggered_gesture = None
