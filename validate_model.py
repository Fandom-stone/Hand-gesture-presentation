import csv
import os
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


MODEL_PATH = "models/gesture_model.joblib"
VALIDATION_FOLDER = "validation"

GESTURES = [
    "FIST",
    "OPEN_PALM",
    "PEACE",
    "THUMBS_UP"
]


def load_validation_data():
    features = []
    labels = []

    for gesture in GESTURES:

        path = os.path.join(
            VALIDATION_FOLDER,
            f"{gesture.lower()}.csv"
        )

        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue

        with open(path, "r", newline="") as file:

            reader = csv.reader(file)

            # Skip header
            next(reader, None)

            count = 0

            for row in reader:

                if len(row) != 43:
                    continue

                sample = [
                    float(value)
                    for value in row[:42]
                ]

                label = row[42]

                features.append(sample)
                labels.append(label)

                count += 1

            print(
                f"{gesture:<12} "
                f"Validation: {count:3} samples"
            )

    return features, labels


def main():

    print("\n==============================")
    print("MODEL VALIDATION")
    print("==============================")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print("\nLoading trained model...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded successfully.")

    print("\nLoading frozen validation data...")

    X_validation, y_validation = load_validation_data()

    print(
        f"\nTotal validation samples: "
        f"{len(X_validation)}"
    )

    if not X_validation:
        print("\nNo validation samples found.")
        return

    print("\nRunning predictions...")

    predictions = model.predict(X_validation)

    accuracy = accuracy_score(
        y_validation,
        predictions
    )

    print("\n==============================")
    print("VALIDATION RESULTS")
    print("==============================")

    print(
        f"\nValidation Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_validation,
            predictions,
            labels=GESTURES,
            zero_division=0
        )
    )

    print("Confusion Matrix:\n")

    matrix = confusion_matrix(
        y_validation,
        predictions,
        labels=GESTURES
    )

    print(f"Labels: {GESTURES}")

    print(matrix)

    print("\nVALIDATION COMPLETE")


if __name__ == "__main__":
    main()