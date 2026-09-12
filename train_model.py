import os
import glob
import warnings

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used with `sklearn.utils.parallel.Parallel`"
)

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# SETTINGS
# ============================================================

DATASET_FOLDER = "dataset"
MODEL_FOLDER = "models"

MODEL_OUTPUT = os.path.join(
    MODEL_FOLDER,
    "gesture_model.joblib"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42

EXPECTED_FEATURES = 42


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    csv_files = glob.glob(
        os.path.join(
            DATASET_FOLDER,
            "*.csv"
        )
    )

    if not csv_files:

        raise RuntimeError(
            "No CSV dataset files found."
        )

    dataframes = []

    print()
    print("Loading datasets...")
    print()

    for file in sorted(csv_files):

        df = pd.read_csv(file)

        if "label" not in df.columns:

            print(
                f"Skipping invalid file: {file}"
            )

            continue

        dataframes.append(df)

        print(
            f"{os.path.basename(file):20s}"
            f" {len(df)} samples"
        )

    if not dataframes:

        raise RuntimeError(
            "No valid datasets found."
        )

    return pd.concat(
        dataframes,
        ignore_index=True
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(data):

    feature_columns = [
        column
        for column in data.columns
        if column != "label"
    ]

    X_raw = data[
        feature_columns
    ].values.astype(
        np.float32
    )

    y = data["label"].values

    # --------------------------------------------------------
    # New static dataset
    #
    # 42 features:
    # 21 landmarks × X/Y
    # --------------------------------------------------------

    if X_raw.shape[1] == EXPECTED_FEATURES:

        print()
        print(
            "Dataset format: "
            "42-feature static gestures"
        )

        X = X_raw

    # --------------------------------------------------------
    # Old sequence dataset
    #
    # 1260 features:
    # 30 frames × 42 features
    #
    # We take the first frame.
    # --------------------------------------------------------

    elif X_raw.shape[1] == 1260:

        print()
        print(
            "Dataset format: "
            "old 30-frame sequence"
        )

        print(
            "Using the first frame "
            "for static gesture training."
        )

        X = X_raw[
            :,
            :EXPECTED_FEATURES
        ]

    else:

        raise RuntimeError(
            f"Unexpected feature count: "
            f"{X_raw.shape[1]}"
        )

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    X_train,
    y_train
):

    print()
    print(
        "Training Random Forest..."
    )
    print()

    model = RandomForestClassifier(

        n_estimators=300,

        max_depth=None,

        min_samples_split=2,

        min_samples_leaf=1,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced"
    )

    model.fit(
        X_train,
        y_train
    )

    return model


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_test,
    y_test
):

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print()
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print()

    print(
        f"Test Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print()

    print(
        "Classification Report:"
    )

    print()

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    labels = sorted(
        np.unique(y_test)
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=labels
    )

    print(
        "Confusion Matrix:"
    )

    print()

    print(
        "Labels:",
        labels
    )

    print()

    print(matrix)

    return accuracy


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model):

    os.makedirs(
        MODEL_FOLDER,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_OUTPUT
    )

    print()

    print(
        "Model saved to:"
    )

    print(
        MODEL_OUTPUT
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("STATIC HAND GESTURE MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    data = load_dataset()

    print()

    print(
        f"Total samples loaded: "
        f"{len(data)}"
    )

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    X, y = prepare_data(
        data
    )

    print()

    print(
        f"Features per sample: "
        f"{X.shape[1]}"
    )

    print()

    print(
        "Gesture classes:"
    )

    for label in sorted(
        np.unique(y)
    ):

        count = np.sum(
            y == label
        )

        print(
            f"  {label:15s} "
            f"{count} samples"
        )

    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(

            X,
            y,

            test_size=TEST_SIZE,

            random_state=RANDOM_STATE,

            stratify=y
        )
    )

    print()

    print(
        f"Training samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    model = train_model(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_model(
        model
    )

    print()
    print("=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()