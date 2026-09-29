"""
Dynamic swipe model training.

Loads data/dynamic_gesture_data.csv (motion-feature clips recorded by
collect_dynamic_data.py), cleans it, splits it, trains a Random Forest
(plus KNN/SVM baselines, mirroring train_model.py), and saves the Random
Forest as models/dynamic_model.pkl for infer_realtime.py.

Two cleaning filters run before the split, both undoing the same recording
artifact -- see drop_direction_contradicting_clips(). Augmentation is applied
to the training split ONLY, so the reported accuracy is measured against real
recorded clips.

Like the static trainer, every model is scored on both a time-based and a
random split, and by both label and action accuracy.

Usage:
    python -m src.train_dynamic_model
    python -m src.train_dynamic_model --csv data/dynamic_gesture_data.csv
    python -m src.train_dynamic_model --split random
"""

from __future__ import annotations

import argparse
import sys
import time

import joblib
import numpy as np
import pandas as pd

from . import config, dataset_io, evaluation

FEATURE_COLUMNS = list(config.DYNAMIC_FEATURE_NAMES)

_DX = FEATURE_COLUMNS.index("dx")
# Features that scale with distance/speed. straightness and
# direction_consistency are ratios, unchanged by moving faster or further.
_MAGNITUDE_FEATURES = [
    FEATURE_COLUMNS.index(n)
    for n in ("dx", "dy", "path_length", "net_displacement", "mean_speed", "max_speed")
]


def load_dataset(csv_path):
    return dataset_io.load_training_csv(
        csv_path, FEATURE_COLUMNS, "collect_dynamic_data.py"
    )


def check_class_balance(df: pd.DataFrame) -> None:
    counts = df["gesture"].value_counts()
    print("Clips per class:")
    for gesture in config.DYNAMIC_GESTURES:
        n = int(counts.get(gesture, 0))
        flag = "" if n >= config.min_clips_for(gesture) else "  <-- below recommended minimum"
        print(f"  {gesture:<10s} {n:5d}{flag}")

    missing = [g for g in config.DYNAMIC_GESTURES if counts.get(g, 0) == 0]
    if missing:
        print(
            f"\nERROR: no clips at all for: {missing}. "
            "Collect clips for every class (including NO_SWIPE!) before training."
        )
        sys.exit(1)

    present = counts[counts > 0]
    imbalance_ratio = present.max() / present.min()
    if imbalance_ratio > 3.0:
        print(
            f"\nWarning: class imbalance ratio is {imbalance_ratio:.1f}x. "
            "NO_SWIPE naturally needs more clips than the swipes (it's the "
            "background class), but if one SWIPE class is starved relative "
            "to the other, collect a few more clips for it."
        )


# --------------------------------------------------------------------------
# Cleaning
#
# Both filters return a keep-mask rather than filtered arrays, so the caller
# can apply the same mask to the features, the labels AND the capture
# timestamps and keep all three aligned (the timestamps are what make the
# time-based split possible).
# --------------------------------------------------------------------------
def direction_contradicting_mask(X, y) -> np.ndarray:
    """False for clips whose sideways travel points the OPPOSITE way to their
    own label.

    The recorder captures a fixed-length window from the moment 'c' is
    pressed, so swiping quickly and bringing the hand back puts the return
    motion inside the same window, cancelling or reversing the net travel.
    Dropped before the split: scoring a model against wrong ground truth is
    meaningless.
    """
    keep = np.ones(len(y), dtype=bool)
    for i, (features, label) in enumerate(zip(X, y)):
        dx = features[_DX]
        if label == config.SWIPE_LEFT and dx >= 0:
            keep[i] = False
        elif label == config.SWIPE_RIGHT and dx <= 0:
            keep[i] = False
    return keep


def degenerate_swipe_mask(X, y) -> np.ndarray:
    """False for swipe clips that barely moved. Same return-motion artifact,
    except the travel cancelled to near-zero rather than reversing, leaving
    clips indistinguishable from a still hand. Left in, they teach the model
    that any faint drift is a swipe."""
    keep = np.ones(len(y), dtype=bool)
    for i, (features, label) in enumerate(zip(X, y)):
        if label in (config.SWIPE_LEFT, config.SWIPE_RIGHT):
            if abs(features[_DX]) < config.MIN_TRAINING_SWIPE_DX:
                keep[i] = False
    return keep


def augment_training_split(X, y) -> tuple[np.ndarray, np.ndarray]:
    """Expand the training split with mirrored and speed-scaled variants.
    Never call this on the test split."""
    X_parts, y_parts = [X], [y]

    mirrored = X.copy()
    mirrored[:, _DX] *= -1
    mirrored_y = np.where(
        y == config.SWIPE_LEFT,
        config.SWIPE_RIGHT,
        np.where(y == config.SWIPE_RIGHT, config.SWIPE_LEFT, y),
    )
    X_parts.append(mirrored)
    y_parts.append(mirrored_y)

    for factor in config.DYNAMIC_AUGMENT_SPEED_FACTORS:
        for base_X, base_y in ((X, y), (mirrored, mirrored_y)):
            scaled = base_X.copy()
            scaled[:, _MAGNITUDE_FEATURES] *= factor
            X_parts.append(scaled)
            y_parts.append(base_y)

    return np.vstack(X_parts), np.concatenate(y_parts)


def train_and_score(X, y, train_idx, test_idx, random_state: int) -> tuple[list[dict], int]:
    """Augment the training split, fit every candidate on it, and score each
    against the REAL (never augmented) test clips."""
    X_train, y_train = X[train_idx], y[train_idx]
    real_train_clips = len(X_train)
    if config.AUGMENT_DYNAMIC_TRAINING_DATA:
        X_train, y_train = augment_training_split(X_train, y_train)
        print(
            f"Augmented training split {real_train_clips} -> {len(X_train)} rows "
            f"(mirrored + speed-scaled {config.DYNAMIC_AUGMENT_SPEED_FACTORS}).\n"
            f"The {len(test_idx)} test clips are REAL recorded clips only -- never augmented."
        )

    results = []
    for name, model in evaluation.build_candidates(
        config.DYNAMIC_FOREST_TREES, random_state
    ).items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_seconds = time.time() - t0
        result = evaluation.evaluate_model(
            name,
            model,
            X[test_idx],
            y[test_idx],
            labels=config.DYNAMIC_GESTURES,
            action_map=config.DYNAMIC_ACTION_MAP,
        )
        result["model"] = model
        result["train_seconds"] = train_seconds
        print(f"Training time: {train_seconds:.3f}s")
        results.append(result)
    return results, len(X_train)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(config.DYNAMIC_DATA_CSV))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("temporal", "random"),
        default=config.EVALUATION_SPLIT,
        help=(
            "Which split the DEPLOYED model is trained and reported on. "
            "Both figures are printed either way. Default: %(default)s"
        ),
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.csv}")
    X, y, timestamps, df = load_dataset(args.csv)
    print(f"Total clips: {len(df)}  |  Feature vector length: {X.shape[1]}")
    check_class_balance(df)

    dropped = degenerate = 0
    if config.DROP_DIRECTION_CONTRADICTING_CLIPS:
        keep = direction_contradicting_mask(X, y)
        dropped = int((~keep).sum())
        X, y, timestamps = X[keep], y[keep], timestamps[keep]
        if dropped:
            print(
                f"\nDropped {dropped} clip(s) whose sideways travel contradicted their own\n"
                f"label (hand's return motion caught inside the recording window)."
            )

        keep = degenerate_swipe_mask(X, y)
        degenerate = int((~keep).sum())
        X, y, timestamps = X[keep], y[keep], timestamps[keep]
        if degenerate:
            print(
                f"Dropped {degenerate} swipe clip(s) that barely moved "
                f"(|dx| < {config.MIN_TRAINING_SWIPE_DX}) -- too small to be\n"
                f"distinguishable from a still hand."
            )
        if dropped or degenerate:
            print(f"{len(y)} clips remain.")

    splits = {
        "temporal": evaluation.temporal_split_indices(y, timestamps, args.test_size),
        "random": evaluation.random_split_indices(y, args.test_size, args.random_state),
    }

    all_results = {}
    augmented_rows = {}
    for split_name, (train_idx, test_idx) in splits.items():
        primary = split_name == args.split
        print(
            f"\n{'=' * 74}\n"
            f"{split_name.upper()} SPLIT: {len(train_idx)} train / {len(test_idx)} test clips"
            f"{'   <-- deployed model trained on this split' if primary else ''}\n"
            f"{'=' * 74}"
        )
        all_results[split_name], augmented_rows[split_name] = train_and_score(
            X, y, train_idx, test_idx, args.random_state
        )
        evaluation.print_summary(all_results[split_name], has_action_accuracy=True)

    print(f"\n{'=' * 74}\nHEADLINE COMPARISON (Random Forest)\n{'=' * 74}")
    print(f"  {'split':<12s} {'label acc':>10s} {'action acc':>12s}")
    for split_name, results in all_results.items():
        rf = next(r for r in results if r["name"].startswith("Random Forest"))
        print(f"  {split_name:<12s} {rf['accuracy']:>10.4f} {rf['action_accuracy']:>12.4f}")
    test_clips = len(splits[args.split][1])
    print(
        f"  Note: only {test_clips} real clips in the test set, so treat these figures as\n"
        f"  trustworthy in direction, not in precision."
    )

    primary = next(
        r for r in all_results[args.split] if r["name"].startswith("Random Forest")
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(primary["model"], config.DYNAMIC_MODEL_PATH)
    print(f"\nSaved deployed dynamic model to {config.DYNAMIC_MODEL_PATH}")
    print(f"  ({config.DYNAMIC_FOREST_TREES} trees, trained on the {args.split} split)")

    evaluation.save_confusion_matrix_plot(
        primary["confusion_matrix"],
        config.DYNAMIC_GESTURES,
        config.DYNAMIC_CONFUSION_MATRIX_PATH,
        f"{primary['name']} - Swipe Confusion Matrix ({args.split} split)",
        cmap="Oranges",
        xticks_rotation=30,
    )
    print(f"Saved confusion matrix plot to {config.DYNAMIC_CONFUSION_MATRIX_PATH}")

    with open(config.DYNAMIC_TRAINING_REPORT_PATH, "w") as f:
        f.write("Gesture-Controlled Presentation System - Dynamic Swipe Training Report\n")
        f.write(f"Dataset: {args.csv}  ({len(df)} clips recorded, {X.shape[1]} features)\n")
        f.write(f"Random Forest size: {config.DYNAMIC_FOREST_TREES} trees\n")
        f.write(f"Deployed model trained on the '{args.split}' split.\n\n")
        if config.DROP_DIRECTION_CONTRADICTING_CLIPS:
            f.write(
                f"CLEANING (applied before any split)\n"
                f"  {dropped} clip(s) dropped: net sideways travel contradicted their own label\n"
                f"    (the hand's return motion was captured inside the fixed recording window).\n"
                f"  {degenerate} swipe clip(s) dropped: |dx| < {config.MIN_TRAINING_SWIPE_DX}, too small to\n"
                f"    distinguish from a still hand.\n"
                f"  {len(y)} clips remain.\n\n"
            )
        if config.AUGMENT_DYNAMIC_TRAINING_DATA:
            f.write(
                f"AUGMENTATION (training split only)\n"
                f"  Mirroring (a left swipe is the mirror of a right swipe) and speed-scaling\n"
                f"  by {config.DYNAMIC_AUGMENT_SPEED_FACTORS} (the same motion performed faster/wider).\n"
                f"  Ratio features (straightness, direction_consistency) are scale-invariant and\n"
                f"  left untouched. Test clips are real recorded clips only -- never augmented.\n\n"
            )
        evaluation.write_split_explanation(
            f, config.DYNAMIC_GESTURES, config.DYNAMIC_ACTION_MAP
        )
        for split_name, results in all_results.items():
            train_idx, test_idx = splits[split_name]
            f.write(f"{'#' * 70}\n")
            f.write(
                f"# {split_name.upper()} SPLIT "
                f"({len(train_idx)} real train clips -> "
                f"{augmented_rows[split_name]} augmented rows / "
                f"{len(test_idx)} real test clips)\n"
            )
            f.write(f"{'#' * 70}\n\n")
            evaluation.write_results_sections(f, results, config.DYNAMIC_GESTURES)
    print(f"Saved full text report to {config.DYNAMIC_TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()
