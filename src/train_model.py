"""
Static pose model training (Section 8.4).

Loads data/gesture_data.csv, splits it, trains a Random Forest (the deployed
model per Section 8.4/10.2) plus KNN and SVM as comparison baselines, reports
accuracy / confusion matrix / per-class precision-recall for all three, and
saves the trained Random Forest as models/model.pkl for infer_realtime.py.

Every model is scored two ways, both of which matter:

  * on a TIME-BASED split (the honest figure, and the default) and on the
    conventional random split (reported for comparison -- it is optimistic
    here, see evaluation.temporal_split_indices);
  * by LABEL accuracy and by ACTION accuracy (see evaluation.action_accuracy).

Usage:
    python -m src.train_model
    python -m src.train_model --csv data/gesture_data.csv --test-size 0.2
    python -m src.train_model --split random
"""

from __future__ import annotations

import argparse
import sys
import time

import joblib
import pandas as pd

from . import config, dataset_io, evaluation

FEATURE_COLUMNS = [f"f{i}" for i in range(config.FEATURE_VECTOR_LENGTH)]


def load_dataset(csv_path):
    return dataset_io.load_training_csv(csv_path, FEATURE_COLUMNS, "collect_data.py")


def check_class_balance(df: pd.DataFrame) -> None:
    counts = df["gesture"].value_counts()
    print("Samples per gesture:")
    for gesture in config.STATIC_GESTURES:
        n = int(counts.get(gesture, 0))
        flag = "" if n >= config.min_samples_for(gesture) else "  <-- below recommended minimum"
        print(f"  {gesture:<10s} {n:5d}{flag}")

    missing_gestures = [g for g in config.STATIC_GESTURES if counts.get(g, 0) == 0]
    if missing_gestures:
        print(
            f"\nERROR: no samples at all for: {missing_gestures}. "
            "Collect data for every gesture before training (Section 8.3)."
        )
        sys.exit(1)

    present = counts[counts > 0]
    imbalance_ratio = present.max() / present.min()
    if imbalance_ratio > 2.0:
        print(
            f"\nWarning: class imbalance ratio is {imbalance_ratio:.1f}x "
            "(largest class / smallest class). Section 8.3 recommends a "
            "balanced dataset -- consider collecting more samples for the "
            "under-represented gesture(s)."
        )


def train_and_score(X, y, train_idx, test_idx, random_state: int) -> list[dict]:
    """Fit every candidate on one split and score it. Returns the results in
    candidate order, with the fitted model attached."""
    results = []
    for name, model in evaluation.build_candidates(
        config.STATIC_FOREST_TREES, random_state
    ).items():
        t0 = time.time()
        model.fit(X[train_idx], y[train_idx])
        train_seconds = time.time() - t0
        result = evaluation.evaluate_model(
            name,
            model,
            X[test_idx],
            y[test_idx],
            labels=config.STATIC_GESTURES,
            action_map=config.GESTURE_ACTION_MAP,
        )
        result["model"] = model
        result["train_seconds"] = train_seconds
        print(f"Training time: {train_seconds:.3f}s")
        results.append(result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=str(config.GESTURE_DATA_CSV))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--split",
        choices=("temporal", "random"),
        default=config.EVALUATION_SPLIT,
        help=(
            "Which split the DEPLOYED model is trained and reported on. "
            "'temporal' splits each class by capture time and is the honest "
            "choice on this data; 'random' is the conventional shuffle. Both "
            "figures are printed either way. Default: %(default)s"
        ),
    )
    args = parser.parse_args()

    print(f"Loading dataset from {args.csv}")
    X, y, timestamps, df = load_dataset(args.csv)
    print(f"Total samples: {len(df)}  |  Feature vector length: {X.shape[1]}")
    check_class_balance(df)

    splits = {
        "temporal": evaluation.temporal_split_indices(y, timestamps, args.test_size),
        "random": evaluation.random_split_indices(y, args.test_size, args.random_state),
    }

    all_results = {}
    for split_name, (train_idx, test_idx) in splits.items():
        primary = split_name == args.split
        print(
            f"\n{'=' * 74}\n"
            f"{split_name.upper()} SPLIT: {len(train_idx)} train / {len(test_idx)} test samples"
            f"{'   <-- deployed model trained on this split' if primary else ''}\n"
            f"{'=' * 74}"
        )
        if split_name == "temporal":
            print(
                "Each class split by capture time: earliest samples train, latest test.\n"
                "Near-duplicate frames therefore stay on one side of the split."
            )
        else:
            print(
                "Conventional stratified shuffle. Expect this to score higher than the\n"
                "temporal split -- near-duplicate frames land on both sides, so much of\n"
                "the test set is effectively already seen. Reported for comparison only."
            )
        all_results[split_name] = train_and_score(
            X, y, train_idx, test_idx, args.random_state
        )
        evaluation.print_summary(all_results[split_name], has_action_accuracy=True)

    print(f"\n{'=' * 74}\nHEADLINE COMPARISON (Random Forest)\n{'=' * 74}")
    print(f"  {'split':<12s} {'label acc':>10s} {'action acc':>12s}")
    for split_name, results in all_results.items():
        rf = next(r for r in results if r["name"].startswith("Random Forest"))
        print(f"  {split_name:<12s} {rf['accuracy']:>10.4f} {rf['action_accuracy']:>12.4f}")
    merged = evaluation.merged_label_groups(
        config.STATIC_GESTURES, config.GESTURE_ACTION_MAP
    )
    if merged:
        print(
            "  Action accuracy is higher because these labels trigger the same action,\n"
            "  so confusing them changes nothing the presenter can see: "
            + " | ".join(", ".join(sorted(g)) for g in merged)
        )

    # Random Forest is always the deployed model (Section 8.4/10.2), regardless
    # of how it ranks against the baselines above -- the comparison is for the
    # report's evaluation/discussion section, not model selection.
    primary = next(
        r for r in all_results[args.split] if r["name"].startswith("Random Forest")
    )

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(primary["model"], config.TRAINED_MODEL_PATH)
    print(f"\nSaved deployed model to {config.TRAINED_MODEL_PATH}")
    print(f"  ({config.STATIC_FOREST_TREES} trees, trained on the {args.split} split)")

    evaluation.save_confusion_matrix_plot(
        primary["confusion_matrix"],
        config.STATIC_GESTURES,
        config.CONFUSION_MATRIX_PATH,
        f"{primary['name']} - Confusion Matrix ({args.split} split)",
    )
    print(f"Saved confusion matrix plot to {config.CONFUSION_MATRIX_PATH}")

    with open(config.TRAINING_REPORT_PATH, "w") as f:
        f.write("Gesture-Controlled Presentation System - Static Pose Training Report\n")
        f.write(f"Dataset: {args.csv}  ({len(df)} samples, {X.shape[1]} features)\n")
        f.write(f"Random Forest size: {config.STATIC_FOREST_TREES} trees\n")
        f.write(f"Deployed model trained on the '{args.split}' split.\n\n")
        evaluation.write_split_explanation(
            f, config.STATIC_GESTURES, config.GESTURE_ACTION_MAP
        )
        for split_name, results in all_results.items():
            train_idx, test_idx = splits[split_name]
            f.write(f"{'#' * 70}\n")
            f.write(
                f"# {split_name.upper()} SPLIT "
                f"({len(train_idx)} train / {len(test_idx)} test)\n"
            )
            f.write(f"{'#' * 70}\n\n")
            evaluation.write_results_sections(f, results, config.STATIC_GESTURES)
    print(f"Saved full text report to {config.TRAINING_REPORT_PATH}")


if __name__ == "__main__":
    main()
