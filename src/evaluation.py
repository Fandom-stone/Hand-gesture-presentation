"""
Shared evaluation helpers for both training scripts.

Three concerns live here, all of which train_model.py and
train_dynamic_model.py previously implemented separately (and identically):

  1. HONEST SPLITTING. A random train/test split is the wrong tool for this
     dataset -- see temporal_split_indices() for why, and what to do instead.
  2. ACTION-LEVEL accuracy. The system's job is to fire the right ACTION, not
     to name the right label; two labels that map to the same action are not
     a user-visible error when confused. See action_accuracy().
  3. Report formatting -- scoring a model, printing it, plotting the
     confusion matrix, writing the text report.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; we only write PNGs, never show a window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

NO_ACTION = "(none)"


def build_candidates(n_trees: int, random_state: int) -> dict:
    """The three algorithms both trainers compare.

    Random Forest is the deployed model either way (Section 8.4/10.2); KNN and
    SVM are trained alongside purely as baselines for the report's evaluation
    section. Defined once so the line-up can't drift between the two trainers
    -- only the forest size differs, and that is the argument.
    """
    return {
        "Random Forest (primary)": RandomForestClassifier(
            n_estimators=n_trees, random_state=random_state, n_jobs=-1
        ),
        "K-Nearest Neighbours (baseline)": KNeighborsClassifier(n_neighbors=5),
        # probability=True is intentionally omitted: it's deprecated as of
        # sklearn 1.9 and unnecessary here, since only .predict() is used for
        # these baselines' accuracy/confusion-matrix comparison.
        "SVM - RBF kernel (baseline)": SVC(kernel="rbf", random_state=random_state),
    }


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------
def temporal_split_indices(
    y: np.ndarray, timestamps: np.ndarray, test_size: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Split each class chronologically: earliest samples train, latest test.

    WHY NOT A RANDOM SPLIT? The collector captures while you hold 'c', so
    consecutive rows are frames a few tens of milliseconds apart -- almost
    the same image. A random split scatters those near-duplicates across
    both sides, so for most test samples an almost identical row sits in the
    training set. The model can score ~100% by recognising rows it has
    effectively already seen, which says nothing about how it will behave on
    a pose held tomorrow.

    Splitting by TIME instead puts each class's near-duplicates on the same
    side of the split. Only the single pair straddling each class's cut
    point stays adjacent, so the test set is genuinely (almost) unseen and
    the resulting accuracy is one worth quoting.

    Splitting is done per class so both sides keep every class, which a
    single global cut would not guarantee.
    """
    train_idx: list[int] = []
    test_idx: list[int] = []
    for label in np.unique(y):
        idx = np.flatnonzero(y == label)
        idx = idx[np.argsort(timestamps[idx], kind="stable")]
        cut = int(round(len(idx) * (1.0 - test_size)))
        # Guarantee at least one sample each side for tiny classes.
        cut = max(1, min(cut, len(idx) - 1)) if len(idx) > 1 else len(idx)
        train_idx.extend(idx[:cut])
        test_idx.extend(idx[cut:])
    return np.array(sorted(train_idx), dtype=int), np.array(sorted(test_idx), dtype=int)


def random_split_indices(
    y: np.ndarray, test_size: float = 0.2, random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """The conventional stratified random split, kept so the two figures can
    be reported side by side. On this dataset it is optimistic -- see
    temporal_split_indices()."""
    all_idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        all_idx, test_size=test_size, random_state=random_state, stratify=y
    )
    return np.sort(train_idx), np.sort(test_idx)


# --------------------------------------------------------------------------
# Action-level scoring
# --------------------------------------------------------------------------
def to_actions(labels: Sequence[str], action_map: dict) -> np.ndarray:
    """Map gesture labels to the action each one triggers. Labels with no
    mapped action all collapse to a single NO_ACTION value."""
    return np.array([action_map.get(label, NO_ACTION) for label in labels])


def action_accuracy(y_true, y_pred, action_map: dict) -> float:
    """Fraction of samples for which the system would take the RIGHT ACTION.

    Label accuracy penalises every confusion equally, but the user only ever
    notices a confusion that changes what the program does. Mixing up two
    labels that both trigger nothing (NEUTRAL and OPEN_PALM, say) produces
    exactly the same behaviour, so it is not an error the presenter can see.
    This is the metric that matches what the system is for.
    """
    return float(np.mean(to_actions(y_true, action_map) == to_actions(y_pred, action_map)))


def merged_label_groups(labels: Sequence[str], action_map: dict) -> list[list[str]]:
    """Groups of two or more labels that share an action, i.e. the confusions
    that action-level accuracy forgives. Reported so the difference between
    the two accuracy figures is never mysterious."""
    by_action: dict = {}
    for label in labels:
        by_action.setdefault(action_map.get(label, NO_ACTION), []).append(label)
    return [group for group in by_action.values() if len(group) > 1]


# --------------------------------------------------------------------------
# Scoring and reporting
# --------------------------------------------------------------------------
def evaluate_model(
    name: str,
    model,
    X_test,
    y_test,
    labels: Sequence[str],
    action_map: Optional[dict] = None,
) -> dict:
    """Score one fitted model and print its results."""
    y_pred = model.predict(X_test)
    result = {
        "name": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=list(labels)),
        "y_pred": y_pred,
    }
    if action_map is not None:
        result["action_accuracy"] = action_accuracy(y_test, y_pred, action_map)

    print(f"\n=== {name} ===")
    print(f"Test accuracy (label) : {result['accuracy']:.4f}")
    if action_map is not None:
        print(f"Test accuracy (action): {result['action_accuracy']:.4f}")
    print(result["report"])
    return result


def print_summary(results: Sequence[dict], has_action_accuracy: bool) -> None:
    print("\n=== Summary (test accuracy) ===")
    header = f"  {'model':<32s} {'label':>8s}"
    if has_action_accuracy:
        header += f" {'action':>8s}"
    print(header + f" {'train time':>12s}")
    for r in sorted(results, key=lambda r: -r["accuracy"]):
        line = f"  {r['name']:<32s} {r['accuracy']:>8.4f}"
        if has_action_accuracy:
            line += f" {r.get('action_accuracy', float('nan')):>8.4f}"
        print(line + f" {r['train_seconds']:>11.3f}s")


def save_confusion_matrix_plot(
    cm, labels, out_path, title: str, cmap: str = "Blues", xticks_rotation=45
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(labels)).plot(
        ax=ax, cmap=cmap, colorbar=False, xticks_rotation=xticks_rotation
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_results_sections(f, results: Sequence[dict], labels: Sequence[str]) -> None:
    """Append the per-model blocks that both reports share."""
    for r in results:
        f.write(f"=== {r['name']} ===\n")
        f.write(f"Accuracy (label) : {r['accuracy']:.4f}\n")
        if "action_accuracy" in r:
            f.write(f"Accuracy (action): {r['action_accuracy']:.4f}\n")
        f.write(f"Training time: {r['train_seconds']:.3f}s\n")
        f.write(r["report"] + "\n")
        f.write(f"Confusion matrix (rows=true, cols=pred), labels={list(labels)}:\n")
        f.write(str(r["confusion_matrix"]) + "\n\n")


def write_split_explanation(f, labels: Sequence[str], action_map: dict) -> None:
    """The methodology note that makes the two headline numbers readable."""
    f.write(
        "EVALUATION METHOD\n"
        "  Two accuracies are reported for every model.\n\n"
        "  Split: samples are split per class by TIME -- earliest 80% train, latest\n"
        "    20% test. A random split is optimistic here because the collector records\n"
        "    many near-identical consecutive frames while a key is held, and a random\n"
        "    split puts near-duplicates of the test rows into the training set. The\n"
        "    random-split figure is printed alongside for comparison.\n\n"
        "  Label accuracy:  did the classifier name the right gesture?\n"
        "  Action accuracy: would the system have taken the right action? Labels that\n"
        "    trigger the same action are not a user-visible error when confused.\n"
    )
    groups = merged_label_groups(labels, action_map)
    if groups:
        for group in groups:
            f.write(f"    Labels sharing one action: {', '.join(sorted(group))}\n")
    else:
        f.write("    (every label triggers a different action, so the two figures match)\n")
    f.write("\n")
