"""Inter-annotator agreement metrics for semantic reliability (RQ1a,
Section 19).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


def cohens_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    return float(cohen_kappa_score(labels_a, labels_b))


def krippendorffs_alpha(rater_labels: list[list[str]], level: str = "nominal") -> float:
    """`rater_labels`: one list per rater, aligned by item index, with None
    for missing judgments. Wraps the `krippendorff` package's reliability
    computation over a value-by-unit matrix."""
    import krippendorff

    all_labels = sorted({l for rater in rater_labels for l in rater if l is not None})
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    matrix = np.full((len(rater_labels), len(rater_labels[0])), np.nan)
    for r, rater in enumerate(rater_labels):
        for i, l in enumerate(rater):
            if l is not None:
                matrix[r, i] = label_to_idx[l]
    try:
        return float(krippendorff.alpha(reliability_data=matrix, level_of_measurement=level))
    except ValueError:
        # Degenerate case (e.g. every item collapsed into the same cluster,
        # typical with a very small shared sample) - alpha is mathematically
        # undefined here, not computable-but-buggy. Match cohen_kappa_score's
        # existing behavior of returning NaN rather than crashing, so a tiny
        # sample surfaces as Gate B failing on NaN < threshold instead of an
        # unhandled traceback.
        return float("nan")


def macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def first_error_step_agreement(steps_a: list[int], steps_b: list[int], tolerance: int = 0) -> float:
    """Fraction of items where the two annotators' first-error step indices
    agree within `tolerance` steps."""
    assert len(steps_a) == len(steps_b)
    agree = sum(1 for a, b in zip(steps_a, steps_b) if abs(a - b) <= tolerance)
    return agree / len(steps_a) if steps_a else float("nan")


def embedding_stability(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    """Mean cosine similarity between independently produced descriptions of
    the same failures (raw-description stability, Section 19)."""
    num = (embeddings_a * embeddings_b).sum(axis=1)
    denom = np.linalg.norm(embeddings_a, axis=1) * np.linalg.norm(embeddings_b, axis=1) + 1e-8
    return float(np.mean(num / denom))
