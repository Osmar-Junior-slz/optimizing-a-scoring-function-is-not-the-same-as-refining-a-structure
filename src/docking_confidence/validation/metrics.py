"""Probability and ranking diagnostics; unavailable evidence is None."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    auc, average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_auc_score, roc_curve,
)


def sigmoid_raw_score(raw: np.ndarray) -> np.ndarray:
    """Convert a logistic decision_function to its uncalibrated probability."""
    raw = np.asarray(raw, dtype=float).reshape(-1)
    if not np.isfinite(raw).all():
        raise ValueError("raw scores must be finite")
    result = np.empty_like(raw)
    positive = raw >= 0
    result[positive] = 1 / (1 + np.exp(-raw[positive]))
    exp_negative = np.exp(raw[~positive])
    result[~positive] = exp_negative / (1 + exp_negative)
    return result


def _validated(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y).reshape(-1)
    probability = np.asarray(p, dtype=float).reshape(-1)
    if len(labels) != len(probability):
        raise ValueError("labels and probabilities must have equal lengths")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("labels must be binary and nonmissing")
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("probabilities must be finite and in [0, 1]")
    return labels.astype(int), probability


def reliability_bins(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    y, p = _validated(y, p)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (p >= left) & ((p <= right) if index == bins - 1 else (p < right))
        count = int(mask.sum())
        rows.append({
            "bin": index, "left": float(left), "right": float(right), "count": count,
            "mean_probability": float(p[mask].mean()) if count else None,
            "observed_positive_fraction": float(y[mask].mean()) if count else None,
        })
    return rows


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    rows = reliability_bins(y, p, bins)
    total = sum(row["count"] for row in rows)
    if not total:
        return None
    return float(sum(
        row["count"] / total * abs(row["mean_probability"] - row["observed_positive_fraction"])
        for row in rows if row["count"]
    ))


def _ranking_scores(p: np.ndarray, ranking_scores: np.ndarray | None) -> np.ndarray:
    scores = p if ranking_scores is None else np.asarray(ranking_scores, dtype=float).reshape(-1)
    if len(scores) != len(p) or not np.isfinite(scores).all():
        raise ValueError("ranking scores must be finite and aligned with probabilities")
    return scores


def probability_metrics(y: np.ndarray, p: np.ndarray, *, ranking_scores: np.ndarray | None = None) -> dict:
    y, p = _validated(y, p)
    scores = _ranking_scores(p, ranking_scores)
    two_classes = len(np.unique(y)) == 2
    ranking_reason = None if two_classes else ("no_rows" if not len(y) else "single_class")
    precision, recall, _ = precision_recall_curve(y, scores) if two_classes else (None, None, None)
    # AP uses a recall-weighted step function, unlike trapezoidal PR-AUC.
    return {
        "n": int(len(y)), "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "positive_prevalence": float(y.mean()) if len(y) else None,
        "brier_score": float(brier_score_loss(y, p)) if len(y) else None,
        "log_loss": float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1])) if len(y) else None,
        "roc_auc": float(roc_auc_score(y, scores)) if two_classes else None,
        "pr_auc_trapezoidal": float(auc(recall, precision)) if two_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if two_classes else None,
        "expected_calibration_error_10bin": expected_calibration_error(y, p, 10),
        "ranking_unavailable_reason": ranking_reason,
    }


def probability_curves(y: np.ndarray, p: np.ndarray, *, ranking_scores: np.ndarray | None = None) -> dict:
    y, p = _validated(y, p)
    scores = _ranking_scores(p, ranking_scores)
    result = {"reliability_bins": reliability_bins(y, p), "roc": [], "precision_recall": []}
    if len(np.unique(y)) != 2:
        result["ranking_unavailable_reason"] = "no_rows" if not len(y) else "single_class"
        return result
    fpr, tpr, _ = roc_curve(y, scores, drop_intermediate=False)
    precision, recall, _ = precision_recall_curve(y, scores)
    result["roc"] = [{"false_positive_rate": float(x), "true_positive_rate": float(z)} for x, z in zip(fpr, tpr)]
    result["precision_recall"] = [{"recall": float(x), "precision": float(z)} for x, z in zip(recall, precision)]
    result["ranking_unavailable_reason"] = None
    return result
