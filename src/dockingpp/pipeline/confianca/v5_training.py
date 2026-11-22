"""Nested training protocol for the preservation calibrator v5.

The implementation deliberately keeps the statistical model and the evidence
graph small and auditable.  Every fitted object is scoped to the training
complexes of its current fold.  FULL controls provide labels only; this module
accepts already paired rows whose predictors came exclusively from REDUCED.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from collections import defaultdict
from statistics import NormalDist, median
from typing import Any, Mapping, Sequence

import numpy as np

from .evidence_graph import EvidenceGraph
from .features import (
    PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES,
    PRESERVATION_MECHANISTIC_GRAPH_V5_PROFILE,
)


V5_MODEL_GRID: tuple[dict[str, Any], ...] = (
    {"type": "logistic_l2", "C": 0.1, "l1_ratio": 0.0},
    {"type": "logistic_l2", "C": 1.0, "l1_ratio": 0.0},
    {"type": "logistic_l2", "C": 10.0, "l1_ratio": 0.0},
    {"type": "logistic_elastic_net", "C": 0.1, "l1_ratio": 0.25},
    {"type": "logistic_elastic_net", "C": 0.1, "l1_ratio": 0.50},
    {"type": "logistic_elastic_net", "C": 0.1, "l1_ratio": 0.75},
    {"type": "logistic_elastic_net", "C": 1.0, "l1_ratio": 0.25},
    {"type": "logistic_elastic_net", "C": 1.0, "l1_ratio": 0.50},
    {"type": "logistic_elastic_net", "C": 1.0, "l1_ratio": 0.75},
)
V5_GRAPH_K_GRID = (3, 5)
V5_DOMAIN_QUANTILE_GRID = (0.80, 0.90, 0.95)
V5_DISAGREEMENT_GRID = (0.10, 0.20, 0.30, 0.40)
V5_PROBABILITY_GRID = (0.50, 0.60, 0.70, 0.80, 0.90)


def _finite_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _row_matrix(
    rows: Sequence[Mapping[str, Any]], feature_paths: Sequence[str]
) -> np.ndarray:
    matrix: list[list[float]] = []
    for row_index, row in enumerate(rows):
        features = row.get("features")
        if not isinstance(features, Mapping):
            raise ValueError(f"row {row_index} has no feature mapping")
        matrix.append(
            [
                _finite_number(
                    features.get(path), field=f"rows[{row_index}].features[{path!r}]"
                )
                for path in feature_paths
            ]
        )
    if not matrix:
        raise ValueError("v5 training requires at least one row")
    return np.asarray(matrix, dtype=float)


def _labels(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    values: list[float] = []
    for row_index, row in enumerate(rows):
        raw_value = row.get("label")
        if isinstance(raw_value, (bool, np.bool_)):
            raise ValueError(
                f"rows[{row_index}].label must be the numeric binary value 0 or 1"
            )
        value = _finite_number(raw_value, field=f"rows[{row_index}].label")
        if value not in (0.0, 1.0):
            raise ValueError("v5 preservation labels must be exactly binary (0 or 1)")
        values.append(value)
    return np.asarray(values, dtype=float)


def _complex_ids(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    values = [str(row.get("complex_id") or "").strip() for row in rows]
    if any(not value for value in values):
        raise ValueError("every v5 row requires complex_id")
    return values


def _complex_balanced_weights(complex_ids: Sequence[str]) -> np.ndarray:
    counts: dict[str, int] = defaultdict(int)
    for complex_id in complex_ids:
        counts[str(complex_id)] += 1
    n_rows = len(complex_ids)
    n_groups = len(counts)
    if n_rows < 1 or n_groups < 1:
        raise ValueError("cannot weight an empty v5 cohort")
    return np.asarray(
        [n_rows / float(n_groups * counts[str(value)]) for value in complex_ids],
        dtype=float,
    )


def _fit_standardizer(
    matrix: np.ndarray, complex_ids: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    weights = _complex_balanced_weights(complex_ids)
    weight_sum = float(np.sum(weights))
    means = np.sum(matrix * weights[:, None], axis=0) / weight_sum
    variance = np.sum(((matrix - means) ** 2) * weights[:, None], axis=0) / weight_sum
    raw_scales = np.sqrt(np.maximum(variance, 0.0))
    constant_indices = np.flatnonzero(raw_scales <= 1e-12).tolist()
    scales = np.where(raw_scales <= 1e-12, 1.0, raw_scales)
    return means, scales, {
        "method": "complex_balanced_weighted_mean_and_population_std",
        "constant_feature_indices": constant_indices,
        "clipping": False,
        "winsorization": False,
        "imputation": False,
    }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    negative_exp = np.exp(values[~positive])
    result[~positive] = negative_exp / (1.0 + negative_exp)
    return result


def _fit_logistic(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    model_spec: Mapping[str, Any],
    *,
    max_iterations: int = 4000,
    tolerance: float = 1e-9,
) -> tuple[np.ndarray, float, dict[str, Any]]:
    if matrix.ndim != 2 or matrix.shape[0] != labels.size:
        raise ValueError("v5 logistic matrix and labels must align")
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("v5 logistic sample weights must be positive")
    model_type = str(model_spec.get("type") or "")
    if model_type not in {"logistic_l2", "logistic_elastic_net"}:
        raise ValueError(f"unsupported v5 global model: {model_type!r}")
    c_value = _finite_number(model_spec.get("C"), field="model.C")
    if c_value <= 0.0:
        raise ValueError("model.C must be positive")
    l1_ratio = (
        0.0
        if model_type == "logistic_l2"
        else _finite_number(model_spec.get("l1_ratio"), field="model.l1_ratio")
    )
    if not 0.0 <= l1_ratio <= 1.0:
        raise ValueError("model.l1_ratio must be within [0, 1]")

    prevalence = float(np.sum(weights * labels) / weight_sum)
    prevalence = min(max(prevalence, 1e-6), 1.0 - 1e-6)
    intercept = math.log(prevalence / (1.0 - prevalence))
    coefficients = np.zeros(matrix.shape[1], dtype=float)
    # C follows the usual sum-loss convention while gradients below use mean loss.
    penalty = 1.0 / (c_value * weight_sum)
    l1_penalty = penalty * l1_ratio
    l2_penalty = penalty * (1.0 - l1_ratio)
    weighted_matrix = matrix * np.sqrt(weights[:, None] / weight_sum)
    spectral = float(np.linalg.norm(weighted_matrix, ord=2) ** 2)
    step = 1.0 / max(0.25 * spectral + l2_penalty, 1e-6)
    step = min(step, 1.0)
    converged = False
    iteration = 0
    for iteration in range(1, int(max_iterations) + 1):
        probabilities = _sigmoid(matrix @ coefficients + intercept)
        residual = weights * (probabilities - labels) / weight_sum
        gradient = matrix.T @ residual + l2_penalty * coefficients
        intercept_gradient = float(np.sum(residual))
        candidate = coefficients - step * gradient
        if l1_penalty > 0.0:
            shrinkage = step * l1_penalty
            candidate = np.sign(candidate) * np.maximum(
                np.abs(candidate) - shrinkage, 0.0
            )
        candidate_intercept = intercept - step * intercept_gradient
        delta = max(
            float(np.max(np.abs(candidate - coefficients))),
            abs(candidate_intercept - intercept),
        )
        coefficients = candidate
        intercept = float(candidate_intercept)
        if delta <= tolerance:
            converged = True
            break
    return coefficients, intercept, {
        "optimizer": "proximal_batch_gradient_descent",
        "iterations": int(iteration),
        "converged": bool(converged),
        "tolerance": float(tolerance),
        "step_size": float(step),
        "regularization_scale": "1/(C*sum_complex_balanced_weights)",
    }


def _predict_logistic(
    matrix: np.ndarray, coefficients: np.ndarray, intercept: float
) -> np.ndarray:
    return _sigmoid(matrix @ coefficients + float(intercept))


def _grouped_folds(
    rows: Sequence[Mapping[str, Any]], *, n_splits: int, seed: int
) -> list[dict[str, Any]]:
    ids = _complex_ids(rows)
    unique = sorted(set(ids))
    if len(unique) < 2:
        raise ValueError("grouped CV requires at least two complexes")
    split_count = min(max(2, int(n_splits)), len(unique))
    labels = _labels(rows)
    group_stats: dict[str, tuple[int, float]] = {}
    for complex_id in unique:
        indices = [index for index, value in enumerate(ids) if value == complex_id]
        group_stats[complex_id] = (len(indices), float(np.mean(labels[indices])))
    rng = np.random.default_rng(int(seed))
    random_tie = {complex_id: float(rng.random()) for complex_id in unique}
    ordered = sorted(
        unique,
        key=lambda value: (
            -group_stats[value][0],
            -abs(group_stats[value][1] - 0.5),
            random_tie[value],
            value,
        ),
    )
    fold_groups: list[list[str]] = [[] for _ in range(split_count)]
    fold_rows = [0] * split_count
    fold_positive = [0.0] * split_count
    for complex_id in ordered:
        size, rate = group_stats[complex_id]
        target = min(
            range(split_count),
            key=lambda index: (
                fold_rows[index],
                fold_positive[index],
                len(fold_groups[index]),
                index,
            ),
        )
        fold_groups[target].append(complex_id)
        fold_rows[target] += size
        fold_positive[target] += rate * size
    folds: list[dict[str, Any]] = []
    for fold_index, validation_ids in enumerate(fold_groups):
        validation_set = set(validation_ids)
        validation_indices = [
            index for index, complex_id in enumerate(ids) if complex_id in validation_set
        ]
        training_indices = [
            index for index, complex_id in enumerate(ids) if complex_id not in validation_set
        ]
        training_ids = sorted(set(ids[index] for index in training_indices))
        if set(training_ids) & validation_set:
            raise RuntimeError("complex leakage detected while creating v5 grouped folds")
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_indices": training_indices,
                "validation_indices": validation_indices,
                "train_complex_ids": training_ids,
                "validation_complex_ids": sorted(validation_set),
            }
        )
    return folds


def _quantile(values: Sequence[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), float(q)))


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-15, 1.0 - 1e-15)
    return float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))


def _brier(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(probabilities, dtype=float) - labels) ** 2))


def _auc(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    positive = int(np.sum(labels == 1))
    negative = int(np.sum(labels == 0))
    if positive == 0 or negative == 0:
        return None
    order = np.argsort(probabilities, kind="mergesort")
    ranks = np.empty(probabilities.size, dtype=float)
    position = 0
    while position < probabilities.size:
        end = position + 1
        while (
            end < probabilities.size
            and probabilities[order[end]] == probabilities[order[position]]
        ):
            end += 1
        average_rank = (position + 1 + end) / 2.0
        ranks[order[position:end]] = average_rank
        position = end
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positive * (positive + 1) / 2.0) / float(positive * negative)


def _ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> float:
    labels = np.asarray(labels, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    total = probabilities.size
    error = 0.0
    for index in range(int(bins)):
        lower = index / float(bins)
        upper = (index + 1) / float(bins)
        mask = (probabilities >= lower) & (
            probabilities <= upper if index == bins - 1 else probabilities < upper
        )
        if np.any(mask):
            error += float(np.sum(mask)) / total * abs(
                float(np.mean(probabilities[mask])) - float(np.mean(labels[mask]))
            )
    return float(error)


def _probability_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    return {
        "n": int(labels.size),
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": int(np.sum(labels == 0)),
        "brier_score": _brier(labels, probabilities),
        "log_loss": _log_loss(labels, probabilities),
        "roc_auc": _auc(labels, probabilities),
        "expected_calibration_error_10bin": _ece(labels, probabilities),
    }


def _wilson_upper(failures: int, trials: int, confidence_level: float) -> float:
    if trials <= 0:
        return 1.0
    z_value = NormalDist().inv_cdf(float(confidence_level))
    proportion = failures / float(trials)
    denominator = 1.0 + z_value**2 / trials
    centre = (proportion + z_value**2 / (2.0 * trials)) / denominator
    radius = z_value * math.sqrt(
        proportion * (1.0 - proportion) / trials + z_value**2 / (4.0 * trials**2)
    ) / denominator
    return float(min(1.0, centre + radius))


def _policy_metrics(
    labels: np.ndarray,
    accepted: np.ndarray,
    complex_ids: Sequence[str],
    *,
    risk_target: float,
    confidence_level: float,
    minimum_coverage: float,
    minimum_authorized_complexes: int,
) -> dict[str, Any]:
    accepted = np.asarray(accepted, dtype=bool)
    if accepted.size != labels.size or accepted.size != len(complex_ids):
        raise ValueError("v5 policy vectors must align")
    groups = sorted(set(str(value) for value in complex_ids))
    accepted_count = int(np.sum(accepted))
    false_count = int(np.sum(accepted & (labels == 0)))
    accepted_groups: list[str] = []
    adverse_groups: list[str] = []
    group_false_rates: list[float] = []
    for complex_id in groups:
        indices = np.asarray(
            [index for index, value in enumerate(complex_ids) if str(value) == complex_id],
            dtype=int,
        )
        group_accepted = accepted[indices]
        if not np.any(group_accepted):
            continue
        accepted_groups.append(complex_id)
        group_labels = labels[indices][group_accepted]
        failures = int(np.sum(group_labels == 0))
        group_false_rates.append(failures / float(group_labels.size))
        if failures > 0:
            adverse_groups.append(complex_id)
    empirical_run_risk = false_count / float(accepted_count) if accepted_count else None
    empirical_cluster_risk = (
        float(np.mean(group_false_rates)) if group_false_rates else None
    )
    complex_adverse_rate = (
        len(adverse_groups) / float(len(accepted_groups)) if accepted_groups else None
    )
    risk_upper = _wilson_upper(
        len(adverse_groups), len(accepted_groups), confidence_level
    )
    coverage = accepted_count / float(labels.size) if labels.size else 0.0
    effective_minimum_complexes = int(minimum_authorized_complexes)
    supported = bool(
        accepted_count > 0
        and empirical_run_risk is not None
        and empirical_run_risk <= risk_target
        and empirical_cluster_risk is not None
        and empirical_cluster_risk <= risk_target
        and risk_upper <= risk_target
        and coverage >= minimum_coverage
        and len(accepted_groups) >= effective_minimum_complexes
    )
    return {
        "supported": supported,
        "risk_target": float(risk_target),
        "confidence_level": float(confidence_level),
        "confidence_method": "one_sided_wilson_on_complexes_with_any_false_acceptance",
        "accepted_row_count": accepted_count,
        "false_acceptance_count": false_count,
        "empirical_run_false_acceptance_risk": empirical_run_risk,
        "empirical_complex_balanced_false_acceptance_risk": empirical_cluster_risk,
        "accepted_complex_count": len(accepted_groups),
        "adverse_accepted_complex_count": len(adverse_groups),
        "complex_adverse_rate": complex_adverse_rate,
        "risk_upper_bound": risk_upper,
        "row_coverage": coverage,
        "complex_coverage": len(accepted_groups) / float(len(groups)),
        "accepted_complex_ids": accepted_groups,
        "adverse_accepted_complex_ids": adverse_groups,
        "minimum_coverage": float(minimum_coverage),
        "requested_minimum_authorized_complexes": int(minimum_authorized_complexes),
        "effective_minimum_authorized_complexes": effective_minimum_complexes,
    }


def _selection_rank(
    metrics: Mapping[str, Any], *, brier_score: float, complexity_rank: int
) -> tuple[Any, ...]:
    basic_coverage = bool(
        float(metrics["row_coverage"]) >= float(metrics["minimum_coverage"])
        and int(metrics["accepted_complex_count"])
        >= int(metrics["effective_minimum_authorized_complexes"])
    )
    upper_violation = max(
        0.0, float(metrics["risk_upper_bound"]) - float(metrics["risk_target"])
    )
    return (
        int(bool(metrics["supported"])),
        int(basic_coverage),
        -upper_violation,
        float(metrics["row_coverage"]),
        -float(brier_score),
        -int(complexity_rank),
    )


def _fit_fold_components(
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str],
    model_spec: Mapping[str, Any],
    k: int,
) -> dict[str, Any]:
    training_ids = _complex_ids(training_rows)
    validation_ids = _complex_ids(validation_rows)
    if set(training_ids) & set(validation_ids):
        raise RuntimeError("v5 fold leakage: a complex appears in train and validation")
    train_matrix = _row_matrix(training_rows, feature_paths)
    validation_matrix = _row_matrix(validation_rows, feature_paths)
    train_labels = _labels(training_rows)
    means, scales, scaler_evidence = _fit_standardizer(train_matrix, training_ids)
    standardized_train = (train_matrix - means) / scales
    standardized_validation = (validation_matrix - means) / scales
    weights = _complex_balanced_weights(training_ids)
    coefficients, intercept, optimizer = _fit_logistic(
        standardized_train, train_labels, weights, model_spec
    )
    p_stat = _predict_logistic(
        standardized_validation, coefficients, intercept
    )
    available_k = int(k)
    maximum_k = len(set(training_ids)) - 1
    if available_k < 1 or available_k > maximum_k:
        raise ValueError(
            "v5 graph k is infeasible for this fold: "
            f"requested={available_k}, maximum={maximum_k}"
        )
    graph = EvidenceGraph.fit(
        standardized_train.tolist(),
        complex_ids=training_ids,
        labels=train_labels.astype(int).tolist(),
        feature_names=feature_paths,
        k=available_k,
        domain_quantile=0.95,
    )
    graph_outputs = [
        graph.evaluate(row.tolist(), p_stat=float(probability))
        for row, probability in zip(standardized_validation, p_stat)
    ]
    loco_kth = EvidenceGraph.leave_one_complex_out_kth_distances(
        graph.nodes, k=min(available_k, len(graph.nodes) - 1)
    )
    thresholds = {
        str(quantile): _quantile(loco_kth, quantile)
        for quantile in V5_DOMAIN_QUANTILE_GRID
    }
    return {
        "p_stat": p_stat,
        "p_graph": np.asarray([item.p_graph for item in graph_outputs], dtype=float),
        "kth_distance": np.asarray(
            [item.kth_distance for item in graph_outputs], dtype=float
        ),
        "nearest_distance": np.asarray(
            [item.nearest_distance for item in graph_outputs], dtype=float
        ),
        "effective_neighbor_count": np.asarray(
            [item.effective_neighbor_count for item in graph_outputs], dtype=float
        ),
        "domain_thresholds": thresholds,
        "train_complex_ids": sorted(set(training_ids)),
        "validation_complex_ids": sorted(set(validation_ids)),
        "graph_node_ids": [node.complex_id for node in graph.nodes],
        "scaler_fit_complex_ids": sorted(set(training_ids)),
        "model_fit_complex_ids": sorted(set(training_ids)),
        "threshold_fit_complex_ids": sorted(set(training_ids)),
        "graph_k": available_k,
        "scaler": scaler_evidence,
        "optimizer": optimizer,
    }


def _inner_oof_for_candidate(
    rows: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str],
    model_spec: Mapping[str, Any],
    k: int,
) -> dict[str, Any]:
    row_count = len(rows)
    p_stat = np.full(row_count, np.nan, dtype=float)
    p_graph = np.full(row_count, np.nan, dtype=float)
    kth_distance = np.full(row_count, np.nan, dtype=float)
    nearest_distance = np.full(row_count, np.nan, dtype=float)
    effective_neighbors = np.full(row_count, np.nan, dtype=float)
    domain_thresholds = {
        quantile: np.full(row_count, np.nan, dtype=float)
        for quantile in V5_DOMAIN_QUANTILE_GRID
    }
    fold_evidence: list[dict[str, Any]] = []
    for fold in folds:
        train_indices = list(fold["train_indices"])
        validation_indices = list(fold["validation_indices"])
        result = _fit_fold_components(
            [rows[index] for index in train_indices],
            [rows[index] for index in validation_indices],
            feature_paths=feature_paths,
            model_spec=model_spec,
            k=k,
        )
        p_stat[validation_indices] = result["p_stat"]
        p_graph[validation_indices] = result["p_graph"]
        kth_distance[validation_indices] = result["kth_distance"]
        nearest_distance[validation_indices] = result["nearest_distance"]
        effective_neighbors[validation_indices] = result[
            "effective_neighbor_count"
        ]
        for quantile in V5_DOMAIN_QUANTILE_GRID:
            domain_thresholds[quantile][validation_indices] = float(
                result["domain_thresholds"][str(quantile)]
            )
        fold_evidence.append(
            {
                "fold_index": int(fold["fold_index"]),
                "train_complex_ids": result["train_complex_ids"],
                "validation_complex_ids": result["validation_complex_ids"],
                "scaler_fit_complex_ids": result["scaler_fit_complex_ids"],
                "model_fit_complex_ids": result["model_fit_complex_ids"],
                "graph_node_ids": result["graph_node_ids"],
                "graph_k": int(result["graph_k"]),
                "optimizer_converged": bool(
                    result["optimizer"].get("converged") is True
                ),
                "threshold_fit_complex_ids": result["threshold_fit_complex_ids"],
                "domain_thresholds": result["domain_thresholds"],
            }
        )
    for name, values in (
        ("p_stat", p_stat),
        ("p_graph", p_graph),
        ("kth_distance", kth_distance),
    ):
        if np.any(~np.isfinite(values)):
            raise RuntimeError(f"inner grouped OOF did not cover every row for {name}")
    return {
        "p_stat": p_stat,
        "p_graph": p_graph,
        "kth_distance": kth_distance,
        "nearest_distance": nearest_distance,
        "effective_neighbor_count": effective_neighbors,
        "domain_thresholds": domain_thresholds,
        "folds": fold_evidence,
        "all_optimizers_converged": all(
            bool(fold["optimizer_converged"]) for fold in fold_evidence
        ),
    }


def _select_nested_configurations(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str],
    seed: int,
    n_splits: int,
    risk_target: float,
    confidence_level: float,
    minimum_coverage: float,
    minimum_authorized_complexes: int,
) -> dict[str, Any]:
    labels = _labels(rows)
    complex_ids = _complex_ids(rows)
    folds = _grouped_folds(rows, n_splits=n_splits, seed=seed)
    minimum_inner_training_complexes = min(
        len(set(str(value) for value in fold["train_complex_ids"]))
        for fold in folds
    )
    best_hybrid: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    best_stat: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    best_graph: tuple[tuple[Any, ...], dict[str, Any]] | None = None
    candidate_count = 0
    candidate_summaries: list[dict[str, Any]] = []
    rejected_candidates: list[dict[str, Any]] = []

    for model_index, model_spec in enumerate(V5_MODEL_GRID):
        for k_index, k in enumerate(V5_GRAPH_K_GRID):
            if k > minimum_inner_training_complexes - 1:
                continue
            oof = _inner_oof_for_candidate(
                rows,
                folds,
                feature_paths=feature_paths,
                model_spec=model_spec,
                k=k,
            )
            stat_brier = _brier(labels, oof["p_stat"])
            graph_brier = _brier(labels, oof["p_graph"])
            disagreement = np.abs(oof["p_stat"] - oof["p_graph"])

            # Statistical-only ablation: graph variables are never consulted.
            if k_index == 0:
                for minimum_probability in V5_PROBABILITY_GRID:
                    accepted = oof["p_stat"] >= minimum_probability
                    metrics = _policy_metrics(
                        labels,
                        accepted,
                        complex_ids,
                        risk_target=risk_target,
                        confidence_level=confidence_level,
                        minimum_coverage=minimum_coverage,
                        minimum_authorized_complexes=minimum_authorized_complexes,
                    )
                    metrics["supported"] = bool(
                        metrics["supported"] and oof["all_optimizers_converged"]
                    )
                    config = {
                        "ablation": "statistical_only",
                        "model": dict(model_spec),
                        "minimum_probability": float(minimum_probability),
                        "inner_policy_metrics": metrics,
                        "inner_probability_metrics": _probability_metrics(
                            labels, oof["p_stat"]
                        ),
                        "inner_fold_evidence": oof["folds"],
                        "inner_optimizer_converged": bool(
                            oof["all_optimizers_converged"]
                        ),
                    }
                    rank = _selection_rank(
                        metrics,
                        brier_score=stat_brier,
                        complexity_rank=model_index,
                    )
                    if best_stat is None or rank > best_stat[0]:
                        best_stat = (rank, config)

            # Graph-only ablation: p_stat and its fitted coefficients are ignored.
            if model_index == 0:
                for domain_quantile in V5_DOMAIN_QUANTILE_GRID:
                    domain_supported = oof["kth_distance"] <= oof[
                        "domain_thresholds"
                    ][domain_quantile]
                    for minimum_graph_probability in V5_PROBABILITY_GRID:
                        accepted = domain_supported & (
                            oof["p_graph"] >= minimum_graph_probability
                        )
                        metrics = _policy_metrics(
                            labels,
                            accepted,
                            complex_ids,
                            risk_target=risk_target,
                            confidence_level=confidence_level,
                            minimum_coverage=minimum_coverage,
                            minimum_authorized_complexes=minimum_authorized_complexes,
                        )
                        metrics["supported"] = bool(
                            metrics["supported"]
                            and oof["all_optimizers_converged"]
                        )
                        config = {
                            "ablation": "graph_only",
                            "k": int(k),
                            "domain_quantile": float(domain_quantile),
                            "minimum_graph_probability": float(
                                minimum_graph_probability
                            ),
                            "inner_policy_metrics": metrics,
                            "inner_probability_metrics": _probability_metrics(
                                labels, oof["p_graph"]
                            ),
                            "inner_fold_evidence": oof["folds"],
                            "inner_optimizer_converged": bool(
                                oof["all_optimizers_converged"]
                            ),
                        }
                        rank = _selection_rank(
                            metrics,
                            brier_score=graph_brier,
                            complexity_rank=k_index,
                        )
                        if best_graph is None or rank > best_graph[0]:
                            best_graph = (rank, config)

            for domain_quantile in V5_DOMAIN_QUANTILE_GRID:
                domain_supported = oof["kth_distance"] <= oof[
                    "domain_thresholds"
                ][domain_quantile]
                for max_disagreement in V5_DISAGREEMENT_GRID:
                    agreement_supported = disagreement <= max_disagreement
                    for minimum_probability in V5_PROBABILITY_GRID:
                        accepted = (
                            domain_supported
                            & agreement_supported
                            & (oof["p_stat"] >= minimum_probability)
                        )
                        metrics = _policy_metrics(
                            labels,
                            accepted,
                            complex_ids,
                            risk_target=risk_target,
                            confidence_level=confidence_level,
                            minimum_coverage=minimum_coverage,
                            minimum_authorized_complexes=minimum_authorized_complexes,
                        )
                        metrics["supported"] = bool(
                            metrics["supported"]
                            and oof["all_optimizers_converged"]
                        )
                        candidate_count += 1
                        config = {
                            "ablation": "hybrid",
                            "model": dict(model_spec),
                            "k": int(k),
                            "domain_quantile": float(domain_quantile),
                            "max_graph_disagreement": float(max_disagreement),
                            "minimum_probability": float(minimum_probability),
                            "inner_policy_metrics": metrics,
                            "inner_statistical_metrics": _probability_metrics(
                                labels, oof["p_stat"]
                            ),
                            "inner_graph_metrics": _probability_metrics(
                                labels, oof["p_graph"]
                            ),
                            "inner_fold_evidence": oof["folds"],
                            "inner_optimizer_converged": bool(
                                oof["all_optimizers_converged"]
                            ),
                        }
                        rank = _selection_rank(
                            metrics,
                            brier_score=stat_brier,
                            complexity_rank=model_index + k_index,
                        )
                        if best_hybrid is None or rank > best_hybrid[0]:
                            best_hybrid = (rank, config)
            candidate_summaries.append(
                {
                    "model": dict(model_spec),
                    "k": int(k),
                    "statistical_metrics": _probability_metrics(
                        labels, oof["p_stat"]
                    ),
                    "graph_metrics": _probability_metrics(labels, oof["p_graph"]),
                }
            )
    if best_hybrid is None or best_stat is None or best_graph is None:
        raise RuntimeError("v5 inner selection did not produce all ablation candidates")
    return {
        "hybrid": best_hybrid[1],
        "statistical_only": best_stat[1],
        "graph_only": best_graph[1],
        "candidate_count": int(candidate_count),
        "grid": {
            "models": [dict(value) for value in V5_MODEL_GRID],
            "graph_k": list(V5_GRAPH_K_GRID),
            "domain_quantiles": list(V5_DOMAIN_QUANTILE_GRID),
            "max_graph_disagreement": list(V5_DISAGREEMENT_GRID),
            "probability_thresholds": list(V5_PROBABILITY_GRID),
        },
        "folds": list(best_hybrid[1]["inner_fold_evidence"]),
        "minimum_inner_training_complexes": minimum_inner_training_complexes,
        "candidate_probability_summaries": candidate_summaries,
        "rejected_candidates": rejected_candidates,
    }


def _fit_graph(
    standardized_train: np.ndarray,
    training_rows: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str],
    k: int,
    domain_quantile: float,
) -> EvidenceGraph:
    training_ids = _complex_ids(training_rows)
    resolved_k = int(k)
    maximum_k = len(set(training_ids)) - 1
    if resolved_k < 1 or resolved_k > maximum_k:
        raise ValueError(
            "v5 graph k is infeasible for the fitted cohort: "
            f"requested={resolved_k}, maximum={maximum_k}"
        )
    return EvidenceGraph.fit(
        standardized_train.tolist(),
        complex_ids=training_ids,
        labels=_labels(training_rows).astype(int).tolist(),
        feature_names=feature_paths,
        k=resolved_k,
        domain_quantile=float(domain_quantile),
    )


def _fit_outer_and_predict(
    training_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str],
    selected: Mapping[str, Any],
) -> dict[str, Any]:
    training_ids = _complex_ids(training_rows)
    validation_ids = _complex_ids(validation_rows)
    if set(training_ids) & set(validation_ids):
        raise RuntimeError("outer LOCO leakage detected before v5 final-fold fit")
    train_matrix = _row_matrix(training_rows, feature_paths)
    validation_matrix = _row_matrix(validation_rows, feature_paths)
    train_labels = _labels(training_rows)
    means, scales, scaler_evidence = _fit_standardizer(train_matrix, training_ids)
    standardized_train = (train_matrix - means) / scales
    standardized_validation = (validation_matrix - means) / scales
    weights = _complex_balanced_weights(training_ids)

    hybrid_config = selected["hybrid"]
    hybrid_coefficients, hybrid_intercept, hybrid_optimizer = _fit_logistic(
        standardized_train,
        train_labels,
        weights,
        hybrid_config["model"],
    )
    hybrid_p_stat = _predict_logistic(
        standardized_validation, hybrid_coefficients, hybrid_intercept
    )
    hybrid_graph = _fit_graph(
        standardized_train,
        training_rows,
        feature_paths=feature_paths,
        k=int(hybrid_config["k"]),
        domain_quantile=float(hybrid_config["domain_quantile"]),
    )
    hybrid_graph_evidence = [
        hybrid_graph.evaluate(vector.tolist(), p_stat=float(probability))
        for vector, probability in zip(standardized_validation, hybrid_p_stat)
    ]
    hybrid_p_graph = np.asarray(
        [item.p_graph for item in hybrid_graph_evidence], dtype=float
    )
    hybrid_domain = np.asarray(
        [item.domain_supported for item in hybrid_graph_evidence], dtype=bool
    )
    hybrid_disagreement = np.abs(hybrid_p_stat - hybrid_p_graph)
    hybrid_accepted = (
        hybrid_domain
        & (
            hybrid_disagreement
            <= float(hybrid_config["max_graph_disagreement"])
        )
        & (hybrid_p_stat >= float(hybrid_config["minimum_probability"]))
    )

    stat_config = selected["statistical_only"]
    stat_coefficients, stat_intercept, stat_optimizer = _fit_logistic(
        standardized_train,
        train_labels,
        weights,
        stat_config["model"],
    )
    stat_probability = _predict_logistic(
        standardized_validation, stat_coefficients, stat_intercept
    )
    stat_accepted = stat_probability >= float(stat_config["minimum_probability"])

    graph_config = selected["graph_only"]
    graph_only = _fit_graph(
        standardized_train,
        training_rows,
        feature_paths=feature_paths,
        k=int(graph_config["k"]),
        domain_quantile=float(graph_config["domain_quantile"]),
    )
    graph_only_evidence = [
        graph_only.evaluate(vector.tolist()) for vector in standardized_validation
    ]
    graph_probability = np.asarray(
        [item.p_graph for item in graph_only_evidence], dtype=float
    )
    graph_accepted = np.asarray(
        [item.domain_supported for item in graph_only_evidence], dtype=bool
    ) & (
        graph_probability >= float(graph_config["minimum_graph_probability"])
    )

    prevalence = float(np.sum(weights * train_labels) / np.sum(weights))
    return {
        "hybrid_p_stat": hybrid_p_stat,
        "hybrid_p_graph": hybrid_p_graph,
        "hybrid_graph_evidence": hybrid_graph_evidence,
        "hybrid_accepted": hybrid_accepted,
        "stat_probability": stat_probability,
        "stat_accepted": stat_accepted,
        "graph_probability": graph_probability,
        "graph_only_evidence": graph_only_evidence,
        "graph_accepted": graph_accepted,
        "baseline_probability": np.full(len(validation_rows), prevalence, dtype=float),
        "fit_evidence": {
            "train_complex_ids": sorted(set(training_ids)),
            "validation_complex_ids": sorted(set(validation_ids)),
            "scaler_fit_complex_ids": sorted(set(training_ids)),
            "hybrid_model_fit_complex_ids": sorted(set(training_ids)),
            "hybrid_graph_node_ids": [
                node.complex_id for node in hybrid_graph.nodes
            ],
            "hybrid_graph_k": int(hybrid_graph.k),
            "hybrid_threshold_fit_complex_ids": sorted(set(training_ids)),
            "statistical_model_fit_complex_ids": sorted(set(training_ids)),
            "graph_only_node_ids": [node.complex_id for node in graph_only.nodes],
            "graph_only_k": int(graph_only.k),
            "graph_only_threshold_fit_complex_ids": sorted(set(training_ids)),
            "hybrid_domain_threshold": float(hybrid_graph.domain_threshold),
            "graph_only_domain_threshold": float(graph_only.domain_threshold),
            "scaler": scaler_evidence,
            "hybrid_optimizer": hybrid_optimizer,
            "statistical_optimizer": stat_optimizer,
            "hybrid_optimizer_converged": bool(
                hybrid_optimizer.get("converged") is True
            ),
            "statistical_optimizer_converged": bool(
                stat_optimizer.get("converged") is True
            ),
        },
    }


def _cluster_bootstrap_risk_upper(
    labels: np.ndarray,
    accepted: np.ndarray,
    complex_ids: Sequence[str],
    *,
    confidence_level: float,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    groups = sorted(set(str(value) for value in complex_ids))
    accepted_groups = [
        complex_id
        for complex_id in groups
        if any(
            bool(accepted[index])
            for index, value in enumerate(complex_ids)
            if str(value) == complex_id
        )
    ]
    if not accepted_groups:
        return {
            "method": "cluster_bootstrap_by_complex",
            "replicates": int(replicates),
            "seed": int(seed),
            "upper_bound": 1.0,
            "valid_replicates": 0,
        }
    group_counts: dict[str, tuple[int, int]] = {}
    for complex_id in accepted_groups:
        indices = [
            index
            for index, value in enumerate(complex_ids)
            if str(value) == complex_id and bool(accepted[index])
        ]
        group_counts[complex_id] = (
            sum(int(labels[index] == 0) for index in indices),
            len(indices),
        )
    rng = np.random.default_rng(int(seed))
    risks = np.empty(int(replicates), dtype=float)
    for replicate in range(int(replicates)):
        sampled = rng.choice(accepted_groups, size=len(accepted_groups), replace=True)
        false_count = sum(group_counts[str(value)][0] for value in sampled)
        accepted_count = sum(group_counts[str(value)][1] for value in sampled)
        risks[replicate] = (
            false_count / float(accepted_count) if accepted_count else 1.0
        )
    return {
        "method": "cluster_bootstrap_by_complex",
        "replicates": int(replicates),
        "seed": int(seed),
        "upper_bound": float(np.quantile(risks, confidence_level)),
        "median": float(np.median(risks)),
        "valid_replicates": int(replicates),
    }


def _runtime_metrics(
    rows: Sequence[Mapping[str, Any]], accepted: np.ndarray
) -> dict[str, Any]:
    reduced_values: list[float] = []
    full_values: list[float] = []
    policy_values: list[float] = []
    fallback_flags: list[bool] = []
    for row, accept in zip(rows, accepted):
        reduced = row.get("reduced_runtime_sec")
        full = row.get("full_runtime_sec")
        try:
            reduced_value = float(reduced)
            full_value = float(full)
        except (TypeError, ValueError):
            continue
        if (
            not math.isfinite(reduced_value)
            or not math.isfinite(full_value)
            or reduced_value < 0.0
            or full_value < 0.0
        ):
            continue
        reduced_values.append(reduced_value)
        full_values.append(full_value)
        policy_values.append(reduced_value + (0.0 if bool(accept) else full_value))
        fallback_flags.append(not bool(accept))
    if not policy_values:
        return {
            "available": False,
            "paired_runtime_count": 0,
            "reason": "paired REDUCED/FULL runtime was not available in source summaries",
        }
    policy_runtime = float(sum(policy_values))
    always_full_runtime = float(sum(full_values))
    always_reduced_runtime = float(sum(reduced_values))
    fallback_count = int(sum(fallback_flags))
    speedup = always_full_runtime / policy_runtime if policy_runtime > 0.0 else None
    return {
        "available": True,
        "paired_runtime_count": len(policy_values),
        "policy_runtime_sec": policy_runtime,
        "always_full_runtime_sec": always_full_runtime,
        "always_reduced_runtime_sec": always_reduced_runtime,
        "fallback_count": fallback_count,
        "fallback_rate": fallback_count / float(len(policy_values)),
        "net_time_saved_sec": always_full_runtime - policy_runtime,
        "net_speedup": speedup,
        "policy_vs_always_full_ratio": (
            policy_runtime / always_full_runtime if always_full_runtime > 0.0 else None
        ),
        "speedup_vs_always_full": speedup,
    }


def _fit_final_components(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str],
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    complex_ids = _complex_ids(rows)
    matrix = _row_matrix(rows, feature_paths)
    labels = _labels(rows)
    means, scales, scaler_evidence = _fit_standardizer(matrix, complex_ids)
    standardized = (matrix - means) / scales
    weights = _complex_balanced_weights(complex_ids)
    coefficients, intercept, optimizer = _fit_logistic(
        standardized,
        labels,
        weights,
        configuration["model"],
    )
    graph = _fit_graph(
        standardized,
        rows,
        feature_paths=feature_paths,
        k=int(configuration["k"]),
        domain_quantile=float(configuration["domain_quantile"]),
    )
    return {
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
        "intercept": float(intercept),
        "graph": graph,
        "scaler_evidence": scaler_evidence,
        "optimizer": optimizer,
    }


def _cohort_sha256(
    rows: Sequence[Mapping[str, Any]], feature_paths: Sequence[str]
) -> str:
    payload = [
        {
            "complex_id": str(row.get("complex_id") or ""),
            "run_id": str(row.get("run_id") or ""),
            "trial_key_sha256": row.get("trial_key_sha256"),
            "label": int(row["label"]),
            "features": {
                path: float(row["features"][path]) for path in feature_paths
            },
        }
        for row in sorted(
            rows,
            key=lambda item: (
                str(item.get("complex_id") or ""),
                str(item.get("run_id") or ""),
                str(item.get("trial_key_sha256") or ""),
            ),
        )
    ]
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _leakage_audit(outer_folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    violations: list[dict[str, Any]] = []
    for fold in outer_folds:
        validation = set(fold["outer_validation_complex_ids"])
        role_fields = (
            "outer_train_complex_ids",
            "scaler_fit_complex_ids",
            "hybrid_model_fit_complex_ids",
            "hybrid_graph_node_ids",
            "hybrid_threshold_fit_complex_ids",
            "statistical_model_fit_complex_ids",
            "graph_only_node_ids",
            "graph_only_threshold_fit_complex_ids",
        )
        for field in role_fields:
            overlap = sorted(validation & set(fold.get(field) or []))
            if overlap:
                violations.append(
                    {
                        "outer_complex_id": fold["outer_complex_id"],
                        "role": field,
                        "overlap": overlap,
                    }
                )
        for inner in fold.get("inner_folds") or []:
            inner_validation = set(inner.get("validation_complex_ids") or [])
            inner_roles = (
                "train_complex_ids",
                "scaler_fit_complex_ids",
                "model_fit_complex_ids",
                "graph_node_ids",
                "threshold_fit_complex_ids",
            )
            for inner_role in inner_roles:
                overlap = sorted(
                    inner_validation & set(inner.get(inner_role) or [])
                )
                if overlap:
                    violations.append(
                        {
                            "outer_complex_id": fold["outer_complex_id"],
                            "inner_fold_index": inner.get("fold_index"),
                            "role": f"inner_{inner_role}",
                            "overlap": overlap,
                        }
                    )
            leaked_outer = sorted(
                validation
                & (
                    set(inner.get("train_complex_ids") or [])
                    | set(inner.get("validation_complex_ids") or [])
                )
            )
            if leaked_outer:
                violations.append(
                    {
                        "outer_complex_id": fold["outer_complex_id"],
                        "role": "outer_complex_present_in_inner_selection",
                        "overlap": leaked_outer,
                    }
                )
    return {
        "passed": not violations,
        "grouping_unit": "complex_id",
        "checked_roles": [
            "split",
            "scaler",
            "global_model",
            "evidence_graph",
            "domain_threshold",
            "probability_threshold",
            "disagreement_threshold",
        ],
        "violation_count": len(violations),
        "violations": violations,
    }


def train_v5_artifact(
    rows: Sequence[Mapping[str, Any]],
    *,
    feature_paths: Sequence[str] = PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES,
    feature_schema_version: str = "1.3",
    seed: int = 2026,
    inner_splits: int = 5,
    risk_target: float = 0.20,
    confidence_level: float = 0.95,
    minimum_coverage: float = 0.10,
    minimum_authorized_complexes: int = 14,
    cluster_bootstrap_replicates: int = 20_000,
    cluster_bootstrap_seed: int = 20_260_829,
    dockq_threshold: float = 0.23,
    max_dockq_loss: float = 0.05,
    dataset_sha256: str | None = None,
    data_selection: Mapping[str, Any] | None = None,
    training_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit v5 with outer LOCO and return an observe-only frozen artifact.

    ``decision_capable`` is intentionally always false here.  A separate,
    untouched external confirmation step is required before activation.
    """

    feature_paths = tuple(str(value) for value in feature_paths)
    if feature_paths != PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES:
        raise ValueError("v5 requires the frozen five-feature mechanistic profile")
    if not 0.0 < float(risk_target) < 1.0:
        raise ValueError("v5 risk_target must be between 0 and 1")
    if not math.isclose(float(risk_target), 0.20, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("v5 frozen preservation protocol requires risk_target=0.20")
    if not 0.5 < float(confidence_level) < 1.0:
        raise ValueError("v5 confidence_level must be between 0.5 and 1")
    if not 0.0 <= float(minimum_coverage) <= 1.0:
        raise ValueError("v5 minimum_coverage must be within [0, 1]")
    if int(minimum_authorized_complexes) < 1:
        raise ValueError("v5 minimum_authorized_complexes must be positive")
    if int(cluster_bootstrap_replicates) < 100:
        raise ValueError("v5 cluster bootstrap requires at least 100 replicates")
    rows = list(rows)
    labels = _labels(rows)
    complex_ids = _complex_ids(rows)
    complexes = sorted(set(complex_ids))
    if len(complexes) < 3:
        raise ValueError("v5 outer LOCO requires at least three independent complexes")
    _row_matrix(rows, feature_paths)

    row_count = len(rows)
    outer_p_stat = np.full(row_count, np.nan, dtype=float)
    outer_p_graph = np.full(row_count, np.nan, dtype=float)
    outer_baseline = np.full(row_count, np.nan, dtype=float)
    outer_hybrid_accept = np.zeros(row_count, dtype=bool)
    outer_stat_probability = np.full(row_count, np.nan, dtype=float)
    outer_stat_accept = np.zeros(row_count, dtype=bool)
    outer_graph_probability = np.full(row_count, np.nan, dtype=float)
    outer_graph_accept = np.zeros(row_count, dtype=bool)
    outer_graph_evidence: list[Any] = [None] * row_count
    outer_folds: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any] | None] = [None] * row_count

    for outer_index, held_complex in enumerate(complexes):
        validation_indices = [
            index for index, value in enumerate(complex_ids) if value == held_complex
        ]
        training_indices = [
            index for index, value in enumerate(complex_ids) if value != held_complex
        ]
        outer_training_rows = [rows[index] for index in training_indices]
        outer_validation_rows = [rows[index] for index in validation_indices]
        selection = _select_nested_configurations(
            outer_training_rows,
            feature_paths=feature_paths,
            seed=int(seed) + 10_007 * (outer_index + 1),
            n_splits=inner_splits,
            risk_target=float(risk_target),
            confidence_level=float(confidence_level),
            minimum_coverage=float(minimum_coverage),
            minimum_authorized_complexes=int(minimum_authorized_complexes),
        )
        prediction = _fit_outer_and_predict(
            outer_training_rows,
            outer_validation_rows,
            feature_paths=feature_paths,
            selected=selection,
        )
        outer_p_stat[validation_indices] = prediction["hybrid_p_stat"]
        outer_p_graph[validation_indices] = prediction["hybrid_p_graph"]
        outer_baseline[validation_indices] = prediction["baseline_probability"]
        outer_hybrid_accept[validation_indices] = prediction["hybrid_accepted"]
        outer_stat_probability[validation_indices] = prediction["stat_probability"]
        outer_stat_accept[validation_indices] = prediction["stat_accepted"]
        outer_graph_probability[validation_indices] = prediction["graph_probability"]
        outer_graph_accept[validation_indices] = prediction["graph_accepted"]
        for local_index, row_index in enumerate(validation_indices):
            graph_item = prediction["hybrid_graph_evidence"][local_index]
            outer_graph_evidence[row_index] = graph_item
            reasons: list[str] = []
            if not graph_item.domain_supported:
                reasons.append("OUT_OF_DOMAIN")
            if float(graph_item.disagreement or 0.0) > float(
                selection["hybrid"]["max_graph_disagreement"]
            ):
                reasons.append("GRAPH_MODEL_DISAGREEMENT")
            if float(prediction["hybrid_p_stat"][local_index]) < float(
                selection["hybrid"]["minimum_probability"]
            ):
                reasons.append("LOW_STATISTICAL_CONFIDENCE")
            if not reasons:
                reasons.append("ACCEPTED")
            source_row = rows[row_index]
            prediction_rows[row_index] = {
                "row_index": int(row_index),
                "complex_id": held_complex,
                "run_id": str(source_row.get("run_id") or ""),
                "trial_key_sha256": source_row.get("trial_key_sha256"),
                "label": int(source_row["label"]),
                "p_stat": float(prediction["hybrid_p_stat"][local_index]),
                "p_graph": float(prediction["hybrid_p_graph"][local_index]),
                "graph": graph_item.to_dict(),
                "hybrid_accepted": bool(
                    prediction["hybrid_accepted"][local_index]
                ),
                "statistical_only_probability": float(
                    prediction["stat_probability"][local_index]
                ),
                "statistical_only_accepted": bool(
                    prediction["stat_accepted"][local_index]
                ),
                "graph_only_probability": float(
                    prediction["graph_probability"][local_index]
                ),
                "graph_only_accepted": bool(
                    prediction["graph_accepted"][local_index]
                ),
                "reason_codes": reasons,
                "fold_selected_policy": {
                    key: value
                    for key, value in selection["hybrid"].items()
                    if not key.startswith("inner_")
                },
            }
        outer_folds.append(
            {
                "outer_fold_index": int(outer_index),
                "outer_complex_id": held_complex,
                "outer_train_complex_ids": sorted(
                    set(complex_ids[index] for index in training_indices)
                ),
                "outer_validation_complex_ids": [held_complex],
                **prediction["fit_evidence"],
                "selected_hybrid": {
                    key: value
                    for key, value in selection["hybrid"].items()
                    if key != "inner_fold_evidence"
                },
                "selected_statistical_only": {
                    key: value
                    for key, value in selection["statistical_only"].items()
                    if key != "inner_fold_evidence"
                },
                "selected_graph_only": {
                    key: value
                    for key, value in selection["graph_only"].items()
                    if key != "inner_fold_evidence"
                },
                "inner_candidate_count": selection["candidate_count"],
                "inner_folds": selection["folds"],
            }
        )

    if any(
        np.any(~np.isfinite(values))
        for values in (
            outer_p_stat,
            outer_p_graph,
            outer_baseline,
            outer_stat_probability,
            outer_graph_probability,
        )
    ):
        raise RuntimeError("outer LOCO did not produce exactly one prediction per v5 row")
    leakage = _leakage_audit(outer_folds)

    hybrid_policy = _policy_metrics(
        labels,
        outer_hybrid_accept,
        complex_ids,
        risk_target=float(risk_target),
        confidence_level=float(confidence_level),
        minimum_coverage=float(minimum_coverage),
        minimum_authorized_complexes=int(minimum_authorized_complexes),
    )
    hybrid_bootstrap = _cluster_bootstrap_risk_upper(
        labels,
        outer_hybrid_accept,
        complex_ids,
        confidence_level=float(confidence_level),
        replicates=int(cluster_bootstrap_replicates),
        seed=int(cluster_bootstrap_seed),
    )
    hybrid_policy["cluster_bootstrap"] = hybrid_bootstrap
    hybrid_policy["supported"] = bool(
        hybrid_policy["supported"]
        and float(hybrid_bootstrap["upper_bound"]) <= float(risk_target)
        and leakage["passed"]
    )

    def ablation(
        name: str, probabilities: np.ndarray, accepted: np.ndarray
    ) -> dict[str, Any]:
        policy = _policy_metrics(
            labels,
            accepted,
            complex_ids,
            risk_target=float(risk_target),
            confidence_level=float(confidence_level),
            minimum_coverage=float(minimum_coverage),
            minimum_authorized_complexes=int(minimum_authorized_complexes),
        )
        bootstrap = _cluster_bootstrap_risk_upper(
            labels,
            accepted,
            complex_ids,
            confidence_level=float(confidence_level),
            replicates=int(cluster_bootstrap_replicates),
            seed=int(cluster_bootstrap_seed) + len(name),
        )
        policy["cluster_bootstrap"] = bootstrap
        policy["supported"] = bool(
            policy["supported"]
            and float(bootstrap["upper_bound"]) <= float(risk_target)
            and leakage["passed"]
        )
        return {
            "name": name,
            "outer_loco_probability_metrics": _probability_metrics(
                labels, probabilities
            ),
            "outer_loco_policy_metrics": policy,
            "runtime": _runtime_metrics(rows, accepted),
        }

    ablation_report = {
        "A_statistical_only": ablation(
            "A_statistical_only", outer_stat_probability, outer_stat_accept
        ),
        "B_graph_only": ablation(
            "B_graph_only", outer_graph_probability, outer_graph_accept
        ),
        "C_hybrid": {
            **ablation("C_hybrid", outer_p_stat, outer_hybrid_accept),
            "outer_loco_graph_probability_metrics": _probability_metrics(
                labels, outer_p_graph
            ),
        },
    }

    statistical_metrics = _probability_metrics(labels, outer_p_stat)
    graph_probability_metrics = _probability_metrics(labels, outer_p_graph)
    baseline_metrics = _probability_metrics(labels, outer_baseline)
    calibration_gate = bool(
        statistical_metrics["brier_score"] <= baseline_metrics["brier_score"]
        and statistical_metrics["log_loss"] <= baseline_metrics["log_loss"]
    )
    calibration_reasons: list[str] = []
    if statistical_metrics["brier_score"] > baseline_metrics["brier_score"]:
        calibration_reasons.append("MODEL_WORSE_THAN_GROUPED_PREVALENCE_BRIER")
    if statistical_metrics["log_loss"] > baseline_metrics["log_loss"]:
        calibration_reasons.append("MODEL_WORSE_THAN_GROUPED_PREVALENCE_LOG_LOSS")
    if not leakage["passed"]:
        calibration_reasons.append("GROUP_LEAKAGE_DETECTED")

    final_selection = _select_nested_configurations(
        rows,
        feature_paths=feature_paths,
        seed=int(seed) + 999_983,
        n_splits=inner_splits,
        risk_target=float(risk_target),
        confidence_level=float(confidence_level),
        minimum_coverage=float(minimum_coverage),
        minimum_authorized_complexes=int(minimum_authorized_complexes),
    )
    final_config = final_selection["hybrid"]
    final_fit = _fit_final_components(
        rows,
        feature_paths=feature_paths,
        configuration=final_config,
    )
    optimizer_gate = bool(
        final_fit["optimizer"].get("converged") is True
        and final_config.get("inner_optimizer_converged") is True
        and all(
            fold.get("hybrid_optimizer_converged") is True
            and (fold.get("selected_hybrid") or {}).get(
                "inner_optimizer_converged"
            )
            is True
            for fold in outer_folds
        )
    )
    if not optimizer_gate:
        calibration_reasons.append("LOGISTIC_OPTIMIZER_DID_NOT_CONVERGE")
    publication_gate = bool(calibration_gate and leakage["passed"] and optimizer_gate)
    cohort_hash = str(dataset_sha256 or _cohort_sha256(rows, feature_paths)).lower()
    if len(cohort_hash) != 64 or any(
        character not in "0123456789abcdef" for character in cohort_hash
    ):
        raise ValueError(
            "v5 training dataset hash must contain 64 hexadecimal characters"
        )
    provenance_source = dict(training_provenance or {})
    forbidden_provenance = (
        "full_controls_used_as_predictors",
        "external_data_used_for_selection",
        "external_data_used_for_training",
        "external_threshold_refit",
    )
    contradicted = [
        field for field in forbidden_provenance if provenance_source.get(field) not in (None, False)
    ]
    if contradicted:
        raise ValueError(
            "v5 training provenance contradicts the no-leakage contract: "
            + ", ".join(contradicted)
        )
    generated_at = dt.datetime.now(dt.timezone.utc)
    target_definition = {
        "type": "reduced_vs_full_structural_preservation",
        "version": "5.0",
        "positive_if": {
            "dockq_reduced_minimum": float(dockq_threshold),
            "dockq_full_minus_reduced_maximum": float(max_dockq_loss),
            "logical_operator": "and",
            "boundary_inclusive": True,
        },
        "features_source": "reduced_only_reference_free",
        "full_control_role": "label_only_never_model_input",
        "prediction_time_full_fields_used": [],
    }
    operating_assessment = {
        **hybrid_policy,
        "supported": bool(hybrid_policy["supported"] and optimizer_gate),
        "optimizer_gate_passed": optimizer_gate,
        "scope": "nested_outer_loco_internal_development",
        "activation_permitted": False,
        "activation_block_reason": "untouched_external_confirmation_required",
    }
    external_confirmation = {
        "supported": False,
        "evaluated": False,
        "required": True,
        "thresholds_may_be_refit_on_external": False,
        "thresholds_refit": False,
        "reason": "no untouched external cohort has confirmed the frozen v5 artifact",
    }
    nested_report = {
        "protocol": "outer_LOCO_with_inner_grouped_CV",
        "schema_version": "5.0",
        "grouping_unit": "complex_id",
        "n_rows": len(rows),
        "n_complexes": len(complexes),
        "feature_paths": list(feature_paths),
        "outer_folds": outer_folds,
        "outer_prediction_rows": [value for value in prediction_rows if value is not None],
        "outer_statistical_metrics": statistical_metrics,
        "outer_graph_probability_metrics": graph_probability_metrics,
        "outer_grouped_prevalence_baseline_metrics": baseline_metrics,
        "outer_operating_point_assessment": operating_assessment,
        "group_leakage_audit": leakage,
        "final_inner_selection": final_selection,
    }
    graph_items = [item for item in outer_graph_evidence if item is not None]
    graph_report = {
        "schema_version": "5.0",
        "method": "complex_prototype_knn_evidence_graph",
        "node_unit": "complex_id",
        "prototype": "coordinate_wise_median_of_standardized_reduced_features",
        "outer_loco": {
            "graph_domain_coverage": float(
                np.mean([item.domain_supported for item in graph_items])
            ),
            "graph_supported_acceptance_count": int(
                np.sum(outer_hybrid_accept)
            ),
            "out_of_domain_count": int(
                sum(not item.domain_supported for item in graph_items)
            ),
            "disagreement_abstention_count": int(
                sum(
                    "GRAPH_MODEL_DISAGREEMENT" in (row or {}).get("reason_codes", [])
                    for row in prediction_rows
                )
            ),
            "nearest_distance_mean": float(
                np.mean([item.nearest_distance for item in graph_items])
            ),
            "mean_nearest_distance": float(
                np.mean([item.nearest_distance for item in graph_items])
            ),
            "nearest_distance_median": float(
                median(item.nearest_distance for item in graph_items)
            ),
            "kth_distance_mean": float(
                np.mean([item.kth_distance for item in graph_items])
            ),
            "mean_kth_distance": float(
                np.mean([item.kth_distance for item in graph_items])
            ),
            "effective_neighbor_count_mean": float(
                np.mean([item.effective_neighbor_count for item in graph_items])
            ),
            "graph_model_disagreement_mean": float(
                np.mean([float(item.disagreement or 0.0) for item in graph_items])
            ),
            "mean_graph_disagreement": float(
                np.mean([float(item.disagreement or 0.0) for item in graph_items])
            ),
            "probability_metrics": graph_probability_metrics,
        },
        "final_graph": final_fit["graph"].to_dict(),
        "final_k": int(final_config["k"]),
        "final_domain_quantile": float(final_config["domain_quantile"]),
    }
    decision_report = {
        "schema_version": "5.0",
        "decision_capable": False,
        "internal_nested_support": bool(operating_assessment["supported"]),
        "calibration_quality_gate": {
            "passed": publication_gate,
            "reasons": calibration_reasons,
            "model_metrics": statistical_metrics,
            "baseline_metrics": baseline_metrics,
        },
        "operating_point_assessment": operating_assessment,
        "external_validation_confirmation": external_confirmation,
        "frozen_policy": {
            "risk_target": float(risk_target),
            "minimum_probability": float(final_config["minimum_probability"]),
            "max_graph_disagreement": float(
                final_config["max_graph_disagreement"]
            ),
            "domain_threshold": float(final_fit["graph"].domain_threshold),
            "k": int(final_config["k"]),
        },
    }
    artifact = {
        "schema_version": 5,
        "name": "dockingpp-preservation-hybrid-v5",
        "version": generated_at.strftime("%Y%m%dT%H%M%SZ"),
        "model_kind": "statistical_plus_evidence_graph",
        "feature_profile": PRESERVATION_MECHANISTIC_GRAPH_V5_PROFILE,
        "feature_schema_version": str(feature_schema_version),
        "calibrated": publication_gate,
        "decision_capable": False,
        "activation_state": "observe_only_pending_untouched_external_confirmation",
        "external_validation_ready": bool(
            publication_gate
            and operating_assessment["supported"]
        ),
        "external_validation_mode": "observe_only_no_automatic_decision",
        "decision_probability_threshold_role": (
            "v5_frozen_nested_policy_observation_only"
        ),
        "decision_target": (
            "reduced_preserves_full:"
            f"dockq_reduced>={float(dockq_threshold):g};"
            f"dockq_full-dockq_reduced<={float(max_dockq_loss):g}"
        ),
        "target_definition": target_definition,
        "training_dataset_sha256": cohort_hash,
        "decision_probability_threshold": float(
            final_config["minimum_probability"]
        ),
        "preprocessing": {
            "standardize": True,
            "winsorize": False,
            "pca": False,
            "impute_for_decision": False,
            "missing_feature_policy": "ABSTAIN_AND_FULL",
            "feature_paths": list(feature_paths),
            "means": dict(zip(feature_paths, final_fit["means"].tolist())),
            "scales": dict(zip(feature_paths, final_fit["scales"].tolist())),
            "fit_scope": "complete_development_cohort_after_nested_evaluation",
            "evidence": final_fit["scaler_evidence"],
        },
        "global_model": {
            "type": str(final_config["model"]["type"]),
            "parameters": {
                "C": float(final_config["model"]["C"]),
                "l1_ratio": float(final_config["model"].get("l1_ratio", 0.0)),
            },
            "coefficients": dict(
                zip(feature_paths, final_fit["coefficients"].tolist())
            ),
            "intercept": float(final_fit["intercept"]),
            "optimizer": final_fit["optimizer"],
        },
        "evidence_graph": final_fit["graph"].to_dict(),
        "decision_policy": {
            "risk_target": float(risk_target),
            "minimum_probability": float(final_config["minimum_probability"]),
            "max_graph_disagreement": float(
                final_config["max_graph_disagreement"]
            ),
            "require_domain_support": True,
            "require_graph_agreement": True,
            "missing_feature_action": "ABSTAIN_AND_FULL",
            "selection_scope": "nested_development_only",
            "automatic_activation": False,
        },
        "calibration_evidence": {
            "protocol": "outer_LOCO_with_inner_grouped_CV",
            "grouping_unit": "complex_id",
            "calibration_quality_gate": decision_report[
                "calibration_quality_gate"
            ],
            "quality_gate": decision_report["calibration_quality_gate"],
            "operating_point_assessment": operating_assessment,
            "external_validation_confirmation": external_confirmation,
            "frozen_policy": {
                "risk_target": float(risk_target),
                "minimum_probability": float(final_config["minimum_probability"]),
                "max_graph_disagreement": float(
                    final_config["max_graph_disagreement"]
                ),
                "domain_threshold": float(final_fit["graph"].domain_threshold),
                "k": int(final_config["k"]),
            },
            "group_leakage_audit": leakage,
            "ablation_summary": ablation_report,
        },
        "training_provenance": {
            "generated_at_utc": generated_at.isoformat(),
            "n_rows": len(rows),
            "n_complexes": len(complexes),
            "complex_ids": complexes,
            "data_selection": dict(data_selection or {}),
            "source": provenance_source,
            "full_controls_used_as_predictors": False,
            "external_data_used_for_selection": False,
        },
    }
    return {
        "artifact": artifact,
        "reports": {
            "v5_nested_loco_report.json": nested_report,
            "v5_graph_report.json": graph_report,
            "v5_decision_report.json": decision_report,
            "v5_ablation_report.json": {
                "schema_version": "5.0",
                "protocol": "nested_outer_LOCO",
                "comparisons": ablation_report,
                "interpretation_guard": (
                    "hybrid superiority may be claimed only if external validation "
                    "also confirms risk, coverage, calibration, and runtime"
                ),
            },
        },
    }


__all__ = [
    "V5_DISAGREEMENT_GRID",
    "V5_DOMAIN_QUANTILE_GRID",
    "V5_GRAPH_K_GRID",
    "V5_MODEL_GRID",
    "V5_PROBABILITY_GRID",
    "train_v5_artifact",
]
