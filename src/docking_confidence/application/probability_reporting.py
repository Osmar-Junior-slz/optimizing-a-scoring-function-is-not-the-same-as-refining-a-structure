"""Retrospective diagnostics from frozen predictions, without docking or gate changes."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from docking_confidence.application.domain import FeatureDomain
from docking_confidence.application.runtime import operational_cost_assessment
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.modeling.logistic import LogisticBaseModel
from docking_confidence.validation.metrics import probability_curves, probability_metrics, sigmoid_raw_score
from docking_confidence.validation.split import GroupPartition


METRIC_NAMES = (
    "roc_auc", "pr_auc_trapezoidal", "average_precision", "brier_score",
    "log_loss", "expected_calibration_error_10bin",
)


@dataclass(frozen=True)
class DiagnosticOptions:
    bootstrap_repetitions: int = 500
    bootstrap_seed: int = 20261010
    confidence_level: float = 0.95

    def __post_init__(self):
        if type(self.bootstrap_repetitions) is not int or not 100 <= self.bootstrap_repetitions <= 100_000:
            raise ValueError("diagnostic bootstrap_repetitions must be an integer in [100, 100000]")
        if type(self.bootstrap_seed) is not int or self.bootstrap_seed < 0:
            raise ValueError("diagnostic bootstrap_seed must be a nonnegative integer")
        if isinstance(self.confidence_level, bool) or not 0 < self.confidence_level < 1:
            raise ValueError("diagnostic confidence_level must be in (0, 1)")


def _cluster_draws(groups: np.ndarray, options: DiagnosticOptions):
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.Generator(np.random.PCG64(options.bootstrap_seed))
    for _ in range(options.bootstrap_repetitions):
        yield np.concatenate([members[index] for index in rng.integers(0, len(unique), len(unique))])


def _interval(values: list[float], options: DiagnosticOptions, reason: str | None) -> dict:
    # A class-deficient resample stays unavailable; it is never assigned AUC=0.5.
    enough = len(values) >= max(50, int(np.ceil(0.8 * options.bootstrap_repetitions)))
    tail = (1 - options.confidence_level) / 2
    endpoints = np.quantile(values, [tail, 1 - tail]) if enough and reason is None else (None, None)
    return {"lower": float(endpoints[0]) if endpoints[0] is not None else None,
            "upper": float(endpoints[1]) if endpoints[1] is not None else None,
            "valid_repetitions": len(values),
            "unavailable_repetitions": options.bootstrap_repetitions - len(values),
            "unavailable_reason": reason or (None if enough else "insufficient_valid_resamples")}


def _resampled_metrics(y: np.ndarray, p: np.ndarray, ranking_scores: np.ndarray | None = None) -> dict:
    """Small validated bootstrap samples avoid repeated sklearn validation overhead.

    Group exact score ties before integrating, matching sklearn's ranking curves.
    Point estimates still use the public sklearn metrics in probability_metrics.
    """
    count = len(y)
    positives = int(y.sum())
    negatives = count - positives
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    bin_ids = np.minimum(np.searchsorted(np.linspace(0, 1, 11), p, side="right") - 1, 9)
    positive_bins = np.bincount(bin_ids, weights=y, minlength=10)
    probability_bins = np.bincount(bin_ids, weights=p, minlength=10)
    result = {"brier_score": float(np.mean((y - p) ** 2)),
              "log_loss": float(-np.mean(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))),
              "expected_calibration_error_10bin": float(np.abs(positive_bins - probability_bins).sum() / count),
              "roc_auc": None, "pr_auc_trapezoidal": None, "average_precision": None}
    if positives and negatives:
        scores = p if ranking_scores is None else ranking_scores
        order = np.argsort(-scores, kind="stable")
        sorted_score, sorted_y = scores[order], y[order]
        endpoints = np.flatnonzero(np.r_[sorted_score[:-1] != sorted_score[1:], True])
        tp = np.cumsum(sorted_y)[endpoints]
        fp = endpoints + 1 - tp
        recall = np.r_[0.0, tp / positives]
        precision = np.r_[1.0, tp / (tp + fp)]
        fpr = np.r_[0.0, fp / negatives]
        result.update(roc_auc=float(np.sum(np.diff(fpr) * (recall[:-1] + recall[1:]) / 2)),
            pr_auc_trapezoidal=float(np.sum(np.diff(recall) * (precision[:-1] + precision[1:]) / 2)),
            average_precision=float(np.sum(np.diff(recall) * precision[1:])))
    return result


def frozen_prediction_report(frame: pd.DataFrame, *, options: DiagnosticOptions | None = None) -> dict:
    """Compare sigmoid(logit) with frozen VA on identical saved test rows."""
    options = options or DiagnosticOptions()
    required = {"preserved", "base_raw_score", "probability", "complex_fingerprint_sha256"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"saved predictions lack diagnostic columns: {sorted(missing)}")
    # Sorting stabilizes the random replay when a CSV is reordered.
    order_columns = ["complex_fingerprint_sha256", "preserved", "base_raw_score", "probability"]
    ordered = frame.sort_values(order_columns, kind="stable").reset_index(drop=True)
    if ordered["complex_fingerprint_sha256"].isna().any() or ordered["complex_fingerprint_sha256"].astype(str).str.strip().eq("").any():
        raise ValueError("diagnostics require nonmissing complex fingerprints")
    groups = ordered["complex_fingerprint_sha256"].to_numpy(dtype=str)
    y = ordered["preserved"].to_numpy()
    probabilities = {
        "before_venn_abers": sigmoid_raw_score(ordered["base_raw_score"].to_numpy(dtype=float)),
        "after_venn_abers": ordered["probability"].to_numpy(dtype=float),
    }
    ranking_scores = {"before_venn_abers": ordered["base_raw_score"].to_numpy(dtype=float),
                      "after_venn_abers": probabilities["after_venn_abers"]}
    evaluations = {name: {"metrics": probability_metrics(y, p, ranking_scores=ranking_scores[name]),
                         "curves": probability_curves(y, p, ranking_scores=ranking_scores[name])}
                   for name, p in probabilities.items()}
    samples = {name: {metric: [] for metric in METRIC_NAMES} for name in probabilities}
    differences = {metric: [] for metric in METRIC_NAMES}
    cluster_count = len(np.unique(groups))
    reason = "no_rows" if not len(y) else ("fewer_than_two_complexes" if cluster_count < 2 else None)
    if reason is None:
        for indices in _cluster_draws(groups, options):
            replicate = {name: _resampled_metrics(y[indices], p[indices], ranking_scores[name][indices]) for name, p in probabilities.items()}
            for metric in METRIC_NAMES:
                for name in probabilities:
                    value = replicate[name][metric]
                    if value is not None:
                        samples[name][metric].append(value)
                before, after = (replicate[name][metric] for name in probabilities)
                if before is not None and after is not None:
                    differences[metric].append(after - before)
    for name in probabilities:
        evaluations[name]["confidence_intervals"] = {
            metric: _interval(samples[name][metric], options,
                reason or (evaluations[name]["metrics"]["ranking_unavailable_reason"]
                           if metric in METRIC_NAMES[:3] else None))
            for metric in METRIC_NAMES
        }
    return {
        "schema_version": "frozen_probability_diagnostics_v1",
        "scope": "retrospective_diagnostic_not_model_selection_or_automatic_authorization",
        "positive_class": "preserved_reduced_structural_quality",
        "positive_prevalence_pr_baseline": float(np.mean(y)) if len(y) else None,
        "n_complexes": cluster_count,
        "bootstrap": {**asdict(options), "method": "whole_complex_percentile",
                      "rng": "PCG64", "weighting": "pooled_runs",
                      "minimum_valid_fraction": 0.8,
                      "interpretation": "approximate conditional on frozen fitted models; no refitting"},
        "evaluations": evaluations,
        "paired_after_minus_before": {
            metric: {"estimate": (
                evaluations["after_venn_abers"]["metrics"][metric] - evaluations["before_venn_abers"]["metrics"][metric]
                if evaluations["after_venn_abers"]["metrics"][metric] is not None else None),
                "confidence_interval": _interval(differences[metric], options,
                    reason or (evaluations["after_venn_abers"]["metrics"]["ranking_unavailable_reason"]
                               if metric in METRIC_NAMES[:3] else None))}
            for metric in METRIC_NAMES
        },
    }


def descriptor_comparison(
    df: pd.DataFrame, split: GroupPartition, feature_columns: list[str], config: dict,
    used_probability_indices: np.ndarray, combined_test_frame: pd.DataFrame,
    *, options: DiagnosticOptions | None = None,
) -> dict:
    """Fit only train/probability roles; evaluate frozen variants on locked test.

    All variants use exactly the original model's in-domain probability-calibration
    rows, so changing descriptors does not silently change the calibration sample.
    No policy/threshold/model selection is performed from the resulting test metrics.
    """
    families = {
        "population": [column for column in feature_columns if not column.startswith("graph_")],
        "interface": [column for column in feature_columns if column.startswith("graph_")],
        "combined": list(feature_columns),
    }
    y = df["preserved"].to_numpy(dtype=int)
    result = {"scope": "retrospective_descriptor_ablation_not_selection",
              "split": split.export(), "calibration_sample": "same_original_in_domain_probability_calibration_rows",
              "calibration_row_count": len(used_probability_indices), "variants": {}}
    for name, columns in families.items():
        if not columns:
            result["variants"][name] = {"available": False, "reason": "feature_family_absent", "features": []}
            continue
        if name == "combined":
            result["variants"][name] = {"available": True, "features": columns,
                "model_origin": "unchanged_primary_frozen_model", "diagnostics": frozen_prediction_report(combined_test_frame, options=options)}
            continue
        X = df[columns].to_numpy(dtype=float)
        params = config["model"]
        model = LogisticBaseModel(C=float(params["C"]),
            class_weight=None if params.get("class_weight") == "none" else params.get("class_weight"),
            max_iter=int(params["max_iter"])).fit(X[split.model_train], y[split.model_train])
        calibrator = VennAbersCalibrator().fit(model.raw_score(X[used_probability_indices]), y[used_probability_indices])
        params = config["applicability_domain"]
        domain = FeatureDomain(lower_quantile=params["lower_quantile"], upper_quantile=params["upper_quantile"]).fit(X[split.model_train])
        frame = df.iloc[split.locked_test][["complex_fingerprint_sha256", "preserved"]].copy()
        raw = model.raw_score(X[split.locked_test])
        lower, upper, point = calibrator.predict_interval(raw)
        frame["base_raw_score"], frame["probability"] = raw, point
        result["variants"][name] = {
            "available": True, "features": columns, "model": model.export(columns),
            "calibration": calibrator.export(), "applicability_domain": domain.export(),
            "locked_predictions": frame.assign(probability_lower=lower, probability_upper=upper,
                in_applicability_domain=domain.contains(X[split.locked_test])).to_dict("records"),
            "diagnostics": frozen_prediction_report(frame, options=options),
        }
    return result


def saved_test_diagnostics(frame: pd.DataFrame, *, options: DiagnosticOptions | None = None) -> dict:
    result = {"probability": frozen_prediction_report(frame, options=options)}
    result["coverage_risk"] = coverage_risk_curve(frame)
    needed = {"runtime_reduced_sec", "runtime_full_sec", "accepted_by_locked_policy"}
    if needed <= set(frame.columns):
        accepted = frame["accepted_by_locked_policy"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
        if accepted.isna().any():
            raise ValueError("saved acceptance flags must be explicit booleans")
        result["operational_cost"] = operational_cost_assessment(frame, accepted.to_numpy(dtype=bool))
    else:
        result["operational_cost"] = {"available": False, "reason": "paired_times_or_policy_flags_absent"}
    return result


def coverage_risk_curve(frame: pd.DataFrame) -> dict:
    needed = {"probability_lower", "in_applicability_domain", "preserved", "complex_fingerprint_sha256"}
    if not needed <= set(frame.columns):
        return {"available": False, "reason": "lower_probability_or_domain_evidence_absent", "points": []}
    lower = frame["probability_lower"].to_numpy(dtype=float)
    if not np.isfinite(lower).all() or ((lower < 0) | (lower > 1)).any():
        raise ValueError("coverage-risk lower probabilities must be finite and in [0, 1]")
    eligible = frame["in_applicability_domain"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    if eligible.isna().any():
        raise ValueError("coverage-risk domain flags must be explicit booleans")
    groups = frame["complex_fingerprint_sha256"].to_numpy(dtype=str)
    total_complexes = len(np.unique(groups))
    labels = frame["preserved"].to_numpy(dtype=int)
    points = []
    for threshold in np.linspace(0, 1, 21):
        accepted = eligible.to_numpy(dtype=bool) & (lower >= threshold)
        accepted_count = int(accepted.sum())
        false_count = int((accepted & (labels == 0)).sum())
        points.append({"threshold": float(threshold), "accepted_count": accepted_count,
            "false_accepted_count": false_count,
            "run_coverage": accepted_count / len(frame) if len(frame) else None,
            "complex_coverage": len(np.unique(groups[accepted])) / total_complexes if total_complexes else None,
            "empirical_run_risk": false_count / accepted_count if accepted_count else None})
    return {"available": bool(len(frame)), "reason": None if len(frame) else "no_rows",
            "scope": "diagnostic_fixed_grid_not_threshold_selection", "points": points}
