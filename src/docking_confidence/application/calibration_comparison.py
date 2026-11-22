"""Exploratory, grouped comparison with separate probability/decision validation.

No production artifact is written. Outer labels never fit models, calibrators,
domains, or thresholds. An unavailable method falls back rather than inventing
a probability. The original docking algorithms are not imported or executed.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from docking_confidence.application.domain import FeatureDomain
from docking_confidence.application.pilot_diagnostics import CANDIDATES, transform_features
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.modeling.logistic import LogisticBaseModel
from docking_confidence.validation.metrics import (
    expected_calibration_error, probability_curves, probability_metrics,
    sigmoid_raw_score,
)

METHODS = ("uncalibrated", "isotonic", "venn_abers")


@dataclass(frozen=True)
class ComparisonOptions:
    seed: int = 20261115
    folds: int = 5
    validation_risk_target: float = .10
    minimum_validation_accepted_groups: int = 3
    bootstrap_repetitions: int = 2000

    def __post_init__(self):
        if self.folds < 3 or not 0 <= self.validation_risk_target < 1:
            raise ValueError("invalid comparison folds or risk target")
        if self.minimum_validation_accepted_groups < 1 or self.bootstrap_repetitions < 100:
            raise ValueError("invalid comparison support or bootstrap repetitions")


def comparison_partitions(groups, options=ComparisonOptions(), planned_groups=None):
    """Assign complete groups without labels, features, or outcome-based retries."""
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups if planned_groups is None else np.r_[groups, np.asarray(planned_groups, str)])
    if len(unique) < max(options.folds, 8):
        raise ValueError("at least eight independent groups are needed")
    shuffled = np.random.default_rng(options.seed).permutation(unique)
    result = []
    for fold, test_groups in enumerate(np.array_split(shuffled, options.folds)):
        development = np.setdiff1d(unique, test_groups)
        development = np.random.default_rng(options.seed + 100 + fold).permutation(development)
        count = math.ceil(len(development) / 4)
        roles = {"probability_calibration": development[:count],
                 "threshold_validation": development[count:2 * count],
                 "model_train": development[2 * count:], "outer_test": test_groups}
        if any(not len(values) for values in roles.values()):
            raise ValueError("insufficient groups for four disjoint roles")
        result.append({"fold": fold,
                       "groups": {role: sorted(values.tolist()) for role, values in roles.items()},
                       "indices": {role: np.flatnonzero(np.isin(groups, values))
                                   for role, values in roles.items()}})
    return result


def select_validation_threshold(scores, y, eligible, groups, *, options=ComparisonOptions(),
                                minimum_threshold=0.0):
    """Empirical exploratory rule, not a confidence guarantee or authorization."""
    scores, y = np.asarray(scores, float), np.asarray(y, int)
    eligible, groups = np.asarray(eligible, bool), np.asarray(groups, str)
    if any(len(value) != len(y) for value in (scores, eligible, groups)):
        raise ValueError("threshold-validation arrays must align")
    common = {"threshold": None, "automatic_authorization": False,
              "risk_target": options.validation_risk_target,
              "selection_data": "threshold_validation_only",
              "selection_objective": "max coverage, min empirical risk, then larger threshold"}
    if not np.isfinite(scores).all():
        return {**common, "reason": "method_unavailable", "candidates": []}
    if len(np.unique(y)) < 2:
        return {**common, "reason": "validation_requires_both_classes", "candidates": []}
    thresholds = np.unique(np.r_[minimum_threshold, scores[scores >= minimum_threshold]])
    candidates = []
    for threshold in thresholds:
        accepted = eligible & (scores >= threshold)
        count = int(accepted.sum())
        wrong = int(((y == 0) & accepted).sum())
        n_groups = len(np.unique(groups[accepted]))
        risk = wrong / count if count else None
        supported = count > 0 and n_groups >= options.minimum_validation_accepted_groups and risk <= options.validation_risk_target
        candidates.append({"threshold": float(threshold), "accepted": count,
                           "false_accepted": wrong, "accepted_groups": n_groups,
                           "empirical_risk": risk, "empirically_feasible": bool(supported)})
    feasible = [candidate for candidate in candidates if candidate["empirically_feasible"]]
    if not feasible:
        return {**common, "reason": "no_empirically_feasible_threshold", "candidates": candidates}
    selected = max(feasible, key=lambda row: (row["accepted"], -row["empirical_risk"], row["threshold"]))
    return {**common, **selected, "reason": "diagnostic_validation_threshold_only",
            "candidates": candidates}


def _mask(scores, eligible, threshold):
    return (np.asarray(eligible, bool) & np.isfinite(scores) & (scores >= threshold)
            if threshold is not None else np.zeros(len(scores), dtype=bool))


def fit_comparison_fold(X, y, groups, *, columns, roles, options=ComparisonOptions(),
                        reduced_quality=None, full_quality=None,
                        rule=PreservationRule()):
    """The only labels read are from model_train, calibration and validation."""
    X = transform_features(X, list(columns), CANDIDATES[0])
    y, groups = np.asarray(y, int), np.asarray(groups, str)
    train, cal, validation, test = [roles[key] for key in
                                  ("model_train", "probability_calibration", "threshold_validation", "outer_test")]
    sets = [set(groups[index]) for index in (train, cal, validation, test)]
    if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
        raise ValueError("a molecular group crosses comparison roles")
    model = LogisticBaseModel(C=1, class_weight="balanced", max_iter=5000).fit(X[train], y[train])
    domain = FeatureDomain(.01, .99).fit(X[train])
    used_cal = cal[domain.contains(X[cal])]
    raw_cal = model.raw_score(X[used_cal]) if len(used_cal) else np.array([])
    raw_test, raw_validation = model.raw_score(X[test]), model.raw_score(X[validation])
    p_test = {name: np.full(len(test), np.nan) for name in METHODS}
    p_validation = {name: np.full(len(validation), np.nan) for name in METHODS}
    p_test["uncalibrated"] = sigmoid_raw_score(raw_test)
    p_validation["uncalibrated"] = sigmoid_raw_score(raw_validation)
    calibration_exports = {}
    if len(used_cal):
        iso = IsotonicRegression(increasing=True, out_of_bounds="clip").fit(raw_cal, y[used_cal])
        p_test["isotonic"], p_validation["isotonic"] = iso.predict(raw_test), iso.predict(raw_validation)
        calibration_exports["isotonic"] = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist(),
                                            "class_deficient": len(np.unique(y[used_cal])) < 2}
    lower_test, lower_validation = np.full(len(test), np.nan), np.full(len(validation), np.nan)
    if len(np.unique(y[used_cal])) == 2:
        va = VennAbersCalibrator().fit(raw_cal, y[used_cal])
        lower_test, _, p_test["venn_abers"] = va.predict_interval(raw_test)
        lower_validation, _, p_validation["venn_abers"] = va.predict_interval(raw_validation)
        calibration_exports["venn_abers"] = va.export()
    test_eligible, validation_eligible = domain.contains(X[test]), domain.contains(X[validation])
    scores_test = {**p_test, "venn_abers_lower": lower_test}
    scores_validation = {**p_validation, "venn_abers_lower": lower_validation}
    cascade = None
    if reduced_quality is not None and full_quality is not None:
        qr, qf = np.asarray(reduced_quality, float), np.asarray(full_quality, float)
        scaler = StandardScaler().fit(X[train])
        reg = Ridge(alpha=1).fit(scaler.transform(X[train]), np.c_[qr[train], qf[train] - qr[train]])
        estimate = {}
        for role, indices in (("test", test), ("validation", validation)):
            pred = reg.predict(scaler.transform(X[indices]))
            estimate[role] = pred
            margin = np.minimum(pred[:, 0] - rule.minimum_reduced_dockq,
                                rule.maximum_dockq_loss - pred[:, 1])
            (scores_test if role == "test" else scores_validation)["cascade_margin"] = margin
        cascade = {"test_quality": estimate["test"][:, 0],
                   "test_gain": estimate["test"][:, 1], "coef": reg.coef_.tolist(),
                   "intercept": reg.intercept_.tolist(), "scaler_mean": scaler.mean_.tolist(),
                   "scaler_scale": scaler.scale_.tolist(), "alpha": 1,
                   "scope": "linear_LtC_inspired_hypothesis_not_paper_reproduction_or_probability"}
    policies, accepted = {}, {}
    for name, scores in scores_validation.items():
        policies[name] = select_validation_threshold(scores, y[validation], validation_eligible,
                                                     groups[validation], options=options)
        accepted[name] = _mask(scores_test[name], test_eligible, policies[name]["threshold"])
    before_negative = cal[y[cal] == 0]
    excluded_negative = before_negative[~domain.contains(X[before_negative])]
    exclusions = [{"row": int(index), "group": str(groups[index]),
                   "features": [columns[j] for j in np.flatnonzero((X[index] < domain.lower) | (X[index] > domain.upper))]}
                  for index in excluded_negative]
    return {"probabilities": p_test, "scores": scores_test,
            "validation_scores": scores_validation, "accepted": accepted, "policies": policies,
            "test_eligible": test_eligible, "validation_eligible": validation_eligible,
            "raw_test_score": raw_test, "model": model.export(list(columns)),
            "domain": domain.export(), "calibrators": calibration_exports, "cascade": cascade,
            "calibration_support": {
                "planned_rows": len(cal), "used_rows": len(used_cal),
                "planned_groups": len(np.unique(groups[cal])), "used_groups": len(np.unique(groups[used_cal])),
                "negative_groups_before_domain": len(np.unique(groups[cal][y[cal] == 0])),
                "positive_groups_after_domain": len(np.unique(groups[used_cal][y[used_cal] == 1])),
                "negative_groups_after_domain": len(np.unique(groups[used_cal][y[used_cal] == 0])),
                "negative_exclusions": exclusions},
            "training_prevalence": float(y[np.r_[train, cal, validation]].mean())}


def metric_bundle(y, probability, ranking_scores=None):
    metrics = probability_metrics(y, probability, ranking_scores=ranking_scores)
    metrics["ece_5bin"] = expected_calibration_error(y, probability, 5)
    metrics["mean_probability"] = float(np.mean(probability)) if len(probability) else None
    metrics["wrong_exact_endpoint_predictions"] = int((((probability == 0) & (y == 1)) | ((probability == 1) & (y == 0))).sum())
    metrics["log_loss_clip_epsilon"] = 1e-12
    reverse = -ranking_scores if ranking_scores is not None else None
    return {"preservation": metrics, "violation": probability_metrics(1 - y, 1 - probability, ranking_scores=reverse),
            "curves": probability_curves(y, probability, ranking_scores=ranking_scores),
            "violation_curves": probability_curves(1 - y, 1 - probability, ranking_scores=reverse)}


def compare_calibrators(frame, columns, *, rule=PreservationRule(), options=ComparisonOptions(), planned_groups=None):
    ordered = frame.sort_values(["complex_fingerprint_sha256", "complex_id", "algorithm", "seed"], kind="stable").reset_index(drop=True)
    X = ordered[list(columns)].to_numpy(float)
    y = np.array([rule.label(r, f) for r, f in zip(ordered.dockq_reduced, ordered.dockq_full)], int)
    groups = ordered.complex_fingerprint_sha256.to_numpy(str)
    partitions = comparison_partitions(groups, options, planned_groups)
    predictions = ordered[["complex_id", "complex_fingerprint_sha256", "algorithm", "seed",
                           "dockq_reduced", "dockq_full", "runtime_reduced_sec", "runtime_full_sec"]].copy()
    predictions["preserved"] = y
    predictions["fold"] = -1
    names = (*METHODS, "venn_abers_lower", "cascade_margin")
    for name in names:
        predictions[name + "__score"] = np.nan
        predictions[name + "__accept"] = False
        predictions[name + "__economic_engine_only_accept"] = False
        predictions[name + "__economic_engine_only_enabled"] = False
    predictions["training_prevalence"] = np.nan
    predictions["in_domain"] = False
    predictions["cascade_quality"] = np.nan
    predictions["cascade_gain"] = np.nan
    predictions["base_raw_score"] = np.nan
    folds = []
    for partition in partitions:
        roles = partition["indices"]; test, validation = roles["outer_test"], roles["threshold_validation"]
        try:
            fitted = fit_comparison_fold(X, y, groups, columns=list(columns), roles=roles, options=options,
                                         reduced_quality=ordered.dockq_reduced.to_numpy(float),
                                         full_quality=ordered.dockq_full.to_numpy(float), rule=rule)
        except ValueError as exc:
            if "both binary classes" not in str(exc):
                raise
            folds.append({"fold": partition["fold"], "groups": partition["groups"], "unavailable_reason": str(exc)})
            predictions.loc[test, "fold"] = partition["fold"]
            continue
        predictions.loc[test, "fold"] = partition["fold"]
        predictions.loc[test, "in_domain"] = fitted["test_eligible"]
        predictions.loc[test, "training_prevalence"] = fitted["training_prevalence"]
        predictions.loc[test, "base_raw_score"] = fitted["raw_test_score"]
        validation_cost = ordered.iloc[validation]
        for name in names:
            predictions.loc[test, name + "__score"] = fitted["scores"][name]
            predictions.loc[test, name + "__accept"] = fitted["accepted"][name]
            val_accept = _mask(fitted["validation_scores"][name], fitted["validation_eligible"], fitted["policies"][name]["threshold"])
            saving = float(validation_cost.loc[val_accept, "runtime_full_sec"].sum() - validation_cost.runtime_reduced_sec.sum())
            enabled = saving > 0
            fitted["policies"][name]["validation_engine_saving_upper_sec"] = saving
            fitted["policies"][name]["economic_engine_only_enabled"] = enabled
            predictions.loc[test, name + "__economic_engine_only_enabled"] = enabled
            predictions.loc[test, name + "__economic_engine_only_accept"] = fitted["accepted"][name] & enabled
        predictions.loc[test, "cascade_quality"] = fitted["cascade"]["test_quality"]
        predictions.loc[test, "cascade_gain"] = fitted["cascade"]["test_gain"]
        cascade_export = {key: value for key, value in fitted["cascade"].items() if not key.startswith("test_")}
        folds.append({"fold": partition["fold"], "groups": partition["groups"], "cascade": cascade_export,
                      **{key: fitted[key] for key in ("model", "domain", "calibrators", "policies", "calibration_support")}})
    common = np.logical_and.reduce([np.isfinite(predictions[name + "__score"].to_numpy(float)) for name in METHODS])
    summaries = {}
    for name in METHODS:
        p = predictions[name + "__score"].to_numpy(float)
        available = np.isfinite(p)
        ranking = predictions.base_raw_score.to_numpy(float) if name == "uncalibrated" else p
        in_domain = common & predictions.in_domain.to_numpy(bool)
        summaries[name] = {"available_rows": int(available.sum()),
                           "common_test_rows": metric_bundle(y[common], p[common], ranking[common]),
                           "own_available_rows": metric_bundle(y[available], p[available], ranking[available]),
                           "common_in_domain_rows": metric_bundle(y[in_domain], p[in_domain], ranking[in_domain])}
    reference = predictions.training_prevalence.to_numpy(float)
    return {"schema_version": "grouped_calibration_comparison_v1", "options": asdict(options),
            "scope": "exploratory_reused_development_data_not_final_confirmation", "automatic_authorization": False,
            "features": list(columns), "preservation_rule": asdict(rule), "folds": folds,
            "counts": {"valid_pairs": len(ordered), "groups": len(np.unique(groups)),
                       "preserved": int(y.sum()), "violations": int((y == 0).sum()),
                       "common_test_rows": int(common.sum()), "common_test_groups": len(np.unique(groups[common]))},
            "metrics": summaries,
            "prevalence_reference_common": metric_bundle(y[common], reference[common]),
            "predictions": predictions.replace({np.nan: None}).to_dict("records")}
