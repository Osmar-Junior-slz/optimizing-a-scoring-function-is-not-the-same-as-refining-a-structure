"""Exploratory model development with whole-complex held-out predictions.

This module never writes a production gate or changes its authorization rules.
The outer outcomes are diagnostic development data, not a pristine final test.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from docking_confidence.application.domain import FeatureDomain
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.modeling.logistic import LogisticBaseModel
from docking_confidence.io.config import REDUCED_SEARCH_FEATURE_COLUMNS
from docking_confidence.features.contact_graph import CONTACT_GRAPH_FEATURE_COLUMNS
from docking_confidence.features.pilot_execution import PILOT_EXECUTION_FEATURE_COLUMNS
from docking_confidence.features.execution_context import EXECUTION_CONTEXT_FEATURE_COLUMNS
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.validation.metrics import (
    probability_curves, probability_metrics, sigmoid_raw_score,
)


@dataclass(frozen=True)
class PilotCandidate:
    name: str
    C: float
    class_weight: str | None
    log_counts: bool = False
    model: str = "logistic"


CANDIDATES = (
    PilotCandidate("legacy_balanced_C1", 1.0, "balanced"),
    PilotCandidate("unweighted_C1", 1.0, None),
    PilotCandidate("balanced_C0.1", 0.1, "balanced"),
    PilotCandidate("log_counts_balanced_C0.1", 0.1, "balanced", True),
    PilotCandidate("log_counts_unweighted_C0.1", 0.1, None, True),
)
LOG_COLUMNS = frozenset({
    "uncertainty_count", "graph_interface_edge_count", "graph_clash_atom_pair_count",
})
PILOT_ALGORITHM_COLUMNS = ("algorithm_ag", "algorithm_pso", "algorithm_hybrid")
PILOT_ALLOWED_FEATURES = frozenset((
    *REDUCED_SEARCH_FEATURE_COLUMNS, *CONTACT_GRAPH_FEATURE_COLUMNS, *PILOT_ALGORITHM_COLUMNS,
    *PILOT_EXECUTION_FEATURE_COLUMNS,
    *EXECUTION_CONTEXT_FEATURE_COLUMNS,
))


class _TreePilotModel:
    """Diagnostic score only; its exports cannot be loaded as production gates."""
    def __init__(self, kind):
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        self.kind = kind
        self.model = (
            HistGradientBoostingClassifier(
                max_iter=60, learning_rate=.05, max_leaf_nodes=4,
                max_depth=2, min_samples_leaf=12, l2_regularization=5.0,
                early_stopping=False, random_state=0,
            ) if kind == "hist_gradient_boosting" else
            RandomForestClassifier(
                n_estimators=200, max_depth=3, min_samples_leaf=8,
                max_features=.7, random_state=0, n_jobs=1,
            )
        )

    def fit(self, X, y):
        if len(np.unique(y)) != 2:
            raise ValueError("pilot tree requires both classes in inner training")
        self.model.fit(X, y)
        return self

    def raw_score(self, X):
        probability = np.clip(self.model.predict_proba(X)[:, 1], 1e-9, 1 - 1e-9)
        return np.log(probability / (1 - probability))

    def export(self, columns):
        return {"method": self.kind, "parameters": self.model.get_params(),
                "feature_columns": list(columns), "production_loadable": False}


def transform_features(X: np.ndarray, columns: list[str], candidate: PilotCandidate) -> np.ndarray:
    if len(set(columns)) != len(columns) or set(columns) - PILOT_ALLOWED_FEATURES:
        raise ValueError("pilot feature allowlist rejects outcomes, references, FULL data or unknown features")
    values = np.asarray(X, dtype=float).copy()
    if values.ndim != 2 or values.shape[1] != len(columns) or not np.isfinite(values).all():
        raise ValueError("pilot features must be a finite matrix in the declared order")
    if candidate.log_counts:
        indices = [i for i, column in enumerate(columns) if column in LOG_COLUMNS]
        if (values[:, indices] < 0).any():
            raise ValueError("count features must be nonnegative before log1p")
        values[:, indices] = np.log1p(values[:, indices])
    return values


def grouped_pilot_partitions(groups, *, seed: int = 20261115, folds: int = 5) -> list[dict]:
    """Freeze roles without inspecting outcomes; all aliases share a role."""
    group_values = np.asarray(groups, dtype=str)
    unique = np.unique(group_values)
    if len(unique) < max(folds, 6):
        raise ValueError("pilot cross-validation requires at least six molecular groups")
    shuffled = np.random.default_rng(seed).permutation(unique)
    test_folds = np.array_split(shuffled, folds)
    result = []
    for fold, test_groups in enumerate(test_folds):
        development = np.setdiff1d(unique, test_groups)
        development = np.random.default_rng(seed + 100 + fold).permutation(development)
        calibration_count = max(2, int(np.ceil(len(development) / 3)))
        calibration_groups = development[:calibration_count]
        training_groups = development[calibration_count:]
        roles = {
            "model_train": sorted(training_groups.tolist()),
            "probability_calibration": sorted(calibration_groups.tolist()),
            "outer_assessment": sorted(test_groups.tolist()),
        }
        role_sets = [set(value) for value in roles.values()]
        if any(role_sets[i] & role_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise RuntimeError("pilot molecular groups leaked across roles")
        result.append({
            "fold": fold, "groups": roles,
            "indices": {role: np.flatnonzero(np.isin(group_values, members))
                        for role, members in roles.items()},
        })
    return result


def fit_pilot_fold(
    X, y, *, columns, candidate, training, calibration, assessment, domain_X=None,
) -> dict:
    """The assessment labels are never read while fitting or predicting."""
    transformed = transform_features(X, columns, candidate)
    if candidate.model == "logistic":
        model = LogisticBaseModel(
            C=candidate.C, class_weight=candidate.class_weight, max_iter=5000,
        )
    elif candidate.model in {"random_forest", "hist_gradient_boosting"}:
        model = _TreePilotModel(candidate.model)
    else:
        raise ValueError("unsupported diagnostic model")
    model.fit(transformed[training], y[training])
    # Keep the exact legacy applicability rule and raw feature domain.
    domain_values = np.asarray(X if domain_X is None else domain_X)
    domain = FeatureDomain(0.01, 0.99).fit(domain_values[training])
    eligible_calibration = calibration[domain.contains(domain_values[calibration])]
    raw = model.raw_score(transformed[assessment])
    result = {
        "raw_score": raw, "before_probability": sigmoid_raw_score(raw),
        "in_domain": domain.contains(domain_values[assessment]),
        "calibration_indices": eligible_calibration,
        "model": model.export(columns),
        "feature_transform": "log1p_declared_counts" if candidate.log_counts else "identity",
        "after_probability": None, "probability_lower": None, "probability_upper": None,
        "calibration_unavailable_reason": None,
    }
    if len(np.unique(y[eligible_calibration])) < 2:
        result["calibration_unavailable_reason"] = "fewer_than_two_classes_in_inner_calibration_domain"
        return result
    calibrator = VennAbersCalibrator().fit(
        model.raw_score(transformed[eligible_calibration]), y[eligible_calibration],
    )
    lower, upper, probability = calibrator.predict_interval(raw)
    result.update(after_probability=probability, probability_lower=lower,
                  probability_upper=upper, calibration=calibrator.export())
    return result


def _metric_bundle(y, probability, ranking=None) -> dict:
    return {
        "preservation": probability_metrics(y, probability, ranking_scores=ranking),
        "violation": probability_metrics(
            1 - y, 1 - probability,
            ranking_scores=None if ranking is None else -ranking,
        ),
        "curves": probability_curves(y, probability, ranking_scores=ranking),
    }


def _score_policy(y, lower, eligible, groups) -> list[dict]:
    rows = []
    for threshold in (0.5, 0.7, 0.8, 0.9):
        accepted = eligible & (lower >= threshold)
        count = int(accepted.sum())
        errors = int(((y == 0) & accepted).sum())
        rows.append({
            "threshold": threshold, "accepted": count, "false_accepted": errors,
            "accepted_complexes": len(np.unique(groups[accepted])),
            "coverage_of_valid_pairs": count / len(y) if len(y) else None,
            "empirical_risk": errors / count if count else None,
        })
    return rows


def compare_pilot_candidates(
    frame: pd.DataFrame, columns: list[str], *, seed=20261115, candidates=CANDIDATES,
    domain_columns=None, group_column="complex_fingerprint_sha256", preservation_rule=None,
) -> dict:
    if group_column not in {"complex_fingerprint_sha256", "pilot_source_cluster"}:
        raise ValueError("pilot grouping must use molecular identity or audited source clusters")
    required = {*columns, "dockq_reduced", "dockq_full", "complex_fingerprint_sha256",
                "complex_id", "algorithm", "seed"}
    if missing := required - set(frame):
        raise ValueError(f"pilot dataset lacks columns: {sorted(missing)}")
    ordered = frame.sort_values(
        ["complex_fingerprint_sha256", "complex_id", "algorithm", "seed"], kind="stable",
    ).reset_index(drop=True)
    X = ordered[columns].to_numpy(dtype=float)
    domain_columns = list(columns if domain_columns is None else domain_columns)
    if set(domain_columns) - PILOT_ALLOWED_FEATURES:
        raise ValueError("pilot domain cannot use outcomes or unknown features")
    domain_X = ordered[domain_columns].to_numpy(dtype=float)
    for column in set(columns) & set(PILOT_ALGORITHM_COLUMNS):
        expected = (ordered.algorithm == column.removeprefix("algorithm_")).to_numpy(dtype=int)
        if not np.array_equal(ordered[column].to_numpy(), expected):
            raise ValueError("pilot algorithm indicators disagree with REDUCED run metadata")
    rule = preservation_rule or PreservationRule()
    y = np.asarray([rule.label(reduced, full) for reduced, full in
                    zip(ordered.dockq_reduced, ordered.dockq_full)], dtype=int)
    groups = ordered[group_column].to_numpy(dtype=str)
    partitions = grouped_pilot_partitions(groups, seed=seed)
    role_report = [{k: v for k, v in fold.items() if k != "indices"} for fold in partitions]
    predictions = ordered[["complex_id", "complex_fingerprint_sha256", "algorithm", "seed"]].copy()
    if group_column != "complex_fingerprint_sha256":
        predictions[group_column] = groups
    predictions["preserved"] = y
    predictions["training_prevalence_probability"] = np.nan
    candidate_reports, results = {}, {}
    for candidate in candidates:
        before, raw, after, lower, upper = (np.full(len(y), np.nan) for _ in range(5))
        eligible = np.zeros(len(y), dtype=bool)
        fold_reports = []
        for fold in partitions:
            roles = fold["indices"]
            train, calibration, test = (roles[key] for key in (
                "model_train", "probability_calibration", "outer_assessment"))
            development = np.r_[train, calibration]
            predictions.loc[test, "training_prevalence_probability"] = float(y[development].mean())
            fitted = fit_pilot_fold(
                X, y, columns=columns, candidate=candidate,
                training=train, calibration=calibration, assessment=test,
                domain_X=domain_X,
            )
            before[test], raw[test], eligible[test] = (
                fitted["before_probability"], fitted["raw_score"], fitted["in_domain"])
            used = fitted["calibration_indices"]
            if fitted["after_probability"] is not None:
                after[test], lower[test], upper[test] = (
                    fitted["after_probability"], fitted["probability_lower"],
                    fitted["probability_upper"])
            fold_reports.append({
                "fold": fold["fold"], "training_rows": len(train), "assessment_rows": len(test),
                "calibration_rows_planned": len(calibration), "calibration_rows_used": len(used),
                "calibration_groups_used": len(np.unique(groups[used])),
                "calibration_positive_groups": len(np.unique(groups[used][y[used] == 1])),
                "calibration_negative_groups": len(np.unique(groups[used][y[used] == 0])),
                "calibration_unavailable_reason": fitted["calibration_unavailable_reason"],
                "model": fitted["model"], "feature_transform": fitted["feature_transform"],
            })
        candidate_reports[candidate.name] = {
            "configuration": asdict(candidate), "before": _metric_bundle(y, before, raw),
            "folds": fold_reports, "after_available_rows": int(np.isfinite(after).sum()),
            "domain_eligible_rows": int(eligible.sum()),
        }
        for key, values in (("before", before), ("raw", raw), ("after", after),
                            ("lower", lower), ("upper", upper), ("in_domain", eligible)):
            predictions[f"{candidate.name}__{key}"] = values
        results[candidate.name] = (before, after, lower, eligible)
    common = np.logical_and.reduce([np.isfinite(value[1]) for value in results.values()])
    reference = predictions.training_prevalence_probability.to_numpy(dtype=float)
    for name, (_, after, lower, eligible) in results.items():
        candidate_reports[name]["after_on_common_rows"] = _metric_bundle(y[common], after[common])
        candidate_reports[name]["hypothetical_policy_on_common_rows"] = _score_policy(
            y[common], lower[common], eligible[common], groups[common])
    feature_health = [{
        "name": name, "unique_values": len(np.unique(X[:, i])),
        "minimum": float(X[:, i].min()), "median": float(np.median(X[:, i])),
        "maximum": float(X[:, i].max()),
        "constant": bool(np.ptp(X[:, i]) == 0),
    } for i, name in enumerate(columns)]
    records = predictions.replace({np.nan: None}).to_dict("records")
    return {
        "schema_version": "calibrator_pilot_comparison_v1",
        "feature_columns": list(columns),
        "domain_feature_columns": domain_columns,
        "scope": "exploratory_model_development_not_final_independent_validation",
        "automatic_authorization": False, "production_gates_changed": False,
        "evaluation": {
            "outer_folds": 5, "seed": seed, "grouping": group_column,
            "assignment": "label_independent", "inner_calibration": "one_third_of_outer_development_groups",
            "domain": "unchanged_legacy_1_to_99_percentile_box_fitted_on_inner_training",
            "candidate_selection_note": "Comparing these outer outcomes is exploratory; no candidate is thereby independently validated.",
        },
        "counts": {
            "rows": len(y), "complexes": len(np.unique(groups)),
            "preserved": int(y.sum()), "violations": int((y == 0).sum()),
            "complexes_with_violations": len(np.unique(groups[y == 0])),
            "common_calibrated_rows": int(common.sum()),
        },
        "partitions": role_report, "feature_health": feature_health,
        "reference": {
            "training_prevalence": _metric_bundle(y, reference),
            "training_prevalence_common_rows": _metric_bundle(y[common], reference[common]),
            "accept_all_valid_risk": float((y == 0).mean()),
        },
        "candidates": candidate_reports, "predictions": records,
    }


def run_pilot_comparison(preparation_dir: str | Path, output_dir: str | Path) -> dict:
    source, destination = Path(preparation_dir).resolve(), Path(output_dir).resolve()
    preparation = json.loads((source / "dataset_build_report.json").read_text(encoding="utf-8"))
    if preparation.get("valid") is not True:
        raise ValueError("pilot requires a valid provenance-checked dataset preparation")
    dataset = Path(preparation["dataset_path"])
    payload = dataset.read_bytes()
    if hashlib.sha256(payload).hexdigest() != preparation["dataset_sha256"]:
        raise ValueError("pilot dataset changed after its preparation")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "pilot_comparison.json"
    if path.exists():
        raise FileExistsError("pilot outputs are immutable; choose a new output directory")
    frame = pd.read_csv(dataset)
    config = json.loads((source / "dataset_config.json").read_text(encoding="utf-8"))
    result = compare_pilot_candidates(
        frame, preparation["feature_columns"],
        preservation_rule=PreservationRule(**config["preservation"]),
    )
    result["dataset_sha256"] = preparation["dataset_sha256"]
    result["preparation_report"] = str(source / "dataset_build_report.json")
    result["source_batch"] = preparation["batch_dir"]
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return result
