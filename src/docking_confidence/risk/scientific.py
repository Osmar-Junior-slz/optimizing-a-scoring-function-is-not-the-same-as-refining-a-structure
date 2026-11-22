"""Exact selective-risk envelopes; no risk tolerance or threshold is invented.

LTT, sections 2.1/2.3/3.2 (arXiv:2110.01052): invert the exact
binomial test and apply Bonferroni to the complete declared family.
Adaptation: each complex contributes one Bernoulli adverse event if any of
its accepted runs is structurally inadequate; correlated runs are not trials.
This is conditional COMPLEX risk, not pooled per-run risk or a pose-level CI.
"""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
import yaml

from docking_confidence.risk.policy import clopper_pearson_upper, threshold_grid


SCHEMA = "article_exact_selective_risk_v1"
QUALITY_CUTOFFS = (0.23, 0.49, 0.80)
METHODS = ("original", "isotonic", "venn_abers")
STAGES = ("risk_calibration", "article_test")


def load_risk_protocol(path=None):
    path = Path(path) if path is not None else Path(__file__).resolve().parents[3] / "configs/article_risk_protocol.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    return validate_risk_protocol(config)


def validate_risk_protocol(config):
    if not isinstance(config, dict) or config.get("schema_version") != SCHEMA:
        raise ValueError("invalid scientific risk protocol")
    expected = {"method": "learn_then_test_exact_binomial_bonferroni_envelope", "mode": "risk_coverage_only",
        "independent_unit": "complex_fingerprint_sha256", "primary_outcome": "reduced_acceptable_no_dockq_class_loss_v1",
        "missing_quality": "adverse_for_upper_bound"}
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("unsupported scientific risk protocol")
    if config.get("maximum_accepted_error_rate") is not None:
        raise ValueError("risk tolerance has not been approved; this protocol only reports risk/coverage")
    if config.get("family_methods") != list(METHODS) or config.get("family_stages") != list(STAGES):
        raise ValueError("declare all three methods and both stages before evaluating outcomes")
    if config.get("quality_cutoffs") != list(QUALITY_CUTOFFS):
        raise ValueError("use the cited DockQ cutoffs; do not optimize structural classes on test data")
    confidence = config.get("confidence_level")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 < confidence < 1:
        raise ValueError("confidence_level must be strictly between 0 and 1")
    low, high, step = (config.get(key) for key in ("threshold_min", "threshold_max", "threshold_step"))
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in (low, high, step)):
        raise ValueError("threshold grid must be finite numeric values")
    if not 0 <= low <= high < 1 or not 0 < step <= 1 or (high - low) / step > 1000:
        raise ValueError("invalid predeclared threshold grid")
    return config


def structural_outcomes(frame):
    """Research endpoint based on published classes; not a noninferiority margin.

    DockQ remains continuous in the report. Equality of quality class does not
    imply identical structures, equal DockQ, or biological equivalence.
    """
    reduced = pd.to_numeric(frame["dockq_reduced"], errors="raise").to_numpy(dtype=float)
    full = pd.to_numeric(frame["dockq_full"], errors="raise").to_numpy(dtype=float)
    for values in (reduced, full):
        if np.isinf(values).any() or ((values[np.isfinite(values)] < 0) | (values[np.isfinite(values)] > 1)).any():
            raise ValueError("DockQ must be in [0,1] or unavailable")
    known = np.isfinite(reduced) & np.isfinite(full)
    reduced_class = np.searchsorted(QUALITY_CUTOFFS, reduced, side="right")
    full_class = np.searchsorted(QUALITY_CUTOFFS, full, side="right")
    safe = known & (reduced_class >= 1) & (reduced_class >= full_class)
    # A known incorrect REDUCED is inadequate even when FULL is unavailable.
    known_incorrect_reduced = np.isfinite(reduced) & (reduced < QUALITY_CUTOFFS[0])
    unknown = ~known & ~known_incorrect_reduced
    return pd.DataFrame({"structural_quality_preserved": safe,
        "quality_unknown": unknown,
        "dockq_class_reduced": np.where(np.isfinite(reduced), reduced_class, -1),
        "dockq_class_full": np.where(np.isfinite(full), full_class, -1)}, index=frame.index)


def risk_coverage_envelope(frame, scores, eligible, config, *, stage, all_complexes=None):
    config = validate_risk_protocol(config)
    if stage not in config["family_stages"] or set(scores) != set(config["family_methods"]):
        raise ValueError("risk family differs from the protocol")
    if "complex_fingerprint_sha256" not in frame:
        raise ValueError("risk evaluation requires canonical complex fingerprints")
    fingerprints = frame["complex_fingerprint_sha256"]
    if fingerprints.isna().any() or fingerprints.astype(str).str.strip().eq("").any():
        raise ValueError("missing molecular grouping identifiers")
    groups = fingerprints.astype(str).to_numpy()
    universe = set(groups) if all_complexes is None else set(all_complexes)
    if any(not isinstance(value, str) or not value.strip() for value in universe) or not set(groups) <= universe:
        raise ValueError("complex universe must include all scored complexes")
    eligible = np.asarray(eligible).reshape(-1)
    if len(eligible) != len(frame) or not np.isin(eligible, [0, 1]).all():
        raise ValueError("eligibility must contain aligned binary flags")
    eligible = eligible.astype(bool)
    outcome = structural_outcomes(frame)
    safe = outcome.structural_quality_preserved.to_numpy(dtype=bool)
    unknown = outcome.quality_unknown.to_numpy(dtype=bool)
    candidates = threshold_grid(config)
    family_size = len(candidates) * len(config["family_methods"]) * len(config["family_stages"])
    error = (1 - config["confidence_level"]) / family_size
    curves = {}
    for method in config["family_methods"]:
        score = np.asarray(scores[method], dtype=float).reshape(-1)
        if len(score) != len(frame) or not np.isfinite(score).all() or ((score < 0) | (score > 1)).any():
            raise ValueError("risk scores must be aligned finite values in [0,1]")
        rows = []
        for threshold in candidates:
            accepted = eligible & (score >= threshold)
            accepted_groups = set(groups[accepted])
            observed_bad = set(groups[accepted & ~safe & ~unknown])
            unknown_groups = set(groups[accepted & unknown])
            adverse = observed_bad | unknown_groups
            count, errors = len(accepted_groups), len(adverse)
            rows.append({"threshold": float(threshold), "accepted_runs": int(accepted.sum()),
                "false_accepted_runs_observed": int((accepted & ~safe & ~unknown).sum()),
                "accepted_runs_quality_unknown": int((accepted & unknown).sum()),
                "total_complexes": len(universe), "accepted_complexes": count,
                "observed_adverse_complexes": len(observed_bad), "unknown_accepted_complexes": len(unknown_groups),
                "worst_case_adverse_complexes": errors,
                "empirical_complex_coverage": count / len(universe) if universe else None,
                "empirical_worst_case_complex_risk": errors / count if count else None,
                "simultaneous_complex_risk_upper_bound": clopper_pearson_upper(errors, count, error) if count else None,
                "risk_status": "available" if count else "no_accepted_complexes"})
        curves[method] = rows
    return {"schema_version": SCHEMA, "stage": stage, "method": config["method"],
        "primary_outcome": config["primary_outcome"], "quality_cutoffs": config["quality_cutoffs"],
        "maximum_accepted_error_rate": None, "selected_threshold": None, "decision_authorized": False,
        "selection_status": "no_approved_risk_tolerance; report_curves_only",
        "loss_definition": "probability of at least one structurally inadequate accepted run in a complex, conditional on at least one accepted run in that complex",
        "family_size": family_size, "confidence_level": config["confidence_level"],
        "error_probability_per_bound": error, "curve_confidence_scope": "simultaneous risk upper bounds over all methods, thresholds and both declared stages",
        "coverage_scope": "empirical complex coverage; no confidence guarantee for coverage",
        "unknown_quality": "counted as adverse for the upper bound; never assigned a fabricated DockQ",
        "score_scope": "frozen scores; Venn-Abers lower multiprobability endpoint is not a structural-quality confidence interval",
        "assumptions": "independent representative complexes; fixed scoring, eligibility, loss, grid and within-complex protocol independently of the relevant stage outcomes; no guarantee under adaptive reuse, homology dependence or distribution shift",
        "curves": curves, "references": config.get("references", {})}


def zero_error_sample_requirements(config, risk_levels=(0.05, 0.10, 0.20)):
    """Illustrative planning levels, not approved tolerances or universal minima."""
    config = validate_risk_protocol(config)
    family_size = len(threshold_grid(config)) * len(METHODS) * len(STAGES)
    tail = (1 - config["confidence_level"]) / family_size
    rows = []
    for risk in risk_levels:
        if isinstance(risk, bool) or not 0 < risk < 1:
            raise ValueError("planning risk levels must be in (0,1)")
        n = int(math.ceil(math.log(tail) / math.log1p(-risk)))
        rows.append({"illustrative_risk_level": risk, "minimum_accepted_independent_complexes_if_zero_errors": n,
                     "approved_tolerance": False})
    return rows
