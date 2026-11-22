from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np

from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.io.config import (
    ARTIFACT_SCHEMA_VERSION, CONFIG_DIGEST_SCHEMA, config_sha256, validate_config,
)
from docking_confidence.risk.policy import (
    assess_cluster_counts,
    binomial_cdf,
    clopper_pearson_lower,
    clopper_pearson_upper,
    threshold_grid,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GATE_NAMES = {
    "profile_allows_automatic_decision",
    "scientific_provenance_passed",
    "risk_calibration_passed",
    "locked_test_passed",
    "runtime_gate_passed",
}


def _mapping(parent: dict, key: str, context: str = "artifact") -> dict:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} field {key!r} must be a JSON object")
    return value


def _strict_bool(parent: dict, key: str, context: str) -> bool:
    value = parent.get(key)
    if type(value) is not bool:
        raise ValueError(f"{context} field {key!r} must be boolean")
    return value


def _finite_number(parent: dict, key: str, context: str) -> float:
    value = parent.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{context} field {key!r} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} field {key!r} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{context} field {key!r} must be a finite number")
    return numeric


def _strict_int(parent: dict, key: str, context: str, *, minimum: int = 0) -> int:
    value = parent.get(key)
    if type(value) is not int or value < minimum:
        raise ValueError(f"{context} field {key!r} must be an integer >= {minimum}")
    return value


def _same_number(actual: float, expected: float, context: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{context} disagrees with resolved configuration or evidence")


def _validate_profile_contract(config: dict) -> None:
    authorization = config["authorization"]
    profile = authorization["profile"]
    automatic = authorization["allow_automatic_decision"]
    if profile in {"pilot_graph_v2", "demo_non_scientific"} and automatic:
        raise ValueError(f"authorization profile {profile!r} cannot enable automatic decisions")
    if automatic:
        if profile != "production_graph_v2":
            raise ValueError("automatic decisions require the production_graph_v2 profile")
        if not authorization["require_scientific_provenance"]:
            raise ValueError("automatic decisions require scientific provenance")
        graph = config["contact_graph"]
        if not graph["enabled"] or not graph["require_pose_provenance"]:
            raise ValueError("automatic decisions require the provenance-bound contact graph")
        runtime = config["runtime_gate"]
        if not runtime["enabled"] or not runtime["require_reduced_faster_than_full"]:
            raise ValueError("automatic decisions require the runtime speedup gate")
        if not config["applicability_domain"]["block_feature_extrapolation"]:
            raise ValueError("automatic decisions require extrapolation blocking")


def _validate_policy_assessment(
    assessment: dict,
    cfg: dict,
    *,
    hypothesis_count: int,
    role: str,
    role_row_count: int,
    role_complex_count: int,
    expected_threshold: float,
    expected_groups: list[str] | None = None,
) -> bool:
    context = f"artifact {role} assessment"
    if cfg.get("method") == "cluster_bootstrap_run_risk":
        records = assessment.get("cluster_counts")
        if not isinstance(records, list) or len(records) != role_complex_count:
            raise ValueError(f"{context} cluster evidence disagrees with the frozen split")
        expected = assess_cluster_counts(
            records, expected_threshold, cfg=cfg,
            hypothesis_count=hypothesis_count, role=role,
        )
        if expected_groups is None or {
            record["complex_fingerprint_sha256"] for record in records
        } != set(expected_groups):
            raise ValueError(f"{context} cluster identities disagree with the frozen split")
        if sum(record["total_run_count"] for record in records) != role_row_count:
            raise ValueError(f"{context} cluster rows disagree with the frozen split")
        for key, value in expected.items():
            actual = assessment.get(key)
            if isinstance(value, float):
                _same_number(_finite_number(assessment, key, context), value, f"{context} {key}")
            elif type(actual) is not type(value) or actual != value:
                raise ValueError(f"{context} {key} disagrees with cluster evidence")
        return expected["supported"]
    if assessment.get("role") != role:
        raise ValueError(f"{context} role is invalid")
    supported = _strict_bool(assessment, "supported", context)
    threshold = _finite_number(assessment, "threshold", context)
    _same_number(threshold, expected_threshold, f"{context} threshold")
    if _strict_int(assessment, "tested_hypothesis_count", context, minimum=1) != hypothesis_count:
        raise ValueError(f"{context} hypothesis count is invalid")

    total_complexes = _strict_int(assessment, "total_complex_count", context, minimum=1)
    if total_complexes != role_complex_count:
        raise ValueError(f"{context} complex count disagrees with the frozen split")
    accepted_complexes = _strict_int(
        assessment, "accepted_complex_count", context, minimum=0
    )
    adverse_complexes = _strict_int(
        assessment, "adverse_accepted_complex_count", context, minimum=0
    )
    accepted_rows = _strict_int(assessment, "accepted_row_count", context, minimum=0)
    eligible_rows = _strict_int(assessment, "eligible_row_count", context, minimum=0)
    false_rows = _strict_int(
        assessment, "false_acceptance_row_count", context, minimum=0
    )
    if (
        accepted_complexes > total_complexes
        or adverse_complexes > accepted_complexes
        or accepted_rows > role_row_count
        or eligible_rows > role_row_count
        or accepted_rows > eligible_rows
        or false_rows > accepted_rows
    ):
        raise ValueError(f"{context} counts are inconsistent")

    confidence = float(cfg["familywise_confidence_level"])
    risk_fraction = float(cfg["risk_error_budget_fraction"])
    total_error = 1.0 - confidence
    risk_error = total_error * risk_fraction / hypothesis_count
    coverage_error = total_error * (1.0 - risk_fraction) / hypothesis_count
    risk_target = float(cfg["risk_target"])
    for key, expected in (
        ("risk_target", risk_target),
        ("familywise_confidence_level", confidence),
        ("risk_error_probability_per_hypothesis", risk_error),
        ("coverage_error_probability_per_hypothesis", coverage_error),
        ("minimum_complex_coverage", float(cfg["minimum_complex_coverage"])),
    ):
        _same_number(_finite_number(assessment, key, context), expected, f"{context} {key}")
    if (
        _strict_int(assessment, "minimum_evaluated_complexes", context, minimum=1)
        != int(cfg["minimum_evaluated_complexes"])
    ):
        raise ValueError(f"{context} minimum evaluated complex count is invalid")
    if (
        _strict_int(assessment, "minimum_authorized_complexes", context, minimum=1)
        != int(cfg["minimum_authorized_complexes"])
    ):
        raise ValueError(f"{context} minimum authorized complex count is invalid")

    if accepted_complexes:
        empirical = adverse_complexes / accepted_complexes
        risk_p = binomial_cdf(adverse_complexes, accepted_complexes, risk_target)
        risk_upper = clopper_pearson_upper(
            adverse_complexes, accepted_complexes, risk_error
        )
        _same_number(
            _finite_number(assessment, "empirical_conditional_complex_risk", context),
            empirical,
            f"{context} empirical risk",
        )
    else:
        if assessment.get("empirical_conditional_complex_risk") is not None:
            raise ValueError(f"{context} empirical risk must be null with no acceptances")
        risk_p = 1.0
        risk_upper = 1.0
    coverage = accepted_complexes / total_complexes
    coverage_lower = clopper_pearson_lower(
        accepted_complexes, total_complexes, coverage_error
    )
    row_coverage = accepted_rows / role_row_count
    for key, expected in (
        ("risk_p_value", risk_p),
        ("risk_p_value_bonferroni", min(1.0, risk_p * hypothesis_count)),
        ("simultaneous_risk_upper_bound", risk_upper),
        ("complex_coverage", coverage),
        ("simultaneous_complex_coverage_lower_bound", coverage_lower),
        ("row_coverage", row_coverage),
    ):
        _same_number(_finite_number(assessment, key, context), expected, f"{context} {key}")

    expected_supported = (
        risk_p <= risk_error
        and coverage_lower >= float(cfg["minimum_complex_coverage"])
        and total_complexes >= int(cfg["minimum_evaluated_complexes"])
        and accepted_complexes >= int(cfg["minimum_authorized_complexes"])
    )
    if supported != expected_supported:
        raise ValueError(f"{context} supported flag disagrees with its exact evidence")
    return supported


def validate_artifact(payload: dict) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("artifact must be a JSON object")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"artifact schema_version must be {ARTIFACT_SCHEMA_VERSION!r}"
        )
    if payload.get("artifact_type") != "auditable_docking_confidence_gate":
        raise ValueError("unsupported artifact_type")
    target = _mapping(payload, "target")
    if target.get("name") != "preserved" or target.get("positive_class") != 1:
        raise ValueError("artifact target orientation is invalid")

    features = payload.get("feature_columns")
    if not isinstance(features, list) or not features or len(features) != len(set(features)):
        raise ValueError("artifact feature_columns is invalid")
    if any(
        not isinstance(feature, str) or not feature or feature != feature.strip()
        for feature in features
    ):
        raise ValueError("artifact contains an invalid feature name")
    dimension = len(features)

    resolved_config = validate_config(payload.get("resolved_config"))
    development_only = payload.get("development_only", False)
    if not isinstance(development_only, bool) or (development_only and (
            resolved_config["authorization"]["allow_automatic_decision"] or payload.get("decision_capable") is not False)):
        raise ValueError("development-only artifacts cannot authorize automatic decisions")
    _validate_profile_contract(resolved_config)
    if resolved_config["features"]["columns"] != features:
        raise ValueError("artifact features disagree with resolved configuration")
    if payload.get("feature_provenance") != resolved_config["features"]["provenance"]:
        raise ValueError("artifact feature provenance disagrees with configuration")

    context_fields = ("docking_task", "feature_schema", "training_config_source")
    if "docking_task" in resolved_config or any(key in payload for key in context_fields):
        if payload.get("docking_task") != resolved_config.get("docking_task") or not payload.get("docking_task"):
            raise ValueError("artifact docking_task disagrees with resolved configuration")
        if payload.get("feature_schema") != resolved_config["features"]["schema_version"]:
            raise ValueError("artifact feature_schema disagrees with resolved configuration")
        source = _mapping(payload, "training_config_source")
        source_path = source.get("path")
        if not isinstance(source_path, str) or not source_path.strip() or source_path != source_path.strip():
            raise ValueError("artifact training configuration source path is invalid")
        if source.get("digest_schema") != CONFIG_DIGEST_SCHEMA:
            raise ValueError("artifact training configuration digest schema is unsupported")
        if source.get("sha256") != config_sha256(resolved_config):
            raise ValueError("artifact training configuration digest disagrees with resolved configuration")

    base = _mapping(payload, "base_model")
    if base.get("method") != "standardized_logistic_regression_v1":
        raise ValueError("unsupported base-model method")
    if _strict_int(base, "positive_class", "artifact base model", minimum=0) != 1:
        raise ValueError("artifact base-model positive class is invalid")
    if base.get("score_orientation") != "larger_means_more_likely_preserved":
        raise ValueError("artifact base-model score orientation is invalid")
    if base.get("formula") != "raw_score = intercept + sum(((x_j - mean_j) / scale_j) * coef_j)":
        raise ValueError("artifact base-model formula is invalid")
    if base.get("solver") != "lbfgs":
        raise ValueError("artifact base-model solver is invalid")
    for key in ("scaler_mean", "scaler_scale", "coef"):
        try:
            values = np.asarray(base.get(key), dtype=float).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"artifact base-model vector {key} is invalid") from exc
        if len(values) != dimension or not np.isfinite(values).all():
            raise ValueError(f"artifact base-model vector {key} is invalid")
        if key == "scaler_scale" and (values <= 0).any():
            raise ValueError("artifact scaler scales must be positive")
    intercept = _finite_number(base, "intercept", "artifact base model")
    base_features = base.get("feature_columns")
    if base_features != features:
        raise ValueError("artifact base-model feature order disagrees with artifact")

    model_cfg = resolved_config["model"]
    _same_number(
        _finite_number(base, "C", "artifact base model"),
        float(model_cfg["C"]),
        "artifact base-model C",
    )
    expected_weight = model_cfg.get("class_weight")
    if expected_weight == "none":
        expected_weight = None
    if base.get("class_weight") != expected_weight:
        raise ValueError("artifact base-model class weight disagrees with configuration")
    max_iter = _strict_int(base, "max_iter", "artifact base model", minimum=1)
    if max_iter != int(model_cfg["max_iter"]):
        raise ValueError("artifact base-model max_iter disagrees with configuration")
    n_iter = _strict_int(base, "n_iter", "artifact base model", minimum=1)
    if n_iter > max_iter:
        raise ValueError("artifact base-model iteration count is invalid")
    mean = np.asarray(base["scaler_mean"], dtype=float)
    scale = np.asarray(base["scaler_scale"], dtype=float)
    coefficients = np.asarray(base["coef"], dtype=float)
    expected_original_coef = coefficients / scale
    expected_original_intercept = intercept - float(
        np.sum(coefficients * mean / scale)
    )
    try:
        original_coef = np.asarray(base.get("original_scale_coef"), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact original-scale coefficients are invalid") from exc
    if (
        len(original_coef) != dimension
        or not np.isfinite(original_coef).all()
        or not np.allclose(original_coef, expected_original_coef, rtol=0.0, atol=1e-12)
    ):
        raise ValueError("artifact original-scale coefficients disagree with base model")
    _same_number(
        _finite_number(base, "original_scale_intercept", "artifact base model"),
        expected_original_intercept,
        "artifact original-scale intercept",
    )

    calibration = _mapping(payload, "calibration")
    VennAbersCalibrator.from_export(calibration)
    if calibration.get("method") != "inductive_venn_abers_pav_v1":
        raise ValueError("artifact calibration method is invalid")
    calibration_count = _strict_int(
        calibration, "calibration_example_count", "artifact calibration", minimum=2
    )
    calibration_cfg = resolved_config["calibration"]
    independent_complexes = _strict_int(
        calibration, "independent_complex_count", "artifact calibration", minimum=2
    )
    positive_complexes = _strict_int(
        calibration, "positive_class_complex_count", "artifact calibration", minimum=1
    )
    negative_complexes = _strict_int(
        calibration, "negative_class_complex_count", "artifact calibration", minimum=1
    )
    if (
        independent_complexes > calibration_count
        or positive_complexes > independent_complexes
        or negative_complexes > independent_complexes
        or (not development_only and (
            independent_complexes < int(calibration_cfg["minimum_independent_complexes"])
            or positive_complexes < int(calibration_cfg["minimum_positive_class_complexes"])
            or negative_complexes < int(calibration_cfg["minimum_negative_class_complexes"])))
    ):
        raise ValueError("artifact calibration complex-level evidence is inconsistent")
    calibration_evidence_passed = (independent_complexes >= int(calibration_cfg["minimum_independent_complexes"])
        and positive_complexes >= int(calibration_cfg["minimum_positive_class_complexes"])
        and negative_complexes >= int(calibration_cfg["minimum_negative_class_complexes"]))
    if "calibration_evidence_passed" in payload and payload["calibration_evidence_passed"] is not calibration_evidence_passed:
        raise ValueError("artifact calibration evidence status disagrees with its counts")

    domain = _mapping(payload, "applicability_domain")
    try:
        lower = np.asarray(domain.get("lower"), dtype=float).reshape(-1)
        upper = np.asarray(domain.get("upper"), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact applicability domain is invalid") from exc
    if (
        len(lower) != dimension
        or len(upper) != dimension
        or not np.isfinite(lower).all()
        or not np.isfinite(upper).all()
        or (lower > upper).any()
    ):
        raise ValueError("artifact applicability domain is invalid")

    domain_cfg = resolved_config["applicability_domain"]
    _same_number(
        _finite_number(domain, "lower_quantile", "artifact applicability domain"),
        float(domain_cfg["lower_quantile"]),
        "artifact applicability-domain lower quantile",
    )
    _same_number(
        _finite_number(domain, "upper_quantile", "artifact applicability domain"),
        float(domain_cfg["upper_quantile"]),
        "artifact applicability-domain upper quantile",
    )
    block_extrapolation = _strict_bool(
        domain, "block_feature_extrapolation", "artifact applicability domain"
    )
    if block_extrapolation != domain_cfg["block_feature_extrapolation"]:
        raise ValueError("artifact extrapolation policy disagrees with configuration")

    preservation = _mapping(payload, "preservation_rule")
    for key in ("minimum_reduced_dockq", "maximum_dockq_loss"):
        _same_number(
            _finite_number(preservation, key, "artifact preservation rule"),
            float(resolved_config["preservation"][key]),
            f"artifact preservation rule {key}",
        )

    graph = _mapping(payload, "contact_graph")
    if graph != resolved_config["contact_graph"]:
        raise ValueError("artifact graph contract disagrees with resolved configuration")

    known = payload.get("known_development_complex_ids")
    if (
        not isinstance(known, list)
        or any(not isinstance(value, str) or not value for value in known)
        or len(known) != len(set(known))
        or known != sorted(known)
    ):
        raise ValueError("artifact development complex list is invalid")
    fingerprints = payload.get("known_development_complex_fingerprints")
    if (
        not isinstance(fingerprints, list)
        or not fingerprints
        or any(
            not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
            for value in fingerprints
        )
        or len(fingerprints) != len(set(fingerprints))
        or fingerprints != sorted(fingerprints)
    ):
        raise ValueError("artifact development complex fingerprint list is invalid")

    split = _mapping(payload, "development_split")
    if (
        split.get("method")
        != "label_independent_assignment_with_postsplit_feasibility_gate_v1"
        or split.get("grouping_unit") != "complex_fingerprint_sha256"
        or _strict_int(split, "random_seed", "artifact development split", minimum=0)
        != int(resolved_config["validation"]["random_seed"])
        or _strict_int(split, "split_folds", "artifact development split", minimum=4)
        != int(resolved_config["validation"]["split_folds"])
    ):
        raise ValueError("artifact development split disagrees with configuration")
    roles = _mapping(split, "roles", "artifact development split")
    expected_roles = {
        "model_train",
        "probability_calibration",
        "risk_calibration",
        "locked_test",
    }
    if set(roles) != expected_roles:
        raise ValueError("artifact development split roles are invalid")
    role_group_sets: list[set[str]] = []
    all_role_groups: list[str] = []
    for role in sorted(expected_roles):
        role_payload = _mapping(roles, role, "artifact development split")
        groups = role_payload.get("groups")
        if (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
                for value in groups
            )
            or groups != sorted(groups)
            or len(groups) != len(set(groups))
        ):
            raise ValueError(f"artifact development split group list is invalid for {role}")
        if _strict_int(role_payload, "group_count", f"artifact split role {role}", minimum=1) != len(groups):
            raise ValueError(f"artifact split group count is invalid for {role}")
        digest = hashlib.sha256(
            json.dumps(groups, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if role_payload.get("groups_sha256") != digest:
            raise ValueError(f"artifact split group hash is invalid for {role}")
        current = set(groups)
        if any(current & previous for previous in role_group_sets):
            raise ValueError("artifact development split contains complex leakage")
        role_group_sets.append(current)
        all_role_groups.extend(groups)
    if sorted(all_role_groups) != fingerprints:
        raise ValueError("artifact known fingerprints disagree with development split")

    summary = _mapping(payload, "development_summary")
    n_rows = _strict_int(summary, "n_rows", "artifact development summary", minimum=1)
    if _strict_int(summary, "n_complexes", "artifact development summary", minimum=1) != len(fingerprints):
        raise ValueError("artifact development complex count is inconsistent")
    positive_count = _strict_int(
        summary, "positive_count", "artifact development summary", minimum=0
    )
    negative_count = _strict_int(
        summary, "negative_count", "artifact development summary", minimum=0
    )
    if positive_count + negative_count != n_rows:
        raise ValueError("artifact development label counts are inconsistent")
    role_row_counts_payload = _mapping(
        summary, "role_row_counts", "artifact development summary"
    )
    if set(role_row_counts_payload) != expected_roles:
        raise ValueError("artifact development row-count roles are invalid")
    role_row_counts = {
        role: _strict_int(
            role_row_counts_payload,
            role,
            "artifact development role row counts",
            minimum=1,
        )
        for role in expected_roles
    }
    if sum(role_row_counts.values()) != n_rows:
        raise ValueError("artifact development role row counts are inconsistent")
    calibration_evidence = _mapping(
        summary, "probability_calibration_evidence", "artifact development summary"
    )
    if (
        _strict_int(
            calibration_evidence,
            "independent_complex_count",
            "artifact probability-calibration evidence",
            minimum=2,
        )
        != independent_complexes
        or _strict_int(
            calibration_evidence,
            "positive_class_complex_count",
            "artifact probability-calibration evidence",
            minimum=1,
        )
        != positive_complexes
        or _strict_int(
            calibration_evidence,
            "negative_class_complex_count",
            "artifact probability-calibration evidence",
            minimum=1,
        )
        != negative_complexes
    ):
        raise ValueError("artifact probability-calibration evidence is inconsistent")
    in_domain_rows = _strict_int(
        calibration_evidence,
        "in_domain_row_count",
        "artifact probability-calibration evidence",
        minimum=0,
    )
    fit_rows = calibration_evidence.get("fit_row_count", in_domain_rows)
    if (isinstance(fit_rows, bool) or not isinstance(fit_rows, int) or fit_rows != calibration_count
            or in_domain_rows > fit_rows
            or (calibration_cfg.get("fit_scope", "in_applicability_domain") == "in_applicability_domain" and in_domain_rows != fit_rows)):
        raise ValueError("artifact Venn-Abers row count disagrees with its evidence")

    source_code_sha256 = payload.get("source_code_sha256")
    if not isinstance(source_code_sha256, str) or _SHA256_RE.fullmatch(
        source_code_sha256
    ) is None:
        raise ValueError("artifact source-code SHA-256 is invalid")
    dataset_snapshot = _mapping(payload, "training_dataset_snapshot")
    if (
        not isinstance(dataset_snapshot.get("path"), str)
        or not dataset_snapshot["path"].strip()
        or not isinstance(dataset_snapshot.get("sha256"), str)
        or _SHA256_RE.fullmatch(dataset_snapshot["sha256"]) is None
    ):
        raise ValueError("artifact training-dataset snapshot metadata is invalid")
    _strict_int(
        dataset_snapshot,
        "size_bytes",
        "artifact training-dataset snapshot",
        minimum=1,
    )

    policy = _mapping(payload, "policy")
    expected_policy_method = (
        "cluster_bootstrap_run_risk_v1"
        if resolved_config["policy"]["method"] == "cluster_bootstrap_run_risk"
        else "learn_then_test_bonferroni_binomial_iid_v1"
    )
    if policy.get("method") != expected_policy_method:
        raise ValueError("unsupported policy method")
    if policy.get("calibration_unit") != "complex_fingerprint_sha256" or policy.get("decision_score") != "venn_abers_probability_lower":
        raise ValueError("artifact policy decision contract is invalid")
    policy_supported = _strict_bool(policy, "supported", "artifact policy")
    policy_cfg = resolved_config["policy"]
    expected_grid = threshold_grid(policy_cfg)
    try:
        artifact_grid = np.asarray(policy.get("candidate_grid"), dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact policy grid is invalid") from exc
    if not np.array_equal(expected_grid, artifact_grid):
        raise ValueError("artifact policy grid disagrees with resolved configuration")
    grid_payload = json.dumps(
        expected_grid.tolist(), separators=(",", ":")
    ).encode("ascii")
    if policy.get("candidate_grid_sha256") != hashlib.sha256(grid_payload).hexdigest():
        raise ValueError("artifact policy grid hash is invalid")
    hypothesis_count = _strict_int(
        policy, "tested_hypothesis_count", "artifact policy", minimum=1
    )
    if hypothesis_count != len(expected_grid):
        raise ValueError("artifact policy hypothesis count is invalid")
    threshold = policy.get("threshold")
    if policy_supported:
        selected_threshold = _finite_number(policy, "threshold", "artifact policy")
        if not np.equal(expected_grid, selected_threshold).any():
            raise ValueError("artifact policy threshold is not in its predeclared grid")
        if not _validate_policy_assessment(
            policy,
            policy_cfg,
            hypothesis_count=hypothesis_count,
            role="risk_calibration",
            role_row_count=role_row_counts["risk_calibration"],
            role_complex_count=len(roles["risk_calibration"]["groups"]),
            expected_threshold=selected_threshold,
            expected_groups=roles["risk_calibration"]["groups"],
        ):
            raise ValueError("artifact supported policy does not satisfy its evidence")
    else:
        if threshold is not None:
            raise ValueError("unsupported artifact policy must not authorize a threshold")
        selected_threshold = None
        if not isinstance(policy.get("best_observed_candidate"), dict):
            raise ValueError("unsupported artifact policy lacks diagnostic evidence")
        if expected_policy_method == "cluster_bootstrap_run_risk_v1":
            diagnostic = policy["best_observed_candidate"]
            diagnostic_threshold = _finite_number(diagnostic, "threshold", "artifact diagnostic")
            if not np.equal(expected_grid, diagnostic_threshold).any():
                raise ValueError("diagnostic threshold is outside the configured candidates")
            if _validate_policy_assessment(
                diagnostic, policy_cfg, hypothesis_count=hypothesis_count,
                role="risk_calibration", role_row_count=role_row_counts["risk_calibration"],
                role_complex_count=len(roles["risk_calibration"]["groups"]),
                expected_threshold=diagnostic_threshold,
                expected_groups=roles["risk_calibration"]["groups"],
            ):
                raise ValueError("unsupported policy contains a supported diagnostic")

    locked = _mapping(payload, "locked_test_audit")
    if selected_threshold is None:
        if (
            locked.get("role") != "locked_test"
            or _strict_bool(locked, "supported", "artifact locked-test audit")
            or locked.get("threshold") is not None
        ):
            raise ValueError("artifact locked-test audit is inconsistent with policy")
        locked_supported = False
    else:
        locked_supported = _validate_policy_assessment(
            locked,
            policy_cfg,
            hypothesis_count=1,
            role="locked_test",
            role_row_count=role_row_counts["locked_test"],
            role_complex_count=len(roles["locked_test"]["groups"]),
            expected_threshold=selected_threshold,
            expected_groups=roles["locked_test"]["groups"],
        )

    runtime = _mapping(payload, "runtime_gate")
    runtime_cfg = resolved_config["runtime_gate"]
    runtime_enabled = _strict_bool(runtime, "enabled", "artifact runtime gate")
    if runtime_enabled != runtime_cfg["enabled"]:
        raise ValueError("artifact runtime enabled flag disagrees with configuration")
    assessed_rows = _strict_int(
        runtime, "assessed_row_count", "artifact runtime gate", minimum=0
    )
    full_runtime = _finite_number(runtime, "full_runtime_sec", "artifact runtime gate")
    reduced_runtime = _finite_number(runtime, "reduced_runtime_sec", "artifact runtime gate")
    speedup = _finite_number(runtime, "speedup_full_over_reduced", "artifact runtime gate")
    minimum_speedup = _finite_number(runtime, "minimum_speedup", "artifact runtime gate")
    if full_runtime < 0 or reduced_runtime < 0 or speedup < 0:
        raise ValueError("artifact runtime evidence cannot be negative")
    _same_number(minimum_speedup, float(runtime_cfg["minimum_speedup"]), "artifact runtime minimum speedup")
    total_cascade = runtime_cfg.get("assessment_scope") == "total_paired_cascade"
    if total_cascade:
        if runtime.get("assessment_scope") != "locked_test_total_paired_cascade_v1":
            raise ValueError("artifact cascade runtime scope disagrees with configuration")
        if runtime.get("overhead_column") != runtime_cfg["overhead_column"]:
            raise ValueError("artifact cascade overhead column disagrees with configuration")
        if assessed_rows != role_row_counts["locked_test"]:
            raise ValueError("cascade runtime must assess every locked paired row")
        accepted_rows = _strict_int(runtime, "accepted_row_count", "artifact cascade runtime", minimum=0)
        if accepted_rows > assessed_rows:
            raise ValueError("cascade accepted rows exceed assessed rows")
        measured_overhead = _strict_bool(runtime, "overhead_measured", "artifact cascade runtime")
        overhead = _finite_number(runtime, "overhead_sec", "artifact cascade runtime") if measured_overhead else 0.0
        if overhead < 0 or (not measured_overhead and runtime.get("overhead_sec") is not None):
            raise ValueError("invalid cascade runtime overhead evidence")
        fallback_runtime = _finite_number(runtime, "fallback_full_sec", "artifact cascade runtime")
        if fallback_runtime < 0 or fallback_runtime > full_runtime + 1e-9:
            raise ValueError("invalid cascade fallback runtime")
        cascade_runtime = _finite_number(runtime, "cascade_runtime_sec", "artifact cascade runtime")
        _same_number(cascade_runtime, reduced_runtime + fallback_runtime + overhead, "cascade runtime accounting")
        _same_number(runtime["saving_upper_bound_sec"], full_runtime - cascade_runtime, "cascade saving bound")
        if measured_overhead:
            _same_number(runtime["measured_saving_sec"], full_runtime - cascade_runtime, "cascade measured saving")
        elif runtime.get("measured_saving_sec") is not None:
            raise ValueError("unknown overhead cannot produce measured savings")
        expected_speedup = full_runtime / cascade_runtime if cascade_runtime > 0 else 0.0
        _same_number(runtime["speedup_full_over_strategy"], expected_speedup, "cascade strategy speedup")
    else:
        expected_speedup = full_runtime / reduced_runtime if reduced_runtime > 0 else 0.0
    _same_number(speedup, expected_speedup, "artifact runtime speedup")
    runtime_passed = _strict_bool(runtime, "passed", "artifact runtime gate")
    expected_runtime_passed = assessed_rows > 0
    if runtime_cfg["enabled"] and runtime_cfg["require_reduced_faster_than_full"]:
        expected_runtime_passed = expected_runtime_passed and speedup >= minimum_speedup
    elif not runtime_cfg["enabled"]:
        expected_runtime_passed = False
    if total_cascade:
        expected_runtime_passed = (expected_runtime_passed and accepted_rows > 0 and cascade_runtime > 0
                                   and full_runtime > cascade_runtime
                                   and (measured_overhead or not runtime_cfg["require_measured_overhead"]))
    if runtime_passed != expected_runtime_passed:
        raise ValueError("artifact runtime gate flag disagrees with runtime evidence")

    provenance = _mapping(payload, "dataset_provenance")
    require_scientific = _strict_bool(
        provenance, "require_scientific_provenance", "artifact dataset provenance"
    )
    if require_scientific != resolved_config["authorization"]["require_scientific_provenance"]:
        raise ValueError("artifact provenance requirement disagrees with configuration")
    if provenance.get("provenance_kind") != "operator_attestation_not_cryptographic_authentication":
        raise ValueError("artifact dataset provenance kind is invalid")
    if provenance.get("trusted_engine_allowlist") != sorted(
        resolved_config["authorization"]["trusted_engine_allowlist"]
    ):
        raise ValueError("artifact engine allowlist disagrees with configuration")
    scientific_rows = _strict_int(
        provenance, "scientific_row_count", "artifact dataset provenance", minimum=0
    )
    provenance_rows = _strict_int(
        provenance, "total_row_count", "artifact dataset provenance", minimum=1
    )
    allowlisted_rows = _strict_int(
        provenance,
        "allowlisted_engine_row_count",
        "artifact dataset provenance",
        minimum=0,
    )
    trusted_rows = _strict_int(
        provenance,
        "trusted_scientific_row_count",
        "artifact dataset provenance",
        minimum=0,
    )
    if (
        provenance_rows != n_rows
        or scientific_rows > provenance_rows
        or allowlisted_rows > provenance_rows
        or trusted_rows > scientific_rows
        or trusted_rows > allowlisted_rows
    ):
        raise ValueError("artifact dataset provenance counts are inconsistent")
    provenance_passed = _strict_bool(provenance, "passed", "artifact dataset provenance")
    expected_provenance_passed = trusted_rows == provenance_rows or not require_scientific
    if provenance_passed != expected_provenance_passed:
        raise ValueError("artifact provenance gate disagrees with row evidence")

    authorization = _mapping(payload, "authorization")
    decision_capable = _strict_bool(payload, "decision_capable", "artifact")
    authorization_decision = _strict_bool(
        authorization, "decision_capable", "artifact authorization"
    )
    if authorization_decision != decision_capable:
        raise ValueError("artifact authorization flags disagree")
    authorization_cfg = resolved_config["authorization"]
    if authorization.get("profile") != authorization_cfg["profile"]:
        raise ValueError("artifact authorization profile disagrees with configuration")
    gates = _mapping(authorization, "gates", "artifact authorization")
    if set(gates) != _GATE_NAMES:
        raise ValueError("artifact authorization gates are incomplete or unsupported")
    expected_gates = {
        "profile_allows_automatic_decision": authorization_cfg["allow_automatic_decision"],
        "scientific_provenance_passed": provenance_passed,
        "risk_calibration_passed": policy_supported,
        "locked_test_passed": locked_supported,
        "runtime_gate_passed": runtime_passed,
    }
    for name, expected in expected_gates.items():
        actual = _strict_bool(gates, name, "artifact authorization gates")
        if actual != expected:
            raise ValueError(f"artifact authorization gate {name!r} disagrees with evidence")
    if decision_capable != all(expected_gates.values()):
        raise ValueError("artifact decision_capable disagrees with its authorization gates")
    if decision_capable and not block_extrapolation:
        raise ValueError("decision-capable artifact must block extrapolation")
    return payload
