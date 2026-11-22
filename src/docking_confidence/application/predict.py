from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd

from docking_confidence.application.runtime import runtime_assessment
from docking_confidence.artifact.schema import validate_artifact
from docking_confidence.audit.manifest import (
    read_bytes_once,
    read_verified_bundle,
    sha256_bytes,
)
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.io.dataset import (
    _validate_training_context, load_reduced_inference_dataset,
)
from docking_confidence.risk.policy import (
    audit_locked_policy,
    operational_acceptance_mask,
    select_policy,
)


_REQUIRED_BUNDLE_FILES = {
    "artifact.json",
    "validation_report.json",
    "probability_calibration_predictions.csv",
    "risk_calibration_predictions.csv",
    "locked_test_predictions.csv",
    "policy_candidates.csv",
    "resolved_config.json",
}


def _json_from_bytes(payload: bytes, description: str) -> dict:
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{description} must be valid UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{description} must be a JSON object")
    return decoded


def _boolean_array(series: pd.Series, description: str) -> np.ndarray:
    if series.dtype == bool:
        return series.to_numpy(dtype=bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError(f"{description} must contain explicit booleans")
    return mapped.to_numpy(dtype=bool)


def _validate_verified_bundle_evidence(artifact: dict, artifact_file: Path, bundle) -> None:
    expected_paths = {
        (artifact_file.parent / name).resolve() for name in _REQUIRED_BUNDLE_FILES
    }
    missing = sorted(str(path.name) for path in expected_paths if path not in bundle.files)
    if missing:
        raise ValueError(f"audit manifest is missing required bundle files: {missing}")
    snapshot = artifact["training_dataset_snapshot"]
    source_matches = [
        entry
        for entry in bundle.source_inputs.values()
        if entry.get("sha256") == snapshot["sha256"]
        and entry.get("size_bytes") == snapshot["size_bytes"]
    ]
    if len(source_matches) != 1:
        raise ValueError(
            "audit manifest source snapshot disagrees with the artifact training dataset"
        )

    resolved_config = _json_from_bytes(
        bundle.bytes_for(artifact_file.parent / "resolved_config.json"),
        "resolved configuration bundle file",
    )
    if resolved_config != artifact["resolved_config"]:
        raise ValueError("bundle resolved configuration disagrees with artifact")
    report = _json_from_bytes(
        bundle.bytes_for(artifact_file.parent / "validation_report.json"),
        "validation report bundle file",
    )
    report_pairs = {
        "decision_capable": "decision_capable",
        "authorization": "authorization",
        "split": "development_split",
        "development_summary": "development_summary",
        "policy": "policy",
        "locked_test_policy_audit": "locked_test_audit",
        "runtime": "runtime_gate",
        "dataset_provenance": "dataset_provenance",
        "validity_scope": "validity_scope",
    }
    for context_key in ("docking_task", "feature_schema", "training_config_source"):
        if context_key in artifact or context_key in report:
            report_pairs[context_key] = context_key
    if report.get("schema_version") != artifact["schema_version"]:
        raise ValueError("bundle validation-report schema disagrees with artifact")
    for report_key, artifact_key in report_pairs.items():
        if report.get(report_key) != artifact.get(artifact_key):
            raise ValueError(
                f"bundle validation-report field {report_key!r} disagrees with artifact"
            )

    role_files = {
        "probability_calibration": "probability_calibration_predictions.csv",
        "risk_calibration": "risk_calibration_predictions.csv",
        "locked_test": "locked_test_predictions.csv",
    }
    expected_counts = artifact["development_summary"]["role_row_counts"]
    role_frames: dict[str, pd.DataFrame] = {}
    for role, filename in role_files.items():
        try:
            frame = pd.read_csv(
                BytesIO(bundle.bytes_for(artifact_file.parent / filename)),
                float_precision="round_trip",
            )
        except Exception as exc:
            raise ValueError(f"bundle file {filename} is not a readable CSV") from exc
        if len(frame) != int(expected_counts[role]):
            raise ValueError(f"bundle file {filename} row count disagrees with artifact")
        if "partition" not in frame or not frame["partition"].astype(str).eq(role).all():
            raise ValueError(f"bundle file {filename} partition labels are invalid")
        if "docking_task" in artifact:
            _validate_training_context(
                frame, artifact["feature_columns"],
                artifact["docking_task"], artifact["feature_schema"],
            )
            source = artifact["training_config_source"]
            if not frame["config_sha256"].eq(source["sha256"]).all() or not frame["config"].eq(source["path"]).all():
                raise ValueError(f"bundle file {filename} configuration context disagrees with artifact")
        role_frames[role] = frame

    features = artifact["feature_columns"]
    base = artifact["base_model"]
    mean = np.asarray(base["scaler_mean"], dtype=float)
    scale = np.asarray(base["scaler_scale"], dtype=float)
    coefficients = np.asarray(base["coef"], dtype=float)
    intercept = float(base["intercept"])
    calibrator = VennAbersCalibrator.from_export(artifact["calibration"])
    domain_lower = np.asarray(artifact["applicability_domain"]["lower"], dtype=float)
    domain_upper = np.asarray(artifact["applicability_domain"]["upper"], dtype=float)
    for role, frame in role_frames.items():
        required = {
            *features,
            "preserved",
            "complex_fingerprint_sha256",
            "base_raw_score",
            "probability_lower",
            "probability_upper",
            "probability",
            "in_applicability_domain",
        }
        missing_columns = sorted(required - set(frame.columns))
        if missing_columns:
            raise ValueError(
                f"bundle {role} evidence lacks required columns: {missing_columns}"
            )
        X = frame[features].to_numpy(dtype=float)
        recomputed_raw = ((X - mean) / scale) @ coefficients + intercept
        stored_raw = frame["base_raw_score"].to_numpy(dtype=float)
        # Venn-Abers deliberately treats exact score ties as one group.  Use the
        # round-tripped frozen score for the trace, while independently checking
        # that the audited features and model reproduce that score numerically.
        trace = calibrator.predict_with_trace(stored_raw)
        in_domain = ((X >= domain_lower) & (X <= domain_upper)).all(axis=1)
        checks = {
            "probability_lower": trace["lower"],
            "probability_upper": trace["upper"],
            "probability": trace["point"],
        }
        if not np.allclose(stored_raw, recomputed_raw, rtol=0.0, atol=1e-12):
            raise ValueError(
                f"bundle {role} base scores disagree with the frozen model and features"
            )
        for column, expected in checks.items():
            actual = frame[column].to_numpy(dtype=float)
            if not np.allclose(actual, expected, rtol=0.0, atol=1e-12):
                raise ValueError(
                    f"bundle {role} column {column} disagrees with the frozen artifact"
                )
        if not np.array_equal(
            _boolean_array(
                frame["in_applicability_domain"],
                f"bundle {role} applicability-domain evidence",
            ),
            in_domain,
        ):
            raise ValueError(
                f"bundle {role} applicability-domain flags disagree with the artifact"
            )

    probability_frame = role_frames["probability_calibration"]
    if "used_to_fit_venn_abers" not in probability_frame:
        raise ValueError("bundle probability-calibration evidence lacks IVAP fit flags")
    used = _boolean_array(
        probability_frame["used_to_fit_venn_abers"],
        "bundle IVAP fit flags",
    )
    expected_used = (np.ones(len(probability_frame), dtype=bool)
        if artifact["resolved_config"]["calibration"].get("fit_scope") == "all_calibration_rows"
        else _boolean_array(probability_frame["in_applicability_domain"], "bundle IVAP domain flags"))
    if not np.array_equal(used, expected_used):
        raise ValueError("bundle IVAP fit flags disagree with calibration scope")
    refitted = VennAbersCalibrator().fit(
        probability_frame.loc[used, "base_raw_score"].to_numpy(dtype=float),
        probability_frame.loc[used, "preserved"].to_numpy(dtype=int),
    ).export()
    for key in (
        "method",
        "point_rule",
        "tie_policy",
        "score_groups",
        "calibration_example_count",
        "calibration_positive_count",
    ):
        if refitted[key] != artifact["calibration"][key]:
            raise ValueError("bundle IVAP fit evidence disagrees with the artifact")

    try:
        candidates = pd.read_csv(
            BytesIO(bundle.bytes_for(artifact_file.parent / "policy_candidates.csv")),
            float_precision="round_trip",
        )
    except Exception as exc:
        raise ValueError("bundle policy_candidates.csv is not a readable CSV") from exc
    if "threshold" not in candidates:
        raise ValueError("bundle policy_candidates.csv lacks threshold evidence")
    candidate_thresholds = candidates["threshold"].to_numpy(dtype=float)
    artifact_thresholds = np.asarray(artifact["policy"]["candidate_grid"], dtype=float)
    if not np.array_equal(candidate_thresholds, artifact_thresholds):
        raise ValueError("bundle policy candidate grid disagrees with artifact")

    risk_frame = role_frames["risk_calibration"]
    risk_eligible = _boolean_array(
        risk_frame["in_applicability_domain"], "bundle risk-calibration eligibility"
    )
    recomputed_policy, recomputed_candidates = select_policy(
        risk_frame,
        risk_frame["probability_lower"].to_numpy(dtype=float),
        risk_eligible,
        artifact["resolved_config"]["policy"],
    )
    if recomputed_policy != artifact["policy"]:
        raise ValueError("bundle risk-calibration evidence does not reproduce the policy")
    candidate_fields = (
        "threshold",
        "supported",
        "total_complex_count",
        "accepted_complex_count",
        "adverse_accepted_complex_count",
        "simultaneous_risk_upper_bound",
        "simultaneous_complex_coverage_lower_bound",
    )
    if artifact["resolved_config"]["policy"]["method"] == "cluster_bootstrap_run_risk":
        candidate_fields += ("empirical_run_risk", "bootstrap_repetitions", "bootstrap_seed")
    else:
        candidate_fields += ("risk_p_value",)
    missing_candidate_fields = sorted(set(candidate_fields) - set(candidates.columns))
    if missing_candidate_fields:
        raise ValueError(
            f"bundle policy candidate evidence lacks fields: {missing_candidate_fields}"
        )
    if len(candidates) != len(recomputed_candidates):
        raise ValueError("bundle policy candidate count is inconsistent")
    for index, expected in enumerate(recomputed_candidates):
        for field in candidate_fields:
            actual = candidates.iloc[index][field]
            if isinstance(expected[field], bool):
                actual_boolean = _boolean_array(
                    candidates[field], "bundle policy candidate support flags"
                )[index]
                if bool(actual_boolean) != expected[field]:
                    raise ValueError("bundle policy candidate evidence is inconsistent")
            elif expected[field] is None:
                if not pd.isna(actual):
                    raise ValueError("bundle policy candidate evidence is inconsistent")
            elif not np.isclose(float(actual), float(expected[field]), rtol=0.0, atol=1e-12):
                raise ValueError("bundle policy candidate evidence is inconsistent")

    locked_frame = role_frames["locked_test"]
    locked_eligible = _boolean_array(
        locked_frame["in_applicability_domain"], "bundle locked-test eligibility"
    )
    threshold = artifact["policy"].get("threshold")
    recomputed_locked = audit_locked_policy(
        locked_frame,
        locked_frame["probability_lower"].to_numpy(dtype=float),
        locked_eligible,
        threshold,
        artifact["resolved_config"]["policy"],
    )
    if recomputed_locked != artifact["locked_test_audit"]:
        raise ValueError("bundle locked-test evidence does not reproduce its audit")
    accepted = (
        np.zeros(len(locked_frame), dtype=bool)
        if threshold is None
        else operational_acceptance_mask(
            locked_frame["probability_lower"].to_numpy(dtype=float),
            locked_eligible,
            float(threshold),
        )
    )
    recomputed_runtime = runtime_assessment(
        locked_frame,
        artifact["resolved_config"]["runtime_gate"],
        accepted,
    )
    if recomputed_runtime != artifact["runtime_gate"]:
        raise ValueError("bundle locked-test runtime evidence is inconsistent")


@dataclass(frozen=True)
class PredictionSession:
    """One verified model snapshot shared by the runs of an immutable batch."""

    artifact: dict
    artifact_sha256: str
    manifest_sha256: str | None
    protected_paths: frozenset[Path]

    def predict_bytes(self, input_bytes: bytes) -> pd.DataFrame:
        return _predict_reduced_snapshot(self.artifact, self.artifact_sha256, input_bytes)


def load_prediction_session(
    artifact_path: str | Path, *, require_bundle: bool = False,
) -> PredictionSession:
    artifact_file = Path(artifact_path).resolve()
    manifest_path = (artifact_file.parent / "audit_manifest.json").resolve()
    protected_paths = {artifact_file}
    if manifest_path.exists():
        bundle = read_verified_bundle(manifest_path)
        artifact_bytes = bundle.bytes_for(artifact_file)
        protected_paths.update(bundle.files)
        protected_paths.update(bundle.source_inputs)
        protected_paths.add(manifest_path)
    else:
        if require_bundle:
            raise ValueError("calibrated docking requires the verified training audit bundle")
        bundle = None
        artifact_bytes = read_bytes_once(artifact_file)
    artifact = validate_artifact(_json_from_bytes(artifact_bytes, "artifact"))
    if bundle is None and artifact["decision_capable"]:
        raise ValueError("decision-capable artifact requires its audit_manifest.json")
    if bundle is not None:
        _validate_verified_bundle_evidence(artifact, artifact_file, bundle)
    return PredictionSession(
        artifact=artifact,
        artifact_sha256=sha256_bytes(artifact_bytes),
        manifest_sha256=sha256_bytes(bundle.manifest_bytes) if bundle is not None else None,
        protected_paths=frozenset(protected_paths),
    )


def predict(artifact_path: str, input_path: str, output_path: str) -> pd.DataFrame:
    from docking_confidence.application.generalized_calibrator import SUPPORTED_SCHEMAS, predict_calibrator as predict_generalized
    schema = _json_from_bytes(read_bytes_once(artifact_path), "artifact").get("schema_version")
    if schema in SUPPORTED_SCHEMAS:
        return predict_generalized(artifact_path, input_path, output_path)
    # Explicitly retained only to replay historical IVAP probability artifacts.
    from docking_confidence.application.legacy_ivap import SCHEMA, predict_calibrator
    if schema == SCHEMA:
        return predict_calibrator(artifact_path, input_path, output_path)
    session = load_prediction_session(artifact_path)
    output_file = Path(output_path).resolve()
    if output_file in session.protected_paths or output_file == Path(input_path).resolve():
        raise ValueError("prediction output must not overwrite artifact, manifest, input, or bundle files")
    result = session.predict_bytes(read_bytes_once(Path(input_path).resolve()))
    result.to_csv(output_file, index=False)
    return result


def _predict_reduced_snapshot(
    artifact: dict, artifact_sha256: str, input_bytes: bytes,
) -> pd.DataFrame:
    features = list(artifact["feature_columns"])
    df = load_reduced_inference_dataset(
        BytesIO(input_bytes),
        features,
        contact_graph_cfg=artifact["contact_graph"],
        docking_task=artifact.get("docking_task"),
        feature_schema=artifact["resolved_config"]["features"]["schema_version"],
    )
    if "docking_task" in artifact and not df["config_sha256"].eq(
        artifact["training_config_source"]["sha256"]
    ).all():
        raise ValueError("inference configuration context disagrees with artifact")
    X = df[features].to_numpy(dtype=float)

    base = artifact["base_model"]
    mean = np.asarray(base["scaler_mean"], dtype=float)
    scale = np.asarray(base["scaler_scale"], dtype=float)
    coefficients = np.asarray(base["coef"], dtype=float)
    contributions = ((X - mean) / scale) * coefficients
    raw_score = contributions.sum(axis=1) + float(base["intercept"])
    if not np.isfinite(raw_score).all():
        raise RuntimeError("artifact produced non-finite base scores")

    calibrator = VennAbersCalibrator.from_export(artifact["calibration"])
    venn_trace = calibrator.predict_with_trace(raw_score)
    probability_lower = venn_trace["lower"]
    probability_upper = venn_trace["upper"]
    probability = venn_trace["point"]

    domain = artifact["applicability_domain"]
    lower = np.asarray(domain["lower"], dtype=float)
    upper = np.asarray(domain["upper"], dtype=float)
    in_domain = ((X >= lower) & (X <= upper)).all(axis=1)
    if "complex_fingerprint_sha256" not in df.columns:
        raise ValueError("inference loader did not produce complex_fingerprint_sha256")
    known_fingerprints = set(artifact["known_development_complex_fingerprints"])
    unseen_complex = ~df["complex_fingerprint_sha256"].astype(str).isin(
        known_fingerprints
    ).to_numpy()
    known_ids = set(artifact["known_development_complex_ids"])
    development_id_match = df["complex_id"].astype(str).isin(known_ids).to_numpy()
    authorization_config = artifact["resolved_config"]["authorization"]
    allowed_engines = set(authorization_config["trusted_engine_allowlist"])
    trusted_input = (
        df["scientific_reduced"].to_numpy(dtype=bool)
        & df["engine_reduced"].astype(str).isin(allowed_engines).to_numpy()
    )

    threshold_raw = artifact["policy"].get("threshold")
    threshold = float(threshold_raw) if threshold_raw is not None else None
    decision_capable = bool(artifact["decision_capable"])
    threshold_kind = (
        "policy_threshold" if artifact["resolved_config"]["policy"]["method"]
        == "cluster_bootstrap_run_risk" else "ltt_threshold"
    )
    actions: list[str] = []
    reasons: list[str] = []
    for index in range(len(df)):
        if not in_domain[index]:
            actions.append("RUN_FULL")
            reasons.append("out_of_applicability_domain")
        elif not unseen_complex[index]:
            actions.append("RUN_FULL")
            reasons.append("complex_seen_during_artifact_development")
        elif not trusted_input[index]:
            actions.append("RUN_FULL")
            reasons.append("reduced_run_provenance_not_scientific_or_engine_not_allowlisted")
        elif not decision_capable:
            actions.append("RUN_FULL")
            reasons.append("artifact_not_authorized_for_automatic_decision")
        elif threshold is not None and probability_lower[index] >= threshold:
            actions.append("ACCEPT_REDUCED")
            reasons.append(f"venn_abers_lower_probability_passed_{threshold_kind}")
        else:
            actions.append("RUN_FULL")
            reasons.append(f"venn_abers_lower_probability_below_{threshold_kind}")

    result = df.copy()
    result["artifact_sha256"] = artifact_sha256
    result["input_snapshot_sha256"] = sha256_bytes(input_bytes)
    result["base_raw_score"] = raw_score
    for column_index, feature in enumerate(features):
        result[f"score_contribution__{feature}"] = contributions[:, column_index]
    result["probability_lower"] = probability_lower
    result["probability_upper"] = probability_upper
    result["probability"] = probability
    result["probability_width"] = probability_upper - probability_lower
    for trace_name in (
        "lower_positive_count",
        "lower_weight",
        "upper_positive_count",
        "upper_weight",
        "point_numerator",
        "point_denominator",
    ):
        result[f"venn_abers_{trace_name}"] = venn_trace[trace_name]
    result["in_applicability_domain"] = in_domain
    result["unseen_complex"] = unseen_complex
    result["trusted_reduced_input_provenance"] = trusted_input
    result["development_complex_id_match_audit_only"] = development_id_match
    result["authorized_threshold"] = threshold
    result["action"] = actions
    result["reason"] = reasons
    return result
