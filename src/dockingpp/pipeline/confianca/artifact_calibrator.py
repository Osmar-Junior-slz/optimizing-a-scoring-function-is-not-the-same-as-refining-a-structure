"""Artifact-backed confidence calibration for reduced-search acceptance."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dockingpp.pipeline.confianca.evidence_graph import EvidenceGraph
from dockingpp.pipeline.confianca.features import (
    PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES,
    PRESERVATION_MECHANISTIC_GRAPH_V5_PROFILE,
    extract_mechanistic_graph_v5_features,
)


ARTIFACT_SCHEMA_VERSION_V5 = "5"
V5_FEATURE_SCHEMA_VERSION = "1.3"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _strict_int(value: Any, *, field: str, minimum: int = 0) -> int:
    """Return a JSON integer while rejecting bools and numeric coercion."""

    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return int(value)


def _complex_id_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in value):
        raise ValueError(f"{field} must contain non-empty strings")
    identifiers = [item.strip() for item in value]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"{field} must contain distinct complex ids")
    return identifiers


def _required_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a 64-character hexadecimal hash")
    digest = value.strip()
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{field} must be a 64-character hexadecimal hash")
    return digest.lower()


def _nested_feature(features: Mapping[str, Any], path: str) -> Any:
    current: Any = features
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _float_mapping(payload: Any, field: str) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"confidence calibrator field '{field}' must be an object")
    converted: dict[str, float] = {}
    for key, value in payload.items():
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"confidence calibrator field '{field}.{key}' must be numeric"
            ) from exc
        if not math.isfinite(number):
            raise ValueError(
                f"confidence calibrator field '{field}.{key}' must be finite"
            )
        converted[str(key)] = number
    return converted


@dataclass(frozen=True)
class V5ArtifactConfidenceCalibrator:
    """Fail-closed statistical + complex-evidence-graph preservation model."""

    name: str
    version: str
    calibrated: bool
    decision_capable: bool
    artifact_path: str
    artifact_sha256: str
    feature_schema_version: str
    feature_profile: str
    decision_target: str
    target_definition: dict[str, Any]
    training_dataset_sha256: str
    preprocessing: dict[str, Any]
    global_model: dict[str, Any]
    graph: EvidenceGraph
    decision_policy: dict[str, Any]
    calibration_evidence: dict[str, Any]
    coefficients: dict[str, float]
    intercept: float
    feature_means: dict[str, float]
    feature_scales: dict[str, float]

    @property
    def decision_probability_threshold(self) -> float:
        return float(self.decision_policy["minimum_probability"])

    @property
    def applicability_domain(self) -> dict[str, Any]:
        return {
            "method": "complex_prototype_knn",
            "distance_metric": self.graph.distance_metric,
            "k": int(self.graph.k),
            "domain_threshold": float(self.graph.domain_threshold),
            "decision_blocking": True,
            "clipping": False,
            "imputation_for_decision": False,
        }

    @classmethod
    def from_path(cls, path: str | Path) -> "V5ArtifactConfidenceCalibrator":
        artifact_path = Path(path).expanduser().resolve()
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("confidence calibrator artifact must contain a JSON object")
        if str(payload.get("schema_version")) != ARTIFACT_SCHEMA_VERSION_V5:
            raise ValueError(
                "unsupported v5 confidence calibrator schema_version: "
                f"{payload.get('schema_version')!r}"
            )
        if payload.get("model_kind") != "statistical_plus_evidence_graph":
            raise ValueError("v5 model_kind must be 'statistical_plus_evidence_graph'")
        if payload.get("feature_profile") != PRESERVATION_MECHANISTIC_GRAPH_V5_PROFILE:
            raise ValueError("v5 artifact uses an unsupported feature profile")

        calibrated = payload.get("calibrated") is True
        raw_decision_capable = payload.get("decision_capable", False)
        if not isinstance(raw_decision_capable, bool):
            raise ValueError("decision_capable must be a boolean")
        decision_capable = bool(raw_decision_capable)
        if decision_capable and not calibrated:
            raise ValueError("decision_capable=true requires calibrated=true")

        training_hash = _required_sha256(
            payload.get("training_dataset_sha256"),
            field="v5 training_dataset_sha256",
        )
        feature_schema_version = str(payload.get("feature_schema_version") or "").strip()
        if feature_schema_version != V5_FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"v5 feature_schema_version must be {V5_FEATURE_SCHEMA_VERSION!r}"
            )
        decision_target = str(payload.get("decision_target") or "").strip()
        if not decision_target.startswith("reduced_preserves_full:"):
            raise ValueError("v5 decision_target must describe REDUCED/FULL preservation")
        target_raw = payload.get("target_definition")
        if not isinstance(target_raw, Mapping):
            raise ValueError("v5 target_definition must be an object")
        target_definition = dict(target_raw)
        if (
            target_definition.get("type")
            != "reduced_vs_full_structural_preservation"
            or target_definition.get("features_source")
            != "reduced_only_reference_free"
            or target_definition.get("full_control_role")
            != "label_only_never_model_input"
        ):
            raise ValueError("v5 artifact violates the REDUCED-only prediction contract")
        positive_if = target_definition.get("positive_if")
        if not isinstance(positive_if, Mapping):
            raise ValueError("v5 target_definition.positive_if must be an object")
        dockq_minimum = _safe_v5_number(
            positive_if.get("dockq_reduced_minimum"),
            field="target_definition.positive_if.dockq_reduced_minimum",
        )
        maximum_loss = _safe_v5_number(
            positive_if.get("dockq_full_minus_reduced_maximum"),
            field="target_definition.positive_if.dockq_full_minus_reduced_maximum",
        )
        if (
            not 0.0 < dockq_minimum <= 1.0
            or not 0.0 <= maximum_loss <= 1.0
            or positive_if.get("logical_operator") != "and"
            or positive_if.get("boundary_inclusive") is not True
        ):
            raise ValueError("v5 preservation target definition is invalid")

        preprocessing_raw = payload.get("preprocessing")
        if not isinstance(preprocessing_raw, Mapping):
            raise ValueError("v5 preprocessing must be an object")
        preprocessing = dict(preprocessing_raw)
        if (
            preprocessing.get("standardize") is not True
            or preprocessing.get("winsorize") is not False
            or preprocessing.get("pca") is not False
            or preprocessing.get("impute_for_decision") is not False
        ):
            raise ValueError(
                "v5 preprocessing must standardize without winsorization, PCA, or imputation"
            )
        feature_paths_raw = preprocessing.get("feature_paths")
        if not isinstance(feature_paths_raw, list):
            raise ValueError("v5 preprocessing.feature_paths must be a list")
        feature_paths = [str(value) for value in feature_paths_raw]
        if tuple(feature_paths) != PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES:
            raise ValueError("v5 artifact feature paths do not match the frozen profile")
        means = _float_mapping(preprocessing.get("means"), "preprocessing.means")
        scales = _float_mapping(preprocessing.get("scales"), "preprocessing.scales")
        if set(means) != set(feature_paths) or set(scales) != set(feature_paths):
            raise ValueError("v5 scaler keys must match feature_paths")
        if any(value <= 0.0 for value in scales.values()):
            raise ValueError("v5 feature scales must be positive")

        global_model_raw = payload.get("global_model")
        if not isinstance(global_model_raw, Mapping):
            raise ValueError("v5 global_model must be an object")
        global_model = dict(global_model_raw)
        model_type = str(global_model.get("type") or "")
        if model_type not in {"logistic_l2", "logistic_elastic_net"}:
            raise ValueError("unsupported v5 global model type")
        coefficients_raw = global_model.get("coefficients")
        if isinstance(coefficients_raw, Mapping):
            coefficients = _float_mapping(coefficients_raw, "global_model.coefficients")
        elif isinstance(coefficients_raw, list):
            if len(coefficients_raw) != len(feature_paths):
                raise ValueError("v5 coefficient list must align with feature_paths")
            coefficients = {
                feature_path: _safe_v5_number(
                    value, field=f"global_model.coefficients[{index}]"
                )
                for index, (feature_path, value) in enumerate(
                    zip(feature_paths, coefficients_raw)
                )
            }
        else:
            raise ValueError("v5 global_model.coefficients must be a list or object")
        if set(coefficients) != set(feature_paths):
            raise ValueError("v5 coefficients must match feature_paths")
        intercept = _safe_v5_number(global_model.get("intercept"), field="global_model.intercept")
        parameters = global_model.get("parameters")
        if parameters is not None and not isinstance(parameters, Mapping):
            raise ValueError("v5 global_model.parameters must be an object")

        graph_raw = payload.get("evidence_graph")
        graph = EvidenceGraph.from_dict(graph_raw if isinstance(graph_raw, Mapping) else {})
        if graph.feature_names != feature_paths:
            raise ValueError("v5 graph features must match the global model features")

        policy_raw = payload.get("decision_policy")
        if not isinstance(policy_raw, Mapping):
            raise ValueError("v5 decision_policy must be an object")
        decision_policy = dict(policy_raw)
        for field in (
            "risk_target",
            "minimum_probability",
            "max_graph_disagreement",
        ):
            decision_policy[field] = _safe_v5_number(
                decision_policy.get(field), field=f"decision_policy.{field}"
            )
        if not 0.0 < decision_policy["risk_target"] < 1.0:
            raise ValueError("v5 risk_target must be between 0 and 1")
        if not 0.0 < decision_policy["minimum_probability"] < 1.0:
            raise ValueError("v5 minimum_probability must be between 0 and 1")
        if not 0.0 <= decision_policy["max_graph_disagreement"] <= 1.0:
            raise ValueError("v5 max_graph_disagreement must be within [0, 1]")
        if (
            decision_policy.get("require_domain_support") is not True
            or decision_policy.get("require_graph_agreement") is not True
        ):
            raise ValueError("v5 decisions must require graph domain and agreement gates")

        evidence_raw = payload.get("calibration_evidence")
        if not isinstance(evidence_raw, Mapping):
            raise ValueError("v5 calibration_evidence must be an object")
        calibration_evidence = dict(evidence_raw)
        quality_gate = calibration_evidence.get("calibration_quality_gate")
        leakage_audit = calibration_evidence.get("group_leakage_audit")
        if calibrated:
            if (
                not isinstance(quality_gate, Mapping)
                or quality_gate.get("passed") is not True
                or not isinstance(leakage_audit, Mapping)
                or leakage_audit.get("passed") is not True
            ):
                raise ValueError(
                    "calibrated v5 artifacts require a passed calibration gate "
                    "and zero group leakage"
                )
            violation_count = _strict_int(
                leakage_audit.get("violation_count"),
                field="calibration_evidence.group_leakage_audit.violation_count",
            )
            if violation_count != 0:
                raise ValueError(
                    "calibrated v5 artifacts require a passed calibration gate "
                    "and zero group leakage"
                )
        if decision_capable:
            operating = calibration_evidence.get("operating_point_assessment")
            external = calibration_evidence.get("external_validation_confirmation")
            frozen_policy = calibration_evidence.get("frozen_policy")
            if (
                not isinstance(operating, Mapping)
                or operating.get("supported") is not True
                or not isinstance(external, Mapping)
                or external.get("supported") is not True
                or external.get("evaluated") is not True
                or external.get("thresholds_refit") is not False
                or not isinstance(frozen_policy, Mapping)
            ):
                raise ValueError(
                    "v5 decision_capable=true requires nested risk support and "
                    "frozen external confirmation"
                )
            risk_upper = _safe_v5_number(
                operating.get("risk_upper_bound"),
                field="calibration_evidence.operating_point_assessment.risk_upper_bound",
            )
            if not 0.0 <= risk_upper <= 1.0:
                raise ValueError("v5 operating risk upper bound must be within [0, 1]")
            if risk_upper > decision_policy["risk_target"]:
                raise ValueError("v5 operating risk upper bound exceeds the frozen target")
            external_risk_upper = _safe_v5_number(
                external.get("risk_upper_bound"),
                field="calibration_evidence.external_validation_confirmation.risk_upper_bound",
            )
            if not 0.0 <= external_risk_upper <= 1.0:
                raise ValueError("v5 external risk upper bound must be within [0, 1]")
            if external_risk_upper > decision_policy["risk_target"]:
                raise ValueError("v5 external risk upper bound exceeds the frozen target")

            operating_risk_target = _safe_v5_number(
                operating.get("risk_target"),
                field="calibration_evidence.operating_point_assessment.risk_target",
            )
            operating_confidence = _safe_v5_number(
                operating.get("confidence_level"),
                field="calibration_evidence.operating_point_assessment.confidence_level",
            )
            operating_row_coverage = _safe_v5_number(
                operating.get("row_coverage"),
                field="calibration_evidence.operating_point_assessment.row_coverage",
            )
            operating_complex_coverage = _safe_v5_number(
                operating.get("complex_coverage"),
                field="calibration_evidence.operating_point_assessment.complex_coverage",
            )
            operating_complex_count = _strict_int(
                operating.get("accepted_complex_count"),
                field=(
                    "calibration_evidence.operating_point_assessment."
                    "accepted_complex_count"
                ),
                minimum=1,
            )
            operating_row_count = _strict_int(
                operating.get("accepted_row_count"),
                field=(
                    "calibration_evidence.operating_point_assessment.accepted_row_count"
                ),
                minimum=1,
            )
            operating_complex_ids = _complex_id_list(
                operating.get("accepted_complex_ids"),
                field=(
                    "calibration_evidence.operating_point_assessment.accepted_complex_ids"
                ),
            )
            if (
                not math.isclose(
                    operating_risk_target,
                    decision_policy["risk_target"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not 0.0 < operating_confidence < 1.0
                or not 0.0 < operating_row_coverage <= 1.0
                or not 0.0 < operating_complex_coverage <= 1.0
                or len(operating_complex_ids) != operating_complex_count
                or operating_row_count < operating_complex_count
            ):
                raise ValueError("v5 operating-point evidence is internally inconsistent")

            frozen_probability = _safe_v5_number(
                frozen_policy.get("minimum_probability"),
                field="calibration_evidence.frozen_policy.minimum_probability",
            )
            frozen_disagreement = _safe_v5_number(
                frozen_policy.get("max_graph_disagreement"),
                field="calibration_evidence.frozen_policy.max_graph_disagreement",
            )
            frozen_domain = _safe_v5_number(
                frozen_policy.get("domain_threshold"),
                field="calibration_evidence.frozen_policy.domain_threshold",
            )
            frozen_risk_target = _safe_v5_number(
                frozen_policy.get("risk_target"),
                field="calibration_evidence.frozen_policy.risk_target",
            )
            frozen_k = _strict_int(
                frozen_policy.get("k"),
                field="calibration_evidence.frozen_policy.k",
                minimum=1,
            )
            artifact_probability = _safe_v5_number(
                payload.get("decision_probability_threshold"),
                field="decision_probability_threshold",
            )
            if (
                not math.isclose(
                    frozen_probability,
                    decision_policy["minimum_probability"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    frozen_disagreement,
                    decision_policy["max_graph_disagreement"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    frozen_domain,
                    graph.domain_threshold,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    frozen_risk_target,
                    decision_policy["risk_target"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    artifact_probability,
                    decision_policy["minimum_probability"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or frozen_k != graph.k
            ):
                raise ValueError("v5 decision thresholds differ from the frozen nested policy")

            training_provenance = payload.get("training_provenance")
            if not isinstance(training_provenance, Mapping):
                raise ValueError("v5 decision-capable artifacts require training_provenance")
            if (
                training_provenance.get("schema_version") != "5.0"
                or training_provenance.get("cohort_role") != "development_only"
                or training_provenance.get("grouping_unit") != "complex_id"
                or training_provenance.get("full_controls_used_as_predictors") is not False
                or training_provenance.get("external_data_used_for_selection") is not False
            ):
                raise ValueError("v5 training provenance violates the frozen cohort contract")
            training_rows = _strict_int(
                training_provenance.get("n_rows"),
                field="training_provenance.n_rows",
                minimum=1,
            )
            training_complex_count = _strict_int(
                training_provenance.get("n_complexes"),
                field="training_provenance.n_complexes",
                minimum=2,
            )
            training_complex_ids = _complex_id_list(
                training_provenance.get("complex_ids"),
                field="training_provenance.complex_ids",
            )
            provenance_dataset_hash = _required_sha256(
                training_provenance.get("dataset_sha256"),
                field="training_provenance.dataset_sha256",
            )
            training_preregistration_hash = _required_sha256(
                training_provenance.get("pre_registration_sha256"),
                field="training_provenance.pre_registration_sha256",
            )
            _required_sha256(
                training_provenance.get("trial_manifest_sha256"),
                field="training_provenance.trial_manifest_sha256",
            )
            training_ids_hash = _required_sha256(
                training_provenance.get("complex_ids_sha256"),
                field="training_provenance.complex_ids_sha256",
            )
            graph_complex_ids = [node.complex_id for node in graph.nodes]
            if (
                provenance_dataset_hash != training_hash.lower()
                or training_ids_hash
                != _canonical_sha256(sorted(training_complex_ids))
                or training_complex_count != len(training_complex_ids)
                or set(training_complex_ids) != set(graph_complex_ids)
                or training_complex_count != len(graph.nodes)
                or training_rows != sum(node.n_runs for node in graph.nodes)
                or not set(operating_complex_ids).issubset(training_complex_ids)
                or not math.isclose(
                    operating_row_coverage,
                    operating_row_count / float(training_rows),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or not math.isclose(
                    operating_complex_coverage,
                    operating_complex_count / float(training_complex_count),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                raise ValueError("v5 training provenance does not match the fitted graph")

            if (
                external.get("schema_version") != "5.0"
                or external.get("cohort_role")
                != "untouched_external_confirmation"
                or external.get("required") is not True
                or external.get("thresholds_may_be_refit_on_external") is not False
                or external.get("external_outcomes_used_for_training") is not False
                or external.get("external_outcomes_used_for_policy_selection") is not False
            ):
                raise ValueError("v5 external confirmation violates the frozen cohort contract")
            external_complex_count = _strict_int(
                external.get("complex_count"),
                field=(
                    "calibration_evidence.external_validation_confirmation.complex_count"
                ),
                minimum=1,
            )
            external_trial_count = _strict_int(
                external.get("paired_trial_count"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "paired_trial_count"
                ),
                minimum=1,
            )
            external_trials_per_complex = _strict_int(
                external.get("trials_per_complex"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "trials_per_complex"
                ),
                minimum=1,
            )
            external_authorized_complex_count = _strict_int(
                external.get("authorized_complex_count"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "authorized_complex_count"
                ),
                minimum=1,
            )
            external_authorized_trial_count = _strict_int(
                external.get("authorized_trial_count"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "authorized_trial_count"
                ),
                minimum=1,
            )
            external_complex_ids = _complex_id_list(
                external.get("complex_ids"),
                field=(
                    "calibration_evidence.external_validation_confirmation.complex_ids"
                ),
            )
            external_confidence = _safe_v5_number(
                external.get("confidence_level"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "confidence_level"
                ),
            )
            external_coverage = _safe_v5_number(
                external.get("authorized_trial_coverage"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "authorized_trial_coverage"
                ),
            )
            external_dataset_hash = _required_sha256(
                external.get("dataset_sha256"),
                field=(
                    "calibration_evidence.external_validation_confirmation.dataset_sha256"
                ),
            )
            external_preregistration_hash = _required_sha256(
                external.get("pre_registration_sha256"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "pre_registration_sha256"
                ),
            )
            for field in (
                "trial_manifest_sha256",
                "analysis_report_sha256",
                "evaluated_artifact_sha256",
            ):
                _required_sha256(
                    external.get(field),
                    field=(
                        "calibration_evidence.external_validation_confirmation."
                        f"{field}"
                    ),
                )
            external_ids_hash = _required_sha256(
                external.get("complex_ids_sha256"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "complex_ids_sha256"
                ),
            )
            frozen_policy_hash = _required_sha256(
                external.get("frozen_policy_sha256"),
                field=(
                    "calibration_evidence.external_validation_confirmation."
                    "frozen_policy_sha256"
                ),
            )
            if (
                len(external_complex_ids) != external_complex_count
                or external_trial_count
                != external_complex_count * external_trials_per_complex
                or external_authorized_complex_count > external_complex_count
                or external_authorized_trial_count > external_trial_count
                or not 0.0 < external_confidence < 1.0
                or not 0.0 < external_coverage <= 1.0
                or not math.isclose(
                    external_coverage,
                    external_authorized_trial_count / float(external_trial_count),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                or external_ids_hash
                != _canonical_sha256(sorted(external_complex_ids))
                or frozen_policy_hash != _canonical_sha256(dict(frozen_policy))
                or not set(external_complex_ids).isdisjoint(training_complex_ids)
                or external_dataset_hash == provenance_dataset_hash
                or external_preregistration_hash == training_preregistration_hash
            ):
                raise ValueError("v5 external confirmation is not independent and consistent")

        return cls(
            name=str(payload.get("name") or "dockingpp-preservation-v5"),
            version=str(payload.get("version") or "unversioned"),
            calibrated=calibrated,
            decision_capable=decision_capable,
            artifact_path=str(artifact_path),
            artifact_sha256=_sha256(artifact_path),
            feature_schema_version=feature_schema_version,
            feature_profile=str(payload["feature_profile"]),
            decision_target=decision_target,
            target_definition=target_definition,
            training_dataset_sha256=training_hash,
            preprocessing=preprocessing,
            global_model=global_model,
            graph=graph,
            decision_policy=decision_policy,
            calibration_evidence=calibration_evidence,
            coefficients=coefficients,
            intercept=intercept,
            feature_means=means,
            feature_scales=scales,
        )

    def _base_prediction(self) -> dict[str, Any]:
        return {
            "artifact_schema_version": 5,
            "prediction_schema_version": "5.0",
            "model_kind": "statistical_plus_evidence_graph",
            "feature_profile": self.feature_profile,
            "feature_schema_version": self.feature_schema_version,
            "model_name": self.name,
            "model_version": self.version,
            "artifact_sha256": self.artifact_sha256,
            "training_dataset_sha256": self.training_dataset_sha256,
            "decision_target": self.decision_target,
            "target_definition": self.target_definition,
            "calibrated": bool(self.calibrated),
            "artifact_decision_capable": bool(self.decision_capable),
            "decision_probability_threshold": self.decision_probability_threshold,
            "applicability_domain": self.applicability_domain,
        }

    def predict(
        self,
        *,
        confidence_features: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        base = self._base_prediction()
        raw_features, missing = extract_mechanistic_graph_v5_features(
            confidence_features
        )
        if missing:
            reasons = ["MISSING_FEATURE"]
            if not self.decision_capable:
                reasons.insert(0, "ARTIFACT_NOT_DECISION_CAPABLE")
            base.update(
                {
                    "estimated_success_probability": None,
                    "estimated_preservation_probability": None,
                    "p_stat": None,
                    "graph": None,
                    "raw_features": raw_features,
                    "standardized_features": {},
                    "missing_feature_paths": sorted(missing),
                    "features_complete": False,
                    "applicability_domain_supported": False,
                    "applicability_domain_reasons": [
                        f"missing_required_feature:{path}" for path in sorted(missing)
                    ],
                    "decision_capable": False,
                    "decision": (
                        "OBSERVE_ONLY" if not self.decision_capable else "ABSTAIN"
                    ),
                    "fallback_required": bool(self.decision_capable),
                    "reason_codes": reasons,
                }
            )
            return base

        standardized = {
            path: (raw_features[path] - self.feature_means[path])
            / self.feature_scales[path]
            for path in PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES
        }
        linear = self.intercept + sum(
            self.coefficients[path] * standardized[path]
            for path in PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES
        )
        probability = (
            1.0 / (1.0 + math.exp(-linear))
            if linear >= 0.0
            else math.exp(linear) / (1.0 + math.exp(linear))
        )
        graph_evidence = self.graph.evaluate(
            [standardized[path] for path in PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES],
            p_stat=probability,
        )
        disagreement = float(graph_evidence.disagreement or 0.0)
        domain_supported = bool(graph_evidence.domain_supported)
        agreement_supported = bool(
            disagreement <= self.decision_policy["max_graph_disagreement"]
        )
        probability_supported = bool(
            probability >= self.decision_policy["minimum_probability"]
        )

        gate_reasons: list[str] = []
        if not domain_supported:
            gate_reasons.append("OUT_OF_DOMAIN")
        if not agreement_supported:
            gate_reasons.append("GRAPH_MODEL_DISAGREEMENT")
        if not probability_supported:
            gate_reasons.append("LOW_STATISTICAL_CONFIDENCE")
        if not self.decision_capable:
            reason_codes = ["ARTIFACT_NOT_DECISION_CAPABLE", *gate_reasons]
            decision = "OBSERVE_ONLY"
        elif gate_reasons:
            reason_codes = gate_reasons
            decision = "ABSTAIN"
        else:
            reason_codes = ["ACCEPTED"]
            decision = "ACCEPT_REDUCED"
        reason_codes = list(dict.fromkeys(reason_codes))
        effective_decision_capable = bool(self.decision_capable and domain_supported)
        base.update(
            {
                "estimated_success_probability": float(probability),
                "estimated_preservation_probability": float(probability),
                "p_stat": float(probability),
                "global_model_type": str(self.global_model.get("type")),
                "global_model_parameters": dict(
                    self.global_model.get("parameters") or {}
                ),
                "graph": graph_evidence.to_dict(),
                "raw_features": raw_features,
                "standardized_features": standardized,
                "missing_feature_paths": [],
                "features_complete": True,
                "applicability_domain_supported": domain_supported,
                "applicability_domain_reasons": (
                    [] if domain_supported else ["graph_kth_distance_above_threshold"]
                ),
                "graph_agreement_supported": agreement_supported,
                "statistical_probability_supported": probability_supported,
                "decision_capable": effective_decision_capable,
                "decision": decision,
                "fallback_required": bool(decision == "ABSTAIN"),
                "reason_codes": reason_codes,
                "policy": {
                    "probability_threshold": self.decision_probability_threshold,
                    "max_graph_disagreement": float(
                        self.decision_policy["max_graph_disagreement"]
                    ),
                    "risk_target": float(self.decision_policy["risk_target"]),
                    "require_domain_support": True,
                    "require_graph_agreement": True,
                },
            }
        )
        return base


def _safe_v5_number(value: Any, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def load_confidence_calibrator(path: str | Path) -> V5ArtifactConfidenceCalibrator:
    """Load the only supported calibrator architecture: preservation v5."""

    artifact_path = Path(path).expanduser().resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("confidence calibrator artifact must contain a JSON object")
    if str(payload.get("schema_version")) != ARTIFACT_SCHEMA_VERSION_V5:
        raise ValueError(
            "unsupported calibrator artifact: only preservation schema_version=5 "
            "is accepted; retrain the calibrator with the v5 protocol"
        )
    return V5ArtifactConfidenceCalibrator.from_path(artifact_path)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION_V5",
    "V5ArtifactConfidenceCalibrator",
    "load_confidence_calibrator",
]

