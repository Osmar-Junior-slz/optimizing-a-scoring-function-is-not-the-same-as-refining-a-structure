from __future__ import annotations

import math
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import yaml

from docking_confidence.features.contact_graph import (
    CONTACT_GRAPH_FEATURE_COLUMNS,
    ContactGraphConfig,
)
from docking_confidence.features.execution_context import EXECUTION_CONTEXT_FEATURE_COLUMNS


ARTIFACT_SCHEMA_VERSION = "2.0"
FEATURE_SCHEMA_VERSION = "reduced_decision_features_v2"
LOCAL_REFINEMENT_FEATURE_SCHEMA_VERSION = "local_refinement_decision_features_v1"
LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION = "local_refinement_decision_features_v2"
CONFIG_DIGEST_SCHEMA = "validated_config_canonical_json_v1"
DOCKING_TASK_FEATURE_SCHEMAS = {
    "surface_discovery": FEATURE_SCHEMA_VERSION,
    "local_refinement": LOCAL_REFINEMENT_FEATURE_SCHEMA_VERSION,
}

REDUCED_SEARCH_FEATURE_COLUMNS = (
    "scan_gap_norm",
    "viable_fraction_mean",
    "elite_dispersion",
    "uncertainty_count",
    "energy_q20_norm",
)

PRODUCTION_FEATURE_COLUMNS = (
    *REDUCED_SEARCH_FEATURE_COLUMNS,
    "graph_has_interface_contact",
    "graph_min_cross_distance_angstrom",
    "graph_interface_edge_count",
    "graph_interface_density",
    "graph_receptor_contact_fraction",
    "graph_ligand_contact_fraction",
    "graph_largest_component_fraction",
    "graph_clash_atom_pair_count",
)

# Local refinement has no multi-pocket scan. Its population diagnostics and
# graph descriptors remain defined; scan gap/scan energy do not.
LOCAL_REFINEMENT_FEATURE_COLUMNS = tuple(
    column for column in PRODUCTION_FEATURE_COLUMNS
    if column not in {"scan_gap_norm", "energy_q20_norm"}
)
LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS = (
    *LOCAL_REFINEMENT_FEATURE_COLUMNS, *EXECUTION_CONTEXT_FEATURE_COLUMNS,
)


def compatible_feature_schema(task: str | None, schema: str) -> bool:
    if not isinstance(schema, str):
        return False
    if task == "local_refinement":
        return schema in {LOCAL_REFINEMENT_FEATURE_SCHEMA_VERSION, LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION}
    return schema == DOCKING_TASK_FEATURE_SCHEMAS.get(task, FEATURE_SCHEMA_VERSION)


def local_feature_columns(schema: str) -> tuple[str, ...]:
    if schema == LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION:
        return LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS
    return LOCAL_REFINEMENT_FEATURE_COLUMNS

_ALLOWED_FEATURES = frozenset(
    (*REDUCED_SEARCH_FEATURE_COLUMNS, *CONTACT_GRAPH_FEATURE_COLUMNS, *EXECUTION_CONTEXT_FEATURE_COLUMNS)
)
_PROFILE_NAMES = {
    "production_graph_v2",
    "pilot_graph_v2",
    "demo_non_scientific",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"configuration section {key!r} must be a YAML object")
    return value


def _number(
    section: dict[str, Any],
    key: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    raw = section.get(key)
    if isinstance(raw, bool):
        raise ValueError(f"configuration value {key!r} must be numeric, not boolean")
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"configuration value {key!r} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"configuration value {key!r} must be finite")
    if strictly_positive and value <= 0:
        raise ValueError(f"configuration value {key!r} must be > 0")
    if minimum is not None and value < minimum:
        raise ValueError(f"configuration value {key!r} must be >= {minimum}")
    if maximum is not None and value > maximum:
        raise ValueError(f"configuration value {key!r} must be <= {maximum}")
    return value


def _integer(section: dict[str, Any], key: str, *, minimum: int) -> int:
    value = _number(section, key, minimum=float(minimum))
    if not value.is_integer():
        raise ValueError(f"configuration value {key!r} must be an integer")
    return int(value)


def _strict_bool(section: dict[str, Any], key: str, *, context: str) -> bool:
    value = section.get(key)
    if type(value) is not bool:
        raise ValueError(f"{context}.{key} must be boolean")
    return value


def validate_config(config: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML object")
    if str(config.get("schema_version")) != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"configuration schema_version must be {ARTIFACT_SCHEMA_VERSION!r}"
        )

    features = _mapping(config, "features")
    feature_schema = features.get("schema_version")
    docking_task = config.get("docking_task")
    if docking_task is not None and docking_task not in DOCKING_TASK_FEATURE_SCHEMAS:
        raise ValueError("docking_task must be local_refinement or surface_discovery")
    expected_schema = DOCKING_TASK_FEATURE_SCHEMAS.get(docking_task, FEATURE_SCHEMA_VERSION)
    if not compatible_feature_schema(docking_task, feature_schema):
        raise ValueError(
            f"features.schema_version must be {expected_schema!r} for docking_task={docking_task!r}"
        )
    if features.get("provenance") != "reduced_only":
        raise ValueError("features.provenance must be 'reduced_only'")
    columns = features.get("columns")
    if not isinstance(columns, list) or not columns:
        raise ValueError("features.columns must be a non-empty list")
    if any(
        not isinstance(column, str) or not column.strip() or column != column.strip()
        for column in columns
    ):
        raise ValueError("every feature name must be a trimmed non-empty string")
    if len(columns) != len(set(columns)):
        raise ValueError("features.columns contains duplicate names")
    unknown = sorted(set(columns) - _ALLOWED_FEATURES)
    if unknown:
        raise ValueError(
            "features must belong to the closed REDUCED feature registry; "
            f"unsupported or potentially leaky columns: {unknown}"
        )
    if set(columns) & set(EXECUTION_CONTEXT_FEATURE_COLUMNS) and (
        docking_task != "local_refinement" or feature_schema != LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION
    ):
        raise ValueError("input-pose and algorithm context require the explicit local-refinement v2 schema")
    production_columns = (
        local_feature_columns(feature_schema)
        if docking_task == "local_refinement" else PRODUCTION_FEATURE_COLUMNS
    )
    if docking_task == "local_refinement" and tuple(columns) != production_columns:
        raise ValueError("local_refinement requires its frozen available feature order")

    contact_graph = _mapping(config, "contact_graph")
    graph_enabled = _strict_bool(
        contact_graph, "enabled", context="contact_graph"
    )
    require_pose = _strict_bool(
        contact_graph, "require_pose_provenance", context="contact_graph"
    )
    graph_config = ContactGraphConfig.from_mapping(contact_graph)
    graph_features = [column for column in columns if column.startswith("graph_")]
    if graph_enabled and not graph_features:
        raise ValueError("contact_graph.enabled=true requires graph-v2 features")
    if graph_enabled and not require_pose:
        raise ValueError("enabled contact graph requires pose provenance")
    if not graph_enabled and graph_features:
        raise ValueError("graph_* features are forbidden when contact_graph is disabled")

    preservation = _mapping(config, "preservation")
    _number(preservation, "minimum_reduced_dockq", minimum=0.0, maximum=1.0)
    _number(preservation, "maximum_dockq_loss", minimum=0.0, maximum=1.0)

    validation = _mapping(config, "validation")
    _integer(validation, "split_folds", minimum=4)
    _integer(validation, "random_seed", minimum=0)
    if validation.get("assignment_commitment") != "predeclared_before_labels":
        raise ValueError(
            "validation.assignment_commitment must be 'predeclared_before_labels'"
        )

    model = _mapping(config, "model")
    _number(model, "C", strictly_positive=True)
    _integer(model, "max_iter", minimum=1)
    if model.get("class_weight") not in {"balanced", "none", None}:
        raise ValueError("model.class_weight must be 'balanced' or 'none'")

    calibration = _mapping(config, "calibration")
    fit_scope = calibration.get("fit_scope", "in_applicability_domain")
    if not isinstance(fit_scope, str) or fit_scope not in {"in_applicability_domain", "all_calibration_rows"}:
        raise ValueError("unsupported calibration.fit_scope")
    if calibration.get("method") != "inductive_venn_abers":
        raise ValueError("calibration.method must be 'inductive_venn_abers'")
    minimum_complexes = _integer(
        calibration, "minimum_independent_complexes", minimum=2
    )
    minimum_positive = _integer(
        calibration, "minimum_positive_class_complexes", minimum=1
    )
    minimum_negative = _integer(
        calibration, "minimum_negative_class_complexes", minimum=1
    )
    if minimum_positive > minimum_complexes or minimum_negative > minimum_complexes:
        raise ValueError("calibration class minima cannot exceed the total complex minimum")

    domain = _mapping(config, "applicability_domain")
    lower = _number(domain, "lower_quantile", minimum=0.0, maximum=1.0)
    upper = _number(domain, "upper_quantile", minimum=0.0, maximum=1.0)
    if lower >= upper:
        raise ValueError("applicability-domain quantiles must satisfy lower < upper")
    block_extrapolation = _strict_bool(
        domain, "block_feature_extrapolation", context="applicability_domain"
    )

    policy = _mapping(config, "policy")
    if policy.get("method") not in {
        "learn_then_test_bonferroni_binomial_iid", "cluster_bootstrap_run_risk",
    }:
        raise ValueError(
            "unsupported policy.method"
        )
    risk_target = _number(policy, "risk_target", minimum=0.0, maximum=1.0)
    if risk_target in {0.0, 1.0}:
        raise ValueError("risk_target must be strictly between 0 and 1")
    confidence = _number(
        policy, "familywise_confidence_level", minimum=0.0, maximum=1.0
    )
    if confidence in {0.0, 1.0}:
        raise ValueError("familywise_confidence_level must be strictly between 0 and 1")
    fraction = _number(
        policy, "risk_error_budget_fraction", minimum=0.0, maximum=1.0
    )
    if fraction in {0.0, 1.0}:
        raise ValueError("risk_error_budget_fraction must be strictly between 0 and 1")
    threshold_min = _number(policy, "threshold_min", minimum=0.0, maximum=1.0)
    threshold_max = _number(policy, "threshold_max", minimum=0.0, maximum=1.0)
    if threshold_max >= 1.0:
        raise ValueError("threshold_max must be < 1 for the IVAP lower endpoint")
    if threshold_min > threshold_max:
        raise ValueError("threshold_min must be less than or equal to threshold_max")
    threshold_step = _number(
        policy, "threshold_step", strictly_positive=True, maximum=1.0
    )
    candidate_count = int((threshold_max - threshold_min) / threshold_step) + 2
    if candidate_count > 1001:
        raise ValueError("threshold grid cannot contain more than 1001 candidates")
    if policy["method"] == "cluster_bootstrap_run_risk":
        repetitions = _integer(policy, "bootstrap_repetitions", minimum=1000)
        if repetitions > 1_000_000:
            raise ValueError("bootstrap_repetitions must be <= 1000000")
        _integer(policy, "bootstrap_seed", minimum=0)
    _number(policy, "minimum_complex_coverage", minimum=0.0, maximum=1.0)
    _integer(policy, "minimum_evaluated_complexes", minimum=1)
    _integer(policy, "minimum_authorized_complexes", minimum=1)

    runtime = _mapping(config, "runtime_gate")
    runtime_enabled = _strict_bool(runtime, "enabled", context="runtime_gate")
    runtime_required = _strict_bool(
        runtime, "require_reduced_faster_than_full", context="runtime_gate"
    )
    _number(runtime, "minimum_speedup", strictly_positive=True)
    runtime_scope = runtime.get("assessment_scope", "accepted_rows_legacy")
    if not isinstance(runtime_scope, str) or runtime_scope not in {"accepted_rows_legacy", "total_paired_cascade"}:
        raise ValueError("runtime_gate.assessment_scope is invalid")
    if runtime_scope == "total_paired_cascade":
        _strict_bool(runtime, "require_measured_overhead", context="runtime_gate")
        overhead_column = runtime.get("overhead_column")
        if not isinstance(overhead_column, str) or not overhead_column.strip() or overhead_column != overhead_column.strip():
            raise ValueError("total cascade runtime requires a nonempty overhead_column")

    authorization = _mapping(config, "authorization")
    profile = authorization.get("profile")
    if profile not in _PROFILE_NAMES:
        raise ValueError(f"authorization.profile must be one of {sorted(_PROFILE_NAMES)}")
    automatic = _strict_bool(
        authorization, "allow_automatic_decision", context="authorization"
    )
    require_scientific = _strict_bool(
        authorization, "require_scientific_provenance", context="authorization"
    )
    allowlist = authorization.get("trusted_engine_allowlist")
    if not isinstance(allowlist, list) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip()
        for item in allowlist
    ) or len(allowlist) != len(set(allowlist)):
        raise ValueError(
            "authorization.trusted_engine_allowlist must contain unique engine IDs"
        )
    commitment = authorization.get("protocol_commitment_sha256")
    commitment_recorded = (
        isinstance(commitment, str)
        and _SHA256_RE.fullmatch(commitment.strip().lower()) is not None
    )

    if automatic:
        failures: list[str] = []
        if profile != "production_graph_v2":
            failures.append("production_graph_v2 profile")
        if tuple(columns) != production_columns:
            failures.append("the frozen production feature order")
        if not graph_enabled or not require_pose:
            failures.append("the provenance-bound graph-v2 contract")
        if not require_scientific or not allowlist:
            failures.append("scientific provenance and a non-empty engine allowlist")
        if not runtime_enabled or not runtime_required:
            failures.append("the locked runtime speedup gate")
        if not block_extrapolation:
            failures.append("applicability-domain extrapolation blocking")
        if not commitment_recorded:
            failures.append("an external protocol commitment SHA-256")
        if failures:
            raise ValueError(
                "automatic decision configuration requires " + "; ".join(failures)
            )
    elif profile == "production_graph_v2" and tuple(columns) != production_columns:
        raise ValueError("production_graph_v2 requires the frozen production feature order")

    # Normalize only values whose canonical form is unambiguous. The caller's
    # mapping remains otherwise intact and is stored verbatim in the artifact.
    contact_graph.update({
        "schema_version": graph_config.schema_version,
        "parser_schema_version": graph_config.parser_schema_version,
        "altloc_policy": graph_config.altloc_policy,
        "node_universe": graph_config.node_universe,
        "model_id": graph_config.model_id,
        "cutoff_angstrom": float(graph_config.cutoff_angstrom),
        "clash_cutoff_angstrom": float(graph_config.clash_cutoff_angstrom),
        "include_hetero": graph_config.include_hetero,
        "heavy_atoms_only": graph_config.heavy_atoms_only,
        "exclude_waters": graph_config.exclude_waters,
    })
    if commitment_recorded:
        authorization["protocol_commitment_sha256"] = commitment.strip().lower()
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return validate_config(config)


def config_sha256(config: dict[str, Any]) -> str:
    """Digest effective validated settings, independent of YAML formatting/path."""
    canonical = json.dumps(
        validate_config(deepcopy(config)),
        sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
