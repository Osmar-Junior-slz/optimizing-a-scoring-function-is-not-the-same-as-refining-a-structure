"""Commit, develop and independently evaluate a decision over a frozen GVA model.

The model predicts a continuous mean margin. This module never interprets a
margin or a generalized prediction bound as a probability. Its acceptance
criteria are explicit empirical requirements, not population-risk guarantees.
Preparation reads identities and configuration, never outcome data. Policy
selection reads only the declared development role; final testing cannot change
the selected cutoff. Artifacts are integrity records, not signed preregistration.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, auc, precision_recall_curve, roc_auc_score

from docking_confidence.application.generalized_calibrator import (
    load_calibrator, predict_calibrator, preservation_margin,
)
from docking_confidence.application.standard_dataset import verify_standard_quality_records
from docking_confidence.audit.manifest import read_bytes_once, sha256_bytes
from docking_confidence.io.config import config_sha256, load_config
from docking_confidence.io.dataset import TRAINING_BASE_COLUMNS, load_paired_dataset
from docking_confidence.io.quality_context import validate_paired_quality_context, validate_quality_metric


PROTOCOL_SCHEMA = "gva_decision_protocol_v1"
POLICY_SCHEMA = "gva_frozen_decision_policy_v1"
EVALUATION_SCHEMA = "gva_frozen_decision_evaluation_v1"
EXPOSURE_SCHEMA = "gva_decision_data_exposure_v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_SCORES = {"prediction", "base_raw_score"}
_IDENTITY = ["complex_fingerprint_sha256", "algorithm", "seed"]
_REGISTRY = Path(__file__).resolve().parents[3] / "configs/decision_data_registry.json"
_EXPOSURE_ROOT = Path(__file__).resolve().parents[3] / "output/decision_exposure_ledger"


def contract_sha256(value: Any) -> str:
    """Canonical SHA-256 shared by quality and execution metadata columns."""
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=False, allow_nan=False).encode("utf-8"))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record(payload: dict) -> dict:
    result = deepcopy(payload)
    result["record_sha256"] = contract_sha256(result)
    return result


def _verify_record(value: dict, schema: str) -> dict:
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ValueError(f"Expected {schema} record")
    core = {k: v for k, v in value.items() if k != "record_sha256"}
    if value.get("record_sha256") != contract_sha256(core):
        raise ValueError("Decision record integrity hash mismatch")
    return value


def _read_record(path, schema):
    return _verify_record(json.loads(read_bytes_once(path)), schema)


def _write_new(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def _exposure_snapshot():
    """Read committed exposure entries; an incomplete/corrupt entry blocks reuse.

    A protocol freezes this snapshot, not a requirement that the global ledger
    remain unchanged. Already committed protocols can consume their own roles.
    """
    groups, records = set(), []
    root = Path(_EXPOSURE_ROOT).resolve()
    for path in sorted(root.glob("*.json")) if root.exists() else []:
        try:
            content = read_bytes_once(path)
            record = _verify_record(json.loads(content), EXPOSURE_SCHEMA)
            if (record.get("role") not in {"validation", "test"}
                    or not _HASH.fullmatch(record.get("protocol_sha256", ""))
                    or not _HASH.fullmatch(record.get("dataset_sha256", ""))):
                raise ValueError("Invalid exposure record identity")
            observed = _fingerprints(record.get("groups"), name="exposure groups")
        except (OSError, TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Global decision exposure ledger is unavailable or corrupt: {path}") from exc
        groups.update(observed)
        records.append({"path": str(path), "file_sha256": sha256_bytes(content),
            "record_sha256": record["record_sha256"], "group_count": len(observed)})
    return groups, {"root": str(root), "records": records,
        "snapshot_sha256": contract_sha256(records), "known_group_count": len(groups)}


def _commit_exposure(protocol, role, dataset_sha256):
    """Check and append exposure under one OS lock before outcome analysis.

    Recheck live foreign exposure even for protocols prepared before another
    protocol consumed the same groups. Their historical snapshot stays intact.
    """
    from docking_confidence.application.batch import _BatchFileLock, BatchAlreadyRunningError

    try:
        with _BatchFileLock(Path(_EXPOSURE_ROOT).resolve() / ".decision_exposure.lock"):
            _, snapshot = _exposure_snapshot()
            if protocol["analysis_role"] == "prospective":
                groups = set(protocol["roles"][role + "_groups"])
                for entry in snapshot["records"]:
                    previous = _read_record(entry["path"], EXPOSURE_SCHEMA)
                    if (previous["protocol_sha256"] != protocol["record_sha256"]
                            and groups.intersection(previous["groups"])):
                        raise ValueError("Molecular groups were already exposed by another decision protocol; "
                            "a previously prepared protocol cannot reuse them as an independent evaluation")
            return _commit_exposure_locked(protocol, role, dataset_sha256)
    except (BatchAlreadyRunningError, OSError) as exc:
        raise ValueError("Global decision exposure ledger is unavailable or in use; retry the unchanged stage") from exc


def _commit_exposure_locked(protocol, role, dataset_sha256):
    """Write once while the global exposure lock is held by the caller."""
    identity = {"protocol_sha256": protocol["record_sha256"], "role": role,
        "dataset_sha256": dataset_sha256}
    path = Path(_EXPOSURE_ROOT).resolve() / (contract_sha256(identity) + ".json")
    record = _record({"schema_version": EXPOSURE_SCHEMA, "created_at_utc": _utc(),
        **identity, "analysis_role": protocol["analysis_role"],
        "groups": list(protocol["roles"][role + "_groups"]),
        "scope": "all planned molecular groups, including failed or missing paired runs",
        "stage": "paired input validated; committed before model predictions or outcome analysis"})
    try:
        _write_new(path, record)
    except FileExistsError:
        existing = _read_record(path, EXPOSURE_SCHEMA)
        if any(existing.get(key) != value for key, value in identity.items()) or existing["groups"] != record["groups"]:
            raise ValueError("Exposure ledger identity collision or changed molecular roles")
        record = existing
    return {"exposure_record_path": str(path), "exposure_record_sha256": record["record_sha256"],
        "exposure_record_file_sha256": sha256_bytes(read_bytes_once(path))}


def _new_output(path, artifact_path) -> Path:
    root, model_root = Path(path).resolve(), Path(artifact_path).resolve().parent
    if root == model_root or model_root in root.parents or root in model_root.parents:
        raise ValueError("Decision outputs must be separate from the frozen model")
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("Choose a new or empty decision output directory")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fingerprints(values, *, name, allow_empty=False):
    if not isinstance(values, (list, tuple, set)):
        raise ValueError(f"{name} must be a list of molecular SHA-256 fingerprints")
    result = []
    for value in values:
        if not isinstance(value, str) or _HASH.fullmatch(value.lower()) is None:
            raise ValueError(f"{name} contains an invalid molecular fingerprint")
        result.append(value.lower())
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate fingerprints")
    if not result and not allow_empty:
        raise ValueError(f"{name} must contain at least one molecular fingerprint")
    return sorted(result)


def _number(value, name, *, minimum=None, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if minimum is not None and value < minimum or maximum is not None and value > maximum:
        raise ValueError(f"{name} is outside the permitted range")
    return float(value)


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _identity(row):
    return str(row["complex_fingerprint_sha256"]), str(row["algorithm"]), int(row["seed"])


def _planned_records(groups, algorithms, seeds, explicit=None):
    if explicit is None:
        return [{"complex_fingerprint_sha256": g, "algorithm": a, "seed": seed}
                for g in groups for a in algorithms for seed in seeds]
    if not isinstance(explicit, (list, tuple)) or not explicit:
        raise ValueError("planned_runs must be a nonempty list of identity records")
    result, seen = [], set()
    for record in explicit:
        if not isinstance(record, dict):
            raise ValueError("planned_runs contains a non-object identity")
        fp = _fingerprints([record.get("complex_fingerprint_sha256")], name="planned_runs")[0]
        algorithm, seed = record.get("algorithm"), record.get("seed")
        if fp not in groups or algorithm not in algorithms:
            raise ValueError("planned_runs contains an undeclared molecular group or algorithm")
        _integer(seed, "planned seed")
        if seed not in seeds:
            raise ValueError("planned_runs contains an undeclared seed")
        key = (fp, algorithm, seed)
        if key in seen:
            raise ValueError("planned_runs contains duplicate molecular/algorithm/seed identities")
        seen.add(key)
        row = {"complex_fingerprint_sha256": fp, "algorithm": algorithm, "seed": seed}
        if "run_id" in record:
            if not isinstance(record["run_id"], str) or not record["run_id"].strip():
                raise ValueError("planned run_id must be a nonempty string")
            row["run_id"] = record["run_id"]
        # Explicit known timings describe recorded failed jobs, never imputed zeros.
        for field in ("runtime_reduced_sec", "runtime_full_sec"):
            if field in record and record[field] is not None:
                row[field] = _number(record[field], field, minimum=0)
        result.append(row)
    if {r["complex_fingerprint_sha256"] for r in result} != set(groups):
        raise ValueError("Every planned molecular group must have a planned run")
    return sorted(result, key=_identity)


def _model_context(artifact_path, config_path):
    artifact_file = Path(artifact_path).resolve(strict=True)
    loaded = load_calibrator(artifact_file)
    artifact = loaded["artifact"]
    config_file = Path(config_path).resolve(strict=True)
    config = load_config(config_file)
    if config_sha256(config) != artifact["settings"]["source_config_sha256"]:
        raise ValueError("Configuration differs from the frozen model training context; use its original configuration")
    return loaded, artifact, config, {
        "artifact_path": str(artifact_file),
        "artifact_sha256": sha256_bytes(loaded["downloads"]["artifact.json"]),
        "manifest_sha256": sha256_bytes(loaded["downloads"]["audit_manifest.json"]),
        "config_path": str(config_file), "config_sha256": config_sha256(config),
        "source_config": config,
    }


def _prospective_context_issues(artifact, config):
    issues = []
    execution = config.get("execution_contract")
    quality = config.get("quality_metric")
    if not isinstance(execution, dict) or not execution:
        issues.append("missing_training_execution_contract")
    elif artifact["settings"].get("execution_contract") != execution:
        issues.append("model_does_not_bind_training_execution_contract")
    if not isinstance(quality, dict) or not quality:
        issues.append("missing_training_quality_metric")
    elif artifact["settings"].get("quality_metric") != quality:
        issues.append("model_does_not_bind_training_quality_metric")
    else:
        try:
            validate_quality_metric(quality)
        except ValueError:
            issues.append("unsupported_or_incomplete_standard_quality_metric")
    return issues


def prepare_decision_protocol(artifact_path, validation_groups, test_groups, out_dir, *,
        config_path, minimum_preservation=.90, minimum_coverage=0.,
        minimum_accepted_complexes=1, score_column="prediction", candidate_thresholds=None,
        prior_exposure_groups=(), analysis_role="prospective", random_seed=2026,
        planned_runs=None, algorithms=("ag", "pso", "hybrid"), seeds=(1,),
        bootstrap_repetitions=2000, require_runtime_improvement=False):
    """Commit roles, metric, candidate cutoffs and criteria without reading labels.

    ``planned_runs`` optionally maps ``validation`` and ``test`` to identity
    records. Otherwise the declared algorithms/seeds/group Cartesian product is
    committed. Missing result rows stay in these planned denominators.
    """
    if analysis_role not in {"prospective", "exploratory"}:
        raise ValueError("analysis_role must be prospective or exploratory")
    if score_column not in _SCORES:
        raise ValueError("score_column must be prediction or base_raw_score")
    preservation = _number(minimum_preservation, "minimum_preservation", minimum=0, maximum=1)
    coverage = _number(minimum_coverage, "minimum_coverage", minimum=0, maximum=1)
    _integer(minimum_accepted_complexes, "minimum_accepted_complexes", 1)
    _integer(random_seed, "random_seed")
    _integer(bootstrap_repetitions, "bootstrap_repetitions", 0)
    if type(require_runtime_improvement) is not bool:
        raise ValueError("require_runtime_improvement must be boolean")
    validation = _fingerprints(validation_groups, name="validation_groups")
    test = _fingerprints(test_groups, name="test_groups")
    exposed = set(_fingerprints(prior_exposure_groups, name="prior_exposure_groups", allow_empty=True))
    ledger_groups, ledger_snapshot = _exposure_snapshot()
    exposed.update(ledger_groups)
    registry = None
    if _REGISTRY.exists():
        registry_bytes = read_bytes_once(_REGISTRY)
        value = json.loads(registry_bytes)
        known = _fingerprints(value.get("known_fingerprints"), name="registry known_fingerprints", allow_empty=True)
        exposed.update(known)
        registry = {"path": str(_REGISTRY), "sha256": sha256_bytes(registry_bytes), "known_group_count": len(known)}
    if set(validation) & set(test):
        raise ValueError("Policy-validation and final-test molecular groups overlap")
    loaded, artifact, config, model = _model_context(artifact_path, config_path)
    fitted = {g for role in artifact["partition"].values() for g in role["groups"]}
    if fitted & (set(validation) | set(test)):
        raise ValueError("Declared validation/test groups overlap model fitting or calibration")
    observed_roles = {"validation": sorted(set(validation) & exposed), "test": sorted(set(test) & exposed)}
    context_issues = _prospective_context_issues(artifact, config)
    if analysis_role == "prospective" and any(observed_roles.values()):
        raise ValueError("Previously exposed molecular groups cannot be declared as a new independent test; use exploratory analysis or new groups")
    if analysis_role == "prospective" and context_issues:
        raise ValueError("Model is not ready for prospective decisions: " + ", ".join(context_issues)
                         + ". Create a model with a bound standard quality metric and matching execution contract.")
    if (not isinstance(algorithms, (list, tuple)) or not algorithms
            or len(set(algorithms)) != len(algorithms)
            or any(a not in {"ag", "pso", "hybrid"} for a in algorithms)):
        raise ValueError("algorithms must be distinct supported algorithm names")
    if not isinstance(seeds, (list, tuple)) or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    for seed in seeds:
        _integer(seed, "seed")
    if planned_runs is not None and (not isinstance(planned_runs, dict)
                                    or set(planned_runs) != {"validation", "test"}):
        raise ValueError("planned_runs must map validation and test to frozen identity lists")
    plans = {role: _planned_records(groups, algorithms, seeds,
             None if planned_runs is None else planned_runs[role])
             for role, groups in (("validation", validation), ("test", test))}
    if candidate_thresholds is None:
        lower, upper = artifact["target"]["outcome_bounds"]
        thresholds = sorted(set([float(x) for x in np.linspace(lower, upper, 101)] + [0.0]))
    else:
        if not isinstance(candidate_thresholds, (list, tuple)) or not candidate_thresholds:
            raise ValueError("candidate_thresholds must be a nonempty numeric list")
        thresholds = sorted(set(_number(x, "candidate threshold") for x in candidate_thresholds))
    root = _new_output(out_dir, artifact_path)
    protocol = _record({
        "schema_version": PROTOCOL_SCHEMA, "created_at_utc": _utc(), "analysis_role": analysis_role,
        "model": model, "roles": {"validation_groups": validation, "test_groups": test},
        "prior_exposure_groups": sorted(exposed), "exposed_role_groups": observed_roles,
        "exposure_registry": registry, "prospective_context_issues": context_issues,
        "exposure_ledger": ledger_snapshot,
        "algorithms": list(algorithms), "seeds": list(seeds), "planned_runs": plans,
        "score_column": score_column, "candidate_thresholds": thresholds,
        "threshold_rule": "score_greater_than_or_equal_threshold",
        "criteria": {"minimum_preservation": preservation, "preservation_comparison": "strictly_greater_than",
            "minimum_coverage": coverage, "minimum_accepted_complexes": minimum_accepted_complexes,
            "require_selective_rule": True, "require_empirical_baseline_improvement": True,
            "require_runtime_improvement": require_runtime_improvement},
        "selection_objective": "maximum_planned_run_coverage_subject_to_criteria_then_minimum_risk_then_lowest_threshold",
        "uncertainty": {"method": "molecular_cluster_percentile_bootstrap", "repetitions": bootstrap_repetitions,
            "random_seed": random_seed, "confidence_level": .95},
        "guarantee": "empirical prespecified criteria only; no population-risk or individual-outcome guarantee",
        "commitment_scope": "local integrity commitment, not independently timestamped preregistration",
        "status": "prepared", "decision_capable": False,
    })
    path = root / "decision_protocol.json"
    _write_new(path, protocol)
    if loaded["downloads"] != load_calibrator(artifact_path)["downloads"]:
        raise ValueError("Frozen model changed during preparation")
    return {"protocol": protocol, "protocol_path": str(path)}


def _checked_protocol(value):
    protocol = _verify_record(value, PROTOCOL_SCHEMA)
    artifact_path = protocol["model"]["artifact_path"]
    loaded = load_calibrator(artifact_path)
    if (sha256_bytes(loaded["downloads"]["artifact.json"]) != protocol["model"]["artifact_sha256"]
            or sha256_bytes(loaded["downloads"]["audit_manifest.json"]) != protocol["model"]["manifest_sha256"]):
        raise ValueError("Frozen model changed after protocol preparation")
    config = protocol["model"]["source_config"]
    if (config_sha256(config) != protocol["model"]["config_sha256"]
            or loaded["artifact"]["settings"]["source_config_sha256"] != protocol["model"]["config_sha256"]):
        raise ValueError("Frozen source configuration changed after preparation")
    return protocol, loaded


def _check_row_context(frame, protocol, *, paired):
    config = protocol["model"]["source_config"]
    if not frame["config_sha256"].eq(protocol["model"]["config_sha256"]).all():
        raise ValueError("Input configuration differs from the frozen protocol/model")
    expected = {}
    if config.get("execution_contract"):
        expected["execution_contract_sha256"] = contract_sha256(config["execution_contract"])
    quality = config.get("quality_metric")
    if paired and quality:
        expected["quality_metric_sha256"] = contract_sha256(quality)
    for column, digest in expected.items():
        if column not in frame or not frame[column].astype(str).eq(digest).all():
            raise ValueError(f"Input is missing or incompatible with frozen {column}")
    if paired:
        validate_paired_quality_context(frame, quality)
    for field in ("scientific_reduced", "scientific_full") if paired else ("scientific_reduced",):
        if field in frame and not frame[field].eq(True).all():
            raise ValueError("Decision evaluation requires scientific docking provenance")
    allowed = config.get("authorization", {}).get("trusted_engine_allowlist", [])
    if protocol["analysis_role"] == "prospective" and allowed:
        for field in ("engine_reduced", "engine_full") if paired else ("engine_reduced",):
            if not frame[field].isin(allowed).all():
                raise ValueError("Docking engine is not in the frozen trusted engine list")


def _stage_dataset(protocol, loaded, dataset_path, output, role, expected_digest=None, planned_runs=None):
    data_file = Path(dataset_path).resolve(strict=True)
    content = read_bytes_once(data_file)
    digest = sha256_bytes(content)
    if expected_digest is not None and digest != expected_digest:
        raise ValueError("Dataset differs from its expected SHA-256 snapshot")
    plan = protocol["planned_runs"][role]
    if planned_runs is not None:
        supplied = _planned_records(protocol["roles"][role + "_groups"], protocol["algorithms"],
                                    protocol["seeds"], planned_runs)
        if supplied != plan:
            raise ValueError("Planned run identities/timings cannot change after preparation")
    artifact = loaded["artifact"]
    settings = artifact["settings"]
    context = settings["features"]["columns"]
    requested = [c for c in artifact["source_feature_columns"] if c in context]
    raw_frame = pd.read_csv(BytesIO(content), dtype=str)
    if raw_frame.empty:
        required = set(TRAINING_BASE_COLUMNS + requested + ["config_sha256", *_IDENTITY])
        if required - set(raw_frame.columns):
            raise ValueError("Empty dataset must retain the declared paired input columns")
        frame = raw_frame
    else:
        frame = load_paired_dataset(BytesIO(content), requested, settings["contact_graph"],
            docking_task=settings["docking_task"], feature_schema=settings["features"]["schema_version"],
            context_feature_columns=context)
        _check_row_context(frame, protocol, paired=True)
    allowed_groups = set(protocol["roles"][role + "_groups"])
    actual_groups = set(frame.complex_fingerprint_sha256.astype(str))
    if not actual_groups.issubset(allowed_groups):
        raise ValueError(f"Dataset contains molecular groups outside its declared {role} role")
    planned_by_key = {_identity(r): r for r in plan}
    keys = [_identity(row) for row in frame.to_dict("records")]
    if len(keys) != len(set(keys)):
        raise ValueError("Paired dataset repeats a planned molecular/algorithm/seed identity")
    if not set(keys).issubset(planned_by_key):
        raise ValueError("Dataset contains unplanned algorithms/seeds/runs")
    for row in frame.to_dict("records"):
        declared = planned_by_key[_identity(row)]
        if "run_id" in declared and row["run_id"] != declared["run_id"]:
            raise ValueError("run_id differs from its predeclared run identity")
    quality_source = {}
    if protocol["analysis_role"] == "prospective" and not frame.empty:
        metric = protocol["model"]["source_config"]["quality_metric"]
        verification = verify_standard_quality_records(frame, data_file.parent, metric)
        quality_source = {"quality_records_root": str(data_file.parent),
            "quality_metric": deepcopy(metric), "quality_records_verification": verification,
            "quality_records_scope": "relative evaluator records remain at the immutable source dataset root"}
    exposure_source = _commit_exposure(protocol, role, digest)
    root = _new_output(output, protocol["model"]["artifact_path"])
    snapshot = root / "inputs/dataset.csv"
    snapshot.parent.mkdir()
    with snapshot.open("xb") as handle:
        handle.write(content)
    predictions_file = root / "predictions.csv"
    if frame.empty:
        predictions = frame.copy()
        for column in ("prediction", "base_raw_score", "prediction_lower", "prediction_upper"):
            predictions[column] = pd.Series(dtype=float)
        predictions.to_csv(predictions_file, index=False)
    else:
        predictions = predict_calibrator(protocol["model"]["artifact_path"], snapshot, predictions_file)
        if (len(predictions) != len(frame) or
                not predictions[_IDENTITY].astype(str).reset_index(drop=True).equals(
                    frame[_IDENTITY].astype(str).reset_index(drop=True))):
            raise ValueError("Model predictions and paired outcome identities are misaligned")
    if sha256_bytes(read_bytes_once(data_file)) != digest:
        raise ValueError("Dataset changed while predictions were generated")
    _checked_protocol(protocol)
    return root, frame, predictions, {"path": str(data_file), "sha256": digest,
        "snapshot_path": str(snapshot), "size_bytes": len(content), "role": role,
        "predictions_path": str(predictions_file), "predictions_sha256": sha256_bytes(read_bytes_once(predictions_file)),
        **quality_source, **exposure_source}


def _timings(frame, plan, accepted, *, include_confidence_overhead=True):
    rows = {_identity(row): row for row in frame.to_dict("records")}
    keys = list(rows)
    accepted_keys = {key for key, flag in zip(keys, accepted) if flag}
    total, full_total = 0., 0.
    missing = []
    for planned in plan:
        key = _identity(planned)
        source = rows.get(key, planned)
        for name in ("runtime_reduced_sec", "runtime_full_sec"):
            value = source.get(name)
            try:
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError()
            except (TypeError, ValueError):
                missing.append({"identity": list(key), "field": name})
                continue
            if name == "runtime_reduced_sec":
                total += value
            else:
                full_total += value
                if key not in accepted_keys:
                    total += value
    # A calibrated decision needs measured descriptor/decision overhead, not a
    # silent zero. This column is optional for structural diagnostics.
    overhead = 0.
    for key, row in rows.items() if include_confidence_overhead else []:
        try:
            value = float(row["confidence_overhead_sec"])
            if not math.isfinite(value) or value < 0:
                raise ValueError()
            overhead += value
        except (KeyError, TypeError, ValueError):
            missing.append({"identity": list(key), "field": "confidence_overhead_sec"})
    complete = not missing
    return {"runtime_total_sec": total + overhead if complete else None,
        "runtime_complete": complete, "full_baseline_runtime_sec": full_total if complete else None,
        "runtime_missing_count": len(missing), "runtime_scope": "all planned REDUCED plus rejected FULL" +
            (" plus measured confidence overhead" if include_confidence_overhead else " without a calibrator"),
        "runtime_estimation": "counterfactual replay of recorded engine runtimes and available confidence overhead; "
            "not total wall time or observed deployment savings",
        "evaluation_collection_saved_full_runs": 0,
        "evaluation_collection_scope": "FULL outcomes were collected to evaluate the paired test; "
            "the policy did not avoid those evaluation executions"}


def _metrics(frame, plan, accepted, preserved, *, include_confidence_overhead=True):
    accepted = np.asarray(accepted, bool)
    preserved = np.asarray(preserved, bool)
    n, k = int(accepted.sum()), int((accepted & preserved).sum())
    planned_groups = {r["complex_fingerprint_sha256"] for r in plan}
    accepted_groups = set(frame.loc[accepted, "complex_fingerprint_sha256"].astype(str))
    values = {"planned_runs": len(plan), "valid_runs": len(frame),
        "accepted_runs": n, "accepted_complexes": len(accepted_groups),
        "planned_complexes": len(planned_groups),
        "paired_complexes": int(frame.complex_fingerprint_sha256.nunique()),
        "failed_or_missing_runs": len(plan) - len(frame),
        "coverage": n/len(plan), "complex_coverage": len(accepted_groups)/len(planned_groups),
        "preserved_runs": k, "adverse_accepted_runs": n-k,
        "preservation": k/n if n else None, "risk": (n-k)/n if n else None,
        "preservation_per_planned_run": k/len(plan),
        "status": "evaluated" if n else "no_acceptance"}
    values.update(_timings(frame, plan, accepted, include_confidence_overhead=include_confidence_overhead))
    if n:
        groups = frame.loc[accepted, "complex_fingerprint_sha256"].astype(str).to_numpy()
        bad_groups = set(groups[~preserved[accepted]])
        values["accepted_complexes_with_any_adverse_run"] = len(bad_groups)
        values["complex_adverse_risk"] = len(bad_groups)/len(accepted_groups)
    else:
        values.update(accepted_complexes_with_any_adverse_run=0, complex_adverse_risk=None)
    return values


def _comparison(selected, baseline):
    available = selected["preservation"] is not None and baseline["preservation"] is not None
    gain = selected["preservation"] - baseline["preservation"] if available else None
    return {"preservation_gain": gain, "adverse_risk_reduction": gain,
        "coverage_difference": selected["coverage"] - baseline["coverage"],
        "selective": 0 < selected["accepted_runs"] < baseline["accepted_runs"],
        "baseline_equal": selected["accepted_runs"] == baseline["accepted_runs"],
        "comparison_scope": "same valid paired outcomes and same predeclared planned denominator"}


def _criteria(protocol, selected, baseline, difference):
    values = protocol["criteria"]
    runtime_met = (selected["runtime_complete"] and selected["runtime_total_sec"] < selected["full_baseline_runtime_sec"])
    return {"preservation_met": selected["preservation"] is not None and selected["preservation"] > values["minimum_preservation"],
        "coverage_met": selected["coverage"] >= values["minimum_coverage"],
        "minimum_complexes_met": selected["accepted_complexes"] >= values["minimum_accepted_complexes"],
        "selective_rule": difference["selective"],
        "empirical_baseline_improvement": difference["preservation_gain"] is not None and difference["preservation_gain"] > 0,
        "runtime_met": runtime_met if values["require_runtime_improvement"] else None}


def _all_criteria_met(criteria):
    return all(value is True for value in criteria.values() if value is not None)


def _cluster_intervals(frame, plan, accepted, preserved, settings):
    count = settings["repetitions"]
    if not count:
        return {"available": False, "reason": "bootstrap_disabled_by_protocol"}
    groups = sorted({r["complex_fingerprint_sha256"] for r in plan})
    if len(groups) < 2:
        return {"available": False, "reason": "fewer_than_two_molecular_groups"}
    valid_groups = frame.complex_fingerprint_sha256.astype(str).to_numpy()
    arrays = []
    for group in groups:
        included = valid_groups == group
        a, y = np.asarray(accepted)[included], np.asarray(preserved)[included]
        arrays.append([sum(r["complex_fingerprint_sha256"] == group for r in plan),
                       int(a.sum()), int((a & y).sum()), int(included.sum()), int(y.sum())])
    values = np.asarray(arrays, float)
    rng = np.random.default_rng(settings["random_seed"])
    samples = values[rng.integers(0, len(groups), size=(count, len(groups)))].sum(axis=1)
    planned, chosen, good, valid, base_good = samples.T
    usable = (chosen > 0) & (valid > 0)
    result = {"available": bool(usable.any()), "method": "molecular_cluster_percentile_bootstrap",
        "confidence_level": .95, "repetitions": count, "valid_preservation_draws": int(usable.sum()),
        "interpretation": "exploratory uncertainty; not a risk guarantee or simultaneous selection correction"}
    if usable.any():
        s, b = good[usable]/chosen[usable], base_good[usable]/valid[usable]
        result["selected_preservation_ci95"] = np.quantile(s, [.025, .975]).tolist()
        result["baseline_preservation_ci95"] = np.quantile(b, [.025, .975]).tolist()
        result["preservation_gain_ci95"] = np.quantile(s-b, [.025, .975]).tolist()
    result["coverage_ci95"] = np.quantile(chosen/planned, [.025, .975]).tolist()
    return result


def _outcomes(frame, artifact):
    if frame.empty:
        return np.array([], dtype=bool)
    return preservation_margin(frame.dockq_reduced.to_numpy(float), frame.dockq_full.to_numpy(float),
                               artifact["target"]["definition"]) >= 0


def _prediction_diagnostics(frame, predictions, artifact, score_column):
    if frame.empty:
        return {"available": False, "reason": "no_valid_paired_outcomes", "valid_runs": 0}
    margin = preservation_margin(frame.dockq_reduced.to_numpy(float), frame.dockq_full.to_numpy(float),
                                 artifact["target"]["definition"])
    point = predictions.prediction.to_numpy(float)
    scores = predictions[score_column].to_numpy(float)
    labels = margin >= 0
    difference = point - margin
    result = {"available": True, "valid_runs": len(frame),
        "target": "continuous_preservation_margin", "ranking_score_column": score_column,
        "mae": float(np.abs(difference).mean()), "rmse": float(np.sqrt(np.square(difference).mean())),
        "bias_prediction_minus_observed": float(difference.mean()),
        "observed_preserved_runs": int(labels.sum()), "observed_adverse_runs": int((~labels).sum()),
        "predicted_nonnegative_runs": int((point >= 0).sum()),
        "score_interpretation": "margin ordering, not calibrated success probability"}
    if labels.any() and (~labels).any():
        precision, recall, _ = precision_recall_curve(labels, scores)
        result.update(roc_auc=float(roc_auc_score(labels, scores)),
                      pr_auc_trapezoidal=float(auc(recall, precision)),
                      average_precision=float(average_precision_score(labels, scores)))
    else:
        result.update(roc_auc=None, pr_auc_trapezoidal=None, average_precision=None,
                      ranking_unavailable_reason="single_observed_preservation_class")
    return result


def _summary(protocol, artifact, frame, predictions, role, threshold):
    plan = protocol["planned_runs"][role]
    y = _outcomes(frame, artifact)
    scores = predictions[protocol["score_column"]].to_numpy(float)
    accepted = np.zeros(len(frame), dtype=bool) if threshold is None else scores >= threshold
    selected = _metrics(frame, plan, accepted, y)
    baseline = _metrics(frame, plan, np.ones(len(frame), dtype=bool), y, include_confidence_overhead=False)
    difference = _comparison(selected, baseline)
    criteria = _criteria(protocol, selected, baseline, difference)
    return {"selected": selected, "baseline": baseline, "difference": difference,
        "criteria": criteria, "prediction_diagnostics": _prediction_diagnostics(frame, predictions, artifact, protocol["score_column"]),
        "uncertainty": _cluster_intervals(frame, plan, accepted, y, protocol["uncertainty"])}, accepted, y


def select_decision_policy(protocol_path, validation_dataset_path, out_dir, *,
                           expected_dataset_sha256=None, planned_runs=None):
    """Select a cutoff exclusively on the committed policy-development role."""
    protocol, loaded = _checked_protocol(_read_record(protocol_path, PROTOCOL_SCHEMA))
    claim = Path(protocol_path).resolve().parent / "policy_selection_claim.json"
    if claim.exists():
        raise FileExistsError("A policy is already frozen for this protocol; its development data cannot be replaced")
    root, frame, predictions, source = _stage_dataset(protocol, loaded, validation_dataset_path, out_dir,
                                                       "validation", expected_dataset_sha256, planned_runs)
    artifact = loaded["artifact"]
    y = _outcomes(frame, artifact)
    plan = protocol["planned_runs"]["validation"]
    baseline = _metrics(frame, plan, np.ones(len(frame), dtype=bool), y, include_confidence_overhead=False)
    scores = predictions[protocol["score_column"]].to_numpy(float)
    candidates = []
    eligible = []
    for threshold in protocol["candidate_thresholds"]:
        selected = _metrics(frame, plan, scores >= threshold, y)
        difference = _comparison(selected, baseline)
        criteria = _criteria(protocol, selected, baseline, difference)
        row = {"threshold": threshold, "selected": selected, "difference": difference,
               "criteria": criteria, "eligible": _all_criteria_met(criteria)}
        candidates.append(row)
        if row["eligible"]:
            eligible.append(row)
    chosen = (sorted(eligible, key=lambda r: (-r["selected"]["coverage"], r["selected"]["risk"], r["threshold"]))[0]
              if eligible else None)
    threshold = chosen["threshold"] if chosen else None
    report, accepted, y = _summary(protocol, artifact, frame, predictions, "validation", threshold)
    report.update({"scope": "policy development only; not independent validation", "source": source,
        "status": "selected" if chosen else "no_useful_rule", "decision_capable": False,
        "candidate_count": len(candidates), "eligible_candidate_count": len(eligible),
        "failure_reason": None if chosen else "no candidate jointly met preservation, coverage, selectivity and empirical baseline improvement"})
    policy = _record({"schema_version": POLICY_SCHEMA, "created_at_utc": _utc(),
        "protocol": protocol, "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": protocol["record_sha256"], "validation_source": source,
        "status": report["status"], "threshold": threshold, "score_column": protocol["score_column"],
        "action_without_supported_policy": "RUN_FULL", "decision_capable": False,
        "validation_report": report, "selection_uses_final_test": False,
        "guarantee": protocol["guarantee"]})
    policy_file, report_file = root / "decision_policy.json", root / "validation_report.json"
    _write_new(policy_file, policy)
    _write_new(report_file, _record({"schema_version": "gva_decision_development_report_v1", **report}))
    _write_new(root / "policy_candidates.json", {"candidates": candidates})
    predictions["observed_preserved"] = y
    predictions["observed_margin"] = preservation_margin(frame.dockq_reduced.to_numpy(float), frame.dockq_full.to_numpy(float),
        artifact["target"]["definition"]) if len(frame) else np.array([], dtype=float)
    predictions["accepted_diagnostic_only"] = accepted
    predictions.to_csv(root / "validation_decisions.csv", index=False)
    _write_new(claim, {"policy_path": str(policy_file), "policy_sha256": policy["record_sha256"],
                       "validation_dataset_sha256": source["sha256"], "created_at_utc": _utc()})
    return {"policy": policy, "policy_path": str(policy_file), "report": report,
            "report_path": str(report_file), "predictions_path": str(root / "validation_decisions.csv")}


def _checked_policy(policy_path):
    policy = _read_record(policy_path, POLICY_SCHEMA)
    protocol, loaded = _checked_protocol(policy["protocol"])
    if policy["protocol_sha256"] != protocol["record_sha256"]:
        raise ValueError("Policy is linked to a different frozen protocol")
    if policy["score_column"] != protocol["score_column"]:
        raise ValueError("Policy score changed after commitment")
    if policy["threshold"] is not None and policy["threshold"] not in protocol["candidate_thresholds"]:
        raise ValueError("Policy threshold was not declared before evaluation")
    claim_path = Path(policy["protocol_path"]).resolve().parent / "policy_selection_claim.json"
    if not claim_path.is_file() or json.loads(read_bytes_once(claim_path)).get("policy_sha256") != policy["record_sha256"]:
        raise ValueError("Policy is not the frozen selection recorded for this protocol")
    _verify_source_snapshot(policy["validation_source"])
    return policy, protocol, loaded


def _verify_source_snapshot(source):
    checks = [("snapshot_path", "sha256"), ("predictions_path", "predictions_sha256")]
    if "exposure_record_path" in source:
        checks.append(("exposure_record_path", "exposure_record_file_sha256"))
    for path_key, digest_key in checks:
        try:
            observed = sha256_bytes(read_bytes_once(source[path_key]))
        except OSError as exc:
            raise ValueError("Committed decision evidence is unavailable") from exc
        if observed != source[digest_key]:
            raise ValueError("Committed decision evidence snapshot changed after it was frozen")
    if "quality_records_root" in source:
        frame = pd.read_csv(source["snapshot_path"], float_precision="round_trip")
        verify_standard_quality_records(frame, source["quality_records_root"], source["quality_metric"])


def evaluate_frozen_policy(policy_path, test_dataset_path, out_dir, *,
                           expected_dataset_sha256=None, planned_runs=None):
    """Evaluate once on the fixed test role; never choose or update a cutoff."""
    policy, protocol, loaded = _checked_policy(policy_path)
    claim = Path(policy_path).resolve().parent / "final_evaluation_claim.json"
    if claim.exists():
        raise FileExistsError("This frozen policy has already consumed its final evaluation; reusing the test cannot be presented as a new validation")
    root, frame, predictions, source = _stage_dataset(protocol, loaded, test_dataset_path, out_dir,
                                                       "test", expected_dataset_sha256, planned_runs)
    if source["sha256"] == policy["validation_source"]["sha256"]:
        raise ValueError("Final test cannot reuse policy-development data")
    report, accepted, y = _summary(protocol, loaded["artifact"], frame, predictions, "test", policy["threshold"])
    prospective = protocol["analysis_role"] == "prospective" and not protocol["prospective_context_issues"]
    passed = policy["status"] == "selected" and _all_criteria_met(report["criteria"])
    capable = bool(passed and prospective)
    report.update({"schema_version": EVALUATION_SCHEMA, "created_at_utc": _utc(),
        "policy_path": str(Path(policy_path).resolve()), "policy_sha256": policy["record_sha256"],
        "protocol_sha256": protocol["record_sha256"], "source": source,
        "status": ("passed" if passed else "failed") if prospective else "exploratory",
        "empirical_criteria_met": bool(passed), "prospective_eligible": bool(prospective),
        "decision_capable": capable, "automatic_installation": False,
        "scope": "frozen final test; cutoff unchanged", "guarantee": protocol["guarantee"]})
    report = _record(report)
    report_file = root / "final_evaluation.json"
    predictions["observed_preserved"] = y
    predictions["observed_margin"] = preservation_margin(frame.dockq_reduced.to_numpy(float), frame.dockq_full.to_numpy(float),
        loaded["artifact"]["target"]["definition"]) if len(frame) else np.array([], dtype=float)
    predictions["accepted_diagnostic_only"] = accepted
    predictions.to_csv(root / "test_decisions.csv", index=False)
    _write_new(report_file, report)
    _write_new(claim, {"report_path": str(report_file), "report_sha256": report["record_sha256"],
                       "test_dataset_sha256": source["sha256"], "created_at_utc": _utc()})
    return {"report": report, "report_path": str(report_file), "predictions_path": str(root / "test_decisions.csv")}


def predict_decisions(policy_path, input_path, output_path, *, evaluation_path=None):
    """Use REDUCED-only descriptors; a linked passing final test is mandatory.

    Missing approval produces RUN_FULL for every input row. A verified approval
    permits only the already-frozen score rule; no reference/FULL labels are
    requested or consulted during inference.
    """
    policy, protocol, loaded = _checked_policy(policy_path)
    output = Path(output_path).resolve()
    protected = {Path(input_path).resolve(), Path(policy_path).resolve(),
                 Path(protocol["model"]["artifact_path"]).resolve()}
    if evaluation_path is not None:
        protected.add(Path(evaluation_path).resolve())
    if output in protected or output.exists():
        raise FileExistsError("Decision output must be new and separate from inputs and model artifacts")
    model_root = Path(protocol["model"]["artifact_path"]).resolve().parent
    if output == model_root or model_root in output.parents:
        raise ValueError("Decision outputs cannot modify the frozen model directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".gva_predictions.csv")
    if temporary.exists():
        raise FileExistsError("Temporary prediction output already exists")
    input_bytes = read_bytes_once(input_path)
    snapshot = output.with_name(output.name + ".input.csv")
    with snapshot.open("xb") as handle:
        handle.write(input_bytes)
    predictions = predict_calibrator(protocol["model"]["artifact_path"], snapshot, temporary)
    # predict_calibrator has already validated the reduced loader/graph context.
    source = pd.read_csv(BytesIO(input_bytes), dtype=str)
    if "execution_contract_sha256" in source:
        expected = contract_sha256(protocol["model"]["source_config"].get("execution_contract"))
        if not source.execution_contract_sha256.eq(expected).all():
            raise ValueError("REDUCED execution contract differs from frozen decision protocol")
    elif protocol["analysis_role"] == "prospective":
        raise ValueError("REDUCED inference must bind execution_contract_sha256")
    if (not predictions.scientific_reduced.eq(True).all()
            or not predictions.algorithm.isin(protocol["algorithms"]).all()
            or not predictions.seed.isin(protocol["seeds"]).all()):
        raise ValueError("REDUCED provenance, algorithm or seed differs from frozen decision protocol")
    allowed_engines = protocol["model"]["source_config"].get("authorization", {}).get("trusted_engine_allowlist", [])
    if protocol["analysis_role"] == "prospective" and allowed_engines and not predictions.engine_reduced.isin(allowed_engines).all():
        raise ValueError("REDUCED engine differs from frozen trusted engine list")
    authorized = False
    reason = "independent_final_evaluation_not_approved"
    if evaluation_path is not None:
        evaluation = _read_record(evaluation_path, EVALUATION_SCHEMA)
        if (evaluation["policy_sha256"] != policy["record_sha256"]
                or evaluation["protocol_sha256"] != protocol["record_sha256"]):
            raise ValueError("Final evaluation belongs to a different frozen policy")
        claim_file = Path(policy_path).resolve().parent / "final_evaluation_claim.json"
        if not claim_file.is_file() or json.loads(read_bytes_once(claim_file)).get("report_sha256") != evaluation["record_sha256"]:
            raise ValueError("Final approval is not the recorded final evaluation for this policy")
        _verify_source_snapshot(evaluation["source"])
        authorized = evaluation.get("decision_capable") is True and evaluation.get("status") == "passed"
        reason = "frozen_rule_after_prespecified_empirical_test" if authorized else "final_evaluation_did_not_approve_policy"
    scores = predictions[policy["score_column"]].to_numpy(float)
    accepted = np.zeros(len(predictions), bool)
    if authorized and policy["status"] == "selected" and policy["threshold"] is not None:
        accepted = scores >= policy["threshold"]
    known = {g for role in loaded["artifact"]["partition"].values() for g in role["groups"]}
    known.update(protocol["roles"]["validation_groups"])
    known.update(protocol["roles"]["test_groups"])
    known.update(protocol["prior_exposure_groups"])
    ledger_groups, _ = _exposure_snapshot()
    known.update(ledger_groups)
    exposed = predictions.complex_fingerprint_sha256.astype(str).isin(known).to_numpy()
    accepted &= ~exposed
    predictions["action"] = np.where(accepted, "ACCEPT_REDUCED", "RUN_FULL")
    predictions["decision_reason"] = np.where(accepted, reason,
        "below_frozen_margin_cutoff" if authorized else reason)
    predictions.loc[exposed, "decision_reason"] = "development_or_evaluation_complex_not_eligible_for_new_decision"
    predictions["decision_policy_sha256"] = policy["record_sha256"]
    predictions["decision_capable"] = bool(authorized)
    if sha256_bytes(read_bytes_once(input_path)) != sha256_bytes(input_bytes):
        raise ValueError("REDUCED input changed during decision inference")
    predictions.to_csv(output, index=False)
    # Keep the separately named margin predictions as an audit record. They do
    # not contain experimental labels or FULL inputs.
    return predictions
