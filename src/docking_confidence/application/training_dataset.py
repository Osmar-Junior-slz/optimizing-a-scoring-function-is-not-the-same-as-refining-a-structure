"""Read a single immutable batch into a provenance-checked paired CSV.

This service does no docking, label-dependent selection, imputation or training.
Invalid records remain visible in the report and invalidate the preparation;
missing counterparts and explicitly failed runs are accounted for separately.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from docking_confidence.application.batch import (
    BATCH_SCHEMA_VERSION, MANIFEST_FILENAME, _file_sha256, _load_json_object,
    _safe_component, _task_directory, _task_sha256, _valid_completed_result,
    _validate_manifest, _payload_sha256,
)
from docking_confidence.application.reduced_evidence import (
    _GRAPH_COLUMNS, MissingReducedFeaturesError as _MissingFeaturesError,
    read_reduced_graph as _read_graph,
)
from docking_confidence.application.progress import ProgressCallback, emit_progress
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.domain.structure import CanonicalComplexFingerprint
from docking_confidence.io.config import (
    DOCKING_TASK_FEATURE_SCHEMAS, LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION, config_sha256, load_config,
)
from docking_confidence.features.execution_context import (
    EXECUTION_CONTEXT_METADATA_COLUMNS, local_execution_context,
)
from docking_confidence.io.dataset import TRAINING_BASE_COLUMNS, load_paired_dataset
from docking_confidence.risk.policy import threshold_grid


_CONTEXT_COLUMNS = (
    "complex_fingerprint_schema", "complex_fingerprint_sha256",
    "docking_task", "feature_schema", "config", "config_sha256",
)


def _read_receipt(directory: Path, manifest: dict, target: dict, job: dict) -> dict | None:
    path = directory / "receipt.json"
    if not path.exists():
        return None
    task_hash = _task_sha256(plan_sha256=manifest["plan_sha256"], target=target, job=job)
    if not _valid_completed_result(path, task_hash):
        raise ValueError("receipt failed its task/record integrity hash")
    receipt = _load_json_object(path)
    for key, expected in {
        "batch_id": manifest["batch_id"], "plan_sha256": manifest["plan_sha256"],
        "complex_id": target["complex_id"], "target": target, "job": job,
    }.items():
        if receipt.get(key) != expected:
            raise ValueError(f"receipt {key} disagrees with batch plan")
    result = receipt["engine_result"]
    expected = {key: job[key] for key in ("job_id", "algorithm", "protocol", "seed")}
    expected.update({key: job["profile"][key] for key in (
        "generations", "population", "top_pockets", "evaluation_budget",
    )})
    expected.update(engine=manifest["plan"]["engine"]["name"],
                    scientific=manifest["plan"]["engine"]["scientific"])
    for key, value in expected.items():
        if result.get(key) != value or type(result.get(key)) is not type(value):
            raise ValueError(f"engine_result {key} disagrees with batch plan")
    if result.get("exit_code") not in (None, 0):
        raise ValueError("successful receipt has a failing engine exit code")
    return receipt


def _read_failure(directory: Path, manifest: dict, target: dict, job: dict) -> dict | None:
    path = directory / "error.json"
    if not path.exists():
        return None
    error = _load_json_object(path)
    expected = {
        "schema_version": BATCH_SCHEMA_VERSION, "status": "FAILED",
        "batch_id": manifest["batch_id"], "plan_sha256": manifest["plan_sha256"],
        "task_sha256": _task_sha256(plan_sha256=manifest["plan_sha256"], target=target, job=job),
        "complex_id": target["complex_id"], "job_id": job["job_id"],
    }
    if any(error.get(key) != value for key, value in expected.items()):
        raise ValueError("failed run record disagrees with batch plan")
    hard_failure = job["protocol"] == "reduced" and error.get("engine_status") == "REDUCED_FAILED"
    return {
        "complex_id": target["complex_id"], "run_id": job["job_id"],
        "algorithm": job["algorithm"], "seed": job["seed"],
        "protocol": job["protocol"], "engine_status": error.get("engine_status"),
        "category": "deterministic_fallback" if hard_failure else "execution_failure",
        "reason": "reduced_hard_failure" if hard_failure else error.get("error_type"),
        "failure_reason": error.get("failure_reason"), "message": error.get("message"),
        "source": str(path),
    }


def build_training_dataset(
    batch_dir: str | Path, output_dir: str | Path, profile: str = "pilot",
    *, fingerprint_alias_policy: str = "reject",
    factorial_initialization: str = "uniform",
    config_path: str | Path | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Prepare a new, immutable snapshot; ``valid`` controls training eligibility.

    A candidate CSV may exist for diagnostics even when ``valid`` is false. It
    must never be confused with the authorized ``dataset_path`` (then None).
    ``first_complex_id`` explicitly selects one whole target per fingerprint
    from the manifest, before reading outcomes. Failed representatives are never
    replaced by another alias, and all receipts remain subject to integrity checks.
    """
    if profile not in {"pilot", "default", "operational"}:
        raise ValueError("profile must be pilot, default or operational")
    if fingerprint_alias_policy not in {"reject", "first_complex_id"}:
        raise ValueError("fingerprint_alias_policy must be reject or first_complex_id")
    if factorial_initialization not in {"uniform", "seeded"}:
        raise ValueError("factorial_initialization must be uniform or seeded")
    emit_progress(progress_callback, 0.0, "Validando pasta e manifesto do batch")
    root = Path(batch_dir).resolve(strict=True)
    destination = Path(output_dir).resolve(strict=False)
    if root == destination or root in destination.parents or destination in root.parents:
        raise ValueError("dataset preparation output must be separate from the batch")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError("dataset preparation already exists in this directory")
    destination.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "paired_batch_training_dataset_v1", "valid": False,
        "batch_dir": str(root), "profile": profile, "dataset_path": None,
        "candidate_dataset_path": None, "report_path": str(destination / "dataset_build_report.json"),
        "config_path": None, "config_sha256": None, "docking_task": None,
        "feature_schema": None, "feature_columns": [], "dataset_sha256": None,
        "loader_passed": False, "issues": [], "warnings": [], "failures": [], "unmatched": [],
        "fingerprint_alias_policy": fingerprint_alias_policy,
        "alias_resolutions": [],
        "missing_feature_details": [], "preview": {},
        "counts": dict.fromkeys((
            "complexes_found", "valid_pairs", "reduced_succeeded", "reduced_failed",
            "unmatched_pairs", "usable_rows", "independent_complexes", "missing_features",
            "provenance_errors", "ambiguous_pairs", "candidate_rows",
            "resolved_alias_groups", "excluded_alias_targets", "excluded_alias_planned_runs",
        ), 0),
    }
    counts = report["counts"]

    def issue(code: str, message: str, **details: Any) -> None:
        report["issues"].append({"code": code, "message": message, **details})
        if code != "missing_features":
            counts["provenance_errors"] += 1

    def finish() -> dict:
        report["valid"] = (report["loader_passed"] and not report["issues"]
                           and counts["candidate_rows"] > 0)
        counts["usable_rows"] = counts["candidate_rows"] if report["valid"] else 0
        if report["valid"]:
            report["dataset_path"] = report["candidate_dataset_path"]
        with Path(report["report_path"]).open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, allow_nan=False, indent=2)
        emit_progress(progress_callback, 1.0,
                      "Preparação concluída" if report["valid"] else "Análise concluída: dataset bloqueado")
        return report

    try:
        manifest = _load_json_object(root / MANIFEST_FILENAME)
        _validate_manifest(manifest)
        plan = manifest["plan"]
        targets, jobs = plan["targets"], plan["jobs"]
        if plan.get("experimental_design") == "factorial_2x2":
            from docking_confidence.application.experimental_protocol import bind_experimental_protocol
            binding = plan.get("experimental_protocol", {})
            unbound_plan = {key: value for key, value in plan.items() if key != "experimental_protocol"}
            checked_binding = bind_experimental_protocol(binding["path"], unbound_plan)
            if checked_binding != binding:
                raise ValueError("factorial protocol changed after the batch commitment")
            jobs = [job for job in jobs if job["profile"].get("initialization") == factorial_initialization]
            report["experimental_initialization"] = factorial_initialization
            report["experimental_protocol"] = binding
            report["warnings"].append({"code": "factorial_matched_pair", "message": "Only matched reduced/expanded runs with the selected initialization are used; the other conditions remain in the factorial report."})
        if not targets or not jobs:
            raise ValueError("batch plan has no targets or jobs")
        report["batch_id"] = manifest["batch_id"]
        report["batch_manifest_sha256"] = manifest["manifest_sha256"]
        report["engine_provenance"] = plan["engine"]
        counts["complexes_found"] = len(targets)
        tasks = {job["profile"]["docking_task"] for job in jobs}
        if len(tasks) != 1 or not tasks.issubset(DOCKING_TASK_FEATURE_SCHEMAS):
            raise ValueError(f"mixed or unsupported docking_task: {sorted(tasks)}")
        task = next(iter(tasks))
        config_name = f"local_refinement_{profile}.yaml" if task == "local_refinement" else f"{profile}.yaml"
        config_path = (Path(config_path).resolve(strict=True) if config_path is not None
                       else Path(__file__).resolve().parents[3] / "configs" / config_name)
        config = load_config(config_path)
        from docking_confidence.application.operational import validate_training_regime
        validate_training_regime(config, plan)
        if config.get("docking_task") is not None and config["docking_task"] != task:
            raise ValueError("selected training configuration disagrees with the batch docking task")
        config["docking_task"] = task
        digest = config_sha256(config)
        feature_columns = config["features"]["columns"]
        report.update(config_path=str(config_path), config_sha256=digest, docking_task=task,
                      feature_schema=config["features"]["schema_version"], feature_columns=feature_columns)
        # This is an audit snapshot, not an alteration of the selected profile.
        with (destination / "dataset_config.json").open("x", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, allow_nan=False, indent=2)
        aliases: dict[str, set[str]] = defaultdict(set)
        identifiers: set[str] = set()
        for target in targets:
            cid = _safe_component(target["complex_id"], field="complex_id")
            if cid in identifiers:
                raise ValueError(f"duplicate complex_id in manifest: {cid}")
            identifiers.add(cid)
            fingerprint = CanonicalComplexFingerprint.from_sequences(
                target["receptor_sequence"], target["ligand_sequence"],
                expected_sha256=target["complex_fingerprint_sha256"],
                schema=target["complex_fingerprint_schema"],
            )
            aliases[fingerprint.sha256].add(cid)
        counts["independent_complexes"] = len(aliases)
        selected_ids = set(identifiers)
        for fingerprint, names in aliases.items():
            if len(names) > 1:
                if fingerprint_alias_policy == "reject":
                    issue("fingerprint_alias", "one canonical sequence fingerprint maps to multiple complex_id aliases",
                          complex_fingerprint_sha256=fingerprint, complex_ids=sorted(names))
                else:
                    selected, *excluded = sorted(names)
                    selected_ids.difference_update(excluded)
                    report["alias_resolutions"].append({
                        "complex_fingerprint_sha256": fingerprint,
                        "selected_complex_id": selected, "excluded_complex_ids": excluded,
                        "selection_basis": "lexicographic_complex_id_from_manifest_before_reading_results",
                    })
                    counts["resolved_alias_groups"] += 1
                    counts["excluded_alias_targets"] += len(excluded)
                    counts["excluded_alias_planned_runs"] += len(excluded) * len(jobs)
        if report["alias_resolutions"]:
            report["warnings"].append({
                "code": "fingerprint_alias_resolved",
                "message": "One whole target per sequence fingerprint selected by complex_id order; excluded aliases remain in the batch audit, with no outcome-based substitution.",
            })
        job_ids: set[str] = set()
        for job in jobs:
            job_id = _safe_component(job["job_id"], field="job_id")
            if job_id in job_ids:
                raise ValueError(f"duplicate job_id in manifest: {job_id}")
            job_ids.add(job_id)
            if job["protocol"] not in {"full", "reduced"} or job["profile"]["mode"] != job["protocol"]:
                raise ValueError("batch requires explicit FULL/REDUCED protocols")
            if isinstance(job["seed"], bool) or not isinstance(job["seed"], int) or job["seed"] < 0:
                raise ValueError("batch seed must be a non-negative integer")
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
        issue("manifest_or_context", str(exc))
        return finish()

    # Group planned tasks, not merely successful files. Duplicate protocol jobs
    # remain ambiguous even if only one happened to succeed.
    groups: dict[tuple, dict[str, list]] = defaultdict(lambda: {"full": [], "reduced": []})
    receipts: dict[tuple[str, str], dict | None] = {}
    total_runs = len(targets) * len(jobs)
    for target in targets:
        for job in jobs:
            cid, jid = target["complex_id"], job["job_id"]
            emit_progress(progress_callback, 0.05 + 0.25 * len(receipts) / total_runs,
                          f"Validando execução {len(receipts) + 1}/{total_runs}: {cid} / {jid}")
            key = (target["complex_fingerprint_sha256"], job["algorithm"], job["seed"])
            if cid in selected_ids:
                groups[key][job["protocol"]].append((target, job))
            receipts[cid, jid] = None
            try:
                directory = _task_directory(root, cid, jid)
                receipt = _read_receipt(directory, manifest, target, job)
                receipts[cid, jid] = receipt
                if receipt is not None:
                    if job["protocol"] == "reduced":
                        counts["reduced_succeeded"] += 1
                else:
                    failure = _read_failure(directory, manifest, target, job)
                    if failure is not None:
                        failure["excluded_by_alias_policy"] = cid not in selected_ids
                        report["failures"].append(failure)
                        if failure["reason"] == "reduced_hard_failure":
                            counts["reduced_failed"] += 1
            except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
                issue("receipt_provenance", str(exc), complex_id=cid, run_id=jid)

    rows: list[dict] = []
    for pair_index, ((fingerprint, algorithm, seed), pair) in enumerate(sorted(groups.items())):
        pair_target = next(t for runs in pair.values() for t, _ in runs)
        emit_progress(progress_callback, 0.30 + 0.65 * pair_index / len(groups),
                      f"Validando par {pair_index + 1}/{len(groups)} e grafo REDUCED: {pair_target['complex_id']} / {algorithm} / seed {seed}")
        identity = dict(complex_fingerprint_sha256=fingerprint, algorithm=algorithm, seed=seed)
        if any(len(pair[mode]) > 1 for mode in ("full", "reduced")):
            counts["ambiguous_pairs"] += 1
            issue("ambiguous_pair", "pairing requires exactly one FULL and one REDUCED planned run",
                  **identity, runs={mode: [f"{t['complex_id']}/{j['job_id']}" for t, j in runs]
                                    for mode, runs in pair.items()})
            continue
        available = {mode: (receipts[t["complex_id"], j["job_id"]] if runs else None)
                     for mode, runs in pair.items() for t, j in (runs or [(None, None)])}
        missing_modes = [mode for mode, receipt in available.items() if receipt is None]
        if missing_modes:
            report["unmatched"].append({**identity, "complex_id": pair_target["complex_id"],
                                        "missing": missing_modes})
            counts["unmatched_pairs"] += 1
            continue
        full, reduced = available["full"], available["reduced"]
        target, job = reduced["target"], reduced["job"]
        try:
            if full["target"] != target:
                raise ValueError("FULL and REDUCED have incompatible target/input snapshots")
            row = {key: target[key] for key in (
                "complex_id", "receptor_sequence", "ligand_sequence",
                "complex_fingerprint_schema", "complex_fingerprint_sha256",
            )}
            row.update(run_id=job["job_id"], algorithm=algorithm, seed=seed,
                       docking_task=task, feature_schema=report["feature_schema"],
                       config=str(config_path), config_sha256=digest)
            for mode, receipt in available.items():
                result = receipt["engine_result"]
                for source, destination_column in (("engine", "engine"), ("scientific", "scientific"),
                                                   ("dockq", "dockq"), ("runtime_sec", "runtime")):
                    column = f"runtime_{mode}_sec" if source == "runtime_sec" else f"{destination_column}_{mode}"
                    row[column] = result.get(source)
            row.update({key: reduced["engine_result"].get(key) for key in feature_columns
                        if not key.startswith("graph_")})
            directory = _task_directory(root, target["complex_id"], job["job_id"])
            decision_path = directory / "confidence_decision.json"
            if decision_path.exists():
                decision = _load_json_object(decision_path)
                core = {key: value for key, value in decision.items() if key != "decision_sha256"}
                if (decision.get("decision_sha256") != _payload_sha256(core)
                        or decision.get("reduced_receipt_record_sha256") != reduced["record_sha256"]):
                    raise ValueError("calibrator timing decision is not bound to the REDUCED receipt")
                overhead = decision.get("confidence_overhead_sec")
                if overhead is not None:
                    if isinstance(overhead, bool) or not isinstance(overhead, (int, float)) or not math.isfinite(overhead) or overhead < 0:
                        raise ValueError("invalid measured calibrator overhead")
                    row["confidence_overhead_sec"] = float(overhead)
            row.update(_read_graph(directory, reduced, config))
            if report["feature_schema"] == LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION:
                input_root = Path(plan["input_root"])
                pose_path = (directory / reduced["engine_result"]["pose_artifact_path"]).resolve(strict=True)
                pose_path.relative_to(directory.resolve())
                context, trace = local_execution_context(
                    algorithm=algorithm,
                    receptor_model_pdb=input_root / target["receptor_path"],
                    ligand_template_pdb=input_root / target["ligand_path"], pose_json=pose_path,
                    expected_receptor_sha256=target["receptor_sha256"],
                    expected_ligand_sha256=target["ligand_sha256"],
                )
                row.update(context)
                report.setdefault("execution_context_evidence", []).append({
                    "complex_id": target["complex_id"], "run_id": job["job_id"], **trace,
                })
            missing = [key for key in feature_columns if row.get(key) is None
                       or isinstance(row[key], bool)
                       or not isinstance(row[key], (int, float)) or not math.isfinite(row[key])]
            if missing:
                detail = dict(complex_id=target["complex_id"], run_id=job["job_id"], features=missing)
                report["missing_feature_details"].append(detail)
                counts["missing_features"] += len(missing)
                issue("missing_features", "required REDUCED features are absent or non-finite; no imputation", **detail)
            else:
                counts["valid_pairs"] += 1
            rows.append(row)
        except _MissingFeaturesError as exc:
            detail = dict(complex_id=target["complex_id"], run_id=job["job_id"], features=exc.features)
            report["missing_feature_details"].append(detail)
            counts["missing_features"] += len(exc.features)
            issue("missing_features", str(exc), **detail)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError) as exc:
            issue("pair_provenance", str(exc), complex_id=target["complex_id"], run_id=job["job_id"])

    emit_progress(progress_callback, 0.95, f"Salvando e validando CSV com {len(rows)} pares candidatos")
    extra_metadata = (EXECUTION_CONTEXT_METADATA_COLUMNS
                      if report["feature_schema"] == LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION else ())
    columns = list(dict.fromkeys((*TRAINING_BASE_COLUMNS, *_CONTEXT_COLUMNS,
                                 *feature_columns, *_GRAPH_COLUMNS, *extra_metadata,
                                 "confidence_overhead_sec")))
    frame = pd.DataFrame(rows, columns=columns)
    counts["candidate_rows"] = len(frame)
    csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
    candidate_path = destination / "paired_dataset.csv"
    with candidate_path.open("xb") as handle:
        handle.write(csv_bytes)
    report.update(candidate_dataset_path=str(candidate_path), dataset_sha256=hashlib.sha256(csv_bytes).hexdigest())
    try:
        checked = load_paired_dataset(io.BytesIO(csv_bytes), feature_columns,
                                      contact_graph_cfg=config["contact_graph"],
                                      docking_task=task, feature_schema=report["feature_schema"])
        if checked.empty:
            raise ValueError("no paired rows available for training")
        report["loader_passed"] = True
        rule = PreservationRule(**config["preservation"])
        labels = [rule.label(r, f) for r, f in zip(checked.dockq_reduced, checked.dockq_full)]
        report["preview"] = {
            "row_count": len(checked), "independent_complex_count": checked.complex_fingerprint_sha256.nunique(),
            "algorithm_count": checked.algorithm.nunique(), "seeds": sorted(map(int, checked.seed.unique())),
            "preserved_count": sum(labels), "non_preserved_count": len(labels) - sum(labels),
            "reduced_failed_count": counts["reduced_failed"],
            "scope": "candidate_csv; usable only when valid=true",
        }
        policy = config["policy"]
        report["evidence_requirements"] = {"ivap": config["calibration"], "policy": policy}
        if policy["method"] == "cluster_bootstrap_run_risk":
            report["evidence_requirements"]["note"] = (
                "Accepted-run error rate with whole-complex percentile bootstrap; "
                "approximate uncertainty, not a finite-sample binomial bound. "
                "Zero-error or degenerate bootstrap evidence does not authorize decisions. "
                "No gate or split was evaluated or altered during preparation."
            )
        else:
            candidates = len(threshold_grid(policy))
            alpha = ((1 - policy["familywise_confidence_level"]) * policy["risk_error_budget_fraction"] / candidates)
            report["evidence_requirements"].update(
                minimum_risk_authorized_groups_even_with_zero_failures=math.ceil(
                    math.log(alpha) / math.log(1 - policy["risk_target"])),
                note="Necessary LTT bound only; no gate or split was evaluated or altered.",
            )
    except (ValueError, KeyError, TypeError) as exc:
        issue("official_loader", str(exc))
    return finish()
