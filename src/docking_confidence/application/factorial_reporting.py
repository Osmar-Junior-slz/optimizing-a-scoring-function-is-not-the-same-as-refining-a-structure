"""Offline paired analysis of the independently controlled 2 x 2 experiment.

The immutable batch plan defines the denominator. Receipts, rather than a
possibly incomplete CSV export, provide successful outcomes. Seeds are first
averaged within each complex; confidence intervals resample whole complexes.
No docking engine is imported or executed by this module.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean
from typing import Any, Mapping

from docking_confidence.domain.execution import (
    FACTORIAL_DOCKQ_THRESHOLD, FACTORIAL_PRIMARY_OUTCOME, FIXED_EVALUATION_BUDGET,
)

CONDITIONS = ("reduced_uniform", "reduced_seeded", "expanded_uniform", "expanded_seeded")
METRICS = ("dockq", "runtime_sec", "actual_eval_count", "task_duration_sec", "success")
# Ordering always follows CONDITIONS. Interaction is the difference between the
# two initialization effects, or equivalently between the two amplitude effects.
CONTRASTS = {
    "extent_at_uniform": (1, 0, -1, 0),
    "extent_at_seeded": (0, 1, 0, -1),
    "initialization_at_reduced": (-1, 1, 0, 0),
    "initialization_at_expanded": (0, 0, -1, 1),
    "interaction": (-1, 1, 1, -1),
    "extent_main": (0.5, 0.5, -0.5, -0.5),
    "initialization_main": (-0.5, 0.5, -0.5, 0.5),
}


class FactorialAnalysisError(ValueError):
    """The persisted experiment is inconsistent or cannot be audited safely."""


def _hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise FactorialAnalysisError(f"Cannot read valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise FactorialAnalysisError(f"Expected a JSON object: {path}")
    return payload


def _inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FactorialAnalysisError(f"Artifact escapes batch directory: {relative}") from exc
    return candidate


def _number(value: Any, *, field: str) -> float | int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise FactorialAnalysisError(f"Invalid {field}: {value!r}")
    try:
        converted = float(value)
    except (TypeError, ValueError) as exc:
        raise FactorialAnalysisError(f"Invalid {field}: {value!r}") from exc
    if not math.isfinite(converted) or converted < 0:
        raise FactorialAnalysisError(f"Invalid {field}: {value!r}")
    if field == "dockq" and converted > 1:
        raise FactorialAnalysisError(f"DockQ outside [0, 1]: {value!r}")
    if field == "actual_eval_count":
        if not converted.is_integer():
            raise FactorialAnalysisError(f"Evaluation count must be an integer: {value!r}")
        return int(converted)
    return converted


def _load_plan(root: Path) -> tuple[dict[str, Any], str, dict[str, Any] | None, dict[str, Any] | None]:
    manifest_path = root / "batch_manifest.json"
    if not manifest_path.is_file():
        raise FactorialAnalysisError("batch_manifest.json is required; CSV alone is not evidence")
    manifest = _json(manifest_path)
    recorded = manifest.get("manifest_sha256")
    if recorded != _hash({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise FactorialAnalysisError("Batch manifest failed its integrity hash")
    plan = manifest.get("plan")
    if not isinstance(plan, dict) or manifest.get("plan_sha256") != _hash(plan):
        raise FactorialAnalysisError("Batch plan failed its integrity hash")
    if manifest.get("batch_id") != manifest["plan_sha256"]:
        raise FactorialAnalysisError("batch_id does not match the immutable plan")
    plan_path = root / "batch_plan.json"
    if plan_path.is_file():
        exported = _json(plan_path)
        # Some exports wrap the plan with presentation-only execution metadata.
        exported_plan = exported.get("plan", exported)
        if isinstance(exported_plan, dict) and "targets" in exported_plan:
            if exported_plan != plan:
                raise FactorialAnalysisError("batch_plan.json disagrees with the manifest plan")
        elif exported.get("jobs") != plan.get("jobs"):
            raise FactorialAnalysisError("batch_plan.json jobs disagree with the manifest")
    protocol = plan.get("experimental_protocol")
    frozen = None
    if protocol is not None:
        if not isinstance(protocol, Mapping):
            raise FactorialAnalysisError("Invalid experimental protocol binding")
        protocol_path = Path(str(protocol.get("path", "experimental_protocol.json")))
        path = protocol_path.resolve() if protocol_path.is_absolute() else _inside(root, str(protocol_path))
        expected = protocol.get("sha256")
        if not path.is_file() or expected != hashlib.sha256(path.read_bytes()).hexdigest():
            raise FactorialAnalysisError("Frozen experimental protocol failed its integrity hash")
        frozen = _json(path)
        canonical_hash = _hash({k: v for k, v in frozen.items() if k != "protocol_sha256"})
        if frozen.get("protocol_sha256") != canonical_hash or protocol.get("protocol_sha256") != canonical_hash:
            raise FactorialAnalysisError("Frozen experimental protocol failed its canonical integrity hash")
        base_plan = {k: v for k, v in plan.items() if k != "experimental_protocol"}
        if frozen.get("schema_version") != "factorial_protocol_v1" or frozen.get("plan") != base_plan:
            raise FactorialAnalysisError("Frozen protocol belongs to a different experiment")
        if frozen.get("base_plan_sha256") != _hash(base_plan):
            raise FactorialAnalysisError("Frozen protocol base plan hash is inconsistent")
        if not isinstance(frozen.get("analysis"), dict):
            raise FactorialAnalysisError("Frozen protocol has no analysis choices")
    return plan, manifest["plan_sha256"], dict(protocol) if protocol is not None else None, frozen


def _job_groups(plan: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    job_ids: set[str] = set()
    jobs = plan.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise FactorialAnalysisError("The plan has no jobs")
    for job in jobs:
        if not isinstance(job, dict) or not isinstance(job.get("profile"), dict):
            raise FactorialAnalysisError("A job has no explicit factorial profile")
        profile = job["profile"]
        extent, init = profile.get("search_extent"), profile.get("initialization")
        if extent not in {"reduced", "expanded"} or init not in {"seeded", "uniform"}:
            raise FactorialAnalysisError("Analysis requires explicit factorial search_extent/initialization metadata")
        executable_mode = "reduced" if extent == "reduced" else "full"
        if job.get("protocol") != executable_mode or profile.get("mode", executable_mode) != executable_mode:
            raise FactorialAnalysisError("Factorial extent disagrees with executable protocol")
        job_id = job.get("job_id")
        if not isinstance(job_id, str) or not job_id or job_id in job_ids:
            raise FactorialAnalysisError("Duplicate or invalid planned job_id")
        job_ids.add(job_id)
        seed = job.get("seed")
        algorithm = job.get("algorithm")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or not isinstance(algorithm, str):
            raise FactorialAnalysisError("Invalid algorithm/seed in plan")
        condition = f"{extent}_{init}"
        key = (algorithm, seed)
        if condition in groups[key]:
            raise FactorialAnalysisError(f"Duplicate condition for {key}: {condition}")
        groups[key][condition] = job
    for key, group in groups.items():
        if set(group) != set(CONDITIONS):
            raise FactorialAnalysisError(f"Plan lacks one or more of the four conditions for {key}")
        for field in ("evaluation_budget", "generations", "population", "top_pockets", "docking_task", "budget_policy",
                      "max_translation", "max_rotation_degrees", "pose_source", "execution_order_seed"):
            values = [job["profile"].get(field) for job in group.values()]
            if any(value != values[0] for value in values):
                raise FactorialAnalysisError(f"Uncontrolled {field} differs across factorial conditions for {key}")
        budget = next(iter(group.values()))["profile"].get("evaluation_budget")
        if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
            raise FactorialAnalysisError(f"Missing positive common evaluation budget for {key}")
    return dict(groups)


def _verify_identity(payload: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for key, value in expected.items():
        if payload.get(key) != value:
            raise FactorialAnalysisError(f"{label} disagrees with planned {key}")


def _read_observation(root: Path, target: dict[str, Any], job: dict[str, Any],
                      plan_hash: str, task: dict[str, Any] | None) -> dict[str, Any]:
    complex_id, job_id = target["complex_id"], job["job_id"]
    identity = {"job_id": job_id, "algorithm": job["algorithm"], "protocol": job["protocol"], "seed": job["seed"]}
    if task is not None:
        _verify_identity(task, {"complex_id": complex_id, **identity}, label="Batch status")
    path = _inside(root, f"targets/{complex_id}/jobs/{job_id}/receipt.json")
    state = task.get("state", "MISSING") if task else "MISSING"
    diagnostics: list[str] = []
    result: dict[str, Any] = {}
    valid_receipt = False
    if path.is_file():
        receipt = _json(path)
        if receipt.get("record_sha256") != _hash({k: v for k, v in receipt.items() if k != "record_sha256"}):
            raise FactorialAnalysisError(f"Receipt failed its integrity hash: {path}")
        task_hash = _hash({"plan_sha256": plan_hash, "target": target, "job": job})
        _verify_identity(receipt, {"complex_id": complex_id, "plan_sha256": plan_hash,
                                  "task_sha256": task_hash, "status": "SUCCEEDED"}, label="Receipt")
        if receipt.get("job") != job or receipt.get("target") != target:
            raise FactorialAnalysisError(f"Receipt target/job binding differs from the plan: {path}")
        result = receipt.get("engine_result", {})
        if not isinstance(result, dict):
            raise FactorialAnalysisError(f"Invalid engine_result: {path}")
        _verify_identity(result, identity, label="Engine result")
        valid_receipt = True
        if task is None:
            state = "SUCCEEDED"
            diagnostics.append("status_missing_verified_receipt")
        elif state != "SUCCEEDED":
            diagnostics.append(f"receipt_present_but_status_{state.lower()}")
            # A failed/restarted task must not silently contribute an old success.
            result = {}
    elif state == "SUCCEEDED":
        diagnostics.append("succeeded_without_receipt")
    elif state == "MISSING":
        diagnostics.append("status_and_receipt_missing")
    succeeded = state == "SUCCEEDED" and valid_receipt
    if not succeeded:
        result = {}
    eval_count = result.get("actual_eval_count")
    values = {
        "dockq": _number(result.get("dockq"), field="dockq"),
        "runtime_sec": _number(result.get("runtime_sec"), field="runtime_sec"),
        "actual_eval_count": _number(eval_count, field="actual_eval_count"),
        "task_duration_sec": _number(task.get("duration_seconds") if task else None, field="task_duration_sec"),
        "success": 1 if succeeded else 0 if state == "FAILED" else None,
    }
    if values["actual_eval_count"] is not None and values["actual_eval_count"] > job["profile"]["evaluation_budget"]:
        diagnostics.append("actual_evaluation_budget_exceeded")
    if job["profile"].get("budget_policy") == FIXED_EVALUATION_BUDGET:
        structural_success = None
        if state == "FAILED":
            structural_success = 0
        elif succeeded:
            exact = (values["actual_eval_count"] == job["profile"]["evaluation_budget"]
                     and result.get("stop_reason") == "objective_evaluation_ceiling")
            if not exact:
                diagnostics.append("fixed_evaluation_budget_not_verified")
            pose_path = result.get("complex_pose_pdb_path")
            has_pose = isinstance(pose_path, str) and bool(pose_path) and _inside(path.parent, pose_path).is_file()
            dockq = values["dockq"]
            if not has_pose or dockq is None:
                diagnostics.append("structural_outcome_not_verified")
            structural_success = int(exact and has_pose and dockq is not None and dockq >= FACTORIAL_DOCKQ_THRESHOLD)
        values[FACTORIAL_PRIMARY_OUTCOME] = structural_success
    return {"complex_id": complex_id, **identity,
            "complex_fingerprint_sha256": target.get("complex_fingerprint_sha256"),
            "condition": f"{job['profile']['search_extent']}_{job['profile']['initialization']}",
            "state": state, "valid_receipt": valid_receipt,
            "evaluation_budget": job["profile"]["evaluation_budget"],
            "stop_reason": result.get("stop_reason"),
            "initial_population_sha256": result.get("initial_population_sha256"),
            "execution_audit_sha256": result.get("execution_audit_sha256"),
            "last_error": task.get("last_error") if task else None,
            "metrics": values, "diagnostics": diagnostics}


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _estimate(complex_values: Mapping[str, float], *, repetitions: int, seed: int,
              tag: str) -> dict[str, Any]:
    values = [complex_values[key] for key in sorted(complex_values)]
    interval = None
    if len(values) >= 2:
        derived = int(hashlib.sha256(f"{seed}:{tag}".encode()).hexdigest(), 16)
        rng = random.Random(derived)
        samples = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repetitions))
        interval = [_quantile(samples, 0.025), _quantile(samples, 0.975)]
    return {"mean": mean(values) if values else None,
            "ci95": interval, "n_complexes": len(values),
            "complex_means": dict(sorted(complex_values.items())),
            "ci_note": "At least two complexes are required" if len(values) < 2 else None}


def _csv_audit(root: Path, observations: list[dict[str, Any]]) -> dict[str, Any]:
    path = root / "batch_results.csv"
    if not path.is_file():
        return {"present": False, "rows": 0, "discrepancies": []}
    observed = {(o["complex_id"], o["job_id"]): o for o in observations}
    issues: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        key = (str(row.get("complex_id", "")), str(row.get("job_id", "")))
        if key in seen:
            issues.append({"complex_id": key[0], "job_id": key[1], "reason": "duplicate_csv_row"})
        seen.add(key)
        observation = observed.get(key)
        if observation is None:
            issues.append({"complex_id": key[0], "job_id": key[1], "reason": "csv_row_outside_plan"})
            continue
        if not observation["valid_receipt"]:
            issues.append({"complex_id": key[0], "job_id": key[1], "reason": "csv_row_without_verified_receipt"})
        for metric in ("dockq", "runtime_sec", "actual_eval_count"):
            if metric not in row:
                continue
            value = _number(row[metric], field=metric)
            expected = observation["metrics"][metric]
            if value != expected and not (value is not None and expected is not None
                                          and math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12)):
                issues.append({"complex_id": key[0], "job_id": key[1], "reason": f"csv_{metric}_differs_from_receipt"})
    return {"present": True, "rows": len(rows), "discrepancies": issues,
            "missing_planned_rows": len(set(observed) - seen), "role": "audit_only"}


def analyze_factorial_batch(batch_dir: str | Path, *, bootstrap_repetitions: int = 2000,
                            bootstrap_seed: int = 2026) -> dict[str, Any]:
    """Return auditable effects and coverage without executing any experiment."""
    if isinstance(bootstrap_repetitions, bool) or not isinstance(bootstrap_repetitions, int) or bootstrap_repetitions < 1:
        raise ValueError("bootstrap_repetitions must be a positive integer")
    if isinstance(bootstrap_seed, bool) or not isinstance(bootstrap_seed, int) or bootstrap_seed < 0:
        raise ValueError("bootstrap_seed must be a non-negative integer")
    root = Path(batch_dir).resolve(strict=True)
    plan, plan_hash, protocol, frozen = _load_plan(root)
    groups = _job_groups(plan)
    fixed_flags = {j["profile"].get("budget_policy") == FIXED_EVALUATION_BUDGET for j in plan["jobs"]}
    if len(fixed_flags) != 1:
        raise FactorialAnalysisError("Cannot mix fixed and historical budget policies in one analysis")
    fixed_evaluations = fixed_flags.pop()
    primary_outcome = FACTORIAL_PRIMARY_OUTCOME if fixed_evaluations else "dockq"
    metrics = (primary_outcome, *METRICS) if fixed_evaluations else METRICS
    targets = plan.get("targets")
    if not isinstance(targets, list) or not targets:
        raise FactorialAnalysisError("The immutable plan has no targets")
    target_ids = [target.get("complex_id") for target in targets if isinstance(target, dict)]
    if len(target_ids) != len(targets) or any(not isinstance(t, str) for t in target_ids) or len(set(target_ids)) != len(target_ids):
        raise FactorialAnalysisError("Invalid or duplicate planned complex_id")
    status_path = root / "batch_status.json"
    status = _json(status_path) if status_path.is_file() else {}
    if status and status.get("plan_sha256") != plan_hash:
        raise FactorialAnalysisError("Batch status belongs to a different plan")
    tasks = status.get("tasks", {})
    if not isinstance(tasks, dict):
        raise FactorialAnalysisError("Batch status tasks must be an object")
    planned_keys = {f"{target['complex_id']}::{job['job_id']}" for target in targets for job in plan["jobs"]}
    unplanned_keys = sorted(set(tasks) - planned_keys)
    observations: list[dict[str, Any]] = []
    quartets: list[dict[str, Any]] = []
    cluster_aliases: dict[str, list[str]] = defaultdict(list)
    for target in targets:
        fingerprint = target.get("complex_fingerprint_sha256")
        cluster = fingerprint if isinstance(fingerprint, str) and fingerprint else target["complex_id"]
        cluster_aliases[cluster].append(target["complex_id"])
    for target in sorted(targets, key=lambda t: t["complex_id"]):
        for (algorithm, seed), jobs in sorted(groups.items()):
            runs: dict[str, dict[str, Any]] = {}
            for condition in CONDITIONS:
                job = jobs[condition]
                task = tasks.get(f"{target['complex_id']}::{job['job_id']}")
                if task is not None and not isinstance(task, dict):
                    raise FactorialAnalysisError("Batch task must be an object")
                runs[condition] = _read_observation(root, target, job, plan_hash, task)
                observations.append(runs[condition])
            effects: dict[str, dict[str, float] | None] = {}
            missing: dict[str, list[str]] = {}
            for metric in metrics:
                absent = [condition for condition in CONDITIONS if runs[condition]["metrics"][metric] is None]
                missing[metric] = absent
                values = [runs[condition]["metrics"][metric] for condition in CONDITIONS]
                effects[metric] = None if absent else {
                    contrast: sum(weight * value for weight, value in zip(weights, values))
                    for contrast, weights in CONTRASTS.items()
                }
            quartets.append({"complex_id": target["complex_id"], "algorithm": algorithm,
                             "independent_cluster": target.get("complex_fingerprint_sha256") or target["complex_id"],
                             "seed": seed, "job_ids": {c: jobs[c]["job_id"] for c in CONDITIONS},
                             "states": {c: runs[c]["state"] for c in CONDITIONS},
                             "effects": effects, "missing_conditions_by_metric": missing})
    by_algorithm: dict[str, dict[str, Any]] = {}
    for algorithm in sorted({key[0] for key in groups}):
        selected = [q for q in quartets if q["algorithm"] == algorithm]
        metric_reports: dict[str, Any] = {}
        for metric in metrics:
            available = [q for q in selected if q["effects"][metric] is not None]
            exclusions = [{"complex_id": q["complex_id"], "seed": q["seed"],
                           "missing_conditions": q["missing_conditions_by_metric"][metric],
                           "states": q["states"]} for q in selected if q["effects"][metric] is None]
            estimates: dict[str, Any] = {}
            for contrast in CONTRASTS:
                per_complex: dict[str, list[float]] = defaultdict(list)
                for quartet in available:
                    per_complex[quartet["independent_cluster"]].append(quartet["effects"][metric][contrast])
                estimates[contrast] = _estimate({k: mean(v) for k, v in per_complex.items()},
                                               repetitions=bootstrap_repetitions, seed=bootstrap_seed,
                                               tag=f"{algorithm}:{metric}:{contrast}")
            metric_reports[metric] = {"planned_quartets": len(selected),
                                      "complete_quartets": len(available),
                                      "excluded_quartets": exclusions, "effects": estimates}
        condition_reports: dict[str, Any] = {}
        for condition in CONDITIONS:
            runs = [o for o in observations if o["algorithm"] == algorithm and o["condition"] == condition]
            condition_reports[condition] = {
                "planned_runs": len(runs), "states": dict(sorted(Counter(o["state"] for o in runs).items())),
                "metric_available": {m: sum(o["metrics"][m] is not None for o in runs) for m in metrics},
            }
            if fixed_evaluations:
                outcomes = [o["metrics"][primary_outcome] for o in runs]
                passed = outcomes.count(1)
                condition_reports[condition]["primary_outcome"] = {
                    "successes": passed, "failures_or_below_threshold": outcomes.count(0),
                    "unresolved": outcomes.count(None), "denominator": len(outcomes),
                    "proportion_of_planned": passed / len(outcomes),
                }
        by_algorithm[algorithm] = {"conditions": condition_reports, "metrics": metric_reports}
    partial = any(q["effects"][primary_outcome] is None for q in quartets)
    diagnostics = [{"complex_id": o["complex_id"], "job_id": o["job_id"], "reasons": o["diagnostics"]}
                   for o in observations if o["diagnostics"]]
    frozen_analysis = frozen["analysis"] if frozen is not None else None
    choices_match = frozen_analysis is not None and all(
        frozen_analysis.get(k) == v for k, v in (("bootstrap_repetitions", bootstrap_repetitions),
                                               ("bootstrap_seed", bootstrap_seed), ("confidence_level", 0.95)))
    if fixed_evaluations and choices_match:
        choices_match = all(frozen_analysis.get(k) == v for k, v in (
            ("primary_outcome", FACTORIAL_PRIMARY_OUTCOME),
            ("primary_dockq_threshold", FACTORIAL_DOCKQ_THRESHOLD),
            ("primary_denominator", "all_planned_runs_per_condition"),
            ("budget_policy", FIXED_EVALUATION_BUDGET),
            ("objective_evaluations_per_run", plan["jobs"][0]["profile"]["evaluation_budget"]),
        ))
    protocol_deviation = any(
        {"actual_evaluation_budget_exceeded", "fixed_evaluation_budget_not_verified",
         "structural_outcome_not_verified"}.intersection(o["diagnostics"])
        for o in observations
    )
    interpretation = "complete_frozen_protocol"
    if partial:
        interpretation = "exploratory_partial"
    elif protocol is None:
        interpretation = "exploratory_unfrozen"
    elif protocol_deviation:
        interpretation = "exploratory_protocol_deviation"
    elif not choices_match:
        interpretation = "exploratory_analysis_deviation"
    if plan.get("engine", {}).get("scientific") is False:
        interpretation = "non_scientific_demo"
    return {
        "schema_version": "factorial_analysis_v1", "source_batch": str(root),
        "plan_sha256": plan_hash, "experimental_protocol": protocol,
        "interpretation": interpretation,
        "analysis_role": frozen_analysis.get("analysis_role") if frozen_analysis else "exploratory",
        "primary_outcome": primary_outcome,
        "frozen_analysis": frozen_analysis, "analysis_choices_match_protocol": choices_match,
        "method": {"extent_direction": "reduced minus expanded", "initialization_direction": "seeded minus uniform",
                   "interaction": "(reduced_seeded - reduced_uniform) - (expanded_seeded - expanded_uniform)",
                   "paired_unit": "complex_id, algorithm, seed; all four conditions required per metric",
                   "aggregation": "mean paired seed effects within each molecular fingerprint cluster, then equally weighted mean over clusters",
                   "bootstrap_unit": "complex", "bootstrap_repetitions": bootstrap_repetitions,
                   "independent_unit": "complex_fingerprint_sha256; complex_id fallback only when no fingerprint is recorded",
                   "bootstrap_seed": bootstrap_seed, "confidence_level": 0.95,
                   "success_definition": "1 = verified successful receipt; 0 = FAILED; pending/missing remains unavailable",
                   **({"primary_outcome_definition": "1 = verified pose, fixed evaluation budget and DockQ >= 0.23; "
                       "0 = failed run or completed run without verified structural success; "
                       "unresolved jobs stay in the planned denominator and make the rate provisional"}
                      if fixed_evaluations else {}),
                   "missing_policy": "No missing DockQ, runtime or evaluation count is replaced with zero",
                   "runtime_sec": "engine runtime of verified successful runs",
                   "task_duration_sec": "persisted task duration, including completed failed attempts when available"},
        "counts": {"complexes": len(targets), "independent_clusters": len(cluster_aliases),
                   "planned_runs": len(observations), "planned_quartets": len(quartets),
                   "states": dict(sorted(Counter(o["state"] for o in observations).items()))},
        "by_algorithm": by_algorithm, "quartets": quartets, "observations": observations,
        "diagnostics": diagnostics, "csv_audit": _csv_audit(root, observations),
        "status_audit": {"unplanned_task_keys": unplanned_keys,
                         "missing_planned_task_keys": sorted(planned_keys - set(tasks))},
        "excluded_targets": plan.get("excluded_targets", []),
        "cluster_aliases": dict(sorted(cluster_aliases.items())),
        "limitations": ["Intervals use available complete quartets and do not correct bias from failed or missing runs.",
                        "Seed repetitions are not treated as independent biological complexes.",
                        "Effect directions must be interpreted per metric: positive DockQ and positive runtime have different meanings.",
                        "A frozen protocol does not by itself establish prospective scientific validity."],
    }


def factorial_markdown(report: Mapping[str, Any]) -> str:
    def fmt(value: Any) -> str:
        if value is None:
            return "indisponível"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = ["# Análise fatorial 2 × 2", "", f"Fonte: `{report['source_batch']}`", "",
             f"Plano: `{report['plan_sha256']}`. Situação: `{report['interpretation']}`.", "",
             "Amplitude = reduzido − ampliado; inicialização = semeada − uniforme. "
             "Interação = (reduzido semeado − reduzido uniforme) − (ampliado semeado − ampliado uniforme).", "",
             "Cada efeito exige as quatro condições da mesma proteína, algoritmo e seed. "
             "As seeds são primeiro promediadas por proteína; identificadores com o mesmo fingerprint molecular compartilham o mesmo agrupamento. "
             "O bootstrap reamostra agrupamentos inteiros. "
             "Ausências e falhas são exibidas e não viram DockQ zero. Intervalos exigem pelo menos duas proteínas.", "",
             "| Algoritmo | Condição | Planejados | Estados | DockQ disponível | Avaliações disponíveis |",
             "|---|---|---:|---|---:|---:|"]
    for algorithm, data in report["by_algorithm"].items():
        for condition, info in data["conditions"].items():
            states = ", ".join(f"{s}: {n}" for s, n in info["states"].items())
            lines.append(f"| {algorithm} | {condition} | {info['planned_runs']} | {states} | "
                         f"{info['metric_available']['dockq']} | {info['metric_available']['actual_eval_count']} |")
    if report.get("primary_outcome") == FACTORIAL_PRIMARY_OUTCOME:
        lines += ["", "## Desfecho principal: pose válida com DockQ ≥ 0,23", "",
                  "O denominador inclui todas as execuções planejadas. Falhas contam como insucesso. "
                  "Com pendências, a proporção é provisória e os efeitos usam apenas quartetos conhecidos.", "",
                  "| Algoritmo | Condição | Sucessos | Insucessos | Pendências | Planejados | Proporção |",
                  "|---|---|---:|---:|---:|---:|---:|"]
        for algorithm, data in report["by_algorithm"].items():
            for condition, info in data["conditions"].items():
                primary = info["primary_outcome"]
                lines.append(f"| {algorithm} | {condition} | {primary['successes']} | "
                             f"{primary['failures_or_below_threshold']} | {primary['unresolved']} | "
                             f"{primary['denominator']} | {primary['proportion_of_planned']:.2%} |")
    lines += ["", "## Efeitos pareados", "",
              "Quartetos incompletos são excluídos apenas da métrica correspondente e listados abaixo. "
              "A média dá o mesmo peso a cada proteína com dados completos; falhas podem causar viés nessa estimativa.", "",
              "| Algoritmo | Métrica | Efeito | Média | IC bootstrap 95% | Proteínas | Quartetos completos / planejados |",
              "|---|---|---|---:|---|---:|---|"]
    for algorithm, data in report["by_algorithm"].items():
        for metric, info in data["metrics"].items():
            for effect, estimate in info["effects"].items():
                ci = estimate["ci95"]
                interval = f"[{fmt(ci[0])}, {fmt(ci[1])}]" if ci else "indisponível"
                lines.append(f"| {algorithm} | {metric} | {effect} | {fmt(estimate['mean'])} | {interval} | "
                             f"{estimate['n_complexes']} | {info['complete_quartets']} / {info['planned_quartets']} |")
    lines += ["", "## Quartetos incompletos por métrica", "",
              "| Proteína | Algoritmo | Seed | Métrica | Condições indisponíveis | Estados |",
              "|---|---|---:|---|---|---|"]
    for quartet in report["quartets"]:
        for metric, missing in quartet["missing_conditions_by_metric"].items():
            if missing:
                states = ", ".join(f"{c}: {s}" for c, s in quartet["states"].items())
                lines.append(f"| {fmt(quartet['complex_id'])} | {quartet['algorithm']} | {quartet['seed']} | "
                             f"{metric} | {', '.join(missing)} | {states} |")
    lines += ["", "## Diagnósticos", ""]
    for diagnostic in report["diagnostics"]:
        lines.append(f"- {fmt(diagnostic['complex_id'])}/{fmt(diagnostic['job_id'])}: {', '.join(diagnostic['reasons'])}")
    audit = report["csv_audit"]
    lines.append(f"- Auditoria CSV: {audit['rows']} linhas; {len(audit['discrepancies'])} divergências. O CSV não fornece evidência de sucesso.")
    if report["status_audit"]["unplanned_task_keys"]:
        lines.append("- Registros de status fora do plano: " + ", ".join(report["status_audit"]["unplanned_task_keys"]))
    if report["experimental_protocol"] is None:
        lines.append("- Sem vínculo com protocolo congelado: análise exploratória.")
    elif not report["analysis_choices_match_protocol"]:
        lines.append("- As escolhas de análise diferem do protocolo congelado: análise exploratória.")
    lines += ["", "## Limitações", ""]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"


def write_factorial_report(report: Mapping[str, Any], output_dir: str | Path, *,
                           plots: bool = True) -> dict[str, str]:
    """Write JSON, Markdown, observation/effect tables and optional static plots."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    json_path, markdown_path = destination / "factorial_analysis.json", destination / "factorial_analysis.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    markdown_path.write_text(factorial_markdown(report), encoding="utf-8")
    paths.update(json=str(json_path.resolve()), markdown=str(markdown_path.resolve()))
    observation_rows = [{k: v for k, v in observation.items() if k not in {"metrics", "last_error", "diagnostics"}} |
                        observation["metrics"] | {"diagnostics": "; ".join(observation["diagnostics"])}
                        for observation in report["observations"]]
    effect_rows = []
    for algorithm, data in report["by_algorithm"].items():
        for metric, info in data["metrics"].items():
            for contrast, estimate in info["effects"].items():
                interval = estimate["ci95"] or [None, None]
                effect_rows.append({"algorithm": algorithm, "metric": metric, "contrast": contrast,
                                    "mean": estimate["mean"], "ci95_lower": interval[0], "ci95_upper": interval[1],
                                    "n_complexes": estimate["n_complexes"], "complete_quartets": info["complete_quartets"],
                                    "planned_quartets": info["planned_quartets"]})
    for name, rows in (("observations", observation_rows), ("effects", effect_rows)):
        path = destination / f"factorial_{name}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["algorithm"])
            writer.writeheader()
            writer.writerows(rows)
        paths[name] = str(path.resolve())
    if plots:
        paths.update(_write_plots(report, destination))
    return paths


def _write_plots(report: Mapping[str, Any], destination: Path) -> dict[str, str]:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    paths: dict[str, str] = {}
    effects = ("extent_main", "initialization_main", "interaction")
    plot_metrics = ("dockq", "runtime_sec", "actual_eval_count", "success")
    if report.get("primary_outcome") == FACTORIAL_PRIMARY_OUTCOME:
        plot_metrics = (FACTORIAL_PRIMARY_OUTCOME, *plot_metrics)
    for metric in plot_metrics:
        fig = Figure(figsize=(10, 5), layout="constrained")
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        labels, values, intervals = [], [], []
        for algorithm, data in report["by_algorithm"].items():
            for effect in effects:
                estimate = data["metrics"][metric]["effects"][effect]
                if estimate["mean"] is None:
                    continue
                labels.append(f"{algorithm} · {effect}")
                values.append(estimate["mean"])
                intervals.append(estimate["ci95"])
        if labels:
            ax.scatter(values, range(len(values)), color="#215b80", zorder=3)
            for index, interval in enumerate(intervals):
                if interval:
                    ax.plot(interval, [index, index], color="#215b80")
            ax.set_yticks(range(len(labels)), labels)
            ax.invert_yaxis()
        else:
            ax.text(0.5, 0.5, "Sem quartetos completos para esta métrica", ha="center", transform=ax.transAxes)
        ax.axvline(0, color="gray", linewidth=1)
        label = "sucesso estrutural (DockQ ≥ 0,23)" if metric == FACTORIAL_PRIMARY_OUTCOME else metric
        ax.set_xlabel(f"Efeito pareado em {label}; IC 95% por proteína")
        ax.set_title("Amplitude: reduzido − ampliado; inicialização: semeada − uniforme")
        ax.grid(axis="x", alpha=0.2)
        path = destination / f"factorial_{metric}.png"
        fig.savefig(path, dpi=160)
        paths[f"plot_{metric}"] = str(path.resolve())
    return paths
