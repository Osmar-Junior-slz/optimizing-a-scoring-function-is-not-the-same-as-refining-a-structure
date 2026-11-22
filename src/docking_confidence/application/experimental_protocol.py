"""Freeze experimental choices before running; local hashes provide integrity, not preregistration."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import math

from docking_confidence.application.batch import (
    BatchError, BatchPlanConflictError, _build_plan, _BatchFileLock, _atomic_write_json, LOCK_FILENAME,
    _file_sha256, _load_json_object, _payload_sha256, _resolved_disjoint_roots,
    _utc_now, _verify_target_inputs,
)
from docking_confidence.domain.execution import (
    FACTORIAL_DOCKQ_THRESHOLD, FACTORIAL_PRIMARY_OUTCOME, FIXED_EVALUATION_BUDGET,
)

PROTOCOL_FILENAME = "experimental_protocol.json"
SCHEMA = "factorial_protocol_v1"


class ExperimentalProtocolPlanConflictError(BatchPlanConflictError):
    """An intact saved protocol differs from the newly requested plan."""

    def __init__(self, differences: list[dict]):
        self.differences = differences
        fields = ", ".join(item["field"] for item in differences)
        super().__init__(
            "inputs, engine, conditions or execution order changed after protocol preparation: " + fields
        )


def _plan_differences(saved, current, *, limit=12) -> list[dict]:
    differences = []

    def display(value):
        text = str(value)
        return text if len(text) <= 160 else text[:157] + "..."

    def compare(before, after, field):
        if before == after or len(differences) >= limit:
            return
        if isinstance(before, dict) and isinstance(after, dict):
            for key in sorted(before.keys() | after.keys()):
                compare(before.get(key), after.get(key), f"{field}.{key}" if field else key)
        elif isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
            for index, (old, new) in enumerate(zip(before, after)):
                compare(old, new, f"{field}[{index}]")
        else:
            differences.append({
                "field": field,
                "saved": f"{len(before)} itens" if isinstance(before, list) else display(before),
                "current": f"{len(after)} itens" if isinstance(after, list) else display(after),
            })

    compare(saved, current, "")
    return differences


def load_experimental_protocol(path) -> dict:
    """Read and verify a saved protocol without changing it or executing docking."""
    payload = _load_json_object(Path(path).expanduser().resolve(strict=True))
    core = {key: value for key, value in payload.items() if key != "protocol_sha256"}
    if payload.get("schema_version") != SCHEMA or payload.get("protocol_sha256") != _payload_sha256(core):
        raise BatchPlanConflictError("experimental protocol failed its integrity hash")
    plan = payload.get("plan")
    if not isinstance(plan, dict) or payload.get("base_plan_sha256") != _payload_sha256(plan):
        raise BatchPlanConflictError("experimental protocol base plan failed its integrity hash")
    validate_factorial_jobs(plan["jobs"])
    return payload


def validate_factorial_jobs(jobs: list[dict]) -> None:
    groups = defaultdict(list)
    common = None
    for job in jobs:
        p = job["profile"]
        condition = (p.get("search_extent"), p.get("initialization"))
        if condition[0] not in ("reduced", "expanded") or condition[1] not in ("seeded", "uniform"):
            raise BatchError("every factorial job requires explicit extent and initialization")
        expected_mode = "reduced" if condition[0] == "reduced" else "full"
        if job["protocol"] != expected_mode or p["mode"] != expected_mode:
            raise BatchError("factorial extent and executable protocol disagree")
        if p.get("docking_task") != "local_refinement" or p.get("budget_policy") not in (
            "common_ceiling", FIXED_EVALUATION_BUDGET,
        ):
            raise BatchError("factorial_2x2 requires local_refinement and a common evaluation budget")
        if not isinstance(p.get("pose_source"), str) or not p["pose_source"].strip():
            raise BatchError("factorial pose provenance is required")
        for field in ("evaluation_budget", "population", "generations", "top_pockets"):
            if isinstance(p.get(field), bool) or not isinstance(p.get(field), int) or p[field] < 1:
                raise BatchError(f"invalid factorial {field}")
        if p["evaluation_budget"] < p["population"]:
            raise BatchError("common budget must cover the initial population")
        for field in ("max_translation", "max_rotation_degrees", "search_scale"):
            value = p.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise BatchError(f"invalid factorial {field}")
        if p["max_rotation_degrees"] > 180 or (condition[0] == "expanded" and p["search_scale"] != 1.0) or (condition[0] == "reduced" and p["search_scale"] >= 1):
            raise BatchError("expanded scale must be 1 and reduced scale strictly between 0 and 1")
        order_seed = p.get("execution_order_seed")
        if isinstance(order_seed, bool) or not isinstance(order_seed, int) or order_seed < 0:
            raise BatchError("invalid factorial execution order seed")
        signature = tuple(p.get(field) for field in (
            "generations", "population", "top_pockets", "evaluation_budget", "budget_policy",
            "docking_task", "max_translation", "max_rotation_degrees", "pose_source", "execution_order_seed",
        ))
        if common is None:
            common = signature
        if common != signature:
            raise BatchError("factorial controls must be identical across all conditions")
        groups[(job["algorithm"], job["seed"])].append(p)
    if not groups:
        raise BatchError("factorial plan is empty")
    reduced_scales = set()
    for profiles in groups.values():
        conditions = {(p["search_extent"], p["initialization"]) for p in profiles}
        if len(profiles) != 4 or len(conditions) != 4:
            raise BatchError("each algorithm/seed needs all four unique factorial conditions")
        reduced_scales.update(p["search_scale"] for p in profiles if p["search_extent"] == "reduced")
    if len(reduced_scales) != 1:
        raise BatchError("reduced scale must be identical across the experiment")


def prepare_experimental_protocol(input_root, output_root, *, jobs, engine, report=None, analysis_role="exploratory") -> dict:
    if analysis_role not in ("exploratory", "confirmatory"):
        raise ValueError("analysis_role must be exploratory or confirmatory")
    source, destination = _resolved_disjoint_roots(input_root, output_root)
    if report is None:
        from docking_confidence.io.batch_discovery import discover_ppi4dock
        report = discover_ppi4dock(source, validate_contents=True)
    plan, _ = _build_plan(report=report, input_root=source, jobs=tuple(jobs), engine=engine, eligible_only=True, skip_invalid=False)
    validate_factorial_jobs(plan["jobs"])
    for target in plan["targets"]:
        _verify_target_inputs(target, source)
    profile = plan["jobs"][0]["profile"]
    fixed_evaluations = profile["budget_policy"] == FIXED_EVALUATION_BUDGET
    settings = {
        "analysis_role": analysis_role,
        "independent_unit": "complex_fingerprint_sha256",
        "primary_outcome": "dockq",
        "secondary_outcomes": ["runtime_sec", "actual_eval_count", "task_duration_sec", "success"],
        "contrasts": ["extent_at_uniform", "extent_at_seeded", "initialization_at_reduced", "initialization_at_expanded", "interaction", "extent_main", "initialization_main"],
        "missing_results": "report_every_failure_and_missing_outcome; metric_contrasts_use_complete_quartets_only",
        "replicates": "seeds_and_algorithms_are_correlated_within_complex",
        "budget": "common_ceiling; actual consumption may differ; generation limit and normal stopping rules apply",
        "initialization": "uniform: all candidates sampled in the 6D parameter box; seeded: replace first candidate with supplied pose identity",
        "reference_usage": "reference PDB used only in post-search structural evaluation; operator must document input pose origin",
        "bootstrap_repetitions": 2000, "bootstrap_seed": 2026, "confidence_level": 0.95,
        "local_commitment_limit": "a local hash and timestamp are not independent preregistration or proof that data were never inspected",
    }
    if fixed_evaluations:
        settings.update(
            primary_outcome=FACTORIAL_PRIMARY_OUTCOME,
            primary_dockq_threshold=FACTORIAL_DOCKQ_THRESHOLD,
            primary_denominator="all_planned_runs_per_condition",
            secondary_outcomes=["dockq", *settings["secondary_outcomes"]],
            missing_results="failed_runs_count_as_zero_for_primary; unfinished_runs_reported_separately; "
                            "paired_contrasts_require_known_outcomes_in_all_four_conditions",
            budget_policy=FIXED_EVALUATION_BUDGET,
            objective_evaluations_per_run=profile["evaluation_budget"],
            budget="exact new search objective calls; no generation or convergence stop; "
                   "cache hits and final rescoring excluded",
        )
    path = destination / PROTOCOL_FILENAME
    destination.mkdir(parents=True, exist_ok=True)
    with _BatchFileLock(destination / LOCK_FILENAME):
        if path.exists():
            existing = load_experimental_protocol(path)
            bind_experimental_protocol(path, plan)
            if existing.get("analysis") != settings:
                raise BatchPlanConflictError("prepared analysis choices differ; use a new output directory")
            return {"path": str(path), "sha256": _file_sha256(path), "protocol": existing}
        if any(child.name != LOCK_FILENAME for child in destination.iterdir()):
            raise BatchPlanConflictError("prepare the factorial protocol in an empty output directory before execution")
        payload = {"schema_version": SCHEMA, "created_at_utc": _utc_now(), "base_plan_sha256": _payload_sha256(plan), "plan": plan, "analysis": settings}
        payload["protocol_sha256"] = _payload_sha256(payload)
        _atomic_write_json(path, payload)
        return {"path": str(path), "sha256": _file_sha256(path), "protocol": payload}


def bind_experimental_protocol(path, plan: dict) -> dict:
    resolved = Path(path).expanduser().resolve(strict=True)
    payload = load_experimental_protocol(resolved)
    if payload["plan"] != plan:
        raise ExperimentalProtocolPlanConflictError(_plan_differences(payload["plan"], plan))
    validate_factorial_jobs(plan["jobs"])
    return {"path": str(resolved), "sha256": _file_sha256(resolved), "protocol_sha256": payload["protocol_sha256"]}
