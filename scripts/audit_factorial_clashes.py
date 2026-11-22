"""Apply the study's secondary, common steric eligibility rule to completed factorial trials."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_standard_factorial import (
    ROOT, _atomic_write_json, _file_sha256, _inside, _json, read_study, require, summarize,
)


def assess_gate(meta, condition):
    """Verify the published Boolean using the actual recorded distance and parameters."""
    scale = 0.5 if condition.startswith("reduced_") else 1.0
    require(meta.get("pocket_id") == "local_refinement"
            and meta.get("docking_task") == "local_refinement"
            and meta.get("search_anchor_mode") == "input_centroid"
            and meta.get("input_relative_placement_used") is True
            and meta.get("local_refinement_search_scale") == scale,
            "Selected pose does not establish the registered refinement geometry")
    gate = meta.get("final_hard_clash_gate") or {}
    require(gate.get("enabled") is True and type(gate.get("triggered")) is bool,
            "Missing or disabled steric eligibility assessment")
    threshold = gate.get("threshold")
    distance = gate.get("min_distance")
    penalty = gate.get("penalty_score")
    for key, number in (("threshold", threshold), ("penalty", penalty)):
        require(type(number) in (int, float) and math.isfinite(number), f"Unverified {key} in eligibility record")
    require(threshold > 0 and penalty < 0, "Invalid eligibility parameters")
    if distance is None:
        pairs = meta.get("vina_pair_stats") or {}
        cutoff = pairs.get("cutoff_pair")
        require("min_distance" in gate and pairs.get("n_pairs") == 0 and "min_distance" in pairs
                and pairs["min_distance"] is None and type(cutoff) in (int, float)
                and math.isfinite(cutoff) and cutoff >= threshold,
                "Unverified distance without documented zero pairs within scoring cutoff")
        expected = False
        distance_basis = "zero_pairs_within_scoring_cutoff"
    else:
        require(type(distance) in (int, float) and math.isfinite(distance) and distance >= 0,
                "Unverified distance in eligibility record")
        expected = distance < threshold
        distance_basis = "recorded_minimum"
    require(gate["triggered"] is expected, "Recorded decision disagrees with minimum distance")
    require(bool(meta.get("cheap_score_variant")) and bool(meta.get("final_score_variant")),
            "Selected pose has no scoring variant")
    return {"gate_triggered": int(gate["triggered"]), "threshold_angstrom": threshold,
            "min_distance_angstrom": distance, "distance_evidence": distance_basis, "penalty_score": penalty,
            "pocket_id": meta["pocket_id"], "cheap_score_variant": meta["cheap_score_variant"],
            "final_score_variant": meta["final_score_variant"]}


def audit_attempt(row, batch):
    jobdir = _inside(batch, f"targets/{row['complex_id']}/jobs/{row['job_id']}")
    state = row["outcome_state"]
    if state == "scored":
        receipt_file = jobdir / "receipt.json"
        require(_file_sha256(receipt_file) == row["receipt_file_sha256"], "Primary receipt hash mismatch")
        receipt = _json(receipt_file)
        engine = receipt["engine_result"]
        path = _inside(jobdir, engine["pose_artifact_path"])
        saved = _json(path)
        control = assess_gate(saved["meta"], row["condition"])
        require(saved.get("pocket_id") == control["pocket_id"], "Inconsistent winner pocket")
        engine_name = saved["engine_name"]
        require(int(row["actual_eval_count"]) == int(row["evaluation_budget"]) == engine["actual_eval_count"],
                "Wrong budget in a published pose")
        if row["condition"].startswith("reduced_"):
            require(not control["gate_triggered"], "Reduced protocol published a sterically rejected pose")
    else:
        require(state == "no_valid_pose", "Primary observation is unresolved")
        require(row["error_file_sha256"] == _file_sha256(jobdir / "error.json"), "Failure evidence changed")
        matches = [p for p in jobdir.glob("engine/dockingpp_runs/*/output/result.json")
                   if _file_sha256(p) == row["engine_result_sha256"]]
        require(len(matches) == 1, "Missing definitive evidence of the winning candidate on failed attempt")
        path = matches[0]
        result = _json(path)
        require(result["final_pose_available"] is False and result["failure_reason"] == "no_valid_reduced_pose",
                "Missing structural explanation for absence of pose")
        pockets = result["per_pocket_results"]
        require(result.get("executed_pockets") == ["local_refinement"] and len(pockets) == 1,
                "Multiple or unidentified competing locations on failed attempt")
        selected = pockets[0]
        control = assess_gate(selected["best_pose_meta"], row["condition"])
        require(control["gate_triggered"] and selected.get("final_hard_clash_triggered") is True
                and selected.get("cross_pocket_selection_eligible") is False
                and selected.get("pocket_id") == control["pocket_id"],
                "No direct physical-rejection evidence for native failure")
        require(selected.get("rescoring_applied") is True, "Final scoring not established for rejected candidate")
        actual = row["actual_eval_count"]
        require(result.get("total_n_eval") == (int(actual) if actual else None),
                "Evaluation count for rejected pose differs from primary evidence")
        engine_name = result["search_engine_name"]
    native_valid = int(row["valid_pose"])
    native_primary = int(row["primary"])
    if state == "scored":
        require(native_valid == 1 and native_primary == int(float(row["dockq"]) >= 0.23),
                "Primary response does not match the official structural score")
    else:
        require(native_valid == native_primary == 0 and not row["dockq"], "Geometric failure has a pose score")
    clear = 1 - control["gate_triggered"]
    common = {k: row[k] for k in ("complex_id", "job_id", "complex_fingerprint_sha256", "algorithm",
              "seed", "batch_shard", "condition", "evaluation_budget", "actual_eval_count", "dockq")}
    return {**common, "native_outcome_state": state, "native_valid_pose": native_valid,
            "native_primary": native_primary, "valid_pose": clear*native_valid, "primary": clear*native_primary,
            **control, "search_engine_name": engine_name, "gate_evidence_sha256": _file_sha256(path),
            "gate_evidence_path_relative_to_job": path.relative_to(jobdir).as_posix()}


def gate_counts(rows):
    return {"planned": len(rows),
            "gate_clear": sum(1-r["gate_triggered"] for r in rows),
            "gate_triggered": sum(r["gate_triggered"] for r in rows),
            "native_structural_rejections": sum(1-r["native_valid_pose"] for r in rows),
            "triggered_on_published_pose": sum(r["gate_triggered"]*r["native_valid_pose"] for r in rows),
            "primary_response_changed": sum(r["native_primary"]-r["primary"] for r in rows),
            "actual_evaluations": dict(Counter(str(r["actual_eval_count"] or "unrecorded") for r in rows))}


def analyze(study, out):
    plan_path = out / "protocol/analysis_plan.json"
    ad_path = out / "protocol/common_eligibility_addendum.json"
    primary_path = out / "factorial_results.json"
    raw_path = out / "factorial_observations.csv"
    plan, primary, addendum = map(_json, (plan_path, primary_path, ad_path))
    require(primary["complete"] is True and primary["observations_sha256"] == _file_sha256(raw_path)
            and primary["analysis_plan_sha256"] == addendum["original_local_plan_sha256"] == _file_sha256(plan_path),
            "Incomplete primary study or mismatched supplemental protocol")
    _, workers = read_study(study, plan_path, plan)
    batches = {Path(shard["batch_dir"]).parent.name: Path(shard["batch_dir"]).resolve(strict=True)
               for shard in workers}
    rows, issues = [], []
    with raw_path.open(encoding="utf-8", newline="") as src:
        for row in csv.DictReader(src):
            try:
                rows.append(audit_attempt(row, batches[row["batch_shard"]]))
            except (KeyError, ValueError, OSError, TypeError) as exc:
                issues.append({"complex_id": row["complex_id"], "job_id": row["job_id"], "issue": str(exc)})
    if issues:
        _atomic_write_json(out/"harmonized_audit_issues.json", {"complete": False, "issues": issues})
        raise ValueError(f"{len(issues)} unresolved eligibility records; review harmonized_audit_issues.json")
    require(len(rows) == plan["design"]["attempts"], "Incomplete diagnostic cohort")
    methods, conditions = plan["design"]["algorithms"], plan["analysis"]["cell_order"]
    comparable = ("threshold_angstrom", "penalty_score", "cheap_score_variant", "final_score_variant", "pocket_id")
    parameters = {}
    for k in comparable:
        values = {r[k] for r in rows}
        require(len(values) == 1, f"The four groups did not share {k}: {values}")
        parameters[k] = next(iter(values))
    engines = {}
    for method in methods:
        names = {r["search_engine_name"] for r in rows if r["algorithm"] == method}
        require(len(names) == 1 and all(names), f"Inconsistent search engine for {method}")
        engines[method] = next(iter(names))
    # The same native grouping utility operates on the two common-gate binary columns.
    contrasts = summarize([{**r, "outcome_state": r["native_outcome_state"]} for r in rows], plan)
    pooled = {c: gate_counts([r for r in rows if r["condition"] == c]) for c in conditions}
    stratified = {m: {c: gate_counts([r for r in rows if r["algorithm"] == m and r["condition"] == c])
                      for c in conditions} for m in methods}
    path = out / "harmonized_observations.csv"
    with TemporaryDirectory(dir=out, prefix=".common-") as tmp:
        staged = Path(tmp) / path.name
        with staged.open("w", encoding="utf-8", newline="") as dest:
            writer = csv.DictWriter(dest, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(staged, path)
    _atomic_write_json(out / "harmonized_results.json", {
        "schema_version": "common_eligibility_factorial_v1", "complete": True,
        "addendum_sha256": _file_sha256(ad_path), "analysis_plan_sha256": _file_sha256(plan_path),
        "primary_results_sha256": _file_sha256(primary_path), "primary_observations_sha256": _file_sha256(raw_path),
        "observations_sha256": _file_sha256(path), "quality_metric_sha256": primary["quality_metric_sha256"],
        "bootstrap": plan["analysis"]["uncertainty"], "shared_parameters": parameters, "search_engines": engines,
        "gate_counts_by_condition": pooled, "gate_counts_by_method": stratified, **contrasts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "runs/fatorial_dockq_20261119")
    parser.add_argument("--out", type=Path, default=ROOT / "output/fatorial_dockq_20261119")
    args = parser.parse_args()
    analyze(args.study.resolve(strict=True), args.out.resolve(strict=True))


if __name__ == "__main__":
    main()
