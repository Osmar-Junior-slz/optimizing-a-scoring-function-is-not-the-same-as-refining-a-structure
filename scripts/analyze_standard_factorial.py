"""Score saved factorial poses using official DockQ and analyze a frozen cohort.

Default runs only verify and score terminal receipts. It is safe to invoke the
command again as new receipts become available. --finalize publishes estimates
only if all predeclared trials have a verified scientific outcome.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from io import StringIO
import json
import math
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from docking_confidence.application.batch import _atomic_write_json, _BatchFileLock, _file_sha256
from docking_confidence.application.factorial_reporting import (
    CONDITIONS, _estimate, _inside, _job_groups, _json, _load_plan, _read_observation,
)
from docking_confidence.evaluation.official_dockq import (
    canonical_sha256, official_dockq_preflight, quality_metric_descriptor,
    quality_metric_sha256, rescore_receipt_standard,
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_study(study, plan_path, analysis):
    selection = _json(study / "selection.json")
    workers = _json(study / "planned_workers.json")
    require(workers["selection_sha256"] == _file_sha256(study / "selection.json")
            and workers["analysis_plan_sha256"] == _file_sha256(plan_path),
            "Selection or analysis protocol differs from the signed worker plan")
    targets = {target["complex_id"]: target for target in selection["targets"]}
    n = analysis["selection"]["number_of_complexes"]
    require(len(selection["targets"]) == len(targets) == n, "Selection size mismatch")
    fingerprints = {t["complex_fingerprint_sha256"] for t in targets.values()}
    require(len(fingerprints) == n, "Repeated molecular fingerprint")
    planned = [cid for shard in workers["shards"] for cid in shard["complex_ids"]]
    require(len(planned) == n and set(planned) == set(targets), "Shard selection mismatch")
    return targets, workers["shards"]


def validate_shard(plan, frozen, analysis):
    design = analysis["design"]
    groups = _job_groups(plan)
    require(set(groups) == {(a, s) for a in design["algorithms"] for s in design["seeds"]},
            "Methods/seeds differ from analysis plan")
    expected_engine = analysis["model_and_engine_identity"]
    engine = plan["engine"]["descriptor"]
    for key, origin in (("source_manifest_sha256", "source_manifest_sha256"),
                        ("config_sha256", "engine_config_sha256"),
                        ("upstream_archive_sha256", "upstream_archive_sha256")):
        require(engine[key] == expected_engine[origin], f"Docking engine mismatch: {key}")
    for job in plan["jobs"]:
        profile = job["profile"]
        expected_scale = 0.5 if profile["search_extent"] == "reduced" else 1.0
        require(profile["evaluation_budget"] == design["evaluation_budget"]
                and profile["budget_policy"] == "fixed_evaluations"
                and profile["population"] == design["population"]
                and profile["docking_task"] == "local_refinement"
                and profile["max_translation"] == 5.0 and profile["max_rotation_degrees"] == 25.0
                and profile["search_scale"] == expected_scale,
                f"Unexpected profile: {job['job_id']}")
    uncertainty = analysis["analysis"]["uncertainty"]
    require(frozen and frozen["analysis"]["bootstrap_seed"] == uncertainty["seed"]
            and frozen["analysis"]["bootstrap_repetitions"] == uncertainty["repetitions"]
            and frozen["analysis"]["primary_dockq_threshold"] == 0.23,
            "Analysis settings differ from frozen shard protocol")


def structural_failure(directory, observation, plan_hash, target, job, attempt_count):
    path = directory / "error.json"
    error = _json(path)
    expected_task = canonical_sha256({"plan_sha256": plan_hash, "target": target, "job": job})
    require(error.get("status") == "FAILED" and error.get("task_sha256") == expected_task
            and error.get("plan_sha256") == plan_hash and error.get("attempt") == attempt_count
            and error.get("complex_id") == target["complex_id"] and error.get("job_id") == job["job_id"],
            f"Failed-attempt identity mismatch: {path}")
    require(error.get("engine_status") == "REDUCED_FAILED"
            and error.get("failure_reason") == "no_valid_reduced_pose"
            and job["protocol"] == "reduced",
            f"Non-geometric or unidentified failure: {error.get('error_type')} {error.get('failure_reason')}")
    result_paths = list(directory.glob("engine/dockingpp_runs/*/output/result.json"))
    count = None
    terminal_reason = error["failure_reason"]
    result_sha = None
    if len(result_paths) == 1:
        result = _json(result_paths[0])
        require(result.get("final_pose_available") is False and result.get("status") == error["engine_status"]
                and result.get("failure_reason") == terminal_reason,
                f"Engine outcome disagrees with geometric failure: {path}")
        count = result.get("total_n_eval")
        require(count is None or type(count) is int and 0 <= count <= observation["evaluation_budget"],
                f"Invalid objective count on geometric failure: {path}")
        result_sha = _file_sha256(result_paths[0])
    elif len(result_paths) > 1:
        raise ValueError(f"Ambiguous engine result across attempts for {path}")
    return {"outcome_state": "no_valid_pose", "valid_pose": 0, "primary": 0, "dockq": None,
            "actual_eval_count": count, "stop_reason": terminal_reason,
            "error_file_sha256": _file_sha256(path), "engine_result_sha256": result_sha,
            "error_path": str(path)}


def score_receipt(directory, target, reference, binding, destination, runtime):
    receipt_path = directory / "receipt.json"
    receipt = _json(receipt_path)
    engine = receipt["engine_result"]
    model = _inside(directory, engine["complex_pose_pdb_path"])
    model_sha = _file_sha256(model)
    digest = canonical_sha256(runtime["quality_metric"])
    binding.update(receipt_file_sha256=_file_sha256(receipt_path),
                   receipt_record_sha256=receipt["record_sha256"])
    output = destination / "standard_dockq.json"
    if not output.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Do not leave a partial record at an immutable destination on interruption.
        with TemporaryDirectory(dir=destination.parent, prefix=".official-") as temp:
            rescore_receipt_standard(receipt_path, reference, output_dir=Path(temp)/"score",
                    dependency_path=runtime["dependency_path"], expected_model_sha256=model_sha,
                    expected_metric_sha256=digest, source_binding=binding, timeout_seconds=600)
            (Path(temp)/"score").rename(destination)
    saved = _json(output)
    require(saved.get("record_sha256") == canonical_sha256({k: v for k, v in saved.items() if k != "record_sha256"})
            and saved.get("source_binding") == binding and saved.get("reference_used_for_search") is False
            and saved.get("quality_metric") == runtime["quality_metric"] and saved.get("quality_metric_sha256") == digest,
            f"Invalid official score or source binding: {output}")
    inputs = saved["inputs"]
    require(inputs["model_sha256"] == model_sha and inputs["reference_sha256"] == target["reference_sha256"]
            and _file_sha256(reference) == inputs["reference_sha256"]
            and saved["native_to_model_chain_map"] == {c: c for c in (target["receptor_chain"], target["ligand_chain"])},
            f"Pose, reference or chain map mismatch: {output}")
    for field in ("implementation_version", "implementation_sha256", "constants_sha256"):
        require(saved["implementation"][field] == runtime["preflight"][field], f"DockQ evaluator mismatch: {field}")
    score = saved["metrics"]["dockq"]
    require(math.isfinite(score) and 0 <= score <= 1, f"Invalid official DockQ: {output}")
    return {"outcome_state": "scored", "valid_pose": 1, "primary": int(score >= 0.23), "dockq": score,
            "actual_eval_count": engine["actual_eval_count"], "stop_reason": engine["stop_reason"],
            "pose_sha256": model_sha, "reference_sha256": inputs["reference_sha256"],
            "receipt_path": str(receipt_path), "receipt_file_sha256": binding["receipt_file_sha256"],
            "official_record_path": str(output), "official_record_sha256": _file_sha256(output)}


def scan_shard(shard, selected, analysis, study):
    root = Path(shard["batch_dir"]).resolve()
    root.relative_to(study)
    require(_file_sha256(Path(shard["experimental_protocol_path"])) == shard["protocol_file_sha256"],
            f"Frozen protocol hash mismatch: {root}")
    if not (root/"batch_manifest.json").exists():
        return []
    plan, plan_hash, _, frozen = _load_plan(root)
    validate_shard(plan, frozen, analysis)
    require({t["complex_id"] for t in plan["targets"]} == set(shard["complex_ids"])
            and len(plan["targets"]) == len(shard["complex_ids"]), f"Shard membership mismatch: {root}")
    status_path = root / "batch_status.json"
    status = _json(status_path) if status_path.exists() else {}
    require(not status or status["plan_sha256"] == plan_hash, f"Status plan mismatch: {root}")
    tasks = status.get("tasks", {})
    work = []
    for target in plan["targets"]:
        cid = target["complex_id"]
        fp = target["complex_fingerprint_sha256"]
        for field in ("complex_fingerprint_sha256", "receptor_sha256", "ligand_sha256", "reference_sha256"):
            require(selected[cid][field] == target[field], f"Unselected input for {cid}: {field}")
        reference = _inside(Path(plan["input_root"]).resolve(), target["reference_path"])
        for job in plan["jobs"]:
            task = tasks.get(f"{cid}::{job['job_id']}", {})
            if task.get("state") not in {"SUCCEEDED","FAILED"}:
                continue
            condition = f"{job['profile']['search_extent']}_{job['profile']['initialization']}"
            row = {"complex_id": cid, "complex_fingerprint_sha256": fp, "algorithm": job["algorithm"],
                   "seed": job["seed"], "condition": condition, "job_id": job["job_id"],
                   "evaluation_budget": job["profile"]["evaluation_budget"]}
            work.append((root, plan_hash, target, job, task, reference, row))
    return work


def score_attempt(item, out, runtime):
    root, plan_hash, target, job, task, reference, row = item
    cid = target["complex_id"]
    directory = _inside(root, f"targets/{cid}/jobs/{job['job_id']}")
    try:
        observation = _read_observation(root, target, job, plan_hash, task)
        if observation["state"] == "FAILED":
            details = structural_failure(directory, observation, plan_hash, target, job, task["attempt_count"])
        else:
            require(observation["valid_receipt"] and not observation["diagnostics"]
                    and observation["metrics"]["actual_eval_count"] == row["evaluation_budget"]
                    and observation["stop_reason"] == "objective_evaluation_ceiling",
                    f"Unverified score/budget on successful receipt: {observation['diagnostics']}")
            binding = {**row, "batch_plan_sha256": plan_hash, "analysis_plan_file_sha256": runtime["analysis_hash"]}
            destination = out / "quality" / cid / job["job_id"]
            details = score_receipt(directory, target, reference, binding, destination, runtime)
            details["official_record_relative_path"] = (destination / "standard_dockq.json").relative_to(out).as_posix()
            details["receipt_relative_to_batch"] = (directory / "receipt.json").relative_to(root).as_posix()
        details["batch_shard"] = root.parent.name
        return {**row, **details}
    except Exception as exc:
        return {**row, "outcome_state": "unresolved", "issue": f"{type(exc).__name__}: {exc}"}


def summarize(rows, analysis):
    design, config = analysis["design"], analysis["analysis"]
    algs = design["algorithms"]
    size = analysis["selection"]["number_of_complexes"]
    require(len(rows) == design["attempts"] and all(r["outcome_state"] in {"scored","no_valid_pose"} for r in rows),
            "Complete and verified outcomes are required to compute effects")
    order = config["cell_order"]
    require(tuple(order) == CONDITIONS and design["seeds"] == [1], "Unsupported cell order or seed count")
    by_method, pooled = {}, {}
    for cond in order:
        subset = [r for r in rows if r["condition"] == cond]
        pooled[cond] = counts(subset)
    for alg in algs:
        per_method = [r for r in rows if r["algorithm"] == alg]
        cells = {cond:counts([r for r in per_method if r["condition"] == cond]) for cond in order}
        require(all(c["planned"] == size for c in cells.values()), f"Incomplete cell: {alg}")
        quartets = {}
        for row in per_method:
            cluster = quartets.setdefault(row["complex_fingerprint_sha256"], {})
            require(row["condition"] not in cluster, "Duplicate observation in quartet")
            cluster[row["condition"]] = row["primary"]
        require(len(quartets) == size and all(set(q)==set(order) for q in quartets.values()), "Incomplete quartet")
        effects = {}
        for name, weights in config["contrast_weights"].items():
            contrasts = {fp:100*sum(w*q[c] for w,c in zip(weights,order,strict=True)) for fp,q in quartets.items()}
            u = config["uncertainty"]
            est = _estimate(contrasts, repetitions=u["repetitions"], seed=u["seed"], tag=f"{alg}:{name}")
            effects[name] = {"estimate_pp": est["mean"], "ci95_pp": est["ci95"], "complexes": est["n_complexes"]}
        by_method[alg] = {"cells":cells,"effects":effects}
    return {"pooled_cells":pooled,"by_method":by_method}


def counts(rows):
    return {"planned": len(rows), "valid_poses": sum(r["valid_pose"] for r in rows),
            "acceptable_poses": sum(r["primary"] for r in rows)}


def evaluate(study, out, workers, finalize, dependency, follow):
    plan_path = out/"protocol"/"analysis_plan.json"
    analysis = _json(plan_path)
    require(analysis["schema_version"] == "official_dockq_factorial_analysis_plan_v1", "Unsupported plan")
    targets, shards = read_study(study, plan_path, analysis)
    # A concurrent docking collection can delay imports even on a valid installation.
    print("Checking official DockQ runtime", flush=True)
    preflight = official_dockq_preflight(dependency_path=dependency, timeout_seconds=180)
    metric = quality_metric_descriptor("2.1.3")
    require(preflight.get("available") and preflight.get("quality_metric") == metric, f"DockQ unavailable: {preflight}")
    print(f"Official DockQ {preflight['implementation_version']} ready", flush=True)
    runtime = {"dependency_path":dependency, "quality_metric":metric,
               "analysis_hash": _file_sha256(plan_path), "preflight":preflight}
    verified = {}
    while True:
        work = []
        for shard in shards:
            work.extend(scan_shard(shard, targets, analysis, study))
        # Cached terminal rows are used only to monitor progress. The publication
        # pass reopens and checks every receipt, pose, reference and DockQ record.
        pending = analysis["design"]["attempts"] - len(work)
        if not pending or not follow:
            verified.clear()
        cached, to_score = [], []
        for item in work:
            target, job, task = item[2:5]
            identity = (target["complex_id"], job["job_id"], task["state"], task["attempt_count"])
            if identity in verified:
                cached.append(verified[identity])
            else:
                to_score.append(item)
        if to_score:
            print(f"Checking {len(to_score)} terminal receipts", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            new_rows = list(pool.map(lambda item: score_attempt(item, out, runtime), to_score))
        for item, row in zip(to_score, new_rows):
            if row["outcome_state"] != "unresolved":
                task = item[4]
                key = (row["complex_id"], row["job_id"], task["state"], task["attempt_count"])
                verified[key] = row
        rows = sorted(cached + new_rows, key=lambda r: (r["complex_id"], r["algorithm"], r["condition"]))
        states = Counter(r["outcome_state"] for r in rows)
        progress = {"planned": analysis["design"]["attempts"], "pending": pending,
                    "states": dict(states), "issues": [r for r in rows if r["outcome_state"] == "unresolved"]}
        _atomic_write_json(out / "scoring_progress.json", progress)
        print(json.dumps({"planned": progress["planned"], "pending": pending, "states": dict(states)}), flush=True)
        if not follow or not pending:
            break
        time.sleep(45)
    if finalize:
        require(not pending and not states["unresolved"], "Collection incomplete; inspect scoring_progress.json")
        table = summarize(rows,analysis)
        text = StringIO(newline="")
        writer = csv.DictWriter(text, list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)
        observations = out/"factorial_observations.csv"
        with TemporaryDirectory(dir=out,prefix=".result-") as tmp:
            partial = Path(tmp)/observations.name
            partial.write_bytes(text.getvalue().encode("utf-8"))
            os.replace(partial,observations)
        _atomic_write_json(out/"factorial_results.json",{
            "schema_version":"official_dockq_factorial_results_v1", "complete":True,
            "analysis_plan_sha256":runtime["analysis_hash"], "quality_metric":metric,
            "quality_metric_sha256":quality_metric_sha256(metric), "evaluator":preflight,
            "selection_sha256":_file_sha256(study/"selection.json"),
            "planned_workers_sha256":_file_sha256(study/"planned_workers.json"),
            "observations_sha256":_file_sha256(observations), "bootstrap":analysis["analysis"]["uncertainty"],
            "budget_deviations_on_no_pose":[r for r in rows if r["outcome_state"]=="no_valid_pose"
                                           and r["actual_eval_count"]!=r["evaluation_budget"]], **table})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",type=Path,default=ROOT/"runs/fatorial_dockq_20261119")
    parser.add_argument("--out",type=Path,default=ROOT/"output/fatorial_dockq_20261119")
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--dependency-path",type=Path,default=ROOT/".local/dockq_standard_runtime")
    parser.add_argument("--finalize",action="store_true")
    parser.add_argument("--follow", action="store_true", help="Score new terminal receipts every 45 seconds")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    for var in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
        os.environ[var] = "1"
    study, out = args.study.resolve(strict=True),args.out.resolve(strict=True)
    with _BatchFileLock(out/".scoring.lock"):
        evaluate(study, out, args.workers, args.finalize, args.dependency_path.resolve(strict=True), args.follow)


if __name__ == "__main__":
    main()
