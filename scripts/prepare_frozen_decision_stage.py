"""Prepare completed frozen pairs, rescore official DockQ and run one decision stage.

Validation selects a rule; test evaluates the frozen rule. Neither role trains a
model, starts docking, changes the split or claims measured GVA runtime savings.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
import run_frozen_decision_study as collection
from docking_confidence.application.batch import _validate_manifest
from docking_confidence.application.decision_validation import (
    _read_record, contract_sha256, evaluate_frozen_policy, select_decision_policy,
)
from docking_confidence.application.standard_dataset import build_standard_paired_dataset
from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.io.config import config_sha256, load_config, validate_config
from docking_confidence.io.dataset import load_paired_dataset


def read_object(path):
    value = json.loads(Path(path).read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def audit_completed_study(args):
    """Verify stage/roles and every planned shard before preparing any CSV."""
    context = collection.frozen_context(args.role, args.protocol, args.policy)
    protocol = context["protocol"]
    study = Path(args.study_dir).resolve(strict=True)
    manifest_path = study / "study_manifest.json"
    manifest = _read_record(manifest_path, collection.RUN_SCHEMA)
    if (manifest["role"] != args.role or manifest["protocol_sha256"] != protocol["record_sha256"]
            or manifest["stage_record_path"] != str(context["record_path"])
            or manifest["stage_file_sha256"] != context["record_file_sha256"]
            or manifest["stage_record_sha256"] != (context["policy"] or protocol)["record_sha256"]):
        raise ValueError("Study is not bound to the requested frozen protocol/policy and role")
    if manifest.get("automatic_skipping") is not False:
        raise ValueError("Decision evidence requires both planned arms; skipped FULL is not paired evidence")
    selection_path = Path(manifest["selection_path"]).resolve(strict=True)
    if collection.file_sha256(selection_path) != manifest["selection_file_sha256"]:
        raise ValueError("Study selection file changed")
    selection = collection.checked_selection(selection_path, protocol)
    if selection["record_sha256"] != manifest["selection_record_sha256"]:
        raise ValueError("Study selection record changed")
    source_config_path = Path(args.source_config).resolve(strict=True)
    source_config = load_config(source_config_path)
    if "quality_metric" in source_config:
        raise ValueError("Source batch preparation must use the original legacy-metric configuration")
    expected_standard = validate_config({**deepcopy(source_config),
        "quality_metric": protocol["model"]["source_config"]["quality_metric"]})
    if config_sha256(expected_standard) != protocol["model"]["config_sha256"]:
        raise ValueError("Source config plus official metric differs from the frozen model configuration")
    selected = {item["complex_id"]: item for item in selection["targets"] if item["role"] == args.role}
    declared_ids = [cid for partition in manifest["partitions"] for cid in partition]
    if len(declared_ids) != len(set(declared_ids)) or set(declared_ids) != set(selected):
        raise ValueError("Study partitions differ from the frozen selected role")
    batches, sources = [], []
    for index, ids in enumerate(manifest["partitions"]):
        if not ids:
            continue
        batch = study / f"shard_{index + 1:02d}" / "batch"
        batch_manifest_path, status_path = batch / "batch_manifest.json", batch / "batch_status.json"
        batch_manifest = read_object(batch_manifest_path)
        _validate_manifest(batch_manifest)
        plan = batch_manifest["plan"]
        if plan["jobs"] != manifest["jobs"]:
            raise ValueError(f"Shard recipe differs from the frozen study: {batch}")
        if plan.get("confidence_gate", {}).get("reason") != "article_paired_data_collection":
            raise ValueError("Shard was not collected with the unconditional paired fallback")
        if {target["complex_id"] for target in plan["targets"]} != set(ids):
            raise ValueError("Shard targets differ from the frozen partition")
        for target in plan["targets"]:
            item = selected[target["complex_id"]]
            for field in ("complex_fingerprint_sha256", "receptor_sha256", "ligand_sha256", "reference_sha256"):
                if target[field] != item[field]:
                    raise ValueError(f"Shard molecular input binding changed: {target['complex_id']} {field}")
        status = read_object(status_path)
        planned_keys = {f"{cid}::{job['job_id']}" for cid in ids for job in plan["jobs"]}
        if (status.get("plan_sha256") != batch_manifest["plan_sha256"]
                or status.get("state") not in {"COMPLETED", "COMPLETED_WITH_ERRORS"}
                or set(status.get("tasks", {})) != planned_keys
                or any(task.get("state") not in {"SUCCEEDED", "FAILED"} for task in status["tasks"].values())):
            raise ValueError(f"Every planned job must be completed or explicitly failed before preparation: {batch}")
        for target in plan["targets"]:
            for job in plan["jobs"]:
                directory = batch / "targets" / target["complex_id"] / "jobs" / job["job_id"]
                if status["tasks"][f"{target['complex_id']}::{job['job_id']}"]["state"] == "SUCCEEDED":
                    result = read_object(directory / "receipt.json")["engine_result"]
                    expected_budget = protocol["model"]["source_config"]["execution_contract"][job["protocol"] + "_evaluation_budget"]
                    if result.get("actual_eval_count") != expected_budget or result.get("stop_reason") != "objective_evaluation_ceiling":
                        raise ValueError(f"Successful arm did not complete its frozen effective budget: {directory}")
        batches.append(batch)
        sources.append({"batch_dir": str(batch), "manifest_sha256": collection.file_sha256(batch_manifest_path),
            "status_sha256": collection.file_sha256(status_path), "batch_id": batch_manifest["batch_id"],
            "planned_jobs": len(planned_keys)})
    if sum(item["planned_jobs"] for item in sources) != manifest["planned_jobs"]:
        raise ValueError("Completed shard jobs do not match the frozen planned denominator")
    return context, manifest, batches, sources, source_config_path, source_config


def empty_failed_preparation(report):
    """An entirely failed shard is retained; corrupt/incomplete evidence is not."""
    return (report["counts"]["candidate_rows"] == 0 and bool(report.get("issues"))
        and all(issue.get("code") == "official_loader" and issue.get("message") in {
            "dataset is empty", "no paired rows available for training"} for issue in report["issues"]))


def merge_preparations(reports, output, config, protocol, role):
    frames, failures, unmatched, source_records = [], [], [], []
    allowed = {(r["complex_fingerprint_sha256"], r["algorithm"], r["seed"])
               for r in protocol["planned_runs"][role]}
    seen = set()
    for report in reports:
        if not report["valid"] and not empty_failed_preparation(report):
            raise ValueError(f"Shard preparation is invalid: {report['report_path']}: {report['issues']}")
        path = Path(report["dataset_path"] or report["candidate_dataset_path"])
        snapshot = path.read_bytes()
        if collection.file_sha256(path) != report["dataset_sha256"]:
            raise ValueError("Prepared source dataset changed")
        frame = pd.read_csv(BytesIO(snapshot), float_precision="round_trip")
        keys = {(row.complex_fingerprint_sha256, row.algorithm, int(row.seed)) for row in frame.itertuples()}
        if len(keys) != len(frame) or not keys.issubset(allowed) or seen & keys:
            raise ValueError("Merged dataset duplicates or introduces unplanned run identities")
        seen.update(keys)
        if not frame.empty and not frame.config_sha256.eq(config_sha256(config)).all():
            raise ValueError("Prepared CSV has a different source configuration")
        # FullFallbackGate was measured during collection; the GVA model was not.
        # Preserve its recorded diagnostic time without assigning it to the GVA.
        if "confidence_overhead_sec" in frame:
            frame = frame.rename(columns={"confidence_overhead_sec": "historical_confidence_overhead_sec"})
        frames.append(frame)
        failures.extend(report.get("failures", []))
        unmatched.extend(report.get("unmatched", []))
        source_records.append({"report_path": report["report_path"],
            "report_sha256": collection.file_sha256(report["report_path"]),
            "dataset_path": str(path), "dataset_sha256": report["dataset_sha256"]})
    merged = pd.concat(frames, ignore_index=True).sort_values(
        ["complex_fingerprint_sha256", "algorithm", "seed"], kind="stable")
    snapshot = merged.to_csv(index=False, lineterminator="\n").encode("utf-8")
    if not merged.empty:
        load_paired_dataset(BytesIO(snapshot), config["features"]["columns"], config["contact_graph"],
            docking_task=config.get("docking_task"), feature_schema=config["features"]["schema_version"])
    path = output / "historical_paired_dataset.csv"
    with path.open("xb") as handle:
        handle.write(snapshot)
    report = {"schema_version": "frozen_decision_stage_merge_v1", "role": role,
        "dataset_path": str(path), "dataset_sha256": collection.file_sha256(path),
        "config_sha256": config_sha256(config), "planned_runs": len(allowed), "valid_paired_rows": len(merged),
        "failed_or_missing_pairs": len(allowed) - len(merged), "failures": failures, "unmatched": unmatched,
        "source_preparations": source_records,
        "runtime_context": "FullFallback overhead retained as historical_confidence_overhead_sec; current GVA overhead was not measured and is unavailable"}
    write_new(output / "merge_report.json", report)
    return merged, report


def prepare_stage(args):
    context, manifest, batches, sources, source_config_path, source_config = audit_completed_study(args)
    protocol = context["protocol"]
    if args.check_only:
        return {"status": "preflight_passed", "role": args.role, "batches": len(batches),
            "planned_runs": len(protocol["planned_runs"][args.role]), "docking_executed": False}
    output = Path(args.out_dir).resolve()
    forbidden = [Path(args.study_dir).resolve(), Path(protocol["model"]["artifact_path"]).resolve().parent]
    if any(output == path or path in output.parents or output in path.parents for path in forbidden):
        raise ValueError("Stage output must be new and separate from the collected study and frozen model")
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "stage_commitment.json", {"schema_version": "frozen_decision_stage_preparation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": args.role,
        "study_manifest_path": str(Path(args.study_dir).resolve() / "study_manifest.json"),
        "study_manifest_sha256": collection.file_sha256(Path(args.study_dir) / "study_manifest.json"),
        "protocol_sha256": protocol["record_sha256"], "source_config_path": str(source_config_path),
        "source_config_file_sha256": collection.file_sha256(source_config_path), "source_batches": sources,
        "automatic_test_collection": False, "training_performed": False})
    preparations = []
    for batch in batches:
        preparation = build_training_dataset(batch, output / "preparations" / batch.parent.name,
            profile="operational", fingerprint_alias_policy="reject", config_path=source_config_path)
        preparations.append(preparation)
    frame, merged = merge_preparations(preparations, output, source_config, protocol, args.role)
    standard_root = output / "standard"
    if not frame.empty:
        converted = build_standard_paired_dataset(merged["dataset_path"], batches, standard_root,
            source_config_path=source_config_path, python_executable=args.python_executable,
            dependency_path=args.dependency_path, max_workers=args.workers,
            expected_source_sha256=merged["dataset_sha256"],
            progress_callback=lambda fraction, message: print(message, flush=True))
        if not converted["valid"]:
            raise ValueError(f"Official DockQ conversion failed; no policy was selected or evaluated: {converted['issues']}")
        if converted["config_sha256"] != protocol["model"]["config_sha256"]:
            raise ValueError("Converted standard configuration differs from the frozen protocol")
    else:
        # Every planned pair failed. Retain an empty declared input so the core
        # reports zero coverage and no useful rule instead of deleting the role.
        standard_root.mkdir()
        path = standard_root / "paired_dataset.csv"
        with path.open("xb") as handle:
            handle.write(frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))
        converted = {"valid": True, "dataset_path": str(path), "dataset_sha256": collection.file_sha256(path),
            "config_sha256": protocol["model"]["config_sha256"], "quality_status": "no_observed_pairs_no_quality_measurements"}
        write_new(standard_root / "empty_role_report.json", converted)
    # Recheck immutable source manifests/statuses after preparation and rescoring.
    for source in sources:
        batch = Path(source["batch_dir"])
        if (collection.file_sha256(batch / "batch_manifest.json") != source["manifest_sha256"]
                or collection.file_sha256(batch / "batch_status.json") != source["status_sha256"]):
            raise ValueError("Collected batch changed during stage preparation")
    if args.role == "validation":
        result = select_decision_policy(args.protocol, converted["dataset_path"], output / "decision",
            expected_dataset_sha256=converted["dataset_sha256"])
    else:
        result = evaluate_frozen_policy(args.policy, converted["dataset_path"], output / "decision",
            expected_dataset_sha256=converted["dataset_sha256"])
    report = result["report"]
    summary = {"schema_version": "frozen_decision_stage_summary_v1", "role": args.role,
        "status": report["status"], "decision_capable": report.get("decision_capable", False),
        "protocol_sha256": protocol["record_sha256"], "standard_dataset_path": converted["dataset_path"],
        "standard_dataset_sha256": converted["dataset_sha256"], "report_path": result["report_path"],
        "policy_path": result.get("policy_path", args.policy), "planned_runs": merged["planned_runs"],
        "valid_paired_rows": merged["valid_paired_rows"], "failed_or_missing_pairs": merged["failed_or_missing_pairs"],
        "selected": report["selected"], "baseline": report["baseline"], "difference": report["difference"],
        "criteria": report["criteria"], "runtime_context": merged["runtime_context"],
        "training_performed": False, "test_collection_started": False,
        "scope": "prespecified empirical preservation and coverage criteria; no population-risk guarantee or measured deployment economy"}
    write_new(output / "stage_summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--role", choices=("validation", "test"), required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--policy")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--source-config", default=str(ROOT / "configs/local_refinement_operational_candidate.yaml"))
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--dependency-path", default=str(ROOT / ".local/dockq_standard_runtime"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    summary = prepare_stage(args)
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
