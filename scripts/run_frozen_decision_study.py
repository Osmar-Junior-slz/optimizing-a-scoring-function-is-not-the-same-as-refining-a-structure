"""Collect frozen validation or test pairs; never train, select or install a model.

The batch's legacy quality sidecars remain legacy. Official DockQ rescoring and
dataset preparation are explicit later steps; this runner reports counts only.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import (
    CANCEL_FILENAME, LOCK_FILENAME, _atomic_write_json, _BatchFileLock, _job_snapshot,
    clear_batch_cancel_request, request_batch_cancel, run_batch,
)
from docking_confidence.application.decision_validation import (
    PROTOCOL_SCHEMA, _checked_policy, _checked_protocol, _read_record, contract_sha256,
)
from docking_confidence.application.operational import load_operational_config
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, ProtocolMode
from docking_confidence.io.batch_discovery import _validate_target_contents, discover_ppi4dock
from docking_confidence.io.config import config_sha256, load_config
from docking_confidence.io.quality_context import validate_quality_metric


ROOT = Path(__file__).resolve().parents[1]
SELECTION_SCHEMA = "frozen_decision_study_selection_v1"
RUN_SCHEMA = "frozen_decision_study_run_v1"
THREAD_LIMITS = {name: "1" for name in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
)}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_context(role, protocol_path=None, policy_path=None):
    """Authenticate the required stage without reading any validation/test CSV."""
    if role == "validation":
        if not protocol_path or policy_path:
            raise ValueError("Validation requires --protocol and does not accept --policy")
        protocol, _ = _checked_protocol(_read_record(protocol_path, PROTOCOL_SCHEMA))
        record_path, policy = Path(protocol_path).resolve(strict=True), None
        claim = record_path.parent / "policy_selection_claim.json"
        if claim.exists():
            raise ValueError("Validation has already selected a policy; collection is closed")
    elif role == "test":
        if not policy_path or protocol_path:
            raise ValueError("Test requires --policy and does not accept --protocol")
        policy, protocol, _ = _checked_policy(policy_path)
        if policy.get("status") != "selected" or policy.get("threshold") is None:
            raise ValueError("No useful frozen rule was selected; do not collect the final test")
        record_path = Path(policy_path).resolve(strict=True)
        if (record_path.parent / "final_evaluation_claim.json").exists():
            raise ValueError("The final test has already been evaluated; collection is closed")
    else:
        raise ValueError("Role must be validation or test")
    if protocol.get("analysis_role") != "prospective" or protocol.get("prospective_context_issues"):
        raise ValueError("This runner requires a prospective protocol with bound metric and execution context")
    source_config = protocol["model"]["source_config"]
    source_path = Path(protocol["model"]["config_path"]).resolve(strict=True)
    if config_sha256(load_config(source_path)) != protocol["model"]["config_sha256"]:
        raise ValueError("Frozen source configuration file changed")
    validate_quality_metric(source_config.get("quality_metric"))
    return {"protocol": protocol, "policy": policy, "record_path": record_path,
            "record_file_sha256": file_sha256(record_path)}


def checked_selection(selection_path, protocol):
    selection_path = Path(selection_path).resolve(strict=True)
    selection = _read_record(selection_path, SELECTION_SCHEMA)
    targets = selection.get("targets")
    if not isinstance(targets, list) or not targets:
        raise ValueError("Selection must contain predeclared targets")
    seen_ids, seen_groups = set(), set()
    for item in targets:
        role = item.get("role")
        fingerprint = item.get("complex_fingerprint_sha256", item.get("fingerprint"))
        if item.get("fingerprint", fingerprint) != fingerprint:
            raise ValueError("Selection fingerprint aliases disagree")
        if role not in {"validation", "test"} or fingerprint in seen_groups or item.get("complex_id") in seen_ids:
            raise ValueError("Selection roles, molecular groups and complex IDs must be unique")
        if not isinstance(fingerprint, str) or len(fingerprint) != 64:
            raise ValueError("Invalid selected molecular fingerprint")
        seen_groups.add(fingerprint)
        seen_ids.add(item["complex_id"])
    for role in ("validation", "test"):
        groups = {item.get("complex_fingerprint_sha256", item.get("fingerprint"))
                  for item in targets if item["role"] == role}
        if groups != set(protocol["roles"][role + "_groups"]):
            raise ValueError(f"Selection {role} groups differ from the frozen protocol")
    if (selection.get("algorithms") != protocol["algorithms"]
            or selection.get("seeds") != protocol["seeds"]):
        raise ValueError("Selection algorithms/seeds differ from the frozen protocol")
    if set(protocol["algorithms"]) != {"ag", "pso", "hybrid"} or protocol["seeds"] != [1]:
        raise ValueError("This collection requires AG, PSO, HYBRID and seed 1")
    return selection


def checked_jobs(protocol):
    base = load_operational_config()
    expected = {"schema_version": "operational_cascade_v1",
        "reduced_evaluation_budget": 5700, "full_evaluation_budget": 7600,
        "population": 100, "initialization": "uniform", "reduced_search_scale": .5,
        "max_translation": 5., "max_rotation_degrees": 25.}
    if protocol["model"]["source_config"].get("execution_contract") != expected:
        raise ValueError("Frozen study must explicitly bind REDUCED 5700 / FULL 7600 and the operational controls")
    jobs = build_execution_plan(algorithms=[Algorithm(value) for value in protocol["algorithms"]],
        protocol_mode=ProtocolMode.OPERATIONAL, base_config=base, seeds=[1])
    for job in jobs:
        profile = job.profile
        if (profile.evaluation_budget != expected[job.protocol.value + "_evaluation_budget"]
                or profile.population != 100 or profile.initialization != "uniform"
                or profile.max_translation != 5. or profile.max_rotation_degrees != 25.
                or profile.search_scale != (.5 if job.protocol.value == "reduced" else 1.)):
            raise ValueError("Current operational recipe differs from the frozen study contract")
    return jobs


def selected_report(selection, role):
    source = Path(selection["input_root"]).resolve(strict=True)
    # Discovery reads molecular inputs only. It neither loads saved outcomes nor
    # calculates a quality label for either study role.
    report = discover_ppi4dock(source, validate_contents=False)
    discovered = {target.complex_id: target for target in report.targets}
    selected = []
    for item in selection["targets"]:
        if item["role"] != role:
            continue
        target = discovered.get(item["complex_id"])
        if target is None or not target.valid:
            raise ValueError(f"Selected target is missing or invalid: {item['complex_id']}")
        target = _validate_target_contents(target, mobile_partner_domain=report.mobile_partner_domain)
        if not target.valid or target.eligible is not True:
            raise ValueError(f"Selected target is invalid or ineligible: {item['complex_id']}")
        if Path(item["path"]).resolve(strict=True) != target.directory.resolve():
            raise ValueError(f"Selected target path changed: {target.complex_id}")
        fingerprint = item.get("complex_fingerprint_sha256", item.get("fingerprint"))
        if target.complex_fingerprint_sha256 != fingerprint:
            raise ValueError(f"Selected molecular identity changed: {target.complex_id}")
        for component in ("receptor", "ligand", "reference"):
            path = Path(getattr(target, component + "_path")).resolve(strict=True)
            if (Path(item[component + "_path"]).resolve(strict=True) != path
                    or item[component + "_sha256"] != getattr(target, component + "_sha256")
                    or file_sha256(path) != item[component + "_sha256"]):
                raise ValueError(f"Selected {component} input changed: {target.complex_id}")
        selected.append(target)
    if not selected:
        raise ValueError("No selected targets for this role")
    return replace(report, targets=tuple(sorted(selected, key=lambda t: t.complex_id)),
                   validation_performed=True, issues=())


def partition_targets(report, workers):
    partitions, loads = [[] for _ in range(workers)], [0] * workers
    for target in sorted(report.targets, key=lambda t: (
            -max(1, t.receptor_atom_count or 1) * max(1, t.ligand_atom_count or 1), t.complex_id)):
        index = min(range(workers), key=lambda i: (loads[i], i))
        partitions[index].append(target)
        loads[index] += max(1, target.receptor_atom_count or 1) * max(1, target.ligand_atom_count or 1)
    return [tuple(sorted(partition, key=lambda t: t.complex_id)) for partition in partitions]


def study_plan(args):
    context = frozen_context(args.role, args.protocol, args.policy)
    protocol = context["protocol"]
    selection_path = Path(args.selection).resolve(strict=True)
    selection = checked_selection(selection_path, protocol)
    jobs = checked_jobs(protocol)
    # Only the frozen grid is supported: no silently omitted planned run IDs.
    for role in ("validation", "test"):
        expected = {(group, algorithm, 1) for group in protocol["roles"][role + "_groups"]
                    for algorithm in protocol["algorithms"]}
        records = protocol["planned_runs"][role]
        actual = {(item["complex_fingerprint_sha256"], item["algorithm"], item["seed"]) for item in records}
        if actual != expected or len(records) != len(expected):
            raise ValueError("Frozen planned runs differ from the supported collection grid")
        for item in records:
            if item.get("run_id") not in (None, f"{item['algorithm']}-operational-reduced-seed1"):
                raise ValueError("Explicit frozen run_id differs from this engine's run identity")
    report = selected_report(selection, args.role)
    output = Path(args.out_dir).resolve()
    forbidden = [report.input_root.resolve(), Path(protocol["model"]["artifact_path"]).resolve().parent]
    if any(output == path or path in output.parents or output in path.parents for path in forbidden):
        raise ValueError("Study output must be separate from molecular inputs and frozen model")
    partitions = partition_targets(report, args.workers)
    manifest = {"schema_version": RUN_SCHEMA, "role": args.role,
        "selection_path": str(selection_path), "selection_file_sha256": file_sha256(selection_path),
        "selection_record_sha256": selection["record_sha256"],
        "stage_record_path": str(context["record_path"]), "stage_file_sha256": context["record_file_sha256"],
        "stage_record_sha256": (context["policy"] or protocol)["record_sha256"],
        "protocol_sha256": protocol["record_sha256"], "config_sha256": protocol["model"]["config_sha256"],
        "quality_metric": protocol["model"]["source_config"]["quality_metric"],
        "execution_contract": protocol["model"]["source_config"]["execution_contract"],
        "operational_config_sha256": file_sha256(ROOT / "configs/operational_cascade.yaml"),
        "runner_sha256": file_sha256(__file__), "workers": args.workers, "timeout_seconds": args.timeout,
        "jobs": [_job_snapshot(job) for job in jobs],
        "partitions": [[target.complex_id for target in partition] for partition in partitions],
        "planned_complexes": len(report.targets), "planned_jobs": len(report.targets) * len(jobs),
        "thread_limits": THREAD_LIMITS, "failed_run_policy": "first_physical_failure_is_final_no_automatic_retry",
        "training": "not_performed", "policy_selection": "not_performed", "automatic_skipping": False,
        "quality_status": "standard_rescoring_pending_legacy_batch_metrics_are_not_official_DockQ"}
    manifest["record_sha256"] = contract_sha256(manifest)
    return context, report, jobs, partitions, manifest


def bind_manifest(output, manifest):
    path = output / "study_manifest.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Resume rejected: selection, stage, inputs, recipe or worker plan changed")
    else:
        unexpected = [p.name for p in output.iterdir() if p.name != ".frozen_study.lock"]
        if unexpected:
            raise FileExistsError(f"Study output is not a fresh directory: {unexpected}")
        with path.open("x", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")


class FrozenRunEngine(DockingPPLocalEngine):
    """Keep first physical failures in the planned denominator across resumes."""

    def run(self, **kwargs):
        directory = Path(kwargs["work_dir"])
        failure = directory.parent / "error.json"
        if failure.exists():
            original = json.loads(failure.read_text(encoding="utf-8"))
            archive = directory.parent / "first_failure.json"
            if not archive.exists():
                with archive.open("x", encoding="utf-8") as handle:
                    json.dump(original, handle, indent=2, ensure_ascii=False)
            raise RuntimeError("Frozen study preserves the first failed attempt; no docking retry was executed")
        started = time.perf_counter()
        try:
            return super().run(**kwargs)
        finally:
            _atomic_write_json(directory / "benchmark_timing.json", {
                "elapsed_sec": time.perf_counter() - started,
                "basis": "complete_physical_engine_call_wall_seconds_including_failed_calls"})


def run_worker(args, report, jobs, partitions):
    output = Path(args.out_dir).resolve()
    folder = output / f"shard_{args.worker_index + 1:02d}"
    selected = partitions[args.worker_index]
    if not selected:
        return {"state": "EMPTY", "planned_jobs": 0, "succeeded": 0, "failed": 0}
    result = run_batch(report.input_root, folder / "batch", jobs=jobs,
        engine=FrozenRunEngine(timeout_seconds=args.timeout),
        report=replace(report, targets=selected), collect_operational_pairs=True)
    return {"state": result.state, "planned_jobs": result.planned_jobs,
            "succeeded": result.succeeded, "failed": result.failed,
            "skipped_existing": result.skipped_existing}


def launch_workers(args, partitions):
    output = Path(args.out_dir).resolve()
    env = os.environ.copy()
    env.update(THREAD_LIMITS)
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
    children, handles = [], []
    try:
        # A previous parent can have exited while its workers finish the current
        # call. Test all shard locks before acknowledging any cancellation.
        with ExitStack() as locks:
            for index, selected in enumerate(partitions):
                if selected:
                    batch = output / f"shard_{index + 1:02d}" / "batch"
                    locks.enter_context(_BatchFileLock(batch / LOCK_FILENAME))
                    if (batch / CANCEL_FILENAME).exists() and not args.resume_cancelled:
                        raise ValueError("Study was cancelled; use --resume-cancelled to acknowledge and resume")
            if args.resume_cancelled:
                for index, selected in enumerate(partitions):
                    if selected:
                        clear_batch_cancel_request(output / f"shard_{index + 1:02d}" / "batch")
        for index, selected in enumerate(partitions):
            if not selected:
                continue
            handle = (output / f"worker_{index + 1:02d}.log").open("a", encoding="utf-8")
            handles.append(handle)
            command = [sys.executable, "-B", str(Path(__file__).resolve()), "--selection", str(Path(args.selection).resolve()),
                "--role", args.role, "--out-dir", str(output), "--workers", str(args.workers),
                "--timeout", str(args.timeout), "--worker-index", str(index)]
            command += ["--protocol", str(Path(args.protocol).resolve())] if args.protocol else ["--policy", str(Path(args.policy).resolve())]
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=handle,
                stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            children.append((index, child))
        print(json.dumps({"phase": "collecting_frozen_pairs", "role": args.role,
            "workers": len(children), "budgets": {"reduced": 5700, "full": 7600}}), flush=True)
        while True:
            workers = [{"index": index, "pid": child.pid, "exit_code": child.poll()} for index, child in children]
            _atomic_write_json(output / "workers.json", {"thread_limits": THREAD_LIMITS, "workers": workers})
            if all(worker["exit_code"] is not None for worker in workers):
                break
            time.sleep(2)
        if any(worker["exit_code"] != 0 for worker in workers):
            raise RuntimeError("Collection worker failed; inspect persisted worker logs and resume with the same command")
    except BaseException:
        # Cooperatively finish the current engine call, then stop each owned
        # shard. Do not terminate another study or retry any completed failure.
        for index, child in children:
            if child.poll() is None:
                request_batch_cancel(output / f"shard_{index + 1:02d}" / "batch")
        raise
    finally:
        for handle in handles:
            handle.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--role", choices=("validation", "test"), required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--protocol")
    parser.add_argument("--policy")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600.)
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--resume-cancelled", action="store_true", help="Acknowledge a prior cooperative cancellation before resuming")
    parser.add_argument("--check-only", action="store_true", help="Verify commitments and inputs without collecting docking results")
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("timeout must be finite and positive")
    if args.worker_index is not None and not 0 <= args.worker_index < args.workers:
        parser.error("worker-index must be inside workers")
    context, report, jobs, partitions, manifest = study_plan(args)
    if args.check_only:
        print(json.dumps({"phase": "preflight_passed", "role": args.role,
            "planned_complexes": len(report.targets), "planned_jobs": manifest["planned_jobs"],
            "protocol_sha256": context["protocol"]["record_sha256"], "docking_executed": False}), flush=True)
        return 0
    output = Path(args.out_dir).resolve()
    if args.worker_index is not None:
        existing = json.loads((output / "study_manifest.json").read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Worker commitment differs from frozen parent manifest")
        result = run_worker(args, report, jobs, partitions)
        _atomic_write_json(output / f"worker_{args.worker_index + 1:02d}_completion.json", result)
        print(json.dumps(result), flush=True)
        return 0 if result["state"] in {"COMPLETED", "COMPLETED_WITH_ERRORS", "EMPTY"} else 2
    with _BatchFileLock(output / ".frozen_study.lock"):
        bind_manifest(output, manifest)
        launch_workers(args, partitions)
        completions = [json.loads((output / f"worker_{index + 1:02d}_completion.json").read_text(encoding="utf-8"))
                       for index, selected in enumerate(partitions) if selected]
        summary = {"phase": "collection_completed", "role": args.role,
            "planned_jobs": manifest["planned_jobs"], "succeeded": sum(item["succeeded"] for item in completions),
            "failed": sum(item["failed"] for item in completions), "workers": completions,
            "quality_status": manifest["quality_status"], "training_performed": False,
            "policy_selection_performed": False, "automatic_decisions_authorized": False}
        _atomic_write_json(output / "collection_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
