"""Reproduce REDUCED provenance, failure states and pose comparisons.

Rejected candidates are exported only under diagnostic_rejected_pose for the
comparison requested by the investigation. They are never batch successes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking_confidence.adapters.dockingpp_archive_engine import DockingPPArchiveEngine, DockingPPReducedError
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch_launcher import load_dockingpp_archive_contract
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ProtocolMode
from docking_confidence.evaluation.reference_dockq import evaluate_reference_pose
from docking_confidence.io.batch_discovery import discover_ppi4dock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit_zip(path, output):
    rows = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if "/jobs/" not in name or not name.endswith("/output/summary.json"):
                continue
            summary = json.loads(archive.read(name))
            prefix, remainder = name.split("/jobs/", 1)
            job = remainder.split("/", 1)[0]
            task_root = prefix + "/jobs/" + job + "/"
            engine_root = name[:-len("summary.json")]
            pose_path = summary.get("pose_output_path")
            pose = json.loads(archive.read(engine_root + pose_path)) if pose_path else {}
            pdb_name = task_root + "postprocess/predicted_complex.pdb"
            reference_name = task_root + "postprocess/reference_dockq.json"
            row = {
                "complex_id": prefix.split("/")[-1], "job_id": job,
                "engine_name": pose.get("engine_name"),
                "search_strategy": summary.get("search_strategy"),
                "generation": pose.get("generation"),
                "score_cheap": pose.get("score_cheap"),
                "final_rescore": pose.get("final_rescore"),
                "total_n_eval": summary.get("total_n_eval", summary.get("n_eval_total")),
                "fallback_reason": summary.get("fallback_reason"),
                "fallback_executed": summary.get("fallback_executed"),
                "hard_clash": pose.get("meta", {}).get("final_hard_clash_gate", {}).get("triggered"),
                "root_metrics_exists": engine_root + "metrics.jsonl" in archive.namelist(),
                "fallback_metrics_exists": engine_root + "fallback_full/metrics.jsonl" in archive.namelist(),
                "pdb_sha256": hashlib.sha256(archive.read(pdb_name)).hexdigest() if pdb_name in archive.namelist() else None,
                "pose_coords_sha256": hashlib.sha256(json.dumps(pose.get("coords")).encode()).hexdigest(),
                "reference_metrics": json.loads(archive.read(reference_name)) if reference_name in archive.namelist() else None,
            }
            rows.append(row)
    output.write_text(json.dumps({"source_zip_sha256": sha(path), "runs": rows}, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, help="optional legacy ZIP; defaults to the bundled motor")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--complex-id", default="1a2y_AB")
    parser.add_argument("--no-pocket-complex-id", default="1a4y_CD")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--top-pockets", type=int, default=32)
    parser.add_argument("--docking-task", choices=["surface_discovery", "local_refinement"], default="surface_discovery",
                        help="historical surface regression by default; quality refinement has a separate paired validator")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--results-zip", type=Path)
    parser.add_argument("--full-control", action="store_true")
    parser.add_argument("--expect-no-feasible-pocket", action="store_true",
                        help="assert the additional historical no-pocket case still has no feasible region")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.results_zip:
        audit_zip(args.results_zip, output / "original_zip_audit.json")
    targets = {target.complex_id: target for target in discover_ppi4dock(args.input_root).eligible_targets}
    if args.archive:
        pin = load_dockingpp_archive_contract()
        engine = DockingPPArchiveEngine(
            args.archive, output / "snapshots", config_entry=pin.config_entry,
            expected_archive_sha256=pin.archive_sha256, expected_config_sha256=pin.config_sha256,
            python_executable=sys.executable,
        )
    else:
        engine = DockingPPLocalEngine()
    jobs = build_execution_plan(
        algorithms=[Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID],
        protocol_mode=ProtocolMode.REDUCED,
        base_config=BaseSearchConfig(args.generations, args.population, args.top_pockets, args.docking_task), seeds=[args.seed],
    )
    assignments = [(args.complex_id, job) for job in jobs]
    if args.no_pocket_complex_id:
        assignments.append((args.no_pocket_complex_id, jobs[0]))
    if args.full_control:
        full = build_execution_plan(
            algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.FULL,
            base_config=BaseSearchConfig(args.generations, args.population, args.top_pockets, args.docking_task), seeds=[args.seed],
        )[0]
        assignments.append((args.complex_id, full))
    rows = []
    for complex_id, job in assignments:
        target = targets[complex_id]
        work = output / complex_id / job.job_id
        print(f"RUN {complex_id} {job.job_id}", flush=True)
        row = {"complex_id": complex_id, "job": asdict(job), "counted_as_success": False}
        try:
            result = engine.run(job=job, protein_path=target.receptor_path, peptide_path=target.ligand_path, work_dir=work)
            row["status"] = "SUCCEEDED"
            row["counted_as_success"] = True
            pose_path = work / result.pose_artifact_path
        except DockingPPReducedError as exc:
            row.update(status=exc.status, reason=exc.reason)
            # Inspect the first rejected candidate for a reproducible diagnostic
            # comparison; do not pass this through the successful batch path.
            pose_path = next(work.glob("dockingpp_runs/*/output/*/artifacts/best_pose.json"), None)
        out = next(work.glob("dockingpp_runs/*/output"))
        summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
        events = [json.loads(line) for path in out.rglob("metrics.jsonl")
                  for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        operator_events = [event for event in events if event.get("name") == "search.operator_completed"]
        row.update(
            executed_engines=dict(Counter(event["engine_name"] for event in operator_events)),
            operator_contexts=[dict(zip(("engine_name", "search_strategy", "seed", "pop_size", "generations", "pocket_id"), values))
                               for values in sorted({tuple(str(event.get(key)) for key in ("engine_name", "search_strategy", "seed", "pop_size", "generations", "pocket_id")) for event in operator_events})],
            root_metrics=str(out / "metrics.jsonl"),
            resolved_config=str(out.parent / "resolved_config.yaml"),
            fallback_executed=summary.get("fallback_executed"),
            total_n_eval=summary.get("total_n_eval", summary.get("n_eval_total")),
        )
        if pose_path:
            role = "postprocess" if row["counted_as_success"] else "diagnostic_rejected_pose"
            evaluated = evaluate_reference_pose(
                receptor_model_pdb=target.receptor_path, ligand_template_pdb=target.ligand_path,
                reference_complex_pdb=target.reference_path, receptor_chain=target.receptor_chain,
                ligand_chain=target.ligand_chain, pose_json=pose_path, output_dir=work / role,
            )
            row.update(pdb_role=role, pdb_path=str(evaluated.complex_pdb_path),
                       pdb_sha256=sha(evaluated.complex_pdb_path),
                       dockq=evaluated.dockq, fnat=evaluated.fnat, irmsd=evaluated.irmsd, lrmsd=evaluated.lrmsd)
        print(json.dumps({key: row.get(key) for key in ("status", "executed_engines", "pdb_role", "pdb_sha256", "dockq")}), flush=True)
        rows.append(row)
        (output / "validation.json").write_text(json.dumps({"engine": engine.audit_descriptor(), "runs": rows}, indent=2), encoding="utf-8")
    expected = {"ag": "AGSearch", "pso": "PSOSearch", "hybrid": "ABCGAVGOSSearch"}
    for row in rows[:3]:
        assert row["fallback_executed"] is False
        if row["status"] != "NO_FEASIBLE_POCKET":
            assert set(row["executed_engines"]) == {expected[row["job"]["algorithm"]]}
    if args.no_pocket_complex_id:
        assert rows[3]["fallback_executed"] is False
        if args.expect_no_feasible_pocket:
            assert rows[3]["status"] == "NO_FEASIBLE_POCKET"
        if rows[3]["status"] == "NO_FEASIBLE_POCKET":
            assert not rows[3]["executed_engines"]
    print(f"VALIDATED: {output / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
