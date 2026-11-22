"""Paired quality/time validation. References are read only after each search."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, DockingTask, ProtocolMode
from docking_confidence.evaluation.reference_dockq import evaluate_reference_pose
from docking_confidence.io.batch_discovery import discover_ppi4dock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--complexes", nargs="+")
    parser.add_argument("--algorithms", nargs="+", choices=[a.value for a in Algorithm], default=["ag", "pso", "hybrid"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--docking-task", choices=[t.value for t in DockingTask], default="local_refinement")
    parser.add_argument("--protocol", choices=[p.value for p in ProtocolMode], default="reduced_vs_full")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    targets = {t.complex_id: t for t in discover_ppi4dock(args.input_root).eligible_targets}
    chosen = args.complexes or sorted(targets)
    if set(chosen) - set(targets):
        raise ValueError("unknown complex IDs")
    engine = DockingPPLocalEngine()
    (output / "engine_audit.json").write_text(json.dumps(engine.audit_descriptor(), indent=2), encoding="utf-8")
    jobs = build_execution_plan(algorithms=[Algorithm(a) for a in args.algorithms],
                                protocol_mode=ProtocolMode(args.protocol),
                                base_config=BaseSearchConfig(args.generations, args.population, 32, DockingTask(args.docking_task)),
                                seeds=args.seeds)
    rows = []
    for complex_id in chosen:
        target = targets[complex_id]
        for job in jobs:
            if (output / "STOP").exists():
                print("STOP requested; completed results preserved", flush=True)
                return
            started = time.perf_counter()
            work = output / complex_id / job.job_id
            row = {"complex_id": complex_id, "job": asdict(job), "status": "RUNNING"}
            print(f"RUN {complex_id} {job.algorithm.value} {job.protocol.value} seed={job.seed}", flush=True)
            try:
                result = engine.run(job=job, protein_path=target.receptor_path, peptide_path=target.ligand_path, work_dir=work)
                row["engine_result"] = asdict(result)
                row["engine_wall_seconds"] = result.runtime_sec
                pose_path = work / result.pose_artifact_path
                metrics = evaluate_reference_pose(receptor_model_pdb=target.receptor_path,
                    ligand_template_pdb=target.ligand_path, reference_complex_pdb=target.reference_path,
                    receptor_chain=target.receptor_chain, ligand_chain=target.ligand_chain,
                    pose_json=pose_path, output_dir=work / "postprocess")
                row.update(status="SUCCEEDED", dockq=metrics.dockq, fnat=metrics.fnat,
                           irmsd=metrics.irmsd, lrmsd=metrics.lrmsd,
                           pdb_sha256=hashlib.sha256(metrics.complex_pdb_path.read_bytes()).hexdigest())
                pose = json.loads(pose_path.read_text(encoding="utf-8"))
                row["pose_generation"] = pose.get("generation")
                row["pose_meta"] = {key: pose.get("meta", {}).get(key) for key in (
                    "docking_task", "search_domain_low", "search_domain_high", "input_relative_placement_used",
                    "initialization_source", "solution_vector", "final_hard_clash_gate")}
                summary = json.loads((work / result.engine_summary_path).read_text(encoding="utf-8"))
                row["actual_objective_evaluations"] = summary.get("total_n_eval", summary.get("n_eval_total"))
                row["fallback_executed"] = summary.get("fallback_executed", False)
            except Exception as exc:
                row.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
            row["total_wall_seconds"] = time.perf_counter() - started
            rows.append(row)
            (output / "comparison.json").write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
            print(json.dumps({key: row.get(key) for key in ("complex_id", "status", "dockq", "fnat", "engine_wall_seconds", "total_wall_seconds", "error")}), flush=True)
    if any(row["status"] != "SUCCEEDED" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
