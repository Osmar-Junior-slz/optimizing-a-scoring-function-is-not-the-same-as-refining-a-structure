"""Reproducible serial 5-complex probe; never fits or installs a calibrator."""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import _atomic_write_json, run_batch
from docking_confidence.application.operational import load_operational_config
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, ProtocolMode
from docking_confidence.io.batch_discovery import discover_ppi4dock


class MeasuredEngine(DockingPPLocalEngine):
    def run(self, **kwargs):
        started = perf_counter()
        try:
            return super().run(**kwargs)
        finally:
            _atomic_write_json(Path(kwargs["work_dir"]) / "benchmark_timing.json", {
                "elapsed_sec": perf_counter() - started,
                "basis": "complete_engine_call_wall_seconds_including_failed_calls",
            })


def protected_snapshot(root: Path) -> dict:
    paths = [root / p for p in (
        "scripts/run_calibrator_pilot50.py", "scripts/analyze_factorial.py",
        "docs/EXPERIMENTO_FATORIAL.md", "configs/local_refinement_default.yaml",
        "configs/local_refinement_pilot.yaml", "src/dockingpp/_standalone_manifest.json")]
    previous = root / "runs/calibrator_pilot50_fixed_parallel_20261115"
    for name in ("batch_manifest.json", "experimental_protocol.json", "receipt.json"):
        paths.extend(previous.rglob(name))
    # Every scientific operator/config source is independently pinned too.
    paths.extend((root / "src/dockingpp").rglob("*.py"))
    return {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(set(paths)) if p.is_file()}


def evaluation_count(raw: dict, result: dict) -> int | None:
    pockets = raw.get("population_confidence_by_pocket", [])
    if pockets:
        return sum(p["n_eval_actual"] for p in pockets)
    # Rejected REDUCED has an aggregate failure report rather than a receipt.
    # Keep its actual work in the audit instead of silently dropping failures.
    for value in (raw.get("total_n_eval"), raw.get("n_eval_total"), result.get("actual_eval_count")):
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    return None


def summarize(study: Path) -> dict:
    batch = study / "batch"
    manifest=json.loads((batch / "batch_manifest.json").read_text(encoding="utf-8"))
    status=json.loads((batch / "batch_status.json").read_text(encoding="utf-8"))
    rows=[]
    for target in manifest["plan"]["targets"]:
        for job in manifest["plan"]["jobs"]:
            directory=batch / "targets" / target["complex_id"] / "jobs" / job["job_id"]
            receipt_path=directory / "receipt.json"
            receipt=json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
            result=receipt.get("engine_result",{})
            raw_paths=list((directory / "engine").rglob("output/summary.json"))
            raw=json.loads(raw_paths[0].read_text(encoding="utf-8")) if len(raw_paths)==1 else {}
            pockets=raw.get("population_confidence_by_pocket",[])
            timing_path=directory / "engine/benchmark_timing.json"
            timing=json.loads(timing_path.read_text()) if timing_path.is_file() else {}
            task=status["tasks"][f"{target['complex_id']}::{job['job_id']}"]
            row={"complex_id":target["complex_id"],"algorithm":job["algorithm"],"protocol":job["protocol"],
                 "state":task["state"],"requested_evaluations":job["profile"]["evaluation_budget"],
                 "actual_evaluations":evaluation_count(raw,result),
                 "engine_call_wall_sec":timing.get("elapsed_sec"),"engine_subprocess_wall_sec":result.get("runtime_sec"),
                 "dockq":result.get("dockq"),"fnat":result.get("fnat"),"irmsd":result.get("irmsd"),"lrmsd":result.get("lrmsd"),
                 "completed_generations":pockets[0].get("completed_generations") if len(pockets)==1 else None,
                 "valid_pose":task["state"]=="SUCCEEDED", "error":task.get("last_error")}
            decision=directory / "confidence_decision.json"
            if decision.is_file():
                decision=json.loads(decision.read_text(encoding="utf-8"))
                row.update(action=decision["action"],reason=decision["reason"],confidence_overhead_sec=decision.get("confidence_overhead_sec"))
            rows.append(row)
    pairs=[]
    for target in manifest["plan"]["targets"]:
        for algorithm in ("ag","pso","hybrid"):
            selected={r["protocol"]:r for r in rows if r["complex_id"]==target["complex_id"] and r["algorithm"]==algorithm}
            r,f=selected["reduced"],selected["full"]
            valid=r["valid_pose"] and f["valid_pose"]
            pairs.append({"complex_id":target["complex_id"],"algorithm":algorithm,"valid_pair":valid,
                "reduced_dockq":r["dockq"],"full_dockq":f["dockq"],
                "dockq_loss":f["dockq"]-r["dockq"] if valid else None,
                "preserved":valid and r["dockq"]>=.23 and f["dockq"]-r["dockq"]<=.05,
                "reduced_wall_sec":r["engine_call_wall_sec"],"full_wall_sec":f["engine_call_wall_sec"]})
    totals={}
    for mode in ("reduced","full"):
        selected=[r for r in rows if r["protocol"]==mode]
        successful=[r for r in selected if r["valid_pose"]]
        totals[mode]={"planned":len(selected),"successful":len(successful),"failed":len(selected)-len(successful),
            "requested_evaluations_per_job":sorted({r["requested_evaluations"] for r in selected}),
            "observed_evaluation_counts":sorted({r["actual_evaluations"] for r in selected if r["actual_evaluations"] is not None}),
            "all_engine_call_wall_sec":sum(r["engine_call_wall_sec"] for r in selected if r["engine_call_wall_sec"] is not None),
            "mean_dockq_successful":statistics.mean(r["dockq"] for r in successful) if successful else None,
            "dockq_ge_023_successes":sum(r["dockq"]>=.23 for r in successful)}
    valid=[p for p in pairs if p["valid_pair"]]
    reduced=sum(p["reduced_wall_sec"] for p in valid); full=sum(p["full_wall_sec"] for p in valid)
    report={"scope":"exploratory_5_complexes_1_seed_serial_not_independent_calibrator_validation",
        "timing_basis":"wall time of complete engine calls; serial execution; failures included in totals",
        "counts":status["counts"],"totals":totals,"paired_valid_count":len(valid),
        "preserved_pairs":sum(p["preserved"] for p in pairs),
        "paired_reduced_wall_sec":reduced,"paired_full_wall_sec":full,
        "paired_reduced_time_saving_fraction":1-reduced/full if full else None,
        "full_avoided":status["counts"].get("skipped_calibrator",0),
        "all_engine_cascade_sec":sum(t["all_engine_call_wall_sec"] for t in totals.values()),
        "early_stopping":"disabled_pending_independent_validation","rows":rows,"pairs":pairs}
    _atomic_write_json(study / "comparison.json",report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--selection-spec",default="output/calibration_audit_20261115/followup_5_spec.json")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=Path(args.output_dir).resolve();study.mkdir(parents=True,exist_ok=True)
    before_path=study / "protected_before.json"
    if not before_path.exists(): _atomic_write_json(before_path,protected_snapshot(root))
    spec=json.loads(Path(args.selection_spec).read_text(encoding="utf-8"))
    expected={r["complex_id"]:r for r in spec["targets"]}
    if len(expected)!=5: raise ValueError("pilot must use exactly five preselected complexes")
    report=discover_ppi4dock(args.input_dir,validate_contents=True)
    targets=tuple(t for t in report.targets if t.complex_id in expected)
    if len(targets)!=5: raise ValueError("preselected inputs are missing")
    for t in targets:
        for field in ("receptor_sha256","ligand_sha256","reference_sha256","complex_fingerprint_sha256"):
            if getattr(t,field)!=expected[t.complex_id][field]: raise ValueError(f"preselected input changed: {t.complex_id} {field}")
    report=replace(report,targets=targets,issues=())
    _atomic_write_json(study / "selection.json",{"selection":spec["selection"],"complex_ids":sorted(expected),
        "seed":1,"execution":"serial; REDUCED before FULL in every cascade pair","training":"not performed"})
    engine=MeasuredEngine(timeout_seconds=600)
    jobs=build_execution_plan(algorithms=[Algorithm.AG,Algorithm.PSO,Algorithm.HYBRID],
        protocol_mode=ProtocolMode.OPERATIONAL,base_config=load_operational_config(),seeds=[1])
    print("Starting 30 operational jobs on exactly five complexes",flush=True)
    result=run_batch(args.input_dir,study / "batch",jobs=jobs,engine=engine,report=report)
    comparison=summarize(study)
    before=json.loads(before_path.read_text(encoding="utf-8"));after=protected_snapshot(root)
    changed=[p for p,digest in before.items() if after.get(p)!=digest]
    _atomic_write_json(study / "preservation_check.json",{"protected_file_count":len(before),"changed":changed,"unchanged":not changed})
    if changed: raise RuntimeError("protected scientific files changed")
    print(json.dumps({"state":result.state,"counts":comparison["counts"],"totals":comparison["totals"]},indent=2),flush=True)


if __name__=="__main__": main()
