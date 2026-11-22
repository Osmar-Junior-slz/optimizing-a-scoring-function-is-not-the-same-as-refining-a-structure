"""Run the complete article workflow on exactly five supplied PPI4DOCK inputs.

Uses the installed frozen calibrator and existing engine; no training or tuning.
Per-engine-call timing includes failed calls, separately from reference scoring.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from docking_confidence.application.article_study import BLOCKS, prepare_article_study, run_article_study
from docking_confidence.application.batch import _atomic_write_json, _file_sha256
from docking_confidence.application.calibrated_batch import configured_calibrator_path
from docking_confidence.io.batch_discovery import discover_ppi4dock
from run_operational_pilot import MeasuredEngine, evaluation_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, out = Path(args.input_dir).resolve(), Path(args.output_dir).resolve()
    report = discover_ppi4dock(source, validate_contents=True)
    if report.has_errors or len(report.targets) != 5 or len(report.eligible_targets) != 5:
        raise ValueError("The authorized probe requires exactly five valid eligible inputs")
    engine = MeasuredEngine(timeout_seconds=600)
    pose_source = ("Modelos PDB fornecidos na pasta de teste; origem experimental e método de posicionamento "
        "não informados pelo usuário. Referência usada apenas na avaliação posterior. "
        "Piloto exploratório de refinamento local; sem declaração de teste cego ou confirmatório.")
    protocol = prepare_article_study(source, out, pose_source=pose_source, report=report, engine=engine)
    model = configured_calibrator_path()
    input_hashes = {str(path): _file_sha256(path) for target in report.targets
        for path in (target.receptor_path, target.ligand_path, target.reference_path)}
    model_hash = _file_sha256(model)
    _atomic_write_json(out / "pilot_commitment.json", {
        "scope": "five-input exploratory execution of the complete article workflow; no model fitting or cutoff selection",
        "protocol_sha256": protocol["protocol_sha256"], "script_sha256": _file_sha256(Path(__file__)),
        "timing_wrapper_sha256": _file_sha256(Path(__file__).with_name("run_operational_pilot.py")),
        "execution": "serial; one native thread; failed engine calls included in wall timing",
        "input_hashes": input_hashes, "artifact_sha256": model_hash,
        "distinct_pdb_entries": len({target.complex_id.split("_")[0] for target in report.targets}),
        "limitations": "same-PDB entries/homology are not proven independent; five cases cannot validate general safety; observed pilot cases must not be reused as fresh confirmation after method changes",
        "platform": platform.platform(), "python": sys.version})
    print("Executing complete five-input article probe: 60 factorial jobs + 30 confidence-validation jobs.", flush=True)
    started = perf_counter()
    result = run_article_study(out, engine=engine)
    elapsed = perf_counter() - started
    rows = []
    for block in BLOCKS:
        manifest = json.loads((out / block / "batch_manifest.json").read_text(encoding="utf-8"))
        status = json.loads((out / block / "batch_status.json").read_text(encoding="utf-8"))
        for target in manifest["plan"]["targets"]:
            for job in manifest["plan"]["jobs"]:
                directory = out / block / "targets" / target["complex_id"] / "jobs" / job["job_id"]
                task = status["tasks"][f"{target['complex_id']}::{job['job_id']}"]
                receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8")) if (directory / "receipt.json").exists() else {}
                engine_result = receipt.get("engine_result", {})
                timing = json.loads((directory / "engine/benchmark_timing.json").read_text(encoding="utf-8"))
                summaries = list((directory / "engine").rglob("output/summary.json"))
                raw = json.loads(summaries[0].read_text(encoding="utf-8")) if len(summaries) == 1 else {}
                decision = json.loads((directory / "confidence_decision.json").read_text(encoding="utf-8")) if (directory / "confidence_decision.json").exists() else {}
                rows.append({"block": block, "complex_id": target["complex_id"], "algorithm": job["algorithm"],
                    "protocol": job["protocol"], "initialization": job["profile"]["initialization"],
                    "requested_evaluations": job["profile"]["evaluation_budget"],
                    "actual_evaluations": evaluation_count(raw, engine_result), "state": task["state"],
                    "engine_call_wall_sec": timing["elapsed_sec"], "task_duration_sec": task["duration_seconds"],
                    "dockq": engine_result.get("dockq"), "fnat": engine_result.get("fnat"),
                    "irmsd": engine_result.get("irmsd"), "lrmsd": engine_result.get("lrmsd"),
                    "probability": decision.get("probability"), "confidence_overhead_sec": decision.get("confidence_overhead_sec"),
                    "error": task.get("last_error")})
    assert input_hashes == {path: _file_sha256(Path(path)) for path in input_hashes}
    assert model_hash == _file_sha256(model)
    joint = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    summary = {"scope": "five-case exploratory pilot; no model or threshold tuning", "study_wall_sec": elapsed,
        "input_and_model_unchanged": True, "actual_evaluation_mismatches": [row for row in rows if row["actual_evaluations"] != row["requested_evaluations"]],
        "counts": joint["counts"], "calibration": joint["calibration"], "timings": rows,
        "report": result, "distinct_pdb_entries": len({target.complex_id.split("_")[0] for target in report.targets})}
    _atomic_write_json(out / "pilot_performance.json", summary)
    import pandas as pd
    pd.DataFrame(rows).to_csv(out / "pilot_job_timings.csv", index=False)
    print(json.dumps({"state": "COMPLETED", "wall_sec": elapsed, "counts": joint["counts"],
        "report": result, "budget_mismatches": len(summary["actual_evaluation_mismatches"])}, indent=2), flush=True)


if __name__ == "__main__":
    main()
