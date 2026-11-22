"""Counterfactual audit procedure using the existing FULL-only batch executor.

Run from the project root with the project virtual environment. Original inputs,
decisions and scores are read only. Reports and FULL runs go into --output-dir.
Re-running the same command resumes verified completed FULL jobs.
"""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import run_batch, _job_snapshot, _target_snapshot
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, DockingTask, ProtocolMode
from docking_confidence.io.batch_discovery import discover_ppi4dock

from scientific_summary import ALGORITHMS, consolidate, dockq_value, preservation, read_cases, read_json, read_score


def case_key(case):
    return case["complex_id"], case["algorithm"].lower(), case["seed"]


def select_cases(cases):
    if len({case_key(c) for c in cases}) != len(cases):
        raise ValueError("Duplicate original case")
    return [c for c in cases if c["action"] == "ACCEPT_REDUCED"
            and dockq_value(c["dockq_reduced"]) is not None
            and c["dockq_reduced"] >= .23 and c["dockq_full"] is None]


def full_job(case, original_jobs):
    _, algorithm, seed = case_key(case)
    matches = [j for j in original_jobs if
               (j["algorithm"], j["seed"], j["protocol"]) == (algorithm, seed, "full")]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original FULL job")
    original = matches[0]
    p = original["profile"]
    jobs = build_execution_plan(
        algorithms=[Algorithm(algorithm)], seeds=[seed], protocol_mode=ProtocolMode.FULL,
        base_config=BaseSearchConfig(generations=p["generations"], population=p["population"],
                                     top_pockets=p["top_pockets"], docking_task=DockingTask(p["docking_task"])))
    if len(jobs) != 1 or _job_snapshot(jobs[0]) != original:
        raise ValueError("FULL parameters differ from the original plan")
    return jobs[0]


def separate_output(output, original_batch, original_inputs, original_summary):
    output = Path(output).resolve()
    for source in (original_batch, original_inputs):
        source = Path(source).resolve()
        if output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError("Audit output must be disjoint from original inputs/results")
    if Path(original_summary).resolve().is_relative_to(output):
        raise ValueError("Audit output would contain the original summary")
    return output


def execute_case(case, job, report, original_target, output, engine, runner=run_batch):
    targets = tuple(t for t in report.targets if t.complex_id == case["complex_id"])
    if len(targets) != 1 or _target_snapshot(targets[0], report.input_root) != original_target:
        raise ValueError("Complex/input hashes differ from the original target")
    if (job.algorithm.value, job.seed, job.protocol.value) != (*case_key(case)[1:], "full"):
        raise ValueError("Only the original algorithm/seed FULL may execute")
    single = replace(report, targets=targets,
                     issues=tuple(i for i in report.issues if i.complex_id in (None, case["complex_id"])))
    destination = Path(output) / case["complex_id"] / job.job_id
    if not destination.resolve().is_relative_to(Path(output).resolve()):
        raise ValueError("Case destination escaped audit output")
    runner(report.input_root, destination, jobs=[job], engine=engine, report=single,
           eligible_only=True, confidence_artifact=None)
    return destination


def merge_audit(cases, scores):
    """Below .23 already disproves the conjunction; missing FULL stays null."""
    merged = deepcopy(cases)
    selected = {case_key(c) for c in select_cases(cases)}
    if set(scores) != selected:
        raise ValueError("Audit FULL scores must match exactly the selected cases")
    for c in merged:
        key = case_key(c)
        if key in scores:
            score, source = scores[key]
            if dockq_value(score) is None:
                raise ValueError("Audit FULL score is missing; cannot finalize PPV")
            c.update(preservation(c["dockq_reduced"], score))
            c["full_source"] = source
        if c["action"] == "ACCEPT_REDUCED" and c["dockq_reduced"] is not None and c["dockq_reduced"] < .23:
            c["Y"] = 0
            c["preservation_status"] = "Y=0: DockQ REDUCED < 0.23; FULL não necessário para refutar a conjunção."
    return merged


def acceptance_metrics(cases):
    accepted = [c for c in cases if c["action"] == "ACCEPT_REDUCED"]
    if any(c["Y"] not in (0, 1) for c in accepted):
        raise ValueError("Unresolved accepted cases; cannot finalize PPV")
    n = len(accepted)
    yes = sum(c["Y"] == 1 for c in accepted)
    no = n - yes
    return {"accepted": n, "Y_1": yes, "Y_0": no,
            "PPV": yes / n if n else None, "false_accept_risk": no / n if n else None,
            "original_coverage_percent": 100 * n / len(cases) if cases else None}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("scientific_summary.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    original = read_json(args.summary)
    batch = Path(original["source_batch"])
    manifest = read_json(batch / "batch_manifest.json")
    plan = manifest["plan"]
    inputs = Path(plan["input_root"])
    output = separate_output(args.output_dir, batch, inputs, args.summary)
    if original["cases"] != read_cases(batch):
        raise ValueError("Original summary differs from persisted decisions/results")
    selected = select_cases(original["cases"])
    if len(selected) != 159 or sum(c["action"] == "ACCEPT_REDUCED" for c in original["cases"]) != 169:
        raise ValueError("Expected 159 FULL audits and 169 original acceptances")
    engine = DockingPPLocalEngine()
    if engine.audit_descriptor() != plan["engine"]["descriptor"]:
        raise ValueError("Engine/config/runtime differs from original experiment")
    report = discover_ppi4dock(inputs, validate_contents=True)
    targets = {t["complex_id"]: t for t in plan["targets"]}
    jobs = [full_job(c, plan["jobs"]) for c in selected]
    # Hash original scientific evidence before/after. No original file is opened for writing.
    protected = {args.summary.resolve(), batch / "batch_manifest.json", batch / "batch_status.json"}
    for t in plan["targets"]:
        protected.update(inputs / t[f"{role}_path"] for role in ("receptor", "ligand", "reference"))
    for pattern in ("*/jobs/*/confidence_decision.json", "*/jobs/*/receipt.json",
                    "*/jobs/*/postprocess/reference_dockq.json"):
        protected.update((batch / "targets").glob(pattern))
    hashes = {str(p): sha256(p) for p in sorted(protected)}
    procedure = {"original_batch": str(batch), "original_summary": str(args.summary.resolve()),
                 "original_sha256": hashes,
                 "cases": [{"complex_id": c["complex_id"], "job": _job_snapshot(j)}
                           for c, j in zip(selected, jobs)]}
    receipt = output / "audit_procedure.json"
    if output.exists() and any(output.iterdir()):
        if not receipt.is_file() or read_json(receipt) != procedure:
            raise ValueError("Output is not an identical resumable audit")
    output.mkdir(parents=True, exist_ok=True)
    write_json(receipt, procedure)
    print(f"Selected {len(selected)} FULL-only cases; output={output}", flush=True)
    scores = {}
    for index, (case, job) in enumerate(zip(selected, jobs), 1):
        print(f"[{index}/{len(selected)}] {case['complex_id']} {job.job_id}", flush=True)
        destination = execute_case(case, job, report, targets[case["complex_id"]], output, engine)
        status = read_json(destination / "batch_status.json")
        task = status["tasks"][f"{case['complex_id']}::{job.job_id}"]
        score = read_score(destination, task)
        if score is None:
            raise RuntimeError(f"FULL failed or DockQ missing: {destination}. Resume the same command after diagnosis.")
        scores[case_key(case)] = (score, str(destination.resolve()))
        print(f"DockQ FULL={score:.9f}", flush=True)
    if any(sha256(p) != digest for p, digest in hashes.items()):
        raise ValueError("Original evidence changed during audit")
    cases = merge_audit(original["cases"], scores)
    result = {"source_batch": str(batch), "audit_directory": str(output),
              "original_evidence_unchanged": True, **consolidate(cases),
              "counterfactual_acceptance": acceptance_metrics(cases),
              "counterfactual_by_algorithm": {a: acceptance_metrics([c for c in cases if c["algorithm"] == a])
                                              for a in ALGORITHMS.values()}}
    write_json(output / "scientific_summary.json", result)
    lines = ["# Auditoria contrafactual dos aceitos", "",
             f"REDUCED e decisões originais: `{batch}`. FULL de auditoria: `{output}`.", "",
             "159 FULL executados com complexo, algoritmo, seed e parâmetros originais. "
             "Os 10 aceitos com DockQ REDUCED < 0.23 têm Y=0 sem precisar de FULL. "
             "Valores FULL ausentes permanecem nulos. Evidências originais verificadas por SHA-256.", "",
             "Y=1 se DockQ REDUCED ≥ 0.23 e FULL − REDUCED ≤ 0.05; caso contrário Y=0. "
             "PPV=Y=1/aceitos; risco de falso aceite=Y=0/aceitos. Cobertura original: 56,3% (169/300).", "",
             "| Grupo | Aceitos | Y=1 | Y=0 | PPV | Risco de falso aceite | Cobertura original |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for label, m in [("Total", result["counterfactual_acceptance"]), *result["counterfactual_by_algorithm"].items()]:
        lines.append(f"| {label} | {m['accepted']} | {m['Y_1']} | {m['Y_0']} | {m['PPV']:.6%} | "
                     f"{m['false_accept_risk']:.6%} | {m['original_coverage_percent']:.6f}% |")
    lines += ["", "| Complexo | Algoritmo | Seed | Decisão original | REDUCED original | FULL auditoria | FULL − REDUCED | Y |",
              "|---|---|---:|---|---:|---:|---:|---:|"]
    for c in cases:
        if c["action"] == "ACCEPT_REDUCED":
            lines.append("| " + " | ".join("Ausente" if c[k] is None else str(c[k]) for k in
                         ("complex_id", "algorithm", "seed", "action", "dockq_reduced", "dockq_full",
                          "dockq_full_minus_reduced", "Y")) + " |")
    (output / "scientific_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["counterfactual_acceptance"], indent=2), flush=True)


if __name__ == "__main__":
    main()
