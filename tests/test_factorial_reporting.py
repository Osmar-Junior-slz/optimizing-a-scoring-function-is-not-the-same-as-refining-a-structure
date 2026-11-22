"""Synthetic persisted evidence only: no docking or optimizer is executed."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from docking_confidence.application.factorial_reporting import (
    CONDITIONS, FactorialAnalysisError, analyze_factorial_batch, factorial_markdown,
    write_factorial_report,
)


def _hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _signed(value, field):
    return {**value, field: _hash(value)}


def _batch(tmp_path, *, complexes=("c1", "c2"), seeds=(1, 2), failed=(), missing_dockq=(), values=None):
    root = tmp_path / "batch"
    jobs = []
    for seed in seeds:
        for condition in CONDITIONS:
            extent, init = condition.split("_")
            protocol = "reduced" if extent == "reduced" else "full"
            jobs.append({"job_id": f"ag-{extent}-{init}-seed{seed}", "algorithm": "ag", "seed": seed,
                         "protocol": protocol, "profile": {"search_extent": extent, "initialization": init,
                                                              "evaluation_budget": 12, "generations": 2,
                                                              "population": 4, "top_pockets": 1,
                                                              "docking_task": "local_refinement",
                                                              "budget_policy": "common_ceiling"}})
    plan = {"jobs": jobs, "targets": [{"complex_id": complex_id} for complex_id in complexes]}
    plan_hash = _hash(plan)
    _write(root / "batch_manifest.json", _signed({"plan": plan, "plan_sha256": plan_hash,
                                                  "batch_id": plan_hash}, "manifest_sha256"))
    tasks = {}
    for target in plan["targets"]:
        for job in jobs:
            cid = target["complex_id"]
            condition = f"{job['profile']['search_extent']}_{job['profile']['initialization']}"
            key = (cid, job["seed"], condition)
            identity = {field: job[field] for field in ("job_id", "algorithm", "protocol", "seed")}
            tasks[f"{cid}::{job['job_id']}"] = {"complex_id": cid, **identity,
                                              "state": "FAILED" if key in failed else "SUCCEEDED",
                                              "duration_seconds": 3.0, "last_error": {"message": "fixture"} if key in failed else None}
            if key in failed:
                continue
            dockq = (values[key] if values else {"reduced_uniform": 0.4, "reduced_seeded": 0.7,
                                                 "expanded_uniform": 0.3, "expanded_seeded": 0.5}[condition])
            result = {**identity, "dockq": None if key in missing_dockq else dockq,
                      "runtime_sec": 2.0, "actual_eval_count": 12}
            receipt = {"complex_id": cid, "plan_sha256": plan_hash, "job": job, "target": target,
                       "task_sha256": _hash({"plan_sha256": plan_hash, "target": target, "job": job}),
                       "status": "SUCCEEDED", "engine_result": result}
            _write(root / "targets" / cid / "jobs" / job["job_id"] / "receipt.json", _signed(receipt, "record_sha256"))
    _write(root / "batch_status.json", {"plan_sha256": plan_hash, "tasks": tasks})
    return root


def _change_manifest(root, mutate):
    path = root / "batch_manifest.json"
    value = json.loads(path.read_text())
    mutate(value["plan"])
    value["plan_sha256"] = value["batch_id"] = _hash(value["plan"])
    value.pop("manifest_sha256")
    _write(path, _signed(value, "manifest_sha256"))


def _change_receipt(root, complex_id, job_id, mutate, *, resign=True):
    path = root / "targets" / complex_id / "jobs" / job_id / "receipt.json"
    value = json.loads(path.read_text())
    mutate(value)
    if resign:
        value.pop("record_sha256")
        value = _signed(value, "record_sha256")
    _write(path, value)


def _rebind_evidence(root):
    manifest = json.loads((root / "batch_manifest.json").read_text())
    plan, plan_hash = manifest["plan"], manifest["plan_sha256"]
    status = json.loads((root / "batch_status.json").read_text())
    status["plan_sha256"] = plan_hash
    _write(root / "batch_status.json", status)
    for target in plan["targets"]:
        for job in plan["jobs"]:
            path = root / "targets" / target["complex_id"] / "jobs" / job["job_id"] / "receipt.json"
            if not path.is_file():
                continue
            value = json.loads(path.read_text())
            value.pop("record_sha256")
            value.update(target=target, job=job, plan_sha256=plan_hash,
                         task_sha256=_hash({"plan_sha256": plan_hash, "target": target, "job": job}))
            _write(path, _signed(value, "record_sha256"))


def _freeze(root):
    manifest = json.loads((root / "batch_manifest.json").read_text())
    base = manifest["plan"]
    frozen = _signed({"schema_version": "factorial_protocol_v1", "plan": base,
                      "base_plan_sha256": _hash(base), "analysis": {"bootstrap_repetitions": 2000,
                      "bootstrap_seed": 2026, "confidence_level": 0.95, "analysis_role": "confirmatory"}},
                     "protocol_sha256")
    path = root / "experimental_protocol.json"
    _write(path, frozen)
    _change_manifest(root, lambda p: p.update(experimental_protocol={"path": str(path.resolve()),
                     "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "protocol_sha256": frozen["protocol_sha256"]}))
    _rebind_evidence(root)
    return path


def test_known_effects_match_factorial_contrasts_and_bootstrap_is_deterministic(tmp_path):
    root = _batch(tmp_path)
    report = analyze_factorial_batch(root, bootstrap_repetitions=200, bootstrap_seed=5)
    assert report == analyze_factorial_batch(root, bootstrap_repetitions=200, bootstrap_seed=5)
    metrics = report["by_algorithm"]["ag"]["metrics"]["dockq"]
    assert metrics["complete_quartets"] == metrics["planned_quartets"] == 4
    expected = {"extent_at_uniform": 0.1, "extent_at_seeded": 0.2,
                "initialization_at_reduced": 0.3, "initialization_at_expanded": 0.2,
                "interaction": 0.1, "extent_main": 0.15, "initialization_main": 0.25}
    for name, value in expected.items():
        estimate = metrics["effects"][name]
        assert estimate["mean"] == pytest.approx(value)
        assert estimate["ci95"] == pytest.approx([value, value])
        assert estimate["n_complexes"] == 2
    assert report["method"]["bootstrap_unit"] == "complex"
    assert report["counts"]["states"] == {"SUCCEEDED": 16}
    assert report["interpretation"] == "exploratory_unfrozen"  # No frozen protocol binding.


def test_failed_and_missing_scores_keep_denominators_and_never_become_zero(tmp_path):
    root = _batch(tmp_path, failed={("c1", 1, "reduced_uniform")},
                  missing_dockq={("c2", 2, "expanded_seeded")})
    report = analyze_factorial_batch(root, bootstrap_repetitions=30)
    data = report["by_algorithm"]["ag"]
    assert data["metrics"]["dockq"]["complete_quartets"] == 2
    assert data["metrics"]["runtime_sec"]["complete_quartets"] == 3
    assert data["metrics"]["success"]["complete_quartets"] == 4
    assert data["metrics"]["task_duration_sec"]["complete_quartets"] == 4
    exclusions = data["metrics"]["dockq"]["excluded_quartets"]
    assert {(row["complex_id"], row["seed"]) for row in exclusions} == {("c1", 1), ("c2", 2)}
    failed = next(o for o in report["observations"] if o["state"] == "FAILED")
    assert failed["metrics"]["dockq"] is None
    assert failed["metrics"]["actual_eval_count"] is None
    assert failed["metrics"]["success"] == 0
    assert failed["last_error"]["message"] == "fixture"
    assert report["interpretation"] == "exploratory_partial"


def test_fixed_primary_counts_failure_and_low_dockq_and_reports_pending(tmp_path):
    root = _batch(tmp_path, complexes=("c1", "c2", "c3", "c4"), seeds=(1,),
                  failed={("c2", 1, "reduced_uniform")})
    _change_manifest(root, lambda p: [j["profile"].update(budget_policy="fixed_evaluations") for j in p["jobs"]])
    _rebind_evidence(root)
    for path in root.rglob("receipt.json"):
        (path.parent / "predicted.pdb").write_text("END\n")
        _change_receipt(root, path.parents[2].name, path.parent.name,
                        lambda r: r["engine_result"].update(
                            complex_pose_pdb_path="predicted.pdb", stop_reason="objective_evaluation_ceiling"))
    job_id = "ag-reduced-uniform-seed1"
    _change_receipt(root, "c1", job_id, lambda r: r["engine_result"].update(dockq=0.23))
    _change_receipt(root, "c3", job_id, lambda r: r["engine_result"].update(dockq=0.22))
    (root / "targets" / "c4" / "jobs" / job_id / "receipt.json").unlink()
    status_path = root / "batch_status.json"
    status = json.loads(status_path.read_text())
    status["tasks"][f"c4::{job_id}"]["state"] = "PENDING"
    _write(status_path, status)

    report = analyze_factorial_batch(root, bootstrap_repetitions=20)
    primary = report["primary_outcome"]
    assert primary == "valid_pose_dockq_ge_023"
    condition = report["by_algorithm"]["ag"]["conditions"]["reduced_uniform"]["primary_outcome"]
    assert condition == {"successes": 1, "failures_or_below_threshold": 2, "unresolved": 1,
                         "denominator": 4, "proportion_of_planned": 0.25}
    metrics = report["by_algorithm"]["ag"]["metrics"]
    assert metrics[primary]["complete_quartets"] == 3
    assert metrics["dockq"]["complete_quartets"] == 2
    assert report["interpretation"] == "exploratory_partial"
    assert "Desfecho principal: pose válida com DockQ ≥ 0,23" in factorial_markdown(report)


def test_complexes_receive_equal_weight_with_unequal_available_seed_counts(tmp_path):
    values = {}
    for cid in ("c1", "c2"):
        for seed in (1, 2):
            for condition in CONDITIONS:
                values[(cid, seed, condition)] = 0.9 if cid == "c1" and condition.startswith("reduced") else 0.1
    root = _batch(tmp_path, values=values, failed={("c2", 2, "expanded_uniform")})
    report = analyze_factorial_batch(root, bootstrap_repetitions=200)
    effect = report["by_algorithm"]["ag"]["metrics"]["dockq"]["effects"]["extent_main"]
    assert effect["complex_means"] == pytest.approx({"c1": 0.8, "c2": 0.0})
    assert effect["mean"] == pytest.approx(0.4)  # A pooled-seed mean would be 0.5333.
    assert effect["ci95"] == pytest.approx([0.0, 0.8])


def test_one_complex_has_point_estimates_but_no_inferential_interval(tmp_path):
    report = analyze_factorial_batch(_batch(tmp_path, complexes=("c1",)), bootstrap_repetitions=20)
    effect = report["by_algorithm"]["ag"]["metrics"]["dockq"]["effects"]["interaction"]
    assert effect["mean"] == pytest.approx(0.1)
    assert effect["ci95"] is None and effect["n_complexes"] == 1


@pytest.mark.parametrize("change,match", [
    (lambda p: p["jobs"].append(p["jobs"][0]), "Duplicate"),
    (lambda p: p["jobs"].pop(), "four conditions"),
    (lambda p: p["jobs"][0]["profile"].update(evaluation_budget=999), "evaluation_budget"),
    (lambda p: p["jobs"][0]["profile"].pop("initialization"), "explicit factorial"),
])
def test_invalid_or_confounded_plan_is_rejected_before_statistics(tmp_path, change, match):
    root = _batch(tmp_path)
    _change_manifest(root, change)
    with pytest.raises(FactorialAnalysisError, match=match):
        analyze_factorial_batch(root)


def test_tampered_receipt_and_signed_wrong_seed_are_rejected(tmp_path):
    root = _batch(tmp_path)
    job_id = "ag-reduced-uniform-seed1"
    _change_receipt(root, "c1", job_id, lambda r: r["engine_result"].update(dockq=0.99), resign=False)
    with pytest.raises(FactorialAnalysisError, match="integrity hash"):
        analyze_factorial_batch(root)
    _change_receipt(root, "c1", job_id, lambda r: r["engine_result"].update(seed=2))
    with pytest.raises(FactorialAnalysisError, match="planned seed"):
        analyze_factorial_batch(root)


def test_csv_alone_cannot_supply_success_and_disagreement_is_visible(tmp_path):
    root = _batch(tmp_path, failed={("c1", 1, "reduced_uniform")})
    with (root / "batch_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["complex_id", "job_id", "dockq"])
        writer.writeheader()
        writer.writerow({"complex_id": "c1", "job_id": "ag-reduced-uniform-seed1", "dockq": 0.99})
    report = analyze_factorial_batch(root, bootstrap_repetitions=20)
    reasons = {row["reason"] for row in report["csv_audit"]["discrepancies"]}
    assert reasons == {"csv_row_without_verified_receipt", "csv_dockq_differs_from_receipt"}
    assert report["by_algorithm"]["ag"]["metrics"]["dockq"]["complete_quartets"] == 3


def test_missing_receipt_is_reported_even_when_status_claims_success(tmp_path):
    root = _batch(tmp_path)
    (root / "targets" / "c1" / "jobs" / "ag-reduced-uniform-seed1" / "receipt.json").unlink()
    report = analyze_factorial_batch(root, bootstrap_repetitions=20)
    assert report["diagnostics"][0]["reasons"] == ["succeeded_without_receipt"]
    assert report["by_algorithm"]["ag"]["metrics"]["dockq"]["complete_quartets"] == 3
    assert report["by_algorithm"]["ag"]["metrics"]["success"]["complete_quartets"] == 3


def test_complete_report_exports_tables_and_static_plots(tmp_path):
    report = analyze_factorial_batch(_batch(tmp_path), bootstrap_repetitions=20)
    paths = write_factorial_report(report, tmp_path / "analysis")
    assert len(paths) == 8
    assert all(Path(path).is_file() for path in paths.values())
    assert json.loads(Path(paths["json"]).read_text(encoding="utf-8")) == report
    assert "Quartetos incompletos" in factorial_markdown(report)
    with Path(paths["observations"]).open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 16


def test_manual_script_only_analyzes_existing_artifacts(tmp_path):
    root = _batch(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_factorial.py"
    completed = subprocess.run([sys.executable, str(script), "--batch", str(root), "--no-plots",
                                "--bootstrap-repetitions", "20"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["counts"]["planned_runs"] == 16
    assert Path(result["files"]["markdown"]).is_file()


def test_frozen_protocol_identity_and_analysis_choices_are_checked(tmp_path):
    root = _batch(tmp_path)
    protocol_path = _freeze(root)
    report = analyze_factorial_batch(root)
    assert report["interpretation"] == "complete_frozen_protocol"
    assert report["analysis_role"] == "confirmatory"
    assert report["analysis_choices_match_protocol"] is True
    override = analyze_factorial_batch(root, bootstrap_repetitions=20)
    assert override["interpretation"] == "exploratory_analysis_deviation"
    protocol_path.write_text(protocol_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(FactorialAnalysisError, match="protocol failed its integrity"):
        analyze_factorial_batch(root)


def test_molecular_aliases_are_one_bootstrap_cluster(tmp_path):
    root = _batch(tmp_path)
    _change_manifest(root, lambda p: [t.update(complex_fingerprint_sha256="a" * 64) for t in p["targets"]])
    _rebind_evidence(root)
    report = analyze_factorial_batch(root, bootstrap_repetitions=20)
    assert report["counts"]["complexes"] == 2
    assert report["counts"]["independent_clusters"] == 1
    assert report["cluster_aliases"] == {"a" * 64: ["c1", "c2"]}
    effect = report["by_algorithm"]["ag"]["metrics"]["dockq"]["effects"]["extent_main"]
    assert effect["n_complexes"] == 1 and effect["ci95"] is None


def test_budget_violation_is_not_presented_as_a_complete_frozen_experiment(tmp_path):
    root = _batch(tmp_path)
    _freeze(root)
    _change_receipt(root, "c1", "ag-reduced-uniform-seed1", lambda r: r["engine_result"].update(actual_eval_count=13))
    report = analyze_factorial_batch(root)
    assert report["interpretation"] == "exploratory_protocol_deviation"
    assert report["diagnostics"][0]["reasons"] == ["actual_evaluation_budget_exceeded"]


def test_reporting_accepts_actual_prepared_batch_schema_without_any_optimizer(tmp_path):
    from docking_confidence.application.batch import run_batch
    from docking_confidence.application.experimental_protocol import prepare_experimental_protocol
    from docking_confidence.application.planning import build_execution_plan
    from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode
    from test_batch_runner import _RecordingEngine, _make_report

    source = tmp_path / "input"
    source.mkdir()
    discovery = _make_report(source, 2)
    engine = _RecordingEngine()
    jobs = build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(generations=2, population=4, top_pockets=1,
            experiment=FactorialExperimentConfig(evaluation_budget=12, pose_source="Synthetic input fixture")), seeds=[1])
    output = tmp_path / "output"
    prepared = prepare_experimental_protocol(source, output, jobs=jobs, engine=engine, report=discovery)
    assert engine.calls == []
    summary = run_batch(source, output, jobs=jobs, engine=engine, report=discovery, experimental_protocol=prepared["path"])
    assert summary.succeeded == 8
    analysis = analyze_factorial_batch(output)
    assert analysis["interpretation"] == "non_scientific_demo"
    assert analysis["analysis_choices_match_protocol"] is True
    assert analysis["counts"]["planned_quartets"] == 2
    assert analysis["by_algorithm"]["ag"]["metrics"]["dockq"]["complete_quartets"] == 2
    assert analysis["by_algorithm"]["ag"]["metrics"]["actual_eval_count"]["complete_quartets"] == 2
    assert analysis["primary_outcome"] == "valid_pose_dockq_ge_023"
