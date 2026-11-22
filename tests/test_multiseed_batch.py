"""Regression coverage of the existing paired workflow; never run real docking.

Large plans are constructed only. Small persistence/dataset checks reuse the
suite's recording and fixture-pose engines, not AG, PSO or Hybrid execution.
"""
from __future__ import annotations

from itertools import product
from dataclasses import replace
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from docking_confidence.application.batch import (
    BatchRunSummary, _build_plan, _initial_status, _manifest_for_plan,
    clear_batch_cancel_request, run_batch,
)
from docking_confidence.application.batch_launcher import real_batch_command
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ProtocolMode
from test_batch_runner import _RecordingEngine, _make_report
from test_training_dataset import make_batch, _build, _receipt, _rewrite_receipt


class _ArtifactRecordingEngine(_RecordingEngine):
    supports_artifact_work_dir = True

    def run(self, *, job, protein_path, peptide_path, work_dir):
        result = super().run(job=job, protein_path=protein_path, peptide_path=peptide_path)
        (work_dir / "summary.json").write_text(json.dumps({
            "fixture": True, "algorithm": job.algorithm.value,
            "protocol": job.protocol.value, "seed": job.seed,
            "total_n_eval": job.seed + 10,
        }), encoding="utf-8")
        return replace(result, engine_summary_path="summary.json")


def _command(tmp_path: Path):
    return real_batch_command(
        input_dir=tmp_path / "input", output_dir=tmp_path / "output",
        algorithms=["ag", "pso", "hybrid"], protocol="reduced_vs_full",
        generations=100, population=100, top_pockets=32,
        seed_start=1, seed_count=5, timeout_seconds=86400,
    )


def _jobs(seeds):
    return build_execution_plan(
        algorithms=list(Algorithm), protocol_mode=ProtocolMode.REDUCED_VS_FULL,
        base_config=BaseSearchConfig(), seeds=seeds,
    )


def test_initial_seed_and_count_expand_to_five_consecutive_cli_seeds(tmp_path):
    command = _command(tmp_path)
    seeds = command[command.index("--seed") + 1:command.index("--generations")]
    assert seeds == ("1", "2", "3", "4", "5")
    assert "--confidence-artifact" not in command
    assert "--seed-count" not in command  # Real CLI retains its list syntax.
    jobs = _jobs(map(int, seeds))
    assert len(jobs) == len({j.job_id for j in jobs}) == 30
    assert {(j.algorithm.value, j.seed, j.protocol.value) for j in jobs} == set(
        product(("ag", "pso", "hybrid"), range(1, 6), ("reduced", "full"))
    )


@pytest.mark.parametrize(("complex_count", "expected_jobs"), [(1, 30), (100, 3000)])
def test_full_batch_plan_counts_all_complex_algorithm_seed_protocol_keys(
    tmp_path, complex_count, expected_jobs,
):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, count=complex_count)
    engine = _RecordingEngine()
    jobs = _jobs(range(1, 6))
    plan, _ = _build_plan(
        report=report, input_root=source.resolve(), jobs=tuple(jobs),
        engine=engine, eligible_only=True, skip_invalid=False,
    )
    status = _initial_status(manifest=_manifest_for_plan(plan), previous=None)
    tasks = status["tasks"]
    assert len(tasks) == expected_jobs
    assert {(t["complex_id"], t["algorithm"], t["seed"], t["protocol"])
            for t in tasks.values()} == set(product(
                (t.complex_id for t in report.targets), ("ag", "pso", "hybrid"),
                range(1, 6), ("reduced", "full"),
            ))
    assert all(t["state"] == "PENDING" for t in tasks.values())
    assert engine.calls == []  # In particular, no 3,000-job experiment.


def test_real_cli_preserves_all_five_seeds_and_both_protocols(tmp_path, monkeypatch, capsys):
    from docking_confidence.cli import main

    engine = object()
    calls = []
    monkeypatch.setattr(
        "docking_confidence.adapters.dockingpp_local_engine.DockingPPLocalEngine",
        lambda **kwargs: engine,
    )

    def record_batch(*args, **kwargs):
        calls.append(kwargs)
        return BatchRunSummary(
            batch_id="fixture", state="COMPLETED", planned_jobs=30,
            succeeded=30, failed=0, pending=0, attempted=30, skipped_existing=0,
            manifest_path=tmp_path / "batch_manifest.json",
            status_path=tmp_path / "batch_status.json",
        )

    monkeypatch.setattr("docking_confidence.application.batch.run_batch", record_batch)
    monkeypatch.setattr(sys, "argv", ["docking-confidence", *_command(tmp_path)[3:]])
    main()
    assert len(calls) == 1
    assert calls[0]["engine"] is engine
    assert calls[0]["confidence_artifact"] is None
    assert {(j.algorithm, j.seed, j.protocol) for j in calls[0]["jobs"]} == set(
        product(Algorithm, range(1, 6), (ProtocolMode.REDUCED, ProtocolMode.FULL))
    )
    assert json.loads(capsys.readouterr().out)["planned_jobs"] == 30


def test_twelve_job_smoke_resume_keeps_completed_seed_and_runs_other_seeds(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.mkdir()
    report = _make_report(source, count=1)
    jobs = _jobs([1, 2])
    engine = _ArtifactRecordingEngine()
    engine.cancel_output = output
    first = run_batch(source, output, jobs=jobs, engine=engine, report=report)
    assert first.planned_jobs == 12
    assert first.attempted == first.succeeded == 1
    assert first.state == "CANCELLED"
    completed_job = engine.calls[0][0]
    assert completed_job.seed == 1
    first_receipt = output / "targets" / report.targets[0].complex_id / "jobs" / completed_job.job_id / "receipt.json"
    first_bytes = first_receipt.read_bytes()
    first_summary = first_receipt.parent / "engine" / "summary.json"
    first_summary_bytes = first_summary.read_bytes()

    clear_batch_cancel_request(output)
    engine.cancel_output = None
    resumed = run_batch(source, output, jobs=jobs, engine=engine, report=report)
    assert resumed.state == "COMPLETED"
    assert resumed.attempted == 11 and resumed.skipped_existing == 1
    assert resumed.succeeded == 12
    assert first_receipt.read_bytes() == first_bytes
    assert first_summary.read_bytes() == first_summary_bytes
    assert len(engine.calls) == len({j.job_id for j, _, _ in engine.calls}) == 12
    receipts = sorted(output.glob("targets/*/jobs/*/receipt.json"))
    assert len(receipts) == 12
    observed = set()
    for path in receipts:
        record = json.loads(path.read_text(encoding="utf-8"))
        job, result = record["job"], record["engine_result"]
        assert record["status"] == "SUCCEEDED"
        assert record["complex_id"] == report.targets[0].complex_id
        for key in ("job_id", "algorithm", "protocol", "seed"):
            assert job[key] == result[key]
        assert path.parent.name == job["job_id"]
        assert f"seed{job['seed']}" in path.parent.name
        assert result["runtime_sec"] == 1.0
        assert result["evaluation_budget"] == job["profile"]["evaluation_budget"]
        assert (output / record["engine_work_dir"]).parent == path.parent
        summary_path = path.parent / result["engine_summary_path"]
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert summary["seed"] == job["seed"]
        assert summary["total_n_eval"] == job["seed"] + 10
        observed.add((job["algorithm"], job["seed"], job["protocol"]))
    assert observed == set(product(("ag", "pso", "hybrid"), (1, 2), ("reduced", "full")))
    artifacts = receipts + sorted(output.glob("targets/*/jobs/*/engine/summary.json"))
    before = {p: p.read_bytes() for p in artifacts}
    again = run_batch(source, output, jobs=jobs, engine=engine, report=report)
    assert again.attempted == 0 and again.skipped_existing == 12
    assert len(engine.calls) == 12
    assert {p: p.read_bytes() for p in artifacts} == before


def test_dataset_pairs_distinct_seed_runtimes_without_collapsing_repetitions(make_batch):
    batch = make_batch(algorithms=tuple(Algorithm), seeds=(1, 2))
    expected = {}
    for algorithm_index, algorithm in enumerate(Algorithm):
        for seed in (1, 2):
            timings = (100 * seed + 10 * algorithm_index + 1,
                       100 * seed + 10 * algorithm_index + 2)
            expected[(algorithm.value, seed)] = timings
            for protocol, runtime in zip(("reduced", "full"), timings):
                _rewrite_receipt(
                    _receipt(batch, algorithm=algorithm.value, seed=seed, protocol=protocol),
                    lambda r, runtime=runtime: r["engine_result"].update(runtime_sec=runtime),
                )
    result = _build(batch)
    assert result["valid"] is True, result["issues"]
    frame = pd.read_csv(result["dataset_path"])
    assert len(frame) == 6
    assert not frame.duplicated(["complex_id", "algorithm", "seed"]).any()
    assert result["counts"]["independent_complexes"] == 1
    for row in frame.itertuples():
        assert (row.runtime_reduced_sec, row.runtime_full_sec) == expected[(row.algorithm, row.seed)]
        assert row.complex_id == "1abc_AB"
        assert row.run_id == f"{row.algorithm}-reduced-seed{row.seed}"


def test_existing_ui_passes_seed_count_without_launching_real_worker(tmp_path, monkeypatch):
    from docking_confidence.ui import batch_page
    from docking_confidence.io.batch_discovery import discover_ppi4dock
    from test_batch_discovery import _write_target

    source = tmp_path / "input"
    _write_target(source)
    calls = []

    def record_launch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pid=123, log_path=tmp_path / "worker.log")

    monkeypatch.setattr(batch_page, "launch_real_batch", record_launch)
    app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_input_dir"] = str(source)
    app.session_state["ppi4dock_output_dir"] = str(tmp_path / "output")
    app.session_state["ppi4dock_validated_report"] = discover_ppi4dock(source)
    app.run(timeout=20)
    app.selectbox(key="ppi4dock_real_protocol").set_value("operational_cascade").run(timeout=20)
    app.multiselect(key="ppi4dock_real_algorithms").set_value(["ag", "pso", "hybrid"])
    app.number_input(key="ppi4dock_real_seed_start").set_value(1)
    app.number_input(key="ppi4dock_real_seed_count").set_value(5)
    assert calls == []
    next(b for b in app.button if b.label == "Executar pares do artigo").click().run(timeout=20)
    assert not app.exception
    assert len(calls) == 1
    call = calls[0]
    assert (call["seed_start"], call["seed_count"]) == (1, 5)
    assert call["protocol"] == "operational_cascade"
    assert call["algorithms"] == ["ag", "pso", "hybrid"]
    assert call["confidence_artifact"] is None
    command = real_batch_command(**call)
    assert command[command.index("--seed")+1:command.index("--generations")] == ("1", "2", "3", "4", "5")
