from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import threading
import time

import pytest

from docking_confidence.application import batch_telemetry as telemetry
from docking_confidence.application.batch import (
    _atomic_write_json, _initial_status, _manifest_for_plan, _job_snapshot,
    _write_status, load_batch_status, run_batch,
)
from docking_confidence.application.batch_monitor import (
    estimate_remaining, load_batch_monitor, plan_breakdown, process_alive, worker_health,
)
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode
from test_batch_runner import _RecordingEngine, _make_report, _jobs


def iso(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


def manifest_fixture(root, complexes=2, *, fixed=False):
    base = BaseSearchConfig(generations=1, experiment=FactorialExperimentConfig(pose_source="synthetic")) if fixed else BaseSearchConfig()
    mode = ProtocolMode.FACTORIAL if fixed else ProtocolMode.REDUCED_VS_FULL
    jobs = build_execution_plan(algorithms=list(Algorithm), protocol_mode=mode, base_config=base, seeds=[1])
    manifest = _manifest_for_plan({
        "input_root": "C:/data/200_proteinas", "jobs": [_job_snapshot(job) for job in jobs],
        "targets": [{"complex_id": f"target{i}_AB", "eligible": True} for i in range(complexes)],
    })
    _atomic_write_json(root / "batch_manifest.json", manifest)
    status = _initial_status(manifest=manifest, previous=None)
    _write_status(root / "batch_status.json", status)
    return manifest, status


def metric(generation, count=100, *, total=75, pocket=0):
    return json.dumps({"generation": generation, "total_generations": total,
                       "pocket_index": pocket, "step": 99999,
                       "name": "n_eval", "value": count}) + "\n"


def test_metrics_generation_zero_incremental_partial_and_duplicate_rows(tmp_path):
    path = tmp_path / "metrics.jsonl"
    reader = telemetry.MetricsReader()
    path.write_text(metric(0), encoding="utf-8")
    assert reader.read(tmp_path)["generation_current"] == 0
    assert reader.read(tmp_path)["evaluations_current"] == 100
    with path.open("a", encoding="utf-8") as handle:
        handle.write(metric(1, 82) + metric(1, 82) + metric(2, 99)[:-1])
    progress = reader.read(tmp_path)
    assert progress["generation_current"] == 1
    assert progress["evaluations_current"] == 182
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert reader.read(tmp_path)["evaluations_current"] == 281
    # Another pocket starts at generation 0; never turn global step into a generation.
    with path.open("a", encoding="utf-8") as handle:
        handle.write(metric(0, 55, pocket=1))
    assert reader.read(tmp_path)["evaluations_current"] == 55
    assert reader.read(tmp_path)["generation_current"] == 0
    path.write_text(metric(0, 10), encoding="utf-8")
    assert reader.read(tmp_path)["evaluations_current"] == 10


def test_metrics_do_not_simulate_or_accept_invalid_progress(tmp_path):
    path = tmp_path / "metrics.jsonl"
    path.write_text("null\n{}\ninvalid\n" + json.dumps({"step": 47}) + "\n" + metric(76), encoding="utf-8")
    assert telemetry.MetricsReader().read(tmp_path) == {}


def test_heartbeat_periodic_persistence_and_current_job(tmp_path):
    manifest, status = manifest_fixture(tmp_path)
    task = next(iter(status["tasks"].values()))
    task.update(state="RUNNING", started_at_utc=iso(time.time()), attempt_count=1)
    job = manifest["plan"]["jobs"][0]
    work = tmp_path / "engine"
    with telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"], interval=0.03) as heartbeat:
        heartbeat.start_job(task, job, work)
        first_stamp = heartbeat.data["updated_at_utc"]
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            second = telemetry.read_json(tmp_path / telemetry.HEARTBEAT_FILENAME)
            if second.get("updated_at_utc") and second["updated_at_utc"] != first_stamp:
                break
            time.sleep(0.02)
        assert second["updated_at_utc"] != first_stamp
        assert second["current_job_id"] == job["job_id"]
        for key in ("complex_id", "algorithm", "protocol", "seed", "generation_current",
                    "generation_total", "evaluations_current", "evaluations_total"):
            assert key in second
        output = work / "dockingpp_runs" / "new_attempt" / "output"
        output.mkdir(parents=True)
        (output / "metrics.jsonl").write_text(metric(0, 100) + metric(47, 4682), encoding="utf-8")
        heartbeat.pulse()
        current = telemetry.read_json(tmp_path / telemetry.HEARTBEAT_FILENAME)
        assert current["generation_current"] == 47
        assert current["evaluations_current"] == 4782
        assert any(e["kind"] == "generation" for e in current["recent_events"])
        heartbeat.end_job({**task, "state": "SUCCEEDED"})
        heartbeat.finish("COMPLETED")
    final = telemetry.read_json(tmp_path / telemetry.HEARTBEAT_FILENAME)
    assert final["worker_state"] == "COMPLETED"
    assert final["current_job_id"] is None
    stamp = (tmp_path / telemetry.HEARTBEAT_FILENAME).stat().st_mtime_ns
    time.sleep(0.06)
    assert (tmp_path / telemetry.HEARTBEAT_FILENAME).stat().st_mtime_ns == stamp


def test_heartbeat_ignores_previous_attempt_metrics_and_bounds_events(tmp_path):
    manifest, status = manifest_fixture(tmp_path)
    job = manifest["plan"]["jobs"][0]
    task = next(iter(status["tasks"].values()))
    task.update(started_at_utc=iso(time.time()), attempt_count=2)
    work = tmp_path / "engine"
    old = work / "dockingpp_runs" / "old" / "output"
    old.mkdir(parents=True)
    (old / "metrics.jsonl").write_text(metric(75), encoding="utf-8")
    heartbeat = telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"])
    heartbeat.start_job(task, job, work)
    for _ in range(telemetry.MAX_EVENTS + 2):
        heartbeat.pulse()
    current = telemetry.read_json(tmp_path / telemetry.HEARTBEAT_FILENAME)
    assert current["generation_current"] is None
    assert len(current["recent_events"]) == telemetry.MAX_EVENTS


def test_metrics_read_failure_keeps_heartbeat_alive(tmp_path, monkeypatch):
    manifest, status = manifest_fixture(tmp_path)
    task = next(iter(status["tasks"].values()))
    task.update(started_at_utc=iso(time.time()), attempt_count=1)
    heartbeat = telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"])
    work = tmp_path / "engine"
    heartbeat.start_job(task, manifest["plan"]["jobs"][0], work)
    (work / "dockingpp_runs" / "attempt" / "output").mkdir(parents=True)
    def unavailable(*args):
        raise PermissionError("temporarily locked")
    monkeypatch.setattr(heartbeat._reader, "read", unavailable)
    heartbeat.pulse()
    data = telemetry.read_json(tmp_path / telemetry.HEARTBEAT_FILENAME)
    assert data["current_job_id"] == task["job_id"]
    assert "temporarily locked" in data["monitor_warning"]
    assert data["recent_events"][-1]["kind"] == "heartbeat"


def test_telemetry_io_failure_does_not_change_batch_results(tmp_path, monkeypatch):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, count=1)
    def broken(*args, **kwargs):
        raise PermissionError("telemetry unavailable")
    monkeypatch.setattr(telemetry, "_atomic_write_json", broken)
    result = run_batch(source, tmp_path / "output", jobs=_jobs(1), engine=_RecordingEngine(), report=report)
    assert result.succeeded == 1
    assert result.failed == 0


@pytest.mark.parametrize("state,age,alive,kind", [
    ("RUNNING", 2, True, "healthy"), ("RUNNING", 120, True, "stale"),
    ("RUNNING", 120, None, "stale"), ("RUNNING", 2, False, "stopped"),
    ("COMPLETED", 200, False, "finished"), ("COMPLETED_WITH_ERRORS", 0, True, "finished"),
    ("CANCELLED", 100, False, "finished"), ("RUNNING", None, None, "unknown"),
])
def test_worker_health_checks_process_without_guessing_death(state, age, alive, kind):
    heartbeat = {"updated_at_utc": iso(1000-age)} if age is not None else {}
    assert worker_health(state, heartbeat, now=1000, alive=alive)["kind"] == kind


def test_process_probe_current_and_nonexistent_pid():
    assert process_alive(os.getpid()) is True
    assert process_alive(-1) is None
    assert process_alive(2_000_000_000) is False


@pytest.mark.parametrize("complexes,expected", [(200, 1200), (216, 1296)])
def test_planned_jobs_uses_manifest_and_reports_folder_name_difference(tmp_path, complexes, expected):
    manifest, _ = manifest_fixture(tmp_path, complexes)
    breakdown = plan_breakdown(manifest)
    assert breakdown["planned_jobs"] == expected
    assert breakdown["cartesian"]
    assert breakdown["folder_count_difference"] == complexes - 200
    assert breakdown["duplicate_complex_ids"] == 0
    assert breakdown["algorithms"] == ["ag", "hybrid", "pso"]


def test_non_cartesian_plan_uses_actual_job_list(tmp_path):
    manifest, _ = manifest_fixture(tmp_path)
    manifest["plan"]["jobs"].pop()
    breakdown = plan_breakdown(manifest)
    assert breakdown["planned_jobs"] == 10
    assert breakdown["cartesian_jobs"] == 12
    assert not breakdown["cartesian"]


def test_global_progress_503_of_1296_and_read_only_reopening(tmp_path):
    manifest, status = manifest_fixture(tmp_path, 216)
    for index, task in enumerate(status["tasks"].values()):
        task["state"] = "SUCCEEDED" if index < 488 else "FAILED" if index < 503 else "RUNNING" if index == 503 else "PENDING"
    _write_status(tmp_path / "batch_status.json", status)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    for _ in range(3):
        monitor = load_batch_monitor(tmp_path, probe=lambda *args: True)
        assert monitor["processed"] == 503
        assert monitor["global_fraction"] == pytest.approx(503/1296)
        assert monitor["counts"]["pending"] == 792
        assert monitor["counts"]["running"] == 1
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_reopens_active_job_heartbeat_events_and_last_success(tmp_path):
    manifest, status = manifest_fixture(tmp_path)
    tasks = list(status["tasks"].values())
    tasks[0].update(state="SUCCEEDED", started_at_utc=iso(800), completed_at_utc=iso(900))
    tasks[1].update(state="RUNNING", started_at_utc=iso(910), attempt_count=1)
    _write_status(tmp_path / "batch_status.json", status)
    heartbeat = telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"])
    heartbeat.start_job(tasks[1], manifest["plan"]["jobs"][1], tmp_path / "engine")
    heartbeat.data.update(generation_current=47, generation_total=75, evaluations_current=4782)
    heartbeat.pulse()
    view = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: True)
    assert view["active_jobs"][0]["progress_fraction"] == pytest.approx(47/75)
    assert view["active_jobs"][0]["elapsed_seconds"] == 90
    assert view["last_completed"]["complex_id"] == tasks[0]["complex_id"]
    assert view["events"]
    # A heartbeat for an earlier attempt must never show current progress.
    heartbeat.data["attempt_count"] = 0
    heartbeat.pulse()
    view = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: True)
    assert view["active_jobs"][0]["progress_fraction"] is None
    stopped = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: False)
    assert stopped["counts"]["running"] == 0
    assert stopped["orphaned_running"] == 1
    status["state"] = "COMPLETED"
    _write_status(tmp_path / "batch_status.json", status)
    view = load_batch_monitor(tmp_path, probe=lambda *args: False)
    assert view["active_jobs"] == []
    assert view["health"]["kind"] == "finished"


def test_fixed_budget_progress_uses_effective_evaluations_past_nominal_generation(tmp_path):
    manifest, status = manifest_fixture(tmp_path, fixed=True)
    task = next(iter(status["tasks"].values()))
    task.update(state="RUNNING", started_at_utc=iso(910), attempt_count=1)
    _write_status(tmp_path / "batch_status.json", status)
    work_dir = tmp_path / "engine"
    heartbeat = telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"])
    heartbeat.start_job(task, manifest["plan"]["jobs"][0], work_dir)
    engine_output = work_dir / "dockingpp_runs" / "run" / "output"
    engine_output.mkdir(parents=True)
    (engine_output / "metrics.jsonl").write_text(metric(0, 100, total=1) + metric(2, 85, total=1))
    heartbeat.pulse()

    active = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: True)["active_jobs"][0]
    assert active["generation_current"] == 2 and active["generation_total"] is None
    assert active["evaluations_current"] == 185
    assert active["progress_basis"] == "evaluations"
    assert active["progress_fraction"] == pytest.approx(185 / 7600)


def test_interrupted_worker_and_heartbeat_from_another_run(tmp_path):
    manifest, status = manifest_fixture(tmp_path)
    heartbeat = telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"])
    status["worker_id"] = heartbeat.data["worker_id"]
    _write_status(tmp_path / "batch_status.json", status)
    heartbeat.finish("INTERRUPTED")
    view = load_batch_monitor(tmp_path, probe=lambda *args: True)
    assert view["state"] == "INTERRUPTED"
    assert view["health"]["kind"] == "finished"
    status["worker_id"] = "a-new-worker"
    _write_status(tmp_path / "batch_status.json", status)
    view = load_batch_monitor(tmp_path, probe=lambda *args: True)
    assert view["state"] == "RUNNING"
    assert view["heartbeat"] == {}


def test_eta_starts_with_first_success_and_marks_sparse_or_missing_groups():
    samples = [{"state": "SUCCEEDED", "algorithm": "pso", "protocol": "full",
                "duration_seconds": 20} for _ in range(5)]
    pending = {"state": "PENDING", "algorithm": "pso", "protocol": "full"}
    assert estimate_remaining([pending], now=1000) is None
    initial = estimate_remaining(samples[:1] + [pending, pending], now=1000)
    assert initial["seconds"] == 40
    assert initial["sample_count"] == 1 and initial["preliminary"]
    stable = estimate_remaining(samples + [pending, pending], now=1000)
    assert stable["seconds"] == 40 and not stable["preliminary"]
    fallback = estimate_remaining(samples + [{**pending, "algorithm": "ag"}], now=1000)
    assert fallback["seconds"] == 20 and fallback["preliminary"]
    assert fallback["fallback_jobs"] == 1
    assert estimate_remaining([{**s, "duration_seconds": float("nan")} for s in samples] + [pending], now=1000) is None
    assert estimate_remaining([{**s, "state": "FAILED"} for s in samples] + [pending], now=1000) is None
    assert estimate_remaining([{**s, "state": "SKIPPED_CALIBRATOR"} for s in samples] + [pending], now=1000) is None
    assert estimate_remaining(samples, now=1000) is None


def test_eta_keeps_factorial_conditions_separate():
    def task(initialization, state, duration=None):
        return {"state": state, "algorithm": "pso", "protocol": "factorial_2x2",
                "profile": {"search_extent": "reduced", "initialization": initialization},
                "duration_seconds": duration}
    samples = [task("uniform", "SUCCEEDED", 20) for _ in range(3)]
    samples += [task("seeded", "SUCCEEDED", 80) for _ in range(3)]
    eta = estimate_remaining(samples + [task("uniform", "PENDING"), task("seeded", "PENDING")], now=1000)
    assert eta["seconds"] == 100 and not eta["preliminary"]
    # A new condition has no matching samples even when algorithm/protocol match.
    pending = task("uniform", "PENDING")
    pending["profile"]["search_extent"] = "expanded"
    eta = estimate_remaining(samples + [pending], now=1000)
    assert eta["seconds"] == 50 and eta["preliminary"]
    assert eta["fallback_jobs"] == 1


def test_eta_deducts_active_elapsed_and_suspends_finish_time_when_overdue():
    sample = {"state": "SUCCEEDED", "algorithm": "pso", "protocol": "full", "duration_seconds": 20}
    active = {**sample, "state": "RUNNING", "started_at_utc": iso(995)}
    pending = {**sample, "state": "PENDING"}
    eta = estimate_remaining([sample, active, pending], now=1000)
    assert eta["seconds"] == 35
    assert telemetry.timestamp(eta["finish_at_utc"]) == 1035
    assert eta["overdue_jobs"] == 0
    overdue = estimate_remaining([sample, {**active, "started_at_utc": iso(970)}], now=1000)
    assert overdue["overdue_jobs"] == 1 and overdue["finish_at_utc"] is None


@pytest.mark.parametrize("kind", ["healthy", "stale", "stopped", "finished"])
def test_monitor_early_eta_requires_live_worker_and_elapsed_freezes_after_finish(tmp_path, kind):
    manifest, status = manifest_fixture(tmp_path)
    task = next(iter(status["tasks"].values()))
    task.update(state="SUCCEEDED", started_at_utc=iso(900), completed_at_utc=iso(920), duration_seconds=20)
    if kind == "finished":
        status.update(state="CANCELLED", completed_at_utc=iso(990))
    _write_status(tmp_path / "batch_status.json", status)
    heartbeat = telemetry.WorkerHeartbeat(tmp_path, manifest["batch_id"])
    heartbeat.data.update(worker_started_at_utc=iso(900))
    heartbeat.pulse()
    saved = telemetry.read_json(tmp_path / telemetry.HEARTBEAT_FILENAME)
    saved["updated_at_utc"] = iso(900 if kind == "stale" else 995)
    _atomic_write_json(tmp_path / telemetry.HEARTBEAT_FILENAME, saved)
    view = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: kind != "stopped")
    assert view["health"]["kind"] == kind
    if kind == "healthy":
        assert view["eta"]["preliminary"] and view["eta"]["sample_count"] == 1
        assert view["eta"]["seconds"] == 220
        assert view["elapsed_seconds"] == 100
    else:
        assert view["eta"] is None
    if kind == "finished":
        assert view["elapsed_seconds"] == 90
        assert load_batch_monitor(tmp_path, now=1100, probe=lambda *args: True)["elapsed_seconds"] == 90


def test_worker_stays_running_while_ui_reopens_and_finishes_after_job(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, count=1)
    output = tmp_path / "output"
    entered, release = threading.Event(), threading.Event()
    results = []
    class BlockingEngine(_RecordingEngine):
        def run(self, **kwargs):
            entered.set()
            assert release.wait(10)
            return super().run(**kwargs)
    def worker():
        results.append(run_batch(source, output, jobs=_jobs(1), engine=BlockingEngine(), report=report))
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(5)
        for _ in range(3):
            view = load_batch_monitor(output)
            assert view["counts"]["running"] == 1
            assert view["active_jobs"][0]["job_id"] == _jobs(1)[0].job_id
            assert view["health"]["kind"] == "healthy"
    finally:
        release.set()
        thread.join(timeout=10)
    assert not thread.is_alive()
    assert results[0].succeeded == 1
    final = load_batch_monitor(output)
    assert final["health"]["kind"] == "finished"
    assert final["last_completed"]["duration_seconds"] > 0
