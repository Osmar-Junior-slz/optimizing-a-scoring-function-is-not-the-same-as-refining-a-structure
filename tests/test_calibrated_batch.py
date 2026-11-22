from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from docking_confidence.application.batch import (
    BatchError, BatchPlanConflictError, clear_batch_cancel_request,
    load_batch_status, request_batch_cancel, run_batch,
)
from docking_confidence.application.batch_monitor import load_batch_monitor
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ProtocolMode
from test_batch_runner import _RecordingEngine, _make_report


def _pairs(*seeds):
    return build_execution_plan(
        algorithms=[Algorithm.PSO], protocol_mode=ProtocolMode.REDUCED_VS_FULL,
        base_config=BaseSearchConfig(generations=4, population=3, top_pockets=2), seeds=seeds,
    )


@pytest.fixture
def batch_case(tmp_path, monkeypatch):
    class Gate:
        action = "ACCEPT_REDUCED"
        capable = True
        fail_prepare = False
        fail_evaluate = False
        evaluations = []
        preparations = []
        validations = []

        def __init__(self, path):
            self.artifact_sha256 = hashlib.sha256(Path(path).read_bytes()).hexdigest()

        def audit_descriptor(self):
            return {"artifact_sha256": self.artifact_sha256, "schema_version": "test_gate"}

        def validate_jobs(self, jobs, engine):
            self.validations.append((jobs, engine))

        def prepare_evidence(self, *, task_directory, target, job, result):
            self.preparations.append(job.job_id)
            if self.fail_prepare:
                raise ValueError("pose preparation failed")
            task_directory.mkdir(parents=True, exist_ok=True)
            (task_directory / "raw_evidence.txt").write_text("verified raw pose", encoding="utf-8")
            (task_directory / "confidence").mkdir(exist_ok=True)
            (task_directory / "confidence" / "evidence.json").write_text('{"schema_version":"test_evidence"}', encoding="utf-8")

        def evaluate(self, *, task_directory, receipt):
            self.evaluations.append(receipt["job"]["job_id"])
            if self.fail_evaluate:
                raise ValueError("model evaluation failed")
            if (task_directory / "raw_evidence.txt").read_text(encoding="utf-8") != "verified raw pose":
                raise ValueError("pose integrity failed")
            return {
                "action": self.action, "reason": "test_calibrated_policy",
                "probability_lower": 0.93, "probability_upper": 0.98,
                "probability": 0.95, "threshold": 0.9,
                "artifact_sha256": self.artifact_sha256, "decision_capable": self.capable,
            }

    module = ModuleType("docking_confidence.application.calibrated_batch")
    module.CalibratedBatchGate = Gate
    monkeypatch.setitem(sys.modules, module.__name__, module)
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, count=1)
    output = tmp_path / "output"
    artifact = tmp_path / "artifact.json"
    artifact.write_text("model version one", encoding="utf-8")
    engine = _RecordingEngine()
    jobs = _pairs(7)
    arguments = dict(jobs=jobs, engine=engine, report=report, confidence_artifact=artifact)

    def run():
        return run_batch(source, output, **arguments)

    def directory(protocol):
        job = next(job for job in jobs if job.protocol.value == protocol)
        return output / "targets" / report.targets[0].complex_id / "jobs" / job.job_id

    return Gate, run, arguments, directory, output


def _protocols(engine):
    return [call[0].protocol.value for call in engine.calls]


def _decision(directory):
    return json.loads((directory("reduced") / "confidence_decision.json").read_text(encoding="utf-8"))


def test_accept_runs_reduced_first_and_never_fabricates_full_receipt(batch_case):
    gate, run, arguments, directory, output = batch_case
    summary = run()
    assert summary.state == "COMPLETED"
    assert (summary.attempted, summary.succeeded, summary.skipped_calibrator) == (1, 1, 1)
    assert _protocols(arguments["engine"]) == ["reduced"]
    assert not (directory("full") / "receipt.json").exists()
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert [job["protocol"] for job in manifest["plan"]["jobs"]] == ["reduced", "full"]
    decision = _decision(directory)
    assert decision["plan_sha256"] == manifest["plan_sha256"]
    assert decision["artifact_sha256"] == manifest["plan"]["confidence_gate"]["artifact_sha256"]
    assert decision["reduced_receipt_sha256"] == hashlib.sha256((directory("reduced") / "receipt.json").read_bytes()).hexdigest()
    assert decision["full_task_sha256"] != decision["reduced_task_sha256"]
    assert gate.preparations == gate.evaluations
    monitor = load_batch_monitor(output, probe=lambda *args: False)
    assert monitor["counts"]["skipped_calibrator"] == 1
    assert monitor["global_fraction"] == 1
    assert monitor["confidence_decisions"][0]["action"] == "ACCEPT_REDUCED"
    status = load_batch_status(output)
    full = next(task for task in status["tasks"].values() if task["protocol"] == "full")
    assert full["state"] == "SKIPPED_CALIBRATOR"
    assert full["attempt_count"] == 0


def test_reject_runs_full_and_records_the_decision(batch_case):
    gate, run, arguments, directory, _ = batch_case
    gate.action = "RUN_FULL"
    summary = run()
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    assert summary.succeeded == 2
    assert summary.skipped_calibrator == 0
    assert _decision(directory)["action"] == "RUN_FULL"


@pytest.mark.parametrize("failure", ["engine", "prepare", "evaluate", "unauthorized"])
def test_reduced_or_calibrator_failure_always_falls_back_to_full(batch_case, failure):
    gate, run, arguments, directory, _ = batch_case
    if failure == "engine":
        original = arguments["engine"].run

        def fail_reduced(**kwargs):
            result = original(**kwargs)
            if kwargs["job"].protocol.value == "reduced":
                raise ValueError("REDUCED engine failure")
            return result

        arguments["engine"].run = fail_reduced
    gate.fail_prepare = failure == "prepare"
    gate.fail_evaluate = failure == "evaluate"
    gate.capable = failure != "unauthorized"
    summary = run()
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    assert (directory("full") / "receipt.json").is_file()
    assert summary.skipped_calibrator == 0
    decision = _decision(directory)
    assert decision["action"] == "RUN_FULL"
    assert decision["reason"] == ("reduced_failed" if failure in {"engine", "prepare"} else "calibrator_evaluation_failed")
    assert summary.failed == (1 if failure in {"engine", "prepare"} else 0)


def test_resume_reuses_reduced_but_recomputes_acceptance(batch_case):
    gate, run, arguments, directory, _ = batch_case
    first = run()
    second = run()
    assert first.batch_id == second.batch_id
    assert second.attempted == 0
    assert second.skipped_existing == 1
    assert second.skipped_calibrator == 1
    assert len(gate.evaluations) == 2
    assert len(gate.preparations) == 1
    assert _protocols(arguments["engine"]) == ["reduced"]
    assert not (directory("full") / "receipt.json").exists()


def test_tampered_old_decision_and_status_cannot_suppress_fallback(batch_case):
    gate, run, arguments, directory, output = batch_case
    run()
    (directory("reduced") / "confidence_decision.json").write_text('{"action":"ACCEPT_REDUCED"}', encoding="utf-8")
    status_path = output / "batch_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    for task in status["tasks"].values():
        task["state"] = "SKIPPED_CALIBRATOR"
    status_path.write_text(json.dumps(status), encoding="utf-8")
    gate.action = "RUN_FULL"
    resumed = run()
    assert resumed.attempted == 1
    assert resumed.skipped_calibrator == 0
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    assert _decision(directory)["action"] == "RUN_FULL"


def test_tampered_receipt_requires_new_reduced_run(batch_case):
    gate, run, arguments, directory, _ = batch_case
    run()
    receipt_path = directory("reduced") / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["engine_result"]["scan_gap_norm"] = 0.999
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    resumed = run()
    assert resumed.attempted == 1
    assert _protocols(arguments["engine"]) == ["reduced", "reduced"]
    assert len(gate.preparations) == 2


def test_rehashed_receipt_cannot_change_the_immutable_target(batch_case):
    from docking_confidence.application.batch import _payload_sha256

    gate, run, arguments, directory, _ = batch_case
    run()
    receipt_path = directory("reduced") / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["target"]["complex_fingerprint_sha256"] = "f" * 64
    receipt["record_sha256"] = _payload_sha256({key: value for key, value in receipt.items() if key != "record_sha256"})
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    resumed = run()
    assert resumed.attempted == 1
    assert resumed.skipped_calibrator == 0
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    assert len(gate.evaluations) == 1
    assert _decision(directory)["error_message"] == "REDUCED receipt disagrees with the immutable plan: target"


def test_changed_evidence_does_not_reuse_previous_acceptance(batch_case):
    _, run, arguments, directory, _ = batch_case
    run()
    (directory("reduced") / "raw_evidence.txt").write_text("altered pose", encoding="utf-8")
    resumed = run()
    assert resumed.attempted == 1
    assert resumed.skipped_calibrator == 0
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    assert _decision(directory)["error_message"] == "pose integrity failed"


def test_rewritten_evidence_manifest_is_rejected_before_model_evaluation(batch_case):
    gate, run, arguments, directory, _ = batch_case
    run()
    (directory("reduced") / "confidence" / "evidence.json").write_text('{"forged":"evidence"}', encoding="utf-8")
    resumed = run()
    assert resumed.attempted == 1
    assert resumed.skipped_calibrator == 0
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    assert len(gate.evaluations) == 1
    assert _decision(directory)["error_message"] == "REDUCED confidence evidence disagrees with its receipt"


def test_changed_artifact_cannot_reuse_batch_directory(batch_case):
    _, run, arguments, _, _ = batch_case
    run()
    arguments["confidence_artifact"].write_text("different model", encoding="utf-8")
    with pytest.raises(BatchPlanConflictError, match="different"):
        run()
    assert _protocols(arguments["engine"]) == ["reduced"]


def test_cancellation_between_reduced_and_decision_is_resumable(batch_case):
    gate, run, arguments, directory, output = batch_case
    arguments["engine"].cancel_output = output
    cancelled = run()
    assert cancelled.state == "CANCELLED"
    assert cancelled.pending == 1
    assert cancelled.skipped_calibrator == 0
    assert not (directory("reduced") / "confidence_decision.json").exists()
    assert not gate.evaluations
    clear_batch_cancel_request(output)
    arguments["engine"].cancel_output = None
    resumed = run()
    assert resumed.state == "COMPLETED"
    assert resumed.skipped_calibrator == 1
    assert resumed.attempted == 0


def test_previously_executed_full_stays_succeeded_on_resume(batch_case):
    gate, run, arguments, _, _ = batch_case
    gate.action = "RUN_FULL"
    run()
    gate.action = "ACCEPT_REDUCED"
    resumed = run()
    assert resumed.attempted == 0
    assert resumed.succeeded == 2
    assert resumed.skipped_calibrator == 0
    assert len(gate.evaluations) == 2
    assert _protocols(arguments["engine"]) == ["reduced", "full"]


def test_cancel_during_calibration_prevents_starting_full(batch_case, monkeypatch):
    gate, run, arguments, _, output = batch_case
    original = gate.evaluate

    def cancel_after_evaluation(self, **kwargs):
        decision = original(self, **kwargs)
        request_batch_cancel(output)
        return decision

    gate.action = "RUN_FULL"
    monkeypatch.setattr(gate, "evaluate", cancel_after_evaluation)
    summary = run()
    assert summary.state == "CANCELLED"
    assert summary.pending == 1
    assert _protocols(arguments["engine"]) == ["reduced"]


def test_failure_to_persist_acceptance_runs_full(batch_case, monkeypatch):
    from docking_confidence.application import batch

    _, run, arguments, _, output = batch_case
    original = batch._atomic_write_json

    def unavailable_decision(path, payload):
        if path.name == "confidence_decision.json":
            raise PermissionError("decision file unavailable")
        original(path, payload)

    monkeypatch.setattr(batch, "_atomic_write_json", unavailable_decision)
    summary = run()
    assert summary.succeeded == 2
    assert summary.skipped_calibrator == 0
    assert _protocols(arguments["engine"]) == ["reduced", "full"]
    status = load_batch_status(output)
    assert all(task["confidence_decision"]["reason"] == "confidence_decision_write_failed"
               for task in status["tasks"].values())


def test_failed_full_is_retried_after_reduced_is_revalidated(batch_case):
    gate, run, arguments, _, _ = batch_case
    gate.action = "RUN_FULL"
    original = arguments["engine"].run

    def fail_full(**kwargs):
        result = original(**kwargs)
        if kwargs["job"].protocol.value == "full":
            raise ValueError("FULL engine failure")
        return result

    arguments["engine"].run = fail_full
    first = run()
    assert first.failed == 1
    arguments["engine"].run = original
    resumed = run()
    assert resumed.attempted == 1
    assert resumed.succeeded == 2
    assert len(gate.evaluations) == 2
    assert _protocols(arguments["engine"]) == ["reduced", "full", "full"]


@pytest.mark.parametrize("protocol", ["full", "reduced"])
def test_calibrated_batch_requires_complete_pairs_before_running(batch_case, protocol):
    gate, run, arguments, _, _ = batch_case
    arguments["jobs"] = [job for job in arguments["jobs"] if job.protocol.value == protocol]
    with pytest.raises(BatchError, match="requires a FULL and REDUCED"):
        run()
    assert not gate.validations
    assert not arguments["engine"].calls


def test_legacy_batch_keeps_plan_without_confidence_configuration(batch_case):
    gate, run, arguments, _, _ = batch_case
    arguments["confidence_artifact"] = None
    summary = run()
    manifest = json.loads(summary.manifest_path.read_text(encoding="utf-8"))
    assert "confidence_gate" not in manifest["plan"]
    assert manifest["plan"]["jobs"][0]["job_id"] == arguments["jobs"][0].job_id
    assert not gate.validations
    assert not gate.evaluations
    assert summary.succeeded == 2
