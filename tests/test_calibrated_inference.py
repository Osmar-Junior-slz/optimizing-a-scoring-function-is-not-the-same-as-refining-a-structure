"""Real pose -> graph -> verified model -> persisted batch decision integration."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pandas as pd
import pytest

from docking_confidence.application.batch import _payload_sha256, load_batch_status, run_batch
from docking_confidence.application.calibrated_batch import CalibratedBatchGate
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.application.predict import predict, load_prediction_session
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, DockingTask, ProtocolMode
from docking_confidence.io.batch_discovery import discover_ppi4dock
from test_training_context import contextual_training
from test_training_dataset import _FixturePoseEngine, make_batch


@pytest.fixture
def connected(contextual_training, make_batch, tmp_path):
    _, _, _, trained, _ = contextual_training
    original = make_batch()
    manifest = json.loads((original / "batch_manifest.json").read_text(encoding="utf-8"))
    source = Path(manifest["plan"]["input_root"])
    jobs = build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.REDUCED_VS_FULL,
                                base_config=BaseSearchConfig(2, 4, 2), seeds=[1])
    engine = _FixturePoseEngine(set())
    output = tmp_path / "connected"
    summary = run_batch(source, output, jobs=jobs, engine=engine,
                        report=discover_ppi4dock(source, validate_contents=True),
                        confidence_artifact=trained["artifact"])
    directory = output / "targets/1abc_AB/jobs/ag-reduced-seed1"
    receipt = json.loads((directory / "receipt.json").read_text(encoding="utf-8"))
    return trained, output, directory, receipt, summary, jobs, engine


def test_live_connection_scores_reduced_and_preserves_blocked_authorization(connected):
    trained, output, directory, receipt, summary, *_ = connected
    assert summary.succeeded == 2 and summary.failed == 0
    decision = json.loads((directory / "confidence_decision.json").read_text(encoding="utf-8"))
    assert decision["action"] == "RUN_FULL"
    prediction = pd.read_csv(directory / "confidence/prediction.csv")
    assert prediction["probability_lower"].between(0, 1).all()
    assert prediction["probability_upper"].between(0, 1).all()
    assert prediction.loc[0, "action"] == "RUN_FULL"
    assert receipt["confidence_evidence_sha256"]
    assert (output / "targets/1abc_AB/jobs/ag-full-seed1/receipt.json").is_file()
    assert load_batch_status(output)["counts"]["skipped_calibrator"] == 0
    assert load_prediction_session(trained["artifact"]).artifact["decision_capable"] is False


def test_input_whitelist_and_shared_prediction_match_official_api(connected, tmp_path):
    trained, _, directory, *_ = connected
    input_path = directory / "confidence/inference_input.csv"
    frame = pd.read_csv(input_path)
    assert not any("dockq" in name or "reference" in name or name.endswith("_full") for name in frame)
    assert not {"fnat", "irmsd", "lrmsd", "runtime_full_sec", "preserved"}.intersection(frame)
    expected = predict(trained["artifact"], str(input_path), str(tmp_path / "prediction.csv"))
    actual = pd.read_csv(directory / "confidence/prediction.csv")
    for column in ("probability", "probability_lower", "probability_upper", "base_raw_score"):
        assert actual[column].to_numpy() == pytest.approx(expected[column].to_numpy())


def test_reference_metrics_and_full_result_are_not_read_by_inference(connected):
    trained, output, directory, receipt, _, jobs, engine = connected
    (directory / receipt["engine_result"]["reference_metrics_path"]).unlink()
    (output / "targets/1abc_AB/jobs/ag-full-seed1/receipt.json").unlink()
    altered = deepcopy(receipt)
    altered["engine_result"].update(dockq=0.0, fnat=0.0, irmsd=999.0, lrmsd=999.0)
    altered["record_sha256"] = _payload_sha256({key: value for key, value in altered.items()
                                              if key != "record_sha256"})
    gate = CalibratedBatchGate(trained["artifact"])
    gate.validate_jobs(jobs, engine)
    a = gate.evaluate(task_directory=directory, receipt=receipt)
    b = gate.evaluate(task_directory=directory, receipt=altered)
    assert a == b


@pytest.mark.parametrize("filename", ["evidence.json", "contact_graph.json", "reduced_complex.pdb"])
def test_modified_live_evidence_cannot_skip_full(connected, filename):
    trained, _, directory, receipt, _, jobs, engine = connected
    path = directory / "confidence" / filename
    path.write_bytes(path.read_bytes() + b"\n ")
    gate = CalibratedBatchGate(trained["artifact"])
    gate.validate_jobs(jobs, engine)
    with pytest.raises(ValueError, match="evidence|hash"):
        gate.evaluate(task_directory=directory, receipt=receipt)


def test_incompatible_task_rejected_before_engine_execution(connected):
    trained, _, _, _, _, jobs, engine = connected
    jobs = [replace(job, profile=replace(job.profile, docking_task=DockingTask.SURFACE_DISCOVERY))
            for job in jobs]
    with pytest.raises(ValueError, match="task is incompatible"):
        CalibratedBatchGate(trained["artifact"]).validate_jobs(jobs, engine)


def test_execution_contract_mismatch_is_explicit_even_with_valid_scores(connected):
    trained, _, directory, receipt, _, jobs, engine = connected
    gate = CalibratedBatchGate(trained["artifact"])
    gate.validate_jobs(jobs, engine)
    assert gate.protocol_issues == ["training_execution_contract_unavailable"]
    result = gate.evaluate(task_directory=directory, receipt=receipt)
    assert result["action"] == "RUN_FULL"
    assert result["reason"] == "training_execution_contract_unavailable"
    assert result["feature_source"] == "reference_free_reduced_pose"
    assert 0 <= result["probability_lower"] <= result["probability_upper"] <= 1
