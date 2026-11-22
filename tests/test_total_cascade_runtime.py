from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from docking_confidence.application.runtime import runtime_assessment
from docking_confidence.artifact.schema import validate_artifact
from docking_confidence.application.train import train
from test_execution_context import context_data
from test_calibrated_batch import batch_case
from test_training_dataset import make_batch
from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.application.batch import _payload_sha256
from docking_confidence.application.predict import load_prediction_session
from docking_confidence.io.config import config_sha256
import json
from pathlib import Path


def config():
    return {"enabled": True, "require_reduced_faster_than_full": True, "minimum_speedup": 1.,
            "assessment_scope": "total_paired_cascade", "require_measured_overhead": True,
            "overhead_column": "confidence_overhead_sec"}


def test_accepted_only_speedup_can_hide_overall_cascade_loss():
    frame = pd.DataFrame({"runtime_reduced_sec": [1., 100.], "runtime_full_sec": [10., 100.],
                          "confidence_overhead_sec": [0., 0.]})
    legacy = {key: value for key, value in config().items() if key not in
              {"assessment_scope", "require_measured_overhead", "overhead_column"}}
    old = runtime_assessment(frame, legacy, [True, False])
    new = runtime_assessment(frame, config(), [True, False])
    assert old["passed"] is True and old["speedup_full_over_reduced"] == 10
    assert new["passed"] is False and new["cascade_runtime_sec"] == 201
    assert new["measured_saving_sec"] == -91


@pytest.mark.parametrize("overhead", [None, [0., np.nan]])
def test_missing_overhead_blocks_economic_authorization_without_inventing_zero(overhead):
    frame = pd.DataFrame({"runtime_reduced_sec": [1., 1.], "runtime_full_sec": [10., 10.]})
    if overhead is not None:
        frame["confidence_overhead_sec"] = overhead
    result = runtime_assessment(frame, config(), [True, True])
    assert result["saving_upper_bound_sec"] == 18
    assert result["measured_saving_sec"] is None and result["overhead_sec"] is None
    assert result["passed"] is False and result["reason"] == "overhead_not_measured"


def test_measured_overhead_and_all_reduced_costs_are_counted():
    frame = pd.DataFrame({"runtime_reduced_sec": [1., 1.], "runtime_full_sec": [10., 10.],
                          "confidence_overhead_sec": [.5, 1.5]})
    result = runtime_assessment(frame, config(), [True, False])
    assert result["cascade_runtime_sec"] == 14
    assert result["measured_saving_sec"] == 6 and result["passed"] is True
    frame.loc[0, "confidence_overhead_sec"] = -1
    with pytest.raises(ValueError, match="overhead"):
        runtime_assessment(frame, config(), [True, False])


def test_new_runtime_scope_round_trips_and_rejects_forged_cost(context_data, tmp_path):
    cfg, path, _ = context_data
    model = train(str(path), cfg, str(tmp_path / "model"))
    artifact = json.loads(Path(model["artifact"]).read_text())
    validate_artifact(artifact)
    assert artifact["runtime_gate"]["assessment_scope"] == "locked_test_total_paired_cascade_v1"
    assert artifact["runtime_gate"]["overhead_measured"] is False
    changed = deepcopy(artifact)
    changed["runtime_gate"]["cascade_runtime_sec"] += 1
    with pytest.raises(ValueError, match="runtime|cascade"):
        validate_artifact(changed)


def test_serialized_local_artifact_without_new_cost_option_keeps_legacy_replay(context_data, tmp_path):
    cfg, path, frame = context_data
    legacy = deepcopy(cfg)
    for key in ("assessment_scope", "require_measured_overhead", "overhead_column"):
        legacy["runtime_gate"].pop(key)
    frame["config_sha256"] = config_sha256(legacy)
    frame.to_csv(path, index=False)
    result = train(str(path), legacy, str(tmp_path / "legacy_model"))
    session = load_prediction_session(result["artifact"])
    assert session.artifact["runtime_gate"]["assessment_scope"] == "locked_test_rows_accepted_by_domain_and_policy"
    assert "overhead_measured" not in session.artifact["runtime_gate"]


def test_live_confidence_timing_is_measured_and_hash_bound(batch_case):
    _, run, _, directory, _ = batch_case
    run()
    receipt = json.loads((directory("reduced") / "receipt.json").read_text())
    decision = json.loads((directory("reduced") / "confidence_decision.json").read_text())
    assert receipt["confidence_preparation_runtime_sec"] >= 0
    assert decision["confidence_evaluation_runtime_sec"] >= 0
    assert decision["confidence_overhead_sec"] == pytest.approx(
        receipt["confidence_preparation_runtime_sec"] + decision["confidence_evaluation_runtime_sec"])
    assert decision["decision_sha256"] == _payload_sha256({k: v for k, v in decision.items() if k != "decision_sha256"})


def test_training_overhead_requires_a_decision_bound_to_reduced_receipt(make_batch, tmp_path):
    batch = make_batch()
    directory = batch / "targets/1abc_AB/jobs/ag-reduced-seed1"
    receipt = json.loads((directory / "receipt.json").read_text())
    decision = {"reduced_receipt_record_sha256": receipt["record_sha256"], "confidence_overhead_sec": .125}
    decision["decision_sha256"] = _payload_sha256(decision)
    path = directory / "confidence_decision.json"
    path.write_text(json.dumps(decision))
    report = build_training_dataset(batch, tmp_path / "valid")
    assert report["valid"] is True
    frame = pd.read_csv(report["dataset_path"])
    assert frame.confidence_overhead_sec.iloc[0] == .125
    decision["confidence_overhead_sec"] = .001
    path.write_text(json.dumps(decision))
    invalid = build_training_dataset(batch, tmp_path / "invalid")
    assert invalid["valid"] is False
