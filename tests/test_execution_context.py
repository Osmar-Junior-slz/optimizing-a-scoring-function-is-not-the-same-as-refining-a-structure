from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from docking_confidence.application.train import train
from docking_confidence.application.predict import predict, load_prediction_session
from docking_confidence.features.execution_context import (
    ALGORITHM_FEATURE_COLUMNS, INPUT_POSE_REFERENCE_ROLE, local_execution_context,
)
from docking_confidence.io.config import (
    config_sha256, load_config, validate_config, LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS,
)
from docking_confidence.io.dataset import load_paired_dataset
from test_training_context import _context_frame
from test_training_dataset import _FixturePoseEngine, make_batch

ROOT = Path(__file__).parents[1]


def test_context_uses_only_supplied_partner_models_and_pose(tmp_path, monkeypatch):
    receptor, ligand, pose = [tmp_path / name for name in ["receptor.pdb", "ligand.pdb", "pose.json"]]
    for path in [receptor, ligand, pose]:
        path.write_text(path.name)
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in [receptor, ligand]]
    observed = []
    def evaluate(**kwargs):
        observed.append(kwargs)
        return SimpleNamespace(as_dict=lambda: {"dockq": .72})
    monkeypatch.setattr("dockingpp.evaluation.structural_metrics.evaluate_saved_pose", evaluate)
    features, trace = local_execution_context(
        algorithm="pso", receptor_model_pdb=receptor, ligand_template_pdb=ligand,
        pose_json=pose, expected_receptor_sha256=hashes[0], expected_ligand_sha256=hashes[1],
    )
    assert observed[0]["native_peptide_pdb"] == ligand.resolve()
    assert observed[0]["receptor_pdb"] == receptor.resolve()
    assert set(observed[0]) == {
        "receptor_pdb", "native_peptide_pdb", "pose_json",
        "contact_cutoff_angstrom", "interface_cutoff_angstrom",
    }
    assert features["algorithm_pso"] == 1 and features["algorithm_ag"] == 0
    assert features["input_pose_retention"] == .72
    assert features["input_pose_retention_reference_role"] == INPUT_POSE_REFERENCE_ROLE
    assert trace["experimental_reference_used"] is False
    assert trace["full_result_used"] is False
    ligand.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        local_execution_context(
            algorithm="pso", receptor_model_pdb=receptor, ligand_template_pdb=ligand,
            pose_json=pose, expected_receptor_sha256=hashes[0], expected_ligand_sha256=hashes[1],
        )


def test_context_profile_preserves_all_scientific_gates_and_frozen_v1():
    previous = load_config(ROOT / "configs/local_refinement_pilot.yaml")
    current = load_config(ROOT / "configs/local_refinement_context_pilot.yaml")
    assert tuple(current["features"]["columns"]) == LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS
    assert len(current["features"]["columns"]) == 15
    for key in previous.keys() - {"features"}:
        assert current[key] == previous[key]
    assert current["authorization"]["allow_automatic_decision"] is False
    old = deepcopy(previous)
    old["features"]["columns"].append("input_pose_retention")
    with pytest.raises(ValueError, match="explicit|frozen"):
        validate_config(old)


@pytest.fixture
def context_data(tmp_path):
    old = load_config(ROOT / "configs/local_refinement_pilot.yaml")
    current = load_config(ROOT / "configs/local_refinement_context_pilot.yaml")
    path = tmp_path / "context.csv"
    frame = _context_frame(path, old, fifty=True)
    frame["feature_schema"] = current["features"]["schema_version"]
    frame["config"] = str(ROOT / "configs/local_refinement_context_pilot.yaml")
    frame["config_sha256"] = config_sha256(current)
    for column in ALGORITHM_FEATURE_COLUMNS:
        frame[column] = (frame.algorithm == column.removeprefix("algorithm_")).astype(int)
    frame["input_pose_retention"] = .2 + .6 * frame["viable_fraction_mean"]
    frame["input_pose_retention_reference_role"] = INPUT_POSE_REFERENCE_ROLE
    for role in ["receptor", "ligand", "pose"]:
        frame[f"input_pose_retention_{role}_sha256"] = "a" * 64
    frame.to_csv(path, index=False)
    return current, path, frame


@pytest.mark.parametrize("field,value", [
    ("input_pose_retention",1.1),
    ("input_pose_retention_reference_role","experimental_complex"),
    ("algorithm_pso",1.0),
])
def test_context_loader_rejects_wrong_reference_range_or_algorithm(context_data, field, value):
    config, path, frame = context_data
    frame.loc[0,field] = value
    frame.to_csv(path,index=False)
    with pytest.raises(ValueError, match="retention|experimental|algorithm"):
        load_paired_dataset(
            path, config["features"]["columns"], config["contact_graph"],
            docking_task="local_refinement",feature_schema=config["features"]["schema_version"],
        )


def test_context_model_round_trip_ignores_experimental_outcomes(context_data, tmp_path):
    config,path,frame = context_data
    result = train(str(path),config,str(tmp_path/"model"))
    session = load_prediction_session(result["artifact"])
    assert session.artifact["feature_columns"] == list(LOCAL_REFINEMENT_CONTEXT_FEATURE_COLUMNS)
    assert session.artifact["decision_capable"] is False
    first = predict(result["artifact"],str(path),str(tmp_path/"first.csv"))
    inference = frame.drop(columns=["dockq_reduced","dockq_full","runtime_full_sec","scientific_full","engine_full"])
    inference_path=tmp_path/"reduced_only.csv"
    inference.to_csv(inference_path,index=False)
    second = predict(result["artifact"],str(inference_path),str(tmp_path/"second.csv"))
    for column in ["base_raw_score","probability","probability_lower","probability_upper"]:
        np.testing.assert_array_equal(first[column],second[column])


def test_venn_abers_matches_independent_isotonic_implementation():
    from sklearn.isotonic import IsotonicRegression
    from docking_confidence.calibration.venn_abers import VennAbersCalibrator
    rng=np.random.default_rng(67)
    scores=np.round(rng.normal(size=60),1)
    labels=rng.integers(0,2,size=60)
    queries=np.r_[scores[:8],-5.,5.]
    lower,upper,_=VennAbersCalibrator().fit(scores,labels).predict_interval(queries)
    for i,query in enumerate(queries):
        for hypothetical,actual in [(0,lower[i]),(1,upper[i])]:
            fitted=IsotonicRegression(increasing=True,out_of_bounds="clip").fit(
                np.r_[scores,query],np.r_[labels,hypothetical])
            assert actual == pytest.approx(fitted.predict([query])[0],abs=1e-12)


def test_live_context_is_frozen_before_experimental_scoring_and_ignores_full(
    context_data, make_batch, tmp_path,
):
    from docking_confidence.application.batch import run_batch,_payload_sha256
    from docking_confidence.application.calibrated_batch import CalibratedBatchGate
    from docking_confidence.application.planning import build_execution_plan
    from docking_confidence.domain.execution import Algorithm,BaseSearchConfig,ProtocolMode
    config,dataset,_=context_data
    model=train(str(dataset),config,str(tmp_path/"model"))
    fixture=make_batch()
    source=Path(json.loads((fixture/"batch_manifest.json").read_text())["plan"]["input_root"])
    jobs=build_execution_plan(
        algorithms=[Algorithm.AG],protocol_mode=ProtocolMode.REDUCED_VS_FULL,
        base_config=BaseSearchConfig(2,4,2),seeds=[1],
    )
    output=tmp_path/"live_context"
    summary=run_batch(
        source,output,jobs=jobs,engine=_FixturePoseEngine(set()),
        confidence_artifact=model["artifact"],
    )
    assert summary.succeeded == 2 and summary.skipped_calibrator == 0
    directory=output/"targets/1abc_AB/jobs/ag-reduced-seed1"
    receipt=json.loads((directory/"receipt.json").read_text())
    evidence=json.loads((directory/"confidence/evidence.json").read_text())
    assert evidence["execution_context"]["input_pose_retention"] == pytest.approx(1.)
    assert evidence["execution_context_evidence"]["experimental_reference_used"] is False
    assert evidence["execution_context"]["input_pose_retention_pose_sha256"] == evidence["engine_pose_sha256"]
    gate=CalibratedBatchGate(model["artifact"])
    first=gate.evaluate(task_directory=directory,receipt=receipt)
    (directory/receipt["engine_result"]["reference_metrics_path"]).unlink()
    (output/"targets/1abc_AB/jobs/ag-full-seed1/receipt.json").unlink()
    changed=deepcopy(receipt)
    changed["engine_result"].update(dockq=0.,fnat=0.,irmsd=999.,lrmsd=999.)
    changed["record_sha256"]=_payload_sha256({k:v for k,v in changed.items() if k!="record_sha256"})
    second=gate.evaluate(task_directory=directory,receipt=changed)
    assert first == second
