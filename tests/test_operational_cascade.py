"""Unequal operational budgets must not modify scientific controls or reuse old models."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docking_confidence.application.batch import _job_snapshot, run_batch
from docking_confidence.application.calibrated_batch import CalibratedBatchGate
from docking_confidence.application.operational import (
    FullFallbackGate, load_operational_config, operational_gate, validate_operational_jobs,
    validate_training_regime,
)
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import (
    Algorithm, BaseSearchConfig, FactorialExperimentConfig, OperationalSearchConfig, ProtocolMode,
)
from test_batch_runner import _RecordingEngine, _make_report
from test_factorial_engine import _execute


def jobs(algorithm="ag", budget=7600):
    base = replace(load_operational_config(), operational=OperationalSearchConfig(full_evaluation_budget=budget))
    return build_execution_plan(algorithms=[Algorithm(algorithm)], protocol_mode=ProtocolMode.OPERATIONAL,
                                base_config=base, seeds=[1])


def test_operational_budgets_and_scientific_recipe_remain_separate():
    operational = jobs()
    assert [(j.protocol.value, j.profile.population, j.profile.evaluation_budget) for j in operational] == [
        ("reduced", 100, 5700), ("full", 100, 7600)]
    validate_operational_jobs([_job_snapshot(j) for j in operational])
    factorial = build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(experiment=FactorialExperimentConfig(pose_source="existing scientific recipe")), seeds=[1])
    assert len(factorial) == 4
    assert {j.profile.evaluation_budget for j in factorial} == {7600}
    assert {j.profile.initialization for j in factorial} == {"uniform", "seeded"}
    assert all(j.profile.execution_strategy is None for j in factorial)
    legacy = build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.REDUCED_VS_FULL,
                                 base_config=BaseSearchConfig(), seeds=[1])
    assert [(j.profile.generations, j.profile.population, j.profile.evaluation_budget) for j in legacy] == [(75,100,7600),(100,100,10100)]


@pytest.mark.parametrize("algorithm", ["ag", "pso", "hybrid"])
def test_real_native_operators_obey_unequal_exact_budgets(algorithm, tmp_path, monkeypatch):
    # A small budget above population also exercises a partially evaluated final generation.
    for job in jobs(algorithm, budget=160):
        pocket, calls, _ = _execute(job, tmp_path / job.job_id, monkeypatch)
        assert len(calls) == job.profile.evaluation_budget
        assert pocket["pop_size"] == 100
        assert len(pocket["initial_population_audit"]["solution_vectors"]) == 100
        assert pocket["execution_audit"]["stop_reason"] == "objective_evaluation_ceiling"
        assert pocket["initial_population_audit"]["stop_rules"]["hybrid_plateau_patience"] == 0
        assert not pocket["initial_population_audit"]["stop_rules"]["generation_limit_enabled"]


@pytest.mark.parametrize("budget", [True, 7601, 0, -4, 100])
def test_invalid_or_population_incompatible_budget_rejected(budget):
    with pytest.raises(ValueError):
        jobs(budget=budget)


def test_new_budget_explicitly_invalidates_old_calibrator_contract():
    current = jobs()
    old = deepcopy([_job_snapshot(j) for j in current])
    old[0]["profile"]["evaluation_budget"] = 7600
    engine = SimpleNamespace(name="test", scientific=False, audit_descriptor=lambda: {})
    gate = CalibratedBatchGate.__new__(CalibratedBatchGate)
    gate.artifact = {"docking_task":"local_refinement"}
    gate.contract = {"jobs":old, "engine":{"name":"test", "scientific":False, "descriptor":{}}}
    gate.validate_jobs(current, engine)
    assert "reduced_evaluation_budget_differs_from_training" in gate.protocol_issues
    assert "execution_jobs_differ_from_training" in gate.protocol_issues


def test_missing_calibrator_runs_full_and_records_reason(tmp_path, monkeypatch):
    from docking_confidence.application import calibrated_batch
    monkeypatch.setattr(calibrated_batch, "configured_calibrator_path", lambda: None)
    source = tmp_path / "input"
    report = _make_report(source, count=1)
    engine = _RecordingEngine()
    summary = run_batch(source,tmp_path / "output",jobs=jobs(),engine=engine,report=report)
    assert summary.succeeded == 2 and summary.skipped_calibrator == 0
    assert [j.protocol.value for j,_,_ in engine.calls] == ["reduced","full"]
    decision = json.loads(next((tmp_path / "output").rglob("confidence_decision.json")).read_text())
    assert decision["action"] == "RUN_FULL"
    assert decision["reason"] == "operational_calibrator_not_installed"
    manifest=json.loads(summary.manifest_path.read_text())
    assert manifest["plan"]["execution_strategy"] == "operational_cascade_v1"
    assert "experimental_design" not in manifest["plan"]


def test_incompatible_artifact_is_not_scored(tmp_path, monkeypatch):
    from docking_confidence.application import calibrated_batch
    artifact=tmp_path / "artifact.json"; artifact.write_text("old model")
    class OldGate:
        def __init__(self, path): pass
        def validate_jobs(self, jobs, engine): self.protocol_issues=["reduced_evaluation_budget_differs_from_training"]
        def evaluate(self, **kwargs): raise AssertionError("old model must not be scored")
    monkeypatch.setattr(calibrated_batch,"CalibratedBatchGate",OldGate)
    gate=operational_gate(artifact,jobs(),object())
    assert isinstance(gate,FullFallbackGate)
    assert gate.evaluate()["reason"] == "operational_calibrator_requires_retraining"


def test_matched_model_keeps_venn_abers_gate(tmp_path, monkeypatch):
    from docking_confidence.application import calibrated_batch
    artifact=tmp_path / "artifact.json"; artifact.write_text("matched verified model")
    class VerifiedGate:
        def __init__(self, path): self.protocol_issues=[]
        def validate_jobs(self, jobs, engine): pass
    monkeypatch.setattr(calibrated_batch,"CalibratedBatchGate",VerifiedGate)
    assert isinstance(operational_gate(artifact,jobs(),object()),VerifiedGate)


@pytest.mark.parametrize("action,capable,expected_skipped", [
    ("ACCEPT_REDUCED", True, 1), ("RUN_FULL", True, 0), ("ACCEPT_REDUCED", False, 0)])
def test_operational_routing_uses_only_authorized_calibrator_decision(
        tmp_path, monkeypatch, action, capable, expected_skipped):
    from docking_confidence.application import operational
    gate=FullFallbackGate("test_verified_decision")
    gate.evaluate=lambda **kwargs: {"action":action,"decision_capable":capable,"reason":"test_verified_decision"}
    monkeypatch.setattr(operational,"operational_gate",lambda *args:gate)
    source=tmp_path / "input";engine=_RecordingEngine()
    summary=run_batch(source,tmp_path / "output",jobs=jobs(),engine=engine,report=_make_report(source,count=1))
    assert summary.skipped_calibrator == expected_skipped
    assert [j.protocol.value for j,_,_ in engine.calls] == (["reduced"] if expected_skipped else ["reduced","full"])


def test_training_cannot_mix_budget_regimes():
    from docking_confidence.io.config import load_config
    plan={"execution_strategy":"operational_cascade_v1","jobs":[_job_snapshot(j) for j in jobs()]}
    cfg=load_config(Path("configs/local_refinement_operational.yaml"))
    validate_training_regime(cfg,plan)
    with pytest.raises(ValueError,match="separate matching"):
        validate_training_regime({},plan)
    with pytest.raises(ValueError,match="separate matching"):
        validate_training_regime(cfg,{"jobs":plan["jobs"]})
    altered=deepcopy(plan);altered["jobs"][0]["profile"]["evaluation_budget"]=7600
    with pytest.raises(ValueError,match="75%"):
        validate_training_regime(cfg,altered)


def test_failed_reduced_work_is_not_dropped_from_evaluation_audit():
    from scripts.run_operational_pilot import evaluation_count
    assert evaluation_count({"status":"REDUCED_FAILED","total_n_eval":5700}, {}) == 5700
    assert evaluation_count({"population_confidence_by_pocket":[{"n_eval_actual":5700}]}, {}) == 5700
    assert evaluation_count({}, {}) is None


def test_article_collection_cannot_skip_full_even_with_an_installed_model(tmp_path,monkeypatch):
    from docking_confidence.application import operational
    def forbidden(*args):
        pytest.fail("Article collection must not consult an installed calibrator")
    monkeypatch.setattr(operational,"operational_gate",forbidden)
    source=tmp_path/"input";engine=_RecordingEngine()
    summary=run_batch(source,tmp_path/"output",jobs=jobs(),engine=engine,
        report=_make_report(source,count=1),collect_operational_pairs=True)
    assert summary.succeeded==2 and summary.skipped_calibrator==0
    assert [j.protocol.value for j,_,_ in engine.calls]==["reduced","full"]
    decision=json.loads(next((tmp_path/"output").rglob("confidence_decision.json")).read_text())
    assert decision["reason"]=="article_paired_data_collection"


def test_operational_ui_is_available_without_changing_scientific_mode():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()").run(timeout=30)
    app.selectbox(key="ppi4dock_real_protocol").set_value("operational_cascade").run(timeout=30)
    assert not app.exception
    assert any("5700" in item.value and "7600" in item.value for item in app.info)
    app.selectbox(key="ppi4dock_real_protocol").set_value("factorial_2x2").run(timeout=30)
    assert not app.exception
    assert any("quatro condições" in item.value for item in app.info)
