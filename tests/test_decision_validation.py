"""Leakage, provenance and selection tests using synthetic docking fixtures.

These tests verify software contracts; synthetic outcomes are not scientific
evidence that any biological acceptance policy is safe or useful.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from docking_confidence.application import decision_validation as service
from docking_confidence.application.generalized_calibrator import create_calibrator, load_calibrator
from docking_confidence.io.config import config_sha256
from docking_confidence.io.quality_context import quality_metric_sha256
from test_authorized_path import _graph_metadata
from test_generalized_creation import _dataset
from standard_quality_fixtures import bind_standard_quality_records


QUALITY = {"schema_version": "structural_quality_metric_v1",
    "method": "official_dockq_v2_protein_protein", "implementation": "DockQ",
    "implementation_version": "2.1.3", "chain_mapping": "explicit_receptor_ligand",
    "capri_peptide": False, "contact_cutoff_angstrom": 5, "interface_cutoff_angstrom": 10}
EXECUTION = {"schema_version": "operational_cascade_v1", "reduced_evaluation_budget": 5700,
    "full_evaluation_budget": 7600, "population": 100, "initialization": "uniform",
    "reduced_search_scale": .5, "max_translation": 5., "max_rotation_degrees": 25.}


@pytest.fixture(autouse=True)
def isolated_global_exposure_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(service, "_EXPOSURE_ROOT", tmp_path / "exposure_ledger")


@pytest.fixture(scope="module")
def frozen_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("decision_real_model")
    dataset, config, frame, config_file = _dataset(root)
    config["quality_metric"] = deepcopy(QUALITY)
    config["execution_contract"] = deepcopy(EXECUTION)
    config["authorization"]["trusted_engine_allowlist"] = ["validated-test-engine"]
    frame["config_sha256"] = config_sha256(config)
    frame["quality_metric_id"] = QUALITY["method"]
    frame["quality_metric_sha256"] = quality_metric_sha256(QUALITY)
    frame["execution_contract_sha256"] = service.contract_sha256(EXECUTION)
    frame["confidence_overhead_sec"] = .01
    config_file.write_text(json.dumps(config), encoding="utf-8")
    frame = bind_standard_quality_records(frame, root, QUALITY)
    frame.to_csv(dataset, index=False)
    result = create_calibrator(dataset, config, root / "model")
    assert result["artifact"]["settings"]["quality_metric"] == QUALITY
    return {"root": root, "config": config, "config_path": config_file,
            "frame": frame, "artifact": Path(result["paths"]["artifact.json"])}


def _role_frame(model, root, name, *, offset, n=20, good=None):
    rows = []
    for i in range(n):
        positive = i % 2 == 0 if good is None else bool(good[i])
        row = model["frame"].iloc[0 if positive else 1].copy()
        row["complex_id"] = f"{name}-{i}"
        row["run_id"] = f"ag-operational-reduced-seed1-{name}-{i}"
        row["algorithm"], row["seed"] = "ag", 1
        row["receptor_sequence"] = "A" * (offset + i + 1) + str(row["receptor_sequence"])
        for key, value in _graph_metadata(row, model["config"]["contact_graph"]).items():
            row[key] = value
        row["graph_receptor_residue_count"] = len(row["receptor_sequence"])
        row["graph_ligand_residue_count"] = len(row["ligand_sequence"])
        row["graph_connected_component_count"] = row["graph_receptor_residue_count"] + row["graph_ligand_residue_count"]
        rows.append(row)
    frame = pd.DataFrame(rows).reset_index(drop=True)
    # Separate roots prevent validation and test fixtures from overwriting
    # one another's signed-by-hash synthetic evaluator records.
    record_root = root / f"quality_{name}"
    frame = bind_standard_quality_records(frame, record_root, QUALITY)
    for mode in ("reduced", "full"):
        field = f"quality_{mode}_record_path"
        frame[field] = frame[field].map(lambda value: str(Path(record_root.name) / value))
    path = root / f"{name}.csv"
    frame.to_csv(path, index=False)
    return frame, path


def _prepare(model, root, monkeypatch, *, val=None, test=None, **kwargs):
    monkeypatch.setattr(service, "_REGISTRY", root / "absent_registry.json")
    if val is None:
        val, _ = _role_frame(model, root, "validation", offset=100)
    if test is None:
        test, _ = _role_frame(model, root, "test", offset=200)
    defaults = {"config_path": model["config_path"], "algorithms": ["ag"], "seeds": [1],
        "candidate_thresholds": [-.95, -.2, 0., .5, .77], "minimum_coverage": .2,
        "minimum_accepted_complexes": 2, "bootstrap_repetitions": 50}
    defaults.update(kwargs)
    return service.prepare_decision_protocol(model["artifact"],
        list(val.complex_fingerprint_sha256.unique()), list(test.complex_fingerprint_sha256.unique()),
        root / "protocol", **defaults)


def test_protocol_is_prepared_from_identities_without_reading_outcomes(frozen_model, tmp_path, monkeypatch):
    prepared = _prepare(frozen_model, tmp_path, monkeypatch)
    p = prepared["protocol"]
    assert p["status"] == "prepared" and not p["decision_capable"]
    assert len(p["planned_runs"]["test"]) == 20
    assert p["criteria"]["preservation_comparison"] == "strictly_greater_than"
    assert not p["prospective_context_issues"]
    assert "outcomes" not in p and "dataset_sha256" not in p
    assert Path(prepared["protocol_path"]).is_file()
    with pytest.raises(FileExistsError):
        _prepare(frozen_model, tmp_path, monkeypatch)


def test_fitting_calibration_and_role_overlap_are_rejected(frozen_model, tmp_path, monkeypatch):
    val, _ = _role_frame(frozen_model, tmp_path, "val", offset=100)
    with pytest.raises(ValueError, match="validation.*test.*overlap"):
        _prepare(frozen_model, tmp_path, monkeypatch, val=val, test=val)
    fit = load_calibrator(frozen_model["artifact"])["artifact"]["partition"]["proper_training"]["groups"][0]
    val.loc[0, "complex_fingerprint_sha256"] = fit
    with pytest.raises(ValueError, match="fitting or calibration"):
        _prepare(frozen_model, tmp_path, monkeypatch, val=val)


def test_known_registry_exposure_cannot_be_omitted_by_caller(frozen_model, tmp_path, monkeypatch):
    val, _ = _role_frame(frozen_model, tmp_path, "val", offset=100)
    test, _ = _role_frame(frozen_model, tmp_path, "test", offset=200)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"known_fingerprints": [test.iloc[0].complex_fingerprint_sha256]}))
    monkeypatch.setattr(service, "_REGISTRY", registry)
    with pytest.raises(ValueError, match="Previously exposed"):
        service.prepare_decision_protocol(frozen_model["artifact"],list(val.complex_fingerprint_sha256),
            list(test.complex_fingerprint_sha256),tmp_path / "protocol",config_path=frozen_model["config_path"])
    p = service.prepare_decision_protocol(frozen_model["artifact"],list(val.complex_fingerprint_sha256),
        list(test.complex_fingerprint_sha256),tmp_path / "exploratory",config_path=frozen_model["config_path"],
        analysis_role="exploratory")["protocol"]
    assert p["exposure_registry"]["sha256"] == service.sha256_bytes(registry.read_bytes())
    assert p["exposed_role_groups"]["test"] == [test.iloc[0].complex_fingerprint_sha256]


def test_criterion_and_configuration_cannot_change_after_commit(frozen_model, tmp_path, monkeypatch):
    prepared = _prepare(frozen_model, tmp_path, monkeypatch)
    path = Path(prepared["protocol_path"])
    value = json.loads(path.read_bytes())
    value["criteria"]["minimum_preservation"] = .5
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="integrity"):
        service.select_decision_policy(path,tmp_path / "validation.csv",tmp_path / "selection")


def test_selection_and_final_test_use_disjoint_frozen_roles(frozen_model, tmp_path, monkeypatch):
    prepared = _prepare(frozen_model, tmp_path, monkeypatch)
    selection = service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    assert selection["policy"]["status"] == "selected"
    assert selection["policy"]["decision_capable"] is False
    assert selection["report"]["difference"]["selective"]
    assert selection["report"]["difference"]["preservation_gain"] > 0
    threshold = selection["policy"]["threshold"]
    with pytest.raises(ValueError, match="outside.*test"):
        service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "validation.csv",tmp_path / "wrong_test")
    evaluated = service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    assert evaluated["report"]["status"] == "passed"
    assert evaluated["report"]["decision_capable"]
    assert evaluated["report"]["selected"]["preservation"] == 1.
    assert json.loads(Path(selection["policy_path"]).read_bytes())["threshold"] == threshold
    assert "not a risk guarantee" in evaluated["report"]["uncertainty"]["interpretation"]
    with pytest.raises(FileExistsError, match="consumed"):
        service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "another_test")
    with pytest.raises(FileExistsError, match="already frozen"):
        service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "another_policy")


def test_all_positive_flat_score_cannot_claim_improvement(frozen_model, tmp_path, monkeypatch):
    prepared = _prepare(frozen_model, tmp_path, monkeypatch)
    real_predict = service.predict_calibrator
    def flat(*args, **kwargs):
        result = real_predict(*args, **kwargs)
        result["prediction"] = .2
        result.to_csv(args[2],index=False)
        return result
    monkeypatch.setattr(service,"predict_calibrator",flat)
    result = service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    assert result["policy"]["status"] == "no_useful_rule"
    assert result["policy"]["threshold"] is None
    assert result["report"]["eligible_candidate_count"] == 0
    assert result["report"]["selected"]["preservation"] is None
    assert result["report"]["selected"]["risk"] is None
    assert not result["report"]["decision_capable"]


def test_more_than_ninety_is_strict_and_always_accept_is_not_a_success(frozen_model, tmp_path, monkeypatch):
    prepared = _prepare(frozen_model, tmp_path, monkeypatch)
    p = prepared["protocol"]
    selected={"preservation": .9,"coverage":1.,"accepted_complexes":20,
              "runtime_complete":False,"runtime_total_sec":None,"full_baseline_runtime_sec":None}
    result = service._criteria(p,selected,{}, {"selective":True,"preservation_gain":.05})
    assert result["preservation_met"] is False
    selected["preservation"] = .95
    result = service._criteria(p,selected,{}, {"selective":False,"preservation_gain":0.})
    assert result["preservation_met"] is True
    assert not service._all_criteria_met(result)


def test_planned_missing_runs_remain_in_denominator(frozen_model, tmp_path, monkeypatch):
    prepared = _prepare(frozen_model, tmp_path, monkeypatch)
    frame=pd.read_csv(tmp_path / "validation.csv")
    frame.iloc[:-1].to_csv(tmp_path / "partial.csv",index=False)
    result=service.select_decision_policy(prepared["protocol_path"],tmp_path / "partial.csv",tmp_path / "selection")
    baseline=result["report"]["baseline"]
    assert baseline["planned_runs"] == 20
    assert baseline["valid_runs"] == 19
    assert baseline["failed_or_missing_runs"] == 1
    assert baseline["coverage"] == pytest.approx(19/20)
    assert not baseline["runtime_complete"]


def test_missing_quality_or_budget_metadata_is_rejected(frozen_model, tmp_path, monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    frame=pd.read_csv(tmp_path / "validation.csv")
    for column in ["quality_metric_sha256","execution_contract_sha256"]:
        invalid=tmp_path/f"without_{column}.csv"
        frame.drop(columns=[column]).to_csv(invalid,index=False)
        with pytest.raises(ValueError,match=column):
            service.select_decision_policy(prepared["protocol_path"],invalid,tmp_path/f"selection_{column}")


def test_unplanned_seeds_and_duplicate_planned_identity_are_rejected(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    frame=pd.read_csv(tmp_path / "validation.csv")
    invalid=pd.concat([frame,frame.iloc[[0]].assign(run_id="duplicate")],ignore_index=True)
    for i in invalid.index:
        for key,value in _graph_metadata(invalid.loc[i],frozen_model["config"]["contact_graph"]).items():
            invalid.loc[i,key]=value
    invalid.to_csv(tmp_path / "duplicate.csv",index=False)
    with pytest.raises(ValueError,match="repeats a planned"):
        service.select_decision_policy(prepared["protocol_path"],tmp_path / "duplicate.csv",tmp_path / "selection")


def test_exploratory_results_never_authorize_inference(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch,analysis_role="exploratory")
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    evaluated=service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    assert evaluated["report"]["empirical_criteria_met"]
    assert evaluated["report"]["status"] == "exploratory"
    assert not evaluated["report"]["decision_capable"]


def test_zero_acceptance_is_unavailable_risk_not_zero(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch,candidate_thresholds=[100.])
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    evaluated=service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    selected=evaluated["report"]["selected"]
    assert selected["accepted_runs"] == 0
    assert selected["risk"] is None and selected["preservation"] is None
    assert not evaluated["report"]["decision_capable"]


def test_empty_paired_test_retains_all_planned_failures(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    pd.read_csv(tmp_path / "test.csv").iloc[:0].to_csv(tmp_path / "empty.csv",index=False)
    evaluated=service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "empty.csv",tmp_path / "final")
    assert evaluated["report"]["selected"]["planned_runs"] == 20
    assert evaluated["report"]["selected"]["failed_or_missing_runs"] == 20
    assert evaluated["report"]["selected"]["preservation"] is None
    assert not evaluated["report"]["decision_capable"]


def test_reduced_only_inference_requires_linked_passing_test(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    evaluated=service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    frame,_=_role_frame(frozen_model,tmp_path,"new",offset=300,n=4)
    paired_only=["dockq_reduced","dockq_full","runtime_full_sec","engine_full","scientific_full",
                 "quality_metric_id","quality_metric_sha256"]
    reduced=tmp_path / "reduced.csv"
    frame.drop(columns=paired_only).to_csv(reduced,index=False)
    unsupported=service.predict_decisions(selection["policy_path"],reduced,tmp_path / "no_approval.csv")
    assert set(unsupported.action)=={"RUN_FULL"}
    supported=service.predict_decisions(selection["policy_path"],reduced,tmp_path / "approved.csv",
                                       evaluation_path=evaluated["report_path"])
    assert "ACCEPT_REDUCED" in set(supported.action)
    assert "RUN_FULL" in set(supported.action)
    assert "dockq_full" not in supported and "dockq_reduced" not in supported
    assert supported.decision_capable.all()


def test_inference_never_accepts_development_or_test_complexes(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    evaluated=service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    result=service.predict_decisions(selection["policy_path"],tmp_path / "test.csv",tmp_path / "known.csv",
                                     evaluation_path=evaluated["report_path"])
    assert set(result.action)=={"RUN_FULL"}
    assert result.decision_reason.str.contains("not_eligible").all()


def test_changed_dataset_snapshot_and_plan_cannot_be_used(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="SHA-256"):
        service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection",
                                       expected_dataset_sha256="0"*64)
    plan=deepcopy(prepared["protocol"]["planned_runs"]["validation"])
    plan.pop()
    with pytest.raises(ValueError,match="cannot change|Every planned"):
        service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection",
                                       planned_runs=plan)


def test_runtime_gate_requires_measured_overhead_and_complete_denominators(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch,require_runtime_improvement=True)
    frame=pd.read_csv(tmp_path / "validation.csv").drop(columns=["confidence_overhead_sec"])
    frame.to_csv(tmp_path / "no_overhead.csv",index=False)
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "no_overhead.csv",tmp_path / "selection")
    assert selection["policy"]["status"]=="no_useful_rule"
    assert selection["report"]["selected"]["runtime_total_sec"] is None
    assert selection["report"]["selected"]["evaluation_collection_saved_full_runs"] == 0
    assert "counterfactual" in selection["report"]["selected"]["runtime_estimation"]


def test_official_quality_evidence_is_verified_and_remains_bound(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    assert selection["policy"]["validation_source"]["quality_records_verification"]["records"] == 40
    frame=pd.read_csv(tmp_path / "validation.csv")
    record=tmp_path / frame.iloc[0].quality_reduced_record_path
    record.write_text(record.read_text(encoding="utf-8")+" ",encoding="utf-8")
    with pytest.raises(ValueError,match="Quality record file SHA"):
        service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")


def test_exposed_roles_cannot_be_redeclared_in_a_new_prospective_protocol(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    # Existing commitments created before the ledger upgrade remain readable;
    # their use must still append exposure for future preparations.
    old_protocol=deepcopy(prepared["protocol"])
    old_protocol.pop("exposure_ledger")
    old_protocol.pop("record_sha256")
    old_protocol=service._record(old_protocol)
    Path(prepared["protocol_path"]).write_text(json.dumps(old_protocol),encoding="utf-8")
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    exposed,snapshot=service._exposure_snapshot()
    assert len(exposed)==20 and len(snapshot["records"])==1
    validation=pd.read_csv(tmp_path / "validation.csv")
    test=pd.read_csv(tmp_path / "test.csv")
    with pytest.raises(ValueError,match="Previously exposed"):
        _prepare(frozen_model,tmp_path / "again",monkeypatch,val=validation,test=test)
    exploratory=_prepare(frozen_model,tmp_path / "exploratory",monkeypatch,val=validation,test=test,
                         analysis_role="exploratory")["protocol"]
    assert exploratory["exposure_ledger"]["snapshot_sha256"]==snapshot["snapshot_sha256"]
    assert len(exploratory["exposed_role_groups"]["validation"])==20
    # The existing policy remains usable as the ledger advances to its test role.
    final=service.evaluate_frozen_policy(selection["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    assert final["report"]["status"]=="passed"
    assert len(service._exposure_snapshot()[0])==40


def test_exposure_is_committed_before_inference_even_if_inference_fails(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    partial=pd.read_csv(tmp_path / "validation.csv").iloc[:-1]
    partial.to_csv(tmp_path / "partial.csv",index=False)
    def failed_inference(*args,**kwargs):
        observed,_=service._exposure_snapshot()
        assert observed==set(prepared["protocol"]["roles"]["validation_groups"])
        raise RuntimeError("synthetic inference failure after exposure commitment")
    monkeypatch.setattr(service,"predict_calibrator",failed_inference)
    with pytest.raises(RuntimeError,match="after exposure commitment"):
        service.select_decision_policy(prepared["protocol_path"],tmp_path / "partial.csv",tmp_path / "selection")
    assert len(service._exposure_snapshot()[0])==20


def test_prepared_protocol_cannot_consume_groups_exposed_later_by_another_protocol(frozen_model,tmp_path,monkeypatch):
    first=_prepare(frozen_model,tmp_path,monkeypatch)
    validation=pd.read_csv(tmp_path / "validation.csv")
    test=pd.read_csv(tmp_path / "test.csv")
    second=_prepare(frozen_model,tmp_path / "second",monkeypatch,val=validation,test=test)
    service.select_decision_policy(first["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    with pytest.raises(ValueError,match="already exposed by another decision protocol"):
        service.select_decision_policy(second["protocol_path"],tmp_path / "validation.csv",tmp_path / "second_selection")
    assert not (tmp_path / "second_selection/decision_policy.json").exists()


def test_global_commit_lock_prevents_parallel_protocol_reuse(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    barrier=Barrier(2)
    group="a"*64
    def consume(index):
        protocol={"record_sha256": str(index)*64,"analysis_role":"prospective",
                  "roles":{"validation_groups":[group]}}
        barrier.wait()
        try:
            service._commit_exposure(protocol,"validation","b"*64)
            return "committed"
        except ValueError:
            return "refused"
    with ThreadPoolExecutor(max_workers=2) as executor:
        results=list(executor.map(consume,[1,2]))
    assert sorted(results)==["committed","refused"]
    assert len(service._exposure_snapshot()[1]["records"])==1


def test_raw_margin_is_a_predeclared_score_option(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch,score_column="base_raw_score")
    selection=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    assert selection["policy"]["score_column"]=="base_raw_score"
    assert selection["policy"]["threshold"] in prepared["protocol"]["candidate_thresholds"]


def test_invalid_quality_binding_cannot_be_declared_prospective(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    p=prepared["protocol"]
    artifact=load_calibrator(frozen_model["artifact"])["artifact"]
    config=deepcopy(frozen_model["config"])
    config["quality_metric"]["contact_cutoff_angstrom"]=5.0
    artifact["settings"]["quality_metric"]=config["quality_metric"]
    assert "unsupported_or_incomplete_standard_quality_metric" in service._prospective_context_issues(artifact,config)


def test_changed_validation_evidence_invalidates_frozen_policy(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    selected=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    snapshot=Path(selected["policy"]["validation_source"]["snapshot_path"])
    snapshot.write_bytes(snapshot.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="evidence snapshot changed"):
        service.evaluate_frozen_policy(selected["policy_path"],tmp_path / "test.csv",tmp_path / "final")


def test_passing_approval_cannot_be_reused_after_test_evidence_changes(frozen_model,tmp_path,monkeypatch):
    prepared=_prepare(frozen_model,tmp_path,monkeypatch)
    selected=service.select_decision_policy(prepared["protocol_path"],tmp_path / "validation.csv",tmp_path / "selection")
    evaluated=service.evaluate_frozen_policy(selected["policy_path"],tmp_path / "test.csv",tmp_path / "final")
    assert evaluated["report"]["prediction_diagnostics"]["roc_auc"] is not None
    decisions=pd.read_csv(evaluated["predictions_path"])
    assert "observed_margin" in decisions
    evidence=Path(evaluated["report"]["source"]["predictions_path"])
    evidence.write_bytes(evidence.read_bytes()+b"\n")
    _,reduced=_role_frame(frozen_model,tmp_path,"new",offset=300,n=4)
    with pytest.raises(ValueError,match="evidence snapshot changed"):
        service.predict_decisions(selected["policy_path"],reduced,tmp_path / "approved.csv",evaluation_path=evaluated["report_path"])


def test_model_and_configuration_context_remain_bound(frozen_model,tmp_path,monkeypatch):
    wrong=deepcopy(frozen_model["config"])
    wrong["execution_contract"]["reduced_evaluation_budget"]=7600
    wrong_path=tmp_path / "wrong_config.json"
    wrong_path.write_text(json.dumps(wrong))
    with pytest.raises(ValueError,match="Configuration differs"):
        _prepare(frozen_model,tmp_path,monkeypatch,config_path=wrong_path)


