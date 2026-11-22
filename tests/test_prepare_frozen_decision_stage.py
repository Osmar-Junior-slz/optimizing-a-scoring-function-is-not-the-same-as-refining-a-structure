"""Stage integration with persisted fixture poses and a simulated quality backend."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from docking_confidence.application.batch import _job_snapshot, run_batch
from docking_confidence.application.operational import load_operational_config
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, ProtocolMode
from docking_confidence.evaluation import official_dockq
from docking_confidence.io.batch_discovery import discover_ppi4dock
from docking_confidence.io.config import config_sha256, load_config, validate_config
from test_training_dataset import _FixturePoseEngine, _component


_SPEC = importlib.util.spec_from_file_location("frozen_stage", Path(__file__).resolve().parents[1] / "scripts/prepare_frozen_decision_stage.py")
stage = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stage)


def _save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def _record(value):
    result = deepcopy(value)
    result["record_sha256"] = stage.contract_sha256(result)
    return result


@pytest.fixture
def completed(tmp_path, monkeypatch, request):
    parameters = getattr(request, "param", {})
    run_role = parameters.get("role", "validation")
    source = tmp_path / "inputs"
    for index, residue in enumerate(("GLY", "SER")):
        directory = source / f"{index + 1}abc_AB"
        directory.mkdir(parents=True)
        receptor = _component("A", "ALA", 0., 1)
        ligand = _component("B", residue, 3., 21)
        (directory / "A_model_st.pdb").write_text(receptor + "END\n")
        (directory / "B_model_st.pdb").write_text(ligand + "END\n")
        (directory / f"{directory.name}_st.pdb").write_text(receptor + ligand + "END\n")
    discovery = discover_ppi4dock(source)
    targets = []
    for role, target in zip(("validation", "test"), discovery.targets):
        targets.append({"role": role, "complex_id": target.complex_id, "path": str(target.directory),
            **{key: str(getattr(target, key)) for key in ("receptor_path", "ligand_path", "reference_path")},
            **{key: getattr(target, key) for key in ("complex_fingerprint_sha256", "receptor_sha256", "ligand_sha256", "reference_sha256")}})
    selection = _record({"schema_version": stage.collection.SELECTION_SCHEMA, "input_root": str(source),
        "targets": targets, "algorithms": ["ag", "pso", "hybrid"], "seeds": [1]})
    selection_path = tmp_path / "selection.json"
    _save(selection_path, selection)
    source_config_path = stage.ROOT / "configs/local_refinement_operational_candidate.yaml"
    source_config = load_config(source_config_path)
    standard_config = validate_config({**deepcopy(source_config), "quality_metric": official_dockq.quality_metric_descriptor("2.1.3")})
    protocol = {"schema_version": stage.collection.PROTOCOL_SCHEMA,
        "model": {"artifact_path": str(tmp_path / "model" / "artifact.json"),
            "config_sha256": config_sha256(standard_config), "source_config": standard_config},
        "roles": {role + "_groups": [target["complex_fingerprint_sha256"]]
                  for role, target in zip(("validation", "test"), targets)},
        "algorithms": ["ag", "pso", "hybrid"], "seeds": [1]}
    protocol["planned_runs"] = {role: [{"complex_fingerprint_sha256": target["complex_fingerprint_sha256"],
        "algorithm": algorithm, "seed": 1} for algorithm in protocol["algorithms"]]
        for role, target in zip(("validation", "test"), targets)}
    protocol = _record(protocol)
    protocol_path = tmp_path / "protocol.json"
    _save(protocol_path, protocol)
    record_path = protocol_path
    policy = None
    if run_role == "test":
        policy = _record({"status": "selected", "threshold": 0.})
        record_path = tmp_path / "policy.json"
        _save(record_path, policy)
    context = {"protocol": protocol, "policy": policy, "record_path": record_path,
               "record_file_sha256": stage.collection.file_sha256(record_path)}
    monkeypatch.setattr(stage.collection, "frozen_context", lambda *a: context)
    jobs = build_execution_plan(algorithms=[Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID],
        protocol_mode=ProtocolMode.OPERATIONAL, base_config=load_operational_config(), seeds=[1])
    study = tmp_path / "study"
    batch = study / "shard_01" / "batch"
    index = 0 if run_role == "validation" else 1
    run_batch(source, batch, jobs=jobs, engine=_FixturePoseEngine(set(parameters.get("failures", ()))),
        report=replace(discovery, targets=(discovery.targets[index],)), collect_operational_pairs=True)
    manifest = _record({"schema_version": stage.collection.RUN_SCHEMA, "role": run_role,
        "stage_record_path": str(record_path), "stage_file_sha256": context["record_file_sha256"],
        "stage_record_sha256": (policy or protocol)["record_sha256"], "protocol_sha256": protocol["record_sha256"],
        "selection_path": str(selection_path), "selection_file_sha256": stage.collection.file_sha256(selection_path),
        "selection_record_sha256": selection["record_sha256"], "automatic_skipping": False,
        "partitions": [[targets[index]["complex_id"]]] + [[] for _ in range(7)],
        "jobs": [_job_snapshot(job) for job in jobs], "planned_jobs": 6})
    _save(study / "study_manifest.json", manifest)
    def backend(request, **runtime):
        return {"available": True, "implementation": "DockQ", "implementation_version": "2.1.3",
            "implementation_sha256": "a" * 64, "constants_sha256": "b" * 64,
            "python_executable": "fixture-isolated-python",
            "metrics": {"DockQ": .8, "fnat": .7, "iRMSD": .2, "LRMSD": .3}}
    monkeypatch.setattr(official_dockq, "_invoke_backend", backend)
    args = SimpleNamespace(study_dir=str(study), role=run_role,
        protocol=str(protocol_path) if run_role == "validation" else None,
        policy=str(record_path) if run_role == "test" else None,
        source_config=str(source_config_path), out_dir=str(tmp_path / "stage"), check_only=False,
        python_executable="fixture-python", dependency_path="fixture-runtime", workers=4)
    return SimpleNamespace(args=args, batch=batch, context=context, manifest=manifest, protocol=protocol)


def test_prepares_real_operational_receipts_without_calling_fallback_time_gva_overhead(completed, monkeypatch):
    calls = []
    receipt_bytes = {path: path.read_bytes() for path in completed.batch.glob("targets/*/jobs/*/receipt.json")}
    def select(protocol_path, dataset_path, output, **kwargs):
        calls.append((protocol_path, dataset_path, output, kwargs))
        frame = pd.read_csv(dataset_path)
        assert len(frame) == 3
        assert "confidence_overhead_sec" not in frame
        assert "historical_confidence_overhead_sec" in frame
        assert frame.dockq_reduced.tolist() == [.8] * 3
        assert frame.config_sha256.eq(completed.protocol["model"]["config_sha256"]).all()
        report = {"status": "no_useful_rule", "decision_capable": False, "selected": {},
            "baseline": {}, "difference": {}, "criteria": {}}
        return {"report": report, "report_path": str(output / "report.json"), "policy_path": str(output / "policy.json")}
    monkeypatch.setattr(stage, "select_decision_policy", select)
    monkeypatch.setattr(stage, "evaluate_frozen_policy", lambda *a, **k: pytest.fail("validation must not evaluate test"))
    result = stage.prepare_stage(completed.args)
    assert len(calls) == 1
    assert result["planned_runs"] == 3 and result["valid_paired_rows"] == 3
    assert result["decision_capable"] is False
    assert result["training_performed"] is False and result["test_collection_started"] is False
    assert all(path.read_bytes() == snapshot for path, snapshot in receipt_bytes.items())
    with pytest.raises(FileExistsError):
        stage.prepare_stage(completed.args)


def test_incomplete_shard_is_rejected_before_creating_stage_output(completed):
    path = completed.batch / "batch_status.json"
    status = json.loads(path.read_bytes())
    status["state"] = "RUNNING"
    _save(path, status)
    with pytest.raises(ValueError, match="Every planned job"):
        stage.prepare_stage(completed.args)
    assert not Path(completed.args.out_dir).exists()


def test_changed_role_is_rejected_before_quality_or_selection(completed):
    completed.args.role = "test"
    with pytest.raises(ValueError, match="not bound"):
        stage.prepare_stage(completed.args)
    assert not Path(completed.args.out_dir).exists()


def test_entirely_failed_shard_is_retained_but_corrupt_evidence_is_blocked():
    empty = {"counts": {"candidate_rows": 0}, "issues": [{"code": "official_loader", "message": "dataset is empty"}]}
    assert stage.empty_failed_preparation(empty)
    empty["issues"].append({"code": "receipt_provenance", "message": "changed receipt"})
    assert not stage.empty_failed_preparation(empty)


@pytest.mark.parametrize("completed", [{"failures": [("1abc_AB", "reduced"), ("1abc_AB", "full")]}], indirect=True)
def test_all_failed_planned_pairs_remain_zero_coverage_instead_of_disappearing(completed, monkeypatch):
    calls = []
    def select(protocol_path, dataset_path, output, **kwargs):
        calls.append(pd.read_csv(dataset_path))
        assert calls[0].empty
        return {"report": {"status": "no_useful_rule", "decision_capable": False, "selected": {},
            "baseline": {}, "difference": {}, "criteria": {}},
            "report_path": str(output / "report.json"), "policy_path": str(output / "policy.json")}
    monkeypatch.setattr(stage, "select_decision_policy", select)
    monkeypatch.setattr(stage, "build_standard_paired_dataset", lambda *a, **k: pytest.fail("no observed pose can be rescored"))
    result = stage.prepare_stage(completed.args)
    assert len(calls) == 1
    assert result["planned_runs"] == 3 and result["valid_paired_rows"] == 0
    assert result["failed_or_missing_pairs"] == 3
    merge = json.loads((Path(completed.args.out_dir) / "merge_report.json").read_bytes())
    assert len(merge["failures"]) == 6 and len(merge["unmatched"]) == 3


@pytest.mark.parametrize("completed", [{"role": "test"}], indirect=True)
def test_final_role_evaluates_frozen_policy_without_selecting_again(completed, monkeypatch):
    calls = []
    def evaluate(policy_path, dataset_path, output, **kwargs):
        calls.append(policy_path)
        assert policy_path == completed.args.policy
        assert len(pd.read_csv(dataset_path)) == 3
        return {"report": {"status": "failed", "decision_capable": False, "selected": {},
            "baseline": {}, "difference": {}, "criteria": {}}, "report_path": str(output / "report.json")}
    monkeypatch.setattr(stage, "evaluate_frozen_policy", evaluate)
    monkeypatch.setattr(stage, "select_decision_policy", lambda *a, **k: pytest.fail("final test cannot select a new rule"))
    result = stage.prepare_stage(completed.args)
    assert len(calls) == 1 and result["role"] == "test" and result["status"] == "failed"
