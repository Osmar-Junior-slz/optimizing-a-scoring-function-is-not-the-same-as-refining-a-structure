from dataclasses import replace
import json

import pytest

from docking_confidence.application.batch import BatchError, BatchPlanConflictError, BatchAlreadyRunningError, _BatchFileLock, run_batch
from docking_confidence.application.experimental_protocol import (
    ExperimentalProtocolPlanConflictError, load_experimental_protocol, prepare_experimental_protocol,
)
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, DockingTask, FactorialExperimentConfig, ProtocolMode
from test_batch_runner import _make_report, _RecordingEngine


def _jobs(**controls):
    experiment = FactorialExperimentConfig(evaluation_budget=20, pose_source="Synthetic supplied pose; reference only evaluated afterward", **controls)
    return build_execution_plan(algorithms=[Algorithm.AG, Algorithm.PSO, Algorithm.HYBRID],
        protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(generations=8, population=4, top_pockets=1, experiment=experiment), seeds=[1, 2])


def test_factorial_crosses_independent_controls_without_legacy_budget_reduction():
    jobs = _jobs()
    assert len(jobs) == 24
    assert len({job.job_id for job in jobs}) == 24
    assert {(job.profile.search_extent, job.profile.initialization) for job in jobs} == {
        ("reduced", "uniform"), ("reduced", "seeded"), ("expanded", "uniform"), ("expanded", "seeded")}
    assert {(job.profile.generations, job.profile.population, job.profile.evaluation_budget) for job in jobs} == {(8, 4, 20)}
    assert {job.profile.budget_policy for job in jobs} == {"fixed_evaluations"}
    assert all(job.protocol.value == ("reduced" if job.profile.search_extent == "reduced" else "full") for job in jobs)


@pytest.mark.parametrize("control", [{"evaluation_budget": 3}, {"reduced_search_scale": 1.0},
    {"reduced_search_scale": float("nan")}, {"max_translation": 0}, {"max_rotation_degrees": 181}, {"pose_source": ""}])
def test_invalid_controls_are_rejected_before_execution(control):
    options = dict(evaluation_budget=20, pose_source="documented input")
    options.update(control)
    with pytest.raises(ValueError):
        build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.FACTORIAL,
            base_config=BaseSearchConfig(population=4, experiment=FactorialExperimentConfig(**options)), seeds=[1])


def test_factorial_rejects_surface_task():
    with pytest.raises(ValueError, match="local_refinement"):
        build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.FACTORIAL,
            base_config=BaseSearchConfig(docking_task=DockingTask.SURFACE_DISCOVERY,
                experiment=FactorialExperimentConfig(pose_source="documented")), seeds=[1])


def test_prepare_executes_nothing_and_batch_runs_all_conditions_in_frozen_order(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 2)
    engine = _RecordingEngine()
    jobs = _jobs()
    output = tmp_path / "output"
    prepared = prepare_experimental_protocol(source, output, jobs=jobs, engine=engine, report=report)
    assert engine.calls == []
    assert prepared["protocol"]["analysis"]["primary_outcome"] == "valid_pose_dockq_ge_023"
    assert prepared["protocol"]["analysis"]["objective_evaluations_per_run"] == 20
    assert not (output / "batch_manifest.json").exists()
    assert prepare_experimental_protocol(source, output, jobs=jobs, engine=engine, report=report) == prepared
    summary = run_batch(source, output, jobs=jobs, engine=engine, report=report, experimental_protocol=prepared["path"])
    assert summary.succeeded == 48
    call_keys = [f"{receptor.parent.name}::{job.job_id}" for job, receptor, _ in engine.calls]
    assert call_keys == prepared["protocol"]["plan"]["execution_order"]
    assert not any(task["state"] == "SKIPPED_CALIBRATOR" for task in json.loads((output / "batch_status.json").read_text())["tasks"].values())
    resumed = run_batch(source, output, jobs=jobs, engine=engine, report=report, experimental_protocol=prepared["path"])
    assert resumed.attempted == 0 and resumed.skipped_existing == 48


def test_batch_records_under_budget_execution_as_failed_without_receipt(tmp_path):
    class UnderBudget(_RecordingEngine):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            return replace(result, actual_eval_count=result.evaluation_budget - 1)

    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    engine = UnderBudget()
    jobs = _jobs()[:4]
    output = tmp_path / "output"
    prepared = prepare_experimental_protocol(source, output, jobs=jobs, engine=engine, report=report)
    summary = run_batch(source, output, jobs=jobs, engine=engine, report=report, experimental_protocol=prepared["path"])
    assert summary.succeeded == 0 and summary.failed == 4
    assert not list(output.rglob("receipt.json"))


def test_unprepared_or_partial_factorial_cannot_run(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    engine = _RecordingEngine()
    with pytest.raises(BatchError, match="prepared"):
        run_batch(source, tmp_path / "out", jobs=_jobs(), engine=engine, report=report)
    with pytest.raises(BatchError, match="four"):
        prepare_experimental_protocol(source, tmp_path / "partial", jobs=_jobs()[:-1], engine=engine, report=report)
    assert engine.calls == []


def test_prepared_protocol_rejects_changed_conditions_and_modified_inputs(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    engine = _RecordingEngine()
    prepared = prepare_experimental_protocol(source, tmp_path / "out", jobs=_jobs(), engine=engine, report=report)
    with pytest.raises(BatchPlanConflictError, match="changed"):
        run_batch(source, tmp_path / "out", jobs=_jobs(reduced_search_scale=.25), engine=engine, report=report, experimental_protocol=prepared["path"])
    report.targets[0].ligand_path.write_text("changed input")
    with pytest.raises(BatchError, match="changed"):
        prepare_experimental_protocol(source, tmp_path / "out", jobs=_jobs(), engine=engine, report=report)
    assert engine.calls == []


def test_runtime_change_reports_exact_fields_and_preserves_saved_protocol(tmp_path):
    class RuntimeEngine(_RecordingEngine):
        def __init__(self, version, executable):
            super().__init__()
            self.version, self.executable = version, executable

        def audit_descriptor(self):
            return {"python_runtime_version": self.version, "python_executable": self.executable}

    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    old_engine = RuntimeEngine("3.11.9", "old/python.exe")
    new_engine = RuntimeEngine("3.12.4", "new/python.exe")
    output = tmp_path / "out"
    prepared = prepare_experimental_protocol(source, output, jobs=_jobs(), engine=old_engine, report=report)
    protocol_path = output / "experimental_protocol.json"
    original = protocol_path.read_bytes()
    with pytest.raises(ExperimentalProtocolPlanConflictError) as caught:
        prepare_experimental_protocol(source, output, jobs=_jobs(), engine=new_engine, report=report)
    differences = {item["field"]: item for item in caught.value.differences}
    assert set(differences) == {"engine.descriptor.python_runtime_version", "engine.descriptor.python_executable"}
    assert differences["engine.descriptor.python_runtime_version"]["saved"] == "3.11.9"
    assert differences["engine.descriptor.python_runtime_version"]["current"] == "3.12.4"
    assert protocol_path.read_bytes() == original
    assert load_experimental_protocol(protocol_path) == prepared["protocol"]
    new_output = tmp_path / "new_out"
    recovered = prepare_experimental_protocol(source, new_output, jobs=_jobs(), engine=new_engine, report=report)
    assert recovered["protocol"]["plan"]["jobs"] == prepared["protocol"]["plan"]["jobs"]
    assert recovered["protocol"]["plan"]["execution_order"] == prepared["protocol"]["plan"]["execution_order"]
    assert old_engine.calls == new_engine.calls == []


def test_modified_protocol_is_rejected_before_engine_call(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    engine = _RecordingEngine()
    prepared = prepare_experimental_protocol(source, tmp_path / "out", jobs=_jobs(), engine=engine, report=report)
    path = tmp_path / "out" / "experimental_protocol.json"
    payload = json.loads(path.read_text())
    payload["analysis"]["primary_outcome"] = "changed_after_preparation"
    path.write_text(json.dumps(payload))
    with pytest.raises(BatchPlanConflictError, match="integrity"):
        run_batch(source, tmp_path / "out", jobs=_jobs(), engine=engine, report=report, experimental_protocol=prepared["path"])
    assert engine.calls == []


def test_preparation_uses_execution_lock_and_does_not_publish_while_worker_owns_it(tmp_path):
    source = tmp_path / "input"
    source.mkdir()
    report = _make_report(source, 1)
    output = tmp_path / "output"
    output.mkdir()
    with _BatchFileLock(output / "batch.lock"):
        with pytest.raises(BatchAlreadyRunningError):
            prepare_experimental_protocol(source, output, jobs=_jobs(), engine=_RecordingEngine(), report=report)
    assert not (output / "experimental_protocol.json").exists()


def test_legacy_resolver_cannot_discard_experimental_controls():
    from docking_confidence.application.protocols import ProtocolResolver
    with pytest.raises(ValueError, match="resolve_condition"):
        ProtocolResolver().resolve(ProtocolMode.FULL, BaseSearchConfig(experiment=FactorialExperimentConfig(pose_source="documented")))
