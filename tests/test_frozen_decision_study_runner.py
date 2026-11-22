"""Collection guard tests; mocked docking is not biological validation evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from docking_confidence.domain.mobile_partner import DEFAULT_MOBILE_PARTNER_DOMAIN
from docking_confidence.io.batch_discovery import BatchDiscoveryReport, PPI4DockTarget


_SPEC = importlib.util.spec_from_file_location("frozen_study_runner", Path(__file__).resolve().parents[1] / "scripts/run_frozen_decision_study.py")
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)

QUALITY = {"schema_version": "structural_quality_metric_v1", "method": "official_dockq_v2_protein_protein",
    "implementation": "DockQ", "implementation_version": "2.1.3", "chain_mapping": "explicit_receptor_ligand",
    "capri_peptide": False, "contact_cutoff_angstrom": 5, "interface_cutoff_angstrom": 10}
EXECUTION = {"schema_version": "operational_cascade_v1", "reduced_evaluation_budget": 5700,
    "full_evaluation_budget": 7600, "population": 100, "initialization": "uniform",
    "reduced_search_scale": .5, "max_translation": 5., "max_rotation_degrees": 25.}


def _record(value):
    result = deepcopy(value)
    result["record_sha256"] = runner.contract_sha256(result)
    return result


def _save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "inputs"
    source.mkdir()
    targets, selected = [], []
    for index, role in enumerate(("validation", "test")):
        complex_id = f"{index + 1}abc_AB"
        directory = source / complex_id
        directory.mkdir()
        paths, hashes = {}, {}
        for component in ("receptor", "ligand", "reference"):
            path = directory / f"{component}.pdb"
            path.write_bytes(f"immutable {role} {component}".encode())
            paths[component + "_path"] = path
            hashes[component + "_sha256"] = runner.file_sha256(path)
        fingerprint = hashlib.sha256(role.encode()).hexdigest()
        target = PPI4DockTarget(complex_id=complex_id, directory=directory,
            receptor_chain="A", ligand_chain="B", valid=True, validation_performed=True,
            complex_fingerprint_sha256=fingerprint, receptor_atom_count=10, ligand_atom_count=10,
            eligible=True, **paths, **hashes)
        targets.append(target)
        selected.append({"role": role, "complex_id": complex_id, "path": str(directory),
            "fingerprint": fingerprint, "complex_fingerprint_sha256": fingerprint,
            **{key: str(value) for key, value in paths.items()}, **hashes})
    report = BatchDiscoveryReport(source, tuple(targets), (), True, DEFAULT_MOBILE_PARTNER_DOMAIN)
    monkeypatch.setattr(runner, "discover_ppi4dock", lambda *a, **k: report)
    monkeypatch.setattr(runner, "_validate_target_contents", lambda target, **kwargs: target)
    protocol = {"schema_version": runner.PROTOCOL_SCHEMA, "analysis_role": "prospective",
        "prospective_context_issues": [], "roles": {role + "_groups": [item["fingerprint"]]
            for role, item in zip(("validation", "test"), selected)},
        "algorithms": ["ag", "pso", "hybrid"], "seeds": [1],
        "model": {"artifact_path": str(tmp_path / "model" / "artifact.json"), "config_sha256": "c" * 64,
            "source_config": {"execution_contract": EXECUTION, "quality_metric": QUALITY}}}
    protocol["planned_runs"] = {role: [{"complex_fingerprint_sha256": item["fingerprint"], "algorithm": algorithm, "seed": 1}
        for algorithm in protocol["algorithms"]] for role, item in zip(("validation", "test"), selected)}
    protocol = _record(protocol)
    protocol_path = tmp_path / "protocol.json"
    _save(protocol_path, protocol)
    selection = _record({"schema_version": runner.SELECTION_SCHEMA, "input_root": str(source),
        "targets": selected, "algorithms": protocol["algorithms"], "seeds": [1]})
    selection_path = tmp_path / "selection.json"
    _save(selection_path, selection)
    context = {"protocol": protocol, "policy": None, "record_path": protocol_path,
               "record_file_sha256": runner.file_sha256(protocol_path)}
    args = SimpleNamespace(selection=str(selection_path), role="validation", protocol=str(protocol_path),
        policy=None, workers=8, timeout=600., out_dir=str(tmp_path / "collection"),
        worker_index=None, resume_cancelled=False)
    return SimpleNamespace(source=source, report=report, protocol=protocol, selection=selection,
        context=context, args=args, selection_path=selection_path)


def test_tampered_selection_is_rejected_before_discovery(inputs, monkeypatch):
    changed = deepcopy(inputs.selection)
    changed["targets"][0]["role"] = "test"
    _save(inputs.selection_path, changed)
    monkeypatch.setattr(runner, "discover_ppi4dock", lambda *a, **k: pytest.fail("discovery must not run"))
    with pytest.raises(ValueError, match="integrity"):
        runner.checked_selection(inputs.selection_path, inputs.protocol)


def test_even_rehashed_role_swaps_cannot_change_protocol_groups(inputs):
    changed = deepcopy(inputs.selection)
    for item in changed["targets"]:
        item["role"] = "test" if item["role"] == "validation" else "validation"
    changed.pop("record_sha256")
    _save(inputs.selection_path, _record(changed))
    with pytest.raises(ValueError, match="groups differ"):
        runner.checked_selection(inputs.selection_path, inputs.protocol)


def test_selected_input_bytes_are_rechecked(inputs):
    selected = runner.checked_selection(inputs.selection_path, inputs.protocol)
    Path(selected["targets"][0]["ligand_path"]).write_bytes(b"changed molecular input")
    with pytest.raises(ValueError, match="ligand input changed"):
        runner.selected_report(selected, "validation")


def test_unselected_targets_are_never_parsed(inputs, monkeypatch):
    discovered, parsed = [], []
    def discover(path, *, validate_contents):
        discovered.append(validate_contents)
        return inputs.report
    def validate(target, **kwargs):
        parsed.append(target.complex_id)
        return target
    monkeypatch.setattr(runner, "discover_ppi4dock", discover)
    monkeypatch.setattr(runner, "_validate_target_contents", validate)
    report = runner.selected_report(inputs.selection, "validation")
    assert discovered == [False]
    assert parsed == [inputs.selection["targets"][0]["complex_id"]]
    assert report.validation_performed is True


def test_only_requested_role_is_collected_and_original_budget_is_explicit(inputs, monkeypatch):
    monkeypatch.setattr(runner, "frozen_context", lambda *a: inputs.context)
    _, report, jobs, partitions, manifest = runner.study_plan(inputs.args)
    assert [item.complex_id for item in report.targets] == [inputs.selection["targets"][0]["complex_id"]]
    assert sum(len(partition) for partition in partitions) == 1
    assert len(jobs) == 6
    assert {job.profile.evaluation_budget for job in jobs if job.protocol.value == "reduced"} == {5700}
    assert {job.profile.evaluation_budget for job in jobs if job.protocol.value == "full"} == {7600}
    assert manifest["planned_jobs"] == 6
    assert manifest["automatic_skipping"] is False
    assert "standard_rescoring_pending" in manifest["quality_status"]
    assert not Path(inputs.args.out_dir).exists()


def test_changed_planned_run_or_resume_settings_are_rejected(inputs, monkeypatch):
    monkeypatch.setattr(runner, "frozen_context", lambda *a: inputs.context)
    _, _, _, _, manifest = runner.study_plan(inputs.args)
    output = Path(inputs.args.out_dir)
    output.mkdir()
    runner.bind_manifest(output, manifest)
    runner.bind_manifest(output, manifest)
    inputs.args.workers = 4
    _, _, _, _, changed = runner.study_plan(inputs.args)
    with pytest.raises(ValueError, match="Resume rejected"):
        runner.bind_manifest(output, changed)
    inputs.protocol["planned_runs"]["test"].pop()
    with pytest.raises(ValueError, match="planned runs"):
        runner.study_plan(inputs.args)


def test_equal_budget_contract_is_not_silently_reused(inputs):
    changed = deepcopy(inputs.protocol)
    changed["model"]["source_config"]["execution_contract"]["reduced_evaluation_budget"] = 7600
    with pytest.raises(ValueError, match="5700 / FULL 7600"):
        runner.checked_jobs(changed)


def test_test_requires_a_selected_frozen_policy_before_inputs_are_read(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Test requires --policy"):
        runner.frozen_context("test")
    monkeypatch.setattr(runner, "_checked_policy", lambda p: ({"status": "no_useful_rule", "threshold": None}, {}, {}))
    with pytest.raises(ValueError, match="No useful frozen rule"):
        runner.frozen_context("test", policy_path=tmp_path / "policy.json")


def test_first_physical_failure_is_not_retried_or_timed_as_a_new_run(tmp_path, monkeypatch):
    calls = []
    def failing_engine(self, **kwargs):
        calls.append(kwargs)
        raise TimeoutError("original physical timeout")
    monkeypatch.setattr(runner.DockingPPLocalEngine, "run", failing_engine)
    engine = object.__new__(runner.FrozenRunEngine)
    work = tmp_path / "job" / "engine"
    with pytest.raises(TimeoutError):
        engine.run(work_dir=work)
    timing_bytes = (work / "benchmark_timing.json").read_bytes()
    original = {"status": "FAILED", "attempt": 1, "message": "original physical timeout"}
    _save(work.parent / "error.json", original)
    with pytest.raises(RuntimeError, match="no docking retry"):
        engine.run(work_dir=work)
    assert len(calls) == 1
    assert json.loads((work.parent / "first_failure.json").read_text()) == original
    assert (work / "benchmark_timing.json").read_bytes() == timing_bytes


def test_workers_launch_hidden_with_single_thread_and_no_training(inputs, monkeypatch):
    output = Path(inputs.args.out_dir)
    output.mkdir()
    launches = []
    class Child:
        pid = 123
        def poll(self):
            return 0
    def popen(command, **kwargs):
        launches.append((command, kwargs))
        return Child()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner.subprocess, "CREATE_NO_WINDOW", 0x08000000, raising=False)
    runner.launch_workers(inputs.args, [(inputs.report.targets[0],)] + [()] * 7)
    command, options = launches[0]
    assert options["creationflags"] & 0x08000000
    assert options["env"]["OMP_NUM_THREADS"] == "1"
    assert options["stdin"] == runner.subprocess.DEVNULL
    assert "--protocol" in command and "--policy" not in command
    assert not any("fit" in token or "train" in token for token in command)


def test_cancelled_collection_needs_explicit_acknowledgment(inputs, monkeypatch):
    batch = Path(inputs.args.out_dir) / "shard_01" / "batch"
    batch.mkdir(parents=True)
    (batch / runner.CANCEL_FILENAME).write_text("cancelled")
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: pytest.fail("cancelled study must not launch"))
    with pytest.raises(ValueError, match="resume-cancelled"):
        runner.launch_workers(inputs.args, [(inputs.report.targets[0],)] + [()] * 7)
