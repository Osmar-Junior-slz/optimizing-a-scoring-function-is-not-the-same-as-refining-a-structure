"""Scientific workflow tests use synthetic software fixtures, never docking evidence."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import os
import subprocess

import pandas as pd
import pytest

from docking_confidence.application import article_study as study, calibrated_batch
from docking_confidence.application.batch import _atomic_write_json, _file_sha256, _job_snapshot, _manifest_for_plan, _payload_sha256, run_batch
from docking_confidence.application.article_reporting import compare_frozen_methods
from docking_confidence.application.calibrated_batch import CalibratedBatchGate
from docking_confidence.application.probability_reporting import DiagnosticOptions
from docking_confidence.application.train import train
from docking_confidence.io.batch_discovery import discover_ppi4dock
from docking_confidence.io.config import config_sha256, load_config
from test_batch_runner import _RecordingEngine, _make_report
from test_training_context import _context_frame
from test_training_dataset import _FixturePoseEngine, _component


@pytest.fixture(scope="module")
def article_model(tmp_path_factory):
    root = tmp_path_factory.mktemp("article_model")
    cfg = load_config(study.ROOT / "configs/local_refinement_operational_candidate.yaml")
    cfg["diagnostics"] = {"bootstrap_repetitions": 100}
    dataset = root / "dataset.csv"
    frame = _context_frame(dataset, cfg, fifty=True)
    result = train(str(dataset), cfg, str(root / "model"), development_only=True)
    source = root / "training_batch"
    source.mkdir()
    engine = _FixturePoseEngine(set())
    jobs = study.study_jobs("Synthetic software test")[study.BLOCKS[1]]
    plan = {"jobs": [_job_snapshot(job) for job in jobs],
        "engine": {"name": engine.name, "scientific": engine.scientific, "descriptor": engine.audit_descriptor()},
        "mobile_partner_domain": {}, "targets": [{"complex_fingerprint_sha256": value} for value in frame.complex_fingerprint_sha256.unique()]}
    manifest = _manifest_for_plan(plan)
    _atomic_write_json(source / "batch_manifest.json", manifest)
    _atomic_write_json(root / "dataset_build_report.json", {"valid": True, "dataset_sha256": _file_sha256(dataset),
        "config_sha256": config_sha256(cfg), "batch_dir": str(source), "batch_manifest_sha256": manifest["manifest_sha256"]})
    return Path(result["artifact"])


def _input(root):
    folder = root / "1abc_AB"
    folder.mkdir(parents=True)
    receptor = _component("A", "ALA", 0, 1)
    ligand = _component("B", "GLY", 4, 100)
    (folder / "A_model_st.pdb").write_text(receptor + "END\n")
    (folder / "B_model_st.pdb").write_text(ligand + "END\n")
    (folder / "1abc_AB_st.pdb").write_text(receptor + ligand + "END\n")
    return root


def test_article_has_all_controls_and_matching_calibration_budget():
    jobs = study.study_jobs("Input pose origin documented")
    assert len(jobs[study.BLOCKS[0]]) == 12
    assert len(jobs[study.BLOCKS[1]]) == 6
    assert {job.profile.evaluation_budget for job in jobs[study.BLOCKS[0]]} == {7600}
    assert {(job.profile.search_extent, job.profile.initialization) for job in jobs[study.BLOCKS[0]]} == {
        (extent, initialization) for extent in ("reduced", "expanded") for initialization in ("uniform", "seeded")}
    assert {job.profile.population for block in jobs.values() for job in block} == {100}
    assert {job.profile.initialization for job in jobs[study.BLOCKS[1]]} == {"uniform"}
    assert [(job.protocol.value, job.profile.evaluation_budget) for job in jobs[study.BLOCKS[1]][:2]] == [("reduced", 5700), ("full", 7600)]


def test_overlap_and_alias_selection_are_molecular_and_label_independent(tmp_path):
    report = discover_ppi4dock(_input(tmp_path / "inputs"), validate_contents=True)
    target = report.targets[0]
    aliases = replace(report, targets=(replace(target, complex_id="z-alias"), target))
    selected, excluded = study.select_independent_targets(aliases, set())
    assert [value.complex_id for value in selected.targets] == ["1abc_AB"]
    assert excluded == [{"selected": "1abc_AB", "excluded": "z-alias"}]
    with pytest.raises(ValueError, match="desenvolvimento"):
        study.select_independent_targets(report, {target.complex_fingerprint_sha256})


def test_shadow_mode_scores_before_full_but_never_skips_it(tmp_path, monkeypatch):
    engine = _RecordingEngine()
    class AcceptingGate:
        def __init__(self, path): self.protocol_issues = []
        def validate_jobs(self, jobs, engine): pass
        def audit_descriptor(self): return {"artifact_sha256": "a" * 64}
        def prepare_evidence(self, *, task_directory, **kwargs):
            _atomic_write_json(task_directory / "confidence/evidence.json", {"test": True})
        def evaluate(self, **kwargs):
            assert [job.protocol.value for job, _, _ in engine.calls] == ["reduced"]
            return {"action": "ACCEPT_REDUCED", "reason": "fixture_supported", "decision_capable": True,
                    "probability": .99, "artifact_sha256": "a" * 64}
    monkeypatch.setattr(calibrated_batch, "CalibratedBatchGate", AcceptingGate)
    (tmp_path / "fixture.json").write_text("{}")
    source = tmp_path / "inputs"
    summary = run_batch(source, tmp_path / "batch", jobs=study.study_jobs("fixture")[study.BLOCKS[1]][:2],
        engine=engine, report=_make_report(source, count=1), confidence_artifact=tmp_path / "fixture.json", shadow_confidence=True)
    assert summary.succeeded == 2 and summary.skipped_calibrator == 0
    assert [job.protocol.value for job, _, _ in engine.calls] == ["reduced", "full"]
    decision = json.loads(next((tmp_path / "batch").rglob("confidence_decision.json")).read_text())
    assert decision["predictor_action"] == "ACCEPT_REDUCED"
    assert decision["action"] == "RUN_FULL" and decision["validation_mode"] is True


def test_incompatible_shadow_gate_is_rejected_before_docking(tmp_path, monkeypatch):
    class Incompatible:
        def __init__(self, path): self.protocol_issues = ["wrong_budget"]
        def validate_jobs(self, jobs, engine): pass
    monkeypatch.setattr(calibrated_batch, "CalibratedBatchGate", Incompatible)
    source = tmp_path / "inputs"
    engine = _RecordingEngine()
    with pytest.raises(RuntimeError, match="matching calibrated"):
        run_batch(source, tmp_path / "out", jobs=study.study_jobs("fixture")[study.BLOCKS[1]][:2],
            engine=engine, report=_make_report(source, count=1), confidence_artifact=tmp_path / "fixture", shadow_confidence=True)
    assert not engine.calls


@pytest.mark.parametrize("changes, runtime_only", [
    ({"python_executable": "global/python.exe"}, True),
    ({"python_runtime_version": "3.13 fixture"}, True),
    ({"config_sha256": "changed-config"}, False),
    ({"python_executable": "global/python.exe", "source_manifest_sha256": "changed-code"}, False),
])
def test_incompatible_engine_explains_differences_without_allowing_preparation(tmp_path, monkeypatch, changes, runtime_only):
    (tmp_path / "inputs").mkdir()
    descriptor = {"python_executable": "project/.venv/python.exe", "python_runtime_version": "3.12 fixture",
                  "source_manifest_sha256": "original-code", "config_sha256": "original-config"}
    saved = {"name": "dockingpp_local", "scientific": True, "descriptor": descriptor}
    current = {**descriptor, **changes}

    class Incompatible:
        contract = {"engine": saved}
        protocol_issues = ["execution_engine_differs_from_training"]
        def __init__(self, path): pass
        def validate_jobs(self, jobs, engine): pass

    engine = SimpleNamespace(name="dockingpp_local", scientific=True, audit_descriptor=lambda: current,
        run=lambda **kwargs: pytest.fail("Mismatch must not execute docking"))
    monkeypatch.setattr(study, "CalibratedBatchGate", Incompatible)
    monkeypatch.setattr(study, "configured_calibrator_path", lambda: tmp_path / "fixture-model.json")
    monkeypatch.setattr(study, "discover_ppi4dock", lambda *a, **k: pytest.fail("Reject the mismatch before discovering inputs"))
    with pytest.raises(ValueError, match="execution_engine_differs_from_training") as exc:
        study.prepare_article_study(tmp_path / "inputs", tmp_path / "output", pose_source="Synthetic error-path fixture", engine=engine)
    text = str(exc.value)
    assert ("iniciar_sistema.cmd" in text) is runtime_only
    for field, value in changes.items():
        assert str(value) in text and descriptor[field] in text
    assert not (tmp_path / "output" / study.PROTOCOL).exists()
    assert saved["descriptor"] == descriptor


def test_preparation_is_immutable_and_never_runs_docking(article_model, tmp_path):
    source = _input(tmp_path / "inputs")
    engine = _FixturePoseEngine(set())
    engine.run = lambda **kwargs: pytest.fail("Preparation must not execute docking")
    output = tmp_path / "article"
    protocol = study.prepare_article_study(source, output, pose_source="Synthetic software test", artifact_path=article_model, engine=engine)
    before = (output / study.PROTOCOL).read_bytes()
    assert len(protocol["block_plans"]) == 2
    assert all(value["threshold"] is None for value in protocol["comparators"]["policies"].values())
    assert not (output / study.BLOCKS[1] / "batch_manifest.json").exists()
    study.prepare_article_study(source, output, pose_source="Synthetic software test", artifact_path=article_model, engine=engine)
    assert before == (output / study.PROTOCOL).read_bytes()
    with pytest.raises(ValueError, match="outros dados"):
        study.prepare_article_study(source, output, pose_source="Changed source claim", artifact_path=article_model, engine=engine)
    assert before == (output / study.PROTOCOL).read_bytes()


def test_end_to_end_article_records_real_prediction_and_joint_report(article_model, tmp_path):
    source = _input(tmp_path / "inputs")
    engine = _FixturePoseEngine(set())
    output = tmp_path / "article"
    study.prepare_article_study(source, output, pose_source="Synthetic software test", artifact_path=article_model, engine=engine)
    result = study.run_article_study(output, engine=engine)
    report = json.loads(Path(result["json"]).read_text(encoding="utf-8"))
    assert report["counts"][study.BLOCKS[0]]["succeeded"] == 12
    assert report["counts"][study.BLOCKS[1]]["succeeded"] == 6
    assert report["calibration"]["paired_rows"] == 3
    assert report["calibration"]["actually_avoided_full"] == 0
    assert report["calibration"]["actually_saved_full_runtime_sec"] == 0
    for evaluation in report["calibration"]["comparison"]["evaluations"].values():
        assert evaluation["metrics"]["n"] == 3
        assert evaluation["metrics"]["roc_auc"] is None  # one class is unavailable, never invented
        assert evaluation["metrics"]["brier_score"] is not None
    assert not report["calibration"]["automatic_acceptance_authorized"]
    profile = report["calibration"]["comparison"]["structural_risk_profile"]
    assert profile["family_size"] == 600 and profile["selected_threshold"] is None
    assert profile["maximum_accepted_error_rate"] is None
    assert report["validation_risk_profile"]["stage"] == "risk_calibration"
    assert len(pd.read_csv(output / "analise_artigo/risco_cobertura_teste.csv")) == 300
    assert (output / "analise_artigo/risco_cobertura_teste.png").is_file()
    predictions = list((output / study.BLOCKS[1]).rglob("confidence/prediction.csv"))
    assert len(predictions) == 3
    for path in predictions:
        assert pd.read_csv(path).probability.between(0, 1).all()
        columns = pd.read_csv(path.parent / "inference_input.csv").columns
        assert not any("dockq" in column or column.endswith("_full") for column in columns)
    engine.run = lambda **kwargs: pytest.fail("A completed study must resume without repeating docking")
    assert study.run_article_study(output, resume=True, engine=engine) == result
    predictions[0].write_bytes(predictions[0].read_bytes() + b"\n")
    from docking_confidence.application.article_reporting import write_article_report
    with pytest.raises(ValueError, match="fontes mudaram"):
        write_article_report(output)


def test_reference_alignment_rounding_uses_audited_live_graph(article_model, tmp_path):
    """A rotated, rounded reference must not substitute the live input graph."""
    import math
    from docking_confidence.application.article_reporting import _recorded_predictions

    source = _input(tmp_path / "inputs")
    reference = source / "1abc_AB/1abc_AB_st.pdb"
    angle = 0.367
    lines = []
    for line in reference.read_text().splitlines(keepends=True):
        if line.startswith("ATOM"):
            x, y, z = (float(line[a:b]) for a, b in ((30, 38), (38, 46), (46, 54)))
            xx = x * math.cos(angle) - y * math.sin(angle) + 1.23456
            yy = x * math.sin(angle) + y * math.cos(angle) + 2.34567
            line = line[:30] + f"{xx:8.3f}{yy:8.3f}{z:8.3f}" + line[54:]
        lines.append(line)
    reference.write_text("".join(lines))
    engine = _FixturePoseEngine(set())
    output = tmp_path / "article"
    study.prepare_article_study(source, output, pose_source="Synthetic rotated-reference regression fixture", artifact_path=article_model, engine=engine)
    result = study.run_article_study(output, engine=engine)
    report = json.loads(Path(result["json"]).read_text())
    assert report["calibration"]["pairs_with_post_reference_graph_rounding_differences"] > 0
    frame = pd.read_csv(output / "analise_artigo/calibrador_previsoes_e_qualidade.csv", float_precision="round_trip")
    for row in frame.to_dict("records"):
        directory = output / study.BLOCKS[1] / "targets" / row["complex_id"] / "jobs" / row["run_id"]
        live = pd.read_csv(directory / "confidence/inference_input.csv", float_precision="round_trip").iloc[0]
        assert row["graph_min_cross_distance_angstrom"] == live["graph_min_cross_distance_angstrom"]
        assert row["audited_feature_source"] == "reference_free_reduced_evidence_v1"
    session = CalibratedBatchGate(article_model).session
    # Non-graph discrepancies are still rejected; tolerance was not loosened.
    with pytest.raises(ValueError, match="características"):
        _recorded_predictions(output / study.BLOCKS[1], frame.assign(elite_dispersion=frame.elite_dispersion + .001), session)
    # A modified raw PDB cannot be accepted as a rounding difference.
    first = frame.iloc[0]
    directory = output / study.BLOCKS[1] / "targets" / first.complex_id / "jobs" / first.run_id
    evidence = json.loads((directory / "confidence/evidence.json").read_text())
    raw_pose = directory / evidence["pose_path"]
    raw_pose.write_bytes(raw_pose.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="hash"):
        _recorded_predictions(output / study.BLOCKS[1], frame, session)


def test_cancel_stops_before_next_block(article_model, tmp_path, monkeypatch):
    output = tmp_path / "article"
    engine = _FixturePoseEngine(set())
    study.prepare_article_study(_input(tmp_path / "inputs"), output, pose_source="Synthetic test", artifact_path=article_model, engine=engine)
    calls = []
    def cancel_after_first(*args, **kwargs):
        calls.append(args)
        study.request_article_cancel(output)
        return SimpleNamespace(state="COMPLETED")
    monkeypatch.setattr(study, "run_batch", cancel_after_first)
    study.run_article_study(output, engine=engine)
    assert len(calls) == 1
    assert json.loads((output / study.STATUS).read_text())["state"] == "CANCELLED"


def test_comparison_never_adjusts_thresholds_from_test_labels():
    frame = pd.DataFrame({"complex_fingerprint_sha256": ["a", "b", "c", "d"], "preserved": [0, 1, 0, 1],
        "base_raw_score": [-2., 2., -1., 1.], "probability": [.1, .9, .2, .8], "probability_lower": [.05, .85, .15, .75],
        "in_applicability_domain": [True] * 4, "unseen_complex": [True] * 4, "trusted_reduced_input_provenance": [True] * 4,
        "runtime_reduced_sec": [3.] * 4, "runtime_full_sec": [4.] * 4, "confidence_overhead_sec": [.1] * 4})
    comparators = {"isotonic_x": [-2., 2.], "isotonic_y": [0., 1.],
                  "policies": {name: {"threshold": .8, "supported": True} for name in ("original", "isotonic", "venn_abers")}}
    saved = deepcopy(comparators)
    first = compare_frozen_methods(frame.copy(), comparators, options=DiagnosticOptions(100))
    second = compare_frozen_methods(frame.assign(preserved=1-frame.preserved), comparators, options=DiagnosticOptions(100))
    assert comparators == saved
    for method in comparators["policies"]:
        assert first["decisions"][method]["threshold"] == second["decisions"][method]["threshold"] == .8
        assert first["decisions"][method]["hypothetical_accepted_reduced"] == second["decisions"][method]["hypothetical_accepted_reduced"]
    assert first["decisions"]["isotonic"]["cost"]["overhead_sec"] is None


@pytest.mark.parametrize("failed_mode", ["reduced", "full"])
def test_missing_quality_pairs_do_not_create_fake_metrics(article_model, tmp_path, failed_mode):
    output = tmp_path / "article"
    engine = _FixturePoseEngine({("1abc_AB", failed_mode)})
    study.prepare_article_study(_input(tmp_path / "inputs"), output, pose_source="Synthetic test", artifact_path=article_model, engine=engine)
    result = study.run_article_study(output, engine=engine)
    report = json.loads(Path(result["json"]).read_text())
    assert report["calibration"]["paired_rows"] == 0
    assert report["calibration"]["scored_reduced_runs"] == (3 if failed_mode == "full" else 0)
    risk_row = report["calibration"]["comparison"]["structural_risk_profile"]["curves"]["venn_abers"][0]
    assert risk_row["total_complexes"] == 1
    if failed_mode == "reduced":
        assert risk_row["accepted_complexes"] == 0
        assert risk_row["simultaneous_complex_risk_upper_bound"] is None
    for evaluation in report["calibration"]["comparison"]["evaluations"].values():
        assert evaluation["metrics"]["n"] == 0
        assert evaluation["metrics"]["brier_score"] is None


def test_preflight_failure_is_saved_before_any_docking(article_model, tmp_path):
    output = tmp_path / "article"
    source = _input(tmp_path / "inputs")
    engine = _FixturePoseEngine(set())
    study.prepare_article_study(source, output, pose_source="Synthetic test", artifact_path=article_model, engine=engine)
    path = source / "1abc_AB/A_model_st.pdb"
    path.write_text(path.read_text() + "\n")
    engine.run = lambda **kwargs: pytest.fail("Changed inputs must not run")
    with pytest.raises(ValueError):
        study.run_article_study(output, engine=engine)
    assert json.loads((output / study.STATUS).read_text())["state"] == "FAILED"


def test_worker_launcher_pins_checkout_and_rejects_duplicate_process(article_model, tmp_path, monkeypatch):
    from docking_confidence.application import batch_launcher, batch_monitor
    output = tmp_path / "article"
    study.prepare_article_study(_input(tmp_path / "inputs"), output, pose_source="Synthetic test", artifact_path=article_model, engine=_FixturePoseEngine(set()))
    commands = []
    study.request_article_cancel(output)
    def launch(command, **kwargs):
        assert not (output / "article_cancel.json").exists()
        commands.append(command)
        study.request_article_cancel(output)  # New request during child startup.
        return SimpleNamespace(pid=123, log_path=output / "worker.log")
    monkeypatch.setattr(batch_launcher, "_launch_batch_process", launch)
    study.launch_article_study(output, resume=True)
    assert (output / "article_cancel.json").is_file()
    assert "--resume" not in commands[0]
    assert "-I" in commands[0] and "-c" in commands[0]
    foreign = tmp_path / "foreign"
    (foreign / "docking_confidence").mkdir(parents=True)
    (foreign / "docking_confidence/__init__.py").write_text("raise RuntimeError('foreign project imported')")
    completed = subprocess.run([*commands[0], "--help"], cwd=foreign,
        env={**os.environ, "PYTHONPATH": str(foreign)}, capture_output=True, encoding="utf-8", timeout=60)
    assert completed.returncode == 0, completed.stderr
    assert "--output-dir" in completed.stdout
    monkeypatch.setattr(batch_monitor, "process_alive", lambda *args: True)
    with pytest.raises(ValueError, match="já está em execução"):
        study.launch_article_study(output)
    assert len(commands) == 1
