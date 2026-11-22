"""Independent sampling, winner-only quality, and auditable resume."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import numpy as np
from dockingpp.domain.models import Pose
from dockingpp.pipeline.execucao.busca_execucao import _rescoring_final

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import revision_random_control as control


def test_pool_returns_best_scores_over_all_draws_and_preserves_first_tie():
    scores = iter([1, 2, 8, 2, 8, -1, 5])
    space = SimpleNamespace(random_pose=lambda: SimpleNamespace(meta={}, score_cheap=None),
                            evaluate_pose_score=lambda pose: next(scores))
    pool = control.uniform_pool(space, 7, 3)
    assert [(p.meta["draw"], p.score_cheap) for p in pool] == [(3, 8), (5, 8), (7, 5)]
    with pytest.raises(StopIteration):
        next(scores)


def test_validity_and_dockq_follow_the_archived_winner_rules():
    assert control.thresholds("reduced", True, None)["success"] == 0
    assert control.thresholds("expanded", True, .5) == dict(valid_pose=1, success=1, success_common_gate=0)
    assert control.thresholds("reduced", False, .23)["success"] == 1
    assert control.thresholds("expanded", False, .2299)["success"] == 0
    with pytest.raises(ValueError):
        control.thresholds("expanded", False, None)


def test_archived_rescorer_scores_unique_rounded_geometries_in_selected_pool(monkeypatch):
    calls = []
    def rescore(pose, *args):
        calls.append(pose.meta["draw"])
        return pose.score_cheap

    monkeypatch.setattr(_rescoring_final, "score_pose_expensive", rescore)
    pool = [Pose(coords=np.array([[x, 0., 0.]]), score_cheap=score,
                 meta={"pocket_id": "local_refinement", "draw": index})
            for index, (x, score) in enumerate([(0., 4.), (1.e-11, 8.), (2., 6.)], start=1)]
    result = control.RunResult(best_pose=pool[1], population=None, metrics={"rescoring_candidates": pool})
    cfg = control.Config(rescoring_enabled=True, rescoring_top_k=1)
    summary = control.ServicoRescoringFinal().aplicar(cfg=cfg, receptor=None, peptide=None,
                                                     result=result, logger=control.RunLogger())
    assert summary["rescoring_alignment_mode"] == "all_candidates"
    assert summary["rescored_candidates_count"] == 2
    assert calls == [2, 3]


def test_dockq_failure_resumes_saved_winner_and_atomic_result(tmp_path, monkeypatch):
    pose = tmp_path / "artifacts/best_pose.json"
    model = tmp_path / "model.pdb"
    model.write_text("model", encoding="utf-8")
    reference = tmp_path / "reference.pdb"
    reference.write_text("reference", encoding="utf-8")
    target = dict(complex_id="fixture", receptor_chain="A", ligand_chain="B", reference_sha256=control._file_sha256(reference))
    calls = []
    def draw(*args):
        calls.append("draw")
        pose.parent.mkdir()
        pose.write_text("fixture", encoding="utf-8")
        (tmp_path / "draws.npz").write_bytes(b"draws")
        return dict(complex_id="fixture", extent="expanded", hard_clash=False), "test-vectors"
    def score(*args, **kwargs):
        calls.append("dockq")
        path = Path(kwargs["output_dir"]) / "standard_dockq.json"
        if calls.count("dockq") == 1:
            path.write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted")
        control.save_record(path, {"quality_metric": control.quality_metric_descriptor(), "source_binding": kwargs["source_binding"],
                           "inputs": {"model_sha256": control._file_sha256(model), "reference_sha256": target["reference_sha256"]},
                           "metrics": {"dockq": .24}})
    monkeypatch.setattr(control, "draw_and_save", draw)
    monkeypatch.setattr(control, "evaluate_standard_complex", score)
    monkeypatch.setattr(control, "prepare_reduced_pose", lambda **kw: model)
    case = (target, dict(receptor=model, ligand=model, reference=reference), {})
    with pytest.raises(RuntimeError, match="interrupted"):
        control.trial(case, "expanded", tmp_path, None, "protocol")
    assert (tmp_path / "selected.json").exists() and not (tmp_path / "standard_dockq.json").exists()
    assert control.trial(case, "expanded", tmp_path, None, "protocol")["success"] == 1
    assert control.trial(case, "expanded", tmp_path, None, "protocol")["success"] == 1
    assert calls == ["draw", "dockq", "dockq"]


def test_protect_other_archived_batch_before_writing(tmp_path):
    study = tmp_path / "study"
    study.mkdir()
    archived = tmp_path / "other"
    archived.mkdir()
    (archived / "batch_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Preserve"):
        control.run(study, archived / "new", None, None)
    assert not (archived / "new").exists()


def test_resume_after_failure_reuses_only_completed_trial(tmp_path, monkeypatch):
    study, output = tmp_path / "study", tmp_path / "analysis"
    study.mkdir()
    selection = study / "selection.json"
    selection.write_text("{}", encoding="utf-8")
    target = {"complex_id": "test"}
    cases = [(target, {}, {extent: selection for extent in control.EXTENTS})]
    monkeypatch.setattr(control, "inputs_for_study", lambda *args: cases)
    monkeypatch.setattr(control, "engine_identity", lambda: "fixture")
    monkeypatch.setattr(control, "official_dockq_preflight", lambda **kw: dict(
        available=True, quality_metric=control.quality_metric_descriptor(), implementation_sha256="test"))
    attempted = []

    def mock_trial(case, extent, directory, deps, protocol_hash):
        attempted.append(extent)
        if attempted == ["reduced", "expanded"]:
            raise RuntimeError("interrupted")
        artifact = directory / "artifacts/best_pose.json"
        artifact.parent.mkdir(exist_ok=True)
        artifact.write_text("{}", encoding="utf-8")
        (directory / "selected.json").write_text("{}", encoding="utf-8")
        (directory / "draws.npz").write_bytes(b"draws")
        return dict(complex_id="test", extent=extent, dockq=None, valid_pose=0, success=0, success_common_gate=0)

    monkeypatch.setattr(control, "trial", mock_trial)
    with pytest.raises(RuntimeError, match="interrupted"):
        control.run(study, output, None, None)
    assert not (output / "summary.json").exists()
    report = control.run(study, output, None, None)
    assert attempted == ["reduced", "expanded", "expanded"]
    assert report["expanded"]["planned"] == 1
    assert report["reduced"]["success_rate"] == 0
    assert len((output / "observations.csv").read_text().splitlines()) == 3
    (output / "trials/test/reduced/artifacts/best_pose.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="incompatível"):
        control.run(study, output, None, None)
