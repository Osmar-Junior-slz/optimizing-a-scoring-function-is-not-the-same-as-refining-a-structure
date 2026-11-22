"""Local refinement uses matched information and a genuinely nested domain."""
from types import SimpleNamespace

import numpy as np
import pytest

from dockingpp.application.config import Config
from dockingpp.domain.models import Pocket
from dockingpp.search.algoritmos.espaco_rigido import RigidSearchSpace
from docking_confidence.adapters.dockingpp_archive_engine import DockingPPArchiveEngine
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, ProtocolMode


def _space(mode, mobile, task="local_refinement", **kwargs):
    cfg = Config(search_space_mode=mode, docking_task=task, **kwargs)
    return RigidSearchSpace(cfg=cfg, rng=np.random.default_rng(1),
                            pocket=Pocket("irrelevant_surface", np.array([100., 200., 300.]), 5.),
                            base_coords=mobile, score_cheap_fn=lambda *args: 0.,
                            max_trans=5., max_rot_deg=25.)


def test_local_arms_share_identity_and_reduced_is_nested():
    mobile = np.array([[10., 1., 2.], [12., 3., 4.], [8., 5., 1.]])
    full, reduced = [_space(mode, mobile) for mode in ("full", "reduced")]
    for space in (full, reduced):
        pose = space.solution_to_pose(np.zeros(6))
        np.testing.assert_allclose(pose.coords, mobile, atol=1e-12)
        assert pose.meta["input_relative_placement_used"]
        assert pose.meta["docking_task"] == "local_refinement"
        assert space.anchor_mode == "input_centroid"
    np.testing.assert_array_equal(reduced.domain_high, full.domain_high * .5)
    np.testing.assert_array_equal(reduced.domain_low, full.domain_low * .5)
    for solution in [reduced.random_solution() for _ in range(10)]:
        np.testing.assert_allclose(full.solution_to_pose(solution).coords,
                                   reduced.solution_to_pose(solution).coords)


def test_local_refinement_rejects_conflicting_anchor():
    with pytest.raises(ValueError, match="local_refinement requires"):
        _space("reduced", np.ones((4, 3)), search_anchor_mode="pocket_center")


def test_resolved_local_config_caps_reduced_and_preserves_full_budget():
    jobs = build_execution_plan(algorithms=[Algorithm.HYBRID],
                                protocol_mode=ProtocolMode.REDUCED_VS_FULL,
                                base_config=BaseSearchConfig(), seeds=[1])
    configs = {job.protocol.value: DockingPPArchiveEngine._resolved_job_config(job, {}) for job in jobs}
    assert configs["reduced"]["objective_evaluation_budget_per_pocket"] == 7600
    assert configs["full"].get("objective_evaluation_budget_per_pocket") is None
    assert configs["full"]["docking_confidence_audit"]["initialization"] == "original_uniform_population"
    assert configs["reduced"]["docking_confidence_audit"]["effective_max_trans"] == 2.5
    for cfg in configs.values():
        assert cfg["docking_task"] == "local_refinement"
        assert cfg["docking_confidence_audit"]["input_relative_placement_used"]
        assert cfg["docking_confidence_audit"]["runtime_basis"] == "engine_subprocess_wall_seconds"
        assert cfg["reduced_geometric_fallback_enabled"] is False


@pytest.mark.parametrize("algorithm", ["ag", "pso", "hybrid"])
def test_local_initial_pose_is_evaluated_normally_and_operator_runs(algorithm, tmp_path, monkeypatch):
    from dockingpp.pipeline.run import run_pipeline
    from dockingpp.pipeline.execucao.busca_execucao import _execucao_unica_coordenador as coordinator
    from dockingpp.pipeline.execucao.busca_execucao import _rescoring_final as rescoring

    seen = []
    def cheap(pose, pocket, weights):
        seen.append(dict(pose.meta))
        # Nonzero displacement is better than the supplied identity pose.
        offset = np.array(pose.meta["solution_vector"][:3])
        score = -float(np.sum((offset - [.4, .5, .6]) ** 2))
        pose.meta.update(contatos=10, colisao_severa=False)
        return score
    def expensive(pose, *args):
        pose.meta["final_hard_clash_gate"] = {"enabled": True, "triggered": False}
        return pose.score_cheap
    monkeypatch.setattr(coordinator, "score_pose_cheap", cheap)
    monkeypatch.setattr(rescoring, "score_pose_expensive", expensive)
    cfg = Config(docking_task="local_refinement", search_space_mode="reduced", full_search=False,
                 search_engine=algorithm, generations=4, pop_size=8, scan={"enabled": True},
                 structural_metrics_enabled=False, confidence_policy={"enabled": False},
                 ga_interval=2, sw_interval=2, sw_elite_iter=2, sw_heavy_iter=2)
    run_pipeline(cfg, "__dummy__", "__dummy__", str(tmp_path))
    initial = [meta for meta in seen if np.array_equal(meta.get("solution_vector"), np.zeros(6))]
    assert initial
    import json
    events = [json.loads(line) for path in tmp_path.rglob("metrics.jsonl")
              for line in path.read_text().splitlines() if line.strip()]
    assert any(event.get("name") == "search.operator_completed" for event in events)
    pose = json.loads(next(tmp_path.glob("*/artifacts/best_pose.json")).read_text())
    assert pose["meta"]["docking_task"] == "local_refinement"
    assert pose["score_cheap"] > -.77
    assert not np.allclose(pose["meta"]["solution_vector"], np.zeros(6))
