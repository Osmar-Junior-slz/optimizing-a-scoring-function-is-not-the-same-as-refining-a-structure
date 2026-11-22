"""Score zero is a valid value in both protocols, including at pool cutoffs."""
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("mode", ["reduced", "full"])
def test_zero_score_survives_limited_cross_pocket_candidate_pool(monkeypatch, mode):
    from dockingpp.domain.models import Pocket, Population, Pose
    from dockingpp.search.algoritmos.base import BasePopulationSearch

    cfg = SimpleNamespace(seed=1, search_space_mode=mode, generations=1,
                          rescoring_candidate_pool=1, topk=1)
    pockets = [Pocket(id=name, center=np.zeros(3), radius=2.0)
               for name in ("negative", "zero")]
    negative = Pose(coords=np.zeros((1, 3)), score_cheap=-2.0,
                    meta={"pocket_id": "negative"})
    zero = Pose(coords=np.ones((1, 3)), score_cheap=0.0,
                meta={"pocket_id": "zero"})
    poses = {"negative": negative, "zero": zero}
    search = BasePopulationSearch(cfg)

    def completed_pocket(**kwargs):
        pose = poses[kwargs["pocket"].id]
        return (pose, Population(poses=[pose], generation=1), [pose],
                {"pocket_id": kwargs["pocket"].id,
                 "best_score_global": pose.score_cheap})

    monkeypatch.setattr(search, "_search_single_pocket", completed_pocket)
    result = search.search(None, np.ones((1, 3)), pockets, cfg,
                           None, None, None, None, None)
    assert result.best_pose is zero
    assert result.metrics["rescoring_candidates"] == [zero]
    assert result.metrics["population_confidence_global"]["dominant_pocket_id"] == "zero"


@pytest.mark.parametrize("scope", ["global", "per_pocket"])
def test_zero_cheap_score_wins_top_k_selection_over_negative_and_missing(scope):
    from dockingpp.domain.models import Pose
    from dockingpp.pipeline.execucao.busca_execucao.rescoring_artefatos import (
        _select_rescoring_candidates,
    )

    candidates = [Pose(coords=np.full((1, 3), index), score_cheap=score,
                       meta={"pocket_id": "one"})
                  for index, score in enumerate((-3.0, None, 0.0))]
    selected, skipped = _select_rescoring_candidates(
        candidates=candidates, top_k=1, scope=scope,
    )
    assert len(selected) == 1
    assert selected[0] is candidates[2]
    assert {id(pose) for pose in skipped} == {id(pose) for pose in candidates[:2]}


@pytest.mark.parametrize("mode", ["reduced", "full"])
def test_zero_final_rescore_beats_negative_and_failed_rescore(monkeypatch, mode):
    from dockingpp.domain.models import Pose, RunResult
    from dockingpp.pipeline.execucao.busca_execucao import _rescoring_final

    candidates = [Pose(coords=np.full((1, 3), index), score_cheap=cheap,
                       meta={"pocket_id": "one", "result_score": final})
                  for index, (cheap, final) in enumerate(((3.0, -2.0), (2.0, None), (1.0, 0.0)))]
    result = RunResult(best_pose=candidates[0], population=None,
                       metrics={"rescoring_candidates": candidates})
    cfg = SimpleNamespace(search_space_mode=mode, rescoring_enabled=True,
                          rescoring_top_k=3, rescoring_scope="global",
                          rescoring_alignment_mode="all_candidates")
    logger = SimpleNamespace(log_metric=lambda *args, **kwargs: None)
    monkeypatch.setattr(_rescoring_final, "score_pose_expensive",
                        lambda pose, *args: pose.meta["result_score"])
    summary = _rescoring_final.ServicoRescoringFinal().aplicar(
        cfg=cfg, receptor=None, peptide=None, result=result, logger=logger,
    )
    assert result.best_pose is candidates[2]
    assert summary["best_final_rescore"] == 0.0
    assert summary["rescored_candidates_count"] == 2
    assert summary["ranking_changed_after_rescoring"] is True
    rows = {row["score_cheap"]: row for row in summary["selected_candidates"]}
    assert rows[1.0]["candidate_rank_after_rescoring"] == 1
    assert rows[2.0]["candidate_rank_after_rescoring"] == -1
