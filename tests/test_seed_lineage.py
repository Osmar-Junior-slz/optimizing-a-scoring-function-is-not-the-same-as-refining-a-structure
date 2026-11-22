"""Operator pedigrees and particle identity do not change numerical search."""
from types import SimpleNamespace

import numpy as np
import pytest

from dockingpp.domain.models import Pose
from dockingpp.search.algoritmos import AGSearch, PSOSearch, ABCGAVGOSSearch
from dockingpp.search.algoritmos.contexto import GenerationContext, SeedOrigin, mark_seed_origin, seed_origin
from dockingpp.search.algoritmos._configuracao_populacional import initial_hybrid_operator_state


def _context(tracked):
    cfg = SimpleNamespace(ag_tournament_k=4, ag_mutation_probability=0.,
                          ag_crossover_probability=0., ga_interval=1)
    rng = np.random.default_rng(37)

    def pose(vector, score=None):
        return Pose(coords=np.array([vector[:3]]), score_cheap=score,
                    meta={"solution_vector": np.asarray(vector).tolist()})

    def evaluate(vector):
        score = -float(np.dot(vector, vector))
        return pose(vector, score), score, -score

    def random_vector():
        return rng.uniform(-1., 1., 6)

    previous = [evaluate(np.full(6, x))[0] for x in [.6, .1, 0., -.4]]
    if tracked:
        for index, member in enumerate(previous):
            mark_seed_origin(member, SeedOrigin.UNCHANGED if index == 2 else SeedOrigin.UNRELATED)
    return GenerationContext(
        cfg=cfg, rng=rng, previous=previous,
        raw_previous_scores=np.array([member.score_cheap for member in previous]),
        generation_index=1, selected=previous[:2], pop_size=4, topk=2,
        dim=6, max_trans=1., max_rot_deg=1.,
        domain_low=np.full(6, -1.), domain_high=np.full(6, 1.),
        random_solution=random_vector, random_pose=lambda: pose(random_vector()),
        existing_solution=lambda member: np.array(member.meta["solution_vector"]),
        evaluate_solution=evaluate, solution_to_pose=pose,
        hybrid_operator_history=[], hybrid_operator_state=initial_hybrid_operator_state(cfg),
    )


@pytest.mark.parametrize("engine", [AGSearch, PSOSearch, ABCGAVGOSSearch])
def test_operators_keep_random_stream_and_seed_provenance_after_reordering(engine):
    control, tracked = _context(False), _context(True)
    expected = engine(control.cfg)._next_generation(control)
    actual = engine(tracked.cfg)._next_generation(tracked)
    assert [p.meta["solution_vector"] for p in expected] == [p.meta["solution_vector"] for p in actual]
    assert tracked.rng.random() == control.rng.random()
    assert all(seed_origin(p) is None for p in expected)
    if engine is AGSearch:  # whole-population tournaments, no variation
        assert all(seed_origin(p) == SeedOrigin.UNCHANGED for p in actual)
    if engine is PSOSearch:  # seeded particle moved and remained in its original slot
        assert [bool(p.meta["seed_lineage"]) for p in actual] == [False, False, True, False]
        assert seed_origin(actual[2]) == SeedOrigin.DESCENDANT
        assert tracked.hybrid_operator_state["pso_global_best_from_seed_particle"] is True


def test_hybrid_retains_accepted_ancestry_and_clears_scout_lineage():
    context = _context(True)
    hybrid = ABCGAVGOSSearch(context.cfg)
    state = hybrid._state_from_context(context)
    hybrid._set_solution(state, 0, np.full(6, .03), .01, donor=2)
    assert seed_origin(hybrid._state_to_poses(state)[0]) == SeedOrigin.DESCENDANT
    hybrid._set_solution(state, 0, np.zeros(6), 0.)
    assert seed_origin(hybrid._state_to_poses(state)[0]) == SeedOrigin.DESCENDANT
    hybrid._set_solution(state, 2, np.zeros(6), 0., scout=True)
    assert seed_origin(hybrid._state_to_poses(state)[2]) == SeedOrigin.UNRELATED
