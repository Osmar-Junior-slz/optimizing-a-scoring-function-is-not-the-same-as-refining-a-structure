"""Rejected scan contacts cannot improve a pocket's usable-pose ranking."""
import numpy as np
import pytest

from dockingpp.domain.models import Pocket
from dockingpp.pipeline import scan


def sample(index, contact, *, feasible=True, energy=-2.0):
    return {
        "sample_idx": index,
        "sample_is_feasible": feasible,
        "sample_proxy_is_feasible": True,
        "sample_hard_clash_gate_passed": feasible,
        "energy_sample": energy,
        "frac_contact": contact,
        "raw_contact_count": contact * 100,
        "severe_binary": 0.0,
        "frac_severe": 0.0,
        "frac_moderate": 0.0,
        "pen_exp": 0.0,
    }


def aggregate(samples):
    result = scan._aggregate_sample_statistics(
        samples, n_samples_requested=len(samples), n_atoms_peptide=100,
    )
    result["robust_score"], result["score_components"] = scan._compute_robust_pocket_score(result, {})
    return result


@pytest.mark.parametrize("mode", ["legacy", "geom_kdtree"])
def test_rejected_contact_bonus_cannot_choose_wrong_pocket(mode):
    # Both pockets have two viable samples. A's apparent interaction comes
    # from rejected poses, including a pose that passed the fractional proxy.
    polluted = aggregate([sample(0, .1), sample(1, .1), sample(2, 1., feasible=False, energy=-20.),
                          sample(3, 1., feasible=False, energy=-20.)])
    usable = aggregate([sample(0, .3), sample(1, .3), sample(2, 0., feasible=False),
                        sample(3, 0., feasible=False)])
    assert polluted["contact_fraction_mean"] > usable["contact_fraction_mean"]
    assert polluted["energy_q20"] < usable["energy_q20"]
    adapter = (scan._adapt_core_stats_to_legacy_schema if mode == "legacy"
               else lambda core: scan._adapt_core_stats_to_geom_schema(core, {}))
    table = {"a": adapter(polluted), "b": adapter(usable)}
    pockets = [Pocket("a", np.zeros(3), 1.), Pocket("b", np.ones(3), 1.)]
    selected = scan.select_pockets_from_scan(pockets, table, 1, selector_mode=mode)
    assert [pocket.id for pocket in selected] == ["b"]
    assert table["a"]["ranking_statistics_scope"] == "feasible_samples_v1"
    stats = table["a"]["feasible_pose_statistics"]
    assert stats["n_samples"] == 2
    assert stats["contact_fraction_mean"] == pytest.approx(.1)
    assert stats["best_sample_meta"]["sample_is_feasible"]
    assert stats["best_sample_meta"]["sample_idx"] == 0


def test_score_uses_usable_pose_statistics_and_preserves_attempt_diagnostics():
    first = aggregate([sample(0, .2, energy=-3.), sample(1, .2, energy=-1.),
                       sample(2, 0., feasible=False, energy=10.)])
    second = aggregate([sample(0, .2, energy=-3.), sample(1, .2, energy=-1.),
                        sample(2, 1., feasible=False, energy=-100.)])
    assert first["robust_score"] == pytest.approx(second["robust_score"])
    assert first["energy_mean"] != second["energy_mean"]
    assert first["contact_fraction_mean"] != second["contact_fraction_mean"]
    assert second["feasible_fraction"] == pytest.approx(2 / 3)
    terms = second["score_components"]["terms"]
    assert terms["contact_fraction_mean"] == pytest.approx(.2)
    assert terms["energy_mean"] == pytest.approx(-2.)
    assert second["score_components"]["pose_statistics_scope"] == "feasible_samples_v1"


@pytest.mark.parametrize("samples", [[], [sample(0, 1., feasible=False, energy=-100.)]])
def test_no_viable_sample_produces_no_usable_contact_or_seed(samples):
    result = aggregate(samples)
    stats = result["feasible_pose_statistics"]
    assert stats["n_samples"] == 0
    assert stats["contact_fraction_mean"] == 0.0
    assert stats["best_contact_count"] == 0.0
    assert stats["best_sample_meta"] is None
    assert result["score_components"]["terms"]["normalized_best_contact"] == 0.0
    assert np.isfinite(result["robust_score"])


def test_preexisting_scan_tables_retain_their_ranking_contract():
    pockets = [Pocket("a", np.zeros(3), 1.), Pocket("b", np.ones(3), 1.)]
    table = {"a": {"feasible_fraction": .5, "contact_fraction_mean": .1},
             "b": {"feasible_fraction": .5, "contact_fraction_mean": .3}}
    selected = scan.select_pockets_from_scan(pockets, table, 1)
    assert [pocket.id for pocket in selected] == ["b"]


@pytest.mark.parametrize("top_k", [None, 0, -1, 2, 10])
def test_selecting_all_pockets_preserves_scan_ranking(top_k):
    pockets = [Pocket("a", np.zeros(3), 1.), Pocket("b", np.ones(3), 1.)]
    table = {"a": {"feasible_fraction": .5, "contact_fraction_mean": .1},
             "b": {"feasible_fraction": .5, "contact_fraction_mean": .3}}
    selected = scan.select_pockets_from_scan(pockets, table, top_k)
    assert [pocket.id for pocket in selected] == ["b", "a"]
