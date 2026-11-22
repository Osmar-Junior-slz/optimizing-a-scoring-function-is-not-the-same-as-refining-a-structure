"""Spatial pruning preserves every scalar rescoring operation and diagnostic."""
from pathlib import Path
from types import ModuleType
import zipfile

import numpy as np
import pytest

from dockingpp.scoring import expensive


def atoms(coords):
    types = ["C", "OA", "HD", "N", "S"]
    return [{"coords": np.asarray(point, dtype=float),
             "line": " " * 70 + f"{(.2 if index % 2 else -.3):6.3f} " + types[index % len(types)]}
            for index, point in enumerate(coords)]


def compare_scalar(monkeypatch, ligand, receptor, **kwargs):
    scorer = expensive.VinaScoringFunction(**kwargs)
    actual = scorer.avaliar_pose_detalhado(ligand, receptor)
    with monkeypatch.context() as patch:
        patch.setattr(expensive, "_candidate_receptor_indices", lambda point, coords, cutoff, count: range(count))
        expected = scorer.avaliar_pose_detalhado(ligand, receptor)
    assert actual == expected
    return actual


@pytest.mark.parametrize("shape", ["sparse", "dense", "distant"])
def test_every_term_and_distance_stat_matches_scalar_reference(monkeypatch, shape):
    rng = np.random.default_rng(412)
    scale = 30. if shape == "sparse" else 2.
    receptor = atoms(rng.normal(0, scale, size=(180, 3)))
    ligand = atoms(rng.normal(0, scale, size=(25, 3)) + (500. if shape == "distant" else 0.))
    payload = compare_scalar(monkeypatch, ligand, receptor, pair_clip_min=-.2, pair_clip_max=.1)
    if shape == "dense":
        assert payload["pair_stats"]["n_pairs_clipped_min"] > 0
        assert payload["pair_stats"]["n_pairs_clipped_max"] > 0
    if shape == "distant":
        assert payload["pair_stats"]["n_pairs"] == 0
        assert payload["pair_stats"]["min_distance"] is None


def test_cutoff_boundary_and_hard_clash_diagnostics_are_identical(monkeypatch):
    cutoff = 8.
    receptor = atoms([[cutoff, 0., 0.], [np.nextafter(cutoff, 0.), 0., 0.],
                      [np.nextafter(cutoff, np.inf), 0., 0.], [4.8, 6.4, 0.],
                      [1.1, 0., 0.], [1e6, 0., 0.]])
    payload = compare_scalar(monkeypatch, atoms([[0., 0., 0.]]), receptor)
    assert payload["pair_stats"]["n_pairs"] == 4
    assert payload["pair_stats"]["min_distance"] == 1.1
    gate = expensive._apply_final_hard_clash_gate(payload, {"hard_clash_min_distance": 1.2})
    assert gate["hard_clash_gate"]["triggered"]


@pytest.mark.parametrize("cutoff", [0., -1., 1e-200, float("inf")])
def test_extreme_cutoffs_keep_original_scalar_behavior(monkeypatch, cutoff):
    compare_scalar(monkeypatch, atoms([[0., 0., 0.]]), atoms([[1e-180, 0., 0.], [3., 0., 0.]]), cutoff_pair=cutoff)


@pytest.mark.parametrize("empty_ligand,empty_receptor", [(True, False), (False, True), (True, True)])
def test_empty_partners_keep_original_payload(monkeypatch, empty_ligand, empty_receptor):
    compare_scalar(monkeypatch, [] if empty_ligand else atoms([[0., 0., 0.]]),
                   [] if empty_receptor else atoms([[0., 0., 0.]]))


def test_candidate_filter_keeps_order_and_prunes_only_distant_pairs():
    coords = np.asarray([[2., 0., 0.], [50., 0., 0.], [-3., 0., 0.], [0., 80., 0.]])
    assert list(expensive._candidate_receptor_indices(np.zeros(3), coords, 8., 4)) == [0, 2]


def test_original_archive_scorer_payload_is_exact_when_archive_available():
    archive_path = next((parent / "DockingReduce_V5.zip" for parent in Path(__file__).resolve().parents
                         if (parent / "DockingReduce_V5.zip").is_file()), None)
    if archive_path is None:
        pytest.skip("optional original-source equivalence; scalar reference tests still run")
    with zipfile.ZipFile(archive_path) as archive:
        source = archive.read("dockingpp/dockingpp/scoring/expensive.py")
    original = ModuleType("original_expensive_reference")
    exec(compile(source, "original_expensive_reference.py", "exec"), original.__dict__)
    rng = np.random.default_rng(831)
    ligand, receptor = atoms(rng.normal(0., 10., (30, 3))), atoms(rng.normal(0., 20., (150, 3)))
    expected = original.VinaScoringFunction().avaliar_pose_detalhado(ligand, receptor)
    actual = expensive.VinaScoringFunction().avaliar_pose_detalhado(ligand, receptor)
    assert actual == expected
