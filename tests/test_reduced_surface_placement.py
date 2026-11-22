"""Surface sites constrain the mobile interface, not a protein's centroid."""
from types import SimpleNamespace

import numpy as np
import pytest

from dockingpp.domain.models import Pocket
from dockingpp.pipeline import scan
from dockingpp.search.algoritmos.espaco_rigido import RigidSearchSpace
from dockingpp.utils.geometry import place_surface_contact, surface_anchor_normal


def example():
    receptor = np.array([[0, y, z] for y in range(-8, 9) for z in range(-8, 9)], dtype=float)
    mobile = np.array([[x, y, z] for x in range(0, 21, 2) for y in (-1, 0, 1) for z in (-1, 0, 1)], dtype=float)
    pocket = Pocket("surface", np.array([3., 0., 0.]), 4., receptor,
                    {"detection_mode": "surface_clearance_grid", "receptor_coords_full": receptor})
    return receptor, mobile, pocket


def space(mobile, pocket, mode="reduced", anchor="auto"):
    cfg = SimpleNamespace(search_space_mode=mode, search_anchor_mode=anchor, cheap_weights={})
    return RigidSearchSpace(cfg=cfg, rng=np.random.default_rng(1), pocket=pocket,
                            base_coords=mobile, score_cheap_fn=lambda *a: 0.,
                            max_trans=5., max_rot_deg=25.)


def test_large_partner_surface_placement_is_rigid_and_keeps_local_contact():
    receptor, mobile, pocket = example()
    actual = space(mobile, pocket).solution_to_pose(np.zeros(6))
    legacy = space(mobile, pocket, anchor="pocket_center").solution_to_pose(np.zeros(6))
    tree = scan.build_receptor_kdtree(receptor)
    assert np.min(scan._nearest_dists(tree, legacy.coords, receptor)) < 1.2
    assert np.min(scan._nearest_dists(tree, actual.coords, receptor)) >= 1.2
    assert actual.meta["search_anchor_mode"] == "surface_contact"
    support = actual.meta["surface_support_atom_index"]
    np.testing.assert_allclose(actual.coords[support], pocket.center)
    np.testing.assert_allclose(np.linalg.norm(actual.coords[:, None] - actual.coords, axis=2),
                               np.linalg.norm(mobile[:, None] - mobile, axis=2), atol=1e-12)


@pytest.mark.parametrize("mode,anchor,detector", [
    ("full", "auto", "surface_clearance_grid"),
    ("reduced", "input_centroid", "surface_clearance_grid"),
    ("reduced", "auto", "provided_file"),
])
def test_full_and_explicit_cavity_frames_are_preserved(mode, anchor, detector):
    _, mobile, pocket = example()
    pocket.meta["detection_mode"] = detector
    rigid = space(mobile, pocket, mode=mode, anchor=anchor)
    solution = np.array([1., -2., 3., 15., -8., 4.])
    pose = rigid.solution_to_pose(solution)
    center = pocket.center if detector == "provided_file" else mobile.mean(axis=0)
    expected = (mobile - mobile.mean(axis=0)) @ rigid.euler_rotation_matrix(solution[3:]).T + center + solution[:3]
    np.testing.assert_allclose(pose.coords, expected, atol=1e-12)
    assert "surface_anchor_version" not in pose.meta


def test_scan_and_search_use_identical_surface_transform_and_exposure_point(monkeypatch):
    receptor, mobile, pocket = example()
    rigid = space(mobile, pocket)
    solution = np.array([.2, -.3, .1, 4., 2., -3.])
    pose = rigid.solution_to_pose(solution)
    transform = {"rotation": rigid.euler_rotation_matrix(solution[3:]),
                 "translation": pocket.center + solution[:3],
                 "sampled_offset": solution[:3], "region_type": "sphere"}
    monkeypatch.setattr(scan, "_sample_pocket_transforms", lambda *a, **k: [transform])
    original = scan._evaluate_sample_geometry
    observed = []
    def observe(**kwargs):
        np.testing.assert_allclose(kwargs["placed_coords"], pose.coords, atol=1e-12)
        value = original(**kwargs)
        observed.append(value)
        return value
    monkeypatch.setattr(scan, "_evaluate_sample_geometry", observe)
    scan._scan_pocket_core(scan.build_receptor_kdtree(receptor), receptor, mobile, pocket,
                          {"samples_per_pocket": 1, "contact_min": 2.4, "contact_max": 5.0},
                          np.random.default_rng(1))
    assert len(observed) == 1
    assert observed[0]["pen_exp"] == 0.0
    assert observed[0]["sample_is_feasible"]
    assert np.linalg.norm(pose.coords.mean(axis=0) - pocket.center) > pocket.radius


def test_surface_placement_is_independent_of_mobile_input_translation():
    _, mobile, pocket = example()
    solution = np.array([1., -2., .4, 5., -3., 9.])
    first = space(mobile, pocket).solution_to_pose(solution)
    second = space(mobile + [1000., -500., 20.], pocket).solution_to_pose(solution)
    np.testing.assert_allclose(first.coords, second.coords, atol=1e-10)


def test_surface_transform_is_equivariant_under_global_rigid_frame():
    receptor, mobile, pocket = example()
    rotation = RigidSearchSpace.euler_rotation_matrix(np.array([17., 32., -9.]))
    frame = RigidSearchSpace.euler_rotation_matrix(np.array([-25., 15., 38.]))
    translation = np.array([50., -12., 7.])
    normal = surface_anchor_normal(pocket)
    first, index = place_surface_contact(mobile, rotation, pocket.center, normal)
    moved = Pocket(pocket.id, pocket.center @ frame.T + translation, pocket.radius,
                   meta={"detection_mode": "surface_clearance_grid",
                         "receptor_coords_full": receptor @ frame.T + translation})
    second, second_index = place_surface_contact(mobile @ frame.T + translation,
        frame @ rotation @ frame.T, moved.center, surface_anchor_normal(moved))
    assert second_index == index
    np.testing.assert_allclose(second, first @ frame.T + translation, atol=1e-10)


@pytest.mark.parametrize("threshold,expected", [(1.2, False), (.8, True)])
def test_fractional_scan_proxy_cannot_override_configured_final_clash_gate(threshold, expected):
    receptor = np.array([[0., 0., 0.]])
    # One severe contact among 100 atoms passes the old 5% fractional proxy.
    placed = np.array([[1., 0., 0.]] + [[3., 0., 0.]] * 99)
    pocket = Pocket("surface", np.array([3., 0., 0.]), 4.)
    result = scan._evaluate_sample_geometry(scan.build_receptor_kdtree(receptor), receptor,
        placed, pocket, {"severe_clash_threshold": 1.4, "tau_severe_fraction": .05,
                        "contact_min": 2.4, "contact_max": 5.,
                        "hard_clash_min_distance": threshold}, contact_point=pocket.center)
    assert result["sample_proxy_is_feasible"]
    assert result["sample_is_feasible"] is expected
    assert result["sample_hard_clash_gate_passed"] is expected
