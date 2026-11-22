"""Geometric KD-tree scan utilities for pocket feasibility.

Both public scan functions now use a common internal core (`_scan_pocket_core`)
so we can evolve geometric logic without breaking the external contract.

Complexity notes (theoretical):
- KD-tree build over receptor atoms (R): O(R log R)
- Per pocket scan with S samples and L peptide atoms:
  O(S * L * log R) in practice (nearest-neighbor queries)
- Total pre-filter over P pockets:
  O(R log R + P * S * L * log R)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from dockingpp.domain.models import Pocket
from dockingpp.utils.kdtree import build_kdtree


def build_receptor_kdtree(receptor_coords: np.ndarray) -> Any:
    """Build a KD-tree for receptor coordinates."""

    return build_kdtree(np.asarray(receptor_coords, dtype=float))


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_float(cfg: Any, key: str, default: float) -> float:
    try:
        return float(_cfg_value(cfg, key, default))
    except (TypeError, ValueError):
        return float(default)


def _cfg_int(cfg: Any, key: str, default: int) -> int:
    try:
        return int(_cfg_value(cfg, key, default))
    except (TypeError, ValueError):
        return int(default)


def _as_3d_vector(value: Any) -> np.ndarray | None:
    try:
        vec = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vec.size != 3:
        return None
    return vec


def _sample_uniform_unit_vector(rng: np.random.Generator) -> np.ndarray:
    vec = rng.normal(size=3)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return vec / norm


def _sample_uniform_unit_quaternion(rng: np.random.Generator) -> np.ndarray:
    """Uniform random unit quaternion (Shoemake method)."""

    u1, u2, u3 = rng.random(3)
    qx = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    qy = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    qz = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    qw = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    quat = np.array([qw, qx, qy, qz], dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat


def _quaternion_from_axis_angle(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        axis = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        axis = axis / norm
    half = 0.5 * float(angle_rad)
    sin_half = np.sin(half)
    quat = np.array([np.cos(half), sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]], dtype=float)
    quat /= max(float(np.linalg.norm(quat)), 1e-12)
    return quat


def _rotation_matrix_from_quaternion(quat_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quat_wxyz, dtype=float)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _sample_rotation_matrix(rng: np.random.Generator, rotation_max_deg: float | None) -> np.ndarray:
    """Sample rigid-body rotation using quaternion parameterization."""

    if rotation_max_deg is None:
        return _rotation_matrix_from_quaternion(_sample_uniform_unit_quaternion(rng))

    max_deg = float(rotation_max_deg)
    if max_deg <= 0.0:
        return np.eye(3, dtype=float)
    if max_deg >= 179.999:
        return _rotation_matrix_from_quaternion(_sample_uniform_unit_quaternion(rng))

    # Local random rotation with isotropic axis sampling and bounded angle.
    axis = _sample_uniform_unit_vector(rng)
    angle = rng.uniform(-np.deg2rad(max_deg), np.deg2rad(max_deg))
    quat = _quaternion_from_axis_angle(axis, angle)
    return _rotation_matrix_from_quaternion(quat)


def _extract_box_region_from_pocket(pocket: Pocket) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(getattr(pocket, "meta", None), dict):
        return None

    meta = pocket.meta
    candidate_pairs = (
        ("box_min", "box_max"),
        ("bbox_min", "bbox_max"),
        ("min_corner", "max_corner"),
        ("region_min", "region_max"),
    )
    for min_key, max_key in candidate_pairs:
        box_min = _as_3d_vector(meta.get(min_key))
        box_max = _as_3d_vector(meta.get(max_key))
        if box_min is None or box_max is None:
            continue
        if np.any(box_max <= box_min):
            continue
        return box_min, box_max
    return None


def _resolve_pocket_region(pocket: Pocket, cfg_scan: Any) -> dict[str, Any]:
    center = _as_3d_vector(getattr(pocket, "center", np.zeros(3, dtype=float)))
    if center is None:
        center = np.zeros(3, dtype=float)

    box_region = _extract_box_region_from_pocket(pocket)
    if box_region is not None:
        box_min, box_max = box_region
        return {"type": "box", "center": center, "box_min": box_min, "box_max": box_max}

    radius = float(getattr(pocket, "radius", 0.0) or 0.0)
    min_radius = _cfg_float(cfg_scan, "min_effective_radius", 1e-3)
    radius = max(radius, min_radius)
    return {"type": "sphere", "center": center, "radius": radius}


def _sample_translation_from_region(
    rng: np.random.Generator,
    region: dict[str, Any],
    jitter_sigma: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    region_type = str(region.get("type", "sphere"))
    center = np.asarray(region.get("center", np.zeros(3, dtype=float)), dtype=float)
    jitter = np.zeros(3, dtype=float)
    if jitter_sigma > 0.0:
        jitter = rng.normal(loc=0.0, scale=float(jitter_sigma), size=3)

    if region_type == "box":
        box_min = np.asarray(region.get("box_min", center), dtype=float)
        box_max = np.asarray(region.get("box_max", center), dtype=float)
        sampled = rng.uniform(low=box_min, high=box_max)
        sampled = np.clip(sampled + jitter, box_min, box_max)
        return sampled, sampled - center

    radius = float(region.get("radius", 0.0) or 0.0)
    if radius <= 0.0:
        sampled = center + jitter
        return sampled, sampled - center

    direction = _sample_uniform_unit_vector(rng)
    rho = radius * (float(rng.random()) ** (1.0 / 3.0))
    sampled = center + (direction * rho) + jitter
    delta = sampled - center
    norm = float(np.linalg.norm(delta))
    if norm > radius and norm > 0.0:
        sampled = center + (delta / norm) * radius
    return sampled, sampled - center


def _sample_pocket_transforms(
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
    n_samples: int,
) -> list[dict[str, Any]]:
    """Sample rigid transforms constrained by the pocket region."""

    _ = np.asarray(peptide_coords, dtype=float)  # future shape-aware sampling hook
    region = _resolve_pocket_region(pocket, cfg_scan)
    rotation_max_deg_raw = _cfg_value(cfg_scan, "rotation_max_deg", None)
    rotation_max_deg = None if rotation_max_deg_raw is None else float(rotation_max_deg_raw)
    jitter_sigma = _cfg_float(
        cfg_scan,
        "translation_local_sigma",
        _cfg_float(cfg_scan, "translation_sigma", 0.0),
    )

    transforms: list[dict[str, Any]] = []
    for _idx in range(max(0, int(n_samples))):
        rotation = _sample_rotation_matrix(rng, rotation_max_deg)
        translation, sampled_offset = _sample_translation_from_region(rng, region, jitter_sigma=jitter_sigma)
        transforms.append(
            {
                "rotation": rotation,
                "translation": translation,
                "sampled_offset": sampled_offset,
                "region_type": region.get("type", "sphere"),
            }
        )
    return transforms


def _rigid_transform(peptide_coords: np.ndarray, rotation: np.ndarray, new_center: np.ndarray) -> np.ndarray:
    coords = np.asarray(peptide_coords, dtype=float)
    if coords.size == 0:
        return coords.reshape(0, 3)
    center = coords.mean(axis=0)
    base_coords = coords - center
    return base_coords @ np.asarray(rotation, dtype=float).T + np.asarray(new_center, dtype=float)


def _receptor_points(receptor_kdtree: Any) -> np.ndarray:
    if isinstance(receptor_kdtree, dict):
        return np.asarray(receptor_kdtree.get("points", []), dtype=float)
    return np.asarray(getattr(receptor_kdtree, "data", []), dtype=float)


def _nearest_dists(receptor_kdtree: Any, points: np.ndarray, receptor_points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return np.zeros((0,), dtype=float)
    if receptor_points.size == 0:
        return np.full((pts.shape[0],), np.inf, dtype=float)

    query = getattr(receptor_kdtree, "query", None)
    if callable(query):
        dists, _ = query(pts, k=1)
        return np.asarray(dists, dtype=float).reshape(-1)

    # Fallback path without scipy.
    deltas = receptor_points[None, :, :] - pts[:, None, :]
    return np.linalg.norm(deltas, axis=2).min(axis=1)


def _distance_outside_box(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> float:
    below = np.maximum(box_min - point, 0.0)
    above = np.maximum(point - box_max, 0.0)
    return float(np.linalg.norm(below + above))


def _evaluate_sample_geometry(
    receptor_kdtree: Any,
    receptor_points: np.ndarray,
    placed_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    contact_point: np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate one sampled pose using KD-tree nearest distances."""

    coords = np.asarray(placed_coords, dtype=float)
    n_atoms = int(coords.shape[0]) if coords.ndim == 2 else 0
    if n_atoms <= 0:
        return {
            "raw_contact_count": 0.0,
            "frac_severe": 1.0,
            "frac_moderate": 0.0,
            "frac_contact": 0.0,
            "severe_atom_count": 0.0,
            "moderate_atom_count": 0.0,
            "pen_severe": 0.0,
            "pen_moderate": 0.0,
            "pen_col": 25.0,
            "pen_exp": 10.0,
            "mean_d": float("inf"),
            "min_d": float("inf"),
            "p10_d": float("inf"),
            "energy_sample": 100.0,
            "sample_is_feasible": False,
            "severe_binary": 1.0,
            "reason": "empty_peptide_pose",
        }

    severe_th = _cfg_float(cfg_scan, "severe_clash_threshold", 1.0)
    moderate_th = _cfg_float(cfg_scan, "moderate_clash_threshold", max(severe_th + 1e-6, 1.5))
    if moderate_th <= severe_th:
        moderate_th = severe_th + 1e-6
    contact_min = _cfg_float(cfg_scan, "contact_min", 1.5)
    contact_max = _cfg_float(cfg_scan, "contact_max", 3.0)
    if contact_max < contact_min:
        contact_max = contact_min

    w_severe = _cfg_float(cfg_scan, "collision_weight_severe", 12.0)
    w_moderate = _cfg_float(cfg_scan, "collision_weight_moderate", 3.0)
    lambda_exp = _cfg_float(cfg_scan, "energy_lambda_exp", 1.0)
    lambda_contact = _cfg_float(cfg_scan, "energy_lambda_contact", 1.0)

    tau_severe = _cfg_float(cfg_scan, "tau_severe_fraction", 0.05)
    tau_moderate = _cfg_float(cfg_scan, "tau_moderate_fraction", 0.35)
    tau_exp = _cfg_float(cfg_scan, "tau_exposure_penalty", 0.5)
    tau_contact_min = _cfg_float(cfg_scan, "tau_contact_fraction_min", 0.01)

    dists = _nearest_dists(receptor_kdtree, coords, receptor_points)
    severe_mask = dists < severe_th
    moderate_mask = (dists >= severe_th) & (dists < moderate_th)
    contact_mask = (dists >= contact_min) & (dists <= contact_max)

    severe_count = float(np.sum(severe_mask))
    moderate_count = float(np.sum(moderate_mask))
    raw_contact_count = float(np.sum(contact_mask))
    denom = float(max(1, n_atoms))

    frac_severe = severe_count / denom
    frac_moderate = moderate_count / denom
    frac_contact = raw_contact_count / denom

    pen_severe = float(np.sum((severe_th - dists[severe_mask]) ** 2))
    pen_moderate = float(np.sum((moderate_th - dists[moderate_mask]) ** 2))
    pen_col = (w_severe * pen_severe) + (w_moderate * pen_moderate)

    # For a surface site, the region constrains the actual contacting support
    # atom; a large mobile partner's centroid is naturally outside that region.
    centroid = coords.mean(axis=0) if contact_point is None else np.asarray(contact_point, dtype=float)
    exposure_margin = _cfg_float(cfg_scan, "exposure_margin", 0.0)
    box_region = _extract_box_region_from_pocket(pocket)
    if box_region is not None:
        box_min, box_max = box_region
        if exposure_margin > 0.0:
            box_min = box_min - exposure_margin
            box_max = box_max + exposure_margin
        pen_exp = _distance_outside_box(centroid, box_min, box_max)
    else:
        pocket_center = _as_3d_vector(getattr(pocket, "center", np.zeros(3, dtype=float)))
        if pocket_center is None:
            pocket_center = np.zeros(3, dtype=float)
        pocket_radius = float(getattr(pocket, "radius", 0.0) or 0.0)
        effective_radius = max(0.0, pocket_radius + exposure_margin)
        delta_center = float(np.linalg.norm(centroid - pocket_center))
        pen_exp = max(0.0, delta_center - effective_radius)

    mean_d = float(np.mean(dists)) if dists.size else float("inf")
    min_d = float(np.min(dists)) if dists.size else float("inf")
    p10_d = float(np.quantile(dists, 0.10)) if dists.size else float("inf")

    sample_is_feasible = bool(
        frac_severe <= tau_severe
        and frac_moderate <= tau_moderate
        and pen_exp <= tau_exp
        and frac_contact >= tau_contact_min
    )
    sample_proxy_is_feasible = sample_is_feasible
    hard_gate_enabled = contact_point is not None and bool(_cfg_value(cfg_scan, "hard_clash_gate_enabled", True))
    hard_gate_threshold = _cfg_float(cfg_scan, "hard_clash_min_distance", 1.2)
    sample_hard_clash_gate_passed = not hard_gate_enabled or min_d >= hard_gate_threshold
    sample_is_feasible = sample_is_feasible and sample_hard_clash_gate_passed
    energy_sample = float(pen_col + (lambda_exp * pen_exp) - (lambda_contact * frac_contact))

    return {
        "raw_contact_count": raw_contact_count,
        "frac_severe": frac_severe,
        "frac_moderate": frac_moderate,
        "frac_contact": frac_contact,
        "severe_atom_count": severe_count,
        "moderate_atom_count": moderate_count,
        "pen_severe": pen_severe,
        "pen_moderate": pen_moderate,
        "pen_col": pen_col,
        "pen_exp": float(pen_exp),
        "mean_d": mean_d,
        "min_d": min_d,
        "p10_d": p10_d,
        "energy_sample": energy_sample,
        "sample_is_feasible": sample_is_feasible,
        "sample_proxy_is_feasible": sample_proxy_is_feasible,
        "sample_hard_clash_gate_passed": sample_hard_clash_gate_passed,
        "sample_hard_clash_threshold": hard_gate_threshold if hard_gate_enabled else None,
        "severe_binary": 1.0 if frac_severe > 0.0 else 0.0,
    }


def _wilson_score_interval(
    successes: int,
    trials: int,
    *,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    """Return a dependency-free 95% Wilson interval for a binomial rate.

    The interval is deliberately computed from integer event counts instead of
    treating a sampled fraction as exact.  With no observations, ``[0, 1]``
    records complete uncertainty instead of manufacturing evidence.
    """

    n = max(0, int(trials))
    if n == 0:
        return 0.0, 1.0
    k = min(n, max(0, int(successes)))
    proportion = float(k) / float(n)
    z_squared = float(z) ** 2
    denominator = 1.0 + (z_squared / float(n))
    center = (proportion + (z_squared / (2.0 * float(n)))) / denominator
    half_width = (
        float(z)
        * np.sqrt(
            (proportion * (1.0 - proportion) / float(n))
            + (z_squared / (4.0 * float(n) ** 2))
        )
        / denominator
    )
    return max(0.0, float(center - half_width)), min(1.0, float(center + half_width))


def _feasible_pose_statistics(sample_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe usable poses without rewarding contacts from rejected samples.

    ``sample_is_feasible`` already includes the surface hard-clash veto. Keep
    the statistics of all attempts separately for acceptance and diagnostics.
    """

    accepted = [item for item in sample_stats if bool(item.get("sample_is_feasible", False))]
    if not accepted:
        return {
            "n_samples": 0,
            "energy_best": 100.0,
            "energy_mean": 100.0,
            "energy_std": 0.0,
            "energy_q20": 100.0,
            "mean_contacts": 0.0,
            "contact_fraction_mean": 0.0,
            "best_contact_count": 0.0,
            "moderate_clash_mean": 1.0,
            "mean_exposure_penalty": 10.0,
            "best_sample_meta": None,
        }
    energies = np.asarray([float(item.get("energy_sample", 100.0)) for item in accepted])
    contacts = np.asarray([float(item.get("raw_contact_count", 0.0)) for item in accepted])
    return {
        "n_samples": len(accepted),
        "energy_best": float(np.min(energies)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "energy_q20": float(np.quantile(energies, 0.20)),
        "mean_contacts": float(np.mean(contacts)),
        "contact_fraction_mean": float(np.mean([float(item.get("frac_contact", 0.0)) for item in accepted])),
        "best_contact_count": float(np.max(contacts)),
        "moderate_clash_mean": float(np.mean([float(item.get("frac_moderate", 1.0)) for item in accepted])),
        "mean_exposure_penalty": float(np.mean([float(item.get("pen_exp", 0.0)) for item in accepted])),
        "best_sample_meta": dict(accepted[int(np.argmin(energies))]),
    }


def _aggregate_sample_statistics(
    sample_stats: list[dict[str, Any]],
    *,
    n_samples_requested: int,
    n_atoms_peptide: int,
    empty_reason: str | None = None,
) -> dict[str, Any]:
    if not sample_stats:
        return {
            "n_samples_requested": int(n_samples_requested),
            "n_samples": 0,
            "n_atoms_peptide": int(n_atoms_peptide),
            "energy_best": 100.0,
            "energy_mean": 100.0,
            "energy_std": 0.0,
            "energy_q20": 100.0,
            "feasible_fraction": 0.0,
            "feasible_samples": 0,
            "feasible_fraction_ci95_low": 0.0,
            "feasible_fraction_ci95_high": 1.0,
            "severe_clash_fraction": 1.0,
            "severe_clash_samples": 0,
            "severe_clash_fraction_ci95_low": 0.0,
            "severe_clash_fraction_ci95_high": 1.0,
            "proportion_ci_method": "wilson_score",
            "proportion_ci_confidence": 0.95,
            "moderate_clash_mean": 1.0,
            "mean_contacts": 0.0,
            "contact_fraction_mean": 0.0,
            "best_contact_count": 0.0,
            "mean_exposure_penalty": 10.0,
            "stability_penalty": 0.0,
            "best_sample_meta": {"reason": empty_reason or "no_samples"},
            "clash_ratio_best_legacy": 1.0,
            "best_clashes_legacy": float(max(1, n_atoms_peptide)),
            "mean_d_global": float("inf"),
            "min_d_global": float("inf"),
            "p10_d_global": float("inf"),
            "raw_samples": [],
            "feasible_pose_statistics": _feasible_pose_statistics([]),
            "ranking_statistics_scope": "feasible_samples_v1",
        }

    energies = np.array([float(item.get("energy_sample", 100.0)) for item in sample_stats], dtype=float)
    feasible = np.array([1.0 if bool(item.get("sample_is_feasible", False)) else 0.0 for item in sample_stats], dtype=float)
    severe_binary = np.array([float(item.get("severe_binary", 1.0)) for item in sample_stats], dtype=float)
    frac_moderate = np.array([float(item.get("frac_moderate", 1.0)) for item in sample_stats], dtype=float)
    contacts = np.array([float(item.get("raw_contact_count", 0.0)) for item in sample_stats], dtype=float)
    frac_contact = np.array([float(item.get("frac_contact", 0.0)) for item in sample_stats], dtype=float)
    exposure = np.array([float(item.get("pen_exp", 0.0)) for item in sample_stats], dtype=float)
    frac_severe = np.array([float(item.get("frac_severe", 1.0)) for item in sample_stats], dtype=float)
    severe_counts = np.array([float(item.get("severe_atom_count", 0.0)) for item in sample_stats], dtype=float)
    moderate_counts = np.array([float(item.get("moderate_atom_count", 0.0)) for item in sample_stats], dtype=float)
    mean_d = np.array([float(item.get("mean_d", np.inf)) for item in sample_stats], dtype=float)
    min_d = np.array([float(item.get("min_d", np.inf)) for item in sample_stats], dtype=float)
    p10_d = np.array([float(item.get("p10_d", np.inf)) for item in sample_stats], dtype=float)

    best_idx = int(np.argmin(energies))
    best_meta = dict(sample_stats[best_idx])
    feasible_samples = int(np.sum(feasible))
    severe_clash_samples = int(np.sum(severe_binary))
    feasible_ci_low, feasible_ci_high = _wilson_score_interval(
        feasible_samples,
        len(sample_stats),
    )
    severe_ci_low, severe_ci_high = _wilson_score_interval(
        severe_clash_samples,
        len(sample_stats),
    )

    return {
        "n_samples_requested": int(n_samples_requested),
        "n_samples": int(len(sample_stats)),
        "n_atoms_peptide": int(n_atoms_peptide),
        "energy_best": float(np.min(energies)),
        "energy_mean": float(np.mean(energies)),
        "energy_std": float(np.std(energies)),
        "energy_q20": float(np.quantile(energies, 0.20)),
        "feasible_fraction": float(np.mean(feasible)),
        "feasible_samples": feasible_samples,
        "feasible_fraction_ci95_low": feasible_ci_low,
        "feasible_fraction_ci95_high": feasible_ci_high,
        "severe_clash_fraction": float(np.mean(severe_binary)),
        "severe_clash_samples": severe_clash_samples,
        "severe_clash_fraction_ci95_low": severe_ci_low,
        "severe_clash_fraction_ci95_high": severe_ci_high,
        "proportion_ci_method": "wilson_score",
        "proportion_ci_confidence": 0.95,
        "moderate_clash_mean": float(np.mean(frac_moderate)),
        "mean_contacts": float(np.mean(contacts)),
        "contact_fraction_mean": float(np.mean(frac_contact)),
        "best_contact_count": float(np.max(contacts)),
        "mean_exposure_penalty": float(np.mean(exposure)),
        "stability_penalty": float(np.std(energies)),
        "best_sample_meta": best_meta,
        "clash_ratio_best_legacy": float(np.min(np.clip(frac_severe + frac_moderate, 0.0, 1.0))),
        "best_clashes_legacy": float(np.min(severe_counts + moderate_counts)),
        "mean_d_global": float(np.mean(mean_d)),
        "min_d_global": float(np.min(min_d)),
        "p10_d_global": float(np.mean(p10_d)),
        "raw_samples": sample_stats,
        "feasible_pose_statistics": _feasible_pose_statistics(sample_stats),
        "ranking_statistics_scope": "feasible_samples_v1",
    }


def _resolve_robust_score_coefficients(cfg_scan: Any) -> dict[str, float]:
    # New names.
    a1 = _cfg_float(cfg_scan, "score_a1", 1.5)
    a2 = _cfg_float(cfg_scan, "score_a2", 0.5)
    a3 = _cfg_float(cfg_scan, "score_a3", 0.35)
    b1 = _cfg_float(cfg_scan, "score_b1", _cfg_float(cfg_scan, "score_alpha", 1.0))
    b2 = _cfg_float(cfg_scan, "score_b2", _cfg_float(cfg_scan, "score_beta", 0.5))
    b3 = _cfg_float(cfg_scan, "score_b3", 0.25)
    c1 = _cfg_float(cfg_scan, "score_c1", _cfg_float(cfg_scan, "score_gamma", 1.0))
    c2 = _cfg_float(cfg_scan, "score_c2", 0.75)
    c3 = _cfg_float(cfg_scan, "score_c3", _cfg_float(cfg_scan, "score_delta", 0.5))
    return {
        "a1": a1,
        "a2": a2,
        "a3": a3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
        "c1": c1,
        "c2": c2,
        "c3": c3,
    }


def _compute_robust_pocket_score(core_stats: dict[str, Any], cfg_scan: Any) -> tuple[float, dict[str, Any]]:
    coeffs = _resolve_robust_score_coefficients(cfg_scan)
    n_atoms = max(1, int(core_stats.get("n_atoms_peptide", 0)))
    # Older scan tables have only aggregate fields; retain that read contract.
    pose_stats = core_stats.get("feasible_pose_statistics")
    if not isinstance(pose_stats, dict):
        pose_stats = core_stats
    normalized_best_contact = float(pose_stats.get("best_contact_count", 0.0)) / float(n_atoms)

    terms = {
        "energy_q20": float(pose_stats.get("energy_q20", 100.0)),
        "energy_mean": float(pose_stats.get("energy_mean", 100.0)),
        "energy_std": float(pose_stats.get("energy_std", 0.0)),
        "feasible_fraction": float(core_stats.get("feasible_fraction", 0.0)),
        "contact_fraction_mean": float(pose_stats.get("contact_fraction_mean", 0.0)),
        "normalized_best_contact": float(normalized_best_contact),
        "severe_clash_fraction": float(core_stats.get("severe_clash_fraction", 1.0)),
        "moderate_clash_mean": float(pose_stats.get("moderate_clash_mean", 1.0)),
        "mean_exposure_penalty": float(pose_stats.get("mean_exposure_penalty", 10.0)),
    }

    contributions = {
        "energy_q20": -coeffs["a1"] * terms["energy_q20"],
        "energy_mean": -coeffs["a2"] * terms["energy_mean"],
        "energy_std": -coeffs["a3"] * terms["energy_std"],
        "feasible_fraction": coeffs["b1"] * terms["feasible_fraction"],
        "contact_fraction_mean": coeffs["b2"] * terms["contact_fraction_mean"],
        "normalized_best_contact": coeffs["b3"] * terms["normalized_best_contact"],
        "severe_clash_fraction": -coeffs["c1"] * terms["severe_clash_fraction"],
        "moderate_clash_mean": -coeffs["c2"] * terms["moderate_clash_mean"],
        "mean_exposure_penalty": -coeffs["c3"] * terms["mean_exposure_penalty"],
    }
    robust_score = float(sum(contributions.values()))
    score_components = {
        "coefficients": coeffs,
        "terms": terms,
        "contributions": contributions,
        "total": robust_score,
        "pose_statistics_scope": core_stats.get("ranking_statistics_scope", "all_samples_legacy"),
    }
    return robust_score, score_components


def _scan_pocket_core(
    receptor_kdtree: Any,
    receptor_coords: np.ndarray,
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Common scan core for both legacy and geom_kdtree public entrypoints."""

    receptor_points = np.asarray(receptor_coords, dtype=float)
    coords = np.asarray(peptide_coords, dtype=float)

    requested_samples = _cfg_int(cfg_scan, "samples_per_pocket", 64)
    min_samples = _cfg_value(cfg_scan, "min_samples_per_pocket", None)
    max_samples = _cfg_value(cfg_scan, "max_samples_per_pocket", None)
    n_samples = max(0, int(requested_samples))
    if min_samples is not None:
        n_samples = max(n_samples, max(0, int(min_samples)))
    if max_samples is not None:
        n_samples = min(n_samples, max(0, int(max_samples)))

    early_stop_enabled = bool(_cfg_value(cfg_scan, "early_stop_enabled", True))
    early_stop_min_samples = max(1, _cfg_int(cfg_scan, "early_stop_min_samples", 8))
    early_stop_severe_clash_fraction_geq = _cfg_float(cfg_scan, "early_stop_severe_clash_fraction_geq", 0.98)
    early_stop_feasible_fraction_leq = _cfg_float(cfg_scan, "early_stop_feasible_fraction_leq", 0.0)
    early_stop_contact_fraction_leq = _cfg_float(cfg_scan, "early_stop_contact_fraction_leq", 0.02)

    empty_reason = None
    early_stop_triggered = False
    early_stop_reason: str | None = None
    early_stop_sample_idx: int | None = None
    sample_stats: list[dict[str, Any]] = []
    if coords.size == 0:
        empty_reason = "empty_peptide"
    elif receptor_points.size == 0:
        empty_reason = "empty_receptor"
    elif n_samples <= 0:
        empty_reason = "no_samples_requested"
    else:
        transforms = _sample_pocket_transforms(coords, pocket, cfg_scan, rng, n_samples=n_samples)
        from dockingpp.utils.geometry import place_surface_contact, surface_anchor_normal
        surface_normal = surface_anchor_normal(pocket) if _cfg_value(cfg_scan, "surface_anchor_enabled", True) else None
        running_feasible_sum = 0.0
        running_severe_sum = 0.0
        running_contact_sum = 0.0
        for sample_idx, transform in enumerate(transforms):
            placed = _rigid_transform(coords, transform["rotation"], transform["translation"])
            contact_point = None
            support_index = None
            if surface_normal is not None:
                contact_point = np.asarray(transform["translation"], dtype=float)
                placed, support_index = place_surface_contact(
                    coords, transform["rotation"], contact_point, surface_normal,
                )
            stats = _evaluate_sample_geometry(
                receptor_kdtree=receptor_kdtree,
                receptor_points=receptor_points,
                placed_coords=placed,
                pocket=pocket,
                cfg_scan=cfg_scan,
                contact_point=contact_point,
            )
            stats["sample_idx"] = int(sample_idx)
            stats["translation"] = np.asarray(transform["translation"], dtype=float).tolist()
            stats["sampled_offset"] = np.asarray(transform["sampled_offset"], dtype=float).tolist()
            stats["rotation_matrix"] = np.asarray(transform["rotation"], dtype=float).tolist()
            stats["region_type"] = str(transform.get("region_type", "sphere"))
            if surface_normal is not None:
                stats["surface_anchor_version"] = "surface_contact_v1"
                stats["surface_support_atom_index"] = support_index
                stats["surface_contact_point"] = contact_point.tolist()
                stats["placed_centroid"] = placed.mean(axis=0).tolist()
            sample_stats.append(stats)

            running_feasible_sum += 1.0 if bool(stats.get("sample_is_feasible", False)) else 0.0
            running_severe_sum += float(stats.get("severe_binary", 1.0))
            running_contact_sum += float(stats.get("frac_contact", 0.0))
            n_seen = float(sample_idx + 1)
            if early_stop_enabled and n_seen >= float(early_stop_min_samples):
                running_feasible_fraction = running_feasible_sum / n_seen
                running_severe_fraction = running_severe_sum / n_seen
                running_contact_fraction = running_contact_sum / n_seen
                if (
                    running_feasible_fraction <= early_stop_feasible_fraction_leq
                    and running_severe_fraction >= early_stop_severe_clash_fraction_geq
                    and running_contact_fraction <= early_stop_contact_fraction_leq
                ):
                    early_stop_triggered = True
                    early_stop_reason = "clearly_inviable_partial_scan"
                    early_stop_sample_idx = int(sample_idx)
                    break

    aggregated = _aggregate_sample_statistics(
        sample_stats,
        n_samples_requested=int(requested_samples),
        n_atoms_peptide=int(coords.shape[0] if coords.ndim == 2 else 0),
        empty_reason=empty_reason,
    )
    aggregated["scan_early_stop_enabled"] = bool(early_stop_enabled)
    aggregated["scan_early_stop_triggered"] = bool(early_stop_triggered)
    aggregated["scan_early_stop_reason"] = early_stop_reason
    aggregated["scan_early_stop_sample_idx"] = early_stop_sample_idx
    robust_score, score_components = _compute_robust_pocket_score(aggregated, cfg_scan)
    aggregated["robust_score"] = robust_score
    aggregated["score_components"] = score_components
    aggregated["selector_mode_used"] = "core"
    return aggregated


def _adapt_core_stats_to_legacy_schema(core_stats: dict[str, Any]) -> dict[str, float]:
    return {
        "best_contacts": float(core_stats.get("best_contact_count", 0.0)),
        "mean_contacts": float(core_stats.get("mean_contacts", 0.0)),
        "best_clashes": float(core_stats.get("best_clashes_legacy", 0.0)),
        "clash_ratio_best": float(core_stats.get("clash_ratio_best_legacy", 1.0)),
        "feasible_fraction": float(core_stats.get("feasible_fraction", 0.0)),
        "feasible_samples": int(core_stats.get("feasible_samples", 0)),
        "feasible_fraction_ci95_low": float(core_stats.get("feasible_fraction_ci95_low", 0.0)),
        "feasible_fraction_ci95_high": float(core_stats.get("feasible_fraction_ci95_high", 1.0)),
        "proportion_ci_method": str(core_stats.get("proportion_ci_method", "wilson_score")),
        "proportion_ci_confidence": float(core_stats.get("proportion_ci_confidence", 0.95)),
        "scan_score": float(core_stats.get("robust_score", float("-inf"))),
        # Optional compatibility extensions.
        "energy_q20": float(core_stats.get("energy_q20", 100.0)),
        "energy_mean": float(core_stats.get("energy_mean", 100.0)),
        "energy_std": float(core_stats.get("energy_std", 0.0)),
        "contact_fraction_mean": float(core_stats.get("contact_fraction_mean", 0.0)),
        "stability_penalty": float(core_stats.get("stability_penalty", 0.0)),
        "feasible_pose_statistics": core_stats.get("feasible_pose_statistics"),
        "ranking_statistics_scope": core_stats.get("ranking_statistics_scope", "all_samples_legacy"),
        "n_samples": int(core_stats.get("n_samples", 0)),
        "scan_early_stop_enabled": bool(core_stats.get("scan_early_stop_enabled", False)),
        "scan_early_stop_triggered": bool(core_stats.get("scan_early_stop_triggered", False)),
        "scan_early_stop_reason": core_stats.get("scan_early_stop_reason"),
        "scan_early_stop_sample_idx": core_stats.get("scan_early_stop_sample_idx"),
        "score_components": core_stats.get("score_components", {}),
        "selector_mode": "legacy",
        "selector_mode_used": "legacy",
    }


def _adapt_core_stats_to_geom_schema(core_stats: dict[str, Any], cfg_scan: Any) -> dict[str, Any]:
    reject_feasible_leq = _cfg_float(cfg_scan, "reject_if_feasible_fraction_leq", 0.0)
    robust_score = float(core_stats.get("robust_score", float("-inf")))
    energy_best = float(core_stats.get("energy_best", 100.0))
    clash_ratio_best = float(core_stats.get("clash_ratio_best_legacy", 1.0))

    return {
        "feasible_fraction": float(core_stats.get("feasible_fraction", 0.0)),
        "feasible_samples": int(core_stats.get("feasible_samples", 0)),
        "feasible_fraction_ci95_low": float(core_stats.get("feasible_fraction_ci95_low", 0.0)),
        "feasible_fraction_ci95_high": float(core_stats.get("feasible_fraction_ci95_high", 1.0)),
        "severe_clash_fraction": float(core_stats.get("severe_clash_fraction", 1.0)),
        "severe_clash_samples": int(core_stats.get("severe_clash_samples", 0)),
        "severe_clash_fraction_ci95_low": float(core_stats.get("severe_clash_fraction_ci95_low", 0.0)),
        "severe_clash_fraction_ci95_high": float(core_stats.get("severe_clash_fraction_ci95_high", 1.0)),
        "proportion_ci_method": str(core_stats.get("proportion_ci_method", "wilson_score")),
        "proportion_ci_confidence": float(core_stats.get("proportion_ci_confidence", 0.95)),
        "moderate_clash_mean": float(core_stats.get("moderate_clash_mean", 1.0)),
        "mean_contacts": float(core_stats.get("mean_contacts", 0.0)),
        "best_contact_count": float(core_stats.get("best_contact_count", 0.0)),
        "best_geom_energy": energy_best,
        "mean_exposure_penalty": float(core_stats.get("mean_exposure_penalty", 10.0)),
        "pocket_scan_score": robust_score,
        "n_samples": int(core_stats.get("n_samples", 0)),
        "best_sample_meta": core_stats.get("best_sample_meta", {}),
        "selector_mode": "geom_kdtree",
        "selector_mode_used": "geom_kdtree",
        "canonical_geom_fields": ["pocket_scan_score", "severe_clash_fraction", "best_geom_energy"],
        "compatibility_fields": ["scan_score", "clash_ratio_best"],
        "scan_score": robust_score,
        "clash_ratio_best": clash_ratio_best,
        "reject_if_feasible_fraction_leq": reject_feasible_leq,
        # New robust fields (optional for downstream).
        "energy_q20": float(core_stats.get("energy_q20", 100.0)),
        "energy_mean": float(core_stats.get("energy_mean", 100.0)),
        "energy_std": float(core_stats.get("energy_std", 0.0)),
        "contact_fraction_mean": float(core_stats.get("contact_fraction_mean", 0.0)),
        "stability_penalty": float(core_stats.get("stability_penalty", 0.0)),
        "feasible_pose_statistics": core_stats.get("feasible_pose_statistics"),
        "ranking_statistics_scope": core_stats.get("ranking_statistics_scope", "all_samples_legacy"),
        "scan_early_stop_enabled": bool(core_stats.get("scan_early_stop_enabled", False)),
        "scan_early_stop_triggered": bool(core_stats.get("scan_early_stop_triggered", False)),
        "scan_early_stop_reason": core_stats.get("scan_early_stop_reason"),
        "scan_early_stop_sample_idx": core_stats.get("scan_early_stop_sample_idx"),
        "score_components": core_stats.get("score_components", {}),
    }


def scan_pocket_feasibility(
    receptor_kdtree: Any,
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """Legacy scan entrypoint kept for compatibility."""

    receptor_points = _receptor_points(receptor_kdtree)
    core_stats = _scan_pocket_core(
        receptor_kdtree=receptor_kdtree,
        receptor_coords=receptor_points,
        peptide_coords=np.asarray(peptide_coords, dtype=float),
        pocket=pocket,
        cfg_scan=cfg_scan,
        rng=rng,
    )
    return _adapt_core_stats_to_legacy_schema(core_stats)


def scan_pocket_feasibility_geom_kdtree(
    receptor_kdtree: Any,
    receptor_coords: np.ndarray,
    peptide_coords: np.ndarray,
    pocket: Pocket,
    cfg_scan: Any,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Geom KD-tree scan entrypoint kept compatible with existing schema."""

    core_stats = _scan_pocket_core(
        receptor_kdtree=receptor_kdtree,
        receptor_coords=np.asarray(receptor_coords, dtype=float),
        peptide_coords=np.asarray(peptide_coords, dtype=float),
        pocket=pocket,
        cfg_scan=cfg_scan,
        rng=rng,
    )
    return _adapt_core_stats_to_geom_schema(core_stats, cfg_scan)


def _safe_sort_float(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(parsed):
        return float(default)
    return parsed


def _scan_selector_score(metrics: Dict[str, Any], mode: str) -> float:
    if mode == "geom_kdtree":
        primary = _safe_sort_float(metrics.get("pocket_scan_score"), float("nan"))
        if not np.isnan(primary):
            return primary
        return _safe_sort_float(metrics.get("scan_score"), float("-inf"))
    primary = _safe_sort_float(metrics.get("scan_score"), float("nan"))
    if not np.isnan(primary):
        return primary
    return _safe_sort_float(metrics.get("pocket_scan_score"), float("-inf"))


def _scan_selector_clash(metrics: Dict[str, Any], mode: str) -> float:
    if mode == "geom_kdtree":
        return _safe_sort_float(metrics.get("severe_clash_fraction", metrics.get("clash_ratio_best")), 1.0)
    return _safe_sort_float(metrics.get("clash_ratio_best", metrics.get("severe_clash_fraction")), 1.0)


def _scan_selector_sort_key(
    metrics: Dict[str, Any],
    *,
    mode: str,
    original_index: int,
) -> tuple[float, float, float, float, float, float, float, int]:
    """Build a stable ranking key prioritizing viable receptor interaction.

    Ordering semantics:
    1. Viability bucket (`feasible_fraction > 0`) descending.
    2. Mean receptor-contact fraction of feasible samples descending.
    3. Feasible fraction descending.
    4. Clash severity ascending.
    5. Robust energy quantile of feasible samples ascending.
    6. Selector score descending.
    7. Number of samples descending.
    8. Original order ascending (stable deterministic fallback).

    A nonzero feasible fraction remains a hard ranking gate.  Inside that
    viable set, contact precedes open-space feasibility so broad exposed
    surfaces do not outrank tighter interaction sites merely because random
    placements fit there more often.
    """

    feasible = _safe_sort_float(metrics.get("feasible_fraction", 0.0), 0.0)
    pose_stats = metrics.get("feasible_pose_statistics")
    if not isinstance(pose_stats, dict):
        pose_stats = metrics
    contact = _safe_sort_float(pose_stats.get("contact_fraction_mean", 0.0), 0.0)
    clash = _scan_selector_clash(metrics, mode)
    energy_q20 = _safe_sort_float(pose_stats.get("energy_q20", float("inf")), float("inf"))
    score = _scan_selector_score(metrics, mode)
    n_samples = _safe_sort_float(metrics.get("n_samples", 0.0), 0.0)
    viable_bucket = 1.0 if feasible > 0.0 else 0.0
    return (
        -viable_bucket,
        -contact,
        -feasible,
        clash,
        energy_q20,
        -score,
        -n_samples,
        int(original_index),
    )


def select_pockets_from_scan(
    pockets: List[Pocket],
    scan_table: Dict[str, Dict[str, float]],
    top_k: int | None,
    selector_mode: str = "legacy",
) -> List[Pocket]:
    """Select pockets from scan table while preserving public behavior."""

    mode = str(selector_mode or "legacy").strip().lower()
    scored: List[
        Tuple[
            tuple[float, float, float, float, float, float, float, int],
            Pocket,
        ]
    ] = []
    for idx, pocket in enumerate(pockets):
        metrics = scan_table.get(str(pocket.id), {})
        sort_key = _scan_selector_sort_key(metrics, mode=mode, original_index=idx)
        scored.append((sort_key, pocket))

    scored.sort(key=lambda item: item[0])
    # The selection count must not change the ranking contract. This order
    # also controls which pocket wins under the scan_priority policy.
    limit = len(scored) if top_k is None or int(top_k) <= 0 else int(top_k)
    return [item[1] for item in scored[:limit]]
