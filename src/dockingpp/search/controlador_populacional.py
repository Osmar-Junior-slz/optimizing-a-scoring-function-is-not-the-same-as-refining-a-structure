"""ControladorPopulacional: confidence-guided population audit and curation.

This module keeps the population-confidence heuristic separate from the search
algorithms. The algorithms still generate and evolve candidates; the controller
only audits the population and provides an auxiliary ordering when confidence is
low enough to justify guidance.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dockingpp.domain.models import Pose
from dockingpp.utils.topk import topk_indices


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def confidence_policy(cfg: Any) -> dict[str, Any]:
    raw = cfg_value(cfg, "confidence_policy", {}) or {}
    return dict(raw) if isinstance(raw, dict) else {}


def confidence_search_control_enabled(cfg: Any) -> bool:
    policy = confidence_policy(cfg)
    if not bool(policy.get("enabled", False)):
        return False
    mode = str(policy.get("mode", "disabled") or "disabled").strip().lower()
    if mode not in {"assist_selection", "assist_all"}:
        return False
    search_cfg = policy.get("search_control")
    if isinstance(search_cfg, dict) and "enabled" in search_cfg:
        return bool(search_cfg.get("enabled"))
    direct = cfg_value(cfg, "confidence_search_control_enabled", None)
    if direct is not None:
        return bool(direct)
    return True


def confidence_search_cfg(cfg: Any) -> dict[str, Any]:
    policy = confidence_policy(cfg)
    search_cfg = policy.get("search_control")
    if isinstance(search_cfg, dict):
        return dict(search_cfg)
    direct = cfg_value(cfg, "confidence_search_control", {}) or {}
    return dict(direct) if isinstance(direct, dict) else {}


def active_confidence_selection_enabled(cfg: Any) -> bool:
    policy = confidence_policy(cfg)
    if not bool(policy.get("enabled", False)):
        return False
    mode = str(policy.get("mode", "disabled") or "disabled").strip().lower()
    return mode in {"assist_selection", "assist_all"}


def population_audit_gate(
    *,
    generation: int,
    min_generation: int,
    viable_fraction: float,
    min_viable_fraction: float,
    severe_clash_fraction: float,
    max_severe_clash_fraction: float,
    zero_score_fraction: float,
    max_zero_score_fraction: float,
    elite_score_gap: float,
    min_elite_gap: float,
    no_improve_count: int,
    no_improve_patience: int,
) -> tuple[bool, list[str]]:
    """Cheap objective-signal gate before running full population curation.

    The search engine already evaluates the objective function for every pose.
    This gate consumes those existing signals and only allows the heavier
    confidence controller to inspect/reorder the population when there is a
    concrete warning sign.
    """

    if generation + 1 < max(1, int(min_generation)):
        return False, []

    reasons: list[str] = []
    if viable_fraction < min_viable_fraction:
        reasons.append("objective_low_viable_fraction")
    if severe_clash_fraction > max_severe_clash_fraction:
        reasons.append("objective_high_severe_clash_fraction")
    if zero_score_fraction > max_zero_score_fraction:
        reasons.append("objective_high_zero_score_fraction")

    stalled_half_patience = no_improve_count >= max(1, int(math.ceil(float(no_improve_patience) / 2.0)))
    if stalled_half_patience and elite_score_gap < min_elite_gap:
        reasons.append("objective_stalled_weak_elite_gap")

    return bool(reasons), reasons


def plateau_stop_decision(
    *,
    generation: int,
    min_generations: int,
    best_score_series: list[float],
    plateau_window: int,
    plateau_abs_improvement_max: float,
    elite_score_std: float,
    max_elite_score_std: float,
    elite_coords_dispersion_mean: float,
    max_elite_coords_dispersion: float,
    viable_fraction: float,
    min_viable_fraction: float,
    severe_clash_fraction: float,
    max_severe_clash_fraction: float,
    no_improve_count: int,
    no_improve_patience: int,
) -> tuple[bool, dict[str, Any]]:
    """Decide whether the population reached an auditable confidence plateau.

    This is the low-cost speed-up point of the controller: it does not create a
    new population and does not rescore poses. It only consumes signals already
    produced by the objective function and authorizes early termination when the
    best global score stopped improving and the population still looks reliable.
    """

    resolved_window = max(2, int(plateau_window or 2))
    series = [
        float(value)
        for value in best_score_series
        if value is not None and math.isfinite(float(value))
    ]
    recent_scores = series[-resolved_window:] if len(series) >= resolved_window else []
    recent_improvement = (
        float(recent_scores[-1] - recent_scores[0])
        if len(recent_scores) >= 2
        else float("inf")
    )
    min_reached = bool(generation + 1 >= max(1, int(min_generations or 1)))
    enough_window = bool(len(recent_scores) >= resolved_window)
    score_plateaued = bool(
        enough_window and recent_improvement <= float(plateau_abs_improvement_max)
    )
    patience_reached = bool(no_improve_count >= max(1, int(no_improve_patience or 1)))
    elite_score_stable = bool(float(elite_score_std) <= float(max_elite_score_std))
    elite_geometry_stable = bool(
        float(elite_coords_dispersion_mean) <= float(max_elite_coords_dispersion)
    )
    population_stability_ok = bool(elite_score_stable or elite_geometry_stable)
    viable_ok = bool(float(viable_fraction) >= float(min_viable_fraction))
    severe_clash_ok = bool(
        float(severe_clash_fraction) <= float(max_severe_clash_fraction)
    )
    conditions = {
        "min_generations_reached": min_reached,
        "enough_plateau_window": enough_window,
        "best_global_score_plateaued": score_plateaued,
        "no_global_improvement_patience_reached": patience_reached,
        "elite_score_std_low": elite_score_stable,
        "elite_coords_dispersion_low": elite_geometry_stable,
        "population_stability_ok": population_stability_ok,
        "viable_fraction_ok": viable_ok,
        "severe_clash_fraction_ok": severe_clash_ok,
    }
    required = {
        "min_generations_reached": min_reached,
        "enough_plateau_window": enough_window,
        "best_global_score_plateaued": score_plateaued,
        "no_global_improvement_patience_reached": patience_reached,
        "population_stability_ok": population_stability_ok,
        "viable_fraction_ok": viable_ok,
        "severe_clash_fraction_ok": severe_clash_ok,
    }
    should_stop = bool(all(required.values()))
    payload: dict[str, Any] = {
        "policy": "auditable_population_plateau_stop",
        "generation": int(generation),
        "min_generations": int(max(1, int(min_generations or 1))),
        "plateau_window": int(resolved_window),
        "recent_scores": [float(value) for value in recent_scores],
        "recent_improvement": float(recent_improvement),
        "plateau_abs_improvement_max": float(plateau_abs_improvement_max),
        "no_improve_count": int(no_improve_count),
        "no_improve_patience": int(max(1, int(no_improve_patience or 1))),
        "elite_score_std": float(elite_score_std),
        "max_elite_score_std": float(max_elite_score_std),
        "elite_coords_dispersion_mean": float(elite_coords_dispersion_mean),
        "max_elite_coords_dispersion": float(max_elite_coords_dispersion),
        "viable_fraction": float(viable_fraction),
        "min_viable_fraction": float(min_viable_fraction),
        "severe_clash_fraction": float(severe_clash_fraction),
        "max_severe_clash_fraction": float(max_severe_clash_fraction),
        "conditions": conditions,
        "required_conditions": required,
        "reasons": [key for key, value in conditions.items() if value],
        "blocking_reasons": [key for key, value in required.items() if not value],
    }
    return should_stop, payload


def pose_centroid(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=float)
    if arr.size == 0:
        return np.zeros(3, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        arr = np.reshape(arr, (-1, 3))
    return np.mean(arr, axis=0)


def elite_coords_dispersion_mean(poses: list[Pose], elite_indices: np.ndarray) -> float:
    if elite_indices.size == 0:
        return 0.0
    centroids = np.asarray(
        [pose_centroid(poses[int(idx)].coords) for idx in elite_indices],
        dtype=float,
    )
    if centroids.size == 0:
        return 0.0
    mean_centroid = np.mean(centroids, axis=0)
    distances = np.linalg.norm(centroids - mean_centroid, axis=1)
    return float(np.mean(distances)) if distances.size else 0.0


def _active_weights(cfg: Any) -> dict[str, float]:
    selection_cfg = confidence_search_cfg(cfg)
    raw_weights = selection_cfg.get("active_weights")
    if not isinstance(raw_weights, dict):
        raw_weights = {}
    defaults = {
        "contact_bonus": 0.010,
        "viability_bonus": 0.010,
        "severe_clash_penalty": -0.050,
        "zero_score_penalty": -0.020,
        "elite_consensus_penalty": -0.010,
    }
    resolved = dict(defaults)
    for key in defaults:
        if key in raw_weights:
            resolved[key] = safe_float(raw_weights[key], default=defaults[key])
    return resolved


def _active_gate_cfg(cfg: Any) -> dict[str, float]:
    selection_cfg = confidence_search_cfg(cfg)
    raw_gate = selection_cfg.get("active_gate")
    if not isinstance(raw_gate, dict):
        raw_gate = {}
    defaults = {
        "min_viable_fraction": 0.25,
        "max_severe_clash_fraction": 0.75,
        "min_score_dynamic_range": 1e-9,
    }
    resolved = dict(defaults)
    for key in defaults:
        if key in raw_gate:
            resolved[key] = safe_float(raw_gate[key], default=defaults[key])
    return resolved


def confidence_adjusted_scores(
    *,
    poses: list[Pose],
    raw_scores: np.ndarray,
    elite_indices: np.ndarray,
    cfg: Any,
    enabled: bool,
) -> np.ndarray:
    """Return auxiliary priority scores without changing the scientific score."""

    raw = np.asarray(raw_scores, dtype=float)
    if raw.size == 0:
        return raw
    if not enabled:
        for pose, score in zip(poses, raw):
            meta = pose.meta if isinstance(pose.meta, dict) else {}
            meta["confidence_active_selection_applied"] = False
            meta["confidence_active_priority_score"] = float(score)
            meta["confidence_active_priority_delta"] = 0.0
            pose.meta = meta
        return raw.copy()

    weights = _active_weights(cfg)
    gate_cfg = _active_gate_cfg(cfg)
    viable_flags = np.asarray(
        [is_viable_pose_proxy(pose, float(score)) for pose, score in zip(poses, raw)],
        dtype=bool,
    )
    severe_flags = np.asarray(
        [
            is_severe_clash(pose.meta if isinstance(pose.meta, dict) else {})
            for pose in poses
        ],
        dtype=bool,
    )
    finite_raw = raw[np.isfinite(raw)]
    score_dynamic_range = float(np.max(finite_raw) - np.min(finite_raw)) if finite_raw.size else 0.0
    viable_fraction = float(np.mean(viable_flags)) if viable_flags.size else 0.0
    severe_clash_fraction = float(np.mean(severe_flags)) if severe_flags.size else 0.0
    gate_reasons: list[str] = []
    if score_dynamic_range <= float(gate_cfg["min_score_dynamic_range"]):
        gate_reasons.append("score_dynamic_range_too_low")
    if viable_fraction < float(gate_cfg["min_viable_fraction"]):
        gate_reasons.append("viable_fraction_too_low")
    if severe_clash_fraction > float(gate_cfg["max_severe_clash_fraction"]):
        gate_reasons.append("severe_clash_fraction_too_high")
    if gate_reasons:
        for pose, score in zip(poses, raw):
            meta = pose.meta if isinstance(pose.meta, dict) else {}
            meta["confidence_active_selection_applied"] = False
            meta["confidence_active_selection_skipped"] = True
            meta["confidence_active_selection_skip_reasons"] = list(gate_reasons)
            meta["confidence_active_priority_score"] = float(score)
            meta["confidence_active_priority_delta"] = 0.0
            meta["confidence_active_gate"] = {
                "score_dynamic_range": float(score_dynamic_range),
                "viable_fraction": float(viable_fraction),
                "severe_clash_fraction": float(severe_clash_fraction),
                "thresholds": dict(gate_cfg),
            }
            pose.meta = meta
        return raw.copy()

    if finite_raw.size >= 2:
        p10, p90 = np.percentile(finite_raw, [10, 90])
        scale = max(float(p90 - p10), 1.0)
    else:
        scale = max(abs(float(finite_raw[0])) if finite_raw.size else 0.0, 1.0)
    raw_best = float(np.max(finite_raw)) if finite_raw.size else 0.0
    competitive_tolerance = max(0.10 * scale, 0.25)
    competitive_cutoff = raw_best - competitive_tolerance
    elite_centroids = np.asarray(
        [
            pose_centroid(poses[int(idx)].coords)
            for idx in elite_indices
            if int(idx) < len(poses)
        ],
        dtype=float,
    )
    if elite_centroids.size:
        consensus_center = np.mean(elite_centroids.reshape((-1, 3)), axis=0)
        distances = np.asarray(
            [float(np.linalg.norm(pose_centroid(pose.coords) - consensus_center)) for pose in poses],
            dtype=float,
        )
        distance_scale = max(float(np.percentile(distances, 75)), 1e-9)
    else:
        distances = np.zeros(len(poses), dtype=float)
        distance_scale = 1.0

    adjusted = np.empty_like(raw, dtype=float)
    for idx, (pose, score) in enumerate(zip(poses, raw)):
        meta = pose.meta if isinstance(pose.meta, dict) else {}
        contacts = extract_contacts(meta)
        contacts_norm = max(0.0, min(1.0, float(contacts or 0.0) / 5.0))
        severe_clash = 1.0 if bool(severe_flags[idx]) else 0.0
        zero_score = 1.0 if abs(float(score)) <= 1e-12 else 0.0
        viable = 1.0 if bool(viable_flags[idx]) else 0.0
        consensus_distance = max(0.0, min(2.0, float(distances[idx]) / distance_scale))
        delta = scale * (
            weights["contact_bonus"] * contacts_norm
            + weights["viability_bonus"] * viable
            + weights["severe_clash_penalty"] * severe_clash
            + weights["zero_score_penalty"] * zero_score
            + weights["elite_consensus_penalty"] * consensus_distance
        )
        max_delta = max(0.02 * max(abs(float(score)), 1.0), 0.10)
        delta = float(max(-max_delta, min(max_delta, delta)))
        priority_score = float(score + delta)
        if float(score) < competitive_cutoff:
            priority_score = float(score)
        adjusted[idx] = priority_score
        meta["confidence_active_selection_applied"] = True
        meta["confidence_active_priority_score"] = priority_score
        meta["confidence_active_priority_delta"] = float(delta)
        meta["confidence_active_components"] = {
            "raw_score": float(score),
            "scale": float(scale),
            "gate": {
                "score_dynamic_range": float(score_dynamic_range),
                "viable_fraction": float(viable_fraction),
                "severe_clash_fraction": float(severe_clash_fraction),
                "thresholds": dict(gate_cfg),
            },
            "contacts_norm": float(contacts_norm),
            "viable": float(viable),
            "severe_clash": float(severe_clash),
            "zero_score": float(zero_score),
            "elite_consensus_distance": float(consensus_distance),
            "competitive_cutoff": float(competitive_cutoff),
            "competitive_for_confidence_priority": bool(float(score) >= competitive_cutoff),
        }
        meta["confidence_active_weights"] = dict(weights)
        pose.meta = meta
    return adjusted


def guided_population_order(
    *,
    poses: list[Pose],
    raw_scores: np.ndarray,
    priority_scores: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    """Return a single mentor ranking for the whole population."""

    raw = np.asarray(raw_scores, dtype=float)
    priority = np.asarray(priority_scores, dtype=float)
    if raw.size == 0:
        return np.asarray([], dtype=int)
    ranking = np.asarray(topk_indices(priority, len(raw), largest=True), dtype=int)
    if enabled and raw.size:
        raw_best_idx = int(np.argmax(raw))
        ranking = np.asarray(
            [raw_best_idx] + [int(idx) for idx in ranking if int(idx) != raw_best_idx],
            dtype=int,
        )
    for rank, idx in enumerate(ranking):
        meta = poses[int(idx)].meta if isinstance(poses[int(idx)].meta, dict) else {}
        meta["population_guidance_rank"] = int(rank + 1)
        meta["population_guidance_role"] = "external_population_mentor"
        meta["population_guidance_raw_score"] = float(raw[int(idx)])
        meta["population_guidance_priority_score"] = float(priority[int(idx)])
        poses[int(idx)].meta = meta
    return ranking


def curated_population_order(
    *,
    poses: list[Pose],
    raw_scores: np.ndarray,
    priority_scores: np.ndarray,
    enabled: bool,
    elite_size: int = 1,
    elite_fraction: float = 0.10,
    min_elite_fragment: int = 3,
) -> np.ndarray:
    """Preserve raw elite and only mix confidence-guided candidates after it."""

    raw = np.asarray(raw_scores, dtype=float)
    priority = np.asarray(priority_scores, dtype=float)
    if raw.size == 0:
        return np.asarray([], dtype=int)
    raw_order = np.asarray(topk_indices(raw, len(raw), largest=True), dtype=int)
    if not enabled:
        return raw_order

    guided_order = guided_population_order(
        poses=poses,
        raw_scores=raw,
        priority_scores=priority,
        enabled=True,
    )
    fragment_count = max(
        int(elite_size or 1),
        int(min_elite_fragment or 1),
        int(np.ceil(float(max(0.0, elite_fraction)) * float(len(raw_order)))),
    )
    fragment_count = max(1, min(fragment_count, len(raw_order)))
    preserve_count = max(1, min(int(elite_size or 1), fragment_count))
    raw_fragment = raw_order[:fragment_count]
    guided_fragment = np.asarray(
        [int(idx) for idx in guided_order if int(idx) in set(int(item) for item in raw_fragment)],
        dtype=int,
    )
    curated: list[int] = []
    seen: set[int] = set()

    for idx in raw_fragment[:preserve_count]:
        item = int(idx)
        curated.append(item)
        seen.add(item)

    for guided_idx, raw_idx in zip(guided_fragment, raw_fragment):
        for idx in (int(guided_idx), int(raw_idx)):
            if idx in seen:
                continue
            curated.append(idx)
            seen.add(idx)
            if len(curated) >= fragment_count:
                break
        if len(curated) >= fragment_count:
            break

    for idx in raw_order:
        item = int(idx)
        if item not in seen:
            curated.append(item)
            seen.add(item)

    ranking = np.asarray(curated[: len(raw)], dtype=int)
    for rank, idx in enumerate(ranking):
        meta = poses[int(idx)].meta if isinstance(poses[int(idx)].meta, dict) else {}
        meta["population_curation_rank"] = int(rank + 1)
        meta["population_curation_role"] = "external_population_curator"
        meta["population_curation_raw_rank"] = int(np.where(raw_order == int(idx))[0][0] + 1)
        guided_positions = np.where(guided_order == int(idx))[0]
        meta["population_curation_guided_rank"] = (
            int(guided_positions[0] + 1) if guided_positions.size else None
        )
        poses[int(idx)].meta = meta
    return ranking


def population_orientation_diagnosis(
    *,
    generation: int,
    min_generation: int,
    viable_fraction: float,
    min_viable_fraction: float,
    severe_clash_fraction: float,
    max_severe_clash_fraction: float,
    zero_score_fraction: float,
    max_zero_score_fraction: float,
    elite_score_gap: float,
    min_elite_gap: float,
    elite_score_std: float,
    max_elite_score_std: float,
    no_improve_count: int,
    no_improve_patience: int,
) -> tuple[bool, list[str]]:
    """Diagnose whether population confidence is low enough to need guidance."""

    if generation + 1 < max(1, int(min_generation)):
        return False, []

    reasons: list[str] = []
    if viable_fraction < min_viable_fraction:
        reasons.append("low_viable_fraction")
    if severe_clash_fraction > max_severe_clash_fraction:
        reasons.append("high_severe_clash_fraction")
    if zero_score_fraction > max_zero_score_fraction:
        reasons.append("high_zero_score_fraction")

    stalled = no_improve_count >= max(1, int(math.ceil(float(no_improve_patience) / 2.0)))
    if stalled and elite_score_gap < min_elite_gap:
        reasons.append("weak_elite_gap_under_stall")
    if stalled and elite_score_std > max_elite_score_std:
        reasons.append("unstable_elite_under_stall")

    return bool(reasons), reasons


def extract_contacts(meta: dict[str, Any]) -> float | None:
    if "contatos" in meta:
        return safe_float(meta.get("contatos"), default=0.0)
    if "contacts" in meta:
        return safe_float(meta.get("contacts"), default=0.0)
    return None


def is_severe_clash(meta: dict[str, Any]) -> bool:
    if "colisao_severa" in meta:
        return bool(meta.get("colisao_severa"))
    if "severe_clash" in meta:
        return bool(meta.get("severe_clash"))
    return False


def is_viable_pose_proxy(pose: Pose, score: float) -> bool:
    if not math.isfinite(float(score)):
        return False
    meta = pose.meta if isinstance(pose.meta, dict) else {}
    if is_severe_clash(meta):
        return False
    contacts = extract_contacts(meta)
    if contacts is not None and contacts <= 0.0:
        return False
    return True
