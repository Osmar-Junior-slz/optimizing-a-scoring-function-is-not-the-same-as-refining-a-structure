"""Funções e constantes compartilhadas pelos motores de busca.

O módulo reúne somente operações puras ou agregações sem estado da engine.
Separá-las evita que detalhes de telemetria e seleção fiquem misturados com as
classes dos três algoritmos.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dockingpp.domain.models import Pose


PLATEAU_WINDOW_MAX_GENERATIONS = 5
PLATEAU_ABS_IMPROVEMENT_EPS = 1e-6
ELITE_SCORE_STD_LOW_MAX = 0.25
ELITE_COORDS_DISPERSION_LOW_MAX = 1.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


def _tournament_select(
    rng: np.random.Generator,
    fitness: np.ndarray,
    tournament_k: int,
) -> int:
    """Select the lowest-fitness individual from a sampled tournament."""

    values = np.asarray(fitness, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("tournament selection requires a non-empty population")
    sample_size = max(1, min(int(tournament_k), values.size))
    candidates = np.asarray(
        rng.choice(values.size, size=sample_size, replace=False),
        dtype=int,
    )
    return int(candidates[int(np.argmin(values[candidates]))])


def _collect_preexisting_metrics(cfg: Any) -> dict[str, Any]:
    """Collect optional preexisting metrics payloads from cfg without failing hard."""

    merged: dict[str, Any] = {}
    if isinstance(cfg, dict):
        for key in ("metrics", "search_metrics"):
            payload = cfg.get(key)
            if isinstance(payload, dict):
                merged.update(payload)
        return merged
    for attr in ("metrics", "search_metrics"):
        payload = getattr(cfg, attr, None)
        if isinstance(payload, dict):
            merged.update(payload)
    return merged


def _best_vs_second_best_gap(best_scores: list[float]) -> float | None:
    ordered = sorted(best_scores, reverse=True)
    if len(ordered) < 2:
        return None
    return float(ordered[0] - ordered[1])


def _build_population_confidence_global(
    pocket_summaries: list[dict[str, Any]],
    rescoring_candidates: list[Pose],
) -> dict[str, Any]:
    ranked = sorted(
        pocket_summaries,
        key=lambda item: float(item.get("best_score_global", float("-inf"))),
        reverse=True,
    )
    ranked_rows = [
        {
            "pocket_id": item.get("pocket_id"),
            "best_score_global": float(item.get("best_score_global", 0.0)),
        }
        for item in ranked
    ]
    best_scores = [float(item.get("best_score_global", 0.0)) for item in ranked]
    candidate_counts: dict[str, int] = {}
    for pose in rescoring_candidates:
        meta = pose.meta if isinstance(pose.meta, dict) else {}
        pocket_id = str(meta.get("pocket_id", "unknown"))
        candidate_counts[pocket_id] = candidate_counts.get(pocket_id, 0) + 1
    dominant_pocket_id = None
    dominant_fraction = 0.0
    total_candidates = int(len(rescoring_candidates))
    if candidate_counts:
        dominant_pocket_id = sorted(
            candidate_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        if total_candidates > 0:
            dominant_fraction = float(candidate_counts[dominant_pocket_id]) / float(total_candidates)
    elif ranked_rows:
        dominant_pocket_id = ranked_rows[0].get("pocket_id")
    stable_count = sum(1 for item in pocket_summaries if bool(item.get("converged_stably", False)))
    uncertain_count = sum(
        1
        for item in pocket_summaries
        if isinstance(item.get("uncertainty_flags"), list) and bool(item["uncertainty_flags"])
    )
    return {
        "n_pockets": int(len(pocket_summaries)),
        "dominant_pocket_id": dominant_pocket_id,
        "dominant_pocket_candidate_fraction": float(dominant_fraction),
        "best_vs_second_best_gap": _best_vs_second_best_gap(best_scores),
        "stable_pockets_count": int(stable_count),
        "uncertain_pockets_count": int(uncertain_count),
        "pockets_ranked_by_best_score": ranked_rows,
    }
