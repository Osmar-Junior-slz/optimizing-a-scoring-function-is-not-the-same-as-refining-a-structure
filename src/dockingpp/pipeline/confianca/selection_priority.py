"""Prioridade modular para selecao de candidatos antes do rescoring final."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


DEFAULT_SELECTION_COMPONENTS = {
    "viability": True,
    "clash": True,
    "zero_score": True,
    "stability": True,
    "dispersion": True,
    "elite_gap": True,
}

DEFAULT_SELECTION_WEIGHTS = {
    "viability_bonus": 0.08,
    "elite_gap_bonus": 0.08,
    "severe_clash_penalty": -0.12,
    "zero_score_penalty": -0.05,
    "elite_std_penalty": -0.015,
    "dispersion_penalty": -0.015,
}


@dataclass(frozen=True)
class ConfidenceSelectionPriority:
    """Resultado auditavel da prioridade de selecao pre-Evina."""

    score: float
    delta: float
    components: dict[str, float]
    enabled_components: dict[str, bool]
    weights: dict[str, float]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bounded_unit(value: Any, *, default: float = 0.0) -> float:
    number = _safe_float(value, default=default)
    return float(max(0.0, min(1.0, number)))


def _bounded_ratio(value: Any, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    number = _safe_float(value, default=0.0)
    return float(max(0.0, min(2.0, number / denominator)))


def _cfg_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _selection_cfg(cfg: Any) -> Mapping[str, Any]:
    direct = _cfg_value(cfg, "confidence_selection", None)
    if isinstance(direct, Mapping):
        return direct
    policy = _cfg_value(cfg, "confidence_policy", None)
    if isinstance(policy, Mapping):
        nested = policy.get("selection")
        if isinstance(nested, Mapping):
            return nested
    return {}


def resolve_selection_components(cfg: Any) -> dict[str, bool]:
    """Resolve flags de ablação dos componentes da heuristica."""

    selection_cfg = _selection_cfg(cfg)
    raw_components = selection_cfg.get("components")
    if not isinstance(raw_components, Mapping):
        raw_components = selection_cfg
    components = dict(DEFAULT_SELECTION_COMPONENTS)
    for key in DEFAULT_SELECTION_COMPONENTS:
        if key in raw_components:
            components[key] = bool(raw_components[key])
        flag_key = f"use_{key}"
        if flag_key in raw_components:
            components[key] = bool(raw_components[flag_key])
    return components


def resolve_selection_weights(cfg: Any) -> dict[str, float]:
    """Resolve pesos configuraveis sem exigir que todos sejam informados."""

    selection_cfg = _selection_cfg(cfg)
    raw_weights = selection_cfg.get("weights")
    if not isinstance(raw_weights, Mapping):
        raw_weights = {}
    weights = dict(DEFAULT_SELECTION_WEIGHTS)
    for key in DEFAULT_SELECTION_WEIGHTS:
        if key in raw_weights:
            weights[key] = _safe_float(raw_weights[key], default=weights[key])
    return weights


def score_confidence_selection_priority(
    *,
    score_cheap: float,
    meta: Mapping[str, Any],
    cfg: Any,
) -> ConfidenceSelectionPriority:
    """Calcula a prioridade modular para selecionar candidatos ao Evina.

    O resultado nao substitui o Evina: ele decide apenas qual candidato barato
    merece passar pelo rescoring final seletivo.
    """

    base_score = float(score_cheap)
    scale = max(abs(base_score), 1.0)
    enabled = resolve_selection_components(cfg)
    weights = resolve_selection_weights(cfg)

    viable = _bounded_unit(meta.get("generation_viable_fraction_proxy"))
    severe_clash = _bounded_unit(meta.get("generation_severe_clash_fraction"))
    zero_score = _bounded_unit(meta.get("generation_zero_score_fraction"))
    elite_std = _bounded_ratio(meta.get("generation_elite_score_std"), 0.25)
    dispersion = _bounded_ratio(meta.get("generation_elite_coords_dispersion_mean"), 1.0)
    elite_gap = _bounded_ratio(meta.get("generation_elite_score_gap_top1_topk"), scale)

    components = {
        "base_score_cheap": base_score,
        "scale": float(scale),
        "viability_bonus": (
            weights["viability_bonus"] * scale * viable if enabled["viability"] else 0.0
        ),
        "elite_gap_bonus": (
            weights["elite_gap_bonus"] * scale * min(elite_gap, 1.0)
            if enabled["elite_gap"]
            else 0.0
        ),
        "severe_clash_penalty": (
            weights["severe_clash_penalty"] * scale * severe_clash
            if enabled["clash"]
            else 0.0
        ),
        "zero_score_penalty": (
            weights["zero_score_penalty"] * scale * zero_score
            if enabled["zero_score"]
            else 0.0
        ),
        "elite_std_penalty": (
            weights["elite_std_penalty"] * scale * elite_std
            if enabled["stability"]
            else 0.0
        ),
        "dispersion_penalty": (
            weights["dispersion_penalty"] * scale * dispersion
            if enabled["dispersion"]
            else 0.0
        ),
    }
    delta = float(
        components["viability_bonus"]
        + components["elite_gap_bonus"]
        + components["severe_clash_penalty"]
        + components["zero_score_penalty"]
        + components["elite_std_penalty"]
        + components["dispersion_penalty"]
    )
    return ConfidenceSelectionPriority(
        score=float(base_score + delta),
        delta=delta,
        components={key: float(value) for key, value in components.items()},
        enabled_components=enabled,
        weights=weights,
    )
