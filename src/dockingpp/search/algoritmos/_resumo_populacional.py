"""Montagem do resumo auditável de uma busca populacional por bolsão.

Separar a serialização evita que o grande schema de métricas esconda o ciclo
numérico. A função é determinística, não altera o estado e não consome RNG.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._configuracao_populacional import (
    ConfidenceSearchSettings,
    PopulationSearchSettings,
    audit_sha256,
)
from ._estado_populacional import PocketSearchRuntime
from .comum import (
    ELITE_COORDS_DISPERSION_LOW_MAX,
    ELITE_SCORE_STD_LOW_MAX,
    PLATEAU_ABS_IMPROVEMENT_EPS,
    PLATEAU_WINDOW_MAX_GENERATIONS,
    _mean,
)
from .espaco_rigido import RigidSearchSpace


@dataclass(frozen=True, slots=True)
class PocketSummaryContext:
    """Entradas necessárias para materializar o schema legado do bolsão."""

    pocket: Any
    cfg: Any
    strategy_name: str
    algorithm_version: str
    settings: PopulationSearchSettings
    confidence: ConfidenceSearchSettings
    runtime: PocketSearchRuntime
    search_space: RigidSearchSpace
    anchor_mode: str
    anchor_center: np.ndarray


def build_pocket_summary(context: PocketSummaryContext) -> dict[str, Any]:
    """Produz exatamente o payload histórico de confiança e convergência."""

    runtime = context.runtime
    settings = context.settings
    confidence = context.confidence
    generation_signals = runtime.generation_signals

    evaluated_populations = int(len(generation_signals))
    completed_generations = max(0, evaluated_populations - 1)

    convergence_series = runtime.best_global_score_series or runtime.best_score_series
    window_size = min(PLATEAU_WINDOW_MAX_GENERATIONS, len(convergence_series))
    final_window_scores = convergence_series[-window_size:] if window_size > 0 else []
    best_score_improvement_last_window = 0.0
    plateau_generations_last_window = 0
    if final_window_scores:
        best_score_improvement_last_window = float(
            final_window_scores[-1] - final_window_scores[0]
        )
        for index in range(1, len(final_window_scores)):
            delta = float(final_window_scores[index] - final_window_scores[index - 1])
            if abs(delta) < PLATEAU_ABS_IMPROVEMENT_EPS:
                plateau_generations_last_window += 1

    elite_score_mean_series = [
        float(item["elite_score_mean"]) for item in generation_signals
    ]
    elite_score_std_series = [
        float(item["elite_score_std"]) for item in generation_signals
    ]
    elite_gap_series = [
        float(item["elite_score_gap_top1_topk"]) for item in generation_signals
    ]
    elite_dispersion_series = [
        float(item["elite_coords_dispersion_mean"]) for item in generation_signals
    ]
    viable_fraction_series = [
        float(item["viable_fraction_proxy"]) for item in generation_signals
    ]
    severe_clash_fraction_series = [
        float(item["severe_clash_fraction"]) for item in generation_signals
    ]
    zero_score_fraction_series = [
        float(item["zero_score_fraction"]) for item in generation_signals
    ]
    selection_priority_mean_series = [
        float(item["selection_priority_mean"]) for item in generation_signals
    ]
    selection_priority_std_series = [
        float(item["selection_priority_std"]) for item in generation_signals
    ]
    elite_fragment_mean_series = [
        float(item.get("elite_fragment_mean_score", 0.0))
        for item in generation_signals
    ]
    archive_injection_series = [
        float(item.get("archive_injections", 0.0)) for item in generation_signals
    ]
    actual_eval_series = [
        int(item.get("n_eval_actual", 0.0)) for item in generation_signals
    ]

    elite_score_std_last = elite_score_std_series[-1] if elite_score_std_series else 0.0
    elite_coords_dispersion_last = (
        elite_dispersion_series[-1] if elite_dispersion_series else 0.0
    )
    viable_fraction_mean = _mean(viable_fraction_series)
    viable_fraction_last = viable_fraction_series[-1] if viable_fraction_series else 0.0
    plateau_min_required = max(2, window_size - 1)

    std_is_low = elite_score_std_last <= ELITE_SCORE_STD_LOW_MAX
    dispersion_is_low = (
        elite_coords_dispersion_last <= ELITE_COORDS_DISPERSION_LOW_MAX
    )
    viable_is_stable = viable_fraction_last >= viable_fraction_mean
    plateau_is_stable = plateau_generations_last_window >= plateau_min_required
    converged_stably = bool(
        std_is_low and dispersion_is_low and viable_is_stable and plateau_is_stable
    )

    uncertainty_flags: list[str] = []
    if not std_is_low:
        uncertainty_flags.append("elite_score_std_high")
    if not dispersion_is_low:
        uncertainty_flags.append("elite_coords_dispersion_high")
    if not viable_is_stable:
        uncertainty_flags.append("viable_fraction_last_below_mean")
    if not plateau_is_stable:
        uncertainty_flags.append("best_score_not_plateaued")

    pocket_summary: dict[str, Any] = {
        "pocket_id": str(getattr(context.pocket, "id", "")),
        "search_strategy": context.strategy_name,
        "search_algorithm_version": context.algorithm_version,
        "generations": int(settings.generations),
        "completed_generations": completed_generations,
        "evaluated_populations": evaluated_populations,
        "generations_saved_by_confidence_stop": int(
            max(0, int(settings.generations) - completed_generations)
        ),
        "evolution_cycles_not_run": int(
            max(0, int(settings.generations) - completed_generations)
        ),
        "pop_size": int(settings.pop_size),
        "search_anchor_mode": context.anchor_mode,
        "search_anchor_center": [float(value) for value in context.anchor_center],
        "n_eval_actual": int(context.search_space.actual_eval_count),
        "n_eval_actual_by_generation": actual_eval_series,
        "objective_evaluation_budget_per_pocket": (
            int(context.search_space.evaluation_budget)
            if context.search_space.evaluation_budget is not None
            else None
        ),
        "objective_evaluation_budget_exhausted": bool(
            runtime.objective_evaluation_budget_exhausted
        ),
        "best_score_final": (
            float(convergence_series[-1])
            if convergence_series
            else float(runtime.best_score)
        ),
        "best_score_global": float(runtime.best_score),
        "best_generation": int(runtime.best_generation),
        "elite_score_mean_last": (
            elite_score_mean_series[-1] if elite_score_mean_series else 0.0
        ),
        "elite_score_std_last": float(elite_score_std_last),
        "elite_score_gap_top1_topk_last": (
            elite_gap_series[-1] if elite_gap_series else 0.0
        ),
        "elite_coords_dispersion_last": float(elite_coords_dispersion_last),
        "viable_fraction_mean": float(viable_fraction_mean),
        "viable_fraction_last": float(viable_fraction_last),
        "severe_clash_fraction_mean": float(_mean(severe_clash_fraction_series)),
        "zero_score_fraction_mean": float(_mean(zero_score_fraction_series)),
        "best_score_improvement_last_window": float(
            best_score_improvement_last_window
        ),
        "plateau_generations_last_window": int(plateau_generations_last_window),
        "converged_stably": bool(converged_stably),
        "uncertainty_flags": uncertainty_flags,
        "confidence_active_selection_enabled": bool(
            confidence.active_selection_enabled
        ),
        "confidence_active_selection_role": "conditional_population_guidance",
        "confidence_population_curation_role": (
            "conditional_population_orientation"
            if confidence.active_selection_enabled
            else "disabled"
        ),
        "confidence_population_orientation_policy": (
            "objective_signals_first_curate_only_when_needed"
        ),
        "confidence_population_signal_checks": int(runtime.confidence_signal_checks),
        "confidence_population_orientation_gate_hits": int(
            runtime.confidence_orientation_gate_hits
        ),
        "confidence_population_orientation_gate_hit_fraction": (
            float(runtime.confidence_orientation_gate_hits)
            / float(runtime.confidence_signal_checks)
            if runtime.confidence_signal_checks
            else 0.0
        ),
        "confidence_population_orientation_gate_reason_counts": dict(
            runtime.confidence_orientation_gate_reason_counts
        ),
        "confidence_population_signal_check_interval": int(
            confidence.signal_check_interval
        ),
        "confidence_population_orientation_checks": int(
            runtime.confidence_orientation_checks
        ),
        "confidence_population_orientation_triggers": int(
            runtime.confidence_orientation_triggers
        ),
        "confidence_population_orientation_trigger_fraction": (
            float(runtime.confidence_orientation_triggers)
            / float(runtime.confidence_orientation_checks)
            if runtime.confidence_orientation_checks
            else 0.0
        ),
        "confidence_population_orientation_reason_counts": dict(
            runtime.confidence_orientation_reason_counts
        ),
        "confidence_population_orientation_min_generation": int(
            confidence.orientation_min_generation
        ),
        "confidence_population_orientation_max_zero_score_fraction": float(
            confidence.max_zero_score_fraction
        ),
        "confidence_population_orientation_min_elite_gap": float(
            confidence.min_elite_gap
        ),
        "confidence_population_curation_elite_fraction": float(
            confidence.elite_fraction
        ),
        "confidence_population_curation_min_elite_fragment": int(
            confidence.min_elite_fragment
        ),
        "confidence_population_curation_archive_inject_count": int(
            confidence.archive_inject_count
        ),
        "confidence_population_curation_archive_comparisons": int(
            runtime.curation_archive_comparisons
        ),
        "confidence_population_curation_archive_injections": int(
            runtime.curation_archive_injections
        ),
        "confidence_population_curation_elite_fragment_mean_last": (
            elite_fragment_mean_series[-1] if elite_fragment_mean_series else None
        ),
        "confidence_population_curation_archive_injections_total": float(
            np.sum(np.asarray(archive_injection_series, dtype=float))
        ),
        "confidence_active_best_priority_final": (
            float(runtime.best_priority_series[-1])
            if runtime.best_priority_series
            else None
        ),
        "confidence_active_best_priority_global": (
            float(runtime.best_priority_score)
            if np.isfinite(runtime.best_priority_score)
            else None
        ),
        "confidence_active_priority_mean_last": (
            selection_priority_mean_series[-1]
            if selection_priority_mean_series
            else None
        ),
        "confidence_active_priority_std_last": (
            selection_priority_std_series[-1]
            if selection_priority_std_series
            else None
        ),
        "confidence_search_control_enabled": bool(
            confidence.search_control_enabled
        ),
        "confidence_search_early_stop_enabled": bool(confidence.early_stop_enabled),
        "confidence_search_early_stop_applied": bool(
            runtime.confidence_stop_applied
        ),
        "confidence_search_stop_generation": runtime.confidence_stop_generation,
        "confidence_search_stop_reasons": runtime.confidence_stop_reasons,
        "confidence_search_stop_policy": runtime.confidence_stop_payload.get(
            "policy", "auditable_population_plateau_stop"
        ),
        "confidence_search_stop_conditions": dict(
            runtime.confidence_stop_payload.get("conditions", {})
        ),
        "confidence_search_stop_required_conditions": dict(
            runtime.confidence_stop_payload.get("required_conditions", {})
        ),
        "confidence_search_stop_blocking_reasons": list(
            runtime.confidence_stop_payload.get("blocking_reasons", [])
        ),
        "confidence_search_stop_recent_improvement": (
            runtime.confidence_stop_payload.get("recent_improvement")
        ),
        "confidence_search_stop_plateau_window": runtime.confidence_stop_payload.get(
            "plateau_window", int(confidence.plateau_window)
        ),
        "confidence_search_stop_no_improve_count": runtime.confidence_stop_payload.get(
            "no_improve_count", int(runtime.confidence_no_improve_count)
        ),
        "thresholds": {
            "plateau_window_max_generations": int(
                PLATEAU_WINDOW_MAX_GENERATIONS
            ),
            "plateau_window_size_used": int(window_size),
            "plateau_abs_improvement_max": float(PLATEAU_ABS_IMPROVEMENT_EPS),
            "plateau_min_generations_required": int(plateau_min_required),
            "elite_score_std_low_max": float(ELITE_SCORE_STD_LOW_MAX),
            "elite_coords_dispersion_low_max": float(
                ELITE_COORDS_DISPERSION_LOW_MAX
            ),
            "confidence_search_min_generations": int(confidence.min_generations),
            "confidence_search_plateau_window": int(confidence.plateau_window),
            "confidence_search_plateau_abs_improvement_max": float(
                confidence.plateau_eps
            ),
            "confidence_search_elite_score_std_max": float(
                confidence.elite_std_max
            ),
            "confidence_search_elite_coords_dispersion_max": float(
                confidence.dispersion_max
            ),
            "confidence_search_min_viable_fraction": float(
                confidence.min_viable_fraction
            ),
            "confidence_search_max_severe_clash_fraction": float(
                confidence.max_severe_clash_fraction
            ),
            "confidence_search_no_improve_patience": int(
                confidence.no_improve_patience
            ),
        },
    }
    if context.strategy_name.lower() in {"hybrid", "hybrid_memory"}:
        pocket_summary["hybrid_plateau_stop_applied"] = bool(
            runtime.hybrid_plateau_stop_applied
        )
        pocket_summary["hybrid_plateau_stop_generation"] = (
            runtime.hybrid_plateau_stop_generation
        )
        pocket_summary["hybrid_plateau_no_improve_count"] = int(
            runtime.hybrid_plateau_no_improve
        )
        pocket_summary["hybrid_operator_mode"] = str(
            getattr(context.cfg, "hybrid_operator_mode", "canonical") or "canonical"
        ).strip().lower()
        operator_state = runtime.hybrid_operator_state
        pocket_summary["hybrid_operator_state"] = {
            "p_crossover_ari": float(operator_state["p_crossover_ari"]),
            "p_mutation_gauss": float(operator_state["p_mutation_gauss"]),
            "ga_phases": int(operator_state["ga_phases"]),
            "adaptation_updates": int(operator_state["adaptation_updates"]),
            "attempts": {
                str(key): int(value)
                for key, value in operator_state["attempts"].items()
            },
            "successes": {
                str(key): int(value)
                for key, value in operator_state["successes"].items()
            },
            "history": list(runtime.hybrid_operator_history),
        }
        memory_archive = list(operator_state.get("memory_archive") or [])
        if context.strategy_name.lower() == "hybrid_memory":
            pocket_summary["hybrid_memory_state"] = {
                "archive_size": int(len(memory_archive)),
                "archive_capacity": int(
                    operator_state.get("memory_archive_capacity", len(memory_archive))
                ),
                "best_score": operator_state.get("memory_best_score"),
                "updates": int(operator_state.get("memory_updates", 0)),
                "injections": int(operator_state.get("memory_injections", 0)),
                "last_injected_generation": operator_state.get(
                    "memory_last_injected_generation"
                ),
                "history": list(operator_state.get("memory_history", [])),
            }
    if runtime.initial_population_audit is not None:
        pocket_summary["initial_population_audit"] = runtime.initial_population_audit
        if runtime.confidence_stop_applied:
            stop_reason = "confidence_early_stop"
        elif runtime.objective_evaluation_budget_exhausted:
            stop_reason = "objective_evaluation_ceiling"
        elif runtime.hybrid_plateau_stop_applied:
            stop_reason = "hybrid_plateau"
        else:
            stop_reason = "generation_limit"
        execution = {
            "schema": "factorial_execution_audit_v1",
            "initial_population_sha256": runtime.initial_population_audit["sha256"],
            "seed": int(context.cfg.seed),
            "algorithm": context.strategy_name,
            "budget_policy": getattr(context.cfg, "experiment_budget_policy", None),
            "objective_evaluation_ceiling": context.search_space.evaluation_budget,
            "objective_evaluations_actual": int(context.search_space.actual_eval_count),
            "objective_evaluations_by_generation": actual_eval_series,
            "evaluation_count_basis": "new_cheap_objective_calls_including_internal_operators_excluding_cache_hits_and_final_rescoring",
            "completed_generations": completed_generations,
            "stop_reason": stop_reason,
            "stop_rules": runtime.initial_population_audit["stop_rules"],
        }
        pocket_summary["execution_audit"] = {**execution, "sha256": audit_sha256(execution)}
    return pocket_summary
