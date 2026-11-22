"""Emissão de telemetria do ciclo populacional.

As funções deste módulo mantêm nomes, payloads e ordem dos eventos legados.
Elas não alteram o estado numérico e não acessam o gerador aleatório.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def emit_pocket_started(
    *,
    debug_logger: Any,
    tracer: Any,
    pocket: Any,
    pocket_index: int,
    generations: int,
    pop_size: int,
    topk: int,
    confidence_active_selection_enabled: bool,
) -> None:
    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "search_pocket_start",
                "pocket_id": getattr(pocket, "id", None),
                "pocket_index": int(pocket_index),
                "generations": int(generations),
                "pop_size": int(pop_size),
                "topk": int(topk),
                "confidence_active_selection_enabled": bool(
                    confidence_active_selection_enabled
                ),
            }
        )
    if tracer is not None:
        tracer.event(
            stage="search",
            event_type="search_started",
            substage="pocket",
            pocket_id=str(getattr(pocket, "id", "")),
            payload={
                "pocket_index": int(pocket_index),
                "generations": int(generations),
                "pop_size": int(pop_size),
            },
            level="TRACE",
        )


@dataclass(frozen=True, slots=True)
class GenerationTelemetry:
    """Valores já calculados que serão publicados para uma geração."""

    generation: int
    total_generations: int
    pocket_index: int
    step: int
    best_score: float
    mean_score: float
    best_priority_score: float
    mean_priority_score: float
    elite_score_mean: float
    elite_score_std: float
    elite_score_gap: float
    elite_coords_dispersion: float
    severe_clash_fraction: float
    zero_score_fraction: float
    viable_fraction: float
    generation_eval_count: int
    n_clashes: float
    n_score_zero: int
    min_score: float
    max_score: float

    @property
    def extra(self) -> dict[str, int]:
        return {
            "generation": self.generation,
            "total_generations": self.total_generations,
            "pocket_index": self.pocket_index,
        }


def emit_generation_metrics(logger: Any, telemetry: GenerationTelemetry) -> None:
    """Publica métricas na mesma sequência usada antes da extração."""

    step = telemetry.step
    extra = telemetry.extra
    logger.log_metric("best_score", telemetry.best_score, step=step, extra=extra)
    logger.log_metric("mean_score", telemetry.mean_score, step=step, extra=extra)
    logger.log_metric(
        "best_confidence_priority_score",
        telemetry.best_priority_score,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "mean_confidence_priority_score",
        telemetry.mean_priority_score,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "elite_score_mean", telemetry.elite_score_mean, step=step, extra=extra
    )
    logger.log_metric(
        "elite_score_std", telemetry.elite_score_std, step=step, extra=extra
    )
    logger.log_metric(
        "elite_score_gap_top1_topk",
        telemetry.elite_score_gap,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "elite_coords_dispersion_mean",
        telemetry.elite_coords_dispersion,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "severe_clash_fraction",
        telemetry.severe_clash_fraction,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "zero_score_fraction",
        telemetry.zero_score_fraction,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "viable_fraction_proxy",
        telemetry.viable_fraction,
        step=step,
        extra=extra,
    )
    logger.log_metric(
        "n_eval", float(telemetry.generation_eval_count), step=step, extra=extra
    )
    logger.log_metric(
        "n_clashes", telemetry.n_clashes, step=step, extra=extra
    )


def emit_confidence_early_stop(
    *,
    logger: Any,
    debug_logger: Any,
    tracer: Any,
    pocket: Any,
    pocket_index: int,
    telemetry: GenerationTelemetry,
    stop_payload: dict[str, Any],
    plateau_eps: float,
) -> None:
    logger.log_metric(
        "confidence_search_early_stop",
        1.0,
        step=telemetry.step,
        extra={
            **telemetry.extra,
            "confidence_search_stop_policy": stop_payload.get("policy"),
            "recent_improvement": float(stop_payload.get("recent_improvement", 0.0)),
            "confidence_plateau_abs_improvement_max": float(plateau_eps),
            "conditions": dict(stop_payload.get("conditions", {})),
            "blocking_reasons": list(stop_payload.get("blocking_reasons", [])),
        },
    )
    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "confidence_search_early_stop",
                "pocket_id": getattr(pocket, "id", None),
                "pocket_index": int(pocket_index),
                "generation": int(telemetry.generation),
                "payload": stop_payload,
            }
        )
    if tracer is not None:
        tracer.event(
            stage="search",
            event_type="confidence_search_early_stop",
            substage="generation",
            pocket_id=str(getattr(pocket, "id", "")),
            payload=stop_payload,
            level="AUDIT",
            decision=True,
        )


def emit_generation_completed(
    *,
    debug_logger: Any,
    tracer: Any,
    pocket: Any,
    pocket_index: int,
    telemetry: GenerationTelemetry,
) -> None:
    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "search_generation",
                "pocket_id": getattr(pocket, "id", None),
                "pocket_index": int(pocket_index),
                "generation": int(telemetry.generation),
                "n_eval": int(telemetry.generation_eval_count),
                "best_score": float(telemetry.best_score),
                "mean_score": float(telemetry.mean_score),
                "min_score": float(telemetry.min_score),
                "max_score": float(telemetry.max_score),
                "best_confidence_priority_score": float(
                    telemetry.best_priority_score
                ),
                "n_score_zero": int(telemetry.n_score_zero),
            }
        )
    if tracer is not None:
        tracer.event(
            stage="search",
            event_type="generation_completed",
            substage="generation",
            pocket_id=str(getattr(pocket, "id", "")),
            payload={
                "generation": int(telemetry.generation),
                "n_eval_total": int(telemetry.generation_eval_count),
                "best_score": float(telemetry.best_score),
                "best_confidence_priority_score": float(
                    telemetry.best_priority_score
                ),
            },
            level="AUDIT",
        )


def emit_pocket_finished(
    *,
    debug_logger: Any,
    tracer: Any,
    pocket: Any,
    best_score: float,
    best_pose: Any,
) -> None:
    best_pose_id = best_pose.meta.get("pose_id") or best_pose.meta.get("id")
    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "search_pocket_end",
                "pocket_id": getattr(pocket, "id", None),
                "best_score": float(best_score),
                "best_pose_id": best_pose_id,
            }
        )
    if tracer is not None:
        tracer.event(
            stage="search",
            event_type="run_finished",
            substage="pocket",
            pocket_id=str(getattr(pocket, "id", "")),
            payload={
                "best_score": float(best_score),
                "best_pose_id": best_pose_id,
            },
            level="TRACE",
        )
