"""Estado mutável do ciclo de busca dentro de um único bolsão.

O objeto apenas agrupa valores que antes eram dezenas de variáveis locais em
``BasePopulationSearch``. Não há cálculos numéricos nem consumo de RNG aqui.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from dockingpp.domain.models import Population, Pose

from ._configuracao_populacional import initial_hybrid_operator_state


@dataclass(slots=True)
class PocketSearchRuntime:
    """Estado observável acumulado durante a evolução de um bolsão."""

    best_pose: Pose
    population: Population
    best_score: float = float("-inf")
    best_generation: int = 0
    candidate_pool: list[Pose] = field(default_factory=list)
    generation_signals: list[dict[str, float]] = field(default_factory=list)
    best_score_series: list[float] = field(default_factory=list)
    best_global_score_series: list[float] = field(default_factory=list)
    best_priority_series: list[float] = field(default_factory=list)
    best_priority_score: float = float("-inf")

    confidence_stop_applied: bool = False
    confidence_stop_generation: int | None = None
    confidence_stop_reasons: list[str] = field(default_factory=list)
    confidence_stop_payload: dict[str, Any] = field(default_factory=dict)
    confidence_no_improve_count: int = 0

    elite_archive: list[Pose] = field(default_factory=list)
    elite_archive_mean_score: float | None = None
    curation_archive_injections: int = 0
    curation_archive_comparisons: int = 0
    confidence_signal_checks: int = 0
    confidence_orientation_gate_hits: int = 0
    confidence_orientation_gate_reason_counts: dict[str, int] = field(
        default_factory=dict
    )
    confidence_orientation_checks: int = 0
    confidence_orientation_triggers: int = 0
    confidence_orientation_reason_counts: dict[str, int] = field(
        default_factory=dict
    )

    objective_evaluation_budget_exhausted: bool = False
    hybrid_plateau_best: float = float("-inf")
    hybrid_plateau_no_improve: int = 0
    hybrid_plateau_stop_applied: bool = False
    hybrid_plateau_stop_generation: int | None = None
    hybrid_operator_state: dict[str, Any] = field(default_factory=dict)
    hybrid_operator_history: list[dict[str, Any]] = field(default_factory=list)
    initial_population_audit: dict[str, Any] | None = None

    @classmethod
    def initialize(cls, poses: list[Pose], cfg: Any) -> "PocketSearchRuntime":
        """Reproduz exatamente o estado inicial histórico do ciclo."""

        return cls(
            best_pose=poses[0],
            population=Population(poses=poses, generation=0),
            hybrid_operator_state=initial_hybrid_operator_state(cfg),
        )
