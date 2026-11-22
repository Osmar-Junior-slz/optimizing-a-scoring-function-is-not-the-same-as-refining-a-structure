"""Leitura tipada da configuração do ciclo populacional.

Este módulo não executa busca e não consome números aleatórios.  Ele apenas
preserva, em objetos imutáveis, as mesmas regras de defaults e compatibilidade
que historicamente ficavam no início de ``BasePopulationSearch``.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import count
import json
from typing import Any, Iterator

import numpy as np

from dockingpp.search.controlador_populacional import (
    active_confidence_selection_enabled,
    cfg_value,
    confidence_search_cfg,
    confidence_search_control_enabled,
)

from .comum import (
    ELITE_COORDS_DISPERSION_LOW_MAX,
    ELITE_SCORE_STD_LOW_MAX,
    PLATEAU_ABS_IMPROVEMENT_EPS,
)


def audit_sha256(payload: Any) -> str:
    """Hash deterministic scientific records without consuming the RNG."""

    data = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


MAX_GENERATIONS_WITHOUT_NEW_EVALUATIONS = 50


def uses_fixed_evaluations(cfg: Any) -> bool:
    return getattr(cfg, "experiment_budget_policy", None) == "fixed_evaluations"


def population_generations(search_space: Any, generations: int) -> Iterator[int]:
    """Keep native operators running until the real objective budget is consumed."""
    if not uses_fixed_evaluations(search_space.cfg):
        yield from range(generations + 1)
        return
    if search_space.evaluation_budget is None:
        raise ValueError("fixed_evaluations requires an objective evaluation budget")
    without_progress = 0
    for generation in count():
        if search_space.evaluation_budget_exhausted:
            return
        before = search_space.actual_eval_count
        yield generation
        without_progress = without_progress + 1 if before == search_space.actual_eval_count else 0
        if without_progress >= MAX_GENERATIONS_WITHOUT_NEW_EVALUATIONS:
            raise RuntimeError(
                f"Orçamento fixo não atingido: {search_space.actual_eval_count}/"
                f"{search_space.evaluation_budget} avaliações efetivas após "
                f"{without_progress} gerações sem novas avaliações."
            )


def require_fixed_evaluation_completion(search_space: Any) -> None:
    if uses_fixed_evaluations(search_space.cfg) and not search_space.evaluation_budget_exhausted:
        raise RuntimeError(
            f"Orçamento fixo não atingido: {search_space.actual_eval_count}/"
            f"{search_space.evaluation_budget} avaliações efetivas."
        )


def initialize_population(
    search_space: Any, pop_size: int, cfg: Any, *, seeded_by_default: bool = False,
) -> tuple[list[Any], dict[str, Any] | None]:
    """Sample identically, replacing only candidate zero for seeded runs.

    Sampling the same number of vectors in both arms preserves paired RNG
    streams. The supplied pose is an ordinary candidate, evaluated normally.
    """

    initialization = getattr(cfg, "initialization", None)
    if initialization is not None:
        if initialization not in {"uniform", "seeded"}:
            raise ValueError("initialization must be uniform or seeded")
        if not search_space.local_refinement:
            raise ValueError("independent initialization requires local_refinement")
    seeded = initialization == "seeded" or (
        initialization is None and seeded_by_default
    )
    poses = [search_space.random_pose() for _ in range(pop_size)]
    if seeded:
        poses[0] = search_space.solution_to_pose(np.zeros(search_space.dim, dtype=float))
        poses[0].meta["initialization_source"] = "supplied_pose_identity"
    if initialization is None:
        return poses, None
    record = {
        "schema": "factorial_initial_population_v1",
        "seed": int(cfg.seed),
        "initialization": initialization,
        "search_extent": getattr(cfg, "search_extent", None),
        "pose_source": getattr(cfg, "pose_source", None),
        "input_pose_coordinates_sha256": audit_sha256(search_space.base_coords.tolist()),
        "domain_low": search_space.domain_low.tolist(),
        "domain_high": search_space.domain_high.tolist(),
        "vector_order": ["tx", "ty", "tz", "rx_degrees", "ry_degrees", "rz_degrees"],
        "sampling": "independent_uniform_coordinates_within_rigid_domain",
        "population_size": pop_size,
        "seeded_candidate_count": 1 if seeded else 0,
        "solution_vectors": [list(pose.meta["solution_vector"]) for pose in poses],
        "recorded_before_objective_evaluations": int(search_space.actual_eval_count),
        "stop_rules": {
            "generations": int(cfg.generations),
            "objective_evaluation_ceiling": search_space.evaluation_budget,
            "budget_policy": getattr(cfg, "experiment_budget_policy", None),
            "hybrid_plateau_patience": int(getattr(cfg, "hybrid_plateau_patience", 80)),
            "hybrid_plateau_eps": float(getattr(cfg, "hybrid_plateau_eps", 1e-6)),
            "confidence_search_control_enabled": confidence_search_control_enabled(cfg),
        },
    }
    if uses_fixed_evaluations(cfg):
        record["stop_rules"].update(
            generation_limit_enabled=False,
            max_generations_without_new_evaluations=MAX_GENERATIONS_WITHOUT_NEW_EVALUATIONS,
        )
    return poses, {**record, "sha256": audit_sha256(record)}


@dataclass(frozen=True, slots=True)
class PopulationSearchSettings:
    """Parâmetros gerais usados pelo ciclo de um bolsão."""

    generations: int
    pop_size: int
    topk: int
    candidate_pool_size: int
    max_trans: float
    max_rot_deg: float
    debug_logger: Any
    tracer: Any

    @classmethod
    def from_config(cls, cfg: Any) -> "PopulationSearchSettings":
        generations = int(getattr(cfg, "generations", 1) or 1)
        pop_size = int(
            getattr(cfg, "pop_size", getattr(cfg, "population_size", 20)) or 20
        )
        topk = max(1, int(getattr(cfg, "topk", 1) or 1))
        rescoring_top_k = int(getattr(cfg, "rescoring_top_k", 1) or 1)
        default_candidate_pool = max(topk, rescoring_top_k)
        candidate_pool_size = max(
            1,
            int(
                getattr(
                    cfg,
                    "rescoring_candidate_pool",
                    default_candidate_pool,
                )
                or default_candidate_pool
            ),
        )
        return cls(
            generations=generations,
            pop_size=pop_size,
            topk=topk,
            candidate_pool_size=candidate_pool_size,
            max_trans=float(getattr(cfg, "max_trans", 0.0) or 0.0),
            max_rot_deg=float(getattr(cfg, "max_rot_deg", 0.0) or 0.0),
            debug_logger=getattr(cfg, "debug_logger", None),
            tracer=getattr(cfg, "audit_tracer", None),
        )


@dataclass(frozen=True, slots=True)
class ConfidenceSearchSettings:
    """Configuração das políticas auxiliares de confiança populacional."""

    active_selection_enabled: bool
    search_control_enabled: bool
    early_stop_enabled: bool
    elite_fraction: float
    min_elite_fragment: int
    archive_inject_count: int
    archive_replace_only_if_better: bool
    orientation_min_generation: int
    signal_check_interval: int
    max_zero_score_fraction: float
    min_elite_gap: float
    min_generations: int
    plateau_window: int
    plateau_eps: float
    elite_std_max: float
    dispersion_max: float
    min_viable_fraction: float
    max_severe_clash_fraction: float
    no_improve_patience: int

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        *,
        generations: int,
    ) -> "ConfidenceSearchSettings":
        search_cfg = confidence_search_cfg(cfg)
        default_min_generations = max(3, int(np.ceil(float(generations) * 0.75)))
        return cls(
            active_selection_enabled=active_confidence_selection_enabled(cfg),
            search_control_enabled=confidence_search_control_enabled(cfg),
            early_stop_enabled=bool(search_cfg.get("early_stop_enabled", False)),
            elite_fraction=float(search_cfg.get("elite_fraction", 0.10) or 0.10),
            min_elite_fragment=int(
                max(1, search_cfg.get("min_elite_fragment", 3) or 3)
            ),
            archive_inject_count=int(
                max(0, search_cfg.get("archive_inject_count", 2) or 2)
            ),
            archive_replace_only_if_better=bool(
                search_cfg.get("archive_replace_only_if_better", True)
            ),
            orientation_min_generation=int(
                max(1, search_cfg.get("orientation_min_generation", 5) or 5)
            ),
            signal_check_interval=int(
                max(1, search_cfg.get("signal_check_interval", 10) or 10)
            ),
            max_zero_score_fraction=float(
                search_cfg.get("max_zero_score_fraction", 0.25) or 0.25
            ),
            min_elite_gap=float(search_cfg.get("min_elite_gap", 0.01) or 0.01),
            min_generations=int(
                max(
                    1,
                    search_cfg.get(
                        "min_generations",
                        cfg_value(
                            cfg,
                            "confidence_min_generations",
                            default_min_generations,
                        ),
                    )
                    or default_min_generations,
                )
            ),
            plateau_window=int(
                max(
                    2,
                    search_cfg.get(
                        "plateau_window",
                        cfg_value(cfg, "confidence_plateau_window", 5),
                    )
                    or 5,
                )
            ),
            plateau_eps=float(
                search_cfg.get(
                    "plateau_abs_improvement_max",
                    cfg_value(
                        cfg,
                        "confidence_plateau_abs_improvement_max",
                        PLATEAU_ABS_IMPROVEMENT_EPS,
                    ),
                )
                or PLATEAU_ABS_IMPROVEMENT_EPS
            ),
            elite_std_max=float(
                search_cfg.get(
                    "elite_score_std_max",
                    cfg_value(
                        cfg,
                        "confidence_elite_score_std_max",
                        ELITE_SCORE_STD_LOW_MAX,
                    ),
                )
                or ELITE_SCORE_STD_LOW_MAX
            ),
            dispersion_max=float(
                search_cfg.get(
                    "elite_coords_dispersion_max",
                    cfg_value(
                        cfg,
                        "confidence_elite_coords_dispersion_max",
                        ELITE_COORDS_DISPERSION_LOW_MAX,
                    ),
                )
                or ELITE_COORDS_DISPERSION_LOW_MAX
            ),
            min_viable_fraction=float(
                search_cfg.get(
                    "min_viable_fraction",
                    cfg_value(cfg, "confidence_min_viable_fraction", 0.50),
                )
                or 0.50
            ),
            max_severe_clash_fraction=float(
                search_cfg.get(
                    "max_severe_clash_fraction",
                    cfg_value(
                        cfg,
                        "confidence_max_severe_clash_fraction",
                        0.35,
                    ),
                )
                or 0.35
            ),
            no_improve_patience=int(
                max(
                    1,
                    search_cfg.get(
                        "no_improve_patience",
                        cfg_value(cfg, "confidence_no_improve_patience", 5),
                    )
                    or 5,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class HybridStopSettings:
    """Política histórica de platô exclusiva do motor híbrido."""

    patience: int
    eps: float

    @classmethod
    def from_config(cls, cfg: Any) -> "HybridStopSettings":
        return cls(
            patience=max(0, int(getattr(cfg, "hybrid_plateau_patience", 80))),
            eps=max(0.0, float(getattr(cfg, "hybrid_plateau_eps", 1e-6))),
        )


def initial_hybrid_operator_state(cfg: Any) -> dict[str, Any]:
    """Cria o payload legado que acompanha as fases do híbrido."""

    return {
        "p_crossover_ari": float(
            0.5
            if getattr(cfg, "p_crossover_ari", 0.5) is None
            else getattr(cfg, "p_crossover_ari", 0.5)
        ),
        "p_mutation_gauss": float(
            0.7
            if getattr(cfg, "p_mutation_gauss", 0.7) is None
            else getattr(cfg, "p_mutation_gauss", 0.7)
        ),
        "ga_phases": 0,
        "adaptation_updates": 0,
        "attempts": {
            "arithmetic": 0,
            "binary": 0,
            "gaussian": 0,
            "random": 0,
        },
        "successes": {
            "arithmetic": 0,
            "binary": 0,
            "gaussian": 0,
            "random": 0,
        },
    }
