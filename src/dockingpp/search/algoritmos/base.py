"""Ciclo compartilhado pelas buscas populacionais AG, PSO e híbrida.

Este módulo contém somente infraestrutura comum: gerações, avaliação,
controle populacional, confiança, telemetria e consolidação de resultados.
Os operadores numéricos concretos ficam nos módulos de cada algoritmo.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from dockingpp.domain.models import Population, Pose, RunResult
from dockingpp.search.controlador_populacional import (
    confidence_adjusted_scores as _confidence_adjusted_scores,
    curated_population_order as _curated_population_order,
    elite_coords_dispersion_mean as _elite_coords_dispersion_mean,
    is_severe_clash as _is_severe_clash,
    is_viable_pose_proxy as _is_viable_pose_proxy,
    plateau_stop_decision as _plateau_stop_decision,
    population_audit_gate as _population_audit_gate,
    population_orientation_diagnosis as _population_orientation_diagnosis,
)
from dockingpp.application.ports.search_engine import SearchEngine
from dockingpp.utils.geometry import apply_transform
from dockingpp.utils.topk import topk_indices

from ._configuracao_populacional import (
    ConfidenceSearchSettings,
    HybridStopSettings,
    PopulationSearchSettings,
    initialize_population,
    population_generations,
    require_fixed_evaluation_completion,
    uses_fixed_evaluations,
)
from ._estado_populacional import PocketSearchRuntime
from ._resumo_populacional import PocketSummaryContext, build_pocket_summary
from ._telemetria_populacional import (
    GenerationTelemetry,
    emit_confidence_early_stop,
    emit_generation_completed,
    emit_generation_metrics,
    emit_pocket_finished,
    emit_pocket_started,
)
from .comum import (
    _build_population_confidence_global,
    _collect_preexisting_metrics,
    _safe_float,
)
from .contexto import GenerationContext, SeedOrigin, mark_seed_origin
from .espaco_rigido import (
    ObjectiveEvaluationBudgetExhausted,
    RigidSearchSpace,
)


def _algorithm_family_label(strategy_name: str) -> str:
    strategy = str(strategy_name or "hybrid").strip().lower()
    if strategy == "hybrid":
        return "ABC+AG+VGOS"
    if strategy == "hybrid_memory":
        return "ABC+AG+VGOS-M"
    return strategy.upper()


class BasePopulationSearch(SearchEngine):
    """Orquestra o fluxo comum das buscas populacionais reproduzíveis.

    As subclasses implementam somente a construção da próxima geração. O
    restante do ciclo (avaliação, confiança, telemetria e agregação entre
    bolsões) permanece centralizado para que AG, PSO e híbrido tenham o mesmo
    contrato externo.
    """

    strategy_name = "base"
    algorithm_version = "base_population_search_v1"
    evaluates_candidates_internally = False

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def _next_generation(self, context: GenerationContext) -> list[Pose]:
        """Fallback legado para subclasses externas com estratégia própria.

        AG, PSO e o híbrido substituem este hook em seus módulos. A versão
        genérica é mantida por compatibilidade com eventuais subclasses de
        terceiros que antes alcançavam o ramo final do arquivo monolítico.
        """

        mutation_trans = context.max_trans * 0.2
        mutation_rot_deg = context.max_rot_deg * 0.2
        next_poses = list(context.selected)
        while len(next_poses) < context.pop_size:
            parent = context.selected[
                context.rng.integers(0, len(context.selected))
            ]
            rotation = self._axis_angle_rotation(context.rng, mutation_rot_deg)
            translation = context.rng.uniform(
                -mutation_trans,
                mutation_trans,
                size=3,
            )
            coords = apply_transform(parent.coords, rotation, translation)
            next_poses.append(Pose(coords=coords))
        return next_poses

    def _get_base_coords(self, peptide: Any, cfg: Any, rng: np.random.Generator) -> np.ndarray:
        if isinstance(peptide, np.ndarray) and peptide.size:
            return np.asarray(peptide, dtype=float)
        num_atoms = int(getattr(cfg, "num_atoms", 0) or 0)
        num_atoms = num_atoms if num_atoms > 0 else 10
        return rng.normal(size=(num_atoms, 3)).astype(float)

    def _axis_angle_rotation(self, rng: np.random.Generator, max_rot_deg: float) -> np.ndarray:
        if max_rot_deg <= 0:
            return np.eye(3, dtype=float)
        angle = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
        axis = rng.normal(size=3)
        norm = np.linalg.norm(axis)
        if norm == 0:
            return np.eye(3, dtype=float)
        axis = axis / norm
        x, y, z = axis
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        one_c = 1.0 - c
        return np.array(
            [
                [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
            ],
            dtype=float,
        )

    @staticmethod
    def _euler_rotation_matrix(rot_deg: np.ndarray) -> np.ndarray:
        """Mantém o helper privado histórico sobre a implementação compartilhada."""

        return RigidSearchSpace.euler_rotation_matrix(rot_deg)

    @staticmethod
    def _clone_candidate_pose(
        pose: Pose,
        *,
        pocket_id: str | None,
        generation: int,
        generation_rank: int,
        generation_confidence: dict[str, float] | None = None,
    ) -> Pose:
        cloned_meta = dict(getattr(pose, "meta", {}) or {})
        cloned_meta["pocket_id"] = pocket_id
        cloned_meta["generation"] = int(generation)
        cloned_meta["candidate_rank_generation"] = int(generation_rank)
        if generation_confidence:
            cloned_meta["population_confidence_generation"] = {
                str(key): float(value) for key, value in generation_confidence.items()
            }
            for key, value in generation_confidence.items():
                cloned_meta[f"generation_{key}"] = float(value)
        return Pose(
            coords=np.asarray(pose.coords, dtype=float).copy(),
            score_cheap=float(pose.score_cheap or 0.0),
            score_expensive=None,
            meta=cloned_meta,
        )

    def _search_single_pocket(
        self,
        pocket: Any,
        rng: np.random.Generator,
        base_coords: np.ndarray,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        score_cheap_fn: Callable[..., float],
        score_expensive_fn: Callable[..., float],
        logger: Any,
        pocket_index: int,
        step_offset: int,
    ) -> tuple[Pose, Population, list[Pose], dict[str, Any]]:
        """Execute search inside a single pocket using only cheap score."""

        _ = (receptor, peptide, score_expensive_fn)
        settings = PopulationSearchSettings.from_config(cfg)
        confidence = ConfidenceSearchSettings.from_config(
            cfg,
            generations=settings.generations,
        )
        hybrid_stop = HybridStopSettings.from_config(cfg)
        fixed_evaluations = uses_fixed_evaluations(cfg)

        generations = settings.generations
        pop_size = settings.pop_size
        topk = settings.topk
        candidate_pool_size = settings.candidate_pool_size
        max_trans = settings.max_trans
        max_rot_deg = settings.max_rot_deg
        debug_logger = settings.debug_logger
        tracer = settings.tracer

        confidence_active_selection_enabled = confidence.active_selection_enabled
        confidence_search_enabled = confidence.search_control_enabled
        confidence_early_stop_enabled = confidence.early_stop_enabled and not fixed_evaluations
        confidence_elite_fraction = confidence.elite_fraction
        confidence_min_elite_fragment = confidence.min_elite_fragment
        confidence_archive_inject_count = confidence.archive_inject_count
        confidence_archive_replace_only_if_better = (
            confidence.archive_replace_only_if_better
        )
        confidence_orientation_min_generation = (
            confidence.orientation_min_generation
        )
        confidence_signal_check_interval = confidence.signal_check_interval
        confidence_max_zero_score_fraction = confidence.max_zero_score_fraction
        confidence_min_elite_gap = confidence.min_elite_gap
        confidence_min_generations = confidence.min_generations
        confidence_plateau_window = confidence.plateau_window
        confidence_plateau_eps = confidence.plateau_eps
        confidence_elite_std_max = confidence.elite_std_max
        confidence_dispersion_max = confidence.dispersion_max
        confidence_min_viable_fraction = confidence.min_viable_fraction
        confidence_max_severe_clash_fraction = confidence.max_severe_clash_fraction
        confidence_no_improve_patience = confidence.no_improve_patience

        search_space = RigidSearchSpace(
            cfg=cfg,
            rng=rng,
            pocket=pocket,
            base_coords=base_coords,
            score_cheap_fn=score_cheap_fn,
            max_trans=max_trans,
            max_rot_deg=max_rot_deg,
        )
        dim = search_space.dim
        domain_low = search_space.domain_low
        domain_high = search_space.domain_high
        anchor_mode = search_space.anchor_mode
        anchor_center = search_space.anchor_center
        solution_to_pose = search_space.solution_to_pose
        random_solution = search_space.random_solution
        random_pose = search_space.random_pose
        evaluate_pose_score = search_space.evaluate_pose_score
        evaluate_solution = search_space.evaluate_solution
        existing_solution = search_space.existing_solution
        if (
            search_space.evaluation_budget is not None
            and search_space.evaluation_budget < pop_size
        ):
            raise ValueError(
                "objective_evaluation_budget_per_pocket deve ser >= pop_size"
            )

        # O contexto é uniforme para os três motores; somente o híbrido altera
        # e expõe este estado de operadores em seus resultados.
        def next_generation_from_previous(
            previous: list[Pose],
            raw_previous_scores: np.ndarray,
            generation_index: int,
        ) -> list[Pose]:
            # O ciclo comum cria apenas o contrato explícito. Cada engine
            # implementa o seu operador em um módulo independente.
            selected = list(previous[:topk])
            if not selected:
                return [random_pose() for _ in range(pop_size)]
            context = GenerationContext(
                cfg=cfg,
                rng=rng,
                previous=previous,
                raw_previous_scores=raw_previous_scores,
                generation_index=generation_index,
                selected=selected,
                pop_size=pop_size,
                topk=topk,
                dim=dim,
                max_trans=max_trans,
                max_rot_deg=max_rot_deg,
                domain_low=domain_low,
                domain_high=domain_high,
                random_solution=random_solution,
                random_pose=random_pose,
                existing_solution=existing_solution,
                evaluate_solution=evaluate_solution,
                solution_to_pose=solution_to_pose,
                hybrid_operator_state=runtime.hybrid_operator_state,
                hybrid_operator_history=runtime.hybrid_operator_history,
            )
            next_poses = self._next_generation(context)
            logger.log_metric(
                "search.operator_completed", 1, step=generation_index,
                extra={"engine_name": type(self).__name__,
                       "search_strategy": self.strategy_name,
                       "seed": cfg.seed, "pop_size": pop_size,
                       "generations": cfg.generations,
                       "pocket_id": str(pocket.id)},
            )
            return next_poses

        # Os três motores começam com a mesma amostragem uniforme no espaço
        # rígido 6D. Isso reproduz a inicialização do híbrido original e evita
        # favorecer artificialmente uma pose nula em qualquer baseline.
        poses, initial_audit = initialize_population(
            search_space, pop_size, cfg,
            seeded_by_default=(search_space.local_refinement
                               and getattr(cfg, "search_space_mode", "full") == "reduced"),
        )
        runtime = PocketSearchRuntime.initialize(poses, cfg)
        runtime.initial_population_audit = initial_audit
        track_seed = (
            initial_audit is not None and initial_audit["initialization"] == "seeded"
            and self.strategy_name in {"ag", "hybrid", "pso"}
        )
        seed_generations = []
        if track_seed:
            for index, pose in enumerate(poses):
                mark_seed_origin(pose, SeedOrigin.UNCHANGED if index == 0 else SeedOrigin.UNRELATED)
        if initial_audit is not None:
            logger.log_metric("search.initial_population", pop_size, step=step_offset,
                              extra={"pocket_id": str(pocket.id), "initial_population_audit": initial_audit})
        hybrid_plateau_patience = 0 if fixed_evaluations else hybrid_stop.patience
        hybrid_plateau_eps = hybrid_stop.eps

        emit_pocket_started(
            debug_logger=debug_logger,
            tracer=tracer,
            pocket=pocket,
            pocket_index=pocket_index,
            generations=generations,
            pop_size=pop_size,
            topk=topk,
            confidence_active_selection_enabled=confidence_active_selection_enabled,
        )

        # A população inicial (índice 0) é avaliada antes de ``generations``
        # ciclos evolutivos. Esta é a semântica de ``max_generations`` usada
        # pelo híbrido original e agora vale igualmente para AG e PSO.
        for generation in population_generations(search_space, generations):
            if generation > 0 and search_space.evaluation_budget_exhausted:
                runtime.objective_evaluation_budget_exhausted = True
                break
            generation_eval_start = int(search_space.actual_eval_count)
            fallback_poses_for_budget: list[Pose] | None = None
            if generation > 0:
                fallback_poses_for_budget = list(poses)
                poses = next_generation_from_previous(poses, guided_raw_scores, generation)

            raw_scores = np.zeros(len(poses), dtype=float)
            n_clashes = 0.0
            n_score_zero = 0
            n_severe_clash = 0
            n_viable_proxy = 0
            use_budget_fallback = False
            for idx, pose in enumerate(poses):
                if use_budget_fallback and fallback_poses_for_budget is not None:
                    pose = fallback_poses_for_budget[idx]
                    poses[idx] = pose
                # Soluções 6D passam pelo cache compartilhado. Assim, uma
                # solução já avaliada dentro do híbrido não chama novamente a
                # função objetivo; AG e PSO usam exatamente o mesmo caminho.
                pose_meta = pose.meta if isinstance(pose.meta, dict) else {}
                saved_solution = pose_meta.get("solution_vector")
                solution_array: np.ndarray | None = None
                if saved_solution is not None:
                    try:
                        candidate_array = np.asarray(saved_solution, dtype=float)
                    except (TypeError, ValueError):
                        candidate_array = np.asarray([], dtype=float)
                    if candidate_array.shape == (dim,) and np.all(
                        np.isfinite(candidate_array)
                    ):
                        solution_array = candidate_array
                if solution_array is not None:
                    try:
                        evaluated_pose, evaluated_score, _fitness = evaluate_solution(
                            solution_array
                        )
                    except ObjectiveEvaluationBudgetExhausted:
                        if fallback_poses_for_budget is None:
                            raise
                        runtime.objective_evaluation_budget_exhausted = True
                        use_budget_fallback = True
                        pose = fallback_poses_for_budget[idx]
                        poses[idx] = pose
                    else:
                        evaluated_pose.meta = {
                            **dict(pose_meta),
                            **dict(evaluated_pose.meta or {}),
                        }
                        poses[idx] = evaluated_pose
                        pose = evaluated_pose
                        pose.score_cheap = float(evaluated_score)
                else:
                    try:
                        pose.score_cheap = evaluate_pose_score(pose)
                    except ObjectiveEvaluationBudgetExhausted:
                        if fallback_poses_for_budget is None:
                            raise
                        runtime.objective_evaluation_budget_exhausted = True
                        use_budget_fallback = True
                        pose = fallback_poses_for_budget[idx]
                        poses[idx] = pose
                raw_scores[idx] = pose.score_cheap or 0.0
                if raw_scores[idx] == 0.0:
                    n_score_zero += 1
                meta = pose.meta if isinstance(pose.meta, dict) else {}
                n_clashes += _safe_float(meta.get("clashes"), default=0.0)
                severe_clash = _is_severe_clash(meta)
                if severe_clash:
                    n_severe_clash += 1
                if _is_viable_pose_proxy(pose, float(raw_scores[idx])):
                    n_viable_proxy += 1

            runtime.population = Population(poses=poses, generation=generation)
            if track_seed:
                seed_generations.append({
                    "generation": generation,
                    "n_eval_cumulative": int(search_space.actual_eval_count),
                    "population_size": len(poses),
                    "unchanged_seed_count": sum(p.meta["seed_unchanged"] for p in poses),
                    "seed_lineage_count": sum(p.meta["seed_lineage"] for p in poses),
                    "pso_global_best_from_seed_particle": runtime.hybrid_operator_state.get(
                        "pso_global_best_from_seed_particle"
                    ),
                })
            raw_order = np.asarray(topk_indices(raw_scores, len(poses), largest=True), dtype=int)
            raw_ranked = raw_order[:topk]
            elite_scores = np.asarray([raw_scores[int(idx)] for idx in raw_ranked], dtype=float)
            elite_score_mean = float(np.mean(elite_scores)) if elite_scores.size else 0.0
            elite_score_std = float(np.std(elite_scores, ddof=0)) if elite_scores.size else 0.0
            elite_score_gap_top1_topk = 0.0
            if elite_scores.size >= 2:
                elite_score_gap_top1_topk = float(elite_scores[0] - elite_scores[-1])
            elite_coords_dispersion_mean = _elite_coords_dispersion_mean(poses, raw_ranked)
            severe_clash_fraction = float(n_severe_clash) / float(len(poses)) if poses else 0.0
            zero_score_fraction = float(n_score_zero) / float(len(poses)) if poses else 0.0
            viable_fraction_proxy = float(n_viable_proxy) / float(len(poses)) if poses else 0.0
            orientation_gate_needed = False
            orientation_gate_reasons: list[str] = []
            severe_signal = bool(
                viable_fraction_proxy < confidence_min_viable_fraction
                or severe_clash_fraction > confidence_max_severe_clash_fraction
                or zero_score_fraction > confidence_max_zero_score_fraction
                or (
                    runtime.confidence_no_improve_count
                    >= confidence_no_improve_patience
                    and elite_score_gap_top1_topk < confidence_min_elite_gap
                )
            )
            scheduled_signal_check = bool(
                generation + 1 >= confidence_orientation_min_generation
                and (generation + 1) % confidence_signal_check_interval == 0
            )
            should_check_population_signal = bool(
                confidence_active_selection_enabled and (severe_signal or scheduled_signal_check)
            )
            if should_check_population_signal:
                runtime.confidence_signal_checks += 1
                orientation_gate_needed, orientation_gate_reasons = _population_audit_gate(
                    generation=generation,
                    min_generation=confidence_orientation_min_generation,
                    viable_fraction=viable_fraction_proxy,
                    min_viable_fraction=confidence_min_viable_fraction,
                    severe_clash_fraction=severe_clash_fraction,
                    max_severe_clash_fraction=confidence_max_severe_clash_fraction,
                    zero_score_fraction=zero_score_fraction,
                    max_zero_score_fraction=confidence_max_zero_score_fraction,
                    elite_score_gap=elite_score_gap_top1_topk,
                    min_elite_gap=confidence_min_elite_gap,
                    no_improve_count=runtime.confidence_no_improve_count,
                    no_improve_patience=confidence_no_improve_patience,
                )
            orientation_needed = False
            orientation_reasons: list[str] = []
            if confidence_active_selection_enabled and orientation_gate_needed:
                runtime.confidence_orientation_gate_hits += 1
                for reason in orientation_gate_reasons:
                    runtime.confidence_orientation_gate_reason_counts[reason] = (
                        runtime.confidence_orientation_gate_reason_counts.get(reason, 0)
                        + 1
                    )
                runtime.confidence_orientation_checks += 1
                orientation_needed, orientation_reasons = _population_orientation_diagnosis(
                    generation=generation,
                    min_generation=confidence_orientation_min_generation,
                    viable_fraction=viable_fraction_proxy,
                    min_viable_fraction=confidence_min_viable_fraction,
                    severe_clash_fraction=severe_clash_fraction,
                    max_severe_clash_fraction=confidence_max_severe_clash_fraction,
                    zero_score_fraction=zero_score_fraction,
                    max_zero_score_fraction=confidence_max_zero_score_fraction,
                    elite_score_gap=elite_score_gap_top1_topk,
                    min_elite_gap=confidence_min_elite_gap,
                    elite_score_std=elite_score_std,
                    max_elite_score_std=confidence_elite_std_max,
                    no_improve_count=runtime.confidence_no_improve_count,
                    no_improve_patience=confidence_no_improve_patience,
                )
            confidence_orientation_applied = bool(
                confidence_active_selection_enabled and orientation_needed
            )
            if confidence_orientation_applied:
                runtime.confidence_orientation_triggers += 1
                for reason in orientation_reasons:
                    runtime.confidence_orientation_reason_counts[reason] = (
                        runtime.confidence_orientation_reason_counts.get(reason, 0)
                        + 1
                    )

            if confidence_orientation_applied:
                selection_scores = _confidence_adjusted_scores(
                    poses=poses,
                    raw_scores=raw_scores,
                    elite_indices=raw_ranked,
                    cfg=cfg,
                    enabled=True,
                )
                guidance_order = _curated_population_order(
                    poses=poses,
                    raw_scores=raw_scores,
                    priority_scores=selection_scores,
                    enabled=True,
                    elite_size=max(1, int(getattr(cfg, "elite_size", 1) or 1)),
                    elite_fraction=confidence_elite_fraction,
                    min_elite_fragment=confidence_min_elite_fragment,
                )
            else:
                selection_scores = raw_scores.copy()
                guidance_order = raw_order
            ranked = guidance_order[:topk]
            guided_poses = [poses[int(idx)] for idx in guidance_order]
            guided_raw_scores = np.asarray([raw_scores[int(idx)] for idx in guidance_order], dtype=float)
            fragment_count = max(
                int(getattr(cfg, "elite_size", 1) or 1),
                int(confidence_min_elite_fragment),
                int(np.ceil(float(max(0.0, confidence_elite_fraction)) * float(len(guided_poses)))),
            )
            fragment_count = max(1, min(fragment_count, len(guided_poses)))
            current_elite_indices = guidance_order[:fragment_count]
            current_elite_scores = np.asarray([raw_scores[int(idx)] for idx in current_elite_indices], dtype=float)
            current_elite_mean_score = (
                float(np.mean(current_elite_scores)) if current_elite_scores.size else float("-inf")
            )
            archive_injected_this_generation = 0
            archive_mean_before = runtime.elite_archive_mean_score
            if (
                confidence_orientation_applied
                and runtime.elite_archive
                and confidence_archive_inject_count > 0
            ):
                runtime.curation_archive_comparisons += 1
                archive_better = (
                    archive_mean_before is not None
                    and archive_mean_before > current_elite_mean_score + confidence_plateau_eps
                )
                if archive_better or not confidence_archive_replace_only_if_better:
                    injection_limit = min(
                        confidence_archive_inject_count,
                        len(runtime.elite_archive),
                        len(guided_poses),
                    )
                    if len(guided_poses) <= 1:
                        start_pos = 0
                    else:
                        start_pos = max(1, min(fragment_count, len(guided_poses) - injection_limit))
                    for offset in range(injection_limit):
                        archived = runtime.elite_archive[offset]
                        target_pos = start_pos + offset
                        meta = dict(getattr(archived, "meta", {}) or {})
                        meta["population_curation_injected_from_archive"] = True
                        meta["population_curation_injection_generation"] = int(step_offset + generation)
                        meta["population_curation_archive_mean_score"] = (
                            float(archive_mean_before) if archive_mean_before is not None else None
                        )
                        guided_poses[target_pos] = Pose(
                            coords=np.asarray(archived.coords, dtype=float).copy(),
                            score_cheap=float(archived.score_cheap or 0.0),
                            score_expensive=None,
                            meta=meta,
                        )
                        guided_raw_scores[target_pos] = float(archived.score_cheap or 0.0)
                        archive_injected_this_generation += 1
                    runtime.curation_archive_injections += (
                        archive_injected_this_generation
                    )

            # Update archive after the optional comparison/injection. The archive
            # stores only a compact elite fragment, not the whole population.
            runtime.elite_archive = []
            for rank, idx in enumerate(current_elite_indices):
                pose = poses[int(idx)]
                meta = dict(getattr(pose, "meta", {}) or {})
                meta["population_curation_archive_rank"] = int(rank + 1)
                meta["population_curation_archive_generation"] = int(step_offset + generation)
                runtime.elite_archive.append(
                    Pose(
                        coords=np.asarray(pose.coords, dtype=float).copy(),
                        score_cheap=float(raw_scores[int(idx)]),
                        score_expensive=None,
                        meta=meta,
                    )
                )
            runtime.elite_archive_mean_score = current_elite_mean_score
            selection_elite_scores = np.asarray([selection_scores[int(idx)] for idx in ranked], dtype=float)
            selection_priority_mean = float(np.mean(selection_elite_scores)) if selection_elite_scores.size else 0.0
            selection_priority_std = float(np.std(selection_elite_scores, ddof=0)) if selection_elite_scores.size else 0.0
            current_generation_confidence = {
                "elite_score_mean": float(elite_score_mean),
                "elite_score_std": float(elite_score_std),
                "elite_score_gap_top1_topk": float(elite_score_gap_top1_topk),
                "elite_coords_dispersion_mean": float(elite_coords_dispersion_mean),
                "severe_clash_fraction": float(severe_clash_fraction),
                "zero_score_fraction": float(zero_score_fraction),
                "viable_fraction_proxy": float(viable_fraction_proxy),
                "selection_priority_mean": float(selection_priority_mean),
                "selection_priority_std": float(selection_priority_std),
            }

            for rank, idx in enumerate(ranked[:candidate_pool_size]):
                runtime.candidate_pool.append(
                    self._clone_candidate_pose(
                        poses[idx],
                        pocket_id=str(getattr(pocket, "id", "")),
                        generation=step_offset + generation,
                        generation_rank=rank,
                        generation_confidence=current_generation_confidence,
                    )
                )

            gen_best_idx = int(ranked[0]) if len(ranked) else int(np.argmax(raw_scores))
            gen_best = poses[gen_best_idx]
            gen_best_score = gen_best.score_cheap or 0.0
            gen_best_priority_score = float(selection_scores[gen_best_idx]) if selection_scores.size else float(gen_best_score)
            runtime.best_score_series.append(float(gen_best_score))
            runtime.best_priority_series.append(float(gen_best_priority_score))
            improved_global_raw = bool(gen_best_score > runtime.best_score)
            improved_global_priority = bool(
                gen_best_priority_score > runtime.best_priority_score
            )
            significant_raw_improvement = bool(
                gen_best_score > runtime.best_score + confidence_plateau_eps
            )
            strategy_key = str(getattr(self, "strategy_name", "") or "").lower()
            if strategy_key in {"hybrid", "hybrid_memory"}:
                if (
                    gen_best_score
                    > runtime.hybrid_plateau_best + hybrid_plateau_eps
                ):
                    runtime.hybrid_plateau_best = float(gen_best_score)
                    runtime.hybrid_plateau_no_improve = 0
                elif generation > 0:
                    runtime.hybrid_plateau_no_improve += 1
                if (
                    hybrid_plateau_patience > 0
                    and runtime.hybrid_plateau_no_improve
                    >= hybrid_plateau_patience
                ):
                    runtime.hybrid_plateau_stop_applied = True
                    runtime.hybrid_plateau_stop_generation = int(
                        step_offset + generation
                    )
            if improved_global_priority:
                runtime.best_priority_score = gen_best_priority_score
            if improved_global_raw:
                runtime.best_score = gen_best_score
                runtime.best_pose = gen_best
                runtime.best_generation = step_offset + generation
                runtime.best_pose.meta["pocket_id"] = getattr(pocket, "id", None)
            if significant_raw_improvement:
                runtime.confidence_no_improve_count = 0
            else:
                runtime.confidence_no_improve_count += 1
            runtime.best_global_score_series.append(float(runtime.best_score))

            step = step_offset + generation
            generation_eval_count = int(
                search_space.actual_eval_count - generation_eval_start
            )
            telemetry = GenerationTelemetry(
                generation=generation,
                total_generations=generations,
                pocket_index=pocket_index,
                step=step,
                best_score=float(gen_best_score),
                mean_score=float(np.mean(raw_scores)),
                best_priority_score=float(gen_best_priority_score),
                mean_priority_score=float(np.mean(selection_scores)),
                elite_score_mean=float(elite_score_mean),
                elite_score_std=float(elite_score_std),
                elite_score_gap=float(elite_score_gap_top1_topk),
                elite_coords_dispersion=float(elite_coords_dispersion_mean),
                severe_clash_fraction=float(severe_clash_fraction),
                zero_score_fraction=float(zero_score_fraction),
                viable_fraction=float(viable_fraction_proxy),
                generation_eval_count=generation_eval_count,
                n_clashes=float(n_clashes),
                n_score_zero=n_score_zero,
                min_score=float(np.min(raw_scores)) if raw_scores.size else 0.0,
                max_score=float(np.max(raw_scores)) if raw_scores.size else 0.0,
            )
            emit_generation_metrics(logger, telemetry)
            runtime.generation_signals.append(
                {
                    "best_score": float(gen_best_score),
                    "mean_score": float(np.mean(raw_scores)),
                    "elite_score_mean": float(elite_score_mean),
                    "elite_score_std": float(elite_score_std),
                    "elite_score_gap_top1_topk": float(elite_score_gap_top1_topk),
                    "elite_coords_dispersion_mean": float(elite_coords_dispersion_mean),
                    "severe_clash_fraction": float(severe_clash_fraction),
                    "zero_score_fraction": float(zero_score_fraction),
                    "viable_fraction_proxy": float(viable_fraction_proxy),
                    "selection_priority_mean": float(selection_priority_mean),
                    "selection_priority_std": float(selection_priority_std),
                    "confidence_signal_check": float(1.0 if confidence_active_selection_enabled else 0.0),
                    "confidence_orientation_gate_needed": float(1.0 if orientation_gate_needed else 0.0),
                    "confidence_orientation_gate_reason_count": float(len(orientation_gate_reasons)),
                    "confidence_orientation_applied": float(1.0 if confidence_orientation_applied else 0.0),
                    "confidence_orientation_reason_count": float(len(orientation_reasons)),
                    "elite_fragment_mean_score": float(current_elite_mean_score),
                    "archive_mean_score_before": (
                        float(archive_mean_before) if archive_mean_before is not None else 0.0
                    ),
                    "archive_injections": float(archive_injected_this_generation),
                    "n_eval_actual": float(generation_eval_count),
                }
            )
            if confidence_search_enabled and confidence_early_stop_enabled:
                stop_now, stop_payload = _plateau_stop_decision(
                    generation=generation,
                    min_generations=confidence_min_generations,
                    best_score_series=runtime.best_global_score_series,
                    plateau_window=confidence_plateau_window,
                    plateau_abs_improvement_max=confidence_plateau_eps,
                    elite_score_std=elite_score_std,
                    max_elite_score_std=confidence_elite_std_max,
                    elite_coords_dispersion_mean=elite_coords_dispersion_mean,
                    max_elite_coords_dispersion=confidence_dispersion_max,
                    viable_fraction=viable_fraction_proxy,
                    min_viable_fraction=confidence_min_viable_fraction,
                    severe_clash_fraction=severe_clash_fraction,
                    max_severe_clash_fraction=confidence_max_severe_clash_fraction,
                    no_improve_count=runtime.confidence_no_improve_count,
                    no_improve_patience=confidence_no_improve_patience,
                )
                runtime.confidence_stop_payload = stop_payload
                if stop_now:
                    runtime.confidence_stop_applied = True
                    runtime.confidence_stop_generation = int(step)
                    runtime.confidence_stop_reasons = list(
                        stop_payload.get("reasons", [])
                    )
                    emit_confidence_early_stop(
                        logger=logger,
                        debug_logger=debug_logger,
                        tracer=tracer,
                        pocket=pocket,
                        pocket_index=pocket_index,
                        telemetry=telemetry,
                        stop_payload=stop_payload,
                        plateau_eps=confidence_plateau_eps,
                    )
                    break
            emit_generation_completed(
                debug_logger=debug_logger,
                tracer=tracer,
                pocket=pocket,
                pocket_index=pocket_index,
                telemetry=telemetry,
            )
            poses = guided_poses
            raw_scores = guided_raw_scores
            if search_space.evaluation_budget_exhausted:
                runtime.objective_evaluation_budget_exhausted = True
                break
            if runtime.hybrid_plateau_stop_applied:
                break

        require_fixed_evaluation_completion(search_space)
        runtime.best_pose.meta["generation"] = runtime.best_generation
        emit_pocket_finished(
            debug_logger=debug_logger,
            tracer=tracer,
            pocket=pocket,
            best_score=runtime.best_score,
            best_pose=runtime.best_pose,
        )

        runtime.candidate_pool.sort(
            key=lambda pose: float(
                (pose.meta if isinstance(pose.meta, dict) else {}).get(
                    "confidence_active_priority_score",
                    pose.score_cheap
                    if pose.score_cheap is not None
                    else float("-inf"),
                )
            ),
            reverse=True,
        )
        if len(runtime.candidate_pool) > candidate_pool_size:
            runtime.candidate_pool = runtime.candidate_pool[:candidate_pool_size]

        pocket_summary = build_pocket_summary(
            PocketSummaryContext(
                pocket=pocket,
                cfg=cfg,
                strategy_name=str(
                    getattr(self, "strategy_name", "hybrid") or "hybrid"
                ),
                algorithm_version=str(
                    getattr(self, "algorithm_version", "unknown") or "unknown"
                ),
                settings=settings,
                confidence=confidence,
                runtime=runtime,
                search_space=search_space,
                anchor_mode=anchor_mode,
                anchor_center=anchor_center,
            )
        )
        if track_seed:
            lineage_audit = {
                "schema": "seed_lineage_v1",
                "algorithm": self.strategy_name,
                "lineage_semantics": ("same_particle_as_supplied_initial_pose"
                                      if self.strategy_name == "pso"
                                      else "operator_pedigree_from_supplied_initial_pose"),
                "generations": seed_generations,
                "search_best_seed_lineage": runtime.best_pose.meta["seed_lineage"],
                "search_best_unchanged_seed": runtime.best_pose.meta["seed_unchanged"],
            }
            pocket_summary["seed_lineage"] = lineage_audit
            logger.log_metric("search.seed_lineage", 1, step=step_offset + generation,
                              extra={"pocket_id": str(pocket.id), "seed_lineage": lineage_audit})
        if initial_audit is not None:
            logger.log_metric("search.execution_audit", search_space.actual_eval_count,
                              step=step_offset + runtime.best_generation,
                              extra={"pocket_id": str(pocket.id),
                                     "execution_audit": pocket_summary["execution_audit"]})
        return (
            runtime.best_pose,
            runtime.population,
            runtime.candidate_pool,
            pocket_summary,
        )
    def search(
        self,
        receptor: Any,
        peptide: Any,
        pockets: list[Any],
        cfg: Any,
        score_cheap_fn: Callable[..., float],
        score_expensive_fn: Callable[..., float],
        prior_pocket: Any,
        prior_pose: Any,
        logger: Any,
    ) -> RunResult:
        """Run cheap-score exploration and return rescoring candidates."""

        _ = (prior_pocket, prior_pose)
        if not pockets:
            raise ValueError("No pockets available for search.")
        seed_seq = np.random.SeedSequence(cfg.seed)
        child_seqs = seed_seq.spawn(len(pockets) + 1)
        base_rng = np.random.default_rng(child_seqs[0])
        base_coords = self._get_base_coords(peptide, cfg, base_rng)

        best_pose: Pose | None = None
        best_score = float("-inf")
        best_population: Population | None = None
        generations = int(getattr(cfg, "generations", 1) or 1)
        rescoring_candidates: list[Pose] = []
        population_confidence_by_pocket: list[dict[str, Any]] = []
        base_metrics = _collect_preexisting_metrics(cfg)
        candidate_pool_limit = max(
            1,
            int(
                getattr(
                    cfg,
                    "rescoring_candidate_pool",
                    max(int(getattr(cfg, "topk", 1) or 1), int(getattr(cfg, "rescoring_top_k", 1) or 1)),
                )
                or max(int(getattr(cfg, "topk", 1) or 1), int(getattr(cfg, "rescoring_top_k", 1) or 1))
            ),
        )

        for pocket_idx, pocket in enumerate(pockets):
            rng = np.random.default_rng(child_seqs[pocket_idx + 1])
            step_offset = pocket_idx * (generations + 1)
            pocket_best, population, pocket_candidates, pocket_summary = self._search_single_pocket(
                pocket=pocket,
                rng=rng,
                base_coords=base_coords,
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                score_cheap_fn=score_cheap_fn,
                score_expensive_fn=score_expensive_fn,
                logger=logger,
                pocket_index=pocket_idx,
                step_offset=step_offset,
            )
            rescoring_candidates.extend(pocket_candidates)
            population_confidence_by_pocket.append(pocket_summary)
            pocket_score = pocket_best.score_cheap or 0.0
            if pocket_score > best_score or best_pose is None:
                best_score = pocket_score
                best_pose = pocket_best
                best_population = population

        if best_pose is None or best_population is None:
            raise ValueError("No pockets available for search.")
        best_pose.meta.setdefault("generation", 0)
        rescoring_candidates.sort(
            key=lambda pose: float(pose.score_cheap if pose.score_cheap is not None else float("-inf")),
            reverse=True,
        )
        if len(rescoring_candidates) > candidate_pool_limit:
            rescoring_candidates = rescoring_candidates[:candidate_pool_limit]
        population_confidence_global = _build_population_confidence_global(
            pocket_summaries=population_confidence_by_pocket,
            rescoring_candidates=rescoring_candidates,
        )
        metrics = dict(base_metrics)
        metrics.update(
            {
                "search_strategy": str(getattr(self, "strategy_name", "hybrid") or "hybrid"),
                "search_algorithm_version": str(
                    getattr(self, "algorithm_version", "unknown") or "unknown"
                ),
                "objective_evaluation_budget_per_pocket": getattr(
                    cfg, "objective_evaluation_budget_per_pocket", None
                )
                if getattr(cfg, "objective_evaluation_budget_per_pocket", None)
                is not None
                else getattr(
                    cfg,
                    "max_objective_evaluations",
                    None,
                ),
                "objective_evaluations_actual_by_pocket": [
                    int(summary.get("n_eval_actual", 0))
                    for summary in population_confidence_by_pocket
                ],
                "objective_evaluation_budget_exhausted_any": any(
                    bool(summary.get("objective_evaluation_budget_exhausted", False))
                    for summary in population_confidence_by_pocket
                ),
                "search_algorithm_family": _algorithm_family_label(
                    str(getattr(self, "strategy_name", "hybrid") or "hybrid")
                ),
                "rescoring_candidates": rescoring_candidates,
                "population_confidence_by_pocket": population_confidence_by_pocket,
                "population_confidence_global": population_confidence_global,
            }
        )
        return RunResult(
            best_pose=best_pose,
            population=best_population,
            metrics=metrics,
        )
