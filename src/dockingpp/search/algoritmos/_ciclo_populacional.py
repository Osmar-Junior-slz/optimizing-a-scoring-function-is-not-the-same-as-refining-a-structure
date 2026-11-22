"""Ciclo interno compartilhado pelos motores populacionais.

O componente mantém o fluxo sequencial histórico. Os operadores concretos são
recebidos como callbacks, portanto ABC+GA+VGOS, AG e PSO continuam responsáveis
somente pela criação da próxima população. Nenhum helper deste módulo consome
RNG fora dos mesmos pontos em que o antigo método monolítico consumia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from dockingpp.domain.models import Population, Pose
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
from .comum import _safe_float
from .contexto import GenerationContext
from .espaco_rigido import (
    ObjectiveEvaluationBudgetExhausted,
    RigidSearchSpace,
)


NextGeneration = Callable[[GenerationContext], list[Pose]]
CloneCandidate = Callable[..., Pose]


@dataclass(slots=True)
class _CycleEnvironment:
    pocket: Any
    rng: np.random.Generator
    cfg: Any
    logger: Any
    pocket_index: int
    step_offset: int
    settings: PopulationSearchSettings
    confidence: ConfidenceSearchSettings
    hybrid_stop: HybridStopSettings
    search_space: RigidSearchSpace
    runtime: PocketSearchRuntime
    strategy_name: str
    algorithm_version: str
    next_generation: NextGeneration
    clone_candidate: CloneCandidate


@dataclass(slots=True)
class _GenerationEvaluation:
    poses: list[Pose]
    raw_scores: np.ndarray
    n_clashes: float
    n_score_zero: int
    n_severe_clash: int
    n_viable_proxy: int
    evaluation_count: int


@dataclass(slots=True)
class _GenerationGuidance:
    guided_poses: list[Pose]
    guided_raw_scores: np.ndarray
    selection_scores: np.ndarray
    best_pose: Pose
    best_score: float
    best_priority_score: float
    elite_score_mean: float
    elite_score_std: float
    elite_score_gap: float
    elite_coords_dispersion: float
    severe_clash_fraction: float
    zero_score_fraction: float
    viable_fraction: float
    selection_priority_mean: float
    selection_priority_std: float
    orientation_gate_needed: bool
    orientation_gate_reasons: list[str]
    orientation_applied: bool
    orientation_reasons: list[str]
    elite_fragment_mean_score: float
    archive_mean_before: float | None
    archive_injections: int


class PopulationPocketSearchCycle:
    """Executa o ciclo comum de um bolsão sem conhecer o motor concreto."""

    def __init__(
        self,
        *,
        strategy_name: str,
        algorithm_version: str,
        next_generation: NextGeneration,
        clone_candidate: CloneCandidate,
    ) -> None:
        self._strategy_name = strategy_name
        self._algorithm_version = algorithm_version
        self._next_generation_callback = next_generation
        self._clone_candidate = clone_candidate

    def run(
        self,
        *,
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
        """Executa as gerações e retorna o mesmo contrato histórico."""

        _ = (receptor, peptide, score_expensive_fn)
        settings = PopulationSearchSettings.from_config(cfg)
        confidence = ConfidenceSearchSettings.from_config(
            cfg,
            generations=settings.generations,
        )
        hybrid_stop = HybridStopSettings.from_config(cfg)
        search_space = RigidSearchSpace(
            cfg=cfg,
            rng=rng,
            pocket=pocket,
            base_coords=base_coords,
            score_cheap_fn=score_cheap_fn,
            max_trans=settings.max_trans,
            max_rot_deg=settings.max_rot_deg,
        )
        if (
            search_space.evaluation_budget is not None
            and search_space.evaluation_budget < settings.pop_size
        ):
            raise ValueError(
                "objective_evaluation_budget_per_pocket deve ser >= pop_size"
            )

        # Esta compreensão é deliberadamente o primeiro consumo de RNG do
        # bolsão, como no ciclo anterior à extração.
        poses, initial_audit = initialize_population(search_space, settings.pop_size, cfg)
        runtime = PocketSearchRuntime.initialize(poses, cfg)
        runtime.initial_population_audit = initial_audit
        if initial_audit is not None:
            logger.log_metric("search.initial_population", settings.pop_size, step=step_offset,
                              extra={"pocket_id": str(pocket.id), "initial_population_audit": initial_audit})
        environment = _CycleEnvironment(
            pocket=pocket,
            rng=rng,
            cfg=cfg,
            logger=logger,
            pocket_index=pocket_index,
            step_offset=step_offset,
            settings=settings,
            confidence=confidence,
            hybrid_stop=hybrid_stop,
            search_space=search_space,
            runtime=runtime,
            strategy_name=self._strategy_name,
            algorithm_version=self._algorithm_version,
            next_generation=self._next_generation_callback,
            clone_candidate=self._clone_candidate,
        )
        emit_pocket_started(
            debug_logger=settings.debug_logger,
            tracer=settings.tracer,
            pocket=pocket,
            pocket_index=pocket_index,
            generations=settings.generations,
            pop_size=settings.pop_size,
            topk=settings.topk,
            confidence_active_selection_enabled=(
                confidence.active_selection_enabled
            ),
        )

        guided_raw_scores: np.ndarray
        for generation in population_generations(search_space, settings.generations):
            if generation > 0 and search_space.evaluation_budget_exhausted:
                runtime.objective_evaluation_budget_exhausted = True
                break
            evaluation_start = int(search_space.actual_eval_count)
            fallback_poses: list[Pose] | None = None
            if generation > 0:
                fallback_poses = list(poses)
                poses = self._produce_next_generation(
                    environment,
                    previous=poses,
                    raw_previous_scores=guided_raw_scores,
                    generation=generation,
                )

            evaluation = self._evaluate_generation(
                environment,
                poses=poses,
                fallback_poses=fallback_poses,
                evaluation_start=evaluation_start,
            )
            poses = evaluation.poses
            runtime.population = Population(poses=poses, generation=generation)
            guidance = self._guide_generation(
                environment,
                evaluation=evaluation,
                generation=generation,
            )
            stop_before_completion = self._record_generation(
                environment,
                evaluation=evaluation,
                guidance=guidance,
                generation=generation,
            )
            if stop_before_completion:
                break

            telemetry = self._generation_telemetry(
                environment,
                evaluation=evaluation,
                guidance=guidance,
                generation=generation,
            )
            emit_generation_completed(
                debug_logger=settings.debug_logger,
                tracer=settings.tracer,
                pocket=pocket,
                pocket_index=pocket_index,
                telemetry=telemetry,
            )
            poses = guidance.guided_poses
            guided_raw_scores = guidance.guided_raw_scores
            if search_space.evaluation_budget_exhausted:
                runtime.objective_evaluation_budget_exhausted = True
                break
            if runtime.hybrid_plateau_stop_applied:
                break

        require_fixed_evaluation_completion(search_space)
        return self._finish(environment)

    @staticmethod
    def _produce_next_generation(
        environment: _CycleEnvironment,
        *,
        previous: list[Pose],
        raw_previous_scores: np.ndarray,
        generation: int,
    ) -> list[Pose]:
        settings = environment.settings
        search_space = environment.search_space
        selected = list(previous[: settings.topk])
        if not selected:
            return [
                search_space.random_pose() for _ in range(settings.pop_size)
            ]
        context = GenerationContext(
            cfg=environment.cfg,
            rng=environment.rng,
            previous=previous,
            raw_previous_scores=raw_previous_scores,
            generation_index=generation,
            selected=selected,
            pop_size=settings.pop_size,
            topk=settings.topk,
            dim=search_space.dim,
            max_trans=settings.max_trans,
            max_rot_deg=settings.max_rot_deg,
            domain_low=search_space.domain_low,
            domain_high=search_space.domain_high,
            random_solution=search_space.random_solution,
            random_pose=search_space.random_pose,
            existing_solution=search_space.existing_solution,
            evaluate_solution=search_space.evaluate_solution,
            solution_to_pose=search_space.solution_to_pose,
            hybrid_operator_state=environment.runtime.hybrid_operator_state,
            hybrid_operator_history=environment.runtime.hybrid_operator_history,
        )
        return environment.next_generation(context)

    @staticmethod
    def _evaluate_generation(
        environment: _CycleEnvironment,
        *,
        poses: list[Pose],
        fallback_poses: list[Pose] | None,
        evaluation_start: int,
    ) -> _GenerationEvaluation:
        search_space = environment.search_space
        runtime = environment.runtime
        raw_scores = np.zeros(len(poses), dtype=float)
        n_clashes = 0.0
        n_score_zero = 0
        n_severe_clash = 0
        n_viable_proxy = 0
        use_budget_fallback = False

        for index, pose in enumerate(poses):
            if use_budget_fallback and fallback_poses is not None:
                pose = fallback_poses[index]
                poses[index] = pose
            pose_meta = pose.meta if isinstance(pose.meta, dict) else {}
            saved_solution = pose_meta.get("solution_vector")
            solution_array: np.ndarray | None = None
            if saved_solution is not None:
                try:
                    candidate_array = np.asarray(saved_solution, dtype=float)
                except (TypeError, ValueError):
                    candidate_array = np.asarray([], dtype=float)
                if candidate_array.shape == (search_space.dim,) and np.all(
                    np.isfinite(candidate_array)
                ):
                    solution_array = candidate_array

            if solution_array is not None:
                try:
                    evaluated_pose, evaluated_score, _fitness = (
                        search_space.evaluate_solution(solution_array)
                    )
                except ObjectiveEvaluationBudgetExhausted:
                    if fallback_poses is None:
                        raise
                    runtime.objective_evaluation_budget_exhausted = True
                    use_budget_fallback = True
                    pose = fallback_poses[index]
                    poses[index] = pose
                else:
                    evaluated_pose.meta = {
                        **dict(pose_meta),
                        **dict(evaluated_pose.meta or {}),
                    }
                    poses[index] = evaluated_pose
                    pose = evaluated_pose
                    pose.score_cheap = float(evaluated_score)
            else:
                try:
                    pose.score_cheap = search_space.evaluate_pose_score(pose)
                except ObjectiveEvaluationBudgetExhausted:
                    if fallback_poses is None:
                        raise
                    runtime.objective_evaluation_budget_exhausted = True
                    use_budget_fallback = True
                    pose = fallback_poses[index]
                    poses[index] = pose

            raw_scores[index] = pose.score_cheap or 0.0
            if raw_scores[index] == 0.0:
                n_score_zero += 1
            meta = pose.meta if isinstance(pose.meta, dict) else {}
            n_clashes += _safe_float(meta.get("clashes"), default=0.0)
            if _is_severe_clash(meta):
                n_severe_clash += 1
            if _is_viable_pose_proxy(pose, float(raw_scores[index])):
                n_viable_proxy += 1

        return _GenerationEvaluation(
            poses=poses,
            raw_scores=raw_scores,
            n_clashes=n_clashes,
            n_score_zero=n_score_zero,
            n_severe_clash=n_severe_clash,
            n_viable_proxy=n_viable_proxy,
            evaluation_count=int(search_space.actual_eval_count - evaluation_start),
        )

    @staticmethod
    def _guide_generation(
        environment: _CycleEnvironment,
        *,
        evaluation: _GenerationEvaluation,
        generation: int,
    ) -> _GenerationGuidance:
        settings = environment.settings
        confidence = environment.confidence
        runtime = environment.runtime
        poses = evaluation.poses
        raw_scores = evaluation.raw_scores

        raw_order = np.asarray(
            topk_indices(raw_scores, len(poses), largest=True),
            dtype=int,
        )
        raw_ranked = raw_order[: settings.topk]
        elite_scores = np.asarray(
            [raw_scores[int(index)] for index in raw_ranked],
            dtype=float,
        )
        elite_score_mean = (
            float(np.mean(elite_scores)) if elite_scores.size else 0.0
        )
        elite_score_std = (
            float(np.std(elite_scores, ddof=0)) if elite_scores.size else 0.0
        )
        elite_score_gap = 0.0
        if elite_scores.size >= 2:
            elite_score_gap = float(elite_scores[0] - elite_scores[-1])
        elite_coords_dispersion = _elite_coords_dispersion_mean(poses, raw_ranked)
        severe_clash_fraction = (
            float(evaluation.n_severe_clash) / float(len(poses)) if poses else 0.0
        )
        zero_score_fraction = (
            float(evaluation.n_score_zero) / float(len(poses)) if poses else 0.0
        )
        viable_fraction = (
            float(evaluation.n_viable_proxy) / float(len(poses)) if poses else 0.0
        )

        severe_signal = bool(
            viable_fraction < confidence.min_viable_fraction
            or severe_clash_fraction > confidence.max_severe_clash_fraction
            or zero_score_fraction > confidence.max_zero_score_fraction
            or (
                runtime.confidence_no_improve_count
                >= confidence.no_improve_patience
                and elite_score_gap < confidence.min_elite_gap
            )
        )
        scheduled_signal_check = bool(
            generation + 1 >= confidence.orientation_min_generation
            and (generation + 1) % confidence.signal_check_interval == 0
        )
        should_check_signal = bool(
            confidence.active_selection_enabled
            and (severe_signal or scheduled_signal_check)
        )
        orientation_gate_needed = False
        orientation_gate_reasons: list[str] = []
        if should_check_signal:
            runtime.confidence_signal_checks += 1
            orientation_gate_needed, orientation_gate_reasons = (
                _population_audit_gate(
                    generation=generation,
                    min_generation=confidence.orientation_min_generation,
                    viable_fraction=viable_fraction,
                    min_viable_fraction=confidence.min_viable_fraction,
                    severe_clash_fraction=severe_clash_fraction,
                    max_severe_clash_fraction=(
                        confidence.max_severe_clash_fraction
                    ),
                    zero_score_fraction=zero_score_fraction,
                    max_zero_score_fraction=confidence.max_zero_score_fraction,
                    elite_score_gap=elite_score_gap,
                    min_elite_gap=confidence.min_elite_gap,
                    no_improve_count=runtime.confidence_no_improve_count,
                    no_improve_patience=confidence.no_improve_patience,
                )
            )

        orientation_needed = False
        orientation_reasons: list[str] = []
        if confidence.active_selection_enabled and orientation_gate_needed:
            runtime.confidence_orientation_gate_hits += 1
            for reason in orientation_gate_reasons:
                runtime.confidence_orientation_gate_reason_counts[reason] = (
                    runtime.confidence_orientation_gate_reason_counts.get(reason, 0)
                    + 1
                )
            runtime.confidence_orientation_checks += 1
            orientation_needed, orientation_reasons = (
                _population_orientation_diagnosis(
                    generation=generation,
                    min_generation=confidence.orientation_min_generation,
                    viable_fraction=viable_fraction,
                    min_viable_fraction=confidence.min_viable_fraction,
                    severe_clash_fraction=severe_clash_fraction,
                    max_severe_clash_fraction=(
                        confidence.max_severe_clash_fraction
                    ),
                    zero_score_fraction=zero_score_fraction,
                    max_zero_score_fraction=confidence.max_zero_score_fraction,
                    elite_score_gap=elite_score_gap,
                    min_elite_gap=confidence.min_elite_gap,
                    elite_score_std=elite_score_std,
                    max_elite_score_std=confidence.elite_std_max,
                    no_improve_count=runtime.confidence_no_improve_count,
                    no_improve_patience=confidence.no_improve_patience,
                )
            )
        orientation_applied = bool(
            confidence.active_selection_enabled and orientation_needed
        )
        if orientation_applied:
            runtime.confidence_orientation_triggers += 1
            for reason in orientation_reasons:
                runtime.confidence_orientation_reason_counts[reason] = (
                    runtime.confidence_orientation_reason_counts.get(reason, 0)
                    + 1
                )

        if orientation_applied:
            selection_scores = _confidence_adjusted_scores(
                poses=poses,
                raw_scores=raw_scores,
                elite_indices=raw_ranked,
                cfg=environment.cfg,
                enabled=True,
            )
            guidance_order = _curated_population_order(
                poses=poses,
                raw_scores=raw_scores,
                priority_scores=selection_scores,
                enabled=True,
                elite_size=max(
                    1,
                    int(getattr(environment.cfg, "elite_size", 1) or 1),
                ),
                elite_fraction=confidence.elite_fraction,
                min_elite_fragment=confidence.min_elite_fragment,
            )
        else:
            selection_scores = raw_scores.copy()
            guidance_order = raw_order

        ranked = guidance_order[: settings.topk]
        guided_poses = [poses[int(index)] for index in guidance_order]
        guided_raw_scores = np.asarray(
            [raw_scores[int(index)] for index in guidance_order],
            dtype=float,
        )
        fragment_count = max(
            int(getattr(environment.cfg, "elite_size", 1) or 1),
            int(confidence.min_elite_fragment),
            int(
                np.ceil(
                    float(max(0.0, confidence.elite_fraction))
                    * float(len(guided_poses))
                )
            ),
        )
        fragment_count = max(1, min(fragment_count, len(guided_poses)))
        current_elite_indices = guidance_order[:fragment_count]
        current_elite_scores = np.asarray(
            [raw_scores[int(index)] for index in current_elite_indices],
            dtype=float,
        )
        current_elite_mean_score = (
            float(np.mean(current_elite_scores))
            if current_elite_scores.size
            else float("-inf")
        )
        archive_injections = PopulationPocketSearchCycle._inject_archive(
            environment,
            guided_poses=guided_poses,
            guided_raw_scores=guided_raw_scores,
            fragment_count=fragment_count,
            current_elite_mean_score=current_elite_mean_score,
            generation=generation,
            orientation_applied=orientation_applied,
        )
        archive_mean_before = runtime.elite_archive_mean_score
        PopulationPocketSearchCycle._replace_archive(
            environment,
            poses=poses,
            raw_scores=raw_scores,
            current_elite_indices=current_elite_indices,
            current_elite_mean_score=current_elite_mean_score,
            generation=generation,
        )

        selection_elite_scores = np.asarray(
            [selection_scores[int(index)] for index in ranked],
            dtype=float,
        )
        selection_priority_mean = (
            float(np.mean(selection_elite_scores))
            if selection_elite_scores.size
            else 0.0
        )
        selection_priority_std = (
            float(np.std(selection_elite_scores, ddof=0))
            if selection_elite_scores.size
            else 0.0
        )
        generation_confidence = {
            "elite_score_mean": float(elite_score_mean),
            "elite_score_std": float(elite_score_std),
            "elite_score_gap_top1_topk": float(elite_score_gap),
            "elite_coords_dispersion_mean": float(elite_coords_dispersion),
            "severe_clash_fraction": float(severe_clash_fraction),
            "zero_score_fraction": float(zero_score_fraction),
            "viable_fraction_proxy": float(viable_fraction),
            "selection_priority_mean": float(selection_priority_mean),
            "selection_priority_std": float(selection_priority_std),
        }
        for rank, index in enumerate(ranked[: settings.candidate_pool_size]):
            runtime.candidate_pool.append(
                environment.clone_candidate(
                    poses[index],
                    pocket_id=str(getattr(environment.pocket, "id", "")),
                    generation=environment.step_offset + generation,
                    generation_rank=rank,
                    generation_confidence=generation_confidence,
                )
            )

        best_index = int(ranked[0]) if len(ranked) else int(np.argmax(raw_scores))
        best_pose = poses[best_index]
        best_score = best_pose.score_cheap or 0.0
        best_priority_score = (
            float(selection_scores[best_index])
            if selection_scores.size
            else float(best_score)
        )
        return _GenerationGuidance(
            guided_poses=guided_poses,
            guided_raw_scores=guided_raw_scores,
            selection_scores=selection_scores,
            best_pose=best_pose,
            best_score=float(best_score),
            best_priority_score=best_priority_score,
            elite_score_mean=elite_score_mean,
            elite_score_std=elite_score_std,
            elite_score_gap=elite_score_gap,
            elite_coords_dispersion=elite_coords_dispersion,
            severe_clash_fraction=severe_clash_fraction,
            zero_score_fraction=zero_score_fraction,
            viable_fraction=viable_fraction,
            selection_priority_mean=selection_priority_mean,
            selection_priority_std=selection_priority_std,
            orientation_gate_needed=orientation_gate_needed,
            orientation_gate_reasons=orientation_gate_reasons,
            orientation_applied=orientation_applied,
            orientation_reasons=orientation_reasons,
            elite_fragment_mean_score=current_elite_mean_score,
            archive_mean_before=archive_mean_before,
            archive_injections=archive_injections,
        )

    @staticmethod
    def _inject_archive(
        environment: _CycleEnvironment,
        *,
        guided_poses: list[Pose],
        guided_raw_scores: np.ndarray,
        fragment_count: int,
        current_elite_mean_score: float,
        generation: int,
        orientation_applied: bool,
    ) -> int:
        runtime = environment.runtime
        confidence = environment.confidence
        archive_mean_before = runtime.elite_archive_mean_score
        injected = 0
        if (
            orientation_applied
            and runtime.elite_archive
            and confidence.archive_inject_count > 0
        ):
            runtime.curation_archive_comparisons += 1
            archive_better = bool(
                archive_mean_before is not None
                and archive_mean_before
                > current_elite_mean_score + confidence.plateau_eps
            )
            if archive_better or not confidence.archive_replace_only_if_better:
                injection_limit = min(
                    confidence.archive_inject_count,
                    len(runtime.elite_archive),
                    len(guided_poses),
                )
                if len(guided_poses) <= 1:
                    start_position = 0
                else:
                    start_position = max(
                        1,
                        min(fragment_count, len(guided_poses) - injection_limit),
                    )
                for offset in range(injection_limit):
                    archived = runtime.elite_archive[offset]
                    target_position = start_position + offset
                    meta = dict(getattr(archived, "meta", {}) or {})
                    meta["population_curation_injected_from_archive"] = True
                    meta["population_curation_injection_generation"] = int(
                        environment.step_offset + generation
                    )
                    meta["population_curation_archive_mean_score"] = (
                        float(archive_mean_before)
                        if archive_mean_before is not None
                        else None
                    )
                    guided_poses[target_position] = Pose(
                        coords=np.asarray(archived.coords, dtype=float).copy(),
                        score_cheap=float(archived.score_cheap or 0.0),
                        score_expensive=None,
                        meta=meta,
                    )
                    guided_raw_scores[target_position] = float(
                        archived.score_cheap or 0.0
                    )
                    injected += 1
                runtime.curation_archive_injections += injected
        return injected

    @staticmethod
    def _replace_archive(
        environment: _CycleEnvironment,
        *,
        poses: list[Pose],
        raw_scores: np.ndarray,
        current_elite_indices: np.ndarray,
        current_elite_mean_score: float,
        generation: int,
    ) -> None:
        runtime = environment.runtime
        runtime.elite_archive = []
        for rank, index in enumerate(current_elite_indices):
            pose = poses[int(index)]
            meta = dict(getattr(pose, "meta", {}) or {})
            meta["population_curation_archive_rank"] = int(rank + 1)
            meta["population_curation_archive_generation"] = int(
                environment.step_offset + generation
            )
            runtime.elite_archive.append(
                Pose(
                    coords=np.asarray(pose.coords, dtype=float).copy(),
                    score_cheap=float(raw_scores[int(index)]),
                    score_expensive=None,
                    meta=meta,
                )
            )
        runtime.elite_archive_mean_score = current_elite_mean_score

    @staticmethod
    def _record_generation(
        environment: _CycleEnvironment,
        *,
        evaluation: _GenerationEvaluation,
        guidance: _GenerationGuidance,
        generation: int,
    ) -> tuple[bool, GenerationTelemetry]:
        runtime = environment.runtime
        confidence = environment.confidence
        hybrid_stop = environment.hybrid_stop

        runtime.best_score_series.append(float(guidance.best_score))
        runtime.best_priority_series.append(float(guidance.best_priority_score))
        improved_raw = bool(guidance.best_score > runtime.best_score)
        improved_priority = bool(
            guidance.best_priority_score > runtime.best_priority_score
        )
        significant_raw_improvement = bool(
            guidance.best_score > runtime.best_score + confidence.plateau_eps
        )

        if environment.strategy_name.lower() == "hybrid":
            if (
                guidance.best_score
                > runtime.hybrid_plateau_best + hybrid_stop.eps
            ):
                runtime.hybrid_plateau_best = float(guidance.best_score)
                runtime.hybrid_plateau_no_improve = 0
            elif generation > 0:
                runtime.hybrid_plateau_no_improve += 1
            if (
                not uses_fixed_evaluations(environment.cfg)
                and hybrid_stop.patience > 0
                and runtime.hybrid_plateau_no_improve >= hybrid_stop.patience
            ):
                runtime.hybrid_plateau_stop_applied = True
                runtime.hybrid_plateau_stop_generation = int(
                    environment.step_offset + generation
                )
        if improved_priority:
            runtime.best_priority_score = guidance.best_priority_score
        if improved_raw:
            best_index = int(
                np.argmax(
                    [
                        pose is candidate
                        for pose in evaluation.poses
                        for candidate in []
                    ]
                )
            ) if False else None
            del best_index
            # A pose vencedora é recuperada pela mesma regra de ranking usada
            # em `_guide_generation`, sem nova avaliação ou ordenação aleatória.
            raw_order = np.asarray(
                topk_indices(
                    guidance.selection_scores,
                    len(evaluation.poses),
                    largest=True,
                ),
                dtype=int,
            )
            selected_index = (
                int(raw_order[0])
                if len(raw_order)
                else int(np.argmax(evaluation.raw_scores))
            )
            runtime.best_score = guidance.best_score
            runtime.best_pose = evaluation.poses[selected_index]
            runtime.best_generation = environment.step_offset + generation
            runtime.best_pose.meta["pocket_id"] = getattr(
                environment.pocket,
                "id",
                None,
            )
        if significant_raw_improvement:
            runtime.confidence_no_improve_count = 0
        else:
            runtime.confidence_no_improve_count += 1
        runtime.best_global_score_series.append(float(runtime.best_score))

        telemetry = PopulationPocketSearchCycle._generation_telemetry(
            environment,
            evaluation=evaluation,
            guidance=guidance,
            generation=generation,
        )
        emit_generation_metrics(environment.logger, telemetry)
        runtime.generation_signals.append(
            {
                "best_score": float(guidance.best_score),
                "mean_score": float(np.mean(evaluation.raw_scores)),
                "elite_score_mean": float(guidance.elite_score_mean),
                "elite_score_std": float(guidance.elite_score_std),
                "elite_score_gap_top1_topk": float(guidance.elite_score_gap),
                "elite_coords_dispersion_mean": float(
                    guidance.elite_coords_dispersion
                ),
                "severe_clash_fraction": float(
                    guidance.severe_clash_fraction
                ),
                "zero_score_fraction": float(guidance.zero_score_fraction),
                "viable_fraction_proxy": float(guidance.viable_fraction),
                "selection_priority_mean": float(
                    guidance.selection_priority_mean
                ),
                "selection_priority_std": float(
                    guidance.selection_priority_std
                ),
                "confidence_signal_check": float(
                    1.0 if confidence.active_selection_enabled else 0.0
                ),
                "confidence_orientation_gate_needed": float(
                    1.0 if guidance.orientation_gate_needed else 0.0
                ),
                "confidence_orientation_gate_reason_count": float(
                    len(guidance.orientation_gate_reasons)
                ),
                "confidence_orientation_applied": float(
                    1.0 if guidance.orientation_applied else 0.0
                ),
                "confidence_orientation_reason_count": float(
                    len(guidance.orientation_reasons)
                ),
                "elite_fragment_mean_score": float(
                    guidance.elite_fragment_mean_score
                ),
                "archive_mean_score_before": (
                    float(guidance.archive_mean_before)
                    if guidance.archive_mean_before is not None
                    else 0.0
                ),
                "archive_injections": float(guidance.archive_injections),
                "n_eval_actual": float(evaluation.evaluation_count),
            }
        )

        if (confidence.search_control_enabled and confidence.early_stop_enabled
                and not uses_fixed_evaluations(environment.cfg)):
            stop_now, stop_payload = _plateau_stop_decision(
                generation=generation,
                min_generations=confidence.min_generations,
                best_score_series=runtime.best_global_score_series,
                plateau_window=confidence.plateau_window,
                plateau_abs_improvement_max=confidence.plateau_eps,
                elite_score_std=guidance.elite_score_std,
                max_elite_score_std=confidence.elite_std_max,
                elite_coords_dispersion_mean=guidance.elite_coords_dispersion,
                max_elite_coords_dispersion=confidence.dispersion_max,
                viable_fraction=guidance.viable_fraction,
                min_viable_fraction=confidence.min_viable_fraction,
                severe_clash_fraction=guidance.severe_clash_fraction,
                max_severe_clash_fraction=confidence.max_severe_clash_fraction,
                no_improve_count=runtime.confidence_no_improve_count,
                no_improve_patience=confidence.no_improve_patience,
            )
            runtime.confidence_stop_payload = stop_payload
            if stop_now:
                runtime.confidence_stop_applied = True
                runtime.confidence_stop_generation = int(telemetry.step)
                runtime.confidence_stop_reasons = list(
                    stop_payload.get("reasons", [])
                )
                emit_confidence_early_stop(
                    logger=environment.logger,
                    debug_logger=environment.settings.debug_logger,
                    tracer=environment.settings.tracer,
                    pocket=environment.pocket,
                    pocket_index=environment.pocket_index,
                    telemetry=telemetry,
                    stop_payload=stop_payload,
                    plateau_eps=confidence.plateau_eps,
                )
                return True, telemetry
        return False, telemetry
