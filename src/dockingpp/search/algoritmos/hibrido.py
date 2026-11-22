"""Classe única do algoritmo híbrido ABC + GA + Solis--Wets.

A classe deste módulo mantém juntas as três fases descritas no algoritmo
original. A infraestrutura comum do DockingPP (bolsões, avaliação, cache,
telemetria e resultados) continua em :class:`BasePopulationSearch`.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np

from dockingpp.domain.models import Pose

from .base import BasePopulationSearch
from .contexto import GenerationContext, SeedOrigin, inherit_seed, mark_seed_origin, seed_origin
from .espaco_rigido import ObjectiveEvaluationBudgetExhausted


class ABCGAVGOSSearch(BasePopulationSearch):
    """Executa uma geração híbrida na ordem ABC, Solis--Wets e GA.

    Esta é a única classe do pacote híbrido. Parâmetros, estado numérico e
    operadores foram mantidos como métodos privados para que o fluxo completo
    possa ser lido em um único arquivo sem criar implementações concorrentes.
    """

    strategy_name = "hybrid"
    algorithm_version = "hybrid_abc_ga_solis_wets_6d_v1"
    evaluates_candidates_internally = True

    # ------------------------------------------------------------------
    # Configuração da geração
    # ------------------------------------------------------------------
    @staticmethod
    def _configured(cfg: object, name: str, default: object) -> object:
        """Usa o valor configurado e recorre ao padrão somente para ``None``."""

        value = cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)
        return default if value is None else value

    @staticmethod
    def _explicitly_configured(cfg: object, name: str) -> bool:
        """Distingue valores fornecidos pelo usuário dos defaults do Pydantic."""

        if isinstance(cfg, Mapping):
            return name in cfg
        fields_set = getattr(cfg, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(cfg, "__fields_set__", None)
        if fields_set is not None:
            return name in fields_set
        return hasattr(cfg, name)

    @staticmethod
    def _onlooker_probabilities(fitness: np.ndarray) -> np.ndarray:
        """Converte fitness de minimizacao em probabilidades robustas do ABC."""

        values = np.asarray(fitness, dtype=float).reshape(-1)
        if values.size == 0:
            return values
        finite = np.isfinite(values)
        if not finite.any():
            return np.full(values.size, 1.0 / values.size, dtype=float)
        worst_finite = float(np.max(values[finite]))
        sanitized = np.where(finite, values, worst_finite + 1.0)
        weights = np.maximum(float(np.max(sanitized)) - sanitized, 0.0)
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(values.size, 1.0 / values.size, dtype=float)
        probabilities = weights / total
        probabilities[-1] += 1.0 - float(np.sum(probabilities))
        if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
            return np.full(values.size, 1.0 / values.size, dtype=float)
        return probabilities

    @staticmethod
    def _tournament_select(
        rng: np.random.Generator,
        fitness: np.ndarray,
        tournament_k: int,
    ) -> int:
        """Seleciona o menor fitness em um torneio sem reposicao."""

        values = np.asarray(fitness, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("tournament selection requires a non-empty population")
        sample_size = max(1, min(int(tournament_k), values.size))
        candidates = np.asarray(
            rng.choice(values.size, size=sample_size, replace=False),
            dtype=int,
        )
        return int(candidates[int(np.argmin(values[candidates]))])

    @staticmethod
    def _adapt_operator_probability(
        current: float,
        *,
        success_a: int,
        attempts_a: int,
        success_b: int,
        attempts_b: int,
        learning_rate: float,
        lower: float,
        upper: float,
    ) -> float:
        """Atualiza a probabilidade pela taxa empirica suavizada de sucesso."""

        rate_a = (float(success_a) + 1.0) / (float(attempts_a) + 2.0)
        rate_b = (float(success_b) + 1.0) / (float(attempts_b) + 2.0)
        target = rate_a / max(rate_a + rate_b, np.finfo(float).eps)
        updated = (
            (1.0 - float(learning_rate)) * float(current)
            + float(learning_rate) * target
        )
        return float(np.clip(updated, float(lower), float(upper)))

    @classmethod
    def _parameters_from_context(cls, context: GenerationContext) -> SimpleNamespace:
        """Resolve os parâmetros sem transformar zeros em valores padrão."""

        cfg = context.cfg
        pop_size = context.pop_size
        operator_state = context.hybrid_operator_state
        operator_mode = str(
            cls._configured(cfg, "hybrid_operator_mode", "canonical") or "canonical"
        ).strip().lower()
        if operator_mode not in {"legacy", "canonical"}:
            raise ValueError("hybrid_operator_mode must be legacy or canonical")

        legacy_sw_iter = cls._configured(cfg, "sw_max_iter", 15)
        sw_elite_default = (
            legacy_sw_iter if cls._explicitly_configured(cfg, "sw_max_iter") else 15
        )
        sw_heavy_default = (
            legacy_sw_iter if cls._explicitly_configured(cfg, "sw_max_iter") else 20
        )
        sw_elite_value = (
            cls._configured(cfg, "sw_elite_iter", sw_elite_default)
            if cls._explicitly_configured(cfg, "sw_elite_iter")
            else sw_elite_default
        )
        sw_heavy_value = (
            cls._configured(cfg, "sw_heavy_iter", sw_heavy_default)
            if cls._explicitly_configured(cfg, "sw_heavy_iter")
            else sw_heavy_default
        )

        return SimpleNamespace(
            operator_mode=operator_mode,
            alpha=float(cls._configured(cfg, "alpha", 0.8)),
            onlooker_count=int(
                max(
                    1,
                    round(
                        pop_size
                        * float(cls._configured(cfg, "onlooker_frac", 0.80))
                    ),
                )
            ),
            elite_size=int(max(1, cls._configured(cfg, "elite_size", 1))),
            tournament_k=int(
                max(
                    1,
                    min(
                        pop_size,
                        int(cls._configured(cfg, "tournament_k", 3)),
                    ),
                )
            ),
            limit_no_improve=int(
                max(1, int(cls._configured(cfg, "limit_no_improve", 30)))
            ),
            ga_interval=int(cls._configured(cfg, "ga_interval", 20)),
            p_crossover_ari=float(operator_state["p_crossover_ari"]),
            p_mutation_gauss=float(operator_state["p_mutation_gauss"]),
            adaptive_operator_probabilities=bool(
                cls._configured(cfg, "adaptive_operator_probabilities", False)
            ),
            operator_adaptation_rate=float(
                cls._configured(cfg, "operator_adaptation_rate", 0.25)
            ),
            operator_probability_min=float(
                cls._configured(cfg, "operator_probability_min", 0.1)
            ),
            operator_probability_max=float(
                cls._configured(cfg, "operator_probability_max", 0.9)
            ),
            top_frac_sw=float(cls._configured(cfg, "top_frac_sw", 0.10)),
            sw_interval=int(
                max(1, int(cls._configured(cfg, "sw_interval", 3)))
            ),
            sw_elite_iter=int(max(0, int(sw_elite_value))),
            sw_heavy_iter=int(max(0, int(sw_heavy_value))),
            sw_patience=int(
                max(1, int(cls._configured(cfg, "sw_patience", 8)))
            ),
            sw_iter_budget=int(
                max(0, int(cls._configured(cfg, "sw_iter_budget", 2000)))
            ),
        )

    # ------------------------------------------------------------------
    # Estado numérico compartilhado por ABC, Solis--Wets e GA
    # ------------------------------------------------------------------
    @staticmethod
    def _state_from_context(context: GenerationContext) -> SimpleNamespace:
        """Converte as poses em vetores 6D mantendo fitness, trials e ordem."""

        solutions = np.asarray(
            [context.existing_solution(pose) for pose in context.previous],
            dtype=float,
        )
        raw_fitness = -np.asarray(context.raw_previous_scores, dtype=float)
        finite_raw = np.isfinite(raw_fitness)
        if not finite_raw.all():
            fallback = (
                float(np.max(raw_fitness[finite_raw])) if finite_raw.any() else 1e12
            )
            raw_fitness = np.where(finite_raw, raw_fitness, fallback)
        trials = np.asarray(
            [
                int(
                    (pose.meta if isinstance(pose.meta, dict) else {}).get(
                        "trials", 0
                    )
                )
                for pose in context.previous
            ],
            dtype=int,
        )
        return SimpleNamespace(
            context=context,
            solutions=solutions,
            origins=[seed_origin(pose) for pose in context.previous],
            raw_fitness=raw_fitness,
            trials=trials,
            sigma=np.asarray([0.2, 0.2, 0.2, 5.0, 5.0, 5.0]),
        )

    @staticmethod
    def _clip_solution(state: SimpleNamespace, solution: np.ndarray) -> np.ndarray:
        return np.clip(
            np.asarray(solution, dtype=float),
            state.context.domain_low,
            state.context.domain_high,
        )

    @staticmethod
    def _select_k_not_i(state: SimpleNamespace, index: int) -> int:
        """Seleciona outro indivíduo por rejeição, preservando a ordem do RNG."""

        if state.context.pop_size <= 1:
            return index
        selected = index
        while selected == index:
            selected = int(
                state.context.rng.integers(0, state.context.pop_size)
            )
        return selected

    def _set_solution(
        self,
        state: SimpleNamespace,
        index: int,
        solution: np.ndarray,
        fitness: float,
        reset_trials: bool = True,
        donor: int | None = None,
        scout: bool = False,
    ) -> None:
        state.solutions[index] = self._clip_solution(state, solution)
        source = state.origins[index]
        if source is not None:
            if scout:
                state.origins[index] = SeedOrigin.UNRELATED
            else:
                donor_origin = state.origins[donor] if donor is not None else None
                state.origins[index] = inherit_seed(state.solutions[index], source, donor_origin)
        state.raw_fitness[index] = float(fitness) if np.isfinite(fitness) else 1e12
        if reset_trials:
            state.trials[index] = 0

    @staticmethod
    def _evaluate_fitness(state: SimpleNamespace, solution: np.ndarray) -> float:
        _pose, _score, fitness = state.context.evaluate_solution(solution)
        return float(fitness) if np.isfinite(fitness) else 1e12

    @staticmethod
    def _state_to_poses(state: SimpleNamespace) -> list[Pose]:
        """Materializa o estado otimizado no contrato público de poses."""

        poses: list[Pose] = []
        for index, solution in enumerate(state.solutions):
            score = -float(state.raw_fitness[index])
            pose = state.context.solution_to_pose(solution, score)
            mark_seed_origin(pose, state.origins[index])
            pose.meta["trials"] = int(state.trials[index])
            poses.append(pose)
        return poses

    # ------------------------------------------------------------------
    # Fase ABC: employed, onlooker e scout
    # ------------------------------------------------------------------
    def _run_abc_phases(
        self,
        state: SimpleNamespace,
        params: SimpleNamespace,
    ) -> None:
        """Atualiza a população in-place pelas três fases ABC históricas."""

        context = state.context
        rng = context.rng

        # Abelhas empregadas: cada fonte tenta uma perturbação diferencial.
        for index in range(context.pop_size):
            peer = self._select_k_not_i(state, index)
            phi = (rng.random(context.dim) * 2.0 - 1.0) * params.alpha
            candidate = self._clip_solution(
                state,
                state.solutions[index]
                + phi * (state.solutions[index] - state.solutions[peer]),
            )
            candidate_fit = self._evaluate_fitness(state, candidate)
            if candidate_fit < state.raw_fitness[index]:
                self._set_solution(state, index, candidate, candidate_fit, donor=peer)
            else:
                state.trials[index] += 1

        # No modo canônico, todas as fontes participam conforme o fitness.
        if params.operator_mode == "canonical":
            source_pool = np.arange(context.pop_size, dtype=int)
            source_probabilities = self._onlooker_probabilities(state.raw_fitness)
        else:
            source_pool = np.arange(
                max(1, min(len(context.selected), context.pop_size)),
                dtype=int,
            )
            source_probabilities = None
        for index in rng.choice(
            source_pool,
            size=params.onlooker_count,
            replace=True,
            p=source_probabilities,
        ):
            index = int(index)
            peer = self._select_k_not_i(state, index)
            phi = (rng.random(context.dim) * 2.0 - 1.0) * params.alpha
            candidate = self._clip_solution(
                state,
                state.solutions[index]
                + phi * (state.solutions[index] - state.solutions[peer]),
            )
            candidate_fit = self._evaluate_fitness(state, candidate)
            if candidate_fit < state.raw_fitness[index]:
                self._set_solution(state, index, candidate, candidate_fit, donor=peer)
            else:
                state.trials[index] += 1

        # Scouts reinicializam fontes estagnadas sem substituir a elite real.
        raw_elite = set(
            int(index)
            for index in np.argsort(state.raw_fitness)[: params.elite_size]
        )
        protected = raw_elite
        if params.operator_mode == "legacy":
            protected |= set(
                range(max(1, min(params.elite_size, len(context.selected))))
            )
        for index in np.where(state.trials >= params.limit_no_improve)[0]:
            if int(index) in protected:
                continue
            candidate = context.random_solution()
            self._set_solution(
                state,
                int(index),
                candidate,
                self._evaluate_fitness(state, candidate),
                scout=True,
            )

    # ------------------------------------------------------------------
    # Fase VGOS: busca local Solis--Wets
    # ------------------------------------------------------------------
    def _solis_wets_once(
        self,
        state: SimpleNamespace,
        solution: np.ndarray,
        best_fitness: float,
        max_iter: int,
        patience: int,
    ) -> tuple[np.ndarray, float, int]:
        """Busca local com aceitação estrita e parada por falta de melhora."""

        best = self._clip_solution(state, solution)
        best_value = float(best_fitness) if np.isfinite(best_fitness) else 1e12
        no_improve = 0
        used = 0
        for _ in range(max_iter):
            step = state.context.rng.uniform(
                -state.sigma,
                state.sigma,
                state.context.dim,
            )
            candidate = self._clip_solution(state, best + step)
            candidate_fit = self._evaluate_fitness(state, candidate)
            used += 1
            if candidate_fit < best_value:
                best = candidate
                best_value = candidate_fit
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
        return best, best_value, used

    def _run_solis_wets_phase(
        self,
        state: SimpleNamespace,
        params: SimpleNamespace,
    ) -> None:
        """Refina a elite sempre e os melhores no intervalo pesado."""

        elite = int(np.argmin(state.raw_fitness))
        budget_left = params.sw_iter_budget
        if params.sw_elite_iter > 0:
            improved, improved_fit, used = self._solis_wets_once(
                state,
                state.solutions[elite],
                float(state.raw_fitness[elite]),
                params.sw_elite_iter,
                params.sw_patience,
            )
            budget_left -= used
            if improved_fit < state.raw_fitness[elite]:
                self._set_solution(state, elite, improved, improved_fit)

        if (
            budget_left > 0
            and params.sw_heavy_iter > 0
            and state.context.generation_index % params.sw_interval == 0
        ):
            top_count = max(
                1,
                int(params.top_frac_sw * state.context.pop_size),
            )
            if params.operator_mode == "canonical":
                top_indices = np.argpartition(
                    state.raw_fitness,
                    top_count - 1,
                )[:top_count]
            else:
                top_indices = np.arange(
                    max(1, min(top_count, state.context.pop_size)),
                    dtype=int,
                )
            for index in top_indices:
                index = int(index)
                if index == elite or budget_left <= 0:
                    continue
                improved, improved_fit, used = self._solis_wets_once(
                    state,
                    state.solutions[index],
                    float(state.raw_fitness[index]),
                    min(params.sw_heavy_iter, budget_left),
                    params.sw_patience,
                )
                budget_left -= used
                if improved_fit < state.raw_fitness[index]:
                    self._set_solution(state, index, improved, improved_fit)

    # ------------------------------------------------------------------
    # Fase GA preservadora do híbrido
    # ------------------------------------------------------------------
    def _run_ga_phase(
        self,
        state: SimpleNamespace,
        params: SimpleNamespace,
    ) -> None:
        """Aplica elitismo, crossover e mutação no intervalo configurado."""

        context = state.context
        generation_number = context.generation_index
        if params.ga_interval <= 0 or generation_number % params.ga_interval != 0:
            return

        rng = context.rng
        new_solutions = np.empty_like(state.solutions)
        new_raw_fitness = np.empty_like(state.raw_fitness)
        new_trials = np.zeros_like(state.trials)
        elite = int(np.argmin(state.raw_fitness))
        new_origins = list(state.origins)
        new_origins[0] = state.origins[elite]
        new_solutions[0] = state.solutions[elite]
        new_raw_fitness[0] = state.raw_fitness[elite]
        parent_pool = np.arange(
            max(1, min(len(context.selected), context.pop_size)),
            dtype=int,
        )
        phase_attempts = {
            "arithmetic": 0,
            "binary": 0,
            "gaussian": 0,
            "random": 0,
        }
        phase_successes = {
            "arithmetic": 0,
            "binary": 0,
            "gaussian": 0,
            "random": 0,
        }

        for position in range(1, context.pop_size):
            if params.operator_mode == "canonical":
                first = self._tournament_select(
                    rng,
                    state.raw_fitness,
                    params.tournament_k,
                )
                second = self._tournament_select(
                    rng,
                    state.raw_fitness,
                    params.tournament_k,
                )
            else:
                first = int(rng.choice(parent_pool))
                second = int(rng.choice(parent_pool))
            base_index = (
                first
                if state.raw_fitness[first] <= state.raw_fitness[second]
                else second
            )

            if rng.random() < params.p_crossover_ari:
                crossover_operator = "arithmetic"
                mix = float(rng.random())
                child = (
                    mix * state.solutions[first]
                    + (1.0 - mix) * state.solutions[second]
                )
            else:
                crossover_operator = "binary"
                mask = rng.integers(0, 2, context.dim).astype(bool)
                child = np.where(
                    mask,
                    state.solutions[first],
                    state.solutions[second],
                )

            if rng.random() < params.p_mutation_gauss:
                mutation_operator = "gaussian"
                child = child + rng.normal(0.0, state.sigma, size=context.dim)
            else:
                mutation_operator = "random"
                coordinate = int(rng.integers(0, context.dim))
                child[coordinate] = rng.uniform(
                    context.domain_low[coordinate],
                    context.domain_high[coordinate],
                )

            phase_attempts[crossover_operator] += 1
            phase_attempts[mutation_operator] += 1
            child = self._clip_solution(state, child)
            child_fit = self._evaluate_fitness(state, child)
            accepted = bool(child_fit <= state.raw_fitness[base_index])
            if accepted:
                phase_successes[crossover_operator] += 1
                phase_successes[mutation_operator] += 1
                new_solutions[position] = child
                new_raw_fitness[position] = child_fit
                new_trials[position] = 0
                new_origins[position] = inherit_seed(child, state.origins[first], state.origins[second])
            else:
                new_solutions[position] = state.solutions[base_index]
                new_raw_fitness[position] = state.raw_fitness[base_index]
                new_trials[position] = state.trials[base_index]
                new_origins[position] = state.origins[base_index]

        state.solutions = new_solutions
        state.raw_fitness = new_raw_fitness
        state.trials = new_trials
        state.origins = new_origins
        operator_state = context.hybrid_operator_state
        operator_state["ga_phases"] += 1
        for name in phase_attempts:
            operator_state["attempts"][name] += int(phase_attempts[name])
            operator_state["successes"][name] += int(phase_successes[name])

        if (
            params.operator_mode == "canonical"
            and params.adaptive_operator_probabilities
        ):
            operator_state["p_crossover_ari"] = self._adapt_operator_probability(
                params.p_crossover_ari,
                success_a=phase_successes["arithmetic"],
                attempts_a=phase_attempts["arithmetic"],
                success_b=phase_successes["binary"],
                attempts_b=phase_attempts["binary"],
                learning_rate=params.operator_adaptation_rate,
                lower=params.operator_probability_min,
                upper=params.operator_probability_max,
            )
            operator_state["p_mutation_gauss"] = self._adapt_operator_probability(
                params.p_mutation_gauss,
                success_a=phase_successes["gaussian"],
                attempts_a=phase_attempts["gaussian"],
                success_b=phase_successes["random"],
                attempts_b=phase_attempts["random"],
                learning_rate=params.operator_adaptation_rate,
                lower=params.operator_probability_min,
                upper=params.operator_probability_max,
            )
            operator_state["adaptation_updates"] += 1

        context.hybrid_operator_history.append(
            {
                "generation": int(generation_number),
                "mode": params.operator_mode,
                "p_crossover_ari_before": float(params.p_crossover_ari),
                "p_crossover_ari_after": float(
                    operator_state["p_crossover_ari"]
                ),
                "p_mutation_gauss_before": float(params.p_mutation_gauss),
                "p_mutation_gauss_after": float(
                    operator_state["p_mutation_gauss"]
                ),
                "attempts": dict(phase_attempts),
                "successes": dict(phase_successes),
            }
        )

    # ------------------------------------------------------------------
    # Composição pública usada pelo ciclo compartilhado
    # ------------------------------------------------------------------
    def _next_generation(self, context: GenerationContext) -> list[Pose]:
        """Executa ABC, Solis--Wets e GA sobre o mesmo estado, nessa ordem."""

        if int(context.dim) != 6:
            raise ValueError(
                "ABCGAVGOSSearch requer o espaço rígido 6D do DockingPP"
            )
        params = self._parameters_from_context(context)
        state = self._state_from_context(context)
        try:
            self._run_abc_phases(state, params)
            self._run_solis_wets_phase(state, params)
            self._run_ga_phase(state, params)
        except ObjectiveEvaluationBudgetExhausted:
            # A tentativa que ultrapassaria o limite é descartada; as soluções
            # mantidas no estado já possuem fitness conhecido.
            poses = self._state_to_poses(state)
            for pose in poses:
                pose.meta["objective_evaluation_budget_exhausted"] = True
            return poses
        return self._state_to_poses(state)
