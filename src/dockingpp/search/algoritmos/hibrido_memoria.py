"""Algoritmo independente ABC+AG+VGOS-M com memoria adaptativa.

Este modulo implementa a variante com memoria como uma engine propria. Ele nao
herda do hibrido base: as fases ABC, VGOS/Solis-Wets, AG e memoria ficam aqui
para que a proposta metodologica possa ser lida e alterada de forma isolada.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

import numpy as np

from dockingpp.domain.models import Pose

from .base import BasePopulationSearch
from .contexto import GenerationContext
from .espaco_rigido import ObjectiveEvaluationBudgetExhausted


class ABCGAVGOSMemorySearch(BasePopulationSearch):
    """Executa ABC, VGOS/Solis-Wets, AG e memoria adaptativa top-k."""

    strategy_name = "hybrid_memory"
    algorithm_version = "hybrid_abc_ga_solis_wets_memory_6d_v1"
    evaluates_candidates_internally = True

    # ------------------------------------------------------------------
    # Configuracao
    # ------------------------------------------------------------------
    @staticmethod
    def _configured(cfg: object, name: str, default: object) -> object:
        value = cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)
        return default if value is None else value

    @classmethod
    def _configured_first(
        cls,
        cfg: object,
        names: tuple[str, ...],
        default: object,
    ) -> object:
        for name in names:
            if cls._explicitly_configured(cfg, name):
                return cls._configured(cfg, name, default)
        return default

    @staticmethod
    def _explicitly_configured(cfg: object, name: str) -> bool:
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
        cfg = context.cfg
        pop_size = context.pop_size
        operator_state = context.hybrid_operator_state
        operator_mode = str(
            cls._configured_first(
                cfg,
                ("hybrid_memory_operator_mode", "hybrid_operator_mode"),
                "canonical",
            )
            or "canonical"
        ).strip().lower()
        if operator_mode not in {"legacy", "canonical"}:
            raise ValueError("hybrid_operator_mode must be legacy or canonical")

        if not operator_state.get("hybrid_memory_defaults_initialized", False):
            if (
                not cls._explicitly_configured(cfg, "hybrid_memory_p_crossover_ari")
                and not cls._explicitly_configured(cfg, "p_crossover_ari")
            ):
                operator_state["p_crossover_ari"] = 0.5
            if (
                not cls._explicitly_configured(cfg, "hybrid_memory_p_mutation_gauss")
                and not cls._explicitly_configured(cfg, "p_mutation_gauss")
            ):
                operator_state["p_mutation_gauss"] = 0.7
            operator_state["hybrid_memory_defaults_initialized"] = True

        legacy_sw_iter = cls._configured_first(
            cfg,
            ("hybrid_memory_sw_max_iter", "sw_max_iter"),
            20,
        )
        sw_elite_default = (
            legacy_sw_iter
            if (
                cls._explicitly_configured(cfg, "hybrid_memory_sw_max_iter")
                or cls._explicitly_configured(cfg, "sw_max_iter")
            )
            else 15
        )
        sw_heavy_default = (
            legacy_sw_iter
            if (
                cls._explicitly_configured(cfg, "hybrid_memory_sw_max_iter")
                or cls._explicitly_configured(cfg, "sw_max_iter")
            )
            else 20
        )
        sw_elite_value = cls._configured_first(
            cfg,
            ("hybrid_memory_sw_elite_iter", "sw_elite_iter"),
            sw_elite_default,
        )
        sw_heavy_value = cls._configured_first(
            cfg,
            ("hybrid_memory_sw_heavy_iter", "sw_heavy_iter"),
            sw_heavy_default,
        )

        return SimpleNamespace(
            operator_mode=operator_mode,
            alpha=float(cls._configured_first(cfg, ("hybrid_memory_alpha", "alpha"), 0.8)),
            onlooker_count=int(
                max(
                    1,
                    round(
                        pop_size
                        * float(
                            cls._configured_first(
                                cfg,
                                ("hybrid_memory_onlooker_frac", "onlooker_frac"),
                                0.80,
                            )
                        )
                    ),
                )
            ),
            elite_size=int(
                max(1, cls._configured_first(cfg, ("hybrid_memory_elite_size", "elite_size"), 1))
            ),
            tournament_k=int(
                max(
                    1,
                    min(
                        pop_size,
                        int(
                            cls._configured_first(
                                cfg,
                                ("hybrid_memory_tournament_k", "tournament_k"),
                                3,
                            )
                        ),
                    ),
                )
            ),
            limit_no_improve=int(
                max(
                    1,
                    int(
                        cls._configured_first(
                            cfg,
                            ("hybrid_memory_limit_no_improve", "limit_no_improve"),
                            30,
                        )
                    ),
                )
            ),
            ga_interval=int(
                cls._configured_first(cfg, ("hybrid_memory_ga_interval", "ga_interval"), 20)
            ),
            p_crossover_ari=float(
                cls._configured_first(
                    cfg,
                    ("hybrid_memory_p_crossover_ari",),
                    operator_state["p_crossover_ari"],
                )
            ),
            p_mutation_gauss=float(
                cls._configured_first(
                    cfg,
                    ("hybrid_memory_p_mutation_gauss",),
                    operator_state["p_mutation_gauss"],
                )
            ),
            adaptive_operator_probabilities=bool(
                cls._configured_first(
                    cfg,
                    (
                        "hybrid_memory_adaptive_operator_probabilities",
                        "adaptive_operator_probabilities",
                    ),
                    False,
                )
            ),
            operator_adaptation_rate=float(
                cls._configured_first(
                    cfg,
                    ("hybrid_memory_operator_adaptation_rate", "operator_adaptation_rate"),
                    0.25,
                )
            ),
            operator_probability_min=float(
                cls._configured_first(
                    cfg,
                    ("hybrid_memory_operator_probability_min", "operator_probability_min"),
                    0.1,
                )
            ),
            operator_probability_max=float(
                cls._configured_first(
                    cfg,
                    ("hybrid_memory_operator_probability_max", "operator_probability_max"),
                    0.9,
                )
            ),
            top_frac_sw=float(
                cls._configured_first(cfg, ("hybrid_memory_top_frac_sw", "top_frac_sw"), 0.10)
            ),
            sw_interval=int(
                max(
                    1,
                    int(
                        cls._configured_first(
                            cfg,
                            ("hybrid_memory_sw_interval", "sw_interval"),
                            3,
                        )
                    ),
                )
            ),
            sw_elite_iter=int(max(0, int(sw_elite_value))),
            sw_heavy_iter=int(max(0, int(sw_heavy_value))),
            sw_patience=int(
                max(
                    1,
                    int(cls._configured_first(cfg, ("hybrid_memory_sw_patience", "sw_patience"), 8)),
                )
            ),
            sw_iter_budget=int(
                max(
                    0,
                    int(
                        cls._configured_first(
                            cfg,
                            ("hybrid_memory_sw_iter_budget", "sw_iter_budget"),
                            2000,
                        )
                    ),
                )
            ),
        )

    @classmethod
    def _memory_parameters_from_context(
        cls,
        context: GenerationContext,
    ) -> SimpleNamespace:
        cfg = context.cfg
        default_size = 15
        default_inject = 2
        archive_size = (
            cls._configured(cfg, "hybrid_memory_size", default_size)
            if cls._explicitly_configured(cfg, "hybrid_memory_size")
            else default_size
        )
        inject_count = (
            cls._configured(cfg, "hybrid_memory_inject_count", default_inject)
            if cls._explicitly_configured(cfg, "hybrid_memory_inject_count")
            else default_inject
        )
        return SimpleNamespace(
            archive_size=max(1, int(archive_size)),
            inject_count=max(0, int(inject_count)),
            warmup_generations=max(
                0,
                int(cls._configured(cfg, "hybrid_memory_warmup_generations", 1)),
            ),
            replace_only_if_better=bool(
                cls._configured(cfg, "hybrid_memory_replace_only_if_better", True)
            ),
            rounding_decimals=max(
                1,
                int(cls._configured(cfg, "hybrid_memory_rounding_decimals", 8)),
            ),
        )

    # ------------------------------------------------------------------
    # Estado 6D compartilhado pelas fases
    # ------------------------------------------------------------------
    @staticmethod
    def _state_from_context(context: GenerationContext) -> SimpleNamespace:
        solutions = np.asarray(
            [context.existing_solution(pose) for pose in context.previous],
            dtype=float,
        )
        raw_fitness = -np.asarray(context.raw_previous_scores, dtype=float)
        finite_raw = np.isfinite(raw_fitness)
        if not finite_raw.all():
            fallback = float(np.max(raw_fitness[finite_raw])) if finite_raw.any() else 1e12
            raw_fitness = np.where(finite_raw, raw_fitness, fallback)
        trials = np.asarray(
            [
                int((pose.meta if isinstance(pose.meta, dict) else {}).get("trials", 0))
                for pose in context.previous
            ],
            dtype=int,
        )
        return SimpleNamespace(
            context=context,
            solutions=solutions,
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
        if state.context.pop_size <= 1:
            return index
        selected = index
        while selected == index:
            selected = int(state.context.rng.integers(0, state.context.pop_size))
        return selected

    def _set_solution(
        self,
        state: SimpleNamespace,
        index: int,
        solution: np.ndarray,
        fitness: float,
        reset_trials: bool = True,
    ) -> None:
        state.solutions[index] = self._clip_solution(state, solution)
        state.raw_fitness[index] = float(fitness) if np.isfinite(fitness) else 1e12
        if reset_trials:
            state.trials[index] = 0

    @staticmethod
    def _evaluate_fitness(state: SimpleNamespace, solution: np.ndarray) -> float:
        _pose, _score, fitness = state.context.evaluate_solution(solution)
        return float(fitness) if np.isfinite(fitness) else 1e12

    @staticmethod
    def _state_to_poses(state: SimpleNamespace) -> list[Pose]:
        poses: list[Pose] = []
        for index, solution in enumerate(state.solutions):
            score = -float(state.raw_fitness[index])
            pose = state.context.solution_to_pose(solution, score)
            pose.meta["trials"] = int(state.trials[index])
            poses.append(pose)
        return poses

    # ------------------------------------------------------------------
    # Fase ABC
    # ------------------------------------------------------------------
    def _run_abc_phases(
        self,
        state: SimpleNamespace,
        params: SimpleNamespace,
    ) -> None:
        context = state.context
        rng = context.rng

        for index in range(context.pop_size):
            peer = self._select_k_not_i(state, index)
            phi = (rng.random(context.dim) * 2.0 - 1.0) * params.alpha
            candidate = self._clip_solution(
                state,
                state.solutions[index] + phi * (state.solutions[index] - state.solutions[peer]),
            )
            candidate_fit = self._evaluate_fitness(state, candidate)
            if candidate_fit < state.raw_fitness[index]:
                self._set_solution(state, index, candidate, candidate_fit)
            else:
                state.trials[index] += 1

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
                state.solutions[index] + phi * (state.solutions[index] - state.solutions[peer]),
            )
            candidate_fit = self._evaluate_fitness(state, candidate)
            if candidate_fit < state.raw_fitness[index]:
                self._set_solution(state, index, candidate, candidate_fit)
            else:
                state.trials[index] += 1

        raw_elite = set(int(index) for index in np.argsort(state.raw_fitness)[: params.elite_size])
        protected = raw_elite
        if params.operator_mode == "legacy":
            protected |= set(range(max(1, min(params.elite_size, len(context.selected)))))
        for index in np.where(state.trials >= params.limit_no_improve)[0]:
            if int(index) in protected:
                continue
            candidate = context.random_solution()
            self._set_solution(
                state,
                int(index),
                candidate,
                self._evaluate_fitness(state, candidate),
            )

    # ------------------------------------------------------------------
    # Fase VGOS/Solis-Wets
    # ------------------------------------------------------------------
    def _solis_wets_once(
        self,
        state: SimpleNamespace,
        solution: np.ndarray,
        best_fitness: float,
        max_iter: int,
        patience: int,
    ) -> tuple[np.ndarray, float, int]:
        best = self._clip_solution(state, solution)
        best_value = float(best_fitness) if np.isfinite(best_fitness) else 1e12
        no_improve = 0
        used = 0
        for _ in range(max_iter):
            step = state.context.rng.uniform(-state.sigma, state.sigma, state.context.dim)
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
            top_count = max(1, int(params.top_frac_sw * state.context.pop_size))
            if params.operator_mode == "canonical":
                top_indices = np.argpartition(state.raw_fitness, top_count - 1)[:top_count]
            else:
                top_indices = np.arange(max(1, min(top_count, state.context.pop_size)), dtype=int)
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
    # Fase AG
    # ------------------------------------------------------------------
    def _run_ga_phase(
        self,
        state: SimpleNamespace,
        params: SimpleNamespace,
    ) -> None:
        context = state.context
        generation_number = context.generation_index
        if params.ga_interval <= 0 or generation_number % params.ga_interval != 0:
            return

        rng = context.rng
        new_solutions = np.empty_like(state.solutions)
        new_raw_fitness = np.empty_like(state.raw_fitness)
        new_trials = np.zeros_like(state.trials)
        elite = int(np.argmin(state.raw_fitness))
        new_solutions[0] = state.solutions[elite]
        new_raw_fitness[0] = state.raw_fitness[elite]
        parent_pool = np.arange(max(1, min(len(context.selected), context.pop_size)), dtype=int)
        phase_attempts = {"arithmetic": 0, "binary": 0, "gaussian": 0, "random": 0}
        phase_successes = {"arithmetic": 0, "binary": 0, "gaussian": 0, "random": 0}

        for position in range(1, context.pop_size):
            if params.operator_mode == "canonical":
                first = self._tournament_select(rng, state.raw_fitness, params.tournament_k)
                second = self._tournament_select(rng, state.raw_fitness, params.tournament_k)
            else:
                first = int(rng.choice(parent_pool))
                second = int(rng.choice(parent_pool))
            base_index = first if state.raw_fitness[first] <= state.raw_fitness[second] else second

            if rng.random() < params.p_crossover_ari:
                crossover_operator = "arithmetic"
                mix = float(rng.random())
                child = mix * state.solutions[first] + (1.0 - mix) * state.solutions[second]
            else:
                crossover_operator = "binary"
                mask = rng.integers(0, 2, context.dim).astype(bool)
                child = np.where(mask, state.solutions[first], state.solutions[second])

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
            else:
                new_solutions[position] = state.solutions[base_index]
                new_raw_fitness[position] = state.raw_fitness[base_index]
                new_trials[position] = state.trials[base_index]

        state.solutions = new_solutions
        state.raw_fitness = new_raw_fitness
        state.trials = new_trials
        operator_state = context.hybrid_operator_state
        operator_state["ga_phases"] += 1
        for name in phase_attempts:
            operator_state["attempts"][name] += int(phase_attempts[name])
            operator_state["successes"][name] += int(phase_successes[name])

        if params.operator_mode == "canonical" and params.adaptive_operator_probabilities:
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
                "p_crossover_ari_after": float(operator_state["p_crossover_ari"]),
                "p_mutation_gauss_before": float(params.p_mutation_gauss),
                "p_mutation_gauss_after": float(operator_state["p_mutation_gauss"]),
                "attempts": dict(phase_attempts),
                "successes": dict(phase_successes),
            }
        )

    # ------------------------------------------------------------------
    # Memoria adaptativa top-k
    # ------------------------------------------------------------------
    @staticmethod
    def _ensure_memory_state(context: GenerationContext) -> dict[str, Any]:
        operator_state = context.hybrid_operator_state
        operator_state.setdefault("memory_archive", [])
        operator_state.setdefault("memory_updates", 0)
        operator_state.setdefault("memory_injections", 0)
        operator_state.setdefault("memory_best_score", None)
        operator_state.setdefault("memory_last_injected_generation", None)
        operator_state.setdefault("memory_history", [])
        return operator_state

    @staticmethod
    def _entry_key(solution: np.ndarray, decimals: int) -> tuple[float, ...]:
        rounded = np.round(np.asarray(solution, dtype=float), decimals=int(decimals))
        return tuple(float(value) for value in rounded)

    def _update_memory_archive(
        self,
        state: SimpleNamespace,
        memory_params: SimpleNamespace,
    ) -> None:
        context = state.context
        operator_state = self._ensure_memory_state(context)
        operator_state["memory_archive_capacity"] = int(memory_params.archive_size)
        archive = list(operator_state.get("memory_archive") or [])
        by_key: dict[tuple[float, ...], dict[str, Any]] = {}

        for entry in archive:
            solution = np.asarray(entry.get("solution", []), dtype=float)
            if solution.shape != (context.dim,):
                continue
            key = self._entry_key(solution, memory_params.rounding_decimals)
            by_key[key] = {
                "solution": [float(value) for value in solution],
                "raw_fitness": float(entry.get("raw_fitness", 1e12)),
                "score": float(entry.get("score", -float(entry.get("raw_fitness", 1e12)))),
                "generation": int(entry.get("generation", context.generation_index)),
            }

        for solution, raw_fitness in zip(state.solutions, state.raw_fitness):
            fitness = float(raw_fitness) if np.isfinite(raw_fitness) else 1e12
            key = self._entry_key(solution, memory_params.rounding_decimals)
            current = by_key.get(key)
            if current is None or fitness < float(current["raw_fitness"]):
                by_key[key] = {
                    "solution": [float(value) for value in np.asarray(solution, dtype=float)],
                    "raw_fitness": fitness,
                    "score": -fitness,
                    "generation": int(context.generation_index),
                }

        ordered = sorted(by_key.values(), key=lambda item: float(item["raw_fitness"]))
        archive = ordered[: int(memory_params.archive_size)]
        operator_state["memory_archive"] = archive
        operator_state["memory_updates"] = int(operator_state["memory_updates"]) + 1
        operator_state["memory_best_score"] = float(archive[0]["score"]) if archive else None

    def _inject_memory_archive(
        self,
        state: SimpleNamespace,
        memory_params: SimpleNamespace,
    ) -> int:
        context = state.context
        operator_state = self._ensure_memory_state(context)
        archive = list(operator_state.get("memory_archive") or [])
        if (
            not archive
            or memory_params.inject_count <= 0
            or context.generation_index < memory_params.warmup_generations
        ):
            return 0

        worst_indices = np.argsort(state.raw_fitness)[::-1]
        injections = 0
        for entry, index in zip(archive, worst_indices):
            if injections >= int(memory_params.inject_count):
                break
            solution = np.asarray(entry.get("solution", []), dtype=float)
            if solution.shape != (context.dim,):
                continue
            fitness = float(entry.get("raw_fitness", 1e12))
            index = int(index)
            if memory_params.replace_only_if_better and fitness >= float(state.raw_fitness[index]):
                continue
            self._set_solution(state, index, solution, fitness, reset_trials=True)
            injections += 1

        if injections:
            operator_state["memory_injections"] = int(operator_state["memory_injections"]) + injections
            operator_state["memory_last_injected_generation"] = int(context.generation_index)
            operator_state["memory_history"].append(
                {
                    "generation": int(context.generation_index),
                    "injections": int(injections),
                    "archive_size": int(len(archive)),
                    "best_score": float(archive[0]["score"]),
                }
            )
        return int(injections)

    # ------------------------------------------------------------------
    # Composicao da geracao ABC+AG+VGOS-M
    # ------------------------------------------------------------------
    def _next_generation(self, context: GenerationContext) -> list[Pose]:
        if int(context.dim) != 6:
            raise ValueError(
                "ABCGAVGOSMemorySearch requer o espaco rigido 6D do DockingPP"
            )
        params = self._parameters_from_context(context)
        memory_params = self._memory_parameters_from_context(context)
        state = self._state_from_context(context)
        try:
            self._update_memory_archive(state, memory_params)
            self._run_abc_phases(state, params)
            self._run_solis_wets_phase(state, params)
            self._run_ga_phase(state, params)
            self._update_memory_archive(state, memory_params)
            self._inject_memory_archive(state, memory_params)
            self._update_memory_archive(state, memory_params)
        except ObjectiveEvaluationBudgetExhausted:
            poses = self._state_to_poses(state)
            for pose in poses:
                pose.meta["objective_evaluation_budget_exhausted"] = True
            return poses
        return self._state_to_poses(state)
