"""Baseline de algoritmo genético real no espaço rígido de seis graus.

O baseline usa torneio, elitismo, *simulated binary crossover* (SBX) e mutação
polinomial limitada. Os operadores reais seguem Deb e Agrawal (1995), enquanto
o ciclo e a função objetivo são compartilhados com os outros motores.
"""

from __future__ import annotations

import numpy as np

from dockingpp.domain.models import Pose

from .base import BasePopulationSearch
from .comum import _tournament_select
from .contexto import GenerationContext, inherit_seed, mark_seed_origin, seed_origin


def _finite_fitness(raw_scores: np.ndarray) -> np.ndarray:
    """Converte scores de maximização em fitness finito de minimização."""

    fitness = -np.asarray(raw_scores, dtype=float).reshape(-1)
    finite = np.isfinite(fitness)
    if finite.all():
        return fitness
    fallback = float(np.max(fitness[finite]) + 1.0) if finite.any() else 1e12
    return np.where(finite, fitness, fallback)


def _sbx_pair(
    rng: np.random.Generator,
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Gera dois descendentes pelo SBX limitado de Deb--Agrawal."""

    first = np.asarray(parent_a, dtype=float)
    second = np.asarray(parent_b, dtype=float)
    child_a = first.copy()
    child_b = second.copy()
    exponent = 1.0 / (max(0.0, float(eta)) + 1.0)
    epsilon = np.finfo(float).eps

    for coordinate in range(first.size):
        if rng.random() > 0.5 or abs(first[coordinate] - second[coordinate]) <= epsilon:
            continue
        x1 = min(first[coordinate], second[coordinate])
        x2 = max(first[coordinate], second[coordinate])
        low = float(lower[coordinate])
        high = float(upper[coordinate])
        random_value = float(rng.random())

        beta = 1.0 + 2.0 * (x1 - low) / (x2 - x1)
        alpha = 2.0 - beta ** -(float(eta) + 1.0)
        beta_q = (
            (random_value * alpha) ** exponent
            if random_value <= 1.0 / alpha
            else (1.0 / (2.0 - random_value * alpha)) ** exponent
        )
        offspring_low = 0.5 * ((x1 + x2) - beta_q * (x2 - x1))

        beta = 1.0 + 2.0 * (high - x2) / (x2 - x1)
        alpha = 2.0 - beta ** -(float(eta) + 1.0)
        beta_q = (
            (random_value * alpha) ** exponent
            if random_value <= 1.0 / alpha
            else (1.0 / (2.0 - random_value * alpha)) ** exponent
        )
        offspring_high = 0.5 * ((x1 + x2) + beta_q * (x2 - x1))

        offspring_low = float(np.clip(offspring_low, low, high))
        offspring_high = float(np.clip(offspring_high, low, high))
        if rng.random() <= 0.5:
            child_a[coordinate], child_b[coordinate] = offspring_high, offspring_low
        else:
            child_a[coordinate], child_b[coordinate] = offspring_low, offspring_high

    return child_a, child_b


def _polynomial_mutation(
    rng: np.random.Generator,
    individual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    probability: float,
    eta: float,
) -> np.ndarray:
    """Aplica mutação polinomial limitada, coordenada a coordenada."""

    mutated = np.asarray(individual, dtype=float).copy()
    exponent = 1.0 / (max(0.0, float(eta)) + 1.0)
    for coordinate in range(mutated.size):
        if rng.random() > probability:
            continue
        low = float(lower[coordinate])
        high = float(upper[coordinate])
        span = high - low
        if span <= 0.0:
            mutated[coordinate] = low
            continue
        value = float(np.clip(mutated[coordinate], low, high))
        delta_low = (value - low) / span
        delta_high = (high - value) / span
        random_value = float(rng.random())
        if random_value <= 0.5:
            xy = 1.0 - delta_low
            val = 2.0 * random_value + (1.0 - 2.0 * random_value) * (
                xy ** (float(eta) + 1.0)
            )
            delta = val**exponent - 1.0
        else:
            xy = 1.0 - delta_high
            val = 2.0 * (1.0 - random_value) + 2.0 * (random_value - 0.5) * (
                xy ** (float(eta) + 1.0)
            )
            delta = 1.0 - val**exponent
        mutated[coordinate] = float(np.clip(value + delta * span, low, high))
    return mutated


class AGSearch(BasePopulationSearch):
    """AG real independente, com indivíduos rígidos ``[t, r]`` em 6D."""

    strategy_name = "ag"
    algorithm_version = "ag_sbx_polynomial_6d_v1"

    def _next_generation(self, context: GenerationContext) -> list[Pose]:
        """Produz uma geração completa sem deformar coordenadas atômicas."""

        if int(context.dim) != 6:
            raise ValueError("AGSearch requer o espaço rígido 6D do DockingPP")
        if context.pop_size <= 0:
            return []
        if not context.previous:
            return [context.random_pose() for _ in range(context.pop_size)]

        solutions = np.asarray(
            [context.existing_solution(pose) for pose in context.previous],
            dtype=float,
        )
        fitness = _finite_fitness(context.raw_previous_scores)
        population_size = min(len(solutions), len(fitness))
        solutions = solutions[:population_size]
        fitness = fitness[:population_size]
        if population_size == 0:
            return [context.random_pose() for _ in range(context.pop_size)]

        cfg = context.cfg
        elite_count = max(
            0,
            min(
                context.pop_size,
                population_size,
                int(getattr(cfg, "ag_elite_size", 1)),
            ),
        )
        tournament_k = max(
            1,
            min(population_size, int(getattr(cfg, "ag_tournament_k", 3))),
        )
        crossover_probability = float(
            np.clip(getattr(cfg, "ag_crossover_probability", 0.9), 0.0, 1.0)
        )
        configured_mutation = getattr(cfg, "ag_mutation_probability", None)
        mutation_probability = (
            1.0 / float(context.dim)
            if configured_mutation is None
            else float(np.clip(configured_mutation, 0.0, 1.0))
        )
        crossover_eta = max(0.0, float(getattr(cfg, "ag_sbx_eta", 15.0)))
        mutation_eta = max(0.0, float(getattr(cfg, "ag_mutation_eta", 20.0)))

        elite = np.argsort(fitness)[:elite_count]
        next_solutions = [solutions[int(index)].copy() for index in elite]
        origins = [seed_origin(pose) for pose in context.previous]
        next_origins = [origins[int(index)] for index in elite]
        while len(next_solutions) < context.pop_size:
            first = _tournament_select(context.rng, fitness, tournament_k)
            second = _tournament_select(context.rng, fitness, tournament_k)
            parent_a = solutions[first]
            parent_b = solutions[second]
            if context.rng.random() <= crossover_probability:
                parents = [(origins[first], origins[second])] * 2
                child_a, child_b = _sbx_pair(
                    context.rng,
                    parent_a,
                    parent_b,
                    context.domain_low,
                    context.domain_high,
                    crossover_eta,
                )
            else:
                child_a, child_b = parent_a.copy(), parent_b.copy()
                parents = [(origins[first],), (origins[second],)]
            for child, child_parents in zip((child_a, child_b), parents):
                mutated = _polynomial_mutation(
                    context.rng,
                    child,
                    context.domain_low,
                    context.domain_high,
                    mutation_probability,
                    mutation_eta,
                )
                next_solutions.append(mutated)
                next_origins.append(inherit_seed(mutated, *child_parents))
                if len(next_solutions) >= context.pop_size:
                    break

        poses = []
        for solution, origin in zip(next_solutions, next_origins):
            pose = context.solution_to_pose(solution, None)
            mark_seed_origin(pose, origin)
            pose.meta["baseline_algorithm"] = "ag_sbx_polynomial"
            poses.append(pose)
        return poses
