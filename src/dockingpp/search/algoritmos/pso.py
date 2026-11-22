"""Baseline PSO clássico no espaço rígido de seis graus de liberdade.

Cada partícula é um vetor ``[tx, ty, tz, rx, ry, rz]``.  O estado necessário
entre gerações fica serializável em :attr:`Pose.meta`, pois o ciclo comum pode
reordenar a população entre duas chamadas deste operador.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from dockingpp.domain.models import Pose

from .base import BasePopulationSearch
from .contexto import GenerationContext, SeedOrigin, inherit_seed, mark_seed_origin, seed_origin


_PSO_STATE_KEY = "pso_state"


@dataclass(slots=True)
class _ParticleState:
    """Estado numérico de uma partícula antes da atualização de velocidade."""

    position: np.ndarray
    velocity: np.ndarray
    personal_best_position: np.ndarray
    personal_best_score: float
    seed_origin: SeedOrigin | None = None


def _config_float(
    cfg: Any,
    names: Iterable[str],
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    """Lê o primeiro parâmetro numérico válido, preservando o valor zero."""

    for name in names:
        value = getattr(cfg, name, None)
        if value is None:
            continue
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(parsed):
            continue
        if minimum is not None and parsed < minimum:
            continue
        if maximum is not None and parsed > maximum:
            continue
        return parsed
    return float(default)


def _finite_score(value: Any, default: float = float("-inf")) -> float:
    """Normaliza scores inválidos sem permitir que NaN vença uma comparação."""

    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _state_vector(value: Any, dim: int) -> np.ndarray | None:
    """Converte um vetor salvo em metadados, rejeitando estado corrompido."""

    try:
        vector = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if vector.shape != (dim,) or not np.all(np.isfinite(vector)):
        return None
    return vector.copy()


class PSOSearch(BasePopulationSearch):
    """Particle Swarm Optimization clássico, limitado ao espaço rígido 6D.

    A equação implementada é a formulação de Kennedy e Eberhart (1995):
    ``v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)`` e ``x = x + v``.
    Os padrões ``w=1`` e ``c1=c2=2`` correspondem à formulação clássica;
    o limite de velocidade (20% da amplitude de cada dimensão) impede saltos
    incompatíveis com os limites translacionais e rotacionais do docking.
    """

    strategy_name = "pso"
    algorithm_version = "pso_kennedy_eberhart_6d_v1"

    def _next_generation(self, context: GenerationContext) -> list[Pose]:
        """Atualiza todas as partículas sem deformar coordenadas atômicas."""

        if int(context.dim) != 6:
            raise ValueError("PSOSearch requer o espaço rígido 6D do DockingPP")
        if context.pop_size <= 0:
            return []

        domain_low = np.asarray(context.domain_low, dtype=float)
        domain_high = np.asarray(context.domain_high, dtype=float)
        if domain_low.shape != (context.dim,) or domain_high.shape != (context.dim,):
            raise ValueError("Os limites do PSO devem possuir uma entrada por dimensão")
        if not np.all(np.isfinite(domain_low)) or not np.all(np.isfinite(domain_high)):
            raise ValueError("Os limites do PSO devem ser finitos")
        if np.any(domain_high < domain_low):
            raise ValueError("Cada limite superior do PSO deve ser >= ao inferior")

        inertia = _config_float(
            context.cfg,
            ("pso_inertia_weight",),
            1.0,
            minimum=0.0,
        )
        cognitive = _config_float(
            context.cfg,
            ("pso_cognitive_coefficient",),
            2.0,
            minimum=0.0,
        )
        social = _config_float(
            context.cfg,
            ("pso_social_coefficient",),
            2.0,
            minimum=0.0,
        )
        velocity_fraction = _config_float(
            context.cfg,
            ("pso_velocity_clamp_fraction",),
            0.2,
            minimum=0.0,
            maximum=1.0,
        )

        velocity_limit = (domain_high - domain_low) * velocity_fraction
        raw_scores = np.asarray(context.raw_previous_scores, dtype=float).reshape(-1)
        particles: list[_ParticleState] = []

        # As poses podem chegar reordenadas pelo controlador populacional. O
        # estado acompanha cada Pose, não o índice ocupado na geração anterior.
        for index, pose in enumerate(context.previous[: context.pop_size]):
            position = np.clip(
                np.asarray(context.existing_solution(pose), dtype=float),
                domain_low,
                domain_high,
            )
            saved = (
                pose.meta.get(_PSO_STATE_KEY, {})
                if isinstance(pose.meta, dict)
                else {}
            )
            if not isinstance(saved, dict):
                saved = {}

            velocity = _state_vector(saved.get("velocity"), context.dim)
            if velocity is None:
                velocity = context.rng.uniform(-velocity_limit, velocity_limit)
            velocity = np.clip(velocity, -velocity_limit, velocity_limit)

            personal_best_position = _state_vector(
                saved.get("personal_best_position"), context.dim
            )
            personal_best_score = _finite_score(saved.get("personal_best_score"))
            current_score = _finite_score(
                raw_scores[index] if index < raw_scores.size else None
            )
            if personal_best_position is None:
                personal_best_position = position.copy()
                personal_best_score = current_score
            elif current_score > personal_best_score:
                personal_best_position = position.copy()
                personal_best_score = current_score
            personal_best_position = np.clip(
                personal_best_position, domain_low, domain_high
            )

            particles.append(
                _ParticleState(
                    position=position,
                    velocity=velocity,
                    personal_best_position=personal_best_position,
                    personal_best_score=personal_best_score,
                    seed_origin=seed_origin(pose),
                )
            )

        # O caminho normal recebe exatamente pop_size poses. O preenchimento
        # defensivo mantém o contrato mesmo após um consumidor externo filtrar
        # indevidamente a população.
        while len(particles) < context.pop_size:
            position = np.clip(
                np.asarray(context.random_solution(), dtype=float),
                domain_low,
                domain_high,
            )
            particles.append(
                _ParticleState(
                    position=position,
                    velocity=context.rng.uniform(-velocity_limit, velocity_limit),
                    personal_best_position=position.copy(),
                    personal_best_score=float("-inf"),
                )
            )

        global_best_particle = max(
            particles,
            key=lambda particle: particle.personal_best_score,
        )
        global_best_position = global_best_particle.personal_best_position.copy()
        global_best_score = float(global_best_particle.personal_best_score)
        if global_best_particle.seed_origin is not None:
            context.hybrid_operator_state["pso_global_best_from_seed_particle"] = (
                global_best_particle.seed_origin != SeedOrigin.UNRELATED
            )

        next_poses: list[Pose] = []
        for particle in particles:
            random_cognitive = context.rng.random(context.dim)
            random_social = context.rng.random(context.dim)
            velocity = (
                inertia * particle.velocity
                + cognitive
                * random_cognitive
                * (particle.personal_best_position - particle.position)
                + social
                * random_social
                * (global_best_position - particle.position)
            )
            velocity = np.clip(velocity, -velocity_limit, velocity_limit)
            position = np.clip(
                particle.position + velocity,
                domain_low,
                domain_high,
            )

            pose = context.solution_to_pose(position, None)
            if not isinstance(pose.meta, dict):
                pose.meta = {}
            mark_seed_origin(pose, inherit_seed(position, particle.seed_origin))
            pose.meta[_PSO_STATE_KEY] = {
                "velocity": [float(value) for value in velocity],
                "personal_best_position": [
                    float(value) for value in particle.personal_best_position
                ],
                "personal_best_score": float(particle.personal_best_score),
                "global_best_position": [
                    float(value) for value in global_best_position
                ],
                "global_best_score": global_best_score,
                "generation": int(context.generation_index),
            }
            pose.meta["baseline_algorithm"] = "pso_kennedy_eberhart"
            next_poses.append(pose)

        return next_poses
