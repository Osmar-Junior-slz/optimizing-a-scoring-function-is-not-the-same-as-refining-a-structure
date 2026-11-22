"""Objetos de contexto compartilhados pelos operadores de busca.

O contexto torna explícitas as dependências que antes eram capturadas por uma
função aninhada de mais de trezentas linhas. Ele não executa operações nem
consome números aleatórios; apenas transporta o estado da geração.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

import numpy as np

from dockingpp.domain.models import Pose


class SeedOrigin(IntEnum):
    """A path from the supplied pose through the realized search operators."""

    UNRELATED = 0
    DESCENDANT = 1
    UNCHANGED = 2


def seed_origin(pose: Pose) -> SeedOrigin | None:
    if not isinstance(pose.meta, dict) or "seed_lineage" not in pose.meta:
        return None
    if pose.meta["seed_unchanged"]:
        return SeedOrigin.UNCHANGED
    return SeedOrigin.DESCENDANT if pose.meta["seed_lineage"] else SeedOrigin.UNRELATED


def inherit_seed(solution: np.ndarray, *parents: SeedOrigin | None) -> SeedOrigin | None:
    """Retain exact survival only along an unbroken path of identity transforms."""
    if all(parent is None for parent in parents):
        return None
    if SeedOrigin.UNCHANGED in parents and not np.any(solution):
        return SeedOrigin.UNCHANGED
    if any(parent in (SeedOrigin.UNCHANGED, SeedOrigin.DESCENDANT) for parent in parents):
        return SeedOrigin.DESCENDANT
    return SeedOrigin.UNRELATED


def mark_seed_origin(pose: Pose, origin: SeedOrigin | None) -> None:
    if origin is not None:
        pose.meta["seed_lineage"] = origin != SeedOrigin.UNRELATED
        pose.meta["seed_unchanged"] = origin == SeedOrigin.UNCHANGED


@dataclass(slots=True)
class GenerationContext:
    """Estado necessário para um algoritmo produzir a próxima população."""

    cfg: Any
    rng: np.random.Generator
    previous: list[Pose]
    raw_previous_scores: np.ndarray
    generation_index: int
    selected: list[Pose]
    pop_size: int
    topk: int
    dim: int
    max_trans: float
    max_rot_deg: float
    domain_low: np.ndarray
    domain_high: np.ndarray
    random_solution: Callable[[], np.ndarray]
    random_pose: Callable[[], Pose]
    existing_solution: Callable[[Pose], np.ndarray]
    evaluate_solution: Callable[[np.ndarray], tuple[Pose, float, float]]
    solution_to_pose: Callable[[np.ndarray, float | None], Pose]
    hybrid_operator_state: dict[str, Any]
    hybrid_operator_history: list[dict[str, Any]]
