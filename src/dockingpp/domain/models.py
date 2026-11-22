"""Entidades centrais do dominio de docking.

As estruturas vivem no nucleo para que algoritmos e casos de uso nao dependam
do adaptador de leitura de dados. ``dockingpp.data.structs`` permanece como
fachada de compatibilidade para consumidores historicos.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Pocket:
    """Regiao candidata de ligacao no receptor."""

    id: str
    center: np.ndarray
    radius: float
    coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), dtype=float))
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pose:
    """Pose candidata de um peptideo."""

    coords: np.ndarray
    score_cheap: Optional[float] = None
    score_expensive: Optional[float] = None
    final_rescore: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Population:
    """Conjunto de poses associado a uma geracao."""

    poses: List[Pose]
    generation: int


@dataclass
class RunResult:
    """Resultado produzido por um caso de uso de docking."""

    best_pose: Pose
    population: Optional[Population]
    metrics: Dict[str, Any] = field(default_factory=dict)


__all__ = ["Pocket", "Pose", "Population", "RunResult"]
