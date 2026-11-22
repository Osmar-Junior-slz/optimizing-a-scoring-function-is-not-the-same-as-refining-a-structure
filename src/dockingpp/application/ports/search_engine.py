"""Porta de saida para motores de busca de docking."""

from __future__ import annotations

from typing import Any, Protocol

from dockingpp.domain.models import RunResult
from dockingpp.application.ports.scoring import CheapObjective, FinalRescorer


class SearchEngine(Protocol):
    """Contrato implementado por hibrido, AG e PSO."""

    def search(
        self,
        receptor: Any,
        peptide: Any,
        pockets: list[Any],
        cfg: Any,
        score_cheap_fn: CheapObjective,
        score_expensive_fn: FinalRescorer,
        prior_pocket: Any,
        prior_pose: Any,
        logger: Any,
    ) -> RunResult:
        """Executa a busca e devolve um resultado de dominio."""

        ...


__all__ = ["SearchEngine"]
