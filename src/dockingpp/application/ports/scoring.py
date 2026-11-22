"""Portas de avaliacao consumidas pelos algoritmos e casos de uso."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

from dockingpp.domain.models import Pocket, Pose


class CheapObjective(Protocol):
    """Funcao objetivo usada durante a exploracao populacional."""

    def __call__(
        self,
        pose: Pose,
        pocket: Pocket,
        weights: Mapping[str, float] | None = None,
    ) -> float:
        ...


class FinalRescorer(Protocol):
    """Rescoring posterior; a assinatura permanece flexivel por compatibilidade."""

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        ...


__all__ = ["CheapObjective", "FinalRescorer"]
