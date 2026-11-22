"""Portas para obter estruturas moleculares de fontes externas."""

from __future__ import annotations

from typing import Any, Protocol


class StructureLoader(Protocol):
    """Carrega receptor e peptideo sem expor formato ou filesystem."""

    def load_receptor(self, source: str) -> Any:
        ...

    def load_peptide(self, source: str) -> Any:
        ...


__all__ = ["StructureLoader"]
