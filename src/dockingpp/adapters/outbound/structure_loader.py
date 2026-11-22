"""Adaptador de carregamento que envolve os leitores existentes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class FileStructureLoader:
    """Implementa a porta usando callables de leitura de arquivos."""

    receptor_loader: Callable[[str], Any]
    peptide_loader: Callable[[str], Any]

    def load_receptor(self, source: str) -> Any:
        return self.receptor_loader(source)

    def load_peptide(self, source: str) -> Any:
        return self.peptide_loader(source)


__all__ = ["FileStructureLoader"]
