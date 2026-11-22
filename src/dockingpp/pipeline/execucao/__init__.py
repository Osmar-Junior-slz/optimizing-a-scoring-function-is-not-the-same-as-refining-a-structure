"""Módulos de execução e fachada histórica do orquestrador."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .orquestrador import OrquestradorPipelineDocking

__all__ = ["OrquestradorPipelineDocking"]


def __getattr__(name: str) -> Any:
    """Carrega a fachada antiga somente quando ela for solicitada."""

    if name == "OrquestradorPipelineDocking":
        from .orquestrador import OrquestradorPipelineDocking

        return OrquestradorPipelineDocking
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
