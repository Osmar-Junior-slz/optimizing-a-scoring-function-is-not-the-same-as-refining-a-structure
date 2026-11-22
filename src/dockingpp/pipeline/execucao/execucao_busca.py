"""Fachada compatível da execução da busca do pipeline.

A implementação foi dividida no pacote :mod:`busca_execucao` para separar
contextos, confiança, rescoring, artefatos e orquestração. Os nomes públicos
históricos permanecem disponíveis neste módulo, evitando alterações nos
consumidores existentes.
"""

from __future__ import annotations

from dockingpp.search.engine import resolve_search_engine

from .busca_execucao import (
    ContextoExecucaoBusca,
    ContextoExecucaoUnica,
    ExecutorBuscaPipeline,
    ExecutorExecucaoUnicaPipeline,
    _persist_rescored_candidates,
    _select_rescoring_candidates,
)

__all__ = [
    "ContextoExecucaoBusca",
    "ContextoExecucaoUnica",
    "ExecutorBuscaPipeline",
    "ExecutorExecucaoUnicaPipeline",
    "_persist_rescored_candidates",
    "_select_rescoring_candidates",
    "resolve_search_engine",
]
