"""Fachada pública da execução isolada de um mecanismo de busca."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from dockingpp.pipeline.logging import RunLogger
from dockingpp.search.engine import resolve_search_engine as default_search_engine_resolver

from ._execucao_unica_coordenador import CoordenadorExecucaoUnica
from ._rescoring_final import ServicoRescoringFinal
from .contextos import ContextoExecucaoUnica


class ExecutorExecucaoUnicaPipeline:
    """Fachada estável que compõe os serviços da execução única."""

    def __init__(
        self,
        search_engine_resolver: Callable[[Any], tuple[Any, str]] | None = None,
    ) -> None:
        """Recebe a fábrica como dependência, preservando o contrato histórico."""

        self._search_engine_resolver = (
            search_engine_resolver or default_search_engine_resolver
        )
        self._servico_rescoring = ServicoRescoringFinal()

    def _aplicar_rescoring_final(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        result: Any,
        logger: RunLogger,
    ) -> dict[str, Any]:
        """Mantém o ponto de extensão histórico para o rescoring final."""

        return self._servico_rescoring.aplicar(
            cfg=cfg,
            receptor=receptor,
            peptide=peptide,
            result=result,
            logger=logger,
        )

    def executar(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        pockets: list[Any],
        out_dir: str,
        contexto: ContextoExecucaoUnica,
    ) -> tuple[Any, dict[str, Any], RunLogger]:
        """Executa o fluxo privado mantendo retorno e efeitos observáveis."""

        coordenador = CoordenadorExecucaoUnica(
            search_engine_resolver=self._search_engine_resolver,
            servico_rescoring=self._servico_rescoring,
        )
        return coordenador.executar(
            cfg=cfg,
            receptor=receptor,
            peptide=peptide,
            pockets=pockets,
            out_dir=out_dir,
            contexto=contexto,
        )
