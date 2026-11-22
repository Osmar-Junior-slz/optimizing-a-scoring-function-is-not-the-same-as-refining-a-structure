"""Fases internas da seleção de bolsões do pipeline."""

from .aceitacao import ResultadoAceitacao, avaliar_aceitacao
from .confianca import ResultadoConfianca, aplicar_politica_confianca
from .preparacao import ResultadoPreparacao, preparar_ranking
from .reducao import aplicar_reducao_condicionada
from .scan import ConfiguracaoScan, ResultadoScan, executar_scan, preparar_configuracao_scan
from .telemetria import registrar_telemetria_selecao

__all__ = [
    "ConfiguracaoScan",
    "ResultadoAceitacao",
    "ResultadoConfianca",
    "ResultadoPreparacao",
    "ResultadoScan",
    "aplicar_politica_confianca",
    "aplicar_reducao_condicionada",
    "avaliar_aceitacao",
    "executar_scan",
    "preparar_configuracao_scan",
    "preparar_ranking",
    "registrar_telemetria_selecao",
]
