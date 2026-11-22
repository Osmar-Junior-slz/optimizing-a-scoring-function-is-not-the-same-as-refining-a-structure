"""Blocos modulares usados pela execução da busca do pipeline.

O módulo histórico `execucao_busca` continua sendo a fachada pública. Este
pacote apenas separa responsabilidades internas para facilitar manutenção e
testes sem alterar o fluxo de execução.
"""

from .contextos import ContextoExecucaoBusca, ContextoExecucaoUnica
from .execucao_unica import ExecutorExecucaoUnicaPipeline
from .executor_busca import ExecutorBuscaPipeline
from .rescoring_artefatos import _persist_rescored_candidates, _select_rescoring_candidates

__all__ = [
    "ContextoExecucaoBusca",
    "ContextoExecucaoUnica",
    "ExecutorBuscaPipeline",
    "ExecutorExecucaoUnicaPipeline",
    "_persist_rescored_candidates",
    "_select_rescoring_candidates",
]
