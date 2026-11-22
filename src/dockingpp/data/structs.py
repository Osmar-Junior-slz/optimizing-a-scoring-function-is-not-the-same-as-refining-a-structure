"""Fachada legada para as entidades do dominio.

Novos modulos devem importar de :mod:`dockingpp.domain.models`. Este caminho e
mantido para preservar configuracoes, scripts e integracoes existentes.
"""

from dockingpp.domain.models import Pocket, Population, Pose, RunResult

__all__ = ["Pocket", "Pose", "Population", "RunResult"]
