"""Acesso tardio às dependências públicas da página de relatórios.

Testes e integrações históricas substituem funções diretamente em
``dockingpp.gui.pages.reports``. O acesso tardio mantém esse contrato enquanto
as implementações ficam distribuídas em mixins menores.
"""

from __future__ import annotations

from types import ModuleType


def legacy_reports_module() -> ModuleType:
    """Retorna o módulo-fachada já inicializado, sem criar ciclo no import."""

    from dockingpp.gui.pages import reports

    return reports
