"""Objetos imutáveis que transportam dados entre as etapas da busca.

Os aliases legados documentados nos campos continuam presentes porque fazem
parte do contrato entre seleção de bolsões, orquestração e agregação.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ContextoExecucaoBusca:
    """Dados necessarios para executar a busca apos a selecao."""

    search_space_mode: str
    modo_legado_pockets: bool
    candidate_pockets: list[Any]
    selected_pockets: list[Any]
    # Compatibilidade: `pockets` permanece como alias de `selected_pockets`.
    pockets: list[Any]
    total_pockets: int
    scan_results: dict[str, dict[str, Any]]
    scan_params: dict[str, Any]
    pocketing_time: float
    scan_time: float
    accepted_pockets: list[tuple[int, Any]]
    # Compatibilidade: `feasible_pockets` permanece como alias dos aceitos.
    feasible_pockets: list[tuple[int, Any]]
    rejected: list[dict[str, Any]]
    selection_confidence_metadata: dict[str, Any] = field(default_factory=dict)
    selection_confidence_policy: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContextoExecucaoUnica:
    """Dados auxiliares para persistencia e auditoria de uma execucao unica."""

    run_id: str
    receptor_path: str
    peptide_path: str
    search_space_mode: str
    total_pockets: int
    selected_pockets: int
    pocketing_time: float
    scan_time: float
    scan_params: dict[str, Any]
    scan_results: dict[str, dict[str, Any]]
    selected_pocket_ids: list[str]
    confidence_policy_metadata: dict[str, Any] = field(default_factory=dict)
    tracer: Any | None = None
    pocket_id: str | None = None
    include_global_overhead: bool = True
