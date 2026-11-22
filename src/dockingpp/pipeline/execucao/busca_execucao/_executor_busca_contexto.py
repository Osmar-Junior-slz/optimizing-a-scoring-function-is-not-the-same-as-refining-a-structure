"""Contratos privados compartilhados pelos fluxos do executor de busca."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from dockingpp.pipeline.confianca.policy import merge_confidence_policy_metadata

from .contextos import ContextoExecucaoBusca


@dataclass(frozen=True)
class DependenciasExecutorBusca:
    """Dependências injetadas pela fachada pública a cada execução."""

    executar_execucao_unica: Callable[..., Any]
    escrever_debug_summary: Callable[..., None]
    alocar_orcamento: Callable[[int, int, int], list[tuple[int, int]]]
    construir_bolsao_global: Callable[..., Any]
    classe_config: Any
    pocket_ids: Callable[[list[Any]], list[str]]
    accepted_pocket_ids: Callable[[list[tuple[int, Any]]], list[str]]
    payload_engine_name: Callable[[dict[str, Any] | None], str | None]
    build_reduced_payload_semantics: Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class EntradaExecucaoBusca:
    """Visão normalizada e imutável dos dados de uma execução."""

    cfg: Any
    receptor: Any
    peptide: Any
    receptor_path: str
    peptide_path: str
    out_dir: str
    run_id: str
    tracer: Any
    contexto: ContextoExecucaoBusca
    search_space_mode: str
    modo_legado_pockets: bool
    candidate_pockets: list[Any]
    selected_pockets: list[Any]
    total_pockets: int
    scan_results: dict[str, Any]
    scan_params: dict[str, Any]
    pocketing_time: float
    scan_time: float
    accepted_pockets: list[tuple[int, Any]]
    feasible_pockets: list[tuple[int, Any]]
    rejected: list[dict[str, Any]]
    candidate_pocket_ids: list[str]
    selected_after_scan_ids: list[str]
    accepted_pocket_ids: list[str]
    feasible_pocket_ids: list[str]
    selection_confidence_metadata: dict[str, Any]
    selection_confidence_policy: dict[str, Any]
    confidence_assisted_fallback_requested: bool

    @classmethod
    def criar(
        cls,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        receptor_path: str,
        peptide_path: str,
        out_dir: str,
        run_id: str,
        tracer: Any,
        contexto: ContextoExecucaoBusca,
        dependencias: DependenciasExecutorBusca,
    ) -> "EntradaExecucaoBusca":
        candidate_pockets = contexto.candidate_pockets or []
        selected_pockets = contexto.selected_pockets or contexto.pockets
        accepted_pockets = contexto.accepted_pockets or contexto.feasible_pockets
        feasible_pockets = contexto.feasible_pockets or accepted_pockets
        selection_confidence_policy = merge_confidence_policy_metadata(
            contexto.selection_confidence_policy
        )
        return cls(
            cfg=cfg,
            receptor=receptor,
            peptide=peptide,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            out_dir=out_dir,
            run_id=run_id,
            tracer=tracer,
            contexto=contexto,
            search_space_mode=contexto.search_space_mode,
            modo_legado_pockets=contexto.modo_legado_pockets,
            candidate_pockets=candidate_pockets,
            selected_pockets=selected_pockets,
            total_pockets=contexto.total_pockets,
            scan_results=contexto.scan_results,
            scan_params=contexto.scan_params,
            pocketing_time=contexto.pocketing_time,
            scan_time=contexto.scan_time,
            accepted_pockets=accepted_pockets,
            feasible_pockets=feasible_pockets,
            rejected=contexto.rejected,
            candidate_pocket_ids=dependencias.pocket_ids(candidate_pockets),
            selected_after_scan_ids=dependencias.pocket_ids(selected_pockets),
            accepted_pocket_ids=dependencias.accepted_pocket_ids(accepted_pockets),
            feasible_pocket_ids=dependencias.accepted_pocket_ids(feasible_pockets),
            selection_confidence_metadata=contexto.selection_confidence_metadata or {},
            selection_confidence_policy=selection_confidence_policy,
            confidence_assisted_fallback_requested=bool(
                selection_confidence_policy.get(
                    "confidence_assisted_fallback",
                    False,
                )
            ),
        )


@dataclass
class EstadoExecucaoReduzida:
    """Estado final do loop por bolsão, consumido apenas pela agregação."""

    selection_stage_policy_metadata: dict[str, Any]
    budget_policy: str
    per_pocket_results: list[dict[str, Any]]
    search_time_total_sec: float
    total_n_eval: int
    total_eval_budget_requested: int
    total_eval_budget_assigned: int
    budget_rounding_applied: bool
    best_result: Any
    best_cheap_value: float | None
    best_cheap_pocket_id: str | None
    best_selection_source: str
    best_expensive: Any
    best_pocket_id: str | None
    best_pose_output_path: str | None
    best_candidate_pose_output_path: str | None
    best_candidate_pose_count: int
    best_final_pose_available: bool
    best_pose_role: str | None
    best_pose_meta: Any
    engine_name_reduced: str | None
    search_strategy_reduced: str | None
    search_algorithm_family_reduced: str | None


@dataclass
class DadosAgregacaoReduzida:
    """Valores derivados uma única vez antes de montar os artefatos agregados."""

    budget_delta: int
    budget_rounding_applied: bool
    total_runtime_sec: float
    executed_pocket_ids: list[str]
    parent_final_pose_available: bool
    population_confidence_by_pocket: list[dict[str, Any]]
    population_confidence_global: dict[str, Any]
    reduced_confidence_metadata: dict[str, Any]
    reduced_policy_metadata: dict[str, Any]
    rescoring_applied_any: bool
    ranking_changed_any: bool
    rescoring_selected_candidates: list[dict[str, Any]]
    rescoring_cheap_top5_candidates: list[dict[str, Any]]
    rescoring_candidates_total: int
    rescoring_selected_count: int
    rescored_candidates_count: int
    rescoring_enabled_any: bool
    rescoring_not_applied_reasons: list[str]
    rescoring_not_applied_reason: str | None
    rescoring_top_k: int
    rescoring_scope: str
    rescoring_alignment_mode: str
    rescoring_selection_criterion: str


@dataclass
class ResultadoFallback:
    """Dados extraídos da execução full usada como fallback."""

    result: Any
    full_payload: dict[str, Any]
    full_logger: Any
    fallback_dir: str
    fallback_reason: str
    fallback_policy_metadata: dict[str, Any]
    engine_name: str | None
    best_score_cheap: Any
    best_score_expensive: Any
    rescoring_applied: bool
    rescoring_selected: list[dict[str, Any]]
    rescoring_cheap_top5: list[dict[str, Any]]
    rescored_candidates_count: int
    rescoring_candidates_total: int
    rescoring_selected_count: int
    rescoring_enabled: bool
    rescoring_top_k: int
    rescoring_scope: str
    rescoring_alignment_mode: str
    rescoring_selection_criterion: str
    rescoring_not_applied_reason: Any
    population_confidence_by_pocket: list[dict[str, Any]]
    population_confidence_global: dict[str, Any]
    pose_output_path: str | None
    final_pose_available: bool
    pose_role: Any
    best_pose_meta: Any
    accepted_ids: list[str]
    feasible_ids: list[str]
    requested_budget: int
    assigned_budget: int


class ComponenteExecutorBusca:
    """Base privada que conserva os nomes históricos das dependências."""

    def __init__(self, dependencias: DependenciasExecutorBusca) -> None:
        self._executar_execucao_unica = dependencias.executar_execucao_unica
        self._escrever_debug_summary = dependencias.escrever_debug_summary
        self._alocar_orcamento = dependencias.alocar_orcamento
        self._construir_bolsao_global = dependencias.construir_bolsao_global
        self._classe_config = dependencias.classe_config
        self._payload_engine_name = dependencias.payload_engine_name
        self._build_reduced_payload_semantics = (
            dependencias.build_reduced_payload_semantics
        )
