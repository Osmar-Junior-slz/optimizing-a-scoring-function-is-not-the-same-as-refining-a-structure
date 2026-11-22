"""Orquestra a execução full ou reduzida após a seleção de bolsões.

O método `executar` conserva a sequência original de decisões, inclusive
fallbacks, divisão de orçamento, agregação e escrita dos artefatos. As rotinas
de baixo nível foram movidas para módulos dedicados para deixar este fluxo
científico mais fácil de acompanhar.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from dockingpp.data.io import molecular_partner_size
from dockingpp.evaluation.pipeline_structural import copy_structural_metrics_fields
from dockingpp.pipeline.confianca.features import (
    build_reduced_aggregate_confidence_metadata,
)
from dockingpp.pipeline.confianca.policy import (
    evaluate_confidence_policy,
    merge_confidence_policy_metadata,
)

from .artifact_io import atomic_write_json, atomic_write_text, sha256_file
from .contextos import ContextoExecucaoBusca
from .rescoring_artefatos import (
    _reproducibility_fields,
    _rescoring_alignment_mode,
    _scientific_state_fields,
    _score_semantics_fields,
    _write_pocket_trace_artifacts,
)
from .serializacao_confianca import (
    _build_population_confidence_global_from_pocket_summaries,
    _coerce_string_list,
    _collect_population_confidence_by_pocket,
    _combine_confidence_policy_metadata,
    _normalize_population_confidence_for_pocket,
    _population_confidence_payload_fields,
    _to_json_object,
    _to_json_object_list,
)



def _fail_reduced(*, cfg, out_dir, run_id, status, reason, evidence=None):
    """Persist a failed REDUCED arm; candidates remain diagnostic artifacts."""
    evidence = evidence or {}
    payload = {
        **evidence,
        "schema_version": "2.0",
        "mode": "reduced",
        "search_space_mode": "reduced",
        "requested_protocol": "reduced",
        "executed_mode": "reduced",
        "run_id": run_id,
        "status": status,
        "failure_reason": reason,
        "next_action": "RUN_FULL_REQUIRED",
        "search_strategy": cfg.search_engine,
        "seed": cfg.seed,
        "generations": cfg.generations,
        "pop_size": cfg.pop_size,
        "fallback_to_full": False,
        "fallback_executed": False,
        "final_pose_available": False,
        "pose_output_path": None,
    }
    for name in ("summary.json", "result.json"):
        atomic_write_json(os.path.join(out_dir, name), payload)
    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"name": "run.status", "status": status,
                                 "reason": reason}) + "\n")
    raise RuntimeError(f"{status}: {reason}; RUN_FULL_REQUIRED")


class ExecutorBuscaPipeline:
    """Executa caminhos full/reduced preservando comportamento existente."""

    def __init__(
        self,
        *,
        executar_execucao_unica: Callable[..., Any],
        escrever_debug_summary: Callable[..., None],
        alocar_orcamento: Callable[[int, int, int], list[tuple[int, int]]],
        construir_bolsao_global: Callable[..., Any],
        classe_config: Any,
    ) -> None:
        self._executar_execucao_unica = executar_execucao_unica
        self._escrever_debug_summary = escrever_debug_summary
        self._alocar_orcamento = alocar_orcamento
        self._construir_bolsao_global = construir_bolsao_global
        self._classe_config = classe_config

    @staticmethod
    def _pocket_ids(pockets: list[Any]) -> list[str]:
        """Extrai ids de bolsoes preservando serializacao estavel."""

        return [str(p.id) for p in pockets]

    @staticmethod
    def _accepted_pocket_ids(pockets: list[tuple[int, Any]]) -> list[str]:
        """Extrai ids de bolsoes aceitos (tuplas indice + bolsao)."""

        return [str(p.id) for _, p in pockets]

    @staticmethod
    def _configured_eval_budget(cfg: Any) -> int:
        """Return the real configured objective budget for one search run."""

        explicit = getattr(cfg, "objective_evaluation_budget_per_pocket", None)
        if explicit is not None:
            return max(1, int(explicit))
        generations = max(1, int(getattr(cfg, "generations", 1)))
        pop_size = max(1, int(getattr(cfg, "pop_size", 1)))
        return (generations + 1) * pop_size

    @staticmethod
    def _payload_engine_name(payload: dict[str, Any] | None) -> str | None:
        """Extrai o nome da engine de payloads de execucao unica."""

        if not isinstance(payload, dict):
            return None
        cfg_subset = payload.get("config_resolved_subset")
        if isinstance(cfg_subset, dict) and cfg_subset.get("search_engine_name") is not None:
            return str(cfg_subset.get("search_engine_name"))
        engine_name = payload.get("search_engine_name")
        if engine_name is None:
            return None
        return str(engine_name)

    @staticmethod
    def _final_hard_clash_triggered(pose_meta: Any) -> bool:
        """Return the audited final-pose clash decision, if one was recorded."""

        if not isinstance(pose_meta, dict):
            return False
        gate = pose_meta.get("final_hard_clash_gate")
        return bool(isinstance(gate, dict) and gate.get("triggered") is True)

    @staticmethod
    def _molecular_partner_size(partner: Any) -> tuple[int | None, int | None]:
        """Return atom/residue counts retained by the PDB coordinate loader."""

        return molecular_partner_size(partner)

    @staticmethod
    def _cross_pocket_selection_policy(cfg: Any) -> str:
        """Resolve the explicit policy used to choose among executed pockets."""

        requested = str(
            getattr(cfg, "cross_pocket_selection_policy", "scan_priority")
            or "scan_priority"
        ).strip().lower()
        if requested in {"final_rescore", "legacy_final_rescore"}:
            return "final_rescore"
        return "scan_priority"

    def _build_reduced_payload_semantics(
        self,
        *,
        selected_pre_filter_pockets: list[str],
        selected_after_scan_pockets: list[str],
        accepted_pockets: list[str],
        feasible_pockets: list[str],
        executed_pockets: list[str],
    ) -> dict[str, Any]:
        """Metadados explicitos para evitar ambiguidade semantica no payload."""

        return {
            "pocket_payload_semantics_version": "1.0",
            "selected_pockets_semantic_role": "compatibility_alias_of_executed_pockets",
            "scan_by_pocket_semantic_role": "contains_scanned_pockets_not_necessarily_executed",
            "pocket_payload_semantics": {
                "candidate_pockets": "after_ranking_and_topk_before_scan_and_final_acceptance",
                "selected_pre_filter_pockets": "legacy_compat_field_currently_aliases_selected_after_scan_pockets",
                "selected_after_scan_pockets": "after_scan_and_optional_conditional_reduction_before_final_acceptance",
                "accepted_pockets": "after_final_acceptance_filter",
                "feasible_pockets": "compatibility_alias_of_accepted_pockets",
                "executed_pockets": "pockets_effectively_executed_in_reduced_aggregate",
                "selected_pockets": "compatibility_alias_of_executed_pockets_in_reduced_aggregate_payload",
                "rejected_pockets": "pockets_rejected_by_final_acceptance_filter_with_reason",
                "scan_by_pocket": "scan_metrics_for_scanned_pockets_not_necessarily_executed",
                "selected_pockets_alias_target": "executed_pockets",
                "feasible_pockets_alias_target": "accepted_pockets",
                "selected_pre_filter_is_redundant_with_selected_after_scan": (
                    selected_pre_filter_pockets == selected_after_scan_pockets
                ),
                "accepted_matches_feasible_alias": accepted_pockets == feasible_pockets,
                "executed_matches_selected_alias": True,
            },
        }

    def _execute_post_reduced_fallback(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        receptor_path: str,
        peptide_path: str,
        out_dir: str,
        run_id: str,
        tracer: Any,
        parent_summary: dict[str, Any],
        fallback_reason_override: str | None = None,
        fallback_trigger_source: str = "confidence_policy",
    ) -> Any:
        """A separate FULL job is required after a REDUCED failure."""
        _fail_reduced(
            cfg=cfg, out_dir=out_dir, run_id=run_id,
            status="RUN_FULL_REQUIRED",
            reason=fallback_reason_override or "confidence_policy_requested_full",
            evidence={"reduced_evidence": parent_summary},
        )

    def executar(
        self,
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
    ) -> Any:
        """Executa a busca no modo correspondente ao contexto recebido."""

        search_space_mode = contexto.search_space_mode
        modo_legado_pockets = contexto.modo_legado_pockets
        candidate_pockets = contexto.candidate_pockets or []
        selected_pockets = contexto.selected_pockets or contexto.pockets
        total_pockets = contexto.total_pockets
        scan_results = contexto.scan_results
        scan_params = contexto.scan_params
        pocketing_time = contexto.pocketing_time
        scan_time = contexto.scan_time
        accepted_pockets = contexto.accepted_pockets or contexto.feasible_pockets
        feasible_pockets = contexto.feasible_pockets or accepted_pockets
        rejected = contexto.rejected
        candidate_pocket_ids = self._pocket_ids(candidate_pockets)
        selected_after_scan_ids = self._pocket_ids(selected_pockets)
        accepted_pocket_ids = self._accepted_pocket_ids(accepted_pockets)
        feasible_pocket_ids = self._accepted_pocket_ids(feasible_pockets)
        selection_confidence_metadata = contexto.selection_confidence_metadata or {}
        receptor_atom_count, receptor_residue_count = self._molecular_partner_size(
            receptor
        )
        peptide_atom_count, peptide_residue_count = self._molecular_partner_size(
            peptide
        )
        selection_confidence_policy = merge_confidence_policy_metadata(
            contexto.selection_confidence_policy
        )
        if search_space_mode == "full":
            result, payload_single, logger_single = self._executar_execucao_unica(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=selected_pockets,
                out_dir=out_dir,
                run_id=run_id,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="full",
                total_pockets=total_pockets,
                selected_pockets=len(selected_pockets),
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results=scan_results,
                selected_pocket_ids=[str(p.id) for p in selected_pockets],
                tracer=tracer,
                pocket_id="global",
                confidence_policy_metadata=selection_confidence_policy,
            )
            engine_name_single = self._payload_engine_name(payload_single)
            best_score_cheap_single = payload_single.get("best_score_cheap")
            best_score_expensive_single = payload_single.get(
                "final_rescore",
                payload_single.get("best_score_expensive"),
            )
            rescoring_applied_single = bool(payload_single.get("rescoring_applied", False))
            best_pose_meta_single = payload_single.get("best_pose_meta")
            best_pocket_single = (
                best_pose_meta_single.get("pocket_id")
                if isinstance(best_pose_meta_single, dict)
                else None
            )
            pose_output_path_single = payload_single.get("pose_output_path")
            final_pose_available_single = bool(
                payload_single.get("final_pose_available", False)
            )
            pose_role_single = payload_single.get("pose_role")
            self._escrever_debug_summary(
                tracer,
                {
                    "success": True,
                    "requested_mode": "full",
                    "executed_mode": "full",
                    "search_space_mode": "full",
                    "budget_policy": str(getattr(cfg, "budget_policy", "split")),
                    "compare_policy": "best_pocket_vs_full",
                    "search_engine_name": engine_name_single,
                    "pose_output_path": pose_output_path_single,
                    "final_pose_available": final_pose_available_single,
                    "pose_role": pose_role_single,
                    "fallback_to_full": False,
                    "fallback_executed": False,
                    "fallback_recommended": bool(selection_confidence_policy.get("confidence_fallback_recommended", False)),
                    "fallback_reason": None,
                    "confidence_features_available": bool(
                        selection_confidence_metadata.get("confidence_features_available", False)
                    ),
                    "confidence_policy_mode": selection_confidence_policy.get("confidence_policy_mode"),
                    "confidence_policy_effect": selection_confidence_policy.get("confidence_policy_effect"),
                    "confidence_assisted_selection": bool(selection_confidence_policy.get("confidence_assisted_selection", False)),
                    "confidence_assisted_fallback": bool(selection_confidence_policy.get("confidence_assisted_fallback", False)),
                    "total_runtime_sec": float(payload_single.get("runtime_sec", 0.0)),
                    "total_n_eval": int(
                        sum(
                            float(r.get("value", 0.0))
                            for r in logger_single.records
                            if r.get("name") == "n_eval"
                        )
                    ),
                    "best_score_cheap": best_score_cheap_single,
                    "best_score_expensive": best_score_expensive_single,
                    "final_rescore": best_score_expensive_single,
                    "rescoring_applied": rescoring_applied_single,
                    "best_pocket_id": best_pocket_single,
                    "n_pockets_total": total_pockets,
                    "n_pockets_used": len(selected_pockets),
                    "total_eval_budget_requested": self._configured_eval_budget(cfg),
                    "total_eval_budget_assigned": self._configured_eval_budget(cfg),
                    "budget_delta": 0,
                    "budget_rounding_applied": False,
                    "warnings_count": tracer.warnings_count,
                    "errors_count": tracer.errors_count,
                    "selected_pockets": [str(p.id) for p in selected_pockets],
                    "rejected_pockets": [],
                },
            )
            return result

        if search_space_mode == "reduced" and modo_legado_pockets:
            result, payload_single, logger_single = self._executar_execucao_unica(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=selected_pockets,
                out_dir=out_dir,
                run_id=run_id,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="reduced",
                total_pockets=total_pockets,
                selected_pockets=len(selected_pockets),
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results=scan_results,
                selected_pocket_ids=[str(p.id) for p in selected_pockets],
                tracer=tracer,
                pocket_id="legacy_pockets",
                confidence_policy_metadata=selection_confidence_policy,
            )
            engine_name_single = self._payload_engine_name(payload_single)
            best_score_cheap_single = payload_single.get("best_score_cheap")
            best_score_expensive_single = payload_single.get(
                "final_rescore",
                payload_single.get("best_score_expensive"),
            )
            rescoring_applied_single = bool(payload_single.get("rescoring_applied", False))
            best_pose_meta_single = payload_single.get("best_pose_meta")
            best_pocket_single = (
                best_pose_meta_single.get("pocket_id")
                if isinstance(best_pose_meta_single, dict)
                else None
            )
            pose_output_path_single = payload_single.get("pose_output_path")
            final_pose_available_single = bool(
                payload_single.get("final_pose_available", False)
            )
            pose_role_single = payload_single.get("pose_role")
            self._escrever_debug_summary(
                tracer,
                {
                    "success": True,
                    "requested_mode": "pockets",
                    "executed_mode": "reduced",
                    "search_space_mode": "reduced",
                    "budget_policy": str(getattr(cfg, "budget_policy", "split")),
                    "compare_policy": "best_pocket_vs_full",
                    "search_engine_name": engine_name_single,
                    "pose_output_path": pose_output_path_single,
                    "final_pose_available": final_pose_available_single,
                    "pose_role": pose_role_single,
                    "fallback_to_full": False,
                    "fallback_executed": False,
                    "fallback_recommended": bool(selection_confidence_policy.get("confidence_fallback_recommended", False)),
                    "fallback_reason": None,
                    "confidence_features_available": bool(
                        selection_confidence_metadata.get("confidence_features_available", False)
                    ),
                    "confidence_policy_mode": selection_confidence_policy.get("confidence_policy_mode"),
                    "confidence_policy_effect": selection_confidence_policy.get("confidence_policy_effect"),
                    "confidence_assisted_selection": bool(selection_confidence_policy.get("confidence_assisted_selection", False)),
                    "confidence_assisted_fallback": bool(selection_confidence_policy.get("confidence_assisted_fallback", False)),
                    "total_runtime_sec": float(payload_single.get("runtime_sec", 0.0)),
                    "total_n_eval": int(
                        sum(
                            float(r.get("value", 0.0))
                            for r in logger_single.records
                            if r.get("name") == "n_eval"
                        )
                    ),
                    "best_score_cheap": best_score_cheap_single,
                    "best_score_expensive": best_score_expensive_single,
                    "final_rescore": best_score_expensive_single,
                    "rescoring_applied": rescoring_applied_single,
                    "best_pocket_id": best_pocket_single,
                    "n_pockets_total": total_pockets,
                    "n_pockets_used": len(selected_pockets),
                    "total_eval_budget_requested": self._configured_eval_budget(cfg),
                    "total_eval_budget_assigned": self._configured_eval_budget(cfg),
                    "budget_delta": 0,
                    "budget_rounding_applied": False,
                    "warnings_count": tracer.warnings_count,
                    "errors_count": tracer.errors_count,
                    "selected_pockets": [str(p.id) for p in selected_pockets],
                    "rejected_pockets": [],
                },
            )
            return result

        # Confidence-assisted fallback is intentionally evaluated only after
        # the reduced search has produced auditable population evidence.
        confidence_triggered_fallback = False
        fallback_triggered = not feasible_pockets
        pre_fallback_population_confidence_by_pocket: list[dict[str, Any]] = []
        pre_fallback_population_confidence_global: dict[str, Any] = {}

        if fallback_triggered:
            _fail_reduced(
                cfg=cfg, out_dir=out_dir, run_id=run_id,
                status="NO_FEASIBLE_POCKET", reason="no_feasible_pocket",
                evidence={"candidate_pockets": candidate_pocket_ids,
                          "rejected_pockets": rejected,
                          "scan_results": scan_results,
                          "executed_pockets": [], "total_n_eval": 0},
            )

        selection_stage_policy_metadata = merge_confidence_policy_metadata(
            selection_confidence_policy
        )
        budget_policy = str(getattr(cfg, "budget_policy", "split") or "split").lower()
        if budget_policy not in {"split", "replicated"}:
            budget_policy = "split"
        configured_budget_per_search = self._configured_eval_budget(cfg)
        n_feasible = max(1, len(feasible_pockets))
        if budget_policy == "split":
            split_base = configured_budget_per_search // n_feasible
            split_remainder = configured_budget_per_search % n_feasible
            split_eval_targets = [
                max(1, split_base + (1 if idx < split_remainder else 0))
                for idx in range(len(feasible_pockets))
            ]
            budgets = []
            for target in split_eval_targets:
                pocket_pop_size = max(1, min(int(cfg.pop_size), int(target)))
                remaining_after_initial = max(0, int(target) - pocket_pop_size)
                pocket_generations = max(
                    1,
                    (remaining_after_initial + pocket_pop_size - 1)
                    // pocket_pop_size,
                )
                budgets.append((pocket_generations, pocket_pop_size))
            total_eval_budget_requested = configured_budget_per_search
        else:
            budgets = [
                (int(cfg.generations), int(cfg.pop_size))
                for _ in feasible_pockets
            ]
            split_eval_targets = [
                configured_budget_per_search for _ in feasible_pockets
            ]
            total_eval_budget_requested = (
                configured_budget_per_search * len(feasible_pockets)
            )
        tracer.event(
            stage="budget",
            event_type="budget_split",
            payload={
                "total_eval_budget_requested": total_eval_budget_requested,
                "configured_eval_budget_per_search": configured_budget_per_search,
                "budget_policy": budget_policy,
                "n_pockets": len(feasible_pockets),
                "allocations": [
                    {
                        "pocket_id": str(p.id),
                        "generations": int(b[0]),
                        "pop_size": int(b[1]),
                        "objective_eval_budget": int(target),
                    }
                    for (_, p), b, target in zip(
                        feasible_pockets,
                        budgets,
                        split_eval_targets,
                    )
                ],
            },
            level="TRACE",
            decision=True,
        )

        per_pocket_results = []
        search_time_total_sec = 0.0
        total_runtime_sec = 0.0
        total_n_eval = 0
        total_eval_budget_assigned = 0
        budget_rounding_applied = False
        best_result = None
        best_cheap = float("-inf")
        best_cheap_value: float | None = None
        max_cheap_over_pockets: float | None = None
        best_cheap_pocket_id: str | None = None
        best_selection_score = float("-inf")
        best_selection_source = "score_cheap"
        best_expensive = None
        max_expensive_over_pockets = None
        best_pocket_id = None
        best_pose_output_path: str | None = None
        best_candidate_pose_output_path: str | None = None
        best_candidate_pose_count = 0
        best_final_pose_available = False
        best_pose_role: str | None = None
        best_pose_meta: Any = None
        best_structural_fields: dict[str, Any] = {}

        fallback_result = None
        fallback_pocket_id: str | None = None
        fallback_cheap_value: float | None = None
        fallback_expensive_value: float | None = None
        fallback_pose_output_path: str | None = None
        fallback_candidate_pose_output_path: str | None = None
        fallback_candidate_pose_count = 0
        fallback_final_pose_available = False
        fallback_pose_role: str | None = None
        fallback_pose_meta: Any = None
        fallback_structural_fields: dict[str, Any] = {}

        engine_name_reduced: str | None = None
        search_strategy_reduced: str | None = None
        search_algorithm_family_reduced: str | None = None
        cross_pocket_selection_policy = self._cross_pocket_selection_policy(cfg)
        for local_idx, (original_idx, pocket) in enumerate(feasible_pockets):
            pocket_cfg = self._classe_config(**cfg.model_dump())
            pocket_cfg.confidence_calibrator = getattr(cfg, "confidence_calibrator", None)
            pocket_cfg.search_space_mode = "reduced"
            pocket_cfg.full_search = False
            if budget_policy == "split":
                pocket_cfg.generations, pocket_cfg.pop_size = budgets[local_idx]
                pocket_cfg.objective_evaluation_budget_per_pocket = split_eval_targets[local_idx]
                assigned_budget = split_eval_targets[local_idx]
            else:
                assigned_budget = self._configured_eval_budget(pocket_cfg)
            total_eval_budget_assigned += assigned_budget
            pocket_out_dir = os.path.join(out_dir, str(pocket.id))
            result, payload, logger = self._executar_execucao_unica(
                cfg=pocket_cfg,
                receptor=receptor,
                peptide=peptide,
                pockets=[pocket],
                out_dir=pocket_out_dir,
                run_id=f"{run_id}-pocket-{pocket.id}",
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                search_space_mode="reduced",
                total_pockets=total_pockets,
                selected_pockets=1,
                pocketing_time=pocketing_time,
                scan_time=scan_time,
                scan_params=scan_params,
                scan_results={str(pocket.id): scan_results.get(str(pocket.id), {})},
                selected_pocket_ids=[str(pocket.id)],
                tracer=tracer,
                pocket_id=str(pocket.id),
                include_global_overhead=False,
                confidence_policy_metadata=selection_stage_policy_metadata,
            )
            resolved_engine_name = self._payload_engine_name(payload)
            if engine_name_reduced is None and resolved_engine_name is not None:
                engine_name_reduced = resolved_engine_name
            cfg_subset_payload = payload.get("config_resolved_subset")
            if isinstance(cfg_subset_payload, dict):
                if search_strategy_reduced is None and cfg_subset_payload.get("search_strategy") is not None:
                    search_strategy_reduced = str(cfg_subset_payload.get("search_strategy"))
                if (
                    search_algorithm_family_reduced is None
                    and cfg_subset_payload.get("search_algorithm_family") is not None
                ):
                    search_algorithm_family_reduced = str(cfg_subset_payload.get("search_algorithm_family"))
            runtime = float(payload.get("runtime_sec", 0.0))
            search_time_sec = float(payload.get("timing", {}).get("search_s", runtime))
            n_eval = int(sum(float(r.get("value", 0.0)) for r in logger.records if r.get("name") == "n_eval"))
            search_time_total_sec += search_time_sec
            total_n_eval += n_eval

            score_cheap = payload.get("best_score_cheap")
            score_expensive = payload.get("final_rescore", payload.get("best_score_expensive"))
            pocket_rescoring_applied = bool(payload.get("rescoring_applied", False))
            pocket_population_confidence_by_pocket_raw = _to_json_object_list(
                payload.get("population_confidence_by_pocket")
            )
            pocket_population_confidence_global = (
                _to_json_object(payload.get("population_confidence_global")) or {}
            )
            (
                pocket_population_confidence_by_pocket,
                pocket_population_confidence_pocket,
            ) = _normalize_population_confidence_for_pocket(
                pocket_id=str(pocket.id),
                by_pocket=pocket_population_confidence_by_pocket_raw,
            )
            pocket_pose_output_path = payload.get("pose_output_path")
            if pocket_pose_output_path:
                pocket_pose_output_path = os.path.join(str(pocket.id), str(pocket_pose_output_path)).replace("\\", "/")
            pocket_candidate_pose_output_path = payload.get("candidate_pose_output_path")
            if pocket_candidate_pose_output_path:
                pocket_candidate_pose_output_path = os.path.join(
                    str(pocket.id), str(pocket_candidate_pose_output_path)
                ).replace("\\", "/")
            pocket_candidate_pose_count = int(payload.get("candidate_pose_count") or 0)
            pocket_final_pose_available = bool(payload.get("final_pose_available", False) and pocket_pose_output_path)
            pocket_pose_role = payload.get("pose_role")
            pocket_best_pose_meta = payload.get("best_pose_meta")
            pocket_structural_fields = copy_structural_metrics_fields(
                payload,
                path_prefix=str(pocket.id),
            )

            if fallback_result is None:
                fallback_result = result
                fallback_pocket_id = str(pocket.id)
                fallback_cheap_value = score_cheap
                fallback_expensive_value = score_expensive
                fallback_pose_output_path = pocket_pose_output_path
                fallback_candidate_pose_output_path = pocket_candidate_pose_output_path
                fallback_candidate_pose_count = pocket_candidate_pose_count
                fallback_final_pose_available = pocket_final_pose_available
                fallback_pose_role = pocket_pose_role
                fallback_pose_meta = pocket_best_pose_meta
                fallback_structural_fields = pocket_structural_fields

            if score_cheap is not None and score_cheap > best_cheap:
                best_cheap = float(score_cheap)
                max_cheap_over_pockets = float(score_cheap)
                best_cheap_pocket_id = str(pocket.id)

            final_hard_clash_triggered = self._final_hard_clash_triggered(
                pocket_best_pose_meta
            )
            selection_score: float | None = None
            selection_source = "pocket_scan_priority"
            selection_eligible = not final_hard_clash_triggered
            should_select = False
            if cross_pocket_selection_policy == "scan_priority":
                # feasible_pockets preserves the scan order. The first valid
                # final pose therefore remains the audited pocket winner.
                selection_score = float(-local_idx)
                should_select = bool(selection_eligible and best_result is None)
            else:
                selection_source = "score_cheap"
                if pocket_rescoring_applied and score_expensive is not None:
                    selection_score = float(score_expensive)
                    selection_source = "final_rescore"
                elif score_cheap is not None:
                    selection_score = float(score_cheap)
                should_select = bool(
                    selection_eligible
                    and selection_score is not None
                    and selection_score > best_selection_score
                )

            if should_select:
                best_selection_score = float(selection_score)
                best_selection_source = selection_source
                best_result = result
                best_pocket_id = str(pocket.id)
                best_cheap_value = (
                    float(score_cheap) if score_cheap is not None else None
                )
                best_pose_output_path = pocket_pose_output_path
                best_candidate_pose_output_path = pocket_candidate_pose_output_path
                best_candidate_pose_count = pocket_candidate_pose_count
                best_final_pose_available = pocket_final_pose_available
                best_pose_role = pocket_pose_role
                best_pose_meta = pocket_best_pose_meta
                best_structural_fields = pocket_structural_fields
                best_expensive = (
                    float(score_expensive) if score_expensive is not None else None
                )
            if score_expensive is not None and (
                max_expensive_over_pockets is None
                or score_expensive > max_expensive_over_pockets
            ):
                max_expensive_over_pockets = float(score_expensive)
            per_pocket_results.append(
                {
                    "pocket_index": int(original_idx),
                    "pocket_id": str(pocket.id),
                    "best_score_cheap": score_cheap,
                    "best_score_expensive": score_expensive,
                    "final_rescore": score_expensive,
                    "cross_pocket_selection_rank": int(local_idx + 1),
                    "cross_pocket_selection_eligible": bool(selection_eligible),
                    "final_hard_clash_triggered": bool(final_hard_clash_triggered),
                    **_score_semantics_fields(
                        triage_score=score_cheap,
                        rescore_proxy=score_expensive,
                        final_rescore=score_expensive,
                        rescoring_applied=pocket_rescoring_applied,
                    ),
                    "runtime_sec": runtime,
                    "search_time_sec": search_time_sec,
                    "n_eval_total": n_eval,
                    "alloc_generations": int(pocket_cfg.generations),
                    "alloc_pop_size": int(pocket_cfg.pop_size),
                    "alloc_eval_budget": assigned_budget,
                    "pose_output_path": pocket_pose_output_path,
                    "candidate_pose_output_path": pocket_candidate_pose_output_path,
                    "candidate_pose_count": pocket_candidate_pose_count,
                    "final_pose_available": pocket_final_pose_available,
                    "pose_role": pocket_pose_role,
                    "best_pose_meta": pocket_best_pose_meta,
                    **pocket_structural_fields,
                    "rescoring_applied": pocket_rescoring_applied,
                    "rescoring_enabled": bool(payload.get("rescoring_enabled", False)),
                    "rescoring_top_k": payload.get("rescoring_top_k"),
                    "rescoring_scope": payload.get("rescoring_scope"),
                    "rescoring_alignment_mode": payload.get("rescoring_alignment_mode"),
                    "rescoring_selection_criterion": payload.get("rescoring_selection_criterion"),
                    "rescoring_candidates_total": payload.get("rescoring_candidates_total"),
                    "rescoring_selected_count": payload.get("rescoring_selected_count"),
                    "rescored_candidates_count": payload.get("rescored_candidates_count"),
                    "ranking_changed_after_rescoring": payload.get("ranking_changed_after_rescoring"),
                    "rescoring_not_applied_reason": payload.get("rescoring_not_applied_reason"),
                    "rescoring_selected_candidates": payload.get("rescoring_selected_candidates", []),
                    "rescoring_cheap_top5_candidates": payload.get("rescoring_cheap_top5_candidates", []),
                    "final_score_variant": payload.get("final_score_variant"),
                    "population_confidence_by_pocket": pocket_population_confidence_by_pocket,
                    "population_confidence_global": pocket_population_confidence_global,
                    "population_confidence_pocket": pocket_population_confidence_pocket,
                }
            )

        if best_result is None:
            _fail_reduced(
                cfg=cfg, out_dir=out_dir, run_id=run_id,
                status="REDUCED_FAILED", reason="no_valid_reduced_pose",
                evidence={"per_pocket_results": per_pocket_results,
                          "executed_pockets": feasible_pocket_ids,
                          "search_engine_name": engine_name_reduced,
                          "total_n_eval": total_n_eval},
            )

        for item in per_pocket_results:
            item["selected_as_parent_winner"] = bool(
                str(item.get("pocket_id")) == str(best_pocket_id)
            )

        budget_delta = int(total_eval_budget_assigned - total_eval_budget_requested)
        budget_rounding_applied = budget_delta != 0
        total_runtime_sec = float(pocketing_time) + float(scan_time) + search_time_total_sec
        executed_pocket_ids = feasible_pocket_ids
        parent_final_pose_available = bool(best_final_pose_available and best_pose_output_path)
        population_confidence_by_pocket = _collect_population_confidence_by_pocket(per_pocket_results)
        population_confidence_global = _build_population_confidence_global_from_pocket_summaries(
            pocket_summaries=population_confidence_by_pocket,
            per_pocket_results=per_pocket_results,
        )
        reduced_confidence_metadata = build_reduced_aggregate_confidence_metadata(
            search_space_mode="reduced",
            executed_mode="reduced",
            fallback_to_full=False,
            best_pocket_id=best_pocket_id,
            best_over_pockets_cheap=best_cheap_value,
            best_over_pockets_expensive=best_expensive,
            candidate_pockets=candidate_pocket_ids,
            selected_after_scan_pockets=selected_after_scan_ids,
            accepted_pockets=accepted_pocket_ids,
            executed_pockets=executed_pocket_ids,
            per_pocket_results=per_pocket_results,
            scan_by_pocket=scan_results,
            population_confidence_by_pocket=population_confidence_by_pocket,
            population_confidence_global=population_confidence_global,
            peptide_atom_count=peptide_atom_count,
            peptide_residue_count=peptide_residue_count,
            receptor_atom_count=receptor_atom_count,
            receptor_residue_count=receptor_residue_count,
            calibrator=getattr(cfg, "confidence_calibrator", None),
        )
        reduced_policy_evaluation = evaluate_confidence_policy(
            cfg=cfg,
            confidence_features=reduced_confidence_metadata.get("confidence_features"),
            confidence_output=reduced_confidence_metadata.get("confidence_output"),
            candidate_pocket_ids=candidate_pocket_ids,
            selected_pocket_ids=selected_after_scan_ids,
            accepted_pocket_ids=accepted_pocket_ids,
        )
        policy_cfg = getattr(cfg, "confidence_policy", {})
        if not isinstance(policy_cfg, dict):
            policy_cfg = {}
        policy_mode = str(policy_cfg.get("mode", "disabled") or "disabled").lower()
        automatic_fallback_mode = bool(
            policy_cfg.get("enabled", False)
            and policy_cfg.get("apply_fallback", False)
            and policy_mode in {"assist_fallback", "assist_all"}
        )
        calibrated_output = reduced_confidence_metadata.get("confidence_output")
        calibrated_output_usable = bool(
            reduced_confidence_metadata.get("confidence_output_available", False)
            and isinstance(calibrated_output, dict)
            and calibrated_output.get("calibrated") is True
            and calibrated_output.get("decision_capable", False) is True
        )
        if (
            automatic_fallback_mode
            and bool(getattr(cfg, "confidence_calibrator_required", False))
            and not calibrated_output_usable
        ):
            calibration_error = reduced_confidence_metadata.get(
                "confidence_calibrator_error_type"
            )
            unavailable_reason = (
                "calibrator_error_conservative_full_fallback"
                if calibration_error
                else (
                    "calibrator_operating_point_unsupported_conservative_full_fallback"
                    if isinstance(calibrated_output, dict)
                    and calibrated_output.get("calibrated") is True
                    and calibrated_output.get("decision_capable", False) is False
                    else "calibrator_output_unavailable_conservative_full_fallback"
                )
            )
            reduced_policy_evaluation = dict(reduced_policy_evaluation)
            failure_details = reduced_policy_evaluation.get(
                "confidence_policy_details"
            )
            if not isinstance(failure_details, dict):
                failure_details = {}
            failure_details = dict(failure_details)
            failure_details.update(
                {
                    "fallback_reason": unavailable_reason,
                    "calibrator_required": True,
                    "calibrated_output_usable": False,
                    "calibrator_error_type": calibration_error,
                    "calibrator_error_message": reduced_confidence_metadata.get(
                        "confidence_calibrator_error_message"
                    ),
                }
            )
            reduced_policy_evaluation.update(
                {
                    "confidence_policy_used": True,
                    "confidence_policy_effect": "fallback_requested",
                    "confidence_assisted_fallback": True,
                    "confidence_fallback_recommended": True,
                    "confidence_fallback_apply_requested": True,
                    "confidence_decision": "fallback_requested",
                    "confidence_recommendation": "execute_full_fallback",
                    "confidence_decision_reasons": _coerce_string_list(
                        [
                            *reduced_policy_evaluation.get(
                                "confidence_decision_reasons", []
                            ),
                            unavailable_reason,
                        ]
                    ),
                    "confidence_warning_flags": _coerce_string_list(
                        [
                            *reduced_policy_evaluation.get(
                                "confidence_warning_flags", []
                            ),
                            "required_calibrator_output_unavailable",
                        ]
                    ),
                    "confidence_policy_details": failure_details,
                }
            )
        reduced_policy_metadata = _combine_confidence_policy_metadata(
            selection_policy=selection_stage_policy_metadata,
            reduced_policy=reduced_policy_evaluation,
        )
        rescoring_applied_any = any(
            bool(item.get("rescoring_applied", False)) for item in per_pocket_results
        )
        ranking_changed_any = any(
            bool(item.get("ranking_changed_after_rescoring", False))
            for item in per_pocket_results
        )
        ranking_changed_parent = bool(
            rescoring_applied_any
            and cross_pocket_selection_policy == "final_rescore"
            and best_cheap_pocket_id
            and best_pocket_id
            and best_cheap_pocket_id != best_pocket_id
        )
        ranking_changed_any = bool(ranking_changed_any or ranking_changed_parent)
        rescoring_selected_candidates: list[dict[str, Any]] = []
        rescoring_cheap_top5_candidates: list[dict[str, Any]] = []
        for item in per_pocket_results:
            pocket_id = item.get("pocket_id")
            candidates = item.get("rescoring_selected_candidates")
            if not isinstance(candidates, list):
                candidates = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                merged = dict(candidate)
                merged.setdefault("pocket_id", pocket_id)
                rescoring_selected_candidates.append(merged)

            top5_rows = item.get("rescoring_cheap_top5_candidates")
            if not isinstance(top5_rows, list):
                continue
            for candidate in top5_rows:
                if not isinstance(candidate, dict):
                    continue
                merged_top5 = dict(candidate)
                merged_top5.setdefault("pocket_id", pocket_id)
                if merged_top5.get("score_cheap") is None:
                    continue
                rescoring_cheap_top5_candidates.append(merged_top5)

        rescoring_cheap_top5_candidates = sorted(
            rescoring_cheap_top5_candidates,
            key=lambda row: float(row.get("score_cheap") or float("-inf")),
            reverse=True,
        )[:5]
        rescoring_candidates_total = int(
            sum(int(item.get("rescoring_candidates_total") or 0) for item in per_pocket_results)
        )
        rescoring_selected_count = int(
            sum(int(item.get("rescoring_selected_count") or 0) for item in per_pocket_results)
        )
        rescored_candidates_count = int(
            sum(int(item.get("rescored_candidates_count") or 0) for item in per_pocket_results)
        )
        if rescoring_candidates_total <= 0:
            rescoring_candidates_total = int(len(rescoring_selected_candidates))
        if rescoring_selected_count <= 0:
            rescoring_selected_count = int(len(rescoring_selected_candidates))
        if rescored_candidates_count <= 0:
            rescored_candidates_count = int(len(rescoring_selected_candidates))
        rescoring_enabled_any = any(bool(item.get("rescoring_enabled", False)) for item in per_pocket_results)
        rescoring_not_applied_reasons = [
            str(item.get("rescoring_not_applied_reason"))
            for item in per_pocket_results
            if item.get("rescoring_not_applied_reason")
        ]
        rescoring_not_applied_reason = None
        if not rescoring_applied_any:
            rescoring_not_applied_reason = (
                rescoring_not_applied_reasons[0]
                if rescoring_not_applied_reasons
                else ("disabled_by_config" if not rescoring_enabled_any else "no_valid_final_rescore")
            )
        rescoring_top_k = int(getattr(cfg, "rescoring_top_k", 1) or 1)
        rescoring_scope = str(getattr(cfg, "rescoring_scope", "global") or "global")
        rescoring_alignment_mode = _rescoring_alignment_mode(cfg)
        criteria_from_pockets = [
            str(item.get("rescoring_selection_criterion"))
            for item in per_pocket_results
            if item.get("rescoring_selection_criterion")
        ]
        unique_criteria = sorted(set(criteria_from_pockets))
        if len(unique_criteria) == 1:
            rescoring_selection_criterion = unique_criteria[0]
        elif len(unique_criteria) > 1:
            rescoring_selection_criterion = "mixed_by_pocket"
        else:
            rescoring_selection_criterion = (
                f"{rescoring_scope}:all_candidates_in_pool"
                if rescoring_alignment_mode == "all_candidates"
                else f"{rescoring_scope}:top_{rescoring_top_k}_by_score_cheap"
            )
        scan_telemetry = scan_params.get("telemetry", {})
        if not isinstance(scan_telemetry, dict):
            scan_telemetry = {}
        all_reduced_final_poses_hard_clash = bool(
            per_pocket_results
            and all(
                bool(item.get("final_hard_clash_triggered", False))
                for item in per_pocket_results
            )
        )
        geometric_fallback_enabled = bool(
            getattr(cfg, "reduced_geometric_fallback_enabled", True)
        )
        geometric_fallback_required = bool(
            geometric_fallback_enabled and all_reduced_final_poses_hard_clash
        )
        parent_summary = {
            "schema_version": "2.0",
            "mode": "reduced_aggregate",
            "run_id": run_id,
            "complex_id": getattr(cfg, "complex_id", None),
            "complex_id_source": getattr(cfg, "complex_id_source", None),
            "complex_id_conflict": getattr(cfg, "complex_id_conflict", None),
            "input_id": getattr(cfg, "input_id", None),
            "search_space_mode": "reduced",
            "budget_policy": budget_policy,
            "cross_pocket_selection_policy": cross_pocket_selection_policy,
            "compare_policy": "best_pocket_vs_full",
            "acceptance_relaxed": bool(scan_telemetry.get("acceptance_relaxed", False)),
            "acceptance_relaxed_reason": scan_telemetry.get("acceptance_relaxed_reason"),
            "acceptance_relaxed_pocket_id": scan_telemetry.get("acceptance_relaxed_pocket_id"),
            **_score_semantics_fields(
                triage_score=best_cheap_value,
                rescore_proxy=best_expensive,
                final_rescore=best_expensive,
                rescoring_applied=rescoring_applied_any,
            ),
            **_scientific_state_fields(),
            **_reproducibility_fields(final_pose_available=parent_final_pose_available),
            **reduced_confidence_metadata,
            **reduced_policy_metadata,
            **_population_confidence_payload_fields(
                by_pocket=population_confidence_by_pocket,
                global_payload=population_confidence_global,
            ),
            "search_engine_name": engine_name_reduced,
            "experimental_round": getattr(cfg, "experimental_round", None),
            "n_rounds": getattr(cfg, "n_rounds", None),
            "protocol_kind": getattr(cfg, "protocol_kind", None),
            "protocol_label": getattr(cfg, "protocol_label", None),
            "protocol_question": getattr(cfg, "protocol_question", None),
            "confidence_policy": getattr(cfg, "confidence_policy", None),
            "search_strategy": search_strategy_reduced,
            "search_algorithm_family": search_algorithm_family_reduced,
            "pose_output_path": best_pose_output_path,
            "candidate_pose_output_path": best_candidate_pose_output_path,
            "candidate_pose_count": best_candidate_pose_count,
            "final_pose_available": parent_final_pose_available,
            "pose_role": best_pose_role or "best_computed_pose",
            "best_pose_meta": best_pose_meta,
            **best_structural_fields,
            "fallback_to_full": False,
            "fallback_executed": False,
            "fallback_recommended": bool(
                geometric_fallback_required
                or reduced_policy_metadata.get("confidence_fallback_recommended", False)
            ),
            "fallback_reason": (
                "all_reduced_final_poses_failed_hard_clash_gate"
                if geometric_fallback_required
                else None
            ),
            "reduced_geometric_fallback_enabled": geometric_fallback_enabled,
            "all_reduced_final_poses_hard_clash": all_reduced_final_poses_hard_clash,
            "reduced_final_pose_geometrically_valid": bool(
                not all_reduced_final_poses_hard_clash
            ),
            "executed_mode": "reduced",
            "pocketing_time_sec": float(pocketing_time),
            "scan_time_sec": float(scan_time),
            "search_time_total_sec": search_time_total_sec,
            "runtime_sec": total_runtime_sec,
            "total_runtime_sec": total_runtime_sec,
            "total_n_eval": total_n_eval,
            "total_eval_budget_requested": total_eval_budget_requested,
            "total_eval_budget_assigned": total_eval_budget_assigned,
            "budget_delta": budget_delta,
            "budget_rounding_applied": budget_rounding_applied,
            "n_pockets_total": total_pockets,
            "n_pockets_used": len(feasible_pockets),
            "candidate_pockets": candidate_pocket_ids,
            "selected_pre_filter_pockets": selected_after_scan_ids,
            "selected_after_scan_pockets": selected_after_scan_ids,
            "accepted_pockets": accepted_pocket_ids,
            "feasible_pockets": feasible_pocket_ids,
            "selected_pockets": executed_pocket_ids,
            "executed_pockets": executed_pocket_ids,
            "best_pocket_id": best_pocket_id,
            "best_score_cheap": best_cheap_value,
            "best_score_expensive": best_expensive,
            "best_over_pockets_cheap": best_cheap_value,
            "best_over_pockets_expensive": best_expensive,
            "max_cheap_score_over_pockets": max_cheap_over_pockets,
            "max_final_rescore_over_pockets": max_expensive_over_pockets,
            "best_over_pockets_cheap_is_selection_score": bool(best_selection_source == "score_cheap"),
            "best_over_pockets_expensive_is_selection_score": bool(best_selection_source == "final_rescore"),
            "aggregate_selection_score_name": best_selection_source,
            "final_rescore": best_expensive,
            "rescoring_enabled": bool(rescoring_enabled_any),
            "rescoring_top_k": int(rescoring_top_k),
            "rescoring_scope": rescoring_scope,
            "rescoring_alignment_mode": rescoring_alignment_mode,
            "rescoring_selection_criterion": rescoring_selection_criterion,
            "rescoring_applied": rescoring_applied_any,
            "rescoring_candidates_total": rescoring_candidates_total,
            "rescoring_selected_count": rescoring_selected_count,
            "rescored_candidates_count": rescored_candidates_count,
            "ranking_changed_after_rescoring": ranking_changed_any,
            "rescoring_not_applied_reason": rescoring_not_applied_reason,
            "rescoring_not_applied_reasons_by_pocket": rescoring_not_applied_reasons,
            "rescoring_selected_candidates": rescoring_selected_candidates,
            "rescoring_cheap_top5_candidates": rescoring_cheap_top5_candidates,
            "cheap_score_variant": str(getattr(cfg, "cheap_score_variant", "geom_kdtree")),
            "final_score_variant": str(getattr(cfg, "final_score_variant", "vina_rescore")),
            "per_pocket_results": per_pocket_results,
            "scan": scan_params,
            "scan_by_pocket": scan_results,
            "rejected_pockets": rejected,
            **self._build_reduced_payload_semantics(
                selected_pre_filter_pockets=selected_after_scan_ids,
                selected_after_scan_pockets=selected_after_scan_ids,
                accepted_pockets=accepted_pocket_ids,
                feasible_pockets=feasible_pocket_ids,
                executed_pockets=executed_pocket_ids,
            ),

        }
        if geometric_fallback_required:
            return self._execute_post_reduced_fallback(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                out_dir=out_dir,
                run_id=run_id,
                tracer=tracer,
                parent_summary=parent_summary,
                fallback_reason_override=(
                    "all_reduced_final_poses_failed_hard_clash_gate"
                ),
                fallback_trigger_source="geometric_hard_clash_gate",
            )
        if bool(reduced_policy_metadata.get("confidence_assisted_fallback", False)):
            return self._execute_post_reduced_fallback(
                cfg=cfg,
                receptor=receptor,
                peptide=peptide,
                receptor_path=receptor_path,
                peptide_path=peptide_path,
                out_dir=out_dir,
                run_id=run_id,
                tracer=tracer,
                parent_summary=parent_summary,
            )
        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(parent_summary, indent=2))
        with open(os.path.join(out_dir, "result.json"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(parent_summary, indent=2))
        with open(os.path.join(out_dir, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"name": "runtime.pocketing_time_sec", "value": float(pocketing_time)}) + "\n")
            handle.write(json.dumps({"name": "runtime.scan_time_sec", "value": float(scan_time)}) + "\n")
            handle.write(json.dumps({"name": "runtime.search_time_total_sec", "value": search_time_total_sec}) + "\n")
            handle.write(json.dumps({"name": "runtime.total_runtime_sec", "value": total_runtime_sec}) + "\n")
            for item in per_pocket_results:
                handle.write(json.dumps({"name": "pocket.best_score_cheap", "pocket_id": item["pocket_id"], "value": item["best_score_cheap"]}) + "\n")
                handle.write(json.dumps({"name": "pocket.n_eval_total", "pocket_id": item["pocket_id"], "value": item["n_eval_total"]}) + "\n")
        with open(os.path.join(out_dir, "metrics.timeseries.jsonl"), "w", encoding="utf-8") as handle:
            for idx, item in enumerate(per_pocket_results):
                handle.write(json.dumps({"step": idx, "pocket_id": item["pocket_id"], "best_score_cheap": item["best_score_cheap"], "n_eval_cumulative": sum(x["n_eval_total"] for x in per_pocket_results[: idx + 1])}) + "\n")
        self._escrever_debug_summary(
            tracer,
            {
                "success": True,
                "requested_mode": "reduced",
                "executed_mode": "reduced",
                "search_space_mode": "reduced",
                "budget_policy": budget_policy,
                "compare_policy": "best_pocket_vs_full",
                "search_engine_name": engine_name_reduced,
                "search_strategy": search_strategy_reduced,
                "search_algorithm_family": search_algorithm_family_reduced,
                "pose_output_path": best_pose_output_path,
                "candidate_pose_output_path": best_candidate_pose_output_path,
                "candidate_pose_count": best_candidate_pose_count,
                "final_pose_available": parent_final_pose_available,
                "pose_role": best_pose_role or "best_computed_pose",
                "fallback_to_full": False,
                "fallback_executed": False,
                "fallback_recommended": bool(reduced_policy_metadata.get("confidence_fallback_recommended", False)),
                "fallback_reason": None,
                "confidence_features_available": bool(
                    selection_confidence_metadata.get("confidence_features_available", False)
                ),
                "confidence_policy_mode": reduced_policy_metadata.get("confidence_policy_mode"),
                "confidence_policy_effect": reduced_policy_metadata.get("confidence_policy_effect"),
                "confidence_assisted_selection": bool(reduced_policy_metadata.get("confidence_assisted_selection", False)),
                "confidence_assisted_fallback": bool(reduced_policy_metadata.get("confidence_assisted_fallback", False)),
                "total_runtime_sec": total_runtime_sec,
                "total_n_eval": total_n_eval,
                "best_score_cheap": best_cheap_value,
                "best_score_expensive": best_expensive,
                "final_rescore": best_expensive,
                "rescoring_enabled": bool(rescoring_enabled_any),
                "rescoring_top_k": int(rescoring_top_k),
                "rescoring_scope": rescoring_scope,
                "rescoring_alignment_mode": rescoring_alignment_mode,
                "rescoring_selection_criterion": rescoring_selection_criterion,
                "rescoring_applied": rescoring_applied_any,
                "rescoring_candidates_total": rescoring_candidates_total,
                "rescoring_selected_count": rescoring_selected_count,
                "rescored_candidates_count": rescored_candidates_count,
                "rescoring_not_applied_reason": rescoring_not_applied_reason,
                "ranking_changed_after_rescoring": ranking_changed_any,
                "rescoring_cheap_top5_candidates": rescoring_cheap_top5_candidates,
                "best_pocket_id": best_pocket_id,
                "n_pockets_total": total_pockets,
                "n_pockets_used": len(feasible_pockets),
                "candidate_pockets": [str(p.id) for p in candidate_pockets],
                "selected_pre_filter_pockets": [str(p.id) for p in selected_pockets],
                "selected_after_scan_pockets": [str(p.id) for p in selected_pockets],
                "accepted_pockets": [str(p.id) for _, p in feasible_pockets],
                "executed_pockets": [str(p.id) for _, p in feasible_pockets],
                "total_eval_budget_requested": total_eval_budget_requested,
                "total_eval_budget_assigned": total_eval_budget_assigned,
                "budget_delta": budget_delta,
                "budget_rounding_applied": budget_rounding_applied,
                "warnings_count": tracer.warnings_count,
                "errors_count": tracer.errors_count,
                "selected_pockets": [str(p.id) for _, p in feasible_pockets],
                "rejected_pockets": rejected,
            },
        )
        for item in per_pocket_results:
            pocket_trace = {
                "success": True,
                "requested_mode": "reduced",
                "executed_mode": "reduced",
                "search_space_mode": "reduced",
                "best_score_cheap": item["best_score_cheap"],
                "best_score_expensive": item["best_score_expensive"],
                "final_rescore": item.get("final_rescore", item["best_score_expensive"]),
                "rescoring_enabled": bool(item.get("rescoring_enabled", False)),
                "rescoring_top_k": item.get("rescoring_top_k"),
                "rescoring_scope": item.get("rescoring_scope"),
                "rescoring_alignment_mode": item.get("rescoring_alignment_mode"),
                "rescoring_selection_criterion": item.get("rescoring_selection_criterion"),
                "rescoring_applied": bool(item.get("rescoring_applied", False)),
                "rescoring_candidates_total": item.get("rescoring_candidates_total"),
                "rescoring_selected_count": item.get("rescoring_selected_count"),
                "rescored_candidates_count": item.get("rescored_candidates_count"),
                "rescoring_not_applied_reason": item.get("rescoring_not_applied_reason"),
                "ranking_changed_after_rescoring": item.get("ranking_changed_after_rescoring"),
                "rescoring_selected_candidates": item.get("rescoring_selected_candidates", []),
                "rescoring_cheap_top5_candidates": item.get("rescoring_cheap_top5_candidates", []),
                "best_pocket_id": item["pocket_id"],
                "pose_output_path": item.get("pose_output_path"),
                "candidate_pose_output_path": item.get("candidate_pose_output_path"),
                "candidate_pose_count": item.get("candidate_pose_count", 0),
                "final_pose_available": bool(item.get("final_pose_available", False)),
                "pose_role": item.get("pose_role"),
            }
            tracer.write_summary(pocket_trace, rel_path=f"pockets/{item['pocket_id']}/debug_summary.json")
            _write_pocket_trace_artifacts(
                tracer,
                pocket_id=str(item["pocket_id"]),
                payload=pocket_trace,
            )

        if best_result is None:
            raise ValueError("Reduced aggregate produced no result.")
        return best_result
