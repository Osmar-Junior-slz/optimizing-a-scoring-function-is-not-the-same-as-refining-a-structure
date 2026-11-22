"""Fluxos diretos full e legado do executor de busca."""

from __future__ import annotations

from typing import Any

from ._executor_busca_contexto import ComponenteExecutorBusca, EntradaExecucaoBusca


def _configured_eval_budget(cfg: Any) -> int:
    explicit = getattr(cfg, "objective_evaluation_budget_per_pocket", None)
    if explicit is not None:
        return max(1, int(explicit))
    return (max(1, int(cfg.generations)) + 1) * max(1, int(cfg.pop_size))


class FluxoBuscaFull(ComponenteExecutorBusca):
    """Executa o modo full e registra seu resumo histórico."""

    def executar(self, entrada: EntradaExecucaoBusca) -> Any:
        cfg = entrada.cfg
        receptor = entrada.receptor
        peptide = entrada.peptide
        receptor_path = entrada.receptor_path
        peptide_path = entrada.peptide_path
        out_dir = entrada.out_dir
        run_id = entrada.run_id
        tracer = entrada.tracer
        selected_pockets = entrada.selected_pockets
        total_pockets = entrada.total_pockets
        scan_results = entrada.scan_results
        scan_params = entrada.scan_params
        pocketing_time = entrada.pocketing_time
        scan_time = entrada.scan_time
        selection_confidence_metadata = entrada.selection_confidence_metadata
        selection_confidence_policy = entrada.selection_confidence_policy

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
                "total_eval_budget_requested": _configured_eval_budget(cfg),
                "total_eval_budget_assigned": _configured_eval_budget(cfg),
                "budget_delta": 0,
                "budget_rounding_applied": False,
                "warnings_count": tracer.warnings_count,
                "errors_count": tracer.errors_count,
                "selected_pockets": [str(p.id) for p in selected_pockets],
                "rejected_pockets": [],
            },
        )
        return result


class FluxoBuscaLegado(ComponenteExecutorBusca):
    """Executa o antigo modo ``pockets`` mantendo sua compatibilidade."""

    def executar(self, entrada: EntradaExecucaoBusca) -> Any:
        cfg = entrada.cfg
        receptor = entrada.receptor
        peptide = entrada.peptide
        receptor_path = entrada.receptor_path
        peptide_path = entrada.peptide_path
        out_dir = entrada.out_dir
        run_id = entrada.run_id
        tracer = entrada.tracer
        selected_pockets = entrada.selected_pockets
        total_pockets = entrada.total_pockets
        scan_results = entrada.scan_results
        scan_params = entrada.scan_params
        pocketing_time = entrada.pocketing_time
        scan_time = entrada.scan_time
        selection_confidence_metadata = entrada.selection_confidence_metadata
        selection_confidence_policy = entrada.selection_confidence_policy

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
                "total_eval_budget_requested": _configured_eval_budget(cfg),
                "total_eval_budget_assigned": _configured_eval_budget(cfg),
                "budget_delta": 0,
                "budget_rounding_applied": False,
                "warnings_count": tracer.warnings_count,
                "errors_count": tracer.errors_count,
                "selected_pockets": [str(p.id) for p in selected_pockets],
                "rejected_pockets": [],
            },
        )
        return result
