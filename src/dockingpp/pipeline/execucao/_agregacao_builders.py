"""Agregacao e persistencia de resultados do pipeline."""

from __future__ import annotations

import json
import os
from typing import Any

from dockingpp.pipeline.confianca.features import build_single_confidence_metadata
from dockingpp.pipeline.confianca.policy import merge_confidence_policy_metadata


class _SemanticaResultados:
    """Regras puras compartilhadas pelos dois documentos de saída."""

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        """Converte para float, retornando None para valores invalidos."""

        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _score_semantics_fields(
        *,
        triage_score: float | None,
        rescore_proxy: float | None,
        final_rescore: float | None = None,
        rescoring_applied: bool = False,
    ) -> dict[str, Any]:
        """Campos canonicos para semantica de score (fase estrutural)."""

        final_available = bool(rescoring_applied and final_rescore is not None)
        triage_outlier = False
        try:
            triage_outlier = bool(triage_score is not None and float(triage_score) <= -1e5)
        except (TypeError, ValueError):
            triage_outlier = False
        return {
            "triage_score": triage_score,
            "rescore_proxy": rescore_proxy,
            "final_rescore": final_rescore,
            "final_score": final_rescore if final_available else None,
            "final_score_available": final_available,
            "score_role": "triage_plus_final_rescore" if final_available else "triage_proxy",
            "is_proxy_score": not final_available,
            "optimization_score_name": "score_cheap",
            "selection_score_name": "final_rescore" if final_available else "score_cheap",
            "score_direction": "maximize",
            "rescoring_applied": bool(rescoring_applied),
            "triage_outlier_detected": bool(triage_outlier),
            "score_semantics_version": "1.0",
        }

    @staticmethod
    def _scientific_state_fields() -> dict[str, Any]:
        """Estado cientifico explicito para evitar claims indevidos."""

        return {
            "scientific_score_available": False,
            "scientific_score_name": None,
            "triage_pipeline": True,
            "scientific_readiness": "triage_only",
        }

    @staticmethod
    def _reproducibility_fields(*, final_pose_available: bool) -> dict[str, Any]:
        """Metadados minimos de reprodutibilidade atualmente disponiveis."""

        return {
            "pose_artifact_available": bool(final_pose_available),
            "score_semantics_available": True,
            "reproducibility_level": "basic_artifacts" if final_pose_available else "score_semantics_only",
        }

    @staticmethod
    def _record_field(record: dict[str, Any], key: str) -> Any:
        """Obtem campo de record preservando compatibilidade de formato."""

        if key in record:
            return record[key]
        payload = record.get("payload")
        if isinstance(payload, dict) and key in payload:
            return payload[key]
        return None


class ConstrutorPayloadExecucao(_SemanticaResultados):
    """Constrói exclusivamente o contrato de ``result.json``."""

    def construir_payload_execucao(
        self,
        *,
        run_id: str,
        mode: str,
        search_space_mode: str,
        runtime_sec: float,
        total_pockets: int,
        selected_pockets: int,
        best_score_cheap: float | None,
        best_score_expensive: float | None,
        best_pose_id: str | None,
        config_resolved_subset: dict[str, Any],
        pocketing_time: float,
        scan_time: float,
        search_time: float,
        pose_output_path: str | None = None,
        final_pose_available: bool = False,
        pose_role: str | None = None,
        best_pose_meta: dict[str, Any] | None = None,
        selected_pocket_ids: list[str] | None = None,
        scan_by_pocket: dict[str, dict[str, Any]] | None = None,
        fallback_to_full: bool = False,
        confidence_policy_metadata: dict[str, Any] | None = None,
        confidence_calibrator: Any = None,
        rescoring_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Monta payload padrao de execucao unica."""

        best_pocket_id = None
        if isinstance(best_pose_meta, dict) and best_pose_meta.get("pocket_id") is not None:
            best_pocket_id = str(best_pose_meta.get("pocket_id"))
        rescoring_payload = dict(rescoring_summary or {})
        final_rescore = rescoring_payload.get("best_final_rescore")
        if final_rescore is None:
            final_rescore = best_score_expensive
        rescoring_applied = bool(rescoring_payload.get("rescoring_applied", False))
        cheap_score_variant = None
        final_score_variant = rescoring_payload.get("final_score_variant")
        if isinstance(config_resolved_subset, dict):
            cheap_score_variant = config_resolved_subset.get("cheap_score_variant")
            if final_score_variant is None:
                final_score_variant = config_resolved_subset.get("final_score_variant")
        confidence_metadata = build_single_confidence_metadata(
            search_space_mode=search_space_mode,
            best_score_cheap=best_score_cheap,
            best_score_expensive=final_rescore,
            total_pockets=total_pockets,
            selected_pockets=selected_pockets,
            best_pocket_id=best_pocket_id,
            selected_pocket_ids=selected_pocket_ids,
            scan_by_pocket=scan_by_pocket,
            fallback_to_full=fallback_to_full,
            calibrator=confidence_calibrator,
        )
        confidence_policy_payload = merge_confidence_policy_metadata(confidence_policy_metadata)
        search_strategy = None
        search_algorithm_family = None
        if isinstance(config_resolved_subset, dict):
            search_strategy = config_resolved_subset.get("search_strategy")
            search_algorithm_family = config_resolved_subset.get("search_algorithm_family")
        fallback_recommended = bool(
            confidence_policy_payload.get("confidence_fallback_recommended", False)
        )

        return {
            "schema_version": "2.0",
            "mode": mode,
            "run_id": run_id,
            "best_score_cheap": best_score_cheap,
            "best_score_expensive": final_rescore,
            "final_rescore": final_rescore,
            **self._score_semantics_fields(
                triage_score=best_score_cheap,
                rescore_proxy=final_rescore,
                final_rescore=final_rescore,
                rescoring_applied=rescoring_applied,
            ),
            **self._scientific_state_fields(),
            **self._reproducibility_fields(final_pose_available=bool(final_pose_available)),
            **confidence_metadata,
            **confidence_policy_payload,
            "rescoring_enabled": bool(rescoring_payload.get("rescoring_enabled", False)),
            "rescoring_applied": bool(rescoring_applied),
            "rescoring_top_k": rescoring_payload.get("rescoring_top_k"),
            "rescoring_scope": rescoring_payload.get("rescoring_scope"),
            "rescoring_alignment_mode": rescoring_payload.get("rescoring_alignment_mode"),
            "rescoring_selection_criterion": rescoring_payload.get("rescoring_selection_criterion"),
            "rescoring_selection_score_name": rescoring_payload.get("rescoring_selection_score_name"),
            "population_confidence_used_for_rescoring_selection": rescoring_payload.get(
                "population_confidence_used_for_rescoring_selection",
                False,
            ),
            "population_confidence_rescoring_level": rescoring_payload.get(
                "population_confidence_rescoring_level"
            ),
            "rescoring_candidates_total": rescoring_payload.get("rescoring_candidates_total"),
            "rescoring_selected_count": rescoring_payload.get("rescoring_selected_count"),
            "rescored_candidates_count": rescoring_payload.get("rescored_candidates_count"),
            "ranking_changed_after_rescoring": rescoring_payload.get("ranking_changed_after_rescoring"),
            "rescoring_not_applied_reason": rescoring_payload.get("rescoring_not_applied_reason"),
            "rescoring_selected_candidates": rescoring_payload.get("selected_candidates", []),
            "rescoring_cheap_top5_candidates": rescoring_payload.get("cheap_top5_candidates", []),
            "candidate_pose_output_path": rescoring_payload.get("candidate_pose_output_path"),
            "candidate_pose_count": int(rescoring_payload.get("candidate_pose_count", 0) or 0),
            "cheap_score_variant": cheap_score_variant,
            "final_score_variant": final_score_variant,
            "best_pose_id": best_pose_id,
            "n_pockets_detected": total_pockets,
            "n_pockets_used": selected_pockets,
            "fallback_to_full": bool(fallback_to_full),
            "fallback_executed": bool(fallback_to_full),
            "fallback_recommended": bool(fallback_recommended),
            "search_space_mode": search_space_mode,
            "search_strategy": search_strategy,
            "search_algorithm_family": search_algorithm_family,
            "runtime_sec": runtime_sec,
            "config_resolved_subset": config_resolved_subset,
            "pose_output_path": pose_output_path,
            "final_pose_available": bool(final_pose_available),
            "pose_role": pose_role,
            "best_pose_meta": best_pose_meta,
            "timing": {
                "total_s": runtime_sec,
                "scoring_cheap_s": None,
                "scoring_expensive_s": None,
                "scoring_final_rescore_s": None,
                "pocketing_s": pocketing_time,
                "scan_s": scan_time,
                "search_s": search_time,
            },
        }

class ConstrutorSummaryExecucao(_SemanticaResultados):
    """Consolida métricas no contrato de ``summary.json``."""

    def construir_summary_execucao(
        self,
        *,
        run_id: str,
        mode: str,
        receptor_path: str,
        peptide_path: str,
        search_space_mode: str,
        runtime_sec: float,
        search_time_sec: float,
        pocketing_sec: float,
        scan_sec: float,
        total_pockets: int,
        selected_pockets: int,
        best_score_cheap: float | None,
        best_score_expensive: float | None,
        best_pose_pocket_id: str | None,
        config_resolved_subset: dict[str, Any],
        records: list[dict[str, Any]],
        pockets: list[Any],
        scan_params: dict[str, Any],
        scan_by_pocket: dict[str, dict[str, Any]],
        selected_pocket_ids: list[str],
        pose_output_path: str | None = None,
        final_pose_available: bool = False,
        pose_role: str | None = None,
        best_pose_meta: dict[str, Any] | None = None,
        confidence_policy_metadata: dict[str, Any] | None = None,
        confidence_calibrator: Any = None,
        rescoring_summary: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Consolida métricas finais sem realizar I/O."""

        expensive_ran = 0.0
        expensive_skipped = 0.0
        final_rescore_ran = 0.0
        final_rescore_skipped = 0.0
        n_eval_total = 0.0
        best_by_pocket: dict[int, float] = {}

        for record in records:
            name = record.get("name")
            value = self._safe_float(record.get("value"))
            if name == "expensive_ran" and value is not None:
                expensive_ran += value
            elif name == "expensive_skipped" and value is not None:
                expensive_skipped += value
            elif name == "final_rescore_ran" and value is not None:
                final_rescore_ran += value
            elif name == "final_rescore_skipped" and value is not None:
                final_rescore_skipped += value
            elif name == "n_eval" and value is not None:
                n_eval_total += value
            elif name == "best_score" and value is not None:
                pocket_index = self._record_field(record, "pocket_index")
                if pocket_index is None:
                    continue
                pocket_idx = int(pocket_index)
                current = best_by_pocket.get(pocket_idx)
                if current is None or value > current:
                    best_by_pocket[pocket_idx] = value

        best_cheap_by_pocket = []
        for pocket_idx in sorted(best_by_pocket):
            pocket_id = pockets[pocket_idx].id if pocket_idx < len(pockets) else str(pocket_idx)
            best_cheap_by_pocket.append(
                {"pocket_id": pocket_id, "best_score_cheap": best_by_pocket[pocket_idx]}
            )

        best_expensive_by_pocket = []
        if best_score_expensive is not None and best_pose_pocket_id is not None:
            best_expensive_by_pocket.append(
                {"pocket_id": best_pose_pocket_id, "best_score_expensive": best_score_expensive}
            )

        rescoring_payload = dict(rescoring_summary or {})
        final_rescore = rescoring_payload.get("best_final_rescore")
        if final_rescore is None:
            final_rescore = best_score_expensive
        rescoring_applied = bool(rescoring_payload.get("rescoring_applied", False))
        cheap_score_variant = None
        final_score_variant = rescoring_payload.get("final_score_variant")
        if isinstance(config_resolved_subset, dict):
            cheap_score_variant = config_resolved_subset.get("cheap_score_variant")
            if final_score_variant is None:
                final_score_variant = config_resolved_subset.get("final_score_variant")

        reduction_ratio = 0.0
        if total_pockets > 0:
            reduction_ratio = max(0.0, 1.0 - (float(selected_pockets) / float(total_pockets)))

        input_id = getattr(config_resolved_subset, "input_id", None)
        if input_id is None and isinstance(config_resolved_subset, dict):
            input_id = config_resolved_subset.get("input_id")
        if input_id is None:
            receptor_name = os.path.basename(receptor_path) if receptor_path else ""
            peptide_name = os.path.basename(peptide_path) if peptide_path else ""
            if receptor_name and peptide_name:
                input_id = f"{receptor_name}__{peptide_name}"
            elif receptor_name:
                input_id = receptor_name
            elif peptide_name:
                input_id = peptide_name
            else:
                input_id = None
        complex_id = None
        complex_id_source = None
        complex_id_conflict = None
        if isinstance(config_resolved_subset, dict):
            # ``input_id`` is only a technical identifier (filenames) and must
            # never be promoted to a scientific ``complex_id``.  Keeping these
            # fields separate is essential for grouped calibration by complex.
            complex_id = config_resolved_subset.get("complex_id")
            complex_id_source = config_resolved_subset.get("complex_id_source")
            complex_id_conflict = config_resolved_subset.get("complex_id_conflict")

        expensive_every = int(getattr(config_resolved_subset, "expensive_every", 0) or 0)
        if isinstance(config_resolved_subset, dict):
            expensive_every = int(config_resolved_subset.get("expensive_every", 0) or 0)
        expensive_topk = None
        if isinstance(config_resolved_subset, dict):
            expensive_topk = config_resolved_subset.get("expensive_topk")
        rescoring_enabled = bool(rescoring_payload.get("rescoring_enabled", False))
        if not rescoring_enabled and isinstance(config_resolved_subset, dict):
            rescoring_enabled = bool(config_resolved_subset.get("rescoring_enabled", False))
        if not rescoring_enabled:
            rescoring_enabled = expensive_every > 0 or (expensive_topk is not None and int(expensive_topk or 0) > 0)
        rescoring_policy = {
            "enabled": bool(rescoring_enabled),
            "top_k": rescoring_payload.get("rescoring_top_k"),
            "scope": rescoring_payload.get("rescoring_scope"),
            "alignment_mode": rescoring_payload.get("rescoring_alignment_mode"),
            "selection_criterion": rescoring_payload.get("rescoring_selection_criterion"),
            "selection_score_name": rescoring_payload.get("rescoring_selection_score_name"),
        }
        expensive_enabled = bool(rescoring_enabled)
        expensive_policy = {
            "every": expensive_every,
            "topk": expensive_topk,
        }

        confidence_metadata = build_single_confidence_metadata(
            search_space_mode=search_space_mode,
            best_score_cheap=best_score_cheap,
            best_score_expensive=final_rescore,
            total_pockets=total_pockets,
            selected_pockets=selected_pockets,
            best_pocket_id=best_pose_pocket_id,
            selected_pocket_ids=selected_pocket_ids,
            scan_by_pocket=scan_by_pocket,
            fallback_to_full=False,
            calibrator=confidence_calibrator,
        )
        confidence_policy_payload = merge_confidence_policy_metadata(confidence_policy_metadata)
        search_strategy = None
        search_algorithm_family = None
        if isinstance(config_resolved_subset, dict):
            search_strategy = config_resolved_subset.get("search_strategy")
            search_algorithm_family = config_resolved_subset.get("search_algorithm_family")
        fallback_recommended = bool(
            confidence_policy_payload.get("confidence_fallback_recommended", False)
        )

        summary_payload = {
            "schema_version": "2.0",
            "run_id": run_id,
            "complex_id": complex_id,
            "complex_id_source": complex_id_source,
            "complex_id_conflict": complex_id_conflict,
            "input_id": input_id,
            "seed": config_resolved_subset.get("seed") if isinstance(config_resolved_subset, dict) else None,
            "search_space_mode": search_space_mode,
            "search_strategy": search_strategy,
            "search_algorithm_family": search_algorithm_family,
            "runtime_sec": runtime_sec,
            "search_time_sec": search_time_sec,
            "pocketing_sec": pocketing_sec,
            "scan_sec": scan_sec,
            "n_eval_total": int(n_eval_total),
            "n_pockets_total": total_pockets,
            "n_pockets_used": selected_pockets,
            "fallback_to_full": False,
            "fallback_executed": False,
            "fallback_recommended": bool(fallback_recommended),
            "reduction_ratio": reduction_ratio,
            "best_score_cheap": best_score_cheap,
            "best_score_expensive": final_rescore,
            "final_rescore": final_rescore,
            **self._score_semantics_fields(
                triage_score=best_score_cheap,
                rescore_proxy=final_rescore,
                final_rescore=final_rescore,
                rescoring_applied=rescoring_applied,
            ),
            **self._scientific_state_fields(),
            **self._reproducibility_fields(final_pose_available=bool(final_pose_available)),
            **confidence_metadata,
            **confidence_policy_payload,
            "rescoring_enabled": bool(rescoring_enabled),
            "rescoring_applied": bool(rescoring_applied),
            "rescoring_policy": rescoring_policy,
            "rescoring_alignment_mode": rescoring_payload.get("rescoring_alignment_mode"),
            "rescoring_selection_criterion": rescoring_payload.get("rescoring_selection_criterion"),
            "rescoring_selection_score_name": rescoring_payload.get("rescoring_selection_score_name"),
            "population_confidence_used_for_rescoring_selection": rescoring_payload.get(
                "population_confidence_used_for_rescoring_selection",
                False,
            ),
            "population_confidence_rescoring_level": rescoring_payload.get(
                "population_confidence_rescoring_level"
            ),
            "rescoring_candidates_total": rescoring_payload.get("rescoring_candidates_total"),
            "rescoring_selected_count": rescoring_payload.get("rescoring_selected_count"),
            "rescored_candidates_count": rescoring_payload.get("rescored_candidates_count"),
            "ranking_changed_after_rescoring": rescoring_payload.get("ranking_changed_after_rescoring"),
            "rescoring_not_applied_reason": rescoring_payload.get("rescoring_not_applied_reason"),
            "rescoring_selected_candidates": rescoring_payload.get("selected_candidates", []),
            "rescoring_cheap_top5_candidates": rescoring_payload.get("cheap_top5_candidates", []),
            "candidate_pose_output_path": rescoring_payload.get("candidate_pose_output_path"),
            "candidate_pose_count": int(rescoring_payload.get("candidate_pose_count", 0) or 0),
            "cheap_score_variant": cheap_score_variant,
            "final_score_variant": final_score_variant,
            "expensive_enabled": expensive_enabled,
            "expensive_policy": expensive_policy,
            "mode": mode,
            "n_pockets_detected": total_pockets,
            "expensive_ran_count": int(final_rescore_ran or expensive_ran),
            "expensive_skipped_count": int(final_rescore_skipped or expensive_skipped),
            "final_rescore_ran_count": int(final_rescore_ran),
            "final_rescore_skipped_count": int(final_rescore_skipped),
            "best_cheap_by_pocket": best_cheap_by_pocket,
            "best_expensive_by_pocket": best_expensive_by_pocket,
            "config_resolved_subset": config_resolved_subset,
            "scan": scan_params,
            "scan_by_pocket": scan_by_pocket,
            "selected_pockets": selected_pocket_ids,
            "pose_output_path": pose_output_path,
            "final_pose_available": bool(final_pose_available),
            "pose_role": pose_role,
            "best_pose_meta": best_pose_meta,
        }

        return summary_payload


class PersistidorResultadosJson:
    """Escreve os documentos JSON usando a formatação histórica."""

    @staticmethod
    def persistir(*, out_dir: str, nome: str, payload: dict[str, Any]) -> str:
        caminho = os.path.join(out_dir, nome)
        with open(caminho, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
        return caminho


class _AgregadorResultadosPipelineImpl(_SemanticaResultados):
    """Composição privada dos builders e do adaptador de persistência."""

    def __init__(self) -> None:
        self._payload_builder = ConstrutorPayloadExecucao()
        self._summary_builder = ConstrutorSummaryExecucao()
        self._persistidor = PersistidorResultadosJson()

    def construir_payload_execucao(self, **kwargs: Any) -> dict[str, Any]:
        return self._payload_builder.construir_payload_execucao(**kwargs)

    def persistir_payload_resultado(self, *, out_dir: str, payload: dict[str, Any]) -> str:
        return self._persistidor.persistir(
            out_dir=out_dir,
            nome="result.json",
            payload=payload,
        )

    def escrever_summary_execucao(self, *, out_dir: str, **kwargs: Any) -> str:
        summary_payload = self._summary_builder.construir_summary_execucao(**kwargs)
        return self._persistidor.persistir(
            out_dir=out_dir,
            nome="summary.json",
            payload=summary_payload,
        )
