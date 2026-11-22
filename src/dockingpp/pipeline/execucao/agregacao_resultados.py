"""Fachada pública para agregação e persistência de resultados do pipeline.

Os builders e o persistidor concretos ficam em módulo privado para manter este
contrato histórico pequeno e estável para os consumidores do pipeline.
"""

from __future__ import annotations

from typing import Any

from ._agregacao_builders import _AgregadorResultadosPipelineImpl


class AgregadorResultadosPipeline:
    """Coordena builders de resultados sem alterar os schemas publicados."""

    def __init__(self) -> None:
        self._impl = _AgregadorResultadosPipelineImpl()

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
        """Monta o payload histórico de ``result.json``."""

        return self._impl.construir_payload_execucao(
            run_id=run_id,
            mode=mode,
            search_space_mode=search_space_mode,
            runtime_sec=runtime_sec,
            total_pockets=total_pockets,
            selected_pockets=selected_pockets,
            best_score_cheap=best_score_cheap,
            best_score_expensive=best_score_expensive,
            best_pose_id=best_pose_id,
            config_resolved_subset=config_resolved_subset,
            pocketing_time=pocketing_time,
            scan_time=scan_time,
            search_time=search_time,
            pose_output_path=pose_output_path,
            final_pose_available=final_pose_available,
            pose_role=pose_role,
            best_pose_meta=best_pose_meta,
            selected_pocket_ids=selected_pocket_ids,
            scan_by_pocket=scan_by_pocket,
            fallback_to_full=fallback_to_full,
            confidence_policy_metadata=confidence_policy_metadata,
            confidence_calibrator=confidence_calibrator,
            rescoring_summary=rescoring_summary,
        )

    def persistir_payload_resultado(self, *, out_dir: str, payload: dict[str, Any]) -> str:
        """Persiste ``result.json`` preservando formatação e caminho históricos."""

        return self._impl.persistir_payload_resultado(out_dir=out_dir, payload=payload)

    def escrever_summary_execucao(
        self,
        *,
        out_dir: str,
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
    ) -> str:
        """Consolida métricas e persiste ``summary.json``."""

        return self._impl.escrever_summary_execucao(
            out_dir=out_dir,
            run_id=run_id,
            mode=mode,
            receptor_path=receptor_path,
            peptide_path=peptide_path,
            search_space_mode=search_space_mode,
            runtime_sec=runtime_sec,
            search_time_sec=search_time_sec,
            pocketing_sec=pocketing_sec,
            scan_sec=scan_sec,
            total_pockets=total_pockets,
            selected_pockets=selected_pockets,
            best_score_cheap=best_score_cheap,
            best_score_expensive=best_score_expensive,
            best_pose_pocket_id=best_pose_pocket_id,
            config_resolved_subset=config_resolved_subset,
            records=records,
            pockets=pockets,
            scan_params=scan_params,
            scan_by_pocket=scan_by_pocket,
            selected_pocket_ids=selected_pocket_ids,
            pose_output_path=pose_output_path,
            final_pose_available=final_pose_available,
            pose_role=pose_role,
            best_pose_meta=best_pose_meta,
            confidence_policy_metadata=confidence_policy_metadata,
            confidence_calibrator=confidence_calibrator,
            rescoring_summary=rescoring_summary,
        )

    # Métodos mantidos na fachada para compatibilidade com consumidores antigos.
    _safe_float = staticmethod(_AgregadorResultadosPipelineImpl._safe_float)
    _score_semantics_fields = staticmethod(
        _AgregadorResultadosPipelineImpl._score_semantics_fields
    )
    _scientific_state_fields = staticmethod(
        _AgregadorResultadosPipelineImpl._scientific_state_fields
    )
    _reproducibility_fields = staticmethod(
        _AgregadorResultadosPipelineImpl._reproducibility_fields
    )
    _record_field = staticmethod(_AgregadorResultadosPipelineImpl._record_field)
