"""Execução isolada de um mecanismo de busca.

Este módulo contém somente o fluxo de uma execução: resolve o engine, aplica o
rescoring final e persiste os artefatos. A escolha entre modo full e reduced
permanece no orquestrador de busca.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from dockingpp.evaluation.pipeline_structural import (
    evaluate_and_persist_structural_metrics,
    inject_structural_metrics_into_json,
)
from dockingpp.pipeline.execucao.agregacao_resultados import AgregadorResultadosPipeline
from dockingpp.pipeline.execucao.auditoria_execucao import AuditoriaExecucaoPipeline
from dockingpp.pipeline.logging import RunLogger
from dockingpp.priors.pocket import PriorNetPocket
from dockingpp.priors.pose import PriorNetPose
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.scoring.expensive import score_pose_expensive
from dockingpp.search.engine import resolve_search_engine as default_search_engine_resolver
from .contextos import ContextoExecucaoUnica
from ._rescoring_final import ServicoRescoringFinal
from .rescoring_artefatos import (
    _best_pose_from_result,
    _best_pose_meta,
    _best_pose_score,
    _persist_best_pose,
    _persist_rescored_candidates,
    _rescoring_scope,
    _rescoring_top_k,
)
from .serializacao_confianca import (
    _extract_population_confidence_from_result,
    _inject_population_confidence_into_summary_file,
    _normalize_population_confidence_for_pocket,
    _population_confidence_payload_fields,
    _result_metric,
)


class CoordenadorExecucaoUnica:
    """Coordena busca, rescoring, agregação e persistência de uma execução."""

    def __init__(
        self,
        search_engine_resolver: Callable[[Any], tuple[Any, str]] | None = None,
        servico_rescoring: ServicoRescoringFinal | None = None,
    ) -> None:
        """Recebe a fabrica como dependencia, sem consultar a fachada."""

        self._search_engine_resolver = (
            search_engine_resolver or default_search_engine_resolver
        )
        self._servico_rescoring = servico_rescoring or ServicoRescoringFinal()

    def _aplicar_rescoring_final(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        result: Any,
        logger: RunLogger,
    ) -> dict[str, Any]:
        """Delega o rescoring ao serviço privado, preservando o método histórico."""

        return self._servico_rescoring.aplicar(
            cfg=cfg,
            receptor=receptor,
            peptide=peptide,
            result=result,
            logger=logger,
        )

    def executar(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        pockets: list[Any],
        out_dir: str,
        contexto: ContextoExecucaoUnica,
    ) -> tuple[Any, dict[str, Any], RunLogger]:
        """Executa busca unica e persiste os artefatos padrao sem alterar semantica."""

        os.makedirs(out_dir, exist_ok=True)
        logger = RunLogger(out_dir=out_dir, live_write=True)
        cfg.expensive_logger = logger
        search, engine_name = self._search_engine_resolver(cfg)
        prior_pocket = PriorNetPocket()
        prior_pose = PriorNetPose()

        pockets_para_busca = pockets
        if isinstance(receptor, dict):
            receptor_coords_raw = receptor.get("coords")
        elif isinstance(receptor, np.ndarray):
            receptor_coords_raw = receptor
        else:
            receptor_coords_raw = getattr(receptor, "coords", None)
        receptor_coords_full = (
            np.asarray(receptor_coords_raw, dtype=float)
            if receptor_coords_raw is not None
            else np.zeros((0, 3), dtype=float)
        )
        for pocket in pockets_para_busca:
            if not isinstance(getattr(pocket, "meta", None), dict):
                pocket.meta = {}
            pocket.meta["receptor_coords_full"] = receptor_coords_full
            pocket.meta.setdefault(
                "local_coords",
                np.asarray(getattr(pocket, "coords", np.zeros((0, 3))), dtype=float),
            )
        # Regra historica: no modo full, `max_pockets_used` limita o conjunto
        # entregue ao motor de busca sem alterar a selecao upstream.
        if contexto.search_space_mode == "full":
            limite_pockets = int(getattr(cfg, "max_pockets_used", 0) or 0)
            if limite_pockets > 0:
                pockets_para_busca = pockets[:limite_pockets]

        auditoria = AuditoriaExecucaoPipeline()
        auditoria.registrar_inicio_busca(
            tracer=contexto.tracer,
            cfg=cfg,
            pocket_id=contexto.pocket_id,
            engine_name=engine_name,
        )
        start_search = time.perf_counter()
        result = search.search(
            receptor=receptor,
            peptide=peptide,
            pockets=pockets_para_busca,
            cfg=cfg,
            score_cheap_fn=score_pose_cheap,
            score_expensive_fn=score_pose_expensive,
            prior_pocket=prior_pocket,
            prior_pose=prior_pose,
            logger=logger,
        )
        end_search = time.perf_counter()
        auditoria.registrar_fim_busca(
            tracer=contexto.tracer,
            pocket_id=contexto.pocket_id,
            runtime_sec=end_search - start_search,
            engine_name=engine_name,
        )
        rescoring_summary = self._aplicar_rescoring_final(
            cfg=cfg,
            receptor=receptor,
            peptide=peptide,
            result=result,
            logger=logger,
        )
        population_confidence_by_pocket, population_confidence_global = _extract_population_confidence_from_result(
            result
        )
        population_confidence_by_pocket, _ = _normalize_population_confidence_for_pocket(
            pocket_id=contexto.pocket_id,
            by_pocket=population_confidence_by_pocket,
        )
        search_strategy = str(_result_metric(result, "search_strategy", getattr(cfg, "search_engine", "hybrid")) or "hybrid")
        search_algorithm_family = str(
            _result_metric(
                result,
                "search_algorithm_family",
                "ABC+AG+VGOS" if search_strategy == "hybrid" else search_strategy.upper(),
            )
            or search_strategy
        )

        config_resolved_subset = {
            "docking_task": getattr(cfg, "docking_task", "surface_discovery"),
            "local_refinement_reduced_scale": getattr(cfg, "local_refinement_reduced_scale", 0.5),
            "max_trans": cfg.max_trans,
            "max_rot_deg": cfg.max_rot_deg,
            "seed": cfg.seed,
            "complex_id": getattr(cfg, "complex_id", None),
            "complex_id_source": getattr(cfg, "complex_id_source", None),
            "complex_id_conflict": getattr(cfg, "complex_id_conflict", None),
            "input_id": getattr(cfg, "input_id", None),
            "experimental_round": getattr(cfg, "experimental_round", None),
            "n_rounds": getattr(cfg, "n_rounds", None),
            "protocol_kind": getattr(cfg, "protocol_kind", None),
            "protocol_label": getattr(cfg, "protocol_label", None),
            "protocol_question": getattr(cfg, "protocol_question", None),
            "generations": cfg.generations,
            "pop_size": cfg.pop_size,
            "topk": cfg.topk,
            "search_engine": str(getattr(cfg, "search_engine", "hybrid") or "hybrid"),
            "search_space_mode": contexto.search_space_mode,
            "full_search": bool(getattr(cfg, "full_search", True)),
            "top_pockets": int(getattr(cfg, "top_pockets", 0) or 0),
            "max_pockets_used": int(getattr(cfg, "max_pockets_used", 0) or 0),
            "expensive_every": int(getattr(cfg, "expensive_every", 0) or 0),
            "expensive_topk": getattr(cfg, "expensive_topk", None),
            "debug_log_enabled": bool(getattr(cfg, "debug_log_enabled", False)),
            "debug_log_path": getattr(cfg, "debug_log_path", None),
            "debug_log_level": getattr(cfg, "debug_log_level", None),
            "search_engine_name": engine_name,
            "search_strategy": search_strategy,
            "search_algorithm_family": search_algorithm_family,
            "scan": contexto.scan_params,
            "budget_policy": getattr(cfg, "budget_policy", "split"),
            "rescoring_enabled": bool(rescoring_summary.get("rescoring_enabled", False)),
            "rescoring_top_k": int(rescoring_summary.get("rescoring_top_k", _rescoring_top_k(cfg))),
            "rescoring_scope": str(rescoring_summary.get("rescoring_scope", _rescoring_scope(cfg))),
            "cheap_score_variant": str(getattr(cfg, "cheap_score_variant", "geom_kdtree")),
            "final_score_variant": str(
                rescoring_summary.get(
                    "final_score_variant",
                    getattr(cfg, "final_score_variant", "vina_rescore"),
                )
            ),
            "structural_metrics_enabled": bool(
                getattr(cfg, "structural_metrics_enabled", True)
            ),
            "structural_contact_cutoff_angstrom": float(
                getattr(cfg, "structural_contact_cutoff_angstrom", 4.0)
            ),
            "structural_interface_cutoff_angstrom": float(
                getattr(cfg, "structural_interface_cutoff_angstrom", 8.0)
            ),
            "confidence_policy": getattr(cfg, "confidence_policy", None),
            "confidence_calibrator_path": getattr(
                cfg, "confidence_calibrator_path", None
            ),
            "confidence_calibrator_required": bool(
                getattr(cfg, "confidence_calibrator_required", False)
            ),
            "confidence_calibrator_load_error": getattr(
                cfg, "confidence_calibrator_load_error", None
            ),
        }
        best_pose = _best_pose_from_result(result)
        best_pose_meta = _best_pose_meta(best_pose)
        best_score_cheap = _best_pose_score(best_pose, "score_cheap")
        best_score_expensive = (
            _best_pose_score(best_pose, "final_rescore")
            or _best_pose_score(best_pose, "score_expensive")
        )
        summary_best_final_rescore = rescoring_summary.get("best_final_rescore")
        if summary_best_final_rescore is not None:
            best_score_expensive = float(summary_best_final_rescore)
        best_pose_id = best_pose_meta.get("pose_id") or best_pose_meta.get("id")
        best_pose_pocket_id = best_pose_meta.get("pocket_id")

        pose_artifact = _persist_best_pose(
            out_dir=out_dir,
            best_pose=best_pose,
            engine_name=engine_name,
            search_space_mode=contexto.search_space_mode,
        )
        pose_output_path = pose_artifact.get("pose_output_path")
        final_pose_available = bool(pose_artifact.get("final_pose_available", False))
        pose_role = pose_artifact.get("pose_role")
        persisted_pose_meta = pose_artifact.get("best_pose_meta")
        structural_fields = evaluate_and_persist_structural_metrics(
            cfg=cfg,
            out_dir=out_dir,
            receptor_path=contexto.receptor_path,
            peptide_path=contexto.peptide_path,
            pose_output_path=pose_output_path,
        )
        candidate_artifact = _persist_rescored_candidates(
            out_dir=out_dir,
            result=result,
            engine_name=engine_name,
            search_space_mode=contexto.search_space_mode,
        )
        rescoring_summary.update(candidate_artifact)

        search_time_sec = end_search - start_search
        pocketing_sec = contexto.pocketing_time if contexto.include_global_overhead else 0.0
        scan_sec = contexto.scan_time if contexto.include_global_overhead else 0.0
        runtime_sec = pocketing_sec + scan_sec + search_time_sec

        agregador = AgregadorResultadosPipeline()
        payload = agregador.construir_payload_execucao(
            run_id=contexto.run_id,
            mode="single",
            search_space_mode=contexto.search_space_mode,
            runtime_sec=runtime_sec,
            total_pockets=contexto.total_pockets,
            selected_pockets=len(pockets_para_busca),
            best_score_cheap=best_score_cheap,
            best_score_expensive=best_score_expensive,
            best_pose_id=best_pose_id,
            config_resolved_subset=config_resolved_subset,
            pocketing_time=pocketing_sec,
            scan_time=scan_sec,
            search_time=search_time_sec,
            pose_output_path=pose_output_path,
            final_pose_available=final_pose_available,
            pose_role=pose_role,
            best_pose_meta=persisted_pose_meta,
            selected_pocket_ids=[str(p.id) for p in pockets_para_busca],
            scan_by_pocket=contexto.scan_results,
            fallback_to_full=False,
            confidence_policy_metadata=contexto.confidence_policy_metadata,
            confidence_calibrator=getattr(cfg, "confidence_calibrator", None),
            rescoring_summary=rescoring_summary,
        )
        payload.update(structural_fields)
        payload.update(
            _population_confidence_payload_fields(
                by_pocket=population_confidence_by_pocket,
                global_payload=population_confidence_global,
            )
        )
        result_path = agregador.persistir_payload_resultado(out_dir=out_dir, payload=payload)
        auditoria.registrar_artefato_escrito(tracer=contexto.tracer, caminho=result_path)
        if pose_output_path:
            auditoria.registrar_artefato_escrito(
                tracer=contexto.tracer,
                caminho=os.path.join(out_dir, pose_output_path),
            )
        candidate_pose_output_path = candidate_artifact.get("candidate_pose_output_path")
        if candidate_pose_output_path:
            auditoria.registrar_artefato_escrito(
                tracer=contexto.tracer,
                caminho=os.path.join(out_dir, str(candidate_pose_output_path)),
            )
        structural_metrics_output_path = structural_fields.get(
            "structural_metrics_output_path"
        )
        if structural_metrics_output_path:
            auditoria.registrar_artefato_escrito(
                tracer=contexto.tracer,
                caminho=os.path.join(out_dir, str(structural_metrics_output_path)),
            )

        logger.flush(out_dir)
        logger.flush_timeseries(out_dir, mode=contexto.search_space_mode)
        summary_path = agregador.escrever_summary_execucao(
            out_dir=out_dir,
            run_id=contexto.run_id,
            mode="single",
            receptor_path=contexto.receptor_path,
            peptide_path=contexto.peptide_path,
            search_space_mode=contexto.search_space_mode,
            runtime_sec=runtime_sec,
            search_time_sec=search_time_sec,
            pocketing_sec=pocketing_sec,
            scan_sec=scan_sec,
            total_pockets=contexto.total_pockets,
            selected_pockets=contexto.selected_pockets,
            best_score_cheap=best_score_cheap,
            best_score_expensive=best_score_expensive,
            best_pose_pocket_id=best_pose_pocket_id,
            config_resolved_subset=config_resolved_subset,
            records=logger.records,
            pockets=pockets_para_busca,
            scan_params=contexto.scan_params,
            scan_by_pocket=contexto.scan_results,
            selected_pocket_ids=[str(p.id) for p in pockets_para_busca],
            pose_output_path=pose_output_path,
            final_pose_available=final_pose_available,
            pose_role=pose_role,
            best_pose_meta=persisted_pose_meta,
            confidence_policy_metadata=contexto.confidence_policy_metadata,
            confidence_calibrator=getattr(cfg, "confidence_calibrator", None),
            rescoring_summary=rescoring_summary,
        )
        _inject_population_confidence_into_summary_file(
            summary_path=summary_path,
            by_pocket=population_confidence_by_pocket,
            global_payload=population_confidence_global,
        )
        inject_structural_metrics_into_json(summary_path, structural_fields)
        auditoria.registrar_artefato_escrito(tracer=contexto.tracer, caminho=summary_path)
        auditoria.registrar_execucao_finalizada(
            tracer=contexto.tracer,
            pocket_id=contexto.pocket_id,
            best_score_cheap=best_score_cheap,
            best_score_expensive=best_score_expensive,
        )
        return result, payload, logger
