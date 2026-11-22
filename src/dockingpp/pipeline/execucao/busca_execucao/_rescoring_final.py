"""Serviço privado responsável apenas pelo rescoring final de candidatos."""

from __future__ import annotations

from typing import Any

from dockingpp.domain.models import Pose
from dockingpp.pipeline.logging import RunLogger
from dockingpp.scoring.expensive import score_pose_expensive

from .rescoring_artefatos import (
    _best_pose_from_result,
    _confidence_rescoring_selection_enabled,
    _log_rescoring_skip,
    _population_controller_acted,
    _pose_confidence_priority_score,
    _pose_rank_id,
    _pose_sort_score,
    _rescoring_alignment_mode,
    _rescoring_enabled,
    _rescoring_scope,
    _rescoring_top_k,
    _select_rescoring_candidates,
)
from .serializacao_confianca import _safe_float_or_none, _to_json_object


class ServicoRescoringFinal:
    """Seleciona, reavalia e reordena candidatos sem realizar persistência."""

    def aplicar(
        self,
        *,
        cfg: Any,
        receptor: Any,
        peptide: Any,
        result: Any,
        logger: RunLogger,
    ) -> dict[str, Any]:
        best_pose = _best_pose_from_result(result)
        raw_candidates: list[Pose] = []
        metrics = getattr(result, "metrics", None)
        if isinstance(metrics, dict):
            metric_candidates = metrics.get("rescoring_candidates")
            if isinstance(metric_candidates, list):
                raw_candidates.extend(
                    pose
                    for pose in metric_candidates
                    if isinstance(pose, Pose)
                )
        if isinstance(best_pose, Pose):
            raw_candidates.append(best_pose)

        dedup: dict[tuple[Any, tuple[float, ...]], Pose] = {}
        for pose in raw_candidates:
            key = _pose_rank_id(pose)
            current = dedup.get(key)
            if current is None or _pose_sort_score(pose) > _pose_sort_score(current):
                dedup[key] = pose
        candidates = sorted(dedup.values(), key=_pose_sort_score, reverse=True)

        enabled = _rescoring_enabled(cfg)
        top_k = _rescoring_top_k(cfg)
        scope = _rescoring_scope(cfg)
        alignment_mode = _rescoring_alignment_mode(cfg)
        confidence_selection_requested = _confidence_rescoring_selection_enabled(cfg)
        confidence_controller_acted = _population_controller_acted(metrics)
        confidence_selection_enabled = bool(
            confidence_selection_requested and confidence_controller_acted
        )
        selection_score_name = (
            "confidence_rescoring_priority_score"
            if confidence_selection_enabled
            else "score_cheap"
        )
        variant = str(getattr(cfg, "final_score_variant", "vina_rescore") or "vina_rescore")
        selection_criterion = (
            f"{scope}:all_candidates_in_pool"
            if alignment_mode == "all_candidates"
            else f"{scope}:top_{top_k}_by_{selection_score_name}"
        )
        summary: dict[str, Any] = {
            "rescoring_enabled": bool(enabled),
            "rescoring_applied": False,
            "rescoring_top_k": int(top_k),
            "rescoring_scope": scope,
            "rescoring_alignment_mode": alignment_mode,
            "rescoring_selection_criterion": selection_criterion,
            "rescoring_selection_score_name": selection_score_name,
            "population_confidence_rescoring_requested": bool(confidence_selection_requested),
            "population_confidence_controller_acted": bool(confidence_controller_acted),
            "population_confidence_used_for_rescoring_selection": bool(confidence_selection_enabled),
            "population_confidence_rescoring_level": 3 if confidence_selection_enabled else 2,
            "rescoring_candidates_total": int(len(candidates)),
            "rescoring_selected_count": 0,
            "rescored_candidates_count": 0,
            "ranking_changed_after_rescoring": False,
            "final_score_variant": variant,
            "selected_candidates": [],
            "cheap_top5_candidates": [],
            "best_final_rescore": None,
            "rescoring_not_applied_reason": None,
        }

        if not enabled:
            summary["rescoring_not_applied_reason"] = "disabled_by_config"
            if isinstance(metrics, dict):
                metrics["final_rescoring"] = summary
            return summary
        if not candidates:
            summary["rescoring_not_applied_reason"] = "no_candidates_available"
            if isinstance(metrics, dict):
                metrics["final_rescoring"] = summary
            return summary

        before_ranked = sorted(
            candidates,
            key=lambda pose: _pose_confidence_priority_score(
                pose,
                cfg=cfg,
                enabled=confidence_selection_enabled,
            ),
            reverse=True,
        )
        before_rank_map = {id(pose): idx for idx, pose in enumerate(before_ranked, start=1)}
        selected_topk, skipped_by_topk = _select_rescoring_candidates(
            candidates=candidates,
            top_k=top_k,
            scope=scope,
            cfg=cfg,
            confidence_selection_enabled=confidence_selection_enabled,
        )
        selected = selected_topk if alignment_mode == "top_k" else list(before_ranked)
        skipped = skipped_by_topk if alignment_mode == "top_k" else []
        selected_topk_ids = {id(pose) for pose in selected_topk}
        selected_ids = {id(pose) for pose in selected}
        cheap_top5_rows: list[dict[str, Any]] = []
        for idx, pose in enumerate(before_ranked[:5], start=1):
            pose_meta = pose.meta if isinstance(pose.meta, dict) else {}
            cheap_top5_rows.append(
                {
                    "candidate_rank_before": int(idx),
                    "pocket_id": pose_meta.get("pocket_id"),
                    "generation": pose_meta.get("generation"),
                    "score_cheap": float(pose.score_cheap) if pose.score_cheap is not None else None,
                    "selection_score": _safe_float_or_none(
                        pose_meta.get("confidence_rescoring_priority_score")
                    ) if confidence_selection_enabled else (
                        float(pose.score_cheap) if pose.score_cheap is not None else None
                    ),
                    "confidence_rescoring_priority_delta": _safe_float_or_none(
                        pose_meta.get("confidence_rescoring_priority_delta")
                    ),
                    "selected_for_rescoring": bool(id(pose) in selected_ids),
                    "selected_by_cheap_topk_gate": bool(id(pose) in selected_topk_ids),
                }
            )
        for pose in skipped:
            pose_meta = pose.meta if isinstance(pose.meta, dict) else {}
            _log_rescoring_skip(
                logger,
                rank=int(before_rank_map.get(id(pose), -1)),
                pocket_id=None if not isinstance(pose_meta, dict) else str(pose_meta.get("pocket_id")),
            )
        if not skipped:
            # Keep metric key present for backward-compatible dashboards/tests.
            logger.log_metric(
                "final_rescore_skipped",
                0.0,
                step=0,
                extra={"reason": "no_candidate_skipped"},
            )
            logger.log_metric(
                "expensive_skipped",
                0.0,
                step=0,
                extra={"reason": "no_candidate_skipped"},
            )

        rescored_valid: list[Pose] = []
        selected_rows: list[dict[str, Any]] = []

        for pose in selected:
            pose_meta = pose.meta if isinstance(pose.meta, dict) else {}
            cheap_score = float(pose.score_cheap) if pose.score_cheap is not None else None
            final_rescore = score_pose_expensive(pose, receptor, peptide, cfg)
            if final_rescore is not None:
                pose.score_expensive = float(final_rescore)
                pose.final_rescore = float(final_rescore)
                pose.meta["final_score_variant"] = variant
                rescored_valid.append(pose)
            selected_rows.append(
                {
                    "candidate_rank_before": int(before_rank_map.get(id(pose), -1)),
                    "pocket_id": pose_meta.get("pocket_id"),
                    "generation": pose_meta.get("generation"),
                    "score_cheap": cheap_score,
                    "selection_score": _safe_float_or_none(
                        pose_meta.get("confidence_rescoring_priority_score")
                    ) if confidence_selection_enabled else cheap_score,
                    "selection_score_name": selection_score_name,
                    "confidence_rescoring_selection_applied": bool(
                        pose_meta.get("confidence_rescoring_selection_applied", False)
                    ),
                    "confidence_rescoring_priority_delta": _safe_float_or_none(
                        pose_meta.get("confidence_rescoring_priority_delta")
                    ),
                    "confidence_rescoring_priority_components": _to_json_object(
                        pose_meta.get("confidence_rescoring_priority_components")
                    ),
                    "confidence_rescoring_enabled_components": _to_json_object(
                        pose_meta.get("confidence_rescoring_enabled_components")
                    ),
                    "confidence_rescoring_component_weights": _to_json_object(
                        pose_meta.get("confidence_rescoring_component_weights")
                    ),
                    "selected_by_cheap_topk_gate": bool(id(pose) in selected_topk_ids),
                    "final_rescore": float(final_rescore) if final_rescore is not None else None,
                    "final_rescore_energy": (
                        float(pose_meta.get("final_rescore_energy"))
                        if pose_meta.get("final_rescore_energy") is not None
                        else None
                    ),
                    "vina_terms": _to_json_object(pose_meta.get("vina_terms")),
                    "pair_stats": _to_json_object(pose_meta.get("vina_pair_stats")),
                    "vina_rescore_debug": _to_json_object(pose_meta.get("vina_rescore_debug")),
                }
            )

        after_ranked = sorted(
            rescored_valid,
            key=lambda pose: float(pose.final_rescore if pose.final_rescore is not None else float("-inf")),
            reverse=True,
        )
        after_rank_map = {id(pose): idx for idx, pose in enumerate(after_ranked, start=1)}
        for row in selected_rows:
            candidate_rank_before = row.get("candidate_rank_before", -1)
            target_pose = next(
                (
                    pose
                    for pose in selected
                    if int(before_rank_map.get(id(pose), -1)) == int(candidate_rank_before)
                ),
                None,
            )
            row["candidate_rank_after_rescoring"] = (
                int(after_rank_map.get(id(target_pose), -1)) if target_pose is not None else -1
            )
        selected_rows_by_rank = {
            int(row.get("candidate_rank_before", -1)): row for row in selected_rows
        }
        for row in cheap_top5_rows:
            rank_before = int(row.get("candidate_rank_before", -1))
            selected_row = selected_rows_by_rank.get(rank_before)
            if selected_row is None:
                row["final_rescore"] = None
                row["final_rescore_energy"] = None
                row["candidate_rank_after_rescoring"] = None
                continue
            row["final_rescore"] = selected_row.get("final_rescore")
            row["final_rescore_energy"] = selected_row.get("final_rescore_energy")
            row["candidate_rank_after_rescoring"] = selected_row.get("candidate_rank_after_rescoring")

        best_final_rescore = None
        ranking_changed = False
        if after_ranked:
            winner = after_ranked[0]
            best_final_rescore = float(winner.final_rescore) if winner.final_rescore is not None else None
            if best_pose is not None:
                ranking_changed = _pose_rank_id(winner) != _pose_rank_id(best_pose)
            result.best_pose = winner

        summary.update(
            {
                "rescoring_applied": bool(after_ranked),
                "rescoring_selected_count": int(len(selected)),
                "rescored_candidates_count": int(len(rescored_valid)),
                "ranking_changed_after_rescoring": bool(ranking_changed),
                "selected_candidates": selected_rows,
                "cheap_top5_candidates": cheap_top5_rows,
                "best_final_rescore": best_final_rescore,
                "best_score_expensive": best_final_rescore,
                "rescoring_not_applied_reason": None if after_ranked else "no_valid_final_rescore",
            }
        )
        if isinstance(metrics, dict):
            metrics["final_rescoring"] = summary
        return summary
