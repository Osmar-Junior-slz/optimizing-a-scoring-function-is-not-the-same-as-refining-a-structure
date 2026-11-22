"""Seleção de candidatos, semântica de scores e artefatos de rescoring.

A separação mantém reunidas as regras auxiliares compartilhadas pela execução
única e pelo agregado reduzido. A direção dos scores e a ordenação permanecem
idênticas à implementação anterior.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from typing import Any, Mapping

import numpy as np

from dockingpp.domain.models import Pose
from dockingpp.pipeline.confianca.selection_priority import (
    score_confidence_selection_priority,
)
from dockingpp.pipeline.logging import RunLogger

from .serializacao_confianca import _to_json_compatible


def _best_pose_from_result(result: Any) -> Any | None:
    """Safely extracts `best_pose` from a run result-like object."""

    if result is None:
        return None
    return getattr(result, "best_pose", None)


def _best_pose_meta(best_pose: Any) -> dict[str, Any]:
    """Returns normalized pose metadata."""

    meta = getattr(best_pose, "meta", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _best_pose_score(best_pose: Any, field: str) -> float | None:
    """Reads a score field from pose and normalizes to float."""

    value = getattr(best_pose, field, None)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None




def _score_semantics_fields(
    *,
    triage_score: float | None,
    rescore_proxy: float | None,
    final_rescore: float | None = None,
    rescoring_applied: bool = False,
) -> dict[str, Any]:
    """Campos canonicos para semantica explicita dos scores no payload."""

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



def _scientific_state_fields() -> dict[str, Any]:
    """Estado cientifico explicito para evitar claims indevidos."""

    return {
        "scientific_score_available": False,
        "scientific_score_name": None,
        "triage_pipeline": True,
        "scientific_readiness": "triage_only",
    }


def _reproducibility_fields(*, final_pose_available: bool) -> dict[str, Any]:
    """Metadados minimos de reprodutibilidade para artefatos atuais."""

    return {
        "pose_artifact_available": bool(final_pose_available),
        "score_semantics_available": True,
        "reproducibility_level": "basic_artifacts" if final_pose_available else "score_semantics_only",
    }


def _rescoring_enabled(cfg: Any) -> bool:
    explicit = getattr(cfg, "rescoring_enabled", None)
    legacy_every = int(getattr(cfg, "expensive_every", 0) or 0)
    legacy_topk = getattr(cfg, "expensive_topk", None)
    legacy_topk_enabled = legacy_topk is not None and int(legacy_topk) > 0
    return bool(explicit) or legacy_every > 0 or legacy_topk_enabled


def _rescoring_top_k(cfg: Any) -> int:
    topk = getattr(cfg, "rescoring_top_k", None)
    if topk is None:
        topk = getattr(cfg, "expensive_topk", None)
    if topk is None:
        topk = 1
    try:
        return max(1, int(topk))
    except (TypeError, ValueError):
        return 1


def _rescoring_scope(cfg: Any) -> str:
    raw = str(getattr(cfg, "rescoring_scope", "global") or "global").strip().lower()
    if raw not in {"global", "per_pocket"}:
        return "global"
    return raw


def _rescoring_alignment_mode(cfg: Any) -> str:
    """
    Defines how candidate gating works before final rescoring.

    - all_candidates: rescore every candidate available in pool (default).
    - top_k: legacy behavior, rescore only top-k by cheap score.
    """
    raw = str(getattr(cfg, "rescoring_alignment_mode", "all_candidates") or "all_candidates")
    mode = raw.strip().lower()
    if mode not in {"all_candidates", "top_k"}:
        return "all_candidates"
    return mode


def _pose_sort_score(pose: Pose) -> float:
    value = getattr(pose, "score_cheap", None)
    return float(value if value is not None else float("-inf"))


def _cfg_mapping_value(cfg: Any, key: str, default: Any = None) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _confidence_rescoring_selection_enabled(cfg: Any) -> bool:
    policy = _cfg_mapping_value(cfg, "confidence_policy", {}) or {}
    if not isinstance(policy, Mapping):
        return False
    enabled = bool(policy.get("enabled", False))
    mode = str(policy.get("mode", "disabled") or "disabled").strip().lower()
    return bool(enabled and mode in {"assist_selection", "assist_all"})


def _population_controller_acted(metrics: Any) -> bool:
    if not isinstance(metrics, Mapping):
        return False
    rows = metrics.get("population_confidence_by_pocket")
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            if int(row.get("confidence_population_orientation_triggers", 0) or 0) > 0:
                return True
            if int(row.get("confidence_population_curation_archive_injections", 0) or 0) > 0:
                return True
        return False
    return bool(
        int(metrics.get("confidence_population_orientation_triggers", 0) or 0) > 0
        or int(metrics.get("confidence_population_curation_archive_injections", 0) or 0) > 0
    )


def _pose_confidence_priority_score(pose: Pose, *, cfg: Any | None, enabled: bool) -> float:
    cheap_score = _pose_sort_score(pose)
    if not enabled:
        return cheap_score

    meta = pose.meta if isinstance(getattr(pose, "meta", {}), dict) else {}
    priority = score_confidence_selection_priority(
        score_cheap=cheap_score,
        meta=meta,
        cfg=cfg,
    )

    meta["confidence_rescoring_selection_applied"] = True
    meta["confidence_rescoring_priority_score"] = priority.score
    meta["confidence_rescoring_priority_delta"] = priority.delta
    meta["confidence_rescoring_priority_components"] = priority.components
    meta["confidence_rescoring_enabled_components"] = priority.enabled_components
    meta["confidence_rescoring_component_weights"] = priority.weights
    pose.meta = meta
    return priority.score


def _pose_rank_id(pose: Pose) -> tuple[Any, tuple[float, ...]]:
    meta = getattr(pose, "meta", {}) if isinstance(getattr(pose, "meta", {}), dict) else {}
    coords = np.asarray(getattr(pose, "coords", []), dtype=float).reshape(-1)
    return (
        meta.get("pocket_id"),
        tuple(np.round(coords, decimals=8).tolist()),
    )


def _select_rescoring_candidates(
    *,
    candidates: list[Pose],
    top_k: int,
    scope: str,
    cfg: Any | None = None,
    confidence_selection_enabled: bool = False,
) -> tuple[list[Pose], list[Pose]]:
    ranked = sorted(
        candidates,
        key=lambda pose: _pose_confidence_priority_score(
            pose,
            cfg=cfg,
            enabled=confidence_selection_enabled,
        ),
        reverse=True,
    )
    if scope == "global":
        selected = ranked[:top_k]
        skipped = ranked[top_k:]
        return selected, skipped

    by_pocket: dict[str, list[Pose]] = defaultdict(list)
    for pose in ranked:
        meta = getattr(pose, "meta", {}) if isinstance(getattr(pose, "meta", {}), dict) else {}
        pocket_id = str(meta.get("pocket_id", "unknown"))
        by_pocket[pocket_id].append(pose)
    selected: list[Pose] = []
    for pocket_id in sorted(by_pocket):
        selected.extend(by_pocket[pocket_id][:top_k])
    selected_set = {id(item) for item in selected}
    skipped = [item for item in ranked if id(item) not in selected_set]
    return selected, skipped


def _log_rescoring_skip(logger: RunLogger, *, rank: int, pocket_id: str | None) -> None:
    extra = {"reason": "rank_outside_topk", "rescoring_rank": int(rank), "pocket_id": pocket_id}
    logger.log_metric("final_rescore_skipped", 1.0, step=0, extra=extra)
    # Backward-compatible metric naming.
    logger.log_metric("expensive_skipped", 1.0, step=0, extra=extra)
def _persist_best_pose(
    *,
    out_dir: str,
    best_pose: Any,
    engine_name: str | None,
    search_space_mode: str,
) -> dict[str, Any]:
    """Persists the best computed pose as an auditable JSON artifact."""

    pose_role = "best_computed_pose"
    empty = {
        "pose_output_path": None,
        "final_pose_available": False,
        "pose_role": pose_role,
        "best_pose_meta": None,
    }
    if best_pose is None:
        return empty

    coords = getattr(best_pose, "coords", None)
    if coords is None:
        return empty

    meta = _best_pose_meta(best_pose)
    rel_path = os.path.join("artifacts", "best_pose.json").replace("\\", "/")
    abs_path = os.path.join(out_dir, rel_path)
    payload = {
        "schema_version": "1.0",
        "pose_role": pose_role,
        "search_space_mode": str(search_space_mode),
        "engine_name": None if engine_name is None else str(engine_name),
        "coords": _to_json_compatible(coords),
        "score_cheap": _best_pose_score(best_pose, "score_cheap"),
        "score_expensive": _best_pose_score(best_pose, "score_expensive"),
        "final_rescore": _best_pose_score(best_pose, "final_rescore")
        or _best_pose_score(best_pose, "score_expensive"),
        "pocket_id": meta.get("pocket_id"),
        "generation": meta.get("generation"),
        "meta": _to_json_compatible(meta),
    }
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
    except OSError:
        return empty

    return {
        "pose_output_path": rel_path,
        "final_pose_available": True,
        "pose_role": pose_role,
        "best_pose_meta": _to_json_compatible(meta),
    }


def _persist_rescored_candidates(
    *,
    out_dir: str,
    result: Any,
    engine_name: str | None,
    search_space_mode: str,
) -> dict[str, Any]:
    """Persist every valid rescored candidate for Top-k structural evaluation."""

    metrics = getattr(result, "metrics", None)
    candidates = metrics.get("rescoring_candidates") if isinstance(metrics, dict) else None
    if not isinstance(candidates, list):
        return {"candidate_pose_output_path": None, "candidate_pose_count": 0}
    valid = [
        pose
        for pose in candidates
        if isinstance(pose, Pose)
        and getattr(pose, "coords", None) is not None
        and getattr(pose, "final_rescore", None) is not None
    ]
    valid.sort(key=lambda pose: float(pose.final_rescore), reverse=True)
    if not valid:
        return {"candidate_pose_output_path": None, "candidate_pose_count": 0}

    rel_path = os.path.join("artifacts", "rescored_candidates.json").replace("\\", "/")
    abs_path = os.path.join(out_dir, rel_path)
    payload = {
        "schema_version": "1.0",
        "pose_role": "rescored_candidate_set",
        "search_space_mode": str(search_space_mode),
        "engine_name": None if engine_name is None else str(engine_name),
        "candidate_count": len(valid),
        "candidates": [
            {
                "rank": index,
                "coords": _to_json_compatible(pose.coords),
                "score_cheap": _best_pose_score(pose, "score_cheap"),
                "final_rescore": _best_pose_score(pose, "final_rescore"),
                "meta": _to_json_compatible(_best_pose_meta(pose)),
            }
            for index, pose in enumerate(valid, start=1)
        ],
    }
    try:
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
    except OSError:
        return {"candidate_pose_output_path": None, "candidate_pose_count": 0}
    return {
        "candidate_pose_output_path": rel_path,
        "candidate_pose_count": len(valid),
    }


def _write_pocket_trace_artifacts(
    tracer: Any,
    *,
    pocket_id: str,
    payload: dict[str, Any],
) -> None:
    """Escreve traÃ§o mÃ­nimo por bolsÃ£o para auditoria local nÃ£o-vazia."""

    if tracer is None or not bool(getattr(tracer, "enabled", False)):
        return
    debug_dir = getattr(tracer, "debug_dir", None)
    run_id = getattr(tracer, "run_id", None)
    if not isinstance(debug_dir, str) or not debug_dir:
        return
    event = {
        "schema_version": "1.0",
        "run_id": run_id,
        "event_id": f"pocket-summary-{pocket_id}",
        "parent_event_id": None,
        "stage": "summary",
        "substage": "pocket",
        "event_type": "pocket_run_summary",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "monotonic_ms": None,
        "level": "TRACE",
        "module": "pipeline.execucao.execucao_busca",
        "function": "ExecutorBuscaPipeline.executar",
        "search_space_mode": "reduced",
        "pocket_id": pocket_id,
        "engine": None,
        "payload": _to_json_compatible(payload),
    }
    for rel in (
        f"pockets/{pocket_id}/trace.jsonl",
        f"pockets/{pocket_id}/decision_trace.jsonl",
    ):
        abs_path = os.path.join(debug_dir, rel)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        mark_file = getattr(tracer, "_mark_file", None)
        if callable(mark_file):
            mark_file(rel)
