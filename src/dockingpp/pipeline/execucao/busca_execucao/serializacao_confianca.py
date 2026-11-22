"""Normalização e serialização dos metadados de confiança.

As funções deste módulo não tomam decisões de busca. Elas apenas convertem,
agregam e persistem o formato de auditoria já produzido pelo pipeline.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from dockingpp.pipeline.confianca.policy import merge_confidence_policy_metadata


def _to_json_compatible(value: Any) -> Any:
    """Best-effort conversion of values to JSON-compatible primitives."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_json_compatible(tolist())
        except Exception:
            return str(value)
    return str(value)


def _to_json_object(value: Any) -> dict[str, Any] | None:
    converted = _to_json_compatible(value)
    return converted if isinstance(converted, dict) else None


def _to_json_object_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    converted: list[dict[str, Any]] = []
    for item in value:
        obj = _to_json_object(item)
        if obj is not None:
            converted.append(obj)
    return converted


def _extract_population_confidence_from_result(
    result: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, dict):
        return [], {}
    by_pocket = _to_json_object_list(metrics.get("population_confidence_by_pocket"))
    global_payload = _to_json_object(metrics.get("population_confidence_global")) or {}
    return by_pocket, global_payload


def _result_metric(result: Any, key: str, default: Any = None) -> Any:
    metrics = getattr(result, "metrics", None)
    if not isinstance(metrics, dict):
        return default
    return metrics.get(key, default)


def _normalize_population_confidence_for_pocket(
    *,
    pocket_id: str | None,
    by_pocket: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    normalized: list[dict[str, Any]] = []
    for item in by_pocket:
        row = dict(item)
        if pocket_id is not None:
            row["pocket_id"] = str(pocket_id)
        normalized.append(row)
    if not normalized:
        return normalized, None
    first = dict(normalized[0])
    if pocket_id is not None:
        first["pocket_id"] = str(pocket_id)
    return normalized, first


def _safe_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_string_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _combine_confidence_policy_metadata(
    *,
    selection_policy: Mapping[str, Any] | None,
    reduced_policy: Mapping[str, Any] | None,
) -> dict[str, Any]:
    selection_payload = merge_confidence_policy_metadata(selection_policy)
    reduced_payload = merge_confidence_policy_metadata(reduced_policy)

    combined = dict(reduced_payload)
    if (
        not bool(combined.get("confidence_policy_used", False))
        and bool(selection_payload.get("confidence_policy_used", False))
    ):
        combined["confidence_policy_used"] = True
        combined["confidence_policy_mode"] = selection_payload.get("confidence_policy_mode")

    combined["confidence_assisted_selection"] = bool(
        selection_payload.get("confidence_assisted_selection", False)
        or reduced_payload.get("confidence_assisted_selection", False)
    )
    combined["confidence_assisted_fallback"] = bool(
        reduced_payload.get("confidence_assisted_fallback", False)
    )
    combined["confidence_fallback_recommended"] = bool(
        selection_payload.get("confidence_fallback_recommended", False)
        or reduced_payload.get("confidence_fallback_recommended", False)
    )
    combined["confidence_fallback_apply_requested"] = bool(
        reduced_payload.get("confidence_fallback_apply_requested", False)
    )

    reduced_effect = str(reduced_payload.get("confidence_policy_effect") or "none")
    selection_effect = str(selection_payload.get("confidence_policy_effect") or "none")
    if (
        reduced_effect == "none"
        and selection_effect not in {"none", "fallback_requested"}
    ):
        combined["confidence_policy_effect"] = selection_effect

    if (
        str(combined.get("confidence_decision") or "neutral") == "neutral"
        and selection_payload.get("confidence_decision") not in {None, "neutral"}
    ):
        combined["confidence_decision"] = selection_payload.get("confidence_decision")
    if (
        str(combined.get("confidence_recommendation") or "keep_current_flow")
        == "keep_current_flow"
        and selection_payload.get("confidence_recommendation") is not None
    ):
        combined["confidence_recommendation"] = selection_payload.get("confidence_recommendation")

    combined["confidence_decision_reasons"] = _coerce_string_list(
        [
            *_coerce_string_list(selection_payload.get("confidence_decision_reasons")),
            *_coerce_string_list(reduced_payload.get("confidence_decision_reasons")),
        ]
    )
    combined["confidence_warning_flags"] = _coerce_string_list(
        [
            *_coerce_string_list(selection_payload.get("confidence_warning_flags")),
            *_coerce_string_list(reduced_payload.get("confidence_warning_flags")),
        ]
    )

    combined_details: dict[str, Any] = {}
    selection_details = selection_payload.get("confidence_policy_details")
    if isinstance(selection_details, Mapping):
        combined_details["selection_stage"] = {
            "mode": selection_payload.get("confidence_policy_mode"),
            "effect": selection_payload.get("confidence_policy_effect"),
            "high_uncertainty": bool(selection_details.get("high_uncertainty", False)),
            "uncertainty_reasons": _coerce_string_list(
                selection_details.get("uncertainty_reasons")
            ),
            "applied_pruned_pockets": _coerce_string_list(
                selection_details.get("applied_pruned_pockets")
            ),
            "fallback_signal_ignored_until_post_reduced": bool(
                selection_payload.get("confidence_assisted_fallback", False)
            ),
        }
    reduced_details = reduced_payload.get("confidence_policy_details")
    if isinstance(reduced_details, Mapping):
        combined_details.update(dict(reduced_details))
    combined["confidence_policy_details"] = combined_details

    return merge_confidence_policy_metadata(combined)


def _build_population_confidence_global_from_pocket_summaries(
    *,
    pocket_summaries: list[dict[str, Any]],
    per_pocket_results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not pocket_summaries:
        return {}
    ranked = sorted(
        pocket_summaries,
        key=lambda row: _safe_float_or_none(row.get("best_score_global")) or float("-inf"),
        reverse=True,
    )
    ranked_rows = [
        {
            "pocket_id": row.get("pocket_id"),
            "best_score_global": _safe_float_or_none(row.get("best_score_global")),
        }
        for row in ranked
    ]
    best_scores = [item["best_score_global"] for item in ranked_rows if item["best_score_global"] is not None]
    best_vs_second_best_gap = None
    if len(best_scores) >= 2:
        best_vs_second_best_gap = float(best_scores[0] - best_scores[1])

    candidate_counts: dict[str, int] = {}
    total_candidates = 0
    for pocket_result in per_pocket_results:
        pocket_id = str(pocket_result.get("pocket_id", "unknown"))
        candidates = pocket_result.get("rescoring_selected_candidates")
        if not isinstance(candidates, list):
            continue
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            candidate_pocket_id = str(candidate.get("pocket_id", pocket_id))
            candidate_counts[candidate_pocket_id] = candidate_counts.get(candidate_pocket_id, 0) + 1
            total_candidates += 1
    if not candidate_counts:
        for pocket_result in per_pocket_results:
            pocket_id = str(pocket_result.get("pocket_id", "unknown"))
            count = int(pocket_result.get("rescoring_selected_count", 0) or 0)
            if count <= 0:
                continue
            candidate_counts[pocket_id] = candidate_counts.get(pocket_id, 0) + count
            total_candidates += count

    dominant_pocket_id = None
    dominant_fraction = 0.0
    if candidate_counts:
        dominant_pocket_id = sorted(
            candidate_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0][0]
        if total_candidates > 0:
            dominant_fraction = float(candidate_counts[dominant_pocket_id]) / float(total_candidates)
    elif ranked_rows:
        dominant_pocket_id = ranked_rows[0].get("pocket_id")

    stable_count = sum(1 for row in pocket_summaries if bool(row.get("converged_stably", False)))
    uncertain_count = sum(
        1
        for row in pocket_summaries
        if isinstance(row.get("uncertainty_flags"), list) and bool(row.get("uncertainty_flags"))
    )
    return {
        "n_pockets": int(len(pocket_summaries)),
        "dominant_pocket_id": dominant_pocket_id,
        "dominant_pocket_candidate_fraction": float(dominant_fraction),
        "best_vs_second_best_gap": best_vs_second_best_gap,
        "stable_pockets_count": int(stable_count),
        "uncertain_pockets_count": int(uncertain_count),
        "pockets_ranked_by_best_score": ranked_rows,
    }


def _collect_population_confidence_by_pocket(
    per_pocket_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered: list[dict[str, Any]] = []
    seen_pockets: set[str] = set()
    for pocket_result in per_pocket_results:
        pocket_id = str(pocket_result.get("pocket_id", ""))
        pocket_summary = pocket_result.get("population_confidence_pocket")
        if not isinstance(pocket_summary, dict):
            rows = pocket_result.get("population_confidence_by_pocket")
            if isinstance(rows, list) and rows and isinstance(rows[0], dict):
                pocket_summary = rows[0]
        if not isinstance(pocket_summary, dict):
            continue
        row = dict(pocket_summary)
        if pocket_id:
            row["pocket_id"] = pocket_id
        normalized_id = str(row.get("pocket_id", ""))
        if normalized_id in seen_pockets:
            continue
        seen_pockets.add(normalized_id)
        ordered.append(row)
    return ordered


def _population_confidence_payload_fields(
    *,
    by_pocket: list[dict[str, Any]],
    global_payload: dict[str, Any],
) -> dict[str, Any]:
    first_pocket = by_pocket[0] if by_pocket and isinstance(by_pocket[0], dict) else {}
    search_control_enabled = any(
        bool(row.get("confidence_search_control_enabled", False))
        for row in by_pocket
        if isinstance(row, dict)
    )
    archive_injections = sum(
        int(row.get("confidence_population_curation_archive_injections", 0) or 0)
        for row in by_pocket
        if isinstance(row, dict)
    )
    archive_comparisons = sum(
        int(row.get("confidence_population_curation_archive_comparisons", 0) or 0)
        for row in by_pocket
        if isinstance(row, dict)
    )
    signal_checks = sum(
        int(row.get("confidence_population_signal_checks", 0) or 0)
        for row in by_pocket
        if isinstance(row, dict)
    )
    orientation_gate_hits = sum(
        int(row.get("confidence_population_orientation_gate_hits", 0) or 0)
        for row in by_pocket
        if isinstance(row, dict)
    )
    orientation_checks = sum(
        int(row.get("confidence_population_orientation_checks", 0) or 0)
        for row in by_pocket
        if isinstance(row, dict)
    )
    orientation_triggers = sum(
        int(row.get("confidence_population_orientation_triggers", 0) or 0)
        for row in by_pocket
        if isinstance(row, dict)
    )
    curation_roles = [
        str(row.get("confidence_population_curation_role"))
        for row in by_pocket
        if isinstance(row, dict) and row.get("confidence_population_curation_role")
    ]

    return {
        "population_confidence_role": "auxiliary_audit_signal",
        "population_confidence_not_scientific_score": True,
        "population_confidence_used_for_final_scoring": False,
        "population_confidence_by_pocket": by_pocket,
        "population_confidence_global": global_payload,
        "confidence_search_control_enabled": bool(search_control_enabled),
        "confidence_search_early_stop_enabled": bool(
            first_pocket.get("confidence_search_early_stop_enabled", False)
        ),
        "confidence_search_early_stop_applied": bool(
            first_pocket.get("confidence_search_early_stop_applied", False)
        ),
        "confidence_search_stop_generation": first_pocket.get(
            "confidence_search_stop_generation"
        ),
        "confidence_search_stop_reasons": first_pocket.get(
            "confidence_search_stop_reasons", []
        ),
        "confidence_search_stop_policy": first_pocket.get(
            "confidence_search_stop_policy"
        ),
        "confidence_search_stop_conditions": first_pocket.get(
            "confidence_search_stop_conditions", {}
        ),
        "confidence_search_stop_blocking_reasons": first_pocket.get(
            "confidence_search_stop_blocking_reasons", []
        ),
        "confidence_search_stop_recent_improvement": first_pocket.get(
            "confidence_search_stop_recent_improvement"
        ),
        "confidence_search_stop_plateau_window": first_pocket.get(
            "confidence_search_stop_plateau_window"
        ),
        "generations_saved_by_confidence_stop": int(
            sum(
                int(row.get("generations_saved_by_confidence_stop", 0) or 0)
                for row in by_pocket
                if isinstance(row, dict)
            )
        ),
        "confidence_population_curation_role": (
            "mixed" if len(set(curation_roles)) > 1 else (curation_roles[0] if curation_roles else None)
        ),
        "confidence_population_orientation_policy": first_pocket.get(
            "confidence_population_orientation_policy"
        ),
        "confidence_population_signal_checks": int(signal_checks),
        "confidence_population_orientation_gate_hits": int(orientation_gate_hits),
        "confidence_population_orientation_gate_hit_fraction": (
            float(orientation_gate_hits) / float(signal_checks) if signal_checks else 0.0
        ),
        "confidence_population_signal_check_interval": first_pocket.get(
            "confidence_population_signal_check_interval"
        ),
        "confidence_population_orientation_checks": int(orientation_checks),
        "confidence_population_orientation_triggers": int(orientation_triggers),
        "confidence_population_orientation_trigger_fraction": (
            float(orientation_triggers) / float(orientation_checks) if orientation_checks else 0.0
        ),
        "confidence_population_curation_archive_injections": int(archive_injections),
        "confidence_population_curation_archive_comparisons": int(archive_comparisons),
    }


def _inject_population_confidence_into_summary_file(
    *,
    summary_path: str,
    by_pocket: list[dict[str, Any]],
    global_payload: dict[str, Any],
) -> None:
    try:
        with open(summary_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(payload, dict):
        return
    payload.update(
        _population_confidence_payload_fields(
            by_pocket=by_pocket,
            global_payload=global_payload,
        )
    )
    try:
        with open(summary_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
    except OSError:
        return
