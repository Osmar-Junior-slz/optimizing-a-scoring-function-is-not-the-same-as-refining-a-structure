"""Extracao de features de confianca (pre-calibracao) para artefatos do pipeline."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from dockingpp.pipeline.confianca.calibration import (
    ConfidenceCalibrator,
    build_confidence_calibration_metadata,
)


CONFIDENCE_FEATURES_VERSION = "1.3"

# Frozen candidate profile for the first statistical + evidence-graph model.
# Every value is available after REDUCED and before FULL/native evaluation.
PRESERVATION_MECHANISTIC_GRAPH_V5_PROFILE = "preservation_mechanistic_graph_v5"
PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES = (
    "scan_overview.scan_score_gap_top1_top2_per_peptide_atom",
    "run.popconf_best_pocket_viable_fraction_mean",
    "run.popconf_best_pocket_elite_coords_dispersion_last",
    "run.popconf_best_pocket_uncertainty_count",
    "best_pocket.features.energy_q20_per_peptide_atom",
)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_confidence_feature_path(
    confidence_features: Mapping[str, Any], path: str
) -> Any:
    """Resolve one dotted confidence feature path without implicit defaults."""

    current: Any = confidence_features
    for part in str(path).split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def extract_mechanistic_graph_v5_features(
    confidence_features: Mapping[str, Any],
) -> tuple[dict[str, float], list[str]]:
    """Return finite v5 inputs and the paths that cannot support a decision.

    Missing, non-numeric, infinite, and NaN values are reported as missing.  No
    imputation, clipping, or fallback value is applied here.
    """

    resolved: dict[str, float] = {}
    unavailable: list[str] = []
    for path in PRESERVATION_MECHANISTIC_GRAPH_V5_FEATURES:
        raw = resolve_confidence_feature_path(confidence_features, path)
        value = _safe_float(raw)
        if value is None or not math.isfinite(value):
            unavailable.append(path)
            continue
        resolved[path] = float(value)
    return resolved, unavailable


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _safe_positive_int(value: Any) -> int | None:
    parsed = _safe_int(value)
    return parsed if parsed is not None and parsed > 0 else None


def _per_size(value: Any, size: int | None) -> float | None:
    parsed = _safe_float(value)
    if parsed is None or size is None or size < 1:
        return None
    return float(parsed) / float(size)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / float(len(values))


def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return 0.0 if values else None
    mu = _mean(values)
    if mu is None:
        return None
    var = sum((v - mu) ** 2 for v in values) / float(len(values))
    return var**0.5


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if q <= 0.0:
        return min(values)
    if q >= 1.0:
        return max(values)
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def _coerce_ids(ids: Sequence[Any] | None) -> list[str]:
    if not ids:
        return []
    out: list[str] = []
    for value in ids:
        if value is None:
            continue
        out.append(str(value))
    return out


def _scan_score(metrics: Mapping[str, Any]) -> float | None:
    primary = _safe_float(metrics.get("pocket_scan_score"))
    if primary is not None:
        return primary
    return _safe_float(metrics.get("scan_score"))


def _scan_clash(metrics: Mapping[str, Any]) -> float:
    selector_mode = str(metrics.get("selector_mode") or "legacy").strip().lower()
    if selector_mode == "geom_kdtree":
        primary = _safe_float(metrics.get("severe_clash_fraction"))
        if primary is not None:
            return primary
        secondary = _safe_float(metrics.get("clash_ratio_best"))
        return secondary if secondary is not None else 1.0
    primary = _safe_float(metrics.get("clash_ratio_best"))
    if primary is not None:
        return primary
    secondary = _safe_float(metrics.get("severe_clash_fraction"))
    return secondary if secondary is not None else 1.0


def _scan_rank_sort_key(
    metrics: Mapping[str, Any],
    index: int,
) -> tuple[float, float, float, float, float, float, float, int]:
    score = _scan_score(metrics)
    feasible = _safe_float(metrics.get("feasible_fraction")) or 0.0
    contact = _safe_float(metrics.get("contact_fraction_mean")) or 0.0
    clash = _scan_clash(metrics)
    energy_q20 = _safe_float(metrics.get("energy_q20"))
    n_samples = _safe_float(metrics.get("n_samples")) or 0.0
    viable_bucket = 1.0 if feasible > 0.0 else 0.0
    return (
        -viable_bucket,
        -contact,
        -feasible,
        clash,
        energy_q20 if energy_q20 is not None else float("inf"),
        -(score if score is not None else float("-inf")),
        -n_samples,
        int(index),
    )


def _scan_row(pocket_id: str, metrics: Mapping[str, Any]) -> dict[str, Any]:
    metric_keys = [
        "scan_score",
        "pocket_scan_score",
        "robust_score",
        "energy_q20",
        "energy_mean",
        "energy_std",
        "feasible_fraction",
        "feasible_samples",
        "feasible_fraction_ci95_low",
        "feasible_fraction_ci95_high",
        "n_samples",
        "contact_fraction_mean",
        "severe_clash_fraction",
        "severe_clash_samples",
        "severe_clash_fraction_ci95_low",
        "severe_clash_fraction_ci95_high",
        "moderate_clash_mean",
        "mean_exposure_penalty",
        "stability_penalty",
    ]
    row: dict[str, Any] = {"pocket_id": str(pocket_id)}
    available: list[str] = []
    missing: list[str] = []
    for key in metric_keys:
        value = _safe_float(metrics.get(key))
        row[key] = value
        if value is None:
            missing.append(key)
        else:
            available.append(key)
    selector_mode = metrics.get("selector_mode_used", metrics.get("selector_mode"))
    if selector_mode is not None:
        row["selector_mode"] = str(selector_mode)
    row["available_metrics"] = available
    row["missing_metrics"] = missing
    return row


def _build_scan_features(
    scan_by_pocket: Mapping[str, Mapping[str, Any]] | None,
    pocket_order: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, int], list[str]]:
    scan_table: dict[str, Mapping[str, Any]] = {}
    if isinstance(scan_by_pocket, Mapping):
        scan_table = {
            str(key): value
            for key, value in scan_by_pocket.items()
            if isinstance(value, Mapping)
        }

    ordered_ids = _coerce_ids(pocket_order)
    if not ordered_ids:
        ordered_ids = sorted(scan_table.keys())
    else:
        seen = set(ordered_ids)
        for pocket_id in sorted(scan_table.keys()):
            if pocket_id not in seen:
                ordered_ids.append(pocket_id)

    per_pocket_scan: list[dict[str, Any]] = []
    for pocket_id in ordered_ids:
        metrics = scan_table.get(pocket_id, {})
        per_pocket_scan.append(_scan_row(pocket_id, metrics))

    ranked = sorted(
        enumerate(per_pocket_scan),
        key=lambda item: _scan_rank_sort_key(item[1], item[0]),
    )
    rank_by_pocket: dict[str, int] = {}
    for idx, (_original_idx, row) in enumerate(ranked, start=1):
        if _scan_score(row) is None:
            continue
        rank_by_pocket[str(row["pocket_id"])] = idx
        row["selector_rank"] = idx

    scan_scores = [v for v in (_scan_score(row) for row in per_pocket_scan) if v is not None]
    feasible_values = [
        v for v in (_safe_float(row.get("feasible_fraction")) for row in per_pocket_scan) if v is not None
    ]
    n_samples_values = [
        v for v in (_safe_float(row.get("n_samples")) for row in per_pocket_scan) if v is not None
    ]

    sorted_scan_scores = sorted(scan_scores, reverse=True)
    top_1 = sorted_scan_scores[0] if sorted_scan_scores else None
    top_2 = sorted_scan_scores[1] if len(sorted_scan_scores) > 1 else None

    selector_ranked_rows = [
        row
        for _original_idx, row in ranked
        if _scan_score(row) is not None
    ]
    selector_top_1 = selector_ranked_rows[0] if selector_ranked_rows else None
    selector_top_2 = selector_ranked_rows[1] if len(selector_ranked_rows) > 1 else None
    selector_top_1_feasible = (
        _safe_float(selector_top_1.get("feasible_fraction"))
        if selector_top_1 is not None
        else None
    )
    selector_top_2_feasible = (
        _safe_float(selector_top_2.get("feasible_fraction"))
        if selector_top_2 is not None
        else None
    )
    selector_top_1_ci_low = (
        _safe_float(selector_top_1.get("feasible_fraction_ci95_low"))
        if selector_top_1 is not None
        else None
    )
    selector_top_1_ci_high = (
        _safe_float(selector_top_1.get("feasible_fraction_ci95_high"))
        if selector_top_1 is not None
        else None
    )
    selector_top_2_ci_low = (
        _safe_float(selector_top_2.get("feasible_fraction_ci95_low"))
        if selector_top_2 is not None
        else None
    )
    selector_top_2_ci_high = (
        _safe_float(selector_top_2.get("feasible_fraction_ci95_high"))
        if selector_top_2 is not None
        else None
    )
    selector_feasible_ci_overlap = None
    if None not in {
        selector_top_1_ci_low,
        selector_top_1_ci_high,
        selector_top_2_ci_low,
        selector_top_2_ci_high,
    }:
        selector_feasible_ci_overlap = bool(
            max(float(selector_top_1_ci_low), float(selector_top_2_ci_low))
            <= min(float(selector_top_1_ci_high), float(selector_top_2_ci_high))
        )

    missing_counts: dict[str, int] = {}
    for row in per_pocket_scan:
        for key in row.get("missing_metrics", []):
            missing_counts[key] = missing_counts.get(key, 0) + 1

    summary = {
        "scan_table_available": bool(scan_table),
        "scanned_pockets_count": len(scan_table),
        "scan_score_best": top_1,
        "scan_score_second_best": top_2,
        "scan_score_gap_top1_top2": (top_1 - top_2)
        if top_1 is not None and top_2 is not None
        else None,
        "scan_score_mean": _mean(scan_scores),
        "scan_score_std": _std(scan_scores),
        "scan_score_iqr": (
            (_quantile(scan_scores, 0.75) - _quantile(scan_scores, 0.25))
            if scan_scores
            else None
        ),
        "feasible_fraction_mean": _mean(feasible_values),
        "feasible_fraction_std": _std(feasible_values),
        "selector_top1_pocket_id": (
            str(selector_top_1.get("pocket_id")) if selector_top_1 is not None else None
        ),
        "selector_top2_pocket_id": (
            str(selector_top_2.get("pocket_id")) if selector_top_2 is not None else None
        ),
        "selector_feasible_fraction_gap_top1_top2": (
            float(selector_top_1_feasible - selector_top_2_feasible)
            if selector_top_1_feasible is not None and selector_top_2_feasible is not None
            else None
        ),
        "selector_top1_feasible_fraction_ci95_low": selector_top_1_ci_low,
        "selector_top1_feasible_fraction_ci95_high": selector_top_1_ci_high,
        "selector_top2_feasible_fraction_ci95_low": selector_top_2_ci_low,
        "selector_top2_feasible_fraction_ci95_high": selector_top_2_ci_high,
        "selector_feasible_fraction_ci95_overlap_top1_top2": selector_feasible_ci_overlap,
        "selector_top1_feasible_fraction_ci95_width": (
            float(selector_top_1_ci_high - selector_top_1_ci_low)
            if selector_top_1_ci_low is not None and selector_top_1_ci_high is not None
            else None
        ),
        "selector_top2_feasible_fraction_ci95_width": (
            float(selector_top_2_ci_high - selector_top_2_ci_low)
            if selector_top_2_ci_low is not None and selector_top_2_ci_high is not None
            else None
        ),
        "n_samples_mean": _mean(n_samples_values),
        "n_samples_min": min(n_samples_values) if n_samples_values else None,
        "n_samples_max": max(n_samples_values) if n_samples_values else None,
    }

    return per_pocket_scan, summary, rank_by_pocket, sorted(missing_counts)


def _normalize_population_confidence_rows(
    rows: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    if not rows:
        return normalized
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        pocket_id = item.get("pocket_id")
        if pocket_id is None:
            continue
        row = dict(item)
        row["pocket_id"] = str(pocket_id)
        normalized.append(row)
    return normalized


def _resolve_population_confidence_best_row(
    *,
    rows: Sequence[Mapping[str, Any]],
    best_pocket_id: str | None,
    dominant_pocket_id: str | None,
) -> dict[str, Any] | None:
    by_id = {
        str(item.get("pocket_id")): dict(item)
        for item in rows
        if isinstance(item, Mapping) and item.get("pocket_id") is not None
    }
    if best_pocket_id is not None and best_pocket_id in by_id:
        return by_id[best_pocket_id]
    if dominant_pocket_id is not None and dominant_pocket_id in by_id:
        return by_id[dominant_pocket_id]
    if by_id:
        ranked = sorted(
            by_id.values(),
            key=lambda item: _safe_float(item.get("best_score_global"))
            if _safe_float(item.get("best_score_global")) is not None
            else float("-inf"),
            reverse=True,
        )
        return dict(ranked[0]) if ranked else None
    return None


def _build_population_confidence_features(
    *,
    best_pocket_id: str | None,
    population_confidence_by_pocket: Sequence[Mapping[str, Any]] | None,
    population_confidence_global: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    by_pocket_rows = _normalize_population_confidence_rows(population_confidence_by_pocket)
    global_payload = (
        dict(population_confidence_global)
        if isinstance(population_confidence_global, Mapping)
        else {}
    )

    dominant_pocket_id = global_payload.get("dominant_pocket_id")
    dominant_pocket_id_str = str(dominant_pocket_id) if dominant_pocket_id is not None else None
    best_row = _resolve_population_confidence_best_row(
        rows=by_pocket_rows,
        best_pocket_id=best_pocket_id,
        dominant_pocket_id=dominant_pocket_id_str,
    )
    if best_row is not None and best_row.get("pocket_id") is not None:
        best_row["pocket_id"] = str(best_row.get("pocket_id"))

    uncertainty_flags = (
        [str(flag) for flag in best_row.get("uncertainty_flags", []) if flag is not None]
        if isinstance(best_row, Mapping) and isinstance(best_row.get("uncertainty_flags"), list)
        else []
    )
    best_pocket_uncertainty_count = int(len(uncertainty_flags)) if best_row is not None else None

    best_pocket_converged_stably = (
        _safe_bool(best_row.get("converged_stably")) if best_row is not None else None
    )
    best_pocket_viable_fraction_last = (
        _safe_float(best_row.get("viable_fraction_last")) if best_row is not None else None
    )
    best_pocket_viable_fraction_mean = (
        _safe_float(best_row.get("viable_fraction_mean")) if best_row is not None else None
    )
    best_pocket_elite_score_std_last = (
        _safe_float(best_row.get("elite_score_std_last")) if best_row is not None else None
    )
    best_pocket_elite_coords_dispersion_last = (
        _safe_float(best_row.get("elite_coords_dispersion_last")) if best_row is not None else None
    )
    best_pocket_plateau_generations_last_window = (
        _safe_int(best_row.get("plateau_generations_last_window")) if best_row is not None else None
    )

    dominant_fraction = _safe_float(global_payload.get("dominant_pocket_candidate_fraction"))
    best_vs_second_best_gap = _safe_float(global_payload.get("best_vs_second_best_gap"))
    stable_pockets_count = _safe_int(global_payload.get("stable_pockets_count"))
    uncertain_pockets_count = _safe_int(global_payload.get("uncertain_pockets_count"))

    popconf_available = bool(by_pocket_rows or global_payload)
    run_features = {
        "popconf_available": popconf_available,
        "popconf_dominant_pocket_candidate_fraction": dominant_fraction,
        "popconf_best_vs_second_best_gap": best_vs_second_best_gap,
        "popconf_stable_pockets_count": stable_pockets_count,
        "popconf_uncertain_pockets_count": uncertain_pockets_count,
        "popconf_best_pocket_converged_stably": best_pocket_converged_stably,
        "popconf_best_pocket_viable_fraction_last": best_pocket_viable_fraction_last,
        "popconf_best_pocket_viable_fraction_mean": best_pocket_viable_fraction_mean,
        "popconf_best_pocket_elite_score_std_last": best_pocket_elite_score_std_last,
        "popconf_best_pocket_elite_coords_dispersion_last": best_pocket_elite_coords_dispersion_last,
        "popconf_best_pocket_plateau_generations_last_window": best_pocket_plateau_generations_last_window,
        "popconf_best_pocket_uncertainty_count": best_pocket_uncertainty_count,
    }

    best_pocket_features = {
        "available": bool(best_row is not None),
        "pocket_id": best_row.get("pocket_id") if best_row is not None else best_pocket_id,
        "converged_stably": best_pocket_converged_stably,
        "viable_fraction_last": best_pocket_viable_fraction_last,
        "viable_fraction_mean": best_pocket_viable_fraction_mean,
        "elite_score_std_last": best_pocket_elite_score_std_last,
        "elite_coords_dispersion_last": best_pocket_elite_coords_dispersion_last,
        "plateau_generations_last_window": best_pocket_plateau_generations_last_window,
        "uncertainty_flags": uncertainty_flags,
        "uncertainty_count": best_pocket_uncertainty_count,
    }

    overview = {
        "available": popconf_available,
        "n_pockets": _safe_int(global_payload.get("n_pockets"))
        if global_payload
        else int(len(by_pocket_rows)),
        "dominant_pocket_id": dominant_pocket_id_str,
        "dominant_pocket_candidate_fraction": dominant_fraction,
        "best_vs_second_best_gap": best_vs_second_best_gap,
        "stable_pockets_count": stable_pockets_count,
        "uncertain_pockets_count": uncertain_pockets_count,
        "best_pocket": best_pocket_features,
    }
    return run_features, best_pocket_features, overview


def _confidence_metadata(
    *,
    features: dict[str, Any],
    ready_for_calibration: bool,
    calibrator: ConfidenceCalibrator | None = None,
) -> dict[str, Any]:
    calibration_metadata = build_confidence_calibration_metadata(
        confidence_features=features if features else {},
        confidence_features_version=CONFIDENCE_FEATURES_VERSION,
        ready_for_calibration=ready_for_calibration,
        calibrator=calibrator,
    )
    return {
        "confidence_features_available": bool(features),
        "confidence_features_version": CONFIDENCE_FEATURES_VERSION,
        "confidence_features": features if features else {},
        "confidence_ready_for_calibration": bool(ready_for_calibration),
        "confidence_semantics": "precalibration_features_only",
        **calibration_metadata,
    }


def build_single_confidence_metadata(
    *,
    search_space_mode: str,
    best_score_cheap: float | None,
    best_score_expensive: float | None,
    total_pockets: int,
    selected_pockets: int,
    best_pocket_id: str | None,
    selected_pocket_ids: Sequence[str] | None = None,
    scan_by_pocket: Mapping[str, Mapping[str, Any]] | None = None,
    fallback_to_full: bool = False,
    calibrator: ConfidenceCalibrator | None = None,
) -> dict[str, Any]:
    selected_ids = _coerce_ids(selected_pocket_ids)
    resolved_best_pocket = (
        str(best_pocket_id)
        if best_pocket_id is not None
        else (selected_ids[0] if selected_ids else None)
    )

    per_pocket_scan, scan_summary, rank_by_pocket, missing_scan = _build_scan_features(
        scan_by_pocket=scan_by_pocket,
        pocket_order=selected_ids,
    )

    best_scan_row = None
    if resolved_best_pocket is not None:
        for row in per_pocket_scan:
            if row.get("pocket_id") == resolved_best_pocket:
                best_scan_row = row
                break

    triage = _safe_float(best_score_cheap)
    rescore = _safe_float(best_score_expensive)

    features = {
        "run": {
            "search_space_mode": str(search_space_mode),
            "n_pockets_detected": int(total_pockets),
            "n_pockets_used": int(selected_pockets),
            "selected_fraction": (float(selected_pockets) / float(total_pockets))
            if total_pockets > 0
            else None,
            "triage_score": triage,
            "rescore_proxy": rescore,
            "cheap_expensive_gap": (triage - rescore)
            if triage is not None and rescore is not None
            else None,
            "rescore_proxy_available": rescore is not None,
            "fallback_to_full": bool(fallback_to_full),
        },
        "best_pocket": {
            "pocket_id": resolved_best_pocket,
            "scan_rank": rank_by_pocket.get(resolved_best_pocket)
            if resolved_best_pocket is not None
            else None,
            "scan_features": best_scan_row,
        },
        "scan_overview": scan_summary,
        "per_pocket_scan": per_pocket_scan,
        "missing_scan_metrics": missing_scan,
    }

    ready = triage is not None and selected_pockets > 0
    return _confidence_metadata(
        features=features,
        ready_for_calibration=ready,
        calibrator=calibrator,
    )


def build_reduced_aggregate_confidence_metadata(
    *,
    search_space_mode: str,
    executed_mode: str,
    fallback_to_full: bool,
    best_pocket_id: str | None,
    best_over_pockets_cheap: float | None,
    best_over_pockets_expensive: float | None,
    candidate_pockets: Sequence[str] | None,
    selected_after_scan_pockets: Sequence[str] | None,
    accepted_pockets: Sequence[str] | None,
    executed_pockets: Sequence[str] | None,
    per_pocket_results: Sequence[Mapping[str, Any]] | None,
    scan_by_pocket: Mapping[str, Mapping[str, Any]] | None,
    population_confidence_by_pocket: Sequence[Mapping[str, Any]] | None = None,
    population_confidence_global: Mapping[str, Any] | None = None,
    peptide_atom_count: int | None = None,
    peptide_residue_count: int | None = None,
    receptor_atom_count: int | None = None,
    receptor_residue_count: int | None = None,
    calibrator: ConfidenceCalibrator | None = None,
) -> dict[str, Any]:
    candidate_ids = _coerce_ids(candidate_pockets)
    selected_after_scan_ids = _coerce_ids(selected_after_scan_pockets)
    accepted_ids = _coerce_ids(accepted_pockets)
    executed_ids = _coerce_ids(executed_pockets)

    execution_by_pocket: dict[str, dict[str, Any]] = {}
    for item in per_pocket_results or []:
        if not isinstance(item, Mapping):
            continue
        pocket_id_raw = item.get("pocket_id")
        if pocket_id_raw is None:
            continue
        pocket_id = str(pocket_id_raw)
        triage = _safe_float(item.get("best_score_cheap"))
        rescore = _safe_float(item.get("best_score_expensive"))
        execution_by_pocket[pocket_id] = {
            "best_score_cheap": triage,
            "best_score_expensive": rescore,
            "cheap_expensive_gap": (triage - rescore)
            if triage is not None and rescore is not None
            else None,
            "runtime_sec": _safe_float(item.get("runtime_sec")),
            "n_eval_total": _safe_float(item.get("n_eval_total")),
        }

    scan_order = list(candidate_ids)
    per_pocket_scan, scan_summary, rank_by_pocket, missing_scan = _build_scan_features(
        scan_by_pocket=scan_by_pocket,
        pocket_order=scan_order,
    )
    scan_by_id = {str(row.get("pocket_id")): row for row in per_pocket_scan}

    ordered_ids = list(candidate_ids)
    for source in (
        selected_after_scan_ids,
        accepted_ids,
        executed_ids,
        list(execution_by_pocket.keys()),
        list(scan_by_id.keys()),
    ):
        for pocket_id in source:
            if pocket_id not in ordered_ids:
                ordered_ids.append(pocket_id)

    candidate_set = set(candidate_ids)
    selected_set = set(selected_after_scan_ids)
    accepted_set = set(accepted_ids)
    executed_set = set(executed_ids)

    per_pocket_confidence: list[dict[str, Any]] = []
    for pocket_id in ordered_ids:
        scan_row = scan_by_id.get(pocket_id, {"pocket_id": pocket_id})
        exec_row = execution_by_pocket.get(pocket_id, {})
        per_pocket_confidence.append(
            {
                "pocket_id": pocket_id,
                "candidate": pocket_id in candidate_set,
                "selected_after_scan": pocket_id in selected_set,
                "accepted": pocket_id in accepted_set,
                "executed": pocket_id in executed_set,
                "scan_rank": rank_by_pocket.get(pocket_id),
                "scan_score": _safe_float(scan_row.get("scan_score")),
                "pocket_scan_score": _safe_float(scan_row.get("pocket_scan_score")),
                "robust_score": _safe_float(scan_row.get("robust_score")),
                "energy_q20": _safe_float(scan_row.get("energy_q20")),
                "feasible_fraction": _safe_float(scan_row.get("feasible_fraction")),
                "n_samples": _safe_float(scan_row.get("n_samples")),
                "best_score_cheap": exec_row.get("best_score_cheap"),
                "best_score_expensive": exec_row.get("best_score_expensive"),
                "cheap_expensive_gap": exec_row.get("cheap_expensive_gap"),
                "runtime_sec": exec_row.get("runtime_sec"),
                "n_eval_total": exec_row.get("n_eval_total"),
            }
        )

    cheap_values = [
        value
        for value in (_safe_float(row.get("best_score_cheap")) for row in per_pocket_confidence)
        if value is not None
    ]
    sorted_cheap = sorted(cheap_values, reverse=True)
    cheap_top1 = sorted_cheap[0] if sorted_cheap else None
    cheap_top2 = sorted_cheap[1] if len(sorted_cheap) > 1 else None

    resolved_best_pocket = str(best_pocket_id) if best_pocket_id is not None else None
    best_pocket_row = next(
        (row for row in per_pocket_confidence if row.get("pocket_id") == resolved_best_pocket),
        None,
    )
    (
        population_confidence_run_features,
        population_confidence_best_pocket,
        population_confidence_overview,
    ) = _build_population_confidence_features(
        best_pocket_id=resolved_best_pocket,
        population_confidence_by_pocket=population_confidence_by_pocket,
        population_confidence_global=population_confidence_global,
    )

    triage = _safe_float(best_over_pockets_cheap)
    rescore = _safe_float(best_over_pockets_expensive)
    peptide_atoms = _safe_positive_int(peptide_atom_count)
    peptide_residues = _safe_positive_int(peptide_residue_count)
    receptor_atoms = _safe_positive_int(receptor_atom_count)
    receptor_residues = _safe_positive_int(receptor_residue_count)

    scan_summary.update(
        {
            "scan_score_best_per_peptide_atom": _per_size(
                scan_summary.get("scan_score_best"), peptide_atoms
            ),
            "scan_score_mean_per_peptide_atom": _per_size(
                scan_summary.get("scan_score_mean"), peptide_atoms
            ),
            "scan_score_std_per_peptide_atom": _per_size(
                scan_summary.get("scan_score_std"), peptide_atoms
            ),
            "scan_score_iqr_per_peptide_atom": _per_size(
                scan_summary.get("scan_score_iqr"), peptide_atoms
            ),
            "scan_score_gap_top1_top2_per_peptide_atom": _per_size(
                scan_summary.get("scan_score_gap_top1_top2"), peptide_atoms
            ),
        }
    )
    if best_pocket_row is not None:
        best_pocket_row = dict(best_pocket_row)
        best_pocket_row["energy_q20_per_peptide_atom"] = _per_size(
            best_pocket_row.get("energy_q20"), peptide_atoms
        )

    run_features = {
        "search_space_mode": str(search_space_mode),
        "executed_mode": str(executed_mode),
        "fallback_to_full": bool(fallback_to_full),
        "n_candidate_pockets": len(candidate_ids),
        "n_selected_after_scan": len(selected_after_scan_ids),
        "n_accepted_pockets": len(accepted_ids),
        "n_executed_pockets": len(executed_ids),
        "accepted_over_candidate_ratio": (float(len(accepted_ids)) / float(len(candidate_ids)))
        if candidate_ids
        else None,
        "executed_over_candidate_ratio": (float(len(executed_ids)) / float(len(candidate_ids)))
        if candidate_ids
        else None,
        "triage_score": triage,
        "rescore_proxy": rescore,
        "triage_score_per_peptide_atom": _per_size(triage, peptide_atoms),
        "triage_score_per_peptide_residue": _per_size(
            triage, peptide_residues
        ),
        "rescore_proxy_per_peptide_atom": _per_size(rescore, peptide_atoms),
        "rescore_proxy_per_peptide_residue": _per_size(
            rescore, peptide_residues
        ),
        "cheap_expensive_gap": (triage - rescore)
        if triage is not None and rescore is not None
        else None,
        "best_cheap_top1": cheap_top1,
        "best_cheap_top2": cheap_top2,
        "best_cheap_gap_top1_top2": (cheap_top1 - cheap_top2)
        if cheap_top1 is not None and cheap_top2 is not None
        else None,
    }
    run_features.update(population_confidence_run_features)

    features = {
        "input": {
            "peptide_atom_count": peptide_atoms,
            "peptide_residue_count": peptide_residues,
            "receptor_atom_count": receptor_atoms,
            "receptor_residue_count": receptor_residues,
            "score_size_normalization": "peptide_atom_and_residue_count",
        },
        "run": run_features,
        "best_pocket": {
            "pocket_id": resolved_best_pocket,
            "features": best_pocket_row,
            "population_confidence": population_confidence_best_pocket,
        },
        "scan_overview": scan_summary,
        "population_confidence_overview": population_confidence_overview,
        "per_pocket": per_pocket_confidence,
        "missing_scan_metrics": missing_scan,
    }

    ready = triage is not None and bool(per_pocket_confidence)
    return _confidence_metadata(
        features=features,
        ready_for_calibration=ready,
        calibrator=calibrator,
    )
