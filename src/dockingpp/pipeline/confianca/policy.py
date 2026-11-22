"""Auditable confidence policy for reduced-search control decisions."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


CONFIDENCE_POLICY_VERSION = "4.0"

DEFAULT_MIN_ACCEPTED_RATIO = 0.34
DEFAULT_MIN_FEASIBLE_FRACTION_MEAN = 0.15
DEFAULT_MIN_SCAN_GAP_TOP1_TOP2 = 0.05
DEFAULT_MIN_CALIBRATED_SUCCESS_PROBABILITY = 0.80

DEFAULT_POPCONF_MIN_DOMINANT_POCKET_CANDIDATE_FRACTION = 0.55
DEFAULT_POPCONF_MIN_BEST_VS_SECOND_BEST_GAP = 0.05
DEFAULT_POPCONF_MIN_STABLE_POCKETS_COUNT = 1
DEFAULT_POPCONF_MAX_UNCERTAIN_POCKETS_COUNT = 1
DEFAULT_POPCONF_MAX_BEST_POCKET_ELITE_SCORE_STD_LAST = 0.25
DEFAULT_POPCONF_MAX_BEST_POCKET_ELITE_COORDS_DISPERSION_LAST = 1.0
DEFAULT_POPCONF_MIN_BEST_POCKET_VIABLE_FRACTION_LAST = 0.30
DEFAULT_POPCONF_MIN_BEST_POCKET_VIABLE_DELTA_FROM_MEAN = 0.0
DEFAULT_POPCONF_MIN_BEST_POCKET_PLATEAU_GENERATIONS_LAST_WINDOW = 2
DEFAULT_POPCONF_MAX_BEST_POCKET_UNCERTAINTY_COUNT = 1
DEFAULT_POPCONF_MIN_POSITIVE_SIGNALS_FOR_SUPPORT = 4
DEFAULT_POPCONF_MAX_NEGATIVE_SIGNALS_FOR_SUPPORT = 0
DEFAULT_POPCONF_MIN_NEGATIVE_SIGNALS_FOR_CAUTION = 2
DEFAULT_POPCONF_MIN_NEGATIVE_SIGNALS_FOR_FALLBACK = 9


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _coerce_ids(ids: Sequence[Any] | None) -> list[str]:
    if not ids:
        return []
    out: list[str] = []
    for value in ids:
        if value is None:
            continue
        out.append(str(value))
    return out


def _coerce_string_list(values: Any) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
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


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _feature_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def default_confidence_policy_metadata() -> dict[str, Any]:
    return {
        "confidence_policy_available": True,
        "confidence_policy_version": CONFIDENCE_POLICY_VERSION,
        "confidence_policy_used": False,
        "confidence_policy_mode": "disabled",
        "confidence_policy_effect": "none",
        "confidence_assisted_fallback": False,
        "confidence_fallback_recommended": False,
        "confidence_fallback_apply_requested": False,
        "confidence_assisted_selection": False,
        "confidence_decision_role": "calibrated_quality_gate_or_auditable_advisory",
        "confidence_not_scientific_score": True,
        "confidence_requires_calibration_for_automatic_acceptance": True,
        "confidence_requires_calibration_for_automatic_fallback": True,
        "confidence_can_trigger_conservative_fallback": False,
        "confidence_can_override_final_scoring": False,
        "confidence_decision": "neutral",
        "confidence_recommendation": "keep_current_flow",
        "confidence_decision_reasons": [],
        "confidence_warning_flags": [],
        "confidence_policy_details": {},
    }


def merge_confidence_policy_metadata(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = default_confidence_policy_metadata()
    if not isinstance(payload, Mapping):
        return merged
    for key in merged:
        if key in {"confidence_decision_reasons", "confidence_warning_flags"}:
            continue
        if key in payload:
            merged[key] = payload[key]
    merged["confidence_decision_reasons"] = _coerce_string_list(
        payload.get("confidence_decision_reasons")
    )
    merged["confidence_warning_flags"] = _coerce_string_list(
        payload.get("confidence_warning_flags")
    )
    details = payload.get("confidence_policy_details")
    if isinstance(details, Mapping):
        merged["confidence_policy_details"] = dict(details)
    return merged


def _evaluate_v5_evidence_policy(
    *,
    base: dict[str, Any],
    policy_cfg: Mapping[str, Any],
    mode: str,
    confidence_output: Mapping[str, Any],
    candidate_ids: list[str],
    selected_ids: list[str],
    accepted_ids: list[str],
) -> dict[str, Any]:
    """Translate the frozen v5 multi-gate decision without legacy heuristics."""

    artifact_decision = str(confidence_output.get("decision") or "").upper()
    artifact_decision_capable = bool(
        confidence_output.get("artifact_decision_capable") is True
    )
    reason_codes = _coerce_string_list(confidence_output.get("reason_codes"))
    graph = _feature_mapping(confidence_output.get("graph"))
    can_assist_fallback = bool(mode in {"assist_fallback", "assist_all"})
    apply_fallback_configured = bool(policy_cfg.get("apply_fallback", False))

    effect = "none"
    decision = "observe_only"
    recommendation = "record_v5_evidence_only"
    fallback_recommended = False
    apply_fallback = False
    decision_reasons = list(reason_codes)
    warning_flags = [
        code
        for code in reason_codes
        if code
        in {
            "ARTIFACT_NOT_DECISION_CAPABLE",
            "MISSING_FEATURE",
            "OUT_OF_DOMAIN",
            "GRAPH_MODEL_DISAGREEMENT",
            "LOW_STATISTICAL_CONFIDENCE",
        }
    ]

    calibrated = confidence_output.get("calibrated") is True
    effective_decision_capable = confidence_output.get("decision_capable") is True
    accept_contract = bool(
        artifact_decision == "ACCEPT_REDUCED"
        and artifact_decision_capable
        and effective_decision_capable
        and calibrated
        and confidence_output.get("features_complete") is True
        and confidence_output.get("applicability_domain_supported") is True
        and confidence_output.get("graph_agreement_supported") is True
        and confidence_output.get("statistical_probability_supported") is True
        and confidence_output.get("fallback_required") is False
        and "ACCEPTED" in reason_codes
        and bool(accepted_ids)
    )
    active_v5_mode = bool(
        can_assist_fallback and artifact_decision_capable and calibrated
    )
    if mode == "observe" or not active_v5_mode:
        if not artifact_decision_capable and "ARTIFACT_NOT_DECISION_CAPABLE" not in decision_reasons:
            decision_reasons.insert(0, "ARTIFACT_NOT_DECISION_CAPABLE")
        decision = "observe_only"
        recommendation = "record_v5_evidence_only"
    elif accept_contract:
        decision = "accept_reduced"
        recommendation = "accept_reduced_result"
        decision_reasons = _coerce_string_list([*decision_reasons, "ACCEPTED"])
    else:
        if artifact_decision == "ACCEPT_REDUCED":
            decision_reasons = _coerce_string_list(
                [*decision_reasons, "V5_ACCEPTANCE_CONTRACT_FAILED"]
            )
        fallback_recommended = True
        apply_fallback = bool(apply_fallback_configured)
        if apply_fallback:
            effect = "fallback_requested"
            decision = "fallback_requested"
            recommendation = "execute_full_fallback"
        else:
            effect = "fallback_recommended"
            decision = "fallback_recommended"
            recommendation = "execute_full_fallback_before_acceptance"

    decision_reasons = _coerce_string_list(decision_reasons)
    warning_flags = _coerce_string_list(warning_flags)
    base.update(
        {
            "confidence_policy_used": True,
            "confidence_policy_mode": mode,
            "confidence_policy_effect": effect,
            "confidence_assisted_fallback": bool(apply_fallback),
            "confidence_fallback_recommended": bool(fallback_recommended),
            "confidence_fallback_apply_requested": bool(apply_fallback),
            "confidence_assisted_selection": False,
            "confidence_can_trigger_conservative_fallback": bool(active_v5_mode),
            "confidence_decision": decision,
            "confidence_recommendation": recommendation,
            "confidence_decision_reasons": decision_reasons,
            "confidence_warning_flags": warning_flags,
            "confidence_policy_details": {
                "protocol": "v5_statistical_plus_evidence_graph",
                "candidate_pockets": candidate_ids,
                "selected_pockets": selected_ids,
                "accepted_pockets": accepted_ids,
                "artifact_decision": artifact_decision,
                "artifact_decision_capable": artifact_decision_capable,
                "effective_decision_capable": effective_decision_capable,
                "calibrated": calibrated,
                "acceptance_contract_satisfied": accept_contract,
                "features_complete": confidence_output.get("features_complete"),
                "p_stat": confidence_output.get("p_stat"),
                "graph": dict(graph),
                "domain_supported": confidence_output.get(
                    "applicability_domain_supported"
                ),
                "graph_agreement_supported": confidence_output.get(
                    "graph_agreement_supported"
                ),
                "statistical_probability_supported": confidence_output.get(
                    "statistical_probability_supported"
                ),
                "decision_policy": confidence_output.get("policy"),
                "fallback_required_by_artifact": confidence_output.get(
                    "fallback_required"
                ),
                "apply_confidence_assisted_fallback": bool(apply_fallback),
                "reason_codes": reason_codes,
            },
        }
    )
    return base


def evaluate_confidence_policy(
    *,
    cfg: Any,
    confidence_features: Mapping[str, Any] | None,
    confidence_output: Mapping[str, Any] | None,
    candidate_pocket_ids: Sequence[str] | None,
    selected_pocket_ids: Sequence[str] | None,
    accepted_pocket_ids: Sequence[str] | None,
) -> dict[str, Any]:
    base = default_confidence_policy_metadata()
    policy_cfg = _cfg_value(cfg, "confidence_policy", {})
    if not isinstance(policy_cfg, Mapping):
        policy_cfg = {}

    enabled = bool(policy_cfg.get("enabled", False))
    mode = str(policy_cfg.get("mode", "disabled") or "disabled").strip().lower()
    if mode not in {"disabled", "observe", "assist_selection", "assist_fallback", "assist_all"}:
        mode = "disabled"
    if not enabled or mode == "disabled":
        base["confidence_policy_mode"] = mode
        base["confidence_policy_details"] = {
            "confidence_features_present": isinstance(confidence_features, Mapping)
            and bool(confidence_features),
            "confidence_output_present": isinstance(confidence_output, Mapping),
            "policy_disabled": True,
        }
        return base

    candidate_ids = _coerce_ids(candidate_pocket_ids)
    selected_ids = _coerce_ids(selected_pocket_ids)
    accepted_ids = _coerce_ids(accepted_pocket_ids)

    if (
        isinstance(confidence_output, Mapping)
        and str(confidence_output.get("artifact_schema_version")) == "5"
        and confidence_output.get("model_kind")
        == "statistical_plus_evidence_graph"
    ):
        return _evaluate_v5_evidence_policy(
            base=base,
            policy_cfg=policy_cfg,
            mode=mode,
            confidence_output=confidence_output,
            candidate_ids=candidate_ids,
            selected_ids=selected_ids,
            accepted_ids=accepted_ids,
        )

    features = confidence_features if isinstance(confidence_features, Mapping) else {}
    run_features = _feature_mapping(features.get("run"))
    best_pocket_payload = _feature_mapping(features.get("best_pocket"))
    best_pocket_popconf = _feature_mapping(best_pocket_payload.get("population_confidence"))
    scan_overview = _feature_mapping(features.get("scan_overview"))
    popconf_overview = _feature_mapping(features.get("population_confidence_overview"))

    per_pocket_scan = features.get("per_pocket_scan")
    if not isinstance(per_pocket_scan, list):
        per_pocket_scan = features.get("per_pocket")
    if not isinstance(per_pocket_scan, list):
        per_pocket_scan = []

    scan_by_id: dict[str, Mapping[str, Any]] = {}
    for row in per_pocket_scan:
        if not isinstance(row, Mapping):
            continue
        pocket_id = row.get("pocket_id")
        if pocket_id is None:
            continue
        scan_by_id[str(pocket_id)] = row

    accepted_ratio = (float(len(accepted_ids)) / float(len(candidate_ids))) if candidate_ids else None
    feasible_fraction_mean = _safe_float(scan_overview.get("feasible_fraction_mean"))
    scan_gap = _safe_float(scan_overview.get("scan_score_gap_top1_top2"))
    selector_feasible_ci_overlap = _safe_bool(
        scan_overview.get("selector_feasible_fraction_ci95_overlap_top1_top2")
    )

    decision_reasons: list[str] = []
    warning_flags: list[str] = []
    uncertainty_reasons: list[str] = []

    if not accepted_ids:
        uncertainty_reasons.append("no_accepted_pockets")

    min_accepted_ratio = _safe_float(
        policy_cfg.get("min_accepted_ratio", DEFAULT_MIN_ACCEPTED_RATIO)
    )
    if (
        accepted_ratio is not None
        and min_accepted_ratio is not None
        and accepted_ratio < min_accepted_ratio
    ):
        uncertainty_reasons.append("accepted_ratio_low")

    min_feasible_fraction_mean = _safe_float(
        policy_cfg.get(
            "min_feasible_fraction_mean",
            DEFAULT_MIN_FEASIBLE_FRACTION_MEAN,
        )
    )
    if (
        feasible_fraction_mean is not None
        and min_feasible_fraction_mean is not None
        and feasible_fraction_mean < min_feasible_fraction_mean
    ):
        uncertainty_reasons.append("feasible_fraction_mean_low")

    min_scan_gap = _safe_float(
        policy_cfg.get("min_scan_gap_top1_top2", DEFAULT_MIN_SCAN_GAP_TOP1_TOP2)
    )
    if scan_gap is not None and min_scan_gap is not None and scan_gap < min_scan_gap:
        uncertainty_reasons.append("scan_gap_low")
    if selector_feasible_ci_overlap is True:
        uncertainty_reasons.append("selector_feasible_ci95_overlap")

    prob_est = None
    calibrated_probability_available = False
    calibrated_probability_decision_capable = False
    calibrated_probability_low = False
    calibrated_probability_supports_acceptance = False
    configured_probability_threshold = None
    artifact_probability_threshold = None
    effective_probability_threshold = None
    probability_semantics = None
    if isinstance(confidence_output, Mapping):
        preservation_probability = confidence_output.get(
            "estimated_preservation_probability"
        )
        prob_est = _safe_float(
            preservation_probability
            if preservation_probability is not None
            else confidence_output.get("estimated_success_probability")
        )
        probability_semantics = (
            "reduced_vs_full_structural_preservation"
            if preservation_probability is not None
            else "legacy_or_unspecified_success"
        )
        configured_probability_threshold = _safe_float(
            policy_cfg.get("min_calibrated_success_probability")
        )
        artifact_probability_threshold = _safe_float(
            confidence_output.get("decision_probability_threshold")
        )
        if artifact_probability_threshold is not None:
            if preservation_probability is not None:
                # Preservation v3 validates one learned operating point.  A
                # numerically higher threshold is not silently assumed safe:
                # the empirical risk curve need not be monotone in a small
                # grouped sample.  Runtime therefore uses the frozen artifact
                # threshold exactly.
                effective_probability_threshold = artifact_probability_threshold
                if (
                    configured_probability_threshold is not None
                    and abs(
                        configured_probability_threshold
                        - artifact_probability_threshold
                    )
                    > 1e-12
                ):
                    warning_flags.append(
                        "configured_probability_threshold_ignored_for_"
                        "preservation_artifact"
                    )
            else:
                # Retain the historical stricter-YAML behavior for legacy
                # observation-only success scores.
                effective_probability_threshold = max(
                    artifact_probability_threshold,
                    configured_probability_threshold
                    if configured_probability_threshold is not None
                    else artifact_probability_threshold,
                )
        else:
            effective_probability_threshold = _safe_float(
                _coalesce(
                    configured_probability_threshold,
                    DEFAULT_MIN_CALIBRATED_SUCCESS_PROBABILITY,
                )
            )
        calibrated_probability_decision_capable = bool(
            confidence_output.get("decision_capable", False) is True
        )
        calibrated_probability_available = bool(
            prob_est is not None
            and confidence_output.get("calibrated") is True
            and calibrated_probability_decision_capable
        )
        if prob_est is not None and not calibrated_probability_available:
            if (
                confidence_output.get("calibrated") is True
                and not calibrated_probability_decision_capable
            ):
                warning_flags.append(
                    "calibrated_output_not_decision_capable"
                )
                uncertainty_reasons.append(
                    "calibrated_operating_point_unsupported"
                )
            else:
                warning_flags.append(
                    "uncalibrated_confidence_output_ignored_for_decision"
                )
        if (
            calibrated_probability_available
            and effective_probability_threshold is not None
        ):
            calibrated_probability_low = bool(
                prob_est < effective_probability_threshold
            )
            calibrated_probability_supports_acceptance = not calibrated_probability_low
            if calibrated_probability_low:
                uncertainty_reasons.append("calibrated_success_probability_low")

    popconf_dominant_fraction = _safe_float(
        _coalesce(
            run_features.get("popconf_dominant_pocket_candidate_fraction"),
            popconf_overview.get("dominant_pocket_candidate_fraction"),
        )
    )
    popconf_best_vs_second_best_gap = _safe_float(
        _coalesce(
            run_features.get("popconf_best_vs_second_best_gap"),
            popconf_overview.get("best_vs_second_best_gap"),
        )
    )
    popconf_stable_pockets_count = _safe_int(
        _coalesce(
            run_features.get("popconf_stable_pockets_count"),
            popconf_overview.get("stable_pockets_count"),
        )
    )
    popconf_uncertain_pockets_count = _safe_int(
        _coalesce(
            run_features.get("popconf_uncertain_pockets_count"),
            popconf_overview.get("uncertain_pockets_count"),
        )
    )
    popconf_best_pocket_converged_stably = _safe_bool(
        _coalesce(
            run_features.get("popconf_best_pocket_converged_stably"),
            best_pocket_popconf.get("converged_stably"),
        )
    )
    popconf_best_pocket_viable_fraction_last = _safe_float(
        _coalesce(
            run_features.get("popconf_best_pocket_viable_fraction_last"),
            best_pocket_popconf.get("viable_fraction_last"),
        )
    )
    popconf_best_pocket_viable_fraction_mean = _safe_float(
        _coalesce(
            run_features.get("popconf_best_pocket_viable_fraction_mean"),
            best_pocket_popconf.get("viable_fraction_mean"),
        )
    )
    popconf_best_pocket_elite_score_std_last = _safe_float(
        _coalesce(
            run_features.get("popconf_best_pocket_elite_score_std_last"),
            best_pocket_popconf.get("elite_score_std_last"),
        )
    )
    popconf_best_pocket_elite_coords_dispersion_last = _safe_float(
        _coalesce(
            run_features.get("popconf_best_pocket_elite_coords_dispersion_last"),
            best_pocket_popconf.get("elite_coords_dispersion_last"),
        )
    )
    popconf_best_pocket_plateau_generations_last_window = _safe_int(
        _coalesce(
            run_features.get("popconf_best_pocket_plateau_generations_last_window"),
            best_pocket_popconf.get("plateau_generations_last_window"),
        )
    )
    popconf_best_pocket_uncertainty_count = _safe_int(
        _coalesce(
            run_features.get("popconf_best_pocket_uncertainty_count"),
            best_pocket_popconf.get("uncertainty_count"),
        )
    )

    popconf_available = bool(
        _safe_bool(_coalesce(run_features.get("popconf_available"), popconf_overview.get("available")))
    ) or any(
        value is not None
        for value in (
            popconf_dominant_fraction,
            popconf_best_vs_second_best_gap,
            popconf_stable_pockets_count,
            popconf_uncertain_pockets_count,
            popconf_best_pocket_converged_stably,
            popconf_best_pocket_viable_fraction_last,
            popconf_best_pocket_viable_fraction_mean,
            popconf_best_pocket_elite_score_std_last,
            popconf_best_pocket_elite_coords_dispersion_last,
            popconf_best_pocket_plateau_generations_last_window,
            popconf_best_pocket_uncertainty_count,
        )
    )

    popconf_thresholds = {
        "min_dominant_pocket_candidate_fraction": _safe_float(
            policy_cfg.get(
                "popconf_min_dominant_pocket_candidate_fraction",
                DEFAULT_POPCONF_MIN_DOMINANT_POCKET_CANDIDATE_FRACTION,
            )
        ),
        "min_best_vs_second_best_gap": _safe_float(
            policy_cfg.get(
                "popconf_min_best_vs_second_best_gap",
                DEFAULT_POPCONF_MIN_BEST_VS_SECOND_BEST_GAP,
            )
        ),
        "min_stable_pockets_count": _safe_int(
            policy_cfg.get(
                "popconf_min_stable_pockets_count",
                DEFAULT_POPCONF_MIN_STABLE_POCKETS_COUNT,
            )
        ),
        "max_uncertain_pockets_count": _safe_int(
            policy_cfg.get(
                "popconf_max_uncertain_pockets_count",
                DEFAULT_POPCONF_MAX_UNCERTAIN_POCKETS_COUNT,
            )
        ),
        "max_best_pocket_elite_score_std_last": _safe_float(
            policy_cfg.get(
                "popconf_max_best_pocket_elite_score_std_last",
                DEFAULT_POPCONF_MAX_BEST_POCKET_ELITE_SCORE_STD_LAST,
            )
        ),
        "max_best_pocket_elite_coords_dispersion_last": _safe_float(
            policy_cfg.get(
                "popconf_max_best_pocket_elite_coords_dispersion_last",
                DEFAULT_POPCONF_MAX_BEST_POCKET_ELITE_COORDS_DISPERSION_LAST,
            )
        ),
        "min_best_pocket_viable_fraction_last": _safe_float(
            policy_cfg.get(
                "popconf_min_best_pocket_viable_fraction_last",
                DEFAULT_POPCONF_MIN_BEST_POCKET_VIABLE_FRACTION_LAST,
            )
        ),
        "min_best_pocket_viable_delta_from_mean": _safe_float(
            policy_cfg.get(
                "popconf_min_best_pocket_viable_delta_from_mean",
                DEFAULT_POPCONF_MIN_BEST_POCKET_VIABLE_DELTA_FROM_MEAN,
            )
        ),
        "min_best_pocket_plateau_generations_last_window": _safe_int(
            policy_cfg.get(
                "popconf_min_best_pocket_plateau_generations_last_window",
                DEFAULT_POPCONF_MIN_BEST_POCKET_PLATEAU_GENERATIONS_LAST_WINDOW,
            )
        ),
        "max_best_pocket_uncertainty_count": _safe_int(
            policy_cfg.get(
                "popconf_max_best_pocket_uncertainty_count",
                DEFAULT_POPCONF_MAX_BEST_POCKET_UNCERTAINTY_COUNT,
            )
        ),
        "min_positive_signals_for_support": _safe_int(
            policy_cfg.get(
                "popconf_min_positive_signals_for_support",
                DEFAULT_POPCONF_MIN_POSITIVE_SIGNALS_FOR_SUPPORT,
            )
        ),
        "max_negative_signals_for_support": _safe_int(
            policy_cfg.get(
                "popconf_max_negative_signals_for_support",
                DEFAULT_POPCONF_MAX_NEGATIVE_SIGNALS_FOR_SUPPORT,
            )
        ),
        "min_negative_signals_for_caution": _safe_int(
            policy_cfg.get(
                "popconf_min_negative_signals_for_caution",
                DEFAULT_POPCONF_MIN_NEGATIVE_SIGNALS_FOR_CAUTION,
            )
        ),
        "min_negative_signals_for_fallback": _safe_int(
            policy_cfg.get(
                "popconf_min_negative_signals_for_fallback",
                DEFAULT_POPCONF_MIN_NEGATIVE_SIGNALS_FOR_FALLBACK,
            )
        ),
    }

    popconf_conditions: dict[str, bool | None] = {}

    def _condition_min(
        *,
        name: str,
        value: float | int | None,
        threshold: float | int | None,
        reason_ok: str,
        warning_low: str,
    ) -> None:
        if value is None or threshold is None:
            popconf_conditions[name] = None
            return
        passed = bool(value >= threshold)
        popconf_conditions[name] = passed
        if passed:
            decision_reasons.append(reason_ok)
        else:
            warning_flags.append(warning_low)

    def _condition_max(
        *,
        name: str,
        value: float | int | None,
        threshold: float | int | None,
        reason_ok: str,
        warning_high: str,
    ) -> None:
        if value is None or threshold is None:
            popconf_conditions[name] = None
            return
        passed = bool(value <= threshold)
        popconf_conditions[name] = passed
        if passed:
            decision_reasons.append(reason_ok)
        else:
            warning_flags.append(warning_high)

    if popconf_available:
        _condition_min(
            name="dominant_pocket_candidate_fraction_ok",
            value=popconf_dominant_fraction,
            threshold=popconf_thresholds["min_dominant_pocket_candidate_fraction"],
            reason_ok="dominant_pocket_well_defined",
            warning_low="dominant_pocket_candidate_fraction_low",
        )
        _condition_min(
            name="best_vs_second_best_gap_ok",
            value=popconf_best_vs_second_best_gap,
            threshold=popconf_thresholds["min_best_vs_second_best_gap"],
            reason_ok="best_vs_second_best_gap_is_clear",
            warning_low="best_vs_second_best_gap_low",
        )
        _condition_min(
            name="stable_pockets_count_ok",
            value=popconf_stable_pockets_count,
            threshold=popconf_thresholds["min_stable_pockets_count"],
            reason_ok="stable_pockets_count_sufficient",
            warning_low="stable_pockets_count_low",
        )
        _condition_max(
            name="uncertain_pockets_count_ok",
            value=popconf_uncertain_pockets_count,
            threshold=popconf_thresholds["max_uncertain_pockets_count"],
            reason_ok="uncertain_pockets_count_controlled",
            warning_high="uncertain_pockets_count_high",
        )
        if popconf_best_pocket_converged_stably is None:
            popconf_conditions["best_pocket_converged_stably"] = None
        elif popconf_best_pocket_converged_stably:
            popconf_conditions["best_pocket_converged_stably"] = True
            decision_reasons.append("best_pocket_converged_stably")
        else:
            popconf_conditions["best_pocket_converged_stably"] = False
            warning_flags.append("best_pocket_not_converged_stably")
        _condition_max(
            name="best_pocket_elite_score_std_low",
            value=popconf_best_pocket_elite_score_std_last,
            threshold=popconf_thresholds["max_best_pocket_elite_score_std_last"],
            reason_ok="best_pocket_elite_score_std_low",
            warning_high="best_pocket_elite_score_std_high",
        )
        _condition_max(
            name="best_pocket_elite_coords_dispersion_low",
            value=popconf_best_pocket_elite_coords_dispersion_last,
            threshold=popconf_thresholds["max_best_pocket_elite_coords_dispersion_last"],
            reason_ok="best_pocket_elite_coords_dispersion_low",
            warning_high="best_pocket_elite_coords_dispersion_high",
        )
        _condition_min(
            name="best_pocket_viable_fraction_last_ok",
            value=popconf_best_pocket_viable_fraction_last,
            threshold=popconf_thresholds["min_best_pocket_viable_fraction_last"],
            reason_ok="best_pocket_viable_fraction_last_adequate",
            warning_low="best_pocket_viable_fraction_last_low",
        )
        viable_delta = None
        if (
            popconf_best_pocket_viable_fraction_last is not None
            and popconf_best_pocket_viable_fraction_mean is not None
        ):
            viable_delta = (
                popconf_best_pocket_viable_fraction_last
                - popconf_best_pocket_viable_fraction_mean
            )
        _condition_min(
            name="best_pocket_viable_delta_from_mean_ok",
            value=viable_delta,
            threshold=popconf_thresholds["min_best_pocket_viable_delta_from_mean"],
            reason_ok="best_pocket_viable_fraction_last_not_below_mean",
            warning_low="best_pocket_viable_fraction_last_below_mean",
        )
        _condition_min(
            name="best_pocket_plateau_generations_ok",
            value=popconf_best_pocket_plateau_generations_last_window,
            threshold=popconf_thresholds["min_best_pocket_plateau_generations_last_window"],
            reason_ok="best_pocket_plateau_generations_sufficient",
            warning_low="best_pocket_plateau_generations_low",
        )
        _condition_max(
            name="best_pocket_uncertainty_count_ok",
            value=popconf_best_pocket_uncertainty_count,
            threshold=popconf_thresholds["max_best_pocket_uncertainty_count"],
            reason_ok="best_pocket_uncertainty_flags_controlled",
            warning_high="best_pocket_uncertainty_count_high",
        )

    positive_popconf_signals = sum(1 for value in popconf_conditions.values() if value is True)
    negative_popconf_signals = sum(1 for value in popconf_conditions.values() if value is False)

    mandatory_support_keys = (
        "dominant_pocket_candidate_fraction_ok",
        "best_vs_second_best_gap_ok",
        "best_pocket_converged_stably",
        "best_pocket_elite_coords_dispersion_low",
        "best_pocket_viable_fraction_last_ok",
    )
    mandatory_support_available = all(
        popconf_conditions.get(key) is not None for key in mandatory_support_keys
    )
    mandatory_support_passed = all(
        popconf_conditions.get(key) is True for key in mandatory_support_keys
    )

    min_positive_support = _safe_int(popconf_thresholds["min_positive_signals_for_support"])
    max_negative_support = _safe_int(popconf_thresholds["max_negative_signals_for_support"])
    min_negative_for_caution = _safe_int(popconf_thresholds["min_negative_signals_for_caution"])
    min_negative_for_fallback = _safe_int(popconf_thresholds["min_negative_signals_for_fallback"])

    popconf_supports_acceptance = bool(
        popconf_available
        and mandatory_support_available
        and mandatory_support_passed
        and (
            min_positive_support is None
            or positive_popconf_signals >= min_positive_support
        )
        and (
            max_negative_support is None
            or negative_popconf_signals <= max_negative_support
        )
    )
    popconf_requires_caution = bool(
        popconf_available
        and min_negative_for_caution is not None
        and negative_popconf_signals >= min_negative_for_caution
    )
    popconf_requires_fallback = bool(
        popconf_available
        and min_negative_for_fallback is not None
        and negative_popconf_signals >= min_negative_for_fallback
    )
    if popconf_supports_acceptance:
        decision_reasons.append("population_confidence_supports_reduced_acceptance")
    if popconf_requires_caution:
        warning_flags.append("population_confidence_uncertain")
        uncertainty_reasons.append("population_confidence_uncertain")
    if popconf_requires_fallback:
        warning_flags.append("population_confidence_fallback_threshold_reached")

    high_uncertainty = bool(uncertainty_reasons)

    can_assist_selection = enabled and mode in {"assist_selection", "assist_all"}
    can_assist_fallback = enabled and mode in {"assist_fallback", "assist_all"}

    recommended_pruned: list[str] = []
    apply_pruning = False
    if can_assist_selection and accepted_ids:
        keep_top_by_scan = int(policy_cfg.get("keep_top_by_scan", 1) or 1)
        min_keep = int(policy_cfg.get("min_keep_pockets", 1) or 1)
        prune_max_feasible_fraction = _safe_float(
            policy_cfg.get("prune_if_feasible_fraction_leq", 0.20)
        )

        accepted_ranked = []
        for pocket_id in accepted_ids:
            row = scan_by_id.get(pocket_id, {})
            score = _safe_float(row.get("pocket_scan_score", row.get("scan_score")))
            if score is None:
                score = float("-inf")
            accepted_ranked.append((pocket_id, score, row))
        accepted_ranked.sort(key=lambda item: item[1], reverse=True)

        for idx, (pocket_id, _score, row) in enumerate(accepted_ranked):
            if idx < keep_top_by_scan:
                continue
            feasible = _safe_float(row.get("feasible_fraction"))
            if (
                feasible is not None
                and prune_max_feasible_fraction is not None
                and feasible <= prune_max_feasible_fraction
            ):
                recommended_pruned.append(pocket_id)

        max_removable = max(0, len(accepted_ids) - min_keep)
        if len(recommended_pruned) > max_removable:
            recommended_pruned = recommended_pruned[:max_removable]
        apply_pruning = bool(recommended_pruned and bool(policy_cfg.get("apply_pruning", False)))

    allow_uncalibrated_fallback = bool(
        policy_cfg.get("allow_uncalibrated_fallback", False)
    )
    fallback_signal = (
        calibrated_probability_low
        if calibrated_probability_available
        else bool(
            any(
                reason != "population_confidence_uncertain"
                for reason in uncertainty_reasons
            )
            or popconf_requires_fallback
        )
    )
    recommend_fallback = bool(
        can_assist_fallback
        and bool(accepted_ids)
        and fallback_signal
        and (
            calibrated_probability_low
            or not popconf_supports_acceptance
        )
    )
    apply_fallback = bool(
        recommend_fallback
        and bool(policy_cfg.get("apply_fallback", False))
        and (calibrated_probability_available or allow_uncalibrated_fallback)
    )

    effect = "none"
    if apply_pruning:
        effect = "selection_pruned"
    elif recommend_fallback and apply_fallback:
        effect = "fallback_requested"
    elif recommend_fallback:
        effect = "fallback_recommended"
    elif high_uncertainty:
        effect = "high_uncertainty_flagged"

    decision = "neutral"
    recommendation = "keep_current_flow"
    if apply_pruning:
        decision = "selection_pruned"
        recommendation = "continue_with_pruned_selection"
        decision_reasons.append("confidence_policy_applied_conservative_pruning")
    elif apply_fallback:
        decision = "fallback_requested"
        recommendation = "execute_full_fallback"
        decision_reasons.append("confidence_policy_requested_full_fallback")
    elif recommend_fallback:
        decision = "fallback_recommended"
        recommendation = "consider_full_fallback"
        decision_reasons.append("confidence_policy_recommends_full_fallback")
    elif (
        calibrated_probability_supports_acceptance
        and not high_uncertainty
        and not popconf_requires_caution
    ):
        decision = "accept_reduced"
        recommendation = "accept_reduced_result"
        decision_reasons.append("calibrated_confidence_supports_reduced_acceptance")
    elif popconf_supports_acceptance:
        decision = "supportive_uncalibrated_evidence"
        recommendation = "await_calibrated_acceptance_decision"
        decision_reasons.append(
            "population_confidence_supports_reduced_but_is_not_calibrated"
        )
    elif high_uncertainty or popconf_requires_caution:
        decision = "caution"
        recommendation = "review_with_caution"
        decision_reasons.append("confidence_policy_detected_high_uncertainty")

    warning_flags = _coerce_string_list([*warning_flags, *uncertainty_reasons])
    decision_reasons = _coerce_string_list(decision_reasons)

    base.update(
        {
            "confidence_policy_used": bool(enabled and mode != "disabled"),
            "confidence_policy_mode": mode,
            "confidence_policy_effect": effect,
            "confidence_assisted_fallback": bool(apply_fallback),
            "confidence_fallback_recommended": bool(recommend_fallback),
            "confidence_fallback_apply_requested": bool(apply_fallback),
            "confidence_assisted_selection": bool(apply_pruning),
            "confidence_can_trigger_conservative_fallback": bool(
                can_assist_fallback
                and (calibrated_probability_available or allow_uncalibrated_fallback)
            ),
            "confidence_decision": decision,
            "confidence_recommendation": recommendation,
            "confidence_decision_reasons": decision_reasons,
            "confidence_warning_flags": warning_flags,
            "confidence_policy_details": {
                "candidate_pockets": candidate_ids,
                "selected_pockets": selected_ids,
                "accepted_pockets": accepted_ids,
                "accepted_ratio": accepted_ratio,
                "feasible_fraction_mean": feasible_fraction_mean,
                "scan_score_gap_top1_top2": scan_gap,
                "selector_feasible_fraction_ci95_overlap_top1_top2": selector_feasible_ci_overlap,
                "prob_estimated_success": prob_est,
                "probability_semantics": probability_semantics,
                "calibrated_probability_available": calibrated_probability_available,
                "calibrated_probability_decision_capable": (
                    calibrated_probability_decision_capable
                ),
                "calibrated_probability_low": calibrated_probability_low,
                "artifact_probability_threshold": artifact_probability_threshold,
                "configured_probability_threshold": configured_probability_threshold,
                "effective_probability_threshold": effective_probability_threshold,
                "allow_uncalibrated_fallback": allow_uncalibrated_fallback,
                "high_uncertainty": high_uncertainty,
                "uncertainty_reasons": _coerce_string_list(uncertainty_reasons),
                "recommended_pruned_pockets": recommended_pruned,
                "apply_conservative_pruning": apply_pruning,
                "recommend_full_fallback": recommend_fallback,
                "apply_confidence_assisted_fallback": apply_fallback,
                "confidence_features_present": bool(features),
                "confidence_output_present": isinstance(confidence_output, Mapping),
                "legacy_thresholds": {
                    "min_accepted_ratio": min_accepted_ratio,
                    "min_feasible_fraction_mean": min_feasible_fraction_mean,
                    "min_scan_gap_top1_top2": min_scan_gap,
                    "min_calibrated_success_probability": _safe_float(
                        policy_cfg.get(
                            "min_calibrated_success_probability",
                            DEFAULT_MIN_CALIBRATED_SUCCESS_PROBABILITY,
                        )
                    ),
                },
                "population_confidence_available": popconf_available,
                "population_confidence_thresholds": popconf_thresholds,
                "population_confidence_condition_flags": popconf_conditions,
                "population_confidence_positive_signals": int(positive_popconf_signals),
                "population_confidence_negative_signals": int(negative_popconf_signals),
                "population_confidence_supports_acceptance": popconf_supports_acceptance,
                "population_confidence_requires_caution": popconf_requires_caution,
                "population_confidence_requires_fallback": popconf_requires_fallback,
                "population_confidence_excerpt": {
                    "dominant_pocket_candidate_fraction": popconf_dominant_fraction,
                    "best_vs_second_best_gap": popconf_best_vs_second_best_gap,
                    "stable_pockets_count": popconf_stable_pockets_count,
                    "uncertain_pockets_count": popconf_uncertain_pockets_count,
                    "best_pocket_converged_stably": popconf_best_pocket_converged_stably,
                    "best_pocket_viable_fraction_last": popconf_best_pocket_viable_fraction_last,
                    "best_pocket_viable_fraction_mean": popconf_best_pocket_viable_fraction_mean,
                    "best_pocket_elite_score_std_last": popconf_best_pocket_elite_score_std_last,
                    "best_pocket_elite_coords_dispersion_last": popconf_best_pocket_elite_coords_dispersion_last,
                    "best_pocket_plateau_generations_last_window": popconf_best_pocket_plateau_generations_last_window,
                    "best_pocket_uncertainty_count": popconf_best_pocket_uncertainty_count,
                },
                "confidence_decision": decision,
                "confidence_recommendation": recommendation,
                "confidence_decision_reasons": decision_reasons,
                "confidence_warning_flags": warning_flags,
            },
        }
    )
    return base
