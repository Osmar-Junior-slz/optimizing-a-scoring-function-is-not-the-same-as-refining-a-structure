from __future__ import annotations

import numpy as np
import pandas as pd


def operational_cost_assessment(
    df: pd.DataFrame, accepted_mask: np.ndarray, *, overhead_sec: float | None = None,
) -> dict:
    """Paired-row cost simulation, including the FULL fallback after REDUCED.

    This diagnostic does not replace the historical accepted-row runtime gate.
    Failed/unpaired runs are unavailable here and must be added from batch receipts.
    """
    raw_mask = np.asarray(accepted_mask).reshape(-1)
    if not np.isin(raw_mask, [False, True]).all():
        raise ValueError("operational cost acceptance mask must be explicit binary flags")
    accepted = raw_mask.astype(bool)
    if len(accepted) != len(df):
        raise ValueError("operational cost acceptance mask must align with the dataframe")
    for column in ("runtime_reduced_sec", "runtime_full_sec"):
        if column not in df:
            return {"available": False, "reason": "paired_times_absent"}
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("operational cost runtimes must be finite and nonnegative")
    if overhead_sec is not None and (isinstance(overhead_sec, bool) or not np.isfinite(overhead_sec) or overhead_sec < 0):
        raise ValueError("overhead_sec must be finite and nonnegative")
    baseline = float(df["runtime_full_sec"].sum())
    reduced = float(df["runtime_reduced_sec"].sum())
    fallback = float(df.loc[~accepted, "runtime_full_sec"].sum())
    cost = reduced + fallback + (overhead_sec or 0)
    return {
        "available": bool(len(df)), "reason": None if len(df) else "no_paired_rows",
        "scope": "paired_rows_frozen_policy_simulation",
        "row_count": len(df), "accepted_row_count": int(accepted.sum()),
        "fallback_row_count": int((~accepted).sum()),
        "baseline_full_sec": baseline, "all_reduced_sec": reduced, "fallback_full_sec": fallback,
        "overhead_sec": overhead_sec, "paired_strategy_sec": cost,
        "saved_sec": baseline - cost,
        "speedup_full_over_strategy": baseline / cost if cost > 0 else None,
        "includes_failed_or_unpaired_runs": False,
        "overhead_included": overhead_sec is not None,
        "interpretation": "hypothetical policy acceptance; sums per-run engine times, not parallel wall time; unknown overhead and unpaired failures require separate accounting",
    }


def runtime_assessment(
    df: pd.DataFrame,
    cfg: dict,
    accepted_mask: np.ndarray | None = None,
) -> dict:
    if cfg.get("assessment_scope") == "total_paired_cascade":
        return cascade_runtime_assessment(df, cfg, accepted_mask)
    if accepted_mask is None:
        accepted_mask = np.ones(len(df), dtype=bool)
    accepted_mask = np.asarray(accepted_mask, dtype=bool).reshape(-1)
    if len(accepted_mask) != len(df):
        raise ValueError("runtime acceptance mask must align with the dataframe")
    assessed = df.loc[accepted_mask]
    full = float(assessed["runtime_full_sec"].sum())
    reduced = float(assessed["runtime_reduced_sec"].sum())
    speedup = full / reduced if reduced > 0 else 0.0
    enabled = cfg.get("enabled") is True
    require = cfg.get("require_reduced_faster_than_full") is True
    minimum = float(cfg.get("minimum_speedup", 1.0))
    passed = len(assessed) > 0
    if enabled and require:
        passed = passed and speedup >= minimum
    elif not enabled:
        # A disabled gate is unavailable evidence, not a successful validation.
        passed = False
    return {
        "enabled": enabled,
        "assessment_scope": "locked_test_rows_accepted_by_domain_and_policy",
        "assessed_row_count": int(len(assessed)),
        "full_runtime_sec": full,
        "reduced_runtime_sec": reduced,
        "speedup_full_over_reduced": float(speedup),
        "minimum_speedup": minimum,
        "passed": bool(passed),
        "reason": (
            "runtime gate passed on locked accepted rows"
            if passed
            else "runtime gate disabled or failed on locked accepted rows"
        ),
    }


def cascade_runtime_assessment(df, cfg, accepted_mask=None) -> dict:
    """Net cost on every paired row; unknown overhead cannot certify savings.

    The absent assessment_scope keeps historical artifacts on their exact legacy
    replay path. New local profiles explicitly opt into this stricter scope.
    Failed/unpaired jobs still require separate receipt-level accounting.
    """
    accepted = np.ones(len(df), dtype=bool) if accepted_mask is None else np.asarray(accepted_mask)
    if accepted.ndim != 1 or len(accepted) != len(df) or not np.isin(accepted, [False, True]).all():
        raise ValueError("cascade acceptance mask must contain aligned binary flags")
    accepted = accepted.astype(bool)
    column = cfg.get("overhead_column", "confidence_overhead_sec")
    measured = column in df
    overhead = None
    if measured:
        values = df[column].to_numpy(dtype=float)
        if np.isinf(values).any() or (values < 0).any():
            raise ValueError("cascade overhead must be nonnegative and cannot be infinite")
        measured = bool(np.isfinite(values).all())
        overhead = float(values.sum()) if measured else None
    cost = operational_cost_assessment(df, accepted, overhead_sec=overhead)
    if not cost["available"] and len(df):
        raise ValueError("total cascade assessment requires paired runtime columns")
    baseline = cost.get("baseline_full_sec", 0.0)
    strategy = cost.get("paired_strategy_sec", 0.0)
    speedup = baseline / strategy if strategy > 0 else 0.0
    minimum = float(cfg.get("minimum_speedup", 1.0))
    enabled = cfg.get("enabled") is True
    evidence_available = measured or cfg.get("require_measured_overhead", True) is False
    passed = bool(enabled and len(df) and accepted.any() and strategy > 0 and evidence_available
                  and cost.get("saved_sec", 0) > 0
                  and (not cfg.get("require_reduced_faster_than_full", True) or speedup >= minimum))
    return {
        "enabled": enabled, "assessment_scope": "locked_test_total_paired_cascade_v1",
        "assessed_row_count": len(df), "accepted_row_count": int(accepted.sum()),
        "full_runtime_sec": baseline, "reduced_runtime_sec": cost.get("all_reduced_sec", 0.0),
        "fallback_full_sec": cost.get("fallback_full_sec", 0.0),
        "overhead_column": column, "overhead_measured": measured,
        "overhead_sec": overhead, "cascade_runtime_sec": strategy,
        "speedup_full_over_reduced": float(speedup),
        "speedup_denominator": "cascade_runtime_sec",
        "speedup_full_over_strategy": float(speedup),
        "saving_upper_bound_sec": cost.get("saved_sec", 0.0),
        "measured_saving_sec": cost.get("saved_sec") if measured else None,
        "includes_failed_or_unpaired_runs": False,
        "minimum_speedup": minimum, "passed": passed,
        "reason": ("overhead_not_measured" if not evidence_available else
                   "net paired cascade saving passed" if passed else
                   "no demonstrated net paired cascade saving"),
    }
