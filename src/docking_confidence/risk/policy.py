from __future__ import annotations

import hashlib
import json
import math
import re

import numpy as np
import pandas as pd


_CLUSTER_RUN_METHOD = "cluster_bootstrap_run_risk"
_LEGACY_METHOD = "learn_then_test_bonferroni_binomial_iid"


def _uses_cluster_run_risk(cfg: dict) -> bool:
    method = cfg.get("method", _LEGACY_METHOD)
    if method not in {_LEGACY_METHOD, _CLUSTER_RUN_METHOD}:
        raise ValueError(f"unsupported risk policy method: {method!r}")
    return method == _CLUSTER_RUN_METHOD


def _log_binomial_probability(k: int, n: int, p: float) -> float:
    if p == 0.0:
        return 0.0 if k == 0 else -math.inf
    if p == 1.0:
        return 0.0 if k == n else -math.inf
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def _sum_log_probabilities(log_values: list[float]) -> float:
    finite = [value for value in log_values if math.isfinite(value)]
    if not finite:
        return 0.0
    maximum = max(finite)
    return min(1.0, math.exp(maximum) * sum(math.exp(value - maximum) for value in finite))


def binomial_cdf(k: int, n: int, p: float) -> float:
    """Return P(Binomial(n, p) <= k) using an explicit log-sum calculation."""
    if n < 0 or not 0.0 <= p <= 1.0:
        raise ValueError("invalid binomial parameters")
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return _sum_log_probabilities(
        [_log_binomial_probability(value, n, p) for value in range(k + 1)]
    )


def binomial_survival(k: int, n: int, p: float) -> float:
    """Return P(Binomial(n, p) >= k) without subtractive cancellation."""
    if n < 0 or not 0.0 <= p <= 1.0:
        raise ValueError("invalid binomial parameters")
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return _sum_log_probabilities(
        [_log_binomial_probability(value, n, p) for value in range(k, n + 1)]
    )


def clopper_pearson_upper(successes: int, trials: int, error_probability: float) -> float:
    """Exact one-sided upper confidence endpoint for a binomial proportion."""
    if not 0.0 < error_probability < 1.0:
        raise ValueError("error_probability must be strictly between 0 and 1")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid successes/trials")
    if successes == trials:
        return 1.0
    low = successes / trials
    high = 1.0
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if binomial_cdf(successes, trials, midpoint) > error_probability:
            low = midpoint
        else:
            high = midpoint
    return float((low + high) / 2.0)


def clopper_pearson_lower(successes: int, trials: int, error_probability: float) -> float:
    """Exact one-sided lower confidence endpoint for a binomial proportion."""
    if not 0.0 < error_probability < 1.0:
        raise ValueError("error_probability must be strictly between 0 and 1")
    if trials < 1 or not 0 <= successes <= trials:
        raise ValueError("invalid successes/trials")
    if successes == 0:
        return 0.0
    low = 0.0
    high = successes / trials
    for _ in range(80):
        midpoint = (low + high) / 2.0
        if binomial_survival(successes, trials, midpoint) < error_probability:
            low = midpoint
        else:
            high = midpoint
    return float((low + high) / 2.0)


def threshold_grid(cfg: dict) -> np.ndarray:
    minimum = float(cfg["threshold_min"])
    maximum = float(cfg["threshold_max"])
    step = float(cfg["threshold_step"])
    count = int(math.floor((maximum - minimum) / step + 1e-12))
    values = [minimum + index * step for index in range(count + 1)]
    if values[-1] < maximum - 1e-12:
        values.append(maximum)
    else:
        values[-1] = maximum
    return np.asarray([round(value, 12) for value in values], dtype=float)


def operational_acceptance_mask(
    probability_lower: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probability_lower = np.asarray(probability_lower, dtype=float).reshape(-1)
    eligible = np.asarray(eligible, dtype=bool).reshape(-1)
    if len(probability_lower) != len(eligible):
        raise ValueError("probability and eligibility vectors must have equal length")
    if not np.isfinite(probability_lower).all():
        raise ValueError("probability_lower must be finite")
    return eligible & (probability_lower >= float(threshold))


def _nonnegative_integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _validated_cluster_counts(records: list[dict]) -> list[dict]:
    if not isinstance(records, (list, tuple)) or not records:
        raise ValueError("cluster_counts must be a nonempty list of records")
    normalized = []
    fingerprints = set()
    count_fields = (
        "total_run_count",
        "eligible_run_count",
        "accepted_run_count",
        "false_accepted_run_count",
    )
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("each cluster_counts record must be an object")
        fingerprint = record.get("complex_fingerprint_sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("complex_fingerprint_sha256 must be a canonical SHA-256 hex digest")
        if fingerprint in fingerprints:
            raise ValueError("cluster_counts contains a duplicate complex_fingerprint_sha256")
        fingerprints.add(fingerprint)
        counts = {
            name: _nonnegative_integer(record.get(name), name)
            for name in count_fields
        }
        if not (
            0 <= counts["false_accepted_run_count"]
            <= counts["accepted_run_count"]
            <= counts["eligible_run_count"]
            <= counts["total_run_count"]
            and counts["total_run_count"] >= 1
        ):
            raise ValueError(
                "cluster_counts requires 0 <= false accepted <= accepted <= eligible "
                "<= total and total >= 1"
            )
        # Every resampled sum must fit in the integer accumulator, including when
        # a large cluster is drawn repeatedly. Reject corrupt artifact counts.
        if counts["total_run_count"] > np.iinfo(np.int64).max // len(records):
            raise ValueError("cluster_counts exceed the supported integer count range")
        normalized.append({"complex_fingerprint_sha256": fingerprint, **counts})
    return sorted(normalized, key=lambda item: item["complex_fingerprint_sha256"])


def assess_cluster_counts(
    records: list[dict],
    threshold: float,
    *,
    cfg: dict,
    hypothesis_count: int,
    role: str,
) -> dict:
    """Replay run-risk evidence from whole-complex sufficient statistics.

    The target is E[false accepted runs per complex] / E[accepted runs per
    complex]. The percentile bootstrap is approximate; only the separate
    complex-coverage endpoint retains its exact binomial interpretation.
    The fitted scores, eligibility rule and threshold are held fixed here.
    """
    if not _uses_cluster_run_risk(cfg):
        raise ValueError("assess_cluster_counts requires cluster_bootstrap_run_risk")
    cluster_counts = _validated_cluster_counts(records)
    hypotheses = _nonnegative_integer(hypothesis_count, "hypothesis_count", minimum=1)
    repetitions = _nonnegative_integer(
        cfg.get("bootstrap_repetitions", 10_000), "bootstrap_repetitions", minimum=1_000
    )
    if repetitions > 1_000_000:
        raise ValueError("bootstrap_repetitions must be at most 1000000")
    seed = _nonnegative_integer(cfg.get("bootstrap_seed", 20_260_909), "bootstrap_seed")
    confidence = float(cfg["familywise_confidence_level"])
    risk_fraction = float(cfg["risk_error_budget_fraction"])
    risk_target = float(cfg["risk_target"])
    minimum_coverage = float(cfg["minimum_complex_coverage"])
    if not 0.0 < confidence < 1.0 or not 0.0 < risk_fraction < 1.0:
        raise ValueError("confidence level and risk error budget fraction must be in (0, 1)")
    if not 0.0 <= risk_target <= 1.0 or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("risk target and minimum complex coverage must be in [0, 1]")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    minimum_evaluated = _nonnegative_integer(
        cfg["minimum_evaluated_complexes"], "minimum_evaluated_complexes", minimum=1
    )
    minimum_authorized = _nonnegative_integer(
        cfg["minimum_authorized_complexes"], "minimum_authorized_complexes", minimum=1
    )
    risk_error = (1.0 - confidence) * risk_fraction / hypotheses
    coverage_error = (1.0 - confidence) * (1.0 - risk_fraction) / hypotheses
    if risk_error <= 0.0 or coverage_error <= 0.0:
        raise ValueError("per-hypothesis error budgets must be positive")

    total_complexes = len(cluster_counts)
    accepted_counts = np.asarray(
        [item["accepted_run_count"] for item in cluster_counts], dtype=np.int64
    )
    false_counts = np.asarray(
        [item["false_accepted_run_count"] for item in cluster_counts], dtype=np.int64
    )
    total_rows = sum(item["total_run_count"] for item in cluster_counts)
    eligible_rows = sum(item["eligible_run_count"] for item in cluster_counts)
    accepted_rows = int(accepted_counts.sum())
    false_rows = int(false_counts.sum())
    accepted_complexes = int(np.count_nonzero(accepted_counts))
    adverse_complexes = int(np.count_nonzero(false_counts))
    empirical_risk = false_rows / accepted_rows if accepted_rows else None

    # Sorting above plus explicit PCG64 makes replay independent of input row
    # order. Keep every complex, including complexes with no accepted runs.
    rng = np.random.Generator(np.random.PCG64(seed))
    bootstrap_risks = np.empty(repetitions, dtype=float)
    zero_denominators = 0
    batch_size = max(1, min(256, 262_144 // total_complexes))
    for start in range(0, repetitions, batch_size):
        stop = min(start + batch_size, repetitions)
        sampled = rng.integers(0, total_complexes, size=(stop - start, total_complexes))
        denominator = accepted_counts[sampled].sum(axis=1)
        numerator = false_counts[sampled].sum(axis=1)
        # An undefined replicate stays in the empirical distribution at the
        # pessimistic endpoint; dropping it would condition away uncertainty.
        risks = np.ones(stop - start, dtype=float)
        np.divide(numerator, denominator, out=risks, where=denominator > 0)
        bootstrap_risks[start:stop] = risks
        zero_denominators += int(np.count_nonzero(denominator == 0))

    constant_cluster_ratio = accepted_rows > 0 and all(
        item["false_accepted_run_count"] * accepted_rows
        == item["accepted_run_count"] * false_rows
        for item in cluster_counts
    )
    if accepted_rows == 0:
        bootstrap_status = "no_accepted_runs"
    elif false_rows == 0:
        bootstrap_status = "no_observed_failures"
    elif constant_cluster_ratio or np.ptp(bootstrap_risks) == 0.0:
        bootstrap_status = "degenerate_cluster_risk"
    elif repetitions * risk_error < 20.0:
        # This is a Monte Carlo quantile-resolution requirement, not a required
        # number of observed failures or a finite-sample coverage guarantee.
        bootstrap_status = "insufficient_bootstrap_tail_resolution"
    else:
        bootstrap_status = "ok"
    risk_upper = (
        max(float(empirical_risk), float(np.quantile(
            bootstrap_risks, 1.0 - risk_error, method="higher"
        )))
        if bootstrap_status == "ok"
        else 1.0
    )
    coverage_lower = clopper_pearson_lower(
        accepted_complexes, total_complexes, coverage_error
    )
    supported = (
        bootstrap_status == "ok"
        and risk_upper <= risk_target
        and coverage_lower >= minimum_coverage
        and total_complexes >= minimum_evaluated
        and accepted_complexes >= minimum_authorized
    )
    return {
        "role": role,
        "supported": bool(supported),
        "threshold": float(threshold),
        "tested_hypothesis_count": hypotheses,
        "risk_target": risk_target,
        "familywise_confidence_level": confidence,
        "risk_error_probability_per_hypothesis": float(risk_error),
        "coverage_error_probability_per_hypothesis": float(coverage_error),
        "total_row_count": total_rows,
        "accepted_row_count": accepted_rows,
        "eligible_row_count": eligible_rows,
        "false_acceptance_row_count": false_rows,
        "total_complex_count": total_complexes,
        "accepted_complex_count": accepted_complexes,
        "adverse_accepted_complex_count": adverse_complexes,
        "empirical_run_risk": empirical_risk,
        "empirical_conditional_complex_risk": (
            adverse_complexes / accepted_complexes if accepted_complexes else None
        ),
        "simultaneous_risk_upper_bound": risk_upper,
        "risk_bound_method": "cluster_bootstrap_percentile",
        "confidence_guarantee": "asymptotic_not_finite_sample",
        "bootstrap_status": bootstrap_status,
        "bootstrap_repetitions": repetitions,
        "bootstrap_completed_repetitions": repetitions,
        "bootstrap_seed": seed,
        "bootstrap_rng": "PCG64",
        "bootstrap_quantile_method": "higher",
        "bootstrap_zero_denominator_repetitions": zero_denominators,
        "bootstrap_zero_denominator_fraction": zero_denominators / repetitions,
        "bootstrap_expected_tail_repetitions": repetitions * risk_error,
        "bootstrap_minimum_tail_repetitions": 20,
        "largest_accepted_cluster_share": (
            int(accepted_counts.max()) / accepted_rows if accepted_rows else None
        ),
        "cluster_counts": cluster_counts,
        "row_coverage": accepted_rows / total_rows,
        "complex_coverage": accepted_complexes / total_complexes,
        "simultaneous_complex_coverage_lower_bound": float(coverage_lower),
        "minimum_complex_coverage": minimum_coverage,
        "minimum_evaluated_complexes": minimum_evaluated,
        "minimum_authorized_complexes": minimum_authorized,
    }


def assess_threshold(
    df: pd.DataFrame,
    probability_lower: np.ndarray,
    eligible: np.ndarray,
    threshold: float,
    *,
    cfg: dict,
    hypothesis_count: int,
    role: str,
) -> dict:
    if len(df) == 0:
        raise ValueError("policy assessment dataset cannot be empty")
    if "complex_fingerprint_sha256" not in df.columns:
        raise ValueError(
            "policy assessment requires complex_fingerprint_sha256 as its "
            "independent grouping unit"
        )
    if hypothesis_count < 1:
        raise ValueError("hypothesis_count must be positive")
    if len(probability_lower) != len(df) or len(eligible) != len(df):
        raise ValueError("policy vectors must align with the dataframe")

    accepted = operational_acceptance_mask(probability_lower, eligible, threshold)
    if _uses_cluster_run_risk(cfg):
        if not df["preserved"].isin([0, 1]).all():
            raise ValueError("preserved labels must be binary and nonmissing")
        probabilities = np.asarray(probability_lower, dtype=float)
        if not ((probabilities >= 0.0) & (probabilities <= 1.0)).all():
            raise ValueError("probability_lower must be in [0, 1]")
        run_bundle = pd.DataFrame(
            {
                "complex_fingerprint_sha256": df["complex_fingerprint_sha256"].to_numpy(),
                "eligible": np.asarray(eligible, dtype=bool).reshape(-1),
                "accepted": accepted,
                "adverse": accepted & (df["preserved"].to_numpy() == 0),
            }
        )
        # Keep missing identifiers visible so canonical-count validation rejects
        # them instead of silently dropping those runs during groupby.
        grouped_counts = run_bundle.groupby(
            "complex_fingerprint_sha256", sort=False, dropna=False
        ).agg(
            total_run_count=("accepted", "size"),
            eligible_run_count=("eligible", "sum"),
            accepted_run_count=("accepted", "sum"),
            false_accepted_run_count=("adverse", "sum"),
        )
        return assess_cluster_counts(
            grouped_counts.reset_index().to_dict("records"),
            threshold,
            cfg=cfg,
            hypothesis_count=hypothesis_count,
            role=role,
        )
    bundle = pd.DataFrame(
        {
            "complex_fingerprint_sha256": (
                df["complex_fingerprint_sha256"].astype(str).to_numpy()
            ),
            "accepted": accepted,
            "adverse": accepted & (df["preserved"].astype(int).to_numpy() == 0),
        }
    )
    grouped = bundle.groupby("complex_fingerprint_sha256", sort=True).agg(
        accepted=("accepted", "any"),
        adverse=("adverse", "any"),
    )
    total_complexes = int(len(grouped))
    accepted_complexes = int(grouped["accepted"].sum())
    adverse_complexes = int((grouped["accepted"] & grouped["adverse"]).sum())

    confidence = float(cfg["familywise_confidence_level"])
    total_error = 1.0 - confidence
    risk_fraction = float(cfg["risk_error_budget_fraction"])
    risk_error = total_error * risk_fraction / hypothesis_count
    coverage_error = total_error * (1.0 - risk_fraction) / hypothesis_count
    risk_target = float(cfg["risk_target"])

    if accepted_complexes:
        empirical_risk: float | None = adverse_complexes / accepted_complexes
        risk_p_value = binomial_cdf(
            adverse_complexes, accepted_complexes, risk_target
        )
        risk_upper = clopper_pearson_upper(
            adverse_complexes, accepted_complexes, risk_error
        )
    else:
        empirical_risk = None
        risk_p_value = 1.0
        risk_upper = 1.0

    complex_coverage = accepted_complexes / total_complexes
    coverage_lower = clopper_pearson_lower(
        accepted_complexes, total_complexes, coverage_error
    )
    row_coverage = float(accepted.mean())
    supported = (
        risk_p_value <= risk_error
        and coverage_lower >= float(cfg["minimum_complex_coverage"])
        and total_complexes >= int(cfg["minimum_evaluated_complexes"])
        and accepted_complexes >= int(cfg["minimum_authorized_complexes"])
    )

    return {
        "role": role,
        "supported": bool(supported),
        "threshold": float(threshold),
        "tested_hypothesis_count": int(hypothesis_count),
        "risk_target": risk_target,
        "familywise_confidence_level": confidence,
        "risk_error_probability_per_hypothesis": float(risk_error),
        "coverage_error_probability_per_hypothesis": float(coverage_error),
        "risk_p_value": float(risk_p_value),
        "risk_p_value_bonferroni": float(min(1.0, risk_p_value * hypothesis_count)),
        "accepted_row_count": int(accepted.sum()),
        "eligible_row_count": int(np.asarray(eligible, dtype=bool).sum()),
        "false_acceptance_row_count": int(
            (accepted & (df["preserved"].astype(int).to_numpy() == 0)).sum()
        ),
        "total_complex_count": total_complexes,
        "accepted_complex_count": accepted_complexes,
        "adverse_accepted_complex_count": adverse_complexes,
        "empirical_conditional_complex_risk": empirical_risk,
        "simultaneous_risk_upper_bound": float(risk_upper),
        "row_coverage": row_coverage,
        "complex_coverage": float(complex_coverage),
        "simultaneous_complex_coverage_lower_bound": float(coverage_lower),
        "minimum_complex_coverage": float(cfg["minimum_complex_coverage"]),
        "minimum_evaluated_complexes": int(cfg["minimum_evaluated_complexes"]),
        "minimum_authorized_complexes": int(cfg["minimum_authorized_complexes"]),
    }


def select_policy(
    df: pd.DataFrame,
    probability_lower: np.ndarray,
    eligible: np.ndarray,
    cfg: dict,
) -> tuple[dict, list[dict]]:
    cluster_run_risk = _uses_cluster_run_risk(cfg)
    candidates = threshold_grid(cfg)
    assessments = [
        assess_threshold(
            df,
            probability_lower,
            eligible,
            float(candidate),
            cfg=cfg,
            hypothesis_count=len(candidates),
            role="risk_calibration",
        )
        for candidate in candidates
    ]
    supported = [item for item in assessments if item["supported"]]
    grid_payload = json.dumps(candidates.tolist(), separators=(",", ":")).encode("ascii")
    common = {
        "method": (
            "cluster_bootstrap_run_risk_v1"
            if cluster_run_risk
            else "learn_then_test_bonferroni_binomial_iid_v1"
        ),
        "loss_definition": (
            "expected false accepted REDUCED runs per complex divided by expected "
            "accepted REDUCED runs per complex; run-weighted error among accepted decisions"
            if cluster_run_risk else
            "conditional probability that a complex has at least one non-preserved "
            "accepted REDUCED run, given that the complex has at least one accepted "
            "REDUCED run"
        ),
        "calibration_unit": "complex_fingerprint_sha256",
        "statistical_assumptions": (
            "complexes are independent and identically distributed (i.i.d.); the "
            "within-complex run-generation and aggregation protocol is fixed; and "
            "the scoring, eligibility, candidate grid, and evaluation pipeline are "
            "frozen before observing the partition's outcomes"
        ),
        "decision_score": "venn_abers_probability_lower",
        "candidate_grid": candidates.tolist(),
        "candidate_grid_sha256": hashlib.sha256(grid_payload).hexdigest(),
        "tested_hypothesis_count": int(len(candidates)),
        "selection_objective": (
            "maximize simultaneous complex-coverage lower bound; then minimize "
            "risk upper bound; then choose the larger threshold"
        ),
    }
    if cluster_run_risk:
        common.update(
            risk_bound_method="cluster_bootstrap_percentile",
            confidence_guarantee="asymptotic_not_finite_sample",
            risk_weighting=(
                "ratio of pooled run counts, not an average of per-complex rates; "
                "all canonical complexes are resampled with their complete run counts, "
                "including complexes with no accepted runs; future use must follow "
                "the same within-complex algorithm, repetition and inclusion procedure"
            ),
        )
    if not supported:
        diagnostic = min(
            assessments,
            key=lambda item: (
                item["simultaneous_risk_upper_bound"],
                -item["accepted_complex_count"],
                -item["threshold"],
            ),
        )
        return (
            {
                **common,
                "supported": False,
                "threshold": None,
                "reason": "no predeclared threshold passed every simultaneous risk and coverage gate",
                "best_observed_candidate": diagnostic,
            },
            assessments,
        )

    selected = max(
        supported,
        key=lambda item: (
            item["simultaneous_complex_coverage_lower_bound"],
            -item["simultaneous_risk_upper_bound"],
            item["threshold"],
        ),
    )
    return (
        {
            **common,
            **selected,
            "reason": (
                "threshold passed approximate cluster-bootstrap run-risk and exact "
                "complex-coverage gates under the recorded i.i.d.-complex assumptions"
                if cluster_run_risk else
                "threshold passed simultaneous binomial risk and coverage gates "
                "under the recorded i.i.d.-complex assumptions"
            ),
        },
        assessments,
    )


def audit_locked_policy(
    df: pd.DataFrame,
    probability_lower: np.ndarray,
    eligible: np.ndarray,
    threshold: float | None,
    cfg: dict,
) -> dict:
    if threshold is None:
        return {
            "role": "locked_test",
            "supported": False,
            "threshold": None,
            "reason": "risk-calibration stage did not select a threshold",
        }
    result = assess_threshold(
        df,
        probability_lower,
        eligible,
        float(threshold),
        cfg=cfg,
        hypothesis_count=1,
        role="locked_test",
    )
    result["reason"] = (
        "locked threshold passed independent test gates"
        if result["supported"]
        else "locked threshold failed at least one independent test gate"
    )
    return result
