from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from docking_confidence.application.domain import FeatureDomain
from docking_confidence.application.progress import ProgressCallback, emit_progress
from docking_confidence.application.probability_reporting import (
    DiagnosticOptions, descriptor_comparison, saved_test_diagnostics,
)
from docking_confidence.application.runtime import operational_cost_assessment, runtime_assessment
from docking_confidence.artifact.store import write_json
from docking_confidence.audit.manifest import build_manifest, read_bytes_once, sha256_bytes
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.io.config import (
    ARTIFACT_SCHEMA_VERSION, CONFIG_DIGEST_SCHEMA, config_sha256, validate_config,
)
from docking_confidence.io.dataset import (
    assess_dataset_provenance,
    load_paired_dataset,
)
from docking_confidence.modeling.logistic import LogisticBaseModel
from docking_confidence.risk.policy import (
    audit_locked_policy,
    operational_acceptance_mask,
    select_policy,
)
from docking_confidence.validation.metrics import probability_metrics
from docking_confidence.validation.split import four_way_group_partition


_OUTPUT_NAMES = {
    "artifact": "artifact.json",
    "report": "validation_report.json",
    "probability_calibration": "probability_calibration_predictions.csv",
    "risk_calibration": "risk_calibration_predictions.csv",
    "locked_test": "locked_test_predictions.csv",
    "policy_candidates": "policy_candidates.csv",
    "config": "resolved_config.json",
    "manifest": "audit_manifest.json",
}


class TrainingEvidenceError(ValueError):
    """An existing scientific gate failed; the recorded evidence is diagnostic."""

    def __init__(self, message: str, *, stage: str, evidence: dict):
        super().__init__(message)
        self.stage = stage
        self.evidence = evidence


def _software_versions() -> dict:
    packages = {}
    for package in ("numpy", "pandas", "scikit-learn", "PyYAML"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    return {"python": platform.python_version(), "packages": packages}


def _source_code_sha256() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        payload = read_bytes_once(source)
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _interval_summary(lower: np.ndarray, upper: np.ndarray) -> dict:
    width = np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)
    return {
        "mean_width": float(width.mean()),
        "median_width": float(np.median(width)),
        "maximum_width": float(width.max()),
    }


def _prediction_frame(
    df: pd.DataFrame,
    indices: np.ndarray,
    X: np.ndarray,
    model: LogisticBaseModel,
    calibrator: VennAbersCalibrator,
    domain: FeatureDomain,
    feature_columns: list[str],
    role: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    role_X = X[indices]
    raw = model.raw_score(role_X)
    trace = calibrator.predict_with_trace(raw)
    lower, upper, point = trace["lower"], trace["upper"], trace["point"]
    eligible = domain.contains(role_X)
    output = df.iloc[indices].copy().reset_index(drop=True)
    output["partition"] = role
    output["base_raw_score"] = raw
    output["probability_lower"] = lower
    output["probability_upper"] = upper
    output["probability"] = point
    output["probability_width"] = upper - lower
    for trace_name in (
        "lower_positive_count",
        "lower_weight",
        "upper_positive_count",
        "upper_weight",
        "point_numerator",
        "point_denominator",
    ):
        output[f"venn_abers_{trace_name}"] = trace[trace_name]
    output["in_applicability_domain"] = eligible

    exported = model.export(feature_columns)
    mean = np.asarray(exported["scaler_mean"], dtype=float)
    scale = np.asarray(exported["scaler_scale"], dtype=float)
    coefficients = np.asarray(exported["coef"], dtype=float)
    contributions = ((role_X - mean) / scale) * coefficients
    for column_index, feature in enumerate(feature_columns):
        output[f"score_contribution__{feature}"] = contributions[:, column_index]
    return output, lower, upper, point, eligible


def train(
    dataset_path: str, config: dict, out_dir: str,
    *, expected_dataset_sha256: str | None = None,
    progress_callback: ProgressCallback | None = None,
    development_only: bool = False,
) -> dict:
    emit_progress(progress_callback, 0.0, "Validando dataset e configuração de treinamento")
    config = validate_config(config)
    if not isinstance(development_only, bool) or (development_only and config["authorization"]["allow_automatic_decision"]):
        raise ValueError("development-only fitting must keep automatic decisions disabled")
    dataset_file = Path(dataset_path).resolve()
    dataset_snapshot = read_bytes_once(dataset_file)
    dataset_snapshot_sha256 = sha256_bytes(dataset_snapshot)
    if expected_dataset_sha256 is not None and dataset_snapshot_sha256 != expected_dataset_sha256:
        raise ValueError("training dataset snapshot differs from the validated preparation SHA-256")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {key: out / name for key, name in _OUTPUT_NAMES.items()}
    collisions = [path.name for path in paths.values() if path.exists()]
    if collisions:
        raise FileExistsError(
            "training outputs are immutable; choose a new out-dir because these files "
            f"already exist: {collisions}"
        )

    feature_columns = list(config["features"]["columns"])
    df = load_paired_dataset(
        BytesIO(dataset_snapshot),
        feature_columns,
        contact_graph_cfg=config["contact_graph"],
        docking_task=config.get("docking_task"),
        feature_schema=config["features"]["schema_version"],
    )
    training_context = {}
    if "docking_task" in df.columns:
        if config.get("docking_task") != df["docking_task"].iloc[0]:
            raise ValueError("contextual dataset requires an explicit matching configuration docking_task")
        effective_digest = config_sha256(config)
        if df["config_sha256"].iloc[0] != effective_digest:
            raise ValueError("dataset config_sha256 disagrees with the effective training configuration")
        training_context = {
            "docking_task": df["docking_task"].iloc[0],
            "feature_schema": df["feature_schema"].iloc[0],
            "training_config_source": {
                "path": df["config"].iloc[0],
                "sha256": effective_digest,
                "digest_schema": CONFIG_DIGEST_SCHEMA,
            },
        }
    provenance = assess_dataset_provenance(df, config["authorization"])

    rule_config = config["preservation"]
    rule = PreservationRule(
        minimum_reduced_dockq=float(rule_config["minimum_reduced_dockq"]),
        maximum_dockq_loss=float(rule_config["maximum_dockq_loss"]),
    )
    df["preserved"] = [
        rule.label(reduced, full)
        for reduced, full in zip(df["dockq_reduced"], df["dockq_full"])
    ]
    if df["preserved"].nunique() < 2:
        raise TrainingEvidenceError(
            "training requires both preserved and non-preserved examples",
            stage="dataset_labels",
            evidence={"positive_count": int(df["preserved"].sum()),
                      "negative_count": int(len(df) - df["preserved"].sum()),
                      "profile": config["authorization"]["profile"]},
        )

    emit_progress(progress_callback, 0.10, "Separando complexos nas quatro partições predeclaradas")
    X = df[feature_columns].to_numpy(dtype=float)
    y = df["preserved"].to_numpy(dtype=int)
    groups = df["complex_fingerprint_sha256"].to_numpy(dtype=str)
    validation_config = config["validation"]
    try:
        split = four_way_group_partition(
            y,
            groups,
            split_folds=int(validation_config["split_folds"]),
            random_seed=int(validation_config["random_seed"]),
        )
    except ValueError as exc:
        raise TrainingEvidenceError(
            str(exc), stage="predeclared_split",
            evidence={"independent_complex_count": int(len(np.unique(groups))),
                      "split_folds": int(validation_config["split_folds"]),
                      "random_seed": int(validation_config["random_seed"]),
                      "profile": config["authorization"]["profile"]},
        ) from exc

    emit_progress(progress_callback, 0.20, "Treinando regressão logística")
    model_config = config["model"]
    class_weight = model_config.get("class_weight")
    if class_weight == "none":
        class_weight = None
    model = LogisticBaseModel(
        C=float(model_config["C"]),
        class_weight=class_weight,
        max_iter=int(model_config["max_iter"]),
    ).fit(X[split.model_train], y[split.model_train])

    emit_progress(progress_callback, 0.35, "Ajustando domínio de aplicabilidade")
    domain_config = config["applicability_domain"]
    domain = FeatureDomain(
        lower_quantile=float(domain_config["lower_quantile"]),
        upper_quantile=float(domain_config["upper_quantile"]),
    ).fit(X[split.model_train])

    emit_progress(progress_callback, 0.45, "Verificando evidência e calibrando probabilidades com Venn–Abers")
    probability_indices = split.probability_calibration
    probability_in_domain = domain.contains(X[probability_indices])
    calibration_config = config["calibration"]
    fit_all = calibration_config.get("fit_scope", "in_applicability_domain") == "all_calibration_rows"
    used_probability_indices = probability_indices if fit_all else probability_indices[probability_in_domain]
    used_probability_frame = df.iloc[used_probability_indices]
    independent_complex_count = int(
        used_probability_frame["complex_fingerprint_sha256"].nunique()
    )
    positive_complex_count = int(
        used_probability_frame.loc[
            used_probability_frame["preserved"] == 1,
            "complex_fingerprint_sha256",
        ].nunique()
    )
    negative_complex_count = int(
        used_probability_frame.loc[
            used_probability_frame["preserved"] == 0,
            "complex_fingerprint_sha256",
        ].nunique()
    )
    minimum_complexes = int(calibration_config["minimum_independent_complexes"])
    minimum_positive = int(calibration_config["minimum_positive_class_complexes"])
    minimum_negative = int(calibration_config["minimum_negative_class_complexes"])
    calibration_evidence_passed = not (
        independent_complex_count < minimum_complexes
        or positive_complex_count < minimum_positive
        or negative_complex_count < minimum_negative
    )
    if not calibration_evidence_passed and not development_only:
        raise TrainingEvidenceError(
            "probability-calibration partition failed its independent-complex "
            "evidence gate; the predeclared split is fail-closed: "
            f"total={independent_complex_count}/{minimum_complexes}, "
            f"positive={positive_complex_count}/{minimum_positive}, "
            f"negative={negative_complex_count}/{minimum_negative}",
            stage="probability_calibration",
            evidence={
                "independent_complex_count": independent_complex_count,
                "minimum_independent_complexes": minimum_complexes,
                "positive_class_complex_count": positive_complex_count,
                "minimum_positive_class_complexes": minimum_positive,
                "negative_class_complex_count": negative_complex_count,
                "minimum_negative_class_complexes": minimum_negative,
                "in_domain_row_count": int(probability_in_domain.sum()),
                "fit_row_count": int(len(used_probability_indices)),
                "partition_row_count": int(len(probability_indices)),
                "profile": config["authorization"]["profile"],
                "split": split.export(),
                "model_trained": True,
                "probability_calibrated": False,
            },
        )
    calibrator = VennAbersCalibrator().fit(
        model.raw_score(X[used_probability_indices]),
        y[used_probability_indices],
    )

    probability_frame, probability_lower, probability_upper, probability_point, probability_eligible = (
        _prediction_frame(
            df,
            probability_indices,
            X,
            model,
            calibrator,
            domain,
            feature_columns,
            "probability_calibration",
        )
    )
    probability_frame["used_to_fit_venn_abers"] = np.isin(probability_indices, used_probability_indices)

    cluster_run_risk = config["policy"]["method"] == "cluster_bootstrap_run_risk"
    emit_progress(progress_callback, 0.60,
                  "Avaliando risco por execução com bootstrap por complexo"
                  if cluster_run_risk else "Calibrando controle de risco e selecionando limiar LTT")
    risk_frame, risk_lower, risk_upper, risk_point, risk_eligible = _prediction_frame(
        df,
        split.risk_calibration,
        X,
        model,
        calibrator,
        domain,
        feature_columns,
        "risk_calibration",
    )
    policy, candidate_assessments = select_policy(
        risk_frame,
        risk_lower,
        risk_eligible,
        config["policy"],
    )
    selected_threshold = policy.get("threshold")
    if selected_threshold is None:
        risk_accepted = np.zeros(len(risk_frame), dtype=bool)
    else:
        risk_accepted = operational_acceptance_mask(
            risk_lower, risk_eligible, float(selected_threshold)
        )
    risk_frame["accepted_by_selected_policy"] = risk_accepted

    emit_progress(progress_callback, 0.75, "Avaliando política congelada no locked test")
    test_frame, test_lower, test_upper, test_point, test_eligible = _prediction_frame(
        df,
        split.locked_test,
        X,
        model,
        calibrator,
        domain,
        feature_columns,
        "locked_test",
    )
    test_audit = audit_locked_policy(
        test_frame,
        test_lower,
        test_eligible,
        selected_threshold,
        config["policy"],
    )
    if selected_threshold is None:
        test_accepted = np.zeros(len(test_frame), dtype=bool)
    else:
        test_accepted = operational_acceptance_mask(
            test_lower, test_eligible, float(selected_threshold)
        )
    test_frame["accepted_by_locked_policy"] = test_accepted
    emit_progress(progress_callback, 0.85, "Avaliando tempo de execução e gates de autorização")
    runtime = runtime_assessment(test_frame, config["runtime_gate"], test_accepted)

    authorization_config = config["authorization"]
    authorization_gates = {
        "profile_allows_automatic_decision": bool(
            authorization_config["allow_automatic_decision"]
        ),
        "scientific_provenance_passed": bool(provenance["passed"]),
        "risk_calibration_passed": bool(policy["supported"]),
        "locked_test_passed": bool(test_audit["supported"]),
        "runtime_gate_passed": bool(runtime["passed"]),
    }
    decision_capable = all(authorization_gates.values())

    emit_progress(progress_callback, 0.90, "Montando artefatos e relatório de auditoria")
    base_model_export = model.export(feature_columns)
    calibration_export = calibrator.export()
    calibration_export["independent_complex_count"] = independent_complex_count
    calibration_export["positive_class_complex_count"] = positive_complex_count
    calibration_export["negative_class_complex_count"] = negative_complex_count
    split_export = split.export()
    known_complexes = sorted(df["complex_id"].astype(str).unique().tolist())
    known_fingerprints = sorted(
        df["complex_fingerprint_sha256"].astype(str).unique().tolist()
    )
    development_summary = {
        "n_rows": int(len(df)),
        "n_complexes": int(len(known_fingerprints)),
        "positive_count": int(y.sum()),
        "negative_count": int(len(y) - y.sum()),
        "role_row_counts": {
            "model_train": int(len(split.model_train)),
            "probability_calibration": int(len(split.probability_calibration)),
            "risk_calibration": int(len(split.risk_calibration)),
            "locked_test": int(len(split.locked_test)),
        },
        "probability_calibration_evidence": {
            "in_domain_row_count": int(probability_in_domain.sum()),
            "fit_row_count": int(len(used_probability_indices)),
            "independent_complex_count": independent_complex_count,
            "positive_class_complex_count": positive_complex_count,
            "negative_class_complex_count": negative_complex_count,
        },
        "locked_test_probability_metrics": probability_metrics(
            test_frame["preserved"].to_numpy(dtype=int), test_point
        ),
        "locked_test_interval_summary": _interval_summary(test_lower, test_upper),
        "risk_calibration_probability_metrics": probability_metrics(
            risk_frame["preserved"].to_numpy(dtype=int), risk_point
        ),
        "risk_calibration_interval_summary": _interval_summary(risk_lower, risk_upper),
        "probability_calibration_trace_interval_summary": _interval_summary(
            probability_lower, probability_upper
        ),
    }

    artifact = {
        **training_context,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "auditable_docking_confidence_gate",
        "development_only": development_only,
        "calibration_evidence_passed": calibration_evidence_passed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "system": "DockingConfidenceStandalone",
        "decision_capable": bool(decision_capable),
        "authorization": {
            "profile": authorization_config["profile"],
            "decision_capable": bool(decision_capable),
            "gates": authorization_gates,
        },
        "target": {
            "name": "preserved",
            "positive_class": 1,
            "meaning": "REDUCED docking preserved the configured structural-quality rule",
        },
        "feature_columns": feature_columns,
        "feature_provenance": "reduced_only",
        "contact_graph": dict(config["contact_graph"]),
        "preservation_rule": {
            "minimum_reduced_dockq": rule.minimum_reduced_dockq,
            "maximum_dockq_loss": rule.maximum_dockq_loss,
        },
        "base_model": base_model_export,
        "calibration": calibration_export,
        "applicability_domain": {
            **domain.export(),
            "block_feature_extrapolation": bool(
                domain_config["block_feature_extrapolation"]
            ),
        },
        "policy": policy,
        "locked_test_audit": test_audit,
        "runtime_gate": runtime,
        "dataset_provenance": provenance,
        "development_split": split_export,
        "known_development_complex_ids": known_complexes,
        "known_development_complex_fingerprints": known_fingerprints,
        "resolved_config": config,
        "software_versions": _software_versions(),
        "source_code_sha256": _source_code_sha256(),
        "training_dataset_snapshot": {
            "path": str(dataset_file),
            "size_bytes": len(dataset_snapshot),
            "sha256": dataset_snapshot_sha256,
        },
        "development_summary": development_summary,
        "validity_scope": {
            "risk_calibration_bound": (
                "approximate one-sided percentile cluster-bootstrap bound for the ratio "
                "of expected false accepted runs to expected accepted runs; whole canonical "
                "complexes are resampled; Bonferroni nominal tail allocation over fixed candidates; "
                "not a finite-sample guarantee; assumes independent representative complexes, "
                "a fixed within-complex protocol and nondegenerate bootstrap evidence"
                if cluster_run_risk else
                "finite-sample Clopper-Pearson/binomial bounds at canonical complex "
                "fingerprint level, simultaneous over the predeclared threshold grid via "
                "Bonferroni, under i.i.d. complexes and a fixed within-complex protocol"
            ),
            "locked_test_bound": (
                "separate approximate cluster-bootstrap audit of a frozen threshold; "
                "a fresh confirmation requires outcomes not previously used for model, "
                "threshold or methodology selection; historical reanalysis is exploratory"
                if cluster_run_risk else
                "a separate finite-sample binomial audit for one threshold frozen before "
                "the locked-test outcomes were opened"
            ),
            "venn_abers": (
                "multiprobability calibration under exchangeability; multiple correlated "
                "runs per complex make row-level Venn validity empirical rather than exact; "
                "the authorization risk gate does not rely on treating [p0,p1] as a CI"
            ),
            "distribution_shift": "no guarantee under distribution shift",
            "integrity": (
                "the SHA-256 manifest detects accidental changes but is unsigned and does "
                "not authenticate the operator or prevent a party from forging a new bundle"
            ),
        },
    }
    report = {
        **training_context,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "protocol": (
            "four_way_fingerprint_split_ivap_cluster_bootstrap_run_risk_locked_test"
            if cluster_run_risk else "four_way_fingerprint_split_ivap_ltt_binomial_iid_locked_test"
        ),
        "grouping_unit": "complex_fingerprint_sha256",
        "decision_capable": bool(decision_capable),
        "development_only": development_only,
        "calibration_evidence_passed": calibration_evidence_passed,
        "authorization": artifact["authorization"],
        "split": split_export,
        "development_summary": development_summary,
        "policy": policy,
        "locked_test_policy_audit": test_audit,
        "runtime": runtime,
        "dataset_provenance": provenance,
        "validity_scope": artifact["validity_scope"],
    }

    # Diagnostics are separate from authorization and never select a threshold
    # or a replacement model from test outcomes. Existing bundle names stay fixed.
    emit_progress(progress_callback, 0.92, "Calculando diagnóstico probabilístico e comparação de descritores")
    diagnostic_options = DiagnosticOptions(**config.get("diagnostics", {}))
    diagnostics = saved_test_diagnostics(test_frame, options=diagnostic_options)
    diagnostics["descriptor_comparison"] = descriptor_comparison(
        df, split, feature_columns, config, used_probability_indices, test_frame,
        options=diagnostic_options,
    )
    diagnostics["operational_cost_authorization_applied"] = operational_cost_assessment(
        test_frame, test_accepted if decision_capable else np.zeros(len(test_frame), dtype=bool),
    )
    diagnostics["operational_cost_authorization_applied"]["scope"] = (
        "paired_rows_authorization_simulation_for_future_unseen_complexes"
    )
    report["diagnostics"] = diagnostics

    emit_progress(progress_callback, 0.95, "Salvando modelo, previsões e manifesto de integridade")
    probability_frame.to_csv(paths["probability_calibration"], index=False)
    risk_frame.to_csv(paths["risk_calibration"], index=False)
    test_frame.to_csv(paths["locked_test"], index=False)
    pd.DataFrame(candidate_assessments).to_csv(paths["policy_candidates"], index=False)
    write_json(paths["config"], config)
    write_json(paths["artifact"], artifact)
    write_json(paths["report"], report)
    manifest_inputs = [
        paths["artifact"],
        paths["report"],
        paths["probability_calibration"],
        paths["risk_calibration"],
        paths["locked_test"],
        paths["policy_candidates"],
        paths["config"],
        dataset_file,
    ]
    write_json(
        paths["manifest"],
        build_manifest(
            manifest_inputs,
            snapshot_bytes={dataset_file: dataset_snapshot},
        ),
    )

    emit_progress(progress_callback, 1.0, "Treinamento e calibração concluídos; artefatos salvos")
    return {
        **{key: str(path) for key, path in paths.items()},
        "decision_capable": bool(decision_capable),
    }
