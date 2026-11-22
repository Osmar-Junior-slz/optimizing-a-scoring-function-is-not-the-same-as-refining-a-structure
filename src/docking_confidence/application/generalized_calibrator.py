"""Generalized Venn–Abers (ICML 2025, Algorithm 2), squared-loss docking margin.

The margin preserves the original scientific criterion but is a continuous
regression outcome. Its calibrated mean and prediction set are not probabilities.
"""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

from docking_confidence.application.progress import emit_progress
from docking_confidence.artifact.store import write_json
from docking_confidence.audit.manifest import build_manifest, read_bytes_once, read_verified_bundle, sha256_bytes
from docking_confidence.calibration.generalized_venn_abers import GeneralizedVennAbersCalibrator
from docking_confidence.io.config import config_sha256, load_config, validate_config
from docking_confidence.io.quality_context import validate_paired_quality_context, validate_quality_metric
from docking_confidence.io.dataset import INFERENCE_BASE_COLUMNS, load_paired_dataset, load_reduced_inference_dataset
from docking_confidence.features.calibrator_descriptors import (
    FEATURE_COLUMNS, SOURCE_CONTEXT_COLUMNS, model_features, transform_metadata,
)
from docking_confidence.modeling.group_validation import group_cross_validation
from docking_confidence.modeling import hybrid


LEGACY_SCHEMA = "generalized_venn_abers_preservation_margin_v1"
PREVIOUS_SCHEMA = "generalized_venn_abers_preservation_margin_v2"
SINGLE_SCHEMA = "generalized_venn_abers_preservation_margin_v3"
FIXED_TREE_SCHEMA = "generalized_venn_abers_preservation_margin_v4"
SCHEMA = "generalized_venn_abers_preservation_margin_v5"
SUPPORTED_SCHEMAS = (SCHEMA, FIXED_TREE_SCHEMA, SINGLE_SCHEMA, PREVIOUS_SCHEMA, LEGACY_SCHEMA)
REFERENCE = "https://proceedings.mlr.press/v267/van-der-laan25a.html"
LEGACY_OUTPUT_NAMES = (
    "artifact.json", "resolved_config.json", "validation_report.json",
    "calibration_examples.csv", "training_assignments.csv", "audit_manifest.json",
)
OUTPUT_NAMES = (*LEGACY_OUTPUT_NAMES, "training_cross_validation_predictions.csv")
COMPONENT_SCORE_COLUMNS = {"ridge": "base_ridge_score", "extra_trees": "base_extra_trees_score"}


def _base_backend(schema):
    """Keep V4 replay readable; newly created artifacts always use the hybrid."""
    if schema == FIXED_TREE_SCHEMA:
        from docking_confidence.modeling import extra_trees
        return extra_trees
    return hybrid


def preservation_margin(reduced, full, rule):
    """Nonnegative iff REDUCED meets both parts of the original preservation rule."""
    r, f = np.asarray(reduced, dtype=float), np.asarray(full, dtype=float)
    if r.shape != f.shape or not np.isfinite(r).all() or not np.isfinite(f).all():
        raise ValueError("DockQ values must be finite and aligned")
    if ((r < 0) | (r > 1) | (f < 0) | (f > 1)).any():
        raise ValueError("DockQ values must be in [0, 1]")
    return np.minimum(r - rule["minimum_reduced_dockq"], rule["maximum_dockq_loss"] - (f - r))


def margin_support(rule):
    a, b = rule["minimum_reduced_dockq"], rule["maximum_dockq_loss"]
    return (min(-a, b - 1.0), min(1.0 - a, b + 1.0))


def proper_calibration_partition(groups, *, calibration_fraction=0.3, random_seed=2026):
    if isinstance(calibration_fraction, bool) or not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("São necessários complexos distintos para treino e calibração.")
    shuffled = np.random.default_rng(random_seed).permutation(unique)
    count = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * calibration_fraction))))
    cal = np.isin(groups, shuffled[:count])
    return np.flatnonzero(~cal), np.flatnonzero(cal)


def create_calibrator(dataset_path, config, out_dir, *, calibration_fraction=0.3, random_seed=2026,
                      expected_dataset_sha256=None, progress_callback=None):
    emit_progress(progress_callback, 0.0, "Validando exemplos de qualidade estrutural")
    config = validate_config(config)
    source, out = Path(dataset_path).resolve(), Path(out_dir).resolve()
    snapshot = read_bytes_once(source)
    digest = sha256_bytes(snapshot)
    if expected_dataset_sha256 is not None and digest != expected_dataset_sha256:
        raise ValueError("O dataset mudou depois da preparação.")
    if any((out / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Já existe um calibrador nesta pasta. Escolha uma nova pasta.")
    context_columns = list(config["features"]["columns"])
    if context_columns != list(SOURCE_CONTEXT_COLUMNS):
        raise ValueError("O contexto deve corresponder à coleta REDUCED original.")
    required_base = [column for column in context_columns if column in FEATURE_COLUMNS]
    frame = load_paired_dataset(BytesIO(snapshot), required_base, config["contact_graph"],
        docking_task=config.get("docking_task"), feature_schema=config["features"]["schema_version"],
        context_feature_columns=context_columns)
    if "config_sha256" in frame and not frame.config_sha256.eq(config_sha256(config)).all():
        raise ValueError("A configuração não corresponde ao dataset fornecido.")
    validate_paired_quality_context(frame, config.get("quality_metric"))
    if config.get("quality_metric") is not None:
        from docking_confidence.application.standard_dataset import verify_standard_quality_records
        verify_standard_quality_records(frame, source.parent, config["quality_metric"])
    rule = config["preservation"]
    outcomes = preservation_margin(frame.dockq_reduced, frame.dockq_full, rule)
    support = margin_support(rule)
    proper, calibration = proper_calibration_partition(frame.complex_fingerprint_sha256,
        calibration_fraction=calibration_fraction, random_seed=random_seed)
    X, model_columns = model_features(frame)
    groups = frame.complex_fingerprint_sha256.to_numpy(str)
    emit_progress(progress_callback, 0.20, "Validando o híbrido em complexos separados")
    oof, development_validation = group_cross_validation(X[proper], outcomes[proper], groups[proper])
    emit_progress(progress_callback, 0.50, "Ajustando parte linear e árvores com pesos fixos")
    model = hybrid.HybridBaseModel().fit(X[proper], outcomes[proper], groups[proper])
    # Calibrate the frozen JSON predictor that will also answer future queries.
    base_payload = model.export(model_columns)
    components = hybrid.component_scores_from_export(base_payload, X[calibration])
    scores = hybrid.predict_from_export(base_payload, X[calibration])
    emit_progress(progress_callback, 0.65, "Criando Generalized Venn–Abers de 2025")
    calibrator = GeneralizedVennAbersCalibrator(outcome_bounds=support).fit(scores, outcomes[calibration])
    keys = ["complex_id", "run_id", "algorithm", "seed", "complex_fingerprint_sha256"]
    assignments = frame[keys].copy()
    assignments["partition"] = "proper_training"
    assignments.loc[calibration, "partition"] = "calibration"
    examples = frame.iloc[calibration][keys + model_columns].copy()
    for component, column in COMPONENT_SCORE_COLUMNS.items():
        examples[column] = components[component]
    examples["outcome"] = outcomes[calibration]
    examples["base_raw_score"] = scores
    target = {
        "name": "preservation_margin", "kind": "continuous_regression",
        "formula": "min(DockQ_reduced - minimum_reduced_dockq, maximum_dockq_loss - (DockQ_full - DockQ_reduced))",
        "definition": rule, "outcome_bounds": list(support), "unit": "DockQ margin",
        "interpretation": "nonnegative iff the existing docking-preservation criterion is met",
        "scope": "docking-specific target chosen by the researcher; not prescribed by the 2025 article",
    }
    settings = {
        "method": "generalized_venn_abers", "publication_year": 2025, "loss": "squared_error",
        "calibration_fraction": float(calibration_fraction), "random_seed": random_seed,
        "model": {"method": hybrid.METHOD, "parameters": base_payload["parameters"]},
        "features": config["features"], "feature_transform": transform_metadata(),
        "contact_graph": config["contact_graph"],
        "docking_task": config.get("docking_task"), "target": target,
        "source_config_sha256": config_sha256(config),
        "point_rule": "calibration_only_isotonic; auxiliary point without the set's marginal guarantee",
    }
    partition = {role: {"row_count": len(indices),
        "complex_count": int(frame.iloc[indices].complex_fingerprint_sha256.nunique()),
        "groups": sorted(frame.iloc[indices].complex_fingerprint_sha256.unique().tolist())}
        for role, indices in [("proper_training", proper), ("calibration", calibration)]}
    if "execution_contract" in config:
        settings["execution_contract"] = dict(config["execution_contract"])
    if "quality_metric" in config:
        settings["quality_metric"] = validate_quality_metric(config["quality_metric"])
    artifact = {
        "schema_version": SCHEMA, "artifact_type": "generalized_quality_calibrator",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "reference": REFERENCE,
        "feature_columns": model_columns, "source_feature_columns": model_columns,
        "base_model": base_payload, "calibration": calibrator.export(),
        "development_validation": development_validation,
        "settings": settings, "target": target, "partition": partition,
        "training_dataset_snapshot": {"path": str(source), "sha256": digest, "size_bytes": len(snapshot)},
        "output": {"set_range": ["prediction_lower", "prediction_upper"], "point": "prediction",
                   "base_component_margins": list(COMPONENT_SCORE_COLUMNS.values()),
                   "meaning": "calibrated quality-preservation margin, not probability or coverage interval"},
    }
    calibrated_points = calibrator.predict_point(scores)
    point_range = [float(calibrated_points.min()), float(calibrated_points.max())]
    negative_groups = frame.iloc[calibration].loc[outcomes[calibration] < 0, "complex_fingerprint_sha256"].nunique()
    calibration_diagnostics = {
        "row_count": len(calibration), "complex_count": len(partition["calibration"]["groups"]),
        "negative_row_count": int((outcomes[calibration] < 0).sum()), "negative_complex_count": int(negative_groups),
        "point_range": point_range, "point_can_be_negative": point_range[0] < 0,
        "all_point_predictions_positive": point_range[0] > 0,
        "interpretation": "descriptive fitting diagnostics, not an individual-failure detection guarantee",
    }
    warnings = []
    if point_range[0] >= 0:
        warnings.append("point_function_cannot_output_negative_margin")
    validation_skill = development_validation.get("skill_against_fold_training_mean")
    if development_validation.get("available") and validation_skill is not None and validation_skill <= 0:
        warnings.append("proper_training_group_validation_did_not_improve_over_training_mean")
    report = {
        "schema_version": SCHEMA, "status": "calibrator_created", "reference": REFERENCE,
        "publication_year": 2025, "calibrator_count": 1, "base_model_component_count": 2,
        "partition": partition, "target": target,
        "calibration_preserved_count": int((outcomes[calibration] >= 0).sum()),
        "calibration_not_preserved_count": int((outcomes[calibration] < 0).sum()),
        "calibration_diagnostics": calibration_diagnostics,
        "development_validation": development_validation,
        "warnings": warnings,
        "evaluation": "not_performed; calibration examples are not an independent test",
        "assumptions": [
            "Generalized Venn-Abers validity concerns a prediction set containing a marginally loss-calibrated oracle prediction under exchangeability.",
            "The set range is not a confidence or individual-outcome coverage interval; no 90% guarantee is asserted.",
            "Standalone isotonic point predictions do not inherit the full set's finite-sample guarantee.",
            "Molecular group splitting prevents exact-complex leakage but does not establish absence of homology or independence of repeated algorithm runs.",
        ],
    }
    emit_progress(progress_callback, 0.85, "Salvando o novo calibrador de qualidade")
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in [("artifact.json", artifact), ("resolved_config.json", settings), ("validation_report.json", report)]:
        write_json(out / name, payload)
    examples.to_csv(out / "calibration_examples.csv", index=False, float_format="%.17g")
    assignments.to_csv(out / "training_assignments.csv", index=False)
    cv_rows = frame.iloc[proper][keys].copy()
    cv_rows["observed_margin"] = outcomes[proper]
    cv_rows["out_of_group_prediction"] = oof if oof is not None else np.nan
    cv_rows.to_csv(out / "training_cross_validation_predictions.csv", index=False, float_format="%.17g")
    write_json(out / "audit_manifest.json", build_manifest(
        [out / name for name in OUTPUT_NAMES if name != "audit_manifest.json"] + [source], snapshot_bytes={source: snapshot}))
    result = load_calibrator(out / "artifact.json")
    emit_progress(progress_callback, 1.0, "Generalized Venn–Abers criado")
    return result


def _verify_validation(validation, groups):
    expected = set(groups)
    if (validation.get("calibration_or_test_used") is not False or
        validation.get("hyperparameter_search") is not False or
        validation.get("feature_selection") is not False or
        validation.get("feature_count") != len(FEATURE_COLUMNS) or
        validation.get("independent_complex_count") != len(expected)):
        raise ValueError("A validação deve usar os 15 descritores somente no treino próprio.")
    if bool(validation.get("available")) != (len(expected) >= 2):
        raise ValueError("Disponibilidade inválida da validação por complexo.")
    visited = []
    for fold in validation.get("folds", []):
        train, val = set(fold["training_groups"]), set(fold["validation_groups"])
        if not train or not val or train & val or train | val != expected:
            raise ValueError("Vazamento de complexos na validação interna.")
        visited.extend(val)
    if validation.get("available") and sorted(visited) != sorted(expected):
        raise ValueError("A validação não cobre cada complexo uma única vez.")
    if not validation.get("available") and visited:
        raise ValueError("Não pode haver etapas sem complexos suficientes.")


def load_calibrator(artifact_path):
    path = Path(artifact_path).resolve()
    bundle = read_verified_bundle(path.parent / "audit_manifest.json")
    a = json.loads(bundle.bytes_for(path))
    schema = a.get("schema_version")
    if schema in (LEGACY_SCHEMA, PREVIOUS_SCHEMA, SINGLE_SCHEMA):
        from docking_confidence.application import generalized_legacy
        reader = (generalized_legacy.load_single_descriptor_calibrator if schema == SINGLE_SCHEMA
                  else generalized_legacy.load_calibrator)
        return reader(path)
    if schema not in (SCHEMA, FIXED_TREE_SCHEMA) or a.get("artifact_type") != "generalized_quality_calibrator":
        raise ValueError("Este arquivo não é o calibrador Generalized de 15 descritores.")
    payloads = {name: bundle.manifest_bytes if name == "audit_manifest.json" else bundle.bytes_for(path.parent / name)
                for name in OUTPUT_NAMES}
    settings, features = a["settings"], a["feature_columns"]
    if (a.get("reference") != REFERENCE or settings.get("publication_year") != 2025 or
        settings.get("method") != "generalized_venn_abers" or settings.get("loss") != "squared_error"):
        raise ValueError("Método ou referência incompatível com Generalized Venn–Abers de 2025.")
    if settings != json.loads(payloads["resolved_config.json"]) or a["target"] != settings["target"]:
        raise ValueError("A configuração salva diverge do calibrador.")
    if "quality_metric" in settings:
        validate_quality_metric(settings["quality_metric"])
    if (features != list(FEATURE_COLUMNS) or a["source_feature_columns"] != features or
        settings["features"]["columns"] != list(SOURCE_CONTEXT_COLUMNS) or
        settings["feature_transform"] != transform_metadata() or
        "descriptor_selection" in a or "candidate_feature_columns" in a):
        raise ValueError("O calibrador deve usar exatamente os 15 descritores fixos.")
    target = a["target"]
    if (target["name"] != "preservation_margin" or target["kind"] != "continuous_regression" or
        target["outcome_bounds"] != list(margin_support(target["definition"]))):
        raise ValueError("Alvo incompatível com o calibrador de qualidade.")
    roles = a["partition"]
    if (set(roles) != {"proper_training", "calibration"} or
        any(not role["groups"] for role in roles.values()) or
        set(roles["proper_training"]["groups"]) & set(roles["calibration"]["groups"])):
        raise ValueError("As partições de treino e calibração precisam ser disjuntas.")
    examples = pd.read_csv(BytesIO(payloads["calibration_examples.csv"]), float_precision="round_trip")
    restored = GeneralizedVennAbersCalibrator(outcome_bounds=target["outcome_bounds"]).fit(
        examples.base_raw_score.to_numpy(), examples.outcome.to_numpy())
    if restored.export() != a["calibration"]:
        raise ValueError("Os exemplos salvos divergem da calibração generalizada.")
    backend = _base_backend(schema)
    base = backend.validate_export(a["base_model"])
    if (base["feature_columns"] != features or settings["model"]["method"] != base["method"] or
        settings["model"]["parameters"] != base["parameters"]):
        raise ValueError("O modelo-base diverge dos 15 descritores.")
    X, _ = model_features(examples)
    if not np.allclose(backend.predict_from_export(base, X), examples.base_raw_score.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("Os escores divergem do modelo-base salvo.")
    if schema == SCHEMA:
        scores = hybrid.component_scores_from_export(base, X)
        for component, column in COMPONENT_SCORE_COLUMNS.items():
            if column not in examples or not np.allclose(scores[component], examples[column], rtol=1e-12, atol=1e-12):
                raise ValueError("Os componentes divergem do híbrido salvo.")
    assignments = pd.read_csv(BytesIO(payloads["training_assignments.csv"]), dtype=str)
    if not assignments.partition.isin(roles).all():
        raise ValueError("Partição desconhecida nos exemplos de ajuste.")
    for role, values in roles.items():
        assigned = assignments.loc[assignments.partition == role]
        if (len(assigned) != values["row_count"] or
            sorted(assigned.complex_fingerprint_sha256.unique()) != values["groups"] or
            len(values["groups"]) != values["complex_count"]):
            raise ValueError("As partições salvas divergem do calibrador.")
    if (len(examples) != roles["calibration"]["row_count"] or
        sorted(examples.complex_fingerprint_sha256.unique()) != roles["calibration"]["groups"]):
        raise ValueError("Os exemplos não correspondem à calibração.")
    report = json.loads(payloads["validation_report.json"])
    validation = a["development_validation"]
    if (report.get("development_validation") != validation or "descriptor_selection" in report or
        validation.get("parameters") != base["parameters"]):
        raise ValueError("O relatório diverge da validação dos descritores fixos.")
    if schema == SCHEMA and (validation.get("base_model_method") != hybrid.METHOD or
        report.get("calibrator_count") != 1 or report.get("base_model_component_count") != 2):
        raise ValueError("O híbrido requer dois componentes e uma única calibração final.")
    _verify_validation(validation, roles["proper_training"]["groups"])
    cv_rows = pd.read_csv(BytesIO(payloads["training_cross_validation_predictions.csv"]),
                          dtype={"complex_fingerprint_sha256": str}, float_precision="round_trip")
    if (len(cv_rows) != roles["proper_training"]["row_count"] or
        sorted(cv_rows.complex_fingerprint_sha256.unique()) != roles["proper_training"]["groups"]):
        raise ValueError("As previsões de validação interna divergem do treino próprio.")
    actual_range = [float(restored.predict_point(examples.base_raw_score).min()),
                    float(restored.predict_point(examples.base_raw_score).max())]
    if report.get("calibration_diagnostics", {}).get("point_range") != actual_range:
        raise ValueError("O diagnóstico diverge da função pontual salva.")
    return {"artifact": a, "report": report,
            "paths": {name: str(path.parent / name) for name in OUTPUT_NAMES},
            "downloads": payloads, "verified": True}


def predict_calibrator(artifact_path, input_path, output_path):
    result = load_calibrator(artifact_path)
    a = result["artifact"]
    output = Path(output_path).resolve()
    if output in {Path(p).resolve() for p in result["paths"].values()} | {
        Path(input_path).resolve(), Path(a["training_dataset_snapshot"]["path"]).resolve()}:
        raise ValueError("A saída não pode sobrescrever as entradas ou o calibrador.")
    s, features = a["settings"], a["feature_columns"]
    source_features = a.get("source_feature_columns", features)
    fixed = a["schema_version"] in (SCHEMA, FIXED_TREE_SCHEMA)
    requested = ([column for column in source_features if column in SOURCE_CONTEXT_COLUMNS]
                 if fixed else source_features)
    frame = load_reduced_inference_dataset(input_path, requested, s["contact_graph"],
        docking_task=s["docking_task"], feature_schema=s["features"]["schema_version"],
        context_feature_columns=s["features"]["columns"] if fixed or a["schema_version"] == SINGLE_SCHEMA else None)
    if "config_sha256" in frame and not frame.config_sha256.eq(s["source_config_sha256"]).all():
        raise ValueError("A configuração da entrada diverge dos exemplos de treinamento.")
    component_scores = {}
    if fixed:
        X, _ = model_features(frame)
        frame[features] = X
        raw = _base_backend(a["schema_version"]).predict_from_export(a["base_model"], X)
        if a["schema_version"] == SCHEMA:
            component_scores = hybrid.component_scores_from_export(a["base_model"], X)
    else:
        from docking_confidence.application.generalized_legacy import raw_scores
        raw = raw_scores(a, frame)
    calibrator = GeneralizedVennAbersCalibrator.from_export(a["calibration"])
    lower, upper = calibrator.predict_bounds(raw)
    point = calibrator.predict_point(raw)
    identity = [*INFERENCE_BASE_COLUMNS, "complex_fingerprint_schema", "complex_fingerprint_sha256",
                "docking_task", "feature_schema", "config", "config_sha256"]
    frame = frame[list(dict.fromkeys(c for c in [*identity, *features] if c in frame))].copy()
    for component, values in component_scores.items():
        frame[COMPONENT_SCORE_COLUMNS[component]] = values
    frame["base_raw_score"] = raw
    frame["prediction_lower"], frame["prediction_upper"], frame["prediction"] = lower, upper, point
    frame["prediction_set_width"] = upper - lower
    frame.to_csv(output, index=False)
    return frame


def create_from_source(source, out_dir, *, config_path=None, calibration_fraction=0.3, random_seed=2026, progress_callback=None):
    source = Path(source).resolve(strict=True)
    out = Path(out_dir).resolve()
    if source == out or (source.is_dir() and (source in out.parents or out in source.parents)):
        raise ValueError("Escolha uma pasta de saída separada dos resultados de entrada.")
    if any((out / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Já existe um calibrador nesta pasta. Escolha uma nova pasta.")
    expected = None
    if source.is_dir():
        from docking_confidence.application.training_dataset import build_training_dataset
        root = source / "avaliacao_calibrador" if (source / "avaliacao_calibrador/batch_manifest.json").is_file() else source
        manifest = json.loads((root / "batch_manifest.json").read_bytes())
        profile = "operational" if manifest["plan"].get("execution_strategy") == "operational_cascade_v1" else "default"
        prepared = build_training_dataset(root, out / "inputs", profile=profile, config_path=config_path,
            fingerprint_alias_policy="first_complex_id", progress_callback=progress_callback)
        if not prepared["valid"]:
            raise ValueError("Dataset inválido: " + json.dumps(prepared["issues"], ensure_ascii=False))
        dataset, config_path = Path(prepared["dataset_path"]), prepared["config_path"]
        expected = prepared["dataset_sha256"]
    elif source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            names = [n for n in archive.namelist() if n.endswith("/analise_artigo/calibrador_previsoes_e_qualidade.csv")
                     or n == "analise_artigo/calibrador_previsoes_e_qualidade.csv"]
            if len(names) != 1:
                raise ValueError("O ZIP deve conter uma única tabela calibrador_previsoes_e_qualidade.csv.")
            payload = archive.read(names[0])
        expected = sha256_bytes(payload)
        dataset = out / "inputs/dataset.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        if dataset.exists():
            if read_bytes_once(dataset) != payload:
                raise FileExistsError("A pasta contém dados de outra origem. Escolha uma nova pasta de saída.")
        else:
            with dataset.open("xb") as handle:
                handle.write(payload)
    else:
        dataset = source
    if config_path is None:
        header = pd.read_csv(dataset, nrows=1)
        if "config" not in header or header.empty:
            raise ValueError("Informe a configuração que foi usada para preparar o CSV.")
        config_path = str(header.iloc[0]["config"])
    return create_calibrator(dataset, load_config(config_path), out, calibration_fraction=calibration_fraction,
        random_seed=random_seed, expected_dataset_sha256=expected, progress_callback=progress_callback)
