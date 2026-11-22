"""Read-only replay of historical GVA V1, V2 and V3 artifacts.

No training or model selection is exposed by this module.
"""
from io import BytesIO
import json
from pathlib import Path
import numpy as np
import pandas as pd
from docking_confidence.audit.manifest import read_verified_bundle
from docking_confidence.calibration.generalized_venn_abers import GeneralizedVennAbersCalibrator
from docking_confidence.features.legacy_calibrator_context import model_features, transform_metadata
from docking_confidence.modeling.extra_trees import predict_from_export, validate_export
from docking_confidence.application.generalized_calibrator import margin_support
from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS

LEGACY_SCHEMA = "generalized_venn_abers_preservation_margin_v1"
SCHEMA = "generalized_venn_abers_preservation_margin_v2"
SUPPORTED_SCHEMAS = (SCHEMA, LEGACY_SCHEMA)
PREVIOUS_SCHEMA = SCHEMA
SINGLE_SCHEMA = "generalized_venn_abers_preservation_margin_v3"
REFERENCE = "https://proceedings.mlr.press/v267/van-der-laan25a.html"
LEGACY_OUTPUT_NAMES = (
    "artifact.json", "resolved_config.json", "validation_report.json",
    "calibration_examples.csv", "training_assignments.csv", "audit_manifest.json",
)
OUTPUT_NAMES = (*LEGACY_OUTPUT_NAMES, "training_cross_validation_predictions.csv")


def load_calibrator(artifact_path):
    path = Path(artifact_path).resolve()
    bundle = read_verified_bundle(path.parent / "audit_manifest.json")
    a = json.loads(bundle.bytes_for(path))
    schema = a.get("schema_version")
    names = OUTPUT_NAMES if schema == SCHEMA else LEGACY_OUTPUT_NAMES
    payloads = {name: bundle.manifest_bytes if name == "audit_manifest.json" else bundle.bytes_for(path.parent / name)
                for name in names}
    if schema not in SUPPORTED_SCHEMAS or a.get("artifact_type") != "generalized_quality_calibrator":
        raise ValueError("Este arquivo não é o Generalized Venn–Abers de qualidade.")
    s = a["settings"]
    if (a.get("reference") != REFERENCE or s.get("publication_year") != 2025 or
        s.get("method") != "generalized_venn_abers" or s.get("loss") != "squared_error"):
        raise ValueError("Método ou referência incompatível com Generalized Venn–Abers de 2025.")
    if s != json.loads(payloads["resolved_config.json"]) or a["target"] != s["target"]:
        raise ValueError("A configuração salva diverge do calibrador.")
    target = a["target"]
    if (target["name"] != "preservation_margin" or target["kind"] != "continuous_regression" or
        target["outcome_bounds"] != list(margin_support(target["definition"])) or s["loss"] != "squared_error"):
        raise ValueError("Alvo ou perda incompatível com o calibrador de qualidade.")
    roles = a["partition"]
    if set(roles) != {"proper_training", "calibration"} or any(not r["groups"] for r in roles.values()) or (
        set(roles["proper_training"]["groups"]) & set(roles["calibration"]["groups"])):
        raise ValueError("As partições de treino e calibração precisam ser disjuntas.")
    examples = pd.read_csv(BytesIO(payloads["calibration_examples.csv"]), float_precision="round_trip")
    replay = GeneralizedVennAbersCalibrator(outcome_bounds=target["outcome_bounds"]).fit(
        examples.base_raw_score.to_numpy(), examples.outcome.to_numpy())
    if replay.export() != a["calibration"]:
        raise ValueError("Os exemplos salvos divergem da calibração generalizada.")
    restored = GeneralizedVennAbersCalibrator.from_export(a["calibration"])
    if list(restored.outcome_bounds) != target["outcome_bounds"]:
        raise ValueError("O suporte da calibração diverge do alvo de qualidade.")
    base, features = a["base_model"], a["feature_columns"]
    if schema == SCHEMA:
        source_features = a["source_feature_columns"]
        transform = transform_metadata(source_features)
        if (s["features"]["columns"] != source_features or s["feature_transform"] != transform or
            features != transform["model_feature_columns"] or base["feature_columns"] != features or
            s["model"]["parameters"] != base["parameters"]):
            raise ValueError("As características ou parâmetros divergem do preditor não linear.")
        validate_export(base)
        example_X, _ = model_features(examples, source_features)
        raw = predict_from_export(base, example_X)
    else:
        if (base["method"] != "standardized_ridge_regression_v1" or base["feature_columns"] != features or
            s["features"]["columns"] != features):
            raise ValueError("O modelo-base diverge do calibrador generalizado.")
        for key in ["coef", "scaler_mean", "scaler_scale"]:
            v = np.asarray(base[key], dtype=float)
            if v.shape != (len(features),) or not np.isfinite(v).all():
                raise ValueError("Parâmetros inválidos no modelo-base.")
        if np.any(np.asarray(base["scaler_scale"]) <= 0) or not np.isfinite(base["intercept"]):
            raise ValueError("Parâmetros inválidos no modelo-base.")
        raw = ((examples[features].to_numpy(float) - base["scaler_mean"]) / base["scaler_scale"]) @ np.asarray(base["coef"]) + base["intercept"]
    if not np.allclose(raw, examples.base_raw_score.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("Os escores de calibração divergem do modelo-base salvo.")
    assignments = pd.read_csv(BytesIO(payloads["training_assignments.csv"]), dtype=str)
    for role, values in roles.items():
        assigned = assignments.loc[assignments.partition == role]
        if (len(assigned) != values["row_count"] or
            sorted(assigned.complex_fingerprint_sha256.unique().tolist()) != values["groups"] or
            len(values["groups"]) != values["complex_count"]):
            raise ValueError("As partições salvas divergem do calibrador.")
    if (len(examples) != roles["calibration"]["row_count"] or
        sorted(examples.complex_fingerprint_sha256.unique().tolist()) != roles["calibration"]["groups"]):
        raise ValueError("Os exemplos não correspondem à partição de calibração.")
    report = json.loads(payloads["validation_report.json"])
    if schema == SCHEMA:
        validation = a["development_validation"]
        if report.get("development_validation") != validation:
            raise ValueError("O relatório diverge da validação interna por proteína.")
        cv_rows = pd.read_csv(BytesIO(payloads["training_cross_validation_predictions.csv"]), dtype={"complex_fingerprint_sha256": str})
        if (len(cv_rows) != roles["proper_training"]["row_count"] or
            sorted(cv_rows.complex_fingerprint_sha256.unique()) != roles["proper_training"]["groups"]):
            raise ValueError("As previsões de validação interna divergem do treino próprio.")
        visited = []
        for fold in validation.get("folds", []):
            train_groups, val_groups = set(fold["training_groups"]), set(fold["validation_groups"])
            if train_groups & val_groups or train_groups | val_groups != set(roles["proper_training"]["groups"]):
                raise ValueError("Vazamento de proteínas na validação interna.")
            visited.extend(val_groups)
        if validation.get("available") and sorted(visited) != roles["proper_training"]["groups"]:
            raise ValueError("A validação interna não cobre cada proteína uma única vez.")
        actual_range = [float(restored.predict_point(examples.base_raw_score).min()),
                        float(restored.predict_point(examples.base_raw_score).max())]
        if report.get("calibration_diagnostics", {}).get("point_range") != actual_range:
            raise ValueError("O diagnóstico de calibração diverge da função pontual salva.")
    return {"artifact": a, "report": report,
            "paths": {name: str(path.parent / name) for name in names}, "downloads": payloads, "verified": True}


def raw_scores(artifact, frame):
    """Replay an old preprocessor/predictor without retraining or new inputs."""
    base, features = artifact["base_model"], artifact["feature_columns"]
    if artifact["schema_version"] == SINGLE_SCHEMA:
        return predict_from_export(base, frame[features].to_numpy(float))
    if artifact["schema_version"] == SCHEMA:
        X, _ = model_features(frame, artifact["source_feature_columns"])
        for index, name in enumerate(features):
            frame[name] = X[:, index]
        return predict_from_export(base, X)
    return ((frame[features].to_numpy(float) - base["scaler_mean"]) / base["scaler_scale"]) @ np.asarray(base["coef"]) + base["intercept"]


def _verify_group_folds(validation, expected_groups, candidates, *, nested=False):
    expected = set(expected_groups)
    if (validation.get("calibration_or_test_used") is not False or
        validation.get("hyperparameter_search") is not False or
        validation.get("feature_count") != 1 or
        validation.get("independent_complex_count") != len(expected)):
        raise ValueError("A seleção deve usar um descritor somente no treino próprio.")
    if bool(validation.get("available")) != (len(expected) >= 2):
        raise ValueError("Disponibilidade inválida da validação por complexo.")
    visited = []
    for fold in validation.get("folds", []):
        train, val = set(fold["training_groups"]), set(fold["validation_groups"])
        if not train or not val or train & val or train | val != expected:
            raise ValueError("Vazamento de complexos na seleção ou validação interna.")
        visited.extend(val)
        if nested:
            if fold["selected_feature"] not in candidates:
                raise ValueError("Descritor inválido na validação aninhada.")
            _verify_group_folds(fold["selection"], train, candidates)
            if fold["selected_feature"] != fold["selection"]["selected_feature"]:
                raise ValueError("A escolha do descritor diverge do treino da etapa.")
    if validation.get("available") and sorted(visited) != sorted(expected):
        raise ValueError("A validação não cobre cada complexo uma única vez.")
    if not validation.get("available") and visited:
        raise ValueError("Não pode haver etapas sem complexos suficientes.")
    if not nested:
        scores = validation.get("candidate_scores", [])
        if validation.get("available"):
            if [item["feature"] for item in scores] != candidates or any(
                not np.isfinite(item["group_mse"]) or item["group_mse"] < 0 for item in scores):
                raise ValueError("Escores inválidos da seleção do descritor.")
            chosen = min(scores, key=lambda item: item["group_mse"])["feature"]
        else:
            chosen = candidates[0]
        if validation.get("selected_feature") != chosen:
            raise ValueError("O descritor não corresponde à seleção registrada.")

def load_single_descriptor_calibrator(artifact_path):
    path = Path(artifact_path).resolve()
    bundle = read_verified_bundle(path.parent / "audit_manifest.json")
    a = json.loads(bundle.bytes_for(path))
    if a.get("schema_version") in (LEGACY_SCHEMA, PREVIOUS_SCHEMA):
        from docking_confidence.application.generalized_legacy import load_calibrator as load_legacy
        return load_legacy(path)
    if a.get("schema_version") != SINGLE_SCHEMA or a.get("artifact_type") != "generalized_quality_calibrator":
        raise ValueError("Este arquivo não é o Generalized Venn–Abers de qualidade.")
    payloads = {name: bundle.manifest_bytes if name == "audit_manifest.json" else bundle.bytes_for(path.parent / name)
                for name in OUTPUT_NAMES}
    s, features = a["settings"], a["feature_columns"]
    candidates, selection = a["candidate_feature_columns"], a["descriptor_selection"]
    if (a.get("reference") != REFERENCE or s.get("publication_year") != 2025 or
        s.get("method") != "generalized_venn_abers" or s.get("loss") != "squared_error"):
        raise ValueError("Método ou referência incompatível com Generalized Venn–Abers de 2025.")
    if s != json.loads(payloads["resolved_config.json"]) or a["target"] != s["target"]:
        raise ValueError("A configuração salva diverge do calibrador.")
    if (len(candidates) != len(LOCAL_REFINEMENT_FEATURE_COLUMNS) or
        set(candidates) != set(LOCAL_REFINEMENT_FEATURE_COLUMNS) or
        s["features"]["columns"] != candidates or len(features) != 1 or
        features != a["source_feature_columns"] or features != [selection["selected_feature"]] or
        features[0] not in candidates or s["feature_transform"] != {
            "schema_version": "gva_single_descriptor_v1", "model_feature_columns": features,
            "algorithm_used_as_predictor": False}):
        raise ValueError("O calibrador deve usar exatamente um descritor REDUCED.")
    target = a["target"]
    if (target["name"] != "preservation_margin" or target["kind"] != "continuous_regression" or
        target["outcome_bounds"] != list(margin_support(target["definition"]))):
        raise ValueError("Alvo incompatível com o calibrador de qualidade.")
    roles = a["partition"]
    if (set(roles) != {"proper_training", "calibration"} or
        any(not r["groups"] for r in roles.values()) or
        set(roles["proper_training"]["groups"]) & set(roles["calibration"]["groups"])):
        raise ValueError("As partições de treino e calibração precisam ser disjuntas.")
    examples = pd.read_csv(BytesIO(payloads["calibration_examples.csv"]), float_precision="round_trip")
    if set(examples.columns) & (set(candidates) - set(features)):
        raise ValueError("Os exemplos do calibrador contêm descritores excluídos.")
    restored = GeneralizedVennAbersCalibrator(outcome_bounds=target["outcome_bounds"]).fit(
        examples.base_raw_score.to_numpy(), examples.outcome.to_numpy())
    if restored.export() != a["calibration"]:
        raise ValueError("Os exemplos salvos divergem da calibração generalizada.")
    base = validate_export(a["base_model"])
    if (base["feature_columns"] != features or s["model"]["method"] != base["method"] or
        s["model"]["parameters"] != base["parameters"]):
        raise ValueError("O modelo-base diverge do descritor único.")
    raw = predict_from_export(base, examples[features].to_numpy(float))
    if not np.allclose(raw, examples.base_raw_score.to_numpy(), rtol=1e-12, atol=1e-12):
        raise ValueError("Os escores divergem do modelo-base salvo.")
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
    if (report.get("development_validation") != validation or
        report.get("descriptor_selection") != selection or
        validation.get("descriptor_selection_nested") is not True):
        raise ValueError("O relatório diverge da seleção e validação aninhada.")
    _verify_group_folds(selection, roles["proper_training"]["groups"], candidates)
    _verify_group_folds(validation, roles["proper_training"]["groups"], candidates, nested=True)
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
