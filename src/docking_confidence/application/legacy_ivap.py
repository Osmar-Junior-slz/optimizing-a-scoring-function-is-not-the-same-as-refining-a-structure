"""Historical IVAP replay; the active creator is generalized_calibrator (2025).

Proper training and calibration are disjoint by molecular fingerprint. Docking
labels and the logistic scoring model are application choices, not IVAP claims.
There is no method comparison, acceptance policy, risk selection or runtime gate.
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
from docking_confidence.audit.manifest import (
    build_manifest, read_bytes_once, read_verified_bundle, sha256_bytes,
)
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.io.config import config_sha256, load_config, validate_config
from docking_confidence.io.dataset import INFERENCE_BASE_COLUMNS, load_paired_dataset, load_reduced_inference_dataset
from docking_confidence.modeling.logistic import LogisticBaseModel


SCHEMA = "inductive_venn_abers_calibrator_v1"
REFERENCE = "https://arxiv.org/abs/1511.00213v2"
OUTPUT_NAMES = (
    "artifact.json", "resolved_config.json", "validation_report.json",
    "calibration_examples.csv", "training_assignments.csv", "audit_manifest.json",
)


def proper_calibration_partition(groups, *, calibration_fraction=0.3, random_seed=2026):
    """Assign groups once, without consulting outcomes or retrying a split."""
    if isinstance(calibration_fraction, bool) or not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError("random_seed must be a non-negative integer")
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    if len(unique) < 2:
        raise ValueError("IVAP precisa de complexos distintos para treino e calibração.")
    shuffled = np.random.default_rng(random_seed).permutation(unique)
    count = min(len(unique) - 1, max(1, int(np.ceil(len(unique) * calibration_fraction))))
    calibration = np.flatnonzero(np.isin(groups, shuffled[:count]))
    proper = np.flatnonzero(~np.isin(groups, shuffled[:count]))
    return proper, calibration


def create_calibrator(
    dataset_path, config, out_dir, *, calibration_fraction=0.3, random_seed=2026,
    expected_dataset_sha256=None, progress_callback=None,
):
    """Fit and export one probability calibrator, independently of policy gates."""
    emit_progress(progress_callback, 0.0, "Validando os exemplos de calibração")
    config = validate_config(config)
    source = Path(dataset_path).resolve()
    snapshot = read_bytes_once(source)
    digest = sha256_bytes(snapshot)
    if expected_dataset_sha256 is not None and digest != expected_dataset_sha256:
        raise ValueError("O dataset mudou depois da preparação.")
    out = Path(out_dir).resolve()
    if any((out / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Já existe um calibrador nesta pasta. Escolha uma nova pasta.")
    features = list(config["features"]["columns"])
    frame = load_paired_dataset(
        BytesIO(snapshot), features, config["contact_graph"],
        docking_task=config.get("docking_task"), feature_schema=config["features"]["schema_version"],
    )
    if "config_sha256" in frame and not frame["config_sha256"].eq(config_sha256(config)).all():
        raise ValueError("A configuração não corresponde ao dataset fornecido.")
    rule = PreservationRule(**config["preservation"])
    labels = np.array([rule.label(r, f) for r, f in zip(frame.dockq_reduced, frame.dockq_full)])
    proper, calibration = proper_calibration_partition(
        frame.complex_fingerprint_sha256, calibration_fraction=calibration_fraction, random_seed=random_seed,
    )
    if len(np.unique(labels[proper])) < 2:
        raise ValueError(
            "O treino do modelo-base precisa de exemplos positivos e negativos. "
            "A partição fixa não contém as duas classes; acrescente exemplos."
        )
    emit_progress(progress_callback, 0.25, "Treinando o modelo-base no conjunto de treino")
    X = frame[features].to_numpy(dtype=float)
    model = LogisticBaseModel(**config["model"]).fit(X[proper], labels[proper])
    emit_progress(progress_callback, 0.65, "Criando um único calibrador Venn–Abers indutivo")
    scores = model.raw_score(X[calibration])
    # IVAP's two hypothetical-label fits also work for a one-class calibration set.
    calibrator = VennAbersCalibrator().fit(scores, labels[calibration], allow_single_class=True)
    assignments = frame[["complex_id", "run_id", "algorithm", "seed", "complex_fingerprint_sha256"]].copy()
    assignments["partition"] = "proper_training"
    assignments.loc[calibration, "partition"] = "calibration"
    examples = frame.iloc[calibration][list(assignments.columns[:-1]) + features].copy()
    examples["label"] = labels[calibration]
    examples["base_raw_score"] = scores
    settings = {
        "method": "inductive_venn_abers", "calibration_fraction": float(calibration_fraction),
        "random_seed": random_seed, "model": config["model"],
        "features": config["features"], "contact_graph": config["contact_graph"],
        "docking_task": config.get("docking_task"), "preservation": config["preservation"],
        "source_config_sha256": config_sha256(config),
    }
    partition = {
        role: {"row_count": len(indices), "complex_count": int(frame.iloc[indices].complex_fingerprint_sha256.nunique()),
               "groups": sorted(frame.iloc[indices].complex_fingerprint_sha256.unique().tolist())}
        for role, indices in [("proper_training", proper), ("calibration", calibration)]
    }
    artifact = {
        "schema_version": SCHEMA, "artifact_type": "probability_calibrator",
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "reference": REFERENCE,
        "feature_columns": features, "base_model": model.export(features),
        "calibration": calibrator.export(), "settings": settings, "partition": partition,
        "target": {"name": "preserved", "definition": config["preservation"],
                   "scope": "application-specific docking label; not defined by the IVAP paper"},
        "training_dataset_snapshot": {"path": str(source), "sha256": digest, "size_bytes": len(snapshot)},
        "output": {"multiprobability": ["p0", "p1"], "point_rule": "p1 / (1 - p0 + p1)"},
    }
    report = {
        "schema_version": SCHEMA, "status": "calibrator_created", "reference": REFERENCE,
        "calibrator_count": 1, "partition": partition,
        "calibration_positive_count": int(labels[calibration].sum()),
        "calibration_negative_count": int(len(calibration) - labels[calibration].sum()),
        "evaluation": "not_performed; calibration examples are not an independent test",
        "assumptions": [
            "IVAP validity assumes independent exchangeable examples from the target population.",
            "Splitting by molecular fingerprint prevents exact-complex leakage, not homology or distribution shift.",
            "Multiple algorithms/seeds for one complex are correlated; group splitting alone does not make rows i.i.d.",
            "The single probability is the log-loss minimax summary; IVAP validity concerns the multiprobability output.",
        ],
    }
    emit_progress(progress_callback, 0.85, "Salvando o calibrador e seus exemplos")
    out.mkdir(parents=True, exist_ok=True)
    for name, payload in [("artifact.json", artifact), ("resolved_config.json", settings), ("validation_report.json", report)]:
        write_json(out / name, payload)
    examples.to_csv(out / "calibration_examples.csv", index=False, float_format="%.17g")
    assignments.to_csv(out / "training_assignments.csv", index=False)
    write_json(out / "audit_manifest.json", build_manifest(
        [out / name for name in OUTPUT_NAMES if name != "audit_manifest.json"] + [source],
        snapshot_bytes={source: snapshot},
    ))
    result = load_calibrator(out / "artifact.json")
    emit_progress(progress_callback, 1.0, "Calibrador Venn–Abers criado")
    return result


def load_calibrator(artifact_path):
    """Read the portable, hash-verified predictor and replay its calibration fit."""
    path = Path(artifact_path).resolve()
    bundle = read_verified_bundle(path.parent / "audit_manifest.json")
    payloads = {name: bundle.manifest_bytes if name == "audit_manifest.json" else bundle.bytes_for(path.parent / name)
                for name in OUTPUT_NAMES}
    artifact = json.loads(payloads["artifact.json"])
    if artifact.get("schema_version") != SCHEMA or artifact.get("artifact_type") != "probability_calibrator":
        raise ValueError("Este arquivo não é um calibrador IVAP.")
    if artifact["settings"] != json.loads(payloads["resolved_config.json"]):
        raise ValueError("A configuração salva diverge do calibrador.")
    roles = artifact["partition"]
    if not roles["proper_training"]["groups"] or not roles["calibration"]["groups"] or (
        set(roles["proper_training"]["groups"]) & set(roles["calibration"]["groups"])
    ):
        raise ValueError("Os conjuntos de treino e calibração precisam ser disjuntos.")
    examples = pd.read_csv(BytesIO(payloads["calibration_examples.csv"]), float_precision="round_trip")
    replay = VennAbersCalibrator().fit(examples.base_raw_score.to_numpy(), examples.label.to_numpy(), allow_single_class=True)
    if replay.export() != artifact["calibration"]:
        raise ValueError("Os exemplos salvos divergem do calibrador Venn–Abers.")
    VennAbersCalibrator.from_export(artifact["calibration"], allow_single_class=True)
    base = artifact["base_model"]
    features = artifact["feature_columns"]
    if base["feature_columns"] != features or artifact["settings"]["features"]["columns"] != features:
        raise ValueError("As características do modelo-base divergem do calibrador.")
    for key in ["coef", "scaler_mean", "scaler_scale"]:
        values = np.asarray(base[key], dtype=float)
        if values.shape != (len(features),) or not np.isfinite(values).all():
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
    return {"artifact": artifact, "report": json.loads(payloads["validation_report.json"]),
            "paths": {name: str(path.parent / name) for name in OUTPUT_NAMES}, "downloads": payloads,
            "verified": True}


def predict_calibrator(artifact_path, input_path, output_path):
    result = load_calibrator(artifact_path)
    artifact = result["artifact"]
    output = Path(output_path).resolve()
    if output in {Path(p).resolve() for p in result["paths"].values()} | {
        Path(input_path).resolve(), Path(artifact["training_dataset_snapshot"]["path"]).resolve(),
    }:
        raise ValueError("A saída não pode sobrescrever as entradas ou o calibrador.")
    settings = artifact["settings"]
    features = artifact["feature_columns"]
    frame = load_reduced_inference_dataset(
        input_path, features, settings["contact_graph"], docking_task=settings["docking_task"],
        feature_schema=settings["features"]["schema_version"],
    )
    if "config_sha256" in frame and not frame.config_sha256.eq(settings["source_config_sha256"]).all():
        raise ValueError("A configuração da entrada diverge dos exemplos de treinamento.")
    base = artifact["base_model"]
    raw = ((frame[features].to_numpy(float) - base["scaler_mean"]) / base["scaler_scale"]) @ np.asarray(base["coef"]) + base["intercept"]
    p0, p1, point = VennAbersCalibrator.from_export(artifact["calibration"], allow_single_class=True).predict_interval(raw)
    # Only REDUCED inputs and probability outputs; no inherited decision columns.
    identity = [*INFERENCE_BASE_COLUMNS, "complex_fingerprint_schema", "complex_fingerprint_sha256",
                "docking_task", "feature_schema", "config", "config_sha256"]
    frame = frame[list(dict.fromkeys(c for c in [*identity, *features] if c in frame))].copy()
    frame["base_raw_score"], frame["p0"], frame["p1"] = raw, p0, p1
    frame["probability_lower"], frame["probability_upper"], frame["probability"] = p0, p1, point
    frame.to_csv(output, index=False)
    return frame


def create_from_source(source, out_dir, *, config_path=None, calibration_fraction=0.3, random_seed=2026, progress_callback=None):
    """Accept an existing paired CSV, result ZIP, or a verified docking batch."""
    source = Path(source).resolve(strict=True)
    out = Path(out_dir).resolve()
    if source == out or (source.is_dir() and (source in out.parents or out in source.parents)):
        raise ValueError("Escolha uma pasta de saída separada dos resultados de entrada.")
    if any((out / name).exists() for name in OUTPUT_NAMES):
        raise FileExistsError("Já existe um calibrador nesta pasta. Escolha uma nova pasta.")
    if source.is_dir():
        from docking_confidence.application.training_dataset import build_training_dataset
        root = source / "avaliacao_calibrador" if (source / "avaliacao_calibrador/batch_manifest.json").is_file() else source
        manifest = json.loads((root / "batch_manifest.json").read_bytes())
        profile = "operational" if manifest["plan"].get("execution_strategy") == "operational_cascade_v1" else "default"
        prepared = build_training_dataset(root, out / "inputs", profile=profile,
            config_path=config_path, fingerprint_alias_policy="first_complex_id", progress_callback=progress_callback)
        if not prepared["valid"]:
            raise ValueError("Dataset inválido: " + json.dumps(prepared["issues"], ensure_ascii=False))
        dataset, config_path = Path(prepared["dataset_path"]), prepared["config_path"]
    elif source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            names = [name for name in archive.namelist() if name.endswith("/analise_artigo/calibrador_previsoes_e_qualidade.csv")
                     or name == "analise_artigo/calibrador_previsoes_e_qualidade.csv"]
            if len(names) != 1:
                raise ValueError("O ZIP deve conter uma única tabela calibrador_previsoes_e_qualidade.csv.")
            payload = archive.read(names[0])
        dataset = out / "inputs/dataset.csv"
        dataset.parent.mkdir(parents=True, exist_ok=True)
        with dataset.open("xb") as handle:
            handle.write(payload)
    else:
        dataset = source
    if config_path is None:
        header = pd.read_csv(dataset, nrows=1)
        if "config" not in header or header.empty:
            raise ValueError("Informe a configuração que foi usada para preparar o CSV.")
        config_path = str(header.iloc[0]["config"])
    return create_calibrator(dataset, load_config(config_path), out,
        calibration_fraction=calibration_fraction, random_seed=random_seed, progress_callback=progress_callback)
