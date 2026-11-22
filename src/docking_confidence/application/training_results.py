"""Verified training presentation and explicit invocation of the existing backend."""
from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pandas as pd

from docking_confidence.application import train as training_backend
from docking_confidence.application.progress import ProgressCallback, emit_progress
from docking_confidence.application.predict import _validate_verified_bundle_evidence
from docking_confidence.artifact.schema import validate_artifact
from docking_confidence.audit.manifest import read_verified_bundle, sha256_file
from docking_confidence.io.config import config_sha256, load_config
from docking_confidence.io.dataset import load_paired_dataset
from docking_confidence.application.probability_reporting import DiagnosticOptions, saved_test_diagnostics


TRAINING_OUTPUT_NAMES = (
    "artifact.json",
    "validation_report.json",
    "probability_calibration_predictions.csv",
    "risk_calibration_predictions.csv",
    "locked_test_predictions.csv",
    "policy_candidates.csv",
    "resolved_config.json",
    "audit_manifest.json",
)


def existing_training_outputs(out_dir: str | Path) -> list[Path]:
    root = Path(out_dir).resolve()
    return [root / name for name in TRAINING_OUTPUT_NAMES if (root / name).exists()]


def load_training_result(out_dir: str | Path) -> dict:
    """Verify the saved bundle before displaying decisions or offering downloads."""
    root = Path(out_dir).resolve()
    artifact_path = root / "artifact.json"
    bundle = read_verified_bundle(root / "audit_manifest.json")
    artifact = validate_artifact(json.loads(bundle.bytes_for(artifact_path)))
    _validate_verified_bundle_evidence(artifact, artifact_path, bundle)
    payloads = {
        name: (
            bundle.manifest_bytes
            if name == "audit_manifest.json"
            else bundle.bytes_for(root / name)
        )
        for name in TRAINING_OUTPUT_NAMES
    }
    report = json.loads(payloads["validation_report.json"])
    frame = pd.read_csv(BytesIO(payloads["locked_test_predictions.csv"]), float_precision="round_trip")
    persisted = report.get("diagnostics")
    if persisted is not None:
        settings = persisted["probability"]["bootstrap"]
        options = DiagnosticOptions(**{key: settings[key] for key in (
            "bootstrap_repetitions", "bootstrap_seed", "confidence_level")})
        derived = saved_test_diagnostics(frame, options=options)
        if any(persisted.get(key) != value for key, value in derived.items()):
            raise ValueError("saved probability diagnostics disagree with verified test predictions")
        diagnostics = persisted
    else:
        # Old bundles remain immutable and usable; newly derived diagnostics are
        # clearly a presentation result, not an update to their historical gates.
        diagnostics = saved_test_diagnostics(frame)
    return {
        "artifact": artifact,
        "report": report,
        "diagnostics": diagnostics,
        "paths": {name: str(root / name) for name in TRAINING_OUTPUT_NAMES},
        "downloads": payloads,
        "verified": True,
    }


def train_prepared_dataset(
    preparation: dict, out_dir: str | Path, *, progress_callback: ProgressCallback | None = None,
) -> dict:
    """Train only an unchanged, explicitly validated preparation, never on rerun."""
    emit_progress(progress_callback, 0.0, "Conferindo dataset preparado e seus hashes")
    if preparation.get("valid") is not True or not preparation.get("dataset_path"):
        raise ValueError("O dataset ainda não está válido para treinamento.")
    collisions = existing_training_outputs(out_dir)
    if collisions:
        raise FileExistsError("Treinamento já existe nesta pasta.")
    dataset_path = Path(preparation["dataset_path"]).resolve()
    if sha256_file(dataset_path) != preparation.get("dataset_sha256"):
        raise ValueError("O dataset mudou depois da preparação. Prepare-o novamente.")
    config = load_config(preparation["config_path"])
    config.setdefault("docking_task", preparation["docking_task"])
    if config_sha256(config) != preparation.get("config_sha256"):
        raise ValueError("A configuração mudou depois da preparação. Prepare novamente.")
    if (
        config["docking_task"] != preparation["docking_task"]
        or config["features"]["schema_version"] != preparation["feature_schema"]
        or config["features"]["columns"] != preparation["feature_columns"]
    ):
        raise ValueError("O protocolo e o schema divergem do dataset preparado.")
    load_paired_dataset(
        dataset_path,
        config["features"]["columns"],
        contact_graph_cfg=config["contact_graph"],
        docking_task=config["docking_task"],
        feature_schema=config["features"]["schema_version"],
    )
    training_backend.train(
        str(dataset_path), config, str(Path(out_dir).resolve()),
        expected_dataset_sha256=preparation["dataset_sha256"],
        progress_callback=(
            (lambda fraction, message: emit_progress(progress_callback, 0.05 + 0.85 * fraction, message))
            if progress_callback is not None else None
        ),
    )
    emit_progress(progress_callback, 0.95, "Verificando integridade e evidências dos artefatos gerados")
    result = load_training_result(out_dir)
    emit_progress(progress_callback, 1.0, "Treinamento concluído e artefatos verificados")
    return result
