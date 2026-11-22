from copy import deepcopy
import json
from pathlib import Path
import shutil

import pandas as pd
import pytest
import yaml

from docking_confidence.application.demo import make_demo
from docking_confidence.application import training_results
from docking_confidence.audit.manifest import build_manifest, sha256_file
from docking_confidence.io.config import config_sha256, load_config


def make_preparation(tmp_path: Path) -> dict:
    dataset = tmp_path / "demo.csv"
    make_demo(str(dataset), seed=99, n_complexes=25, runs_per_complex=8)
    config = load_config(Path(__file__).parents[1] / "configs/demo.yaml")
    config["docking_task"] = "surface_discovery"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    frame = pd.read_csv(dataset)
    frame["docking_task"] = "surface_discovery"
    frame["feature_schema"] = config["features"]["schema_version"]
    frame["config"] = str(config_path)
    frame["config_sha256"] = config_sha256(config)
    frame.to_csv(dataset, index=False)
    return {
        "valid": True, "dataset_path": str(dataset),
        "dataset_sha256": sha256_file(dataset), "config_path": str(config_path),
        "config_sha256": config_sha256(config), "docking_task": "surface_discovery",
        "feature_schema": config["features"]["schema_version"],
        "feature_columns": config["features"]["columns"],
    }


@pytest.fixture(scope="module")
def verified_training(tmp_path_factory):
    root = tmp_path_factory.mktemp("verified_training_ui")
    prepared = make_preparation(root)
    out = root / "trained"
    result = training_results.train_prepared_dataset(prepared, out)
    return prepared, out, result


def test_prepared_training_calls_existing_backend_and_preserves_outputs(tmp_path, monkeypatch):
    prepared = make_preparation(tmp_path)
    called = []
    backend = training_results.training_backend.train

    def observed(*args, **kwargs):
        called.append(args)
        assert kwargs["expected_dataset_sha256"] == prepared["dataset_sha256"]
        return backend(*args, **kwargs)

    monkeypatch.setattr(training_results.training_backend, "train", observed)
    out = tmp_path / "trained"
    events = []
    def progress(fraction, message):
        events.append((fraction, message))
        if fraction == 1.0:
            assert training_results.load_training_result(out)["verified"] is True
    result = training_results.train_prepared_dataset(prepared, out, progress_callback=progress)
    assert len(called) == 1
    assert result["verified"] is True
    assert len(result["paths"]) == 8
    assert result["artifact"]["decision_capable"] is False
    fractions = [fraction for fraction, _ in events]
    assert fractions == sorted(fractions)
    assert fractions[0] == 0.0 and fractions[-1] == 1.0
    for stage in ("regressão", "domínio", "Venn–Abers", "LTT", "locked test", "integridade"):
        assert any(stage in message for _, message in events)
    initial = (out / "artifact.json").read_bytes()
    with pytest.raises(FileExistsError, match="Treinamento já existe nesta pasta"):
        training_results.train_prepared_dataset(prepared, out)
    assert len(called) == 1
    assert (out / "artifact.json").read_bytes() == initial


@pytest.mark.parametrize("change", ["invalid", "dataset", "config", "protocol", "schema"])
def test_training_rejects_invalid_or_changed_preparation(tmp_path, monkeypatch, change):
    prepared = make_preparation(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Backend must not run for an invalid preparation")
    monkeypatch.setattr(training_results.training_backend, "train", forbidden)
    if change == "invalid":
        prepared["valid"] = False
    elif change == "dataset":
        Path(prepared["dataset_path"]).write_text("changed\n", encoding="utf-8")
    elif change == "config":
        cfg = load_config(prepared["config_path"])
        cfg["model"]["C"] = 2.0
        Path(prepared["config_path"]).write_text(yaml.safe_dump(cfg), encoding="utf-8")
    else:
        prepared["docking_task" if change == "protocol" else "feature_schema"] = "changed"
    with pytest.raises(ValueError):
        training_results.train_prepared_dataset(prepared, tmp_path / "trained")


def test_result_loader_rejects_tampered_evidence_even_with_rehashed_manifest(tmp_path, verified_training):
    _, source, _ = verified_training
    out = tmp_path / "copied"
    shutil.copytree(source, out)
    manifest_path = out / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    evidence_path = out / "locked_test_predictions.csv"
    evidence = pd.read_csv(evidence_path)
    evidence.loc[0, "probability"] = 0.123456
    evidence.to_csv(evidence_path, index=False)
    with pytest.raises(ValueError, match="manifest"):
        training_results.load_training_result(out)
    # Retain source-snapshot identity; a forged replacement manifest must still
    # fail the existing numerical evidence verifier.
    for entry in manifest["bundle_files"]:
        path = out / entry["path"]
        entry["size_bytes"] = path.stat().st_size
        entry["sha256"] = sha256_file(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="disagrees|disagree"):
        training_results.load_training_result(out)


def test_backend_snapshot_rejects_changes_after_preparation_check(tmp_path, monkeypatch):
    prepared = make_preparation(tmp_path)
    original_reader = training_results.training_backend.read_bytes_once

    def changed_snapshot(path):
        return original_reader(path) + b"\n"

    monkeypatch.setattr(training_results.training_backend, "read_bytes_once", changed_snapshot)
    out = tmp_path / "trained"
    with pytest.raises(ValueError, match="snapshot differs"):
        training_results.train_prepared_dataset(prepared, out)
    assert not out.exists()
