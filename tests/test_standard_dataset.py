"""Saved-run conversion must bind the actual pose, protocol and new labels."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from test_training_dataset import make_batch
from docking_confidence.application import standard_dataset as subject
from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.evaluation import official_dockq
from docking_confidence.io.config import config_sha256, load_config


@pytest.fixture
def source(make_batch, monkeypatch):
    batch = make_batch()
    prepared = build_training_dataset(batch, batch.parent / "prepared")
    assert prepared["valid"], prepared["issues"]
    def backend(request, **runtime):
        return {"available": True, "implementation": "DockQ", "implementation_version": "2.1.3",
            "implementation_sha256": "a" * 64, "constants_sha256": "b" * 64,
            "python_executable": "fixture-isolated-python",
            "metrics": {"DockQ": .8, "fnat": .7, "iRMSD": .2, "LRMSD": .3}}
    monkeypatch.setattr(official_dockq, "_invoke_backend", backend)
    return batch, prepared


def _convert(source, output="standard", **kwargs):
    batch, prepared = source
    return subject.build_standard_paired_dataset(prepared["dataset_path"], batch, batch.parent / output,
        source_config_path=prepared["config_path"], **kwargs)


def test_measures_new_quality_preserves_source_features_and_binds_records(source):
    batch, prepared = source
    old_path = Path(prepared["dataset_path"])
    old_bytes = old_path.read_bytes()
    old = pd.read_csv(old_path, float_precision="round_trip")
    result = _convert(source)
    assert result["valid"], result["issues"]
    new = pd.read_csv(result["dataset_path"], float_precision="round_trip")
    assert old_path.read_bytes() == old_bytes
    assert new.dockq_reduced.tolist() == [.8]
    assert new.historical_quality_reduced.tolist() == old.dockq_reduced.tolist()
    features = load_config(prepared["config_path"])["features"]["columns"]
    pd.testing.assert_frame_equal(new[features], old[features])
    assert new.graph_source_file_sha256.tolist() == old.graph_source_file_sha256.tolist()
    assert new.config_sha256.iloc[0] != old.config_sha256.iloc[0]
    verified = subject.verify_standard_quality_records(new, Path(result["dataset_path"]).parent, result["quality_metric"])
    assert verified["records"] == 2
    with pytest.raises(FileExistsError):
        _convert(source)


def test_same_identifiers_with_another_pose_hash_never_match(source):
    _, prepared = source
    path = Path(prepared["dataset_path"])
    frame = pd.read_csv(path)
    frame["graph_source_file_sha256"] = "f" * 64
    frame.to_csv(path, index=False)
    result = _convert(source)
    assert not result["valid"] and result["dataset_path"] is None
    assert result["counts"]["failed_rows"] == 1
    assert "exact source pair" in result["issues"][0]["message"]


def test_reference_content_must_match_original_manifest(source):
    batch, _ = source
    manifest = json.loads((batch / "batch_manifest.json").read_bytes())
    reference = Path(manifest["plan"]["input_root"]) / manifest["plan"]["targets"][0]["reference_path"]
    reference.write_bytes(reference.read_bytes() + b"REMARK changed reference\n")
    result = _convert(source)
    assert not result["valid"] and result["dataset_path"] is None
    assert "No reference with the manifest SHA-256" in result["issues"][0]["message"]


def test_operational_config_cannot_absorb_historical_equal_budget_batch(source):
    batch, prepared = source
    config_path = Path(__file__).resolve().parents[1] / "configs/local_refinement_operational_candidate.yaml"
    config = load_config(config_path)
    path = Path(prepared["dataset_path"])
    frame = pd.read_csv(path)
    frame["config"], frame["config_sha256"] = str(config_path), config_sha256(config)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="execution_contract"):
        subject.build_standard_paired_dataset(path, batch, batch.parent / "standard",
            source_config_path=config_path)


def test_quality_record_cannot_be_swapped_or_have_new_metric_identity(source):
    result = _convert(source)
    assert result["valid"], result["issues"]
    frame = pd.read_csv(result["dataset_path"], float_precision="round_trip")
    root = Path(result["dataset_path"]).parent
    record_path = root / frame.quality_reduced_record_path.iloc[0]
    record = json.loads(record_path.read_bytes())
    record["source_binding"]["complex_id"] = "another_complex"
    record["record_sha256"] = subject.canonical_sha256({k:v for k,v in record.items() if k != "record_sha256"})
    record_path.write_text(json.dumps(record), encoding="utf-8")
    frame["quality_reduced_record_sha256"] = subject._sha(record_path)
    with pytest.raises(ValueError, match="another run"):
        subject.verify_standard_quality_records(frame, root, result["quality_metric"])


def test_fallback_placeholder_does_not_masquerade_as_reference_free_evidence(source):
    batch, _ = source
    job = next((batch / "targets/1abc_AB/jobs").glob("*reduced*"))
    (job / "confidence").mkdir(exist_ok=True)
    (job / "confidence/evidence.json").write_text(json.dumps({
        "schema_version": "operational_calibrator_unavailable_v1"}), encoding="utf-8")
    result = _convert(source)
    assert result["valid"], result["issues"]
    assert result["pairs"][0]["arms"]["reduced"]["feature_source"] == "postprocess_saved_reduced_pose_v1"
    assert result["pairs"][0]["arms"]["reduced"]["confidence_evidence_sha256"] is None


def test_missing_quality_columns_or_record_file_raise_readable_value_error(source):
    result = _convert(source)
    frame = pd.read_csv(result["dataset_path"], float_precision="round_trip")
    root = Path(result["dataset_path"]).parent
    with pytest.raises(ValueError, match="missing columns"):
        subject.verify_standard_quality_records(frame.drop(columns="quality_full_record_path"), root, result["quality_metric"])
    (root / frame.quality_full_record_path.iloc[0]).unlink()
    with pytest.raises(ValueError, match="absent or malformed"):
        subject.verify_standard_quality_records(frame, root, result["quality_metric"])


def test_old_quality_predictions_cannot_follow_new_observed_labels(source):
    _, prepared = source
    path = Path(prepared["dataset_path"])
    frame = pd.read_csv(path, float_precision="round_trip")
    for column in ("probability", "prediction", "prediction_lower", "preserved"):
        frame[column] = 1.0
    frame.to_csv(path, index=False)
    result = _convert(source)
    assert result["valid"], result["issues"]
    converted = pd.read_csv(result["dataset_path"])
    assert not {"probability", "prediction", "prediction_lower", "preserved"} & set(converted.columns)
