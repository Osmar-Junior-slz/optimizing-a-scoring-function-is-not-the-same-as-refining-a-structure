from io import BytesIO
import json
from zipfile import ZipFile

import numpy as np
import pytest

from docking_confidence.application.generalized_calibrator import create_calibrator, load_calibrator, predict_calibrator
from docking_confidence.application.portable_calibrator_export import export_calibrator_zip
from docking_confidence.audit.manifest import read_verified_bundle, sha256_bytes
from test_generalized_creation import _dataset


def test_portable_zip_survives_removed_original_source_and_preserves_predictions(tmp_path):
    source_root = tmp_path / "external_source"
    source_root.mkdir()
    dataset, config, frame, config_file = _dataset(source_root)
    result = create_calibrator(dataset, config, tmp_path / "frozen_model")
    artifact = tmp_path / "frozen_model/artifact.json"
    manifest = artifact.parent / "audit_manifest.json"
    originals = artifact.read_bytes(), manifest.read_bytes()
    before = predict_calibrator(artifact, dataset, tmp_path / "before.csv")
    payload = export_calibrator_zip(artifact)
    assert (artifact.read_bytes(), manifest.read_bytes()) == originals
    with ZipFile(BytesIO(payload)) as archive:
        assert archive.read("artifact.json") == originals[0]
        assert archive.read("provenance/audit_manifest_original.json") == originals[1]
        assert archive.read("inputs/dataset.csv") == dataset.read_bytes()
        exported_manifest = json.loads(archive.read("audit_manifest.json"))
        assert exported_manifest["source_inputs"][0]["path"] == "inputs/dataset.csv"
        archive.extractall(tmp_path / "moved_export")
    dataset.unlink()
    config_file.unlink()
    moved = tmp_path / "moved_export"
    assert load_calibrator(moved / "artifact.json")["verified"]
    verified = read_verified_bundle(moved / "audit_manifest.json")
    for path, entry in verified.source_inputs.items():
        assert sha256_bytes(path.read_bytes()) == entry["sha256"]
    after = predict_calibrator(moved / "artifact.json", moved / "inputs/dataset.csv", tmp_path / "after.csv")
    for column in ("prediction", "prediction_lower", "prediction_upper", "base_raw_score"):
        np.testing.assert_array_equal(before[column].to_numpy(), after[column].to_numpy())
    (moved / "inputs/dataset.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        export_calibrator_zip(moved / "artifact.json")
