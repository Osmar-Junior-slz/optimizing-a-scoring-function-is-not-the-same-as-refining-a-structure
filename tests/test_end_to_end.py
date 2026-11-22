import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from docking_confidence.application.demo import make_demo
from docking_confidence.application.predict import predict
from docking_confidence.application.train import train
from docking_confidence.audit.manifest import build_manifest
from docking_confidence.io.config import load_config


def test_end_to_end_demo_is_auditable_but_never_authorized(tmp_path: Path):
    dataset = tmp_path / "demo.csv"
    out = tmp_path / "run"
    make_demo(str(dataset), seed=99, n_complexes=25, runs_per_complex=8)
    config_path = Path(__file__).parents[1] / "configs" / "demo.yaml"
    result = train(str(dataset), load_config(config_path), str(out))

    for key in (
        "artifact",
        "report",
        "probability_calibration",
        "risk_calibration",
        "locked_test",
        "policy_candidates",
        "config",
        "manifest",
    ):
        assert Path(result[key]).exists()
    assert result["decision_capable"] is False

    artifact = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))
    assert artifact["schema_version"] == "2.0"
    assert artifact["calibration"]["method"] == "inductive_venn_abers_pav_v1"
    assert artifact["policy"]["method"] == "learn_then_test_bonferroni_binomial_iid_v1"
    assert artifact["dataset_provenance"]["passed"] is False
    assert artifact["authorization"]["gates"]["profile_allows_automatic_decision"] is False

    source = (
        pd.read_csv(dataset)
        .groupby("complex_id", sort=False, as_index=False)
        .head(1)
        .iloc[:4]
        .copy()
    )
    source["complex_id"] = [f"NEW-{index}" for index in range(len(source))]
    source["ligand_sequence"] = source["ligand_sequence"].astype(str) + "A"
    inference = tmp_path / "inference.csv"
    source[
        [
            "complex_id",
            "run_id",
            "algorithm",
            "seed",
            "receptor_sequence",
            "ligand_sequence",
            "engine_reduced",
            "scientific_reduced",
            *artifact["feature_columns"],
        ]
    ].to_csv(
        inference, index=False
    )
    output = tmp_path / "predictions.csv"
    predictions = predict(result["artifact"], str(inference), str(output))
    assert output.exists()
    assert set(predictions["action"]) == {"RUN_FULL"}
    assert (
        (predictions["probability_lower"] < predictions["probability"])
        & (predictions["probability"] < predictions["probability_upper"])
    ).all()

    base = artifact["base_model"]
    X = source[artifact["feature_columns"]].to_numpy(dtype=float)
    expected_raw = (
        (X - np.asarray(base["scaler_mean"]))
        / np.asarray(base["scaler_scale"])
    ) @ np.asarray(base["coef"]) + float(base["intercept"])
    np.testing.assert_allclose(predictions["base_raw_score"], expected_raw, rtol=0, atol=1e-12)

    artifact_path = Path(result["artifact"])
    manifest_path = Path(result["manifest"])
    original_artifact = artifact_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    artifact["decision_capable"] = "false"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    manifest_payload = json.loads(original_manifest.decode("utf-8"))
    manifested_paths = [
        (manifest_path.parent / entry["path"]).resolve()
        for section in ("bundle_files", "source_inputs")
        for entry in manifest_payload[section]
    ]
    manifest_path.write_text(
        json.dumps(build_manifest(manifested_paths)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="boolean"):
        predict(result["artifact"], str(inference), str(tmp_path / "forged.csv"))

    artifact_path.write_bytes(original_artifact)
    manifest_path.write_bytes(original_manifest)
    artifact = json.loads(original_artifact.decode("utf-8"))
    artifact["created_at_utc"] = "tampered"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        predict(result["artifact"], str(inference), str(tmp_path / "tampered.csv"))
