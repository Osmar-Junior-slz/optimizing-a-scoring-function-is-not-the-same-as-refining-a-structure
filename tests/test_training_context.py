from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest

from docking_confidence.application.predict import predict
from docking_confidence.application.train import TrainingEvidenceError, train
from docking_confidence.artifact.schema import validate_artifact
from docking_confidence.io.config import (
    CONFIG_DIGEST_SCHEMA, FEATURE_SCHEMA_VERSION,
    LOCAL_REFINEMENT_FEATURE_COLUMNS, LOCAL_REFINEMENT_FEATURE_SCHEMA_VERSION,
    config_sha256, load_config, validate_config,
)
from docking_confidence.io.dataset import load_paired_dataset, load_reduced_inference_dataset
from test_authorized_path import _graph_metadata, _make_trusted_dataset


ROOT = Path(__file__).parents[1]


def _local_config():
    return load_config(ROOT / "configs/local_refinement_pilot.yaml")


def _context_frame(path, config, *, fifty=False):
    frame = _make_trusted_dataset(path, config)
    if fifty:
        additional = frame.copy()
        additional["complex_id"] = "extra-" + additional["complex_id"]
        additional["ligand_sequence"] += "A"
        for index in additional.index:
            for key, value in _graph_metadata(additional.loc[index], config["contact_graph"]).items():
                additional.loc[index, key] = value
        frame = pd.concat([frame, additional], ignore_index=True)
    frame["docking_task"] = config["docking_task"]
    frame["feature_schema"] = config["features"]["schema_version"]
    frame["config"] = str(ROOT / "configs/local_refinement_pilot.yaml")
    frame["config_sha256"] = config_sha256(config)
    if config["docking_task"] == "local_refinement":
        frame["scan_gap_norm"] = None
        frame["energy_q20_norm"] = None
    frame.to_csv(path, index=False)
    return frame


def _load(path, config, *, inference=False):
    loader = load_reduced_inference_dataset if inference else load_paired_dataset
    return loader(path, config["features"]["columns"], config["contact_graph"],
                  docking_task=config.get("docking_task"),
                  feature_schema=config["features"]["schema_version"])


@pytest.mark.parametrize("original,local", [
    ("pilot.yaml", "local_refinement_pilot.yaml"),
    ("default.yaml", "local_refinement_default.yaml"),
])
def test_local_profiles_preserve_scientific_gates(original, local):
    previous = load_config(ROOT / "configs" / original)
    config = load_config(ROOT / "configs" / local)
    assert config["docking_task"] == "local_refinement"
    assert config["features"]["schema_version"] == LOCAL_REFINEMENT_FEATURE_SCHEMA_VERSION
    assert tuple(config["features"]["columns"]) == LOCAL_REFINEMENT_FEATURE_COLUMNS
    assert len(config["features"]["columns"]) == 11
    for section in previous.keys() - {"features", "authorization", "runtime_gate"}:
        assert config[section] == previous[section]
    # Local profiles now assess the complete paired cascade, including fallback
    # and measured feature/inference overhead. Numeric legacy gates are retained.
    for key, value in previous["runtime_gate"].items():
        assert config["runtime_gate"][key] == value
    assert config["runtime_gate"]["assessment_scope"] == "total_paired_cascade"
    assert config["runtime_gate"]["require_measured_overhead"] is True
    assert config["runtime_gate"]["overhead_column"] == "confidence_overhead_sec"
    previous_authorization = previous["authorization"]
    authorization = config["authorization"]
    assert authorization["profile"] == previous_authorization["profile"]
    assert authorization["require_scientific_provenance"] is True
    assert previous_authorization["allow_automatic_decision"] is False
    # This independent development base has no authorized model or new commitment.
    assert authorization["allow_automatic_decision"] is False
    assert authorization["protocol_commitment_sha256"] is None
    if local == "local_refinement_default.yaml":
        assert authorization["trusted_engine_allowlist"] == ["dockingpp_local"]
    else:
        assert authorization == previous_authorization
        assert authorization["allow_automatic_decision"] is False


def test_config_digest_canonical_and_effective_settings_bound():
    config = _local_config()
    assert config_sha256(config) == config_sha256(dict(reversed(list(config.items()))))
    changed = deepcopy(config)
    changed["model"]["C"] = 2
    assert config_sha256(config) != config_sha256(changed)
    changed = deepcopy(config)
    changed["features"]["columns"].append("scan_gap_norm")
    with pytest.raises(ValueError, match="frozen"):
        validate_config(changed)


def test_policy_configuration_accepts_fixed_threshold():
    config = _local_config()
    config["policy"].update(threshold_min=0.9, threshold_max=0.9)
    validated = validate_config(config)
    assert validated["policy"]["threshold_min"] == 0.9
    assert validated["policy"]["threshold_max"] == 0.9


def test_policy_configuration_rejects_reversed_threshold_bounds():
    config = _local_config()
    config["policy"].update(threshold_min=0.95, threshold_max=0.9)
    with pytest.raises(ValueError, match="threshold_min"):
        validate_config(config)


def test_local_dataset_without_scan_features_passes_official_loaders(tmp_path):
    config = _local_config()
    path = tmp_path / "local.csv"
    source = _context_frame(path, config)
    for inference in (False, True):
        result = _load(path, config, inference=inference)
        assert len(result) == len(source)
        assert result["scan_gap_norm"].isna().all()
        assert result["energy_q20_norm"].isna().all()


@pytest.mark.parametrize("column,value,match", [
    ("docking_task", "surface_discovery", "mixed training context"),
    ("feature_schema", FEATURE_SCHEMA_VERSION, "mixed training context"),
    ("config_sha256", "bad", "mixed training context"),
    ("viable_fraction_mean", None, "finite"),
])
def test_context_or_available_features_cannot_be_mixed_or_missing(tmp_path, column, value, match):
    config = _local_config()
    path = tmp_path / "local.csv"
    frame = _context_frame(path, config)
    frame.loc[0, column] = value
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        _load(path, config)


def test_local_context_cannot_be_removed_or_disguised(tmp_path):
    config = _local_config()
    path = tmp_path / "local.csv"
    frame = _context_frame(path, config)
    frame.drop(columns=["docking_task", "feature_schema", "config", "config_sha256"]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        _load(path, config)
    frame["docking_task"] = "surface_discovery"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="incompatible"):
        _load(path, config)


def test_surface_context_retains_scan_schema_and_legacy_remains_compatible(tmp_path):
    config = load_config(ROOT / "configs/pilot.yaml")
    path = tmp_path / "surface.csv"
    _make_trusted_dataset(path, config)
    assert len(_load(path, config)) == 50
    config["docking_task"] = "surface_discovery"
    frame = _context_frame(path, config)
    frame["config"] = str(ROOT / "configs/pilot.yaml")
    frame.to_csv(path, index=False)
    loaded = _load(path, config)
    assert loaded["scan_gap_norm"].notna().all()
    assert loaded["energy_q20_norm"].notna().all()


def test_train_rejects_configuration_changed_after_dataset_build(tmp_path):
    config = _local_config()
    path = tmp_path / "local.csv"
    _context_frame(path, config)
    config["model"]["C"] = 3.0
    with pytest.raises(ValueError, match="config_sha256 disagrees"):
        train(str(path), config, str(tmp_path / "out"))
    assert not (tmp_path / "out/artifact.json").exists()


def test_calibration_failure_exposes_exact_existing_gate_without_artifact(tmp_path):
    config = _local_config()
    path = tmp_path / "local.csv"
    _context_frame(path, config)
    events = []
    with pytest.raises(TrainingEvidenceError, match="total=5/10") as caught:
        train(str(path), config, str(tmp_path / "out"),
              progress_callback=lambda fraction, message: events.append((fraction, message)))
    assert caught.value.stage == "probability_calibration"
    evidence = caught.value.evidence
    assert evidence["independent_complex_count"] == 5
    assert evidence["minimum_independent_complexes"] == 10
    assert evidence["positive_class_complex_count"] == 5
    assert evidence["negative_class_complex_count"] == 5
    assert evidence["model_trained"] is True
    assert evidence["probability_calibrated"] is False
    assert evidence["split"]["random_seed"] == config["validation"]["random_seed"]
    assert not (tmp_path / "out/artifact.json").exists()
    assert all(fraction < 1.0 for fraction, _ in events)
    assert "Venn–Abers" in events[-1][1]


@pytest.fixture
def contextual_training(tmp_path):
    config = _local_config()
    path = tmp_path / "local.csv"
    frame = _context_frame(path, config, fifty=True)
    result = train(str(path), config, str(tmp_path / "out"))
    artifact = json.loads(Path(result["artifact"]).read_text(encoding="utf-8"))
    return config, path, frame, result, artifact


def test_context_is_bound_in_artifact_report_and_predict(contextual_training, tmp_path):
    config, path, frame, result, artifact = contextual_training
    assert validate_artifact(artifact) is artifact
    assert artifact["docking_task"] == "local_refinement"
    assert artifact["feature_schema"] == LOCAL_REFINEMENT_FEATURE_SCHEMA_VERSION
    assert artifact["training_config_source"] == {
        "path": str(ROOT / "configs/local_refinement_pilot.yaml"),
        "sha256": config_sha256(config), "digest_schema": CONFIG_DIGEST_SCHEMA,
    }
    report = json.loads(Path(result["report"]).read_text(encoding="utf-8"))
    for key in ("docking_task", "feature_schema", "training_config_source"):
        assert report[key] == artifact[key]
    # This perfectly separated fixture has no observed acceptance errors.
    # The bootstrap must expose its degeneracy rather than certify zero risk.
    assert artifact["calibration"]["independent_complex_count"] == 10
    assert artifact["policy"]["candidate_grid"] == [0.9]
    assert artifact["policy"]["threshold"] is None
    assert artifact["policy"]["supported"] is False
    assert artifact["policy"]["best_observed_candidate"]["bootstrap_status"] == "no_observed_failures"
    assert artifact["decision_capable"] is False
    prediction = predict(result["artifact"], str(path), str(tmp_path / "prediction.csv"))
    assert len(prediction) == len(frame)
    assert set(prediction["action"]) == {"RUN_FULL"}
    source = frame.iloc[:2].copy()
    source["config_sha256"] = "a" * 64
    wrong = tmp_path / "wrong.csv"
    source.to_csv(wrong, index=False)
    with pytest.raises(ValueError, match="configuration context disagrees"):
        predict(result["artifact"], str(wrong), str(tmp_path / "wrong_predictions.csv"))


@pytest.mark.parametrize("field", ["docking_task", "feature_schema", "training_config_source"])
def test_artifact_context_cannot_be_removed_or_forged(contextual_training, field):
    *_, artifact = contextual_training
    changed = deepcopy(artifact)
    del changed[field]
    with pytest.raises(ValueError):
        validate_artifact(changed)


def test_contextual_training_outputs_remain_immutable(contextual_training, tmp_path):
    config, path, _, result, _ = contextual_training
    artifact = Path(result["artifact"])
    before = artifact.read_bytes()
    with pytest.raises(FileExistsError, match="immutable"):
        train(str(path), config, str(artifact.parent))
    assert artifact.read_bytes() == before
