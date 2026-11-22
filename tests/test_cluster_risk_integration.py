"""Exercise run-risk policy evidence across training, persistence and inference."""
from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest

from docking_confidence.application.predict import predict
from docking_confidence.application.train import train
from docking_confidence.application.training_results import load_training_result
from docking_confidence.artifact.schema import validate_artifact
from docking_confidence.audit.manifest import sha256_file
from docking_confidence.io.config import load_config, validate_config
from docking_confidence.risk.policy import assess_cluster_counts
from test_authorized_path import _graph_metadata, _make_trusted_dataset, _trusted_config


ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize("profile", [
    "default.yaml", "pilot.yaml",
    "local_refinement_default.yaml", "local_refinement_pilot.yaml",
])
def test_current_profiles_declare_reproducible_run_risk(profile):
    config = load_config(ROOT / "configs" / profile)
    assert config["policy"]["method"] == "cluster_bootstrap_run_risk"
    assert config["policy"]["bootstrap_repetitions"] == 10_000
    assert config["policy"]["bootstrap_seed"] == 20_260_909
    assert config["policy"]["threshold_min"] == 0.9
    assert config["policy"]["threshold_max"] == 0.9
    assert config["authorization"]["allow_automatic_decision"] is False


@pytest.mark.parametrize("key,value", [
    ("bootstrap_repetitions", 999),
    ("bootstrap_repetitions", 1_000_001),
    ("bootstrap_repetitions", 1000.5),
    ("bootstrap_repetitions", True),
    ("bootstrap_repetitions", None),
    ("bootstrap_seed", -1),
    ("bootstrap_seed", 1.5),
    ("bootstrap_seed", True),
    ("bootstrap_seed", None),
])
def test_bootstrap_configuration_rejects_unreplayable_settings(key, value):
    config = load_config(ROOT / "configs/default.yaml")
    config["policy"][key] = value
    with pytest.raises(ValueError, match=key):
        validate_config(config)


@pytest.fixture(scope="module")
def bootstrap_training(tmp_path_factory):
    directory = tmp_path_factory.mktemp("cluster_training")
    config = _trusted_config(ROOT)
    config["policy"].update(
        method="cluster_bootstrap_run_risk",
        bootstrap_repetitions=10_000,
        bootstrap_seed=20_260_909,
        threshold_min=0.0,
        threshold_max=0.0,
    )
    config = validate_config(config)
    dataset = directory / "trusted.csv"
    frame = _make_trusted_dataset(dataset, config, n_complexes=100)
    # Some low-score runs preserve quality. The fixed fixture pattern gives
    # different error ratios across complexes while retaining both classes.
    # The permissive 0.9 risk target is for exercising authorization plumbing,
    # and is deliberately independent of the scientific default risk target.
    frame.loc[frame.index % 6 == 1, "dockq_reduced"] = 0.8
    frame.to_csv(dataset, index=False)
    out = directory / "trained"
    result = train(str(dataset), config, str(out))
    verified = load_training_result(out)
    return config, frame, out, result, verified


def _new_inference_rows(frame, config):
    source = frame.iloc[[0, 1]].copy().reset_index(drop=True)
    source["complex_id"] = ["UNSEEN-TRUSTED", "UNSEEN-UNTRUSTED"]
    source["receptor_sequence"] = "ACDEFGHIKLMNPQRSTVWYACDEFGHIK"
    source["ligand_sequence"] = ["WYVTSRQPN", "WYVTSRQPD"]
    source.loc[1, "engine_reduced"] = "untrusted-engine"
    for index in source.index:
        for key, value in _graph_metadata(source.loc[index], config["contact_graph"]).items():
            source.loc[index, key] = value
    columns = [
        "complex_id", "run_id", "algorithm", "seed", "receptor_sequence",
        "ligand_sequence", "engine_reduced", "scientific_reduced",
        *config["features"]["columns"],
        *[column for column in source if column.startswith("graph_")],
    ]
    return source[list(dict.fromkeys(columns))]


def test_fresh_cluster_training_bundle_loads_and_respects_inference_authorization(
    bootstrap_training, tmp_path,
):
    config, frame, _, result, verified = bootstrap_training
    assert verified["verified"] is True
    artifact = verified["artifact"]
    assert artifact["policy"]["method"] == "cluster_bootstrap_run_risk_v1"
    assert artifact["decision_capable"] is True
    assert all(artifact["authorization"]["gates"].values())
    for assessment in (artifact["policy"], artifact["locked_test_audit"]):
        assert assessment["bootstrap_status"] == "ok"
        assert assessment["supported"] is True
        assert 0 < assessment["false_acceptance_row_count"] < assessment["accepted_row_count"]
        assert assessment["empirical_run_risk"] == pytest.approx(
            assessment["false_acceptance_row_count"] / assessment["accepted_row_count"]
        )
        assert assessment["empirical_run_risk"] < assessment["empirical_conditional_complex_risk"]
        assert assessment["confidence_guarantee"] == "asymptotic_not_finite_sample"

    inference = tmp_path / "unseen.csv"
    _new_inference_rows(frame, config).to_csv(inference, index=False)
    predictions = predict(result["artifact"], str(inference), str(tmp_path / "pred.csv"))
    assert predictions.loc[0, "action"] == "ACCEPT_REDUCED"
    assert bool(predictions.loc[0, "unseen_complex"])
    assert bool(predictions.loc[0, "trusted_reduced_input_provenance"])
    assert predictions.loc[1, "action"] == "RUN_FULL"
    assert predictions.loc[1, "reason"] == "reduced_run_provenance_not_scientific_or_engine_not_allowlisted"


def test_supported_cluster_policy_still_requires_profile_authorization(
    bootstrap_training, tmp_path,
):
    config, frame, *_ = bootstrap_training
    config = deepcopy(config)
    config["authorization"]["allow_automatic_decision"] = False
    dataset = tmp_path / "profile_disabled.csv"
    frame.to_csv(dataset, index=False)
    out = tmp_path / "trained"
    result = train(str(dataset), config, str(out))
    artifact = load_training_result(out)["artifact"]
    assert artifact["policy"]["supported"] is True
    assert artifact["locked_test_audit"]["supported"] is True
    assert artifact["authorization"]["gates"]["profile_allows_automatic_decision"] is False
    assert artifact["decision_capable"] is False
    inference = tmp_path / "unseen.csv"
    _new_inference_rows(frame, config).iloc[[0]].to_csv(inference, index=False)
    predictions = predict(result["artifact"], str(inference), str(tmp_path / "pred.csv"))
    assert predictions.loc[0, "action"] == "RUN_FULL"
    assert predictions.loc[0, "reason"] == "artifact_not_authorized_for_automatic_decision"


@pytest.mark.parametrize("assessment_name", ["policy", "locked_test_audit"])
def test_artifact_rejects_tampered_cluster_error_evidence(
    bootstrap_training, assessment_name,
):
    artifact = deepcopy(bootstrap_training[4]["artifact"])
    assessment = artifact[assessment_name]
    record = next(item for item in assessment["cluster_counts"] if item["false_accepted_run_count"])
    record["false_accepted_run_count"] -= 1
    with pytest.raises(ValueError, match="disagree"):
        validate_artifact(artifact)


def test_artifact_method_must_match_its_resolved_configuration(bootstrap_training):
    artifact = deepcopy(bootstrap_training[4]["artifact"])
    artifact["policy"]["method"] = "learn_then_test_bonferroni_binomial_iid_v1"
    with pytest.raises(ValueError, match="policy method"):
        validate_artifact(artifact)


def test_bundle_rejects_regrouped_errors_even_when_artifact_and_manifest_are_recomputed(
    bootstrap_training, tmp_path,
):
    _, _, original, _, verified = bootstrap_training
    out = tmp_path / "forged_bundle"
    shutil.copytree(original, out)
    artifact = deepcopy(verified["artifact"])
    records = artifact["policy"]["cluster_counts"]
    positive = next(item for item in records if item["false_accepted_run_count"] > 0)
    zero = next(item for item in records if item["false_accepted_run_count"] == 0)
    positive["false_accepted_run_count"], zero["false_accepted_run_count"] = (
        zero["false_accepted_run_count"], positive["false_accepted_run_count"],
    )
    recomputed = assess_cluster_counts(
        records, artifact["policy"]["threshold"],
        cfg=artifact["resolved_config"]["policy"],
        hypothesis_count=artifact["policy"]["tested_hypothesis_count"],
        role="risk_calibration",
    )
    artifact["policy"].update(recomputed)
    # Aggregate evidence and schema are self-consistent, but the recorded
    # errors no longer belong to the complexes in the original scored CSV.
    validate_artifact(artifact)
    (out / "artifact.json").write_text(json.dumps(artifact), encoding="utf-8")
    report = json.loads((out / "validation_report.json").read_text(encoding="utf-8"))
    report["policy"] = artifact["policy"]
    (out / "validation_report.json").write_text(json.dumps(report), encoding="utf-8")
    manifest_path = out / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for entry in manifest["bundle_files"]:
        file = out / entry["path"]
        entry["sha256"] = sha256_file(file)
        entry["size_bytes"] = file.stat().st_size
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="risk-calibration evidence does not reproduce"):
        load_training_result(out)


def test_legacy_config_needs_no_bootstrap_settings_and_legacy_bundle_remains_readable(tmp_path):
    config = _trusted_config(ROOT)
    config["policy"].pop("bootstrap_repetitions", None)
    config["policy"].pop("bootstrap_seed", None)
    config = validate_config(config)
    dataset = tmp_path / "legacy.csv"
    _make_trusted_dataset(dataset, config)
    out = tmp_path / "legacy"
    train(str(dataset), config, str(out))
    loaded = load_training_result(out)
    assert loaded["verified"] is True
    assert loaded["artifact"]["policy"]["method"] == "learn_then_test_bonferroni_binomial_iid_v1"
    assert loaded["artifact"]["decision_capable"] is True
