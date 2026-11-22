"""Continuous Generalized Venn–Abers creation, persistence and active entrypoints."""
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from docking_confidence.application.generalized_calibrator import (
    SCHEMA, create_calibrator, create_from_source, load_calibrator,
    margin_support, preservation_margin, proper_calibration_partition,
)
from docking_confidence.application.predict import predict
from docking_confidence.calibration.generalized_venn_abers import GeneralizedVennAbersCalibrator
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.features.calibrator_descriptors import FEATURE_COLUMNS
from docking_confidence.modeling.hybrid import METHOD as HYBRID_METHOD
from test_training_context import _context_frame, _local_config


APP = Path(__file__).parents[1] / "src/docking_confidence/ui/app.py"




def test_default_binding_installs_generalized_quality_and_retires_legacy_decision_gate():
    import yaml
    from docking_confidence.application.calibrated_batch import configured_calibrator_path
    root = Path(__file__).parents[1]
    binding = yaml.safe_load((root / "configs/calibrator.yaml").read_text(encoding="utf-8"))
    assert binding["purpose"] == "generalized_quality_prediction"
    assert "calibrador_gva_2025_hibrido_15" in binding["artifact_path"]
    assert configured_calibrator_path() is None


def _dataset(tmp_path):
    config = _local_config()
    # Historical policy gates do not participate in generalized regression fitting.
    config["calibration"]["minimum_independent_complexes"] = 10000
    dataset = tmp_path / "dataset.csv"
    frame = _context_frame(dataset, config, fifty=True)
    # Synthetic graph summaries for this I/O fixture; real extraction is tested separately.
    frame["graph_receptor_residue_count"] = frame.receptor_sequence.str.len().astype(float)
    frame["graph_ligand_residue_count"] = frame.ligand_sequence.str.len().astype(float)
    frame["graph_degree_std"] = 0.0
    frame["graph_max_degree"] = 0.0
    frame["graph_connected_component_count"] = frame.graph_receptor_residue_count + frame.graph_ligand_residue_count
    config_file = tmp_path / "config.json"
    config_file.write_text(json.dumps(config), encoding="utf-8")
    frame["config"] = str(config_file)
    frame.to_csv(dataset, index=False)
    return dataset, config, frame, config_file


def test_continuous_margin_sign_matches_both_original_preservation_conditions():
    rule = {"minimum_reduced_dockq": .23, "maximum_dockq_loss": .05}
    reduced = np.array([0., .23, .23, .5, .5, 1., 1., .229, .8])
    full = np.array([1., .23, .28, .56, .54, 0., 1., .229, .8])
    actual = preservation_margin(reduced, full, rule)
    expected = [PreservationRule(**rule).label(r, f) for r, f in zip(reduced, full)]
    np.testing.assert_array_equal(actual >= 0, np.asarray(expected, dtype=bool))
    # Check a large collection without using labels to pick convenient examples.
    rng = np.random.default_rng(2026)
    reduced, full = rng.random(1000), rng.random(1000)
    actual = preservation_margin(reduced, full, rule)
    expected = [PreservationRule(**rule).label(r, f) for r, f in zip(reduced, full)]
    np.testing.assert_array_equal(actual >= 0, np.asarray(expected, dtype=bool))


def test_margin_support_is_derived_from_dockq_domain_and_attained_at_extremes():
    rule = {"minimum_reduced_dockq": .23, "maximum_dockq_loss": .05}
    lower, upper = margin_support(rule)
    assert lower == pytest.approx(-.95) and upper == pytest.approx(.77)
    np.testing.assert_allclose(preservation_margin(np.array([0., 1.]), np.array([1., 0.]), rule), [lower, upper])
    grid_r, grid_f = np.meshgrid(np.linspace(0., 1., 101), np.linspace(0., 1., 101))
    margins = preservation_margin(grid_r, grid_f, rule)
    assert np.all(margins >= lower) and np.all(margins <= upper)


@pytest.mark.parametrize("reduced,full", [([np.nan], [.5]), ([.5], [np.inf]), ([-.01], [.5]), ([.5], [1.01]), ([.2, .3], [.5])])
def test_margin_rejects_invalid_or_misaligned_dockq(reduced, full):
    with pytest.raises(ValueError):
        preservation_margin(reduced, full, {"minimum_reduced_dockq": .23, "maximum_dockq_loss": .05})


def test_generalized_creation_has_one_artifact_two_roles_and_continuous_outputs(tmp_path, monkeypatch):
    from docking_confidence.risk import policy
    monkeypatch.setattr(policy, "select_policy", lambda *a, **k: pytest.fail("must not select a policy"))
    dataset, config, frame, _ = _dataset(tmp_path)
    result = create_calibrator(dataset, config, tmp_path / "gva")
    artifact = result["artifact"]
    assert artifact["schema_version"] == SCHEMA
    assert artifact["artifact_type"] == "generalized_quality_calibrator"
    assert artifact["settings"]["publication_year"] == 2025
    assert artifact["settings"]["loss"] == "squared_error"
    base = artifact["base_model"]
    assert base["method"] == HYBRID_METHOD
    assert base["parameters"]["weights"] == {"ridge": .5, "extra_trees": .5}
    assert base["parameters"]["ridge"]["alpha"] == 1.0
    assert base["parameters"]["extra_trees"]["max_depth"] == 4
    assert set(base["components"]) == {"ridge", "extra_trees"}
    assert all("calibration" not in component for component in base["components"].values())
    assert artifact["feature_columns"] == list(FEATURE_COLUMNS)
    assert len(artifact["feature_columns"]) == 15
    assert artifact["source_feature_columns"] == artifact["feature_columns"]
    assert "candidate_feature_columns" not in artifact and "descriptor_selection" not in artifact
    assert not any(c.startswith("algorithm_") for c in artifact["feature_columns"])
    assert artifact["target"]["kind"] == "continuous_regression"
    np.testing.assert_allclose(artifact["target"]["outcome_bounds"], [-.95, .77])
    assert set(artifact["partition"]) == {"proper_training", "calibration"}
    train, cal = [set(artifact["partition"][role]["groups"]) for role in ["proper_training", "calibration"]]
    assert not train & cal and train | cal == set(frame.complex_fingerprint_sha256)
    assert result["report"]["calibrator_count"] == 1
    assert result["report"]["base_model_component_count"] == 2
    assert all(key not in artifact for key in ["policy", "decision_capable", "runtime_gate", "comparators", "probability"])
    assert not (tmp_path / "gva/policy_candidates.csv").exists()
    assert artifact["calibration"]["calibration_example_count"] == artifact["partition"]["calibration"]["row_count"]
    assert load_calibrator(result["paths"]["artifact.json"])["artifact"] == artifact
    validation = artifact["development_validation"]
    assert validation["calibration_or_test_used"] is False
    assert validation["hyperparameter_search"] is False
    assert validation["feature_selection"] is False
    assert validation["feature_count"] == 15
    assert validation["base_model_method"] == HYBRID_METHOD
    proper_frame = frame.loc[frame.complex_fingerprint_sha256.isin(train)]
    counts = proper_frame.complex_fingerprint_sha256.map(proper_frame.complex_fingerprint_sha256.value_counts())
    expected_mean = np.average(proper_frame[list(FEATURE_COLUMNS)].to_numpy(float), axis=0, weights=1.0 / counts)
    np.testing.assert_allclose(base["components"]["ridge"]["scaler_mean"], expected_mean, atol=1e-12)
    for fold in validation["folds"]:
        assert not set(fold["training_groups"]) & set(fold["validation_groups"])
        assert not set(fold["training_groups"]) & cal
        assert not set(fold["validation_groups"]) & cal
    cv_rows = pd.read_csv(result["paths"]["training_cross_validation_predictions.csv"])
    assert set(cv_rows.complex_fingerprint_sha256) == train
    assert result["report"]["calibration_diagnostics"]["negative_row_count"] >= 0

    output = predict(result["paths"]["artifact.json"], str(dataset), str(tmp_path / "predictions.csv"))
    forbidden = ["p0", "p1", "probability", "probability_lower", "probability_upper", "action",
                 "authorized_threshold", "dockq_full", "dockq_reduced", "preserved"]
    assert all(column not in output for column in forbidden)
    assert np.isfinite(output[["prediction", "prediction_lower", "prediction_upper"]].to_numpy()).all()
    assert all(pd.api.types.is_numeric_dtype(output[column]) for column in FEATURE_COLUMNS)
    np.testing.assert_array_equal(output.base_raw_score.to_numpy(),
        .5 * output.base_ridge_score.to_numpy() + .5 * output.base_extra_trees_score.to_numpy())
    assert output.prediction_lower.le(output.prediction_upper).all()
    assert output.prediction_lower.ge(-.95).all() and output.prediction_upper.le(.77).all()
    fitted = GeneralizedVennAbersCalibrator.from_export(artifact["calibration"])
    np.testing.assert_array_equal(output.prediction.to_numpy(), fitted.predict_point(output.base_raw_score.to_numpy()))
    low, high = fitted.predict_bounds(output.base_raw_score.to_numpy())
    np.testing.assert_array_equal(output.prediction_lower.to_numpy(), low)
    np.testing.assert_array_equal(output.prediction_upper.to_numpy(), high)

    # No FULL result, observed DockQ or training target is needed at inference.
    reduced = tmp_path / "reduced_only.csv"
    frame.drop(columns=["dockq_full", "dockq_reduced", "runtime_full_sec", "engine_full", "scientific_full"]).to_csv(reduced, index=False)
    inference = predict(result["paths"]["artifact.json"], str(reduced), str(tmp_path / "reduced_predictions.csv"))
    for column in ["prediction", "prediction_lower", "prediction_upper"]:
        np.testing.assert_array_equal(output[column].to_numpy(), inference[column].to_numpy())
    # The constant contact flag is excluded; all fifteen declared descriptors are required.
    discarded = set(config["features"]["columns"]) - set(artifact["feature_columns"])
    minimal = tmp_path / "fixed_descriptors.csv"
    frame.drop(columns=list(discarded)).to_csv(minimal, index=False)
    single = predict(result["paths"]["artifact.json"], str(minimal), str(tmp_path / "single_predictions.csv"))
    np.testing.assert_array_equal(single.prediction.to_numpy(), output.prediction.to_numpy())
    examples = pd.read_csv(result["paths"]["calibration_examples.csv"])
    assert not set(examples.columns) & discarded
    assert not set(single.columns) & discarded
    np.testing.assert_allclose(examples.base_raw_score,
        .5 * examples.base_ridge_score + .5 * examples.base_extra_trees_score, atol=1e-12)


def test_fixed_training_does_not_require_excluded_contact_flag(tmp_path):
    dataset, config, frame, _ = _dataset(tmp_path)
    first = create_calibrator(dataset, config, tmp_path / "with_flag")
    frame.drop(columns=["graph_has_interface_contact"]).to_csv(dataset, index=False)
    second = create_calibrator(dataset, config, tmp_path / "without_flag")
    assert first["artifact"]["base_model"] == second["artifact"]["base_model"]
    assert first["artifact"]["calibration"] == second["artifact"]["calibration"]
    assert second["verified"]


def test_calibration_outcomes_do_not_change_base_and_constant_calibration_is_allowed(tmp_path):
    dataset, config, frame, _ = _dataset(tmp_path)
    first = create_calibrator(dataset, config, tmp_path / "first")
    _, calibration = proper_calibration_partition(frame.complex_fingerprint_sha256)
    frame.loc[calibration, "dockq_reduced"] = 0.
    frame.loc[calibration, "dockq_full"] = .9
    frame.loc[calibration, "viable_fraction_mean"] = .987654321
    frame.to_csv(dataset, index=False)
    second = create_calibrator(dataset, config, tmp_path / "second")
    assert first["artifact"]["base_model"] == second["artifact"]["base_model"]
    assert "descriptor_selection" not in first["artifact"]
    assert first["artifact"]["calibration"] != second["artifact"]["calibration"]
    assert second["report"]["calibration_preserved_count"] == 0
    assert second["report"]["calibration_not_preserved_count"] == len(calibration)
    examples = pd.read_csv(second["paths"]["calibration_examples.csv"])
    np.testing.assert_allclose(examples.outcome.to_numpy(), -.85)
    assert load_calibrator(second["paths"]["artifact.json"])["verified"]


def test_constant_zero_target_has_no_undefined_skill_warning_and_ui_can_reload(tmp_path):
    dataset, config, frame, _ = _dataset(tmp_path)
    frame["dockq_reduced"] = config["preservation"]["minimum_reduced_dockq"]
    frame["dockq_full"] = frame["dockq_reduced"]
    frame.to_csv(dataset, index=False)
    result = create_calibrator(dataset, config, tmp_path / "constant")
    validation = result["report"]["development_validation"]
    assert validation["available"]
    assert validation["group_mse"] == 0
    assert validation["skill_against_fold_training_mean"] is None
    assert "proper_training_group_validation_did_not_improve_over_training_mean" not in result["report"]["warnings"]
    app = AppTest.from_file(str(APP))
    app.session_state["gva_output"] = str(tmp_path / "constant")
    app.run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.get("download_button")) == 1


def test_zip_creation_reload_immutability_and_hash_tampering(tmp_path):
    dataset, _, _, _ = _dataset(tmp_path)
    source = tmp_path / "results.zip"
    with ZipFile(source, "w") as archive:
        archive.write(dataset, "study/analise_artigo/calibrador_previsoes_e_qualidade.csv")
    result = create_from_source(source, tmp_path / "gva")
    assert result["verified"] and result["report"]["calibrator_count"] == 1
    original = Path(result["paths"]["artifact.json"]).read_bytes()
    with pytest.raises(FileExistsError):
        create_from_source(source, tmp_path / "gva")
    assert Path(result["paths"]["artifact.json"]).read_bytes() == original
    examples = Path(result["paths"]["calibration_examples.csv"])
    examples.write_bytes(examples.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="mismatch"):
        load_calibrator(result["paths"]["artifact.json"])


def test_zip_retry_after_config_error_reuses_only_identical_snapshot(tmp_path):
    dataset, _, _, config_file = _dataset(tmp_path)
    source = tmp_path / "results.zip"
    member = "study/analise_artigo/calibrador_previsoes_e_qualidade.csv"
    with ZipFile(source, "w") as archive:
        archive.write(dataset, member)
    output = tmp_path / "retry"
    with pytest.raises(FileNotFoundError):
        create_from_source(source, output, config_path=tmp_path / "missing.json")
    snapshot = output / "inputs/dataset.csv"
    original = snapshot.read_bytes()
    assert original == dataset.read_bytes()
    changed_source = tmp_path / "changed.zip"
    with ZipFile(changed_source, "w") as archive:
        archive.writestr(member, original + b"\n")
    with pytest.raises(FileExistsError, match="outra origem"):
        create_from_source(changed_source, output, config_path=config_file)
    assert snapshot.read_bytes() == original
    result = create_from_source(source, output, config_path=config_file)
    assert result["verified"]
    assert snapshot.read_bytes() == original


def test_prediction_refuses_to_overwrite_input_or_bundle_files(tmp_path):
    dataset, config, _, _ = _dataset(tmp_path)
    result = create_calibrator(dataset, config, tmp_path / "gva")
    artifact = result["paths"]["artifact.json"]
    for path in [dataset, *map(Path, result["paths"].values())]:
        original = path.read_bytes()
        with pytest.raises(ValueError, match="sobrescrever"):
            predict(artifact, str(dataset), str(path))
        assert path.read_bytes() == original


def test_active_ui_creates_generalized_once_and_reloads_existing_bundle(tmp_path, monkeypatch):
    from docking_confidence.ui import calibrator_page
    dataset, _, _, config_file = _dataset(tmp_path)
    calls = []
    original = calibrator_page.create_from_source
    def observed(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(calibrator_page, "create_from_source", observed)
    app = AppTest.from_file(str(APP))
    app.session_state["gva_source"] = str(dataset)
    app.session_state["gva_output"] = str(tmp_path / "gva")
    app.session_state["gva_config"] = str(config_file)
    app.run(timeout=30)
    assert not app.exception and not calls
    assert not any(button.key.startswith("article_") or button.key.startswith("ivap_") for button in app.button)
    app.button(key="gva_create").click().run(timeout=30)
    assert not app.exception and not app.error and len(calls) == 1
    assert len(app.get("download_button")) == 1 and len(app.table[0].value) == 2
    artifact = json.loads((tmp_path / "gva/artifact.json").read_bytes())
    assert artifact["schema_version"] == SCHEMA
    app.run(timeout=30)
    assert not app.exception and len(calls) == 1
    restored = AppTest.from_file(str(APP))
    restored.session_state["gva_output"] = str(tmp_path / "gva")
    restored.run(timeout=30)
    assert not restored.exception and not restored.error
    assert len(restored.get("download_button")) == 1


def test_active_cli_emits_paths_for_one_continuous_generalized_calibrator(tmp_path, monkeypatch, capsys):
    import sys
    from docking_confidence.cli import main
    dataset, _, _, config_file = _dataset(tmp_path)
    monkeypatch.setattr(sys, "argv", ["docking-confidence", "train", "--dataset", str(dataset),
        "--config", str(config_file), "--out-dir", str(tmp_path / "gva")])
    main()
    paths = json.loads(capsys.readouterr().out)
    artifact = json.loads(Path(paths["artifact.json"]).read_bytes())
    assert artifact["schema_version"] == SCHEMA
    assert artifact["target"]["kind"] == "continuous_regression" and "policy" not in artifact


def test_invalid_ui_input_reports_failure(tmp_path):
    app = AppTest.from_file(str(APP))
    app.session_state["gva_source"] = str(tmp_path / "missing.csv")
    app.session_state["gva_output"] = str(tmp_path / "gva")
    app.run(timeout=30)
    app.button(key="gva_create").click().run(timeout=30)
    assert not app.exception and app.error
    assert not (tmp_path / "gva/artifact.json").exists()

