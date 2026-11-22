"""A subset of model inputs retains the REDUCED acquisition contract."""
from copy import deepcopy

import numpy as np
import pytest

from docking_confidence.io.dataset import load_paired_dataset, load_reduced_inference_dataset
from test_training_context import _context_frame, _local_config


SELECTED = "graph_interface_density"


@pytest.fixture(scope="module")
def original_rows(tmp_path_factory):
    config = _local_config()
    frame = _context_frame(tmp_path_factory.mktemp("single-input") / "source.csv", config)
    return config, frame


def _load(path, config, model_columns, *, source_columns=None):
    return load_reduced_inference_dataset(
        path, model_columns, config["contact_graph"],
        docking_task=config["docking_task"], feature_schema=config["features"]["schema_version"],
        context_feature_columns=source_columns,
    )


def _selected_only(frame, config):
    excluded = [column for column in config["features"]["columns"] if column != SELECTED]
    return frame.drop(columns=[*excluded, "dockq_reduced", "dockq_full", "engine_full", "scientific_full"])


def test_single_input_without_unused_descriptors_retains_full_context(original_rows, tmp_path):
    config, original = original_rows
    config_snapshot = deepcopy(config)
    source_columns = config["features"]["columns"]
    full = tmp_path / "full.csv"
    original.to_csv(full, index=False)
    reference = _load(full, config, source_columns)
    reduced = tmp_path / "single.csv"
    _selected_only(original, config).to_csv(reduced, index=False)
    loaded = _load(reduced, config, [SELECTED], source_columns=source_columns)
    assert set(source_columns) & set(loaded.columns) == {SELECTED}
    np.testing.assert_array_equal(loaded[SELECTED], reference[SELECTED])
    assert loaded["complex_fingerprint_sha256"].tolist() == reference["complex_fingerprint_sha256"].tolist()
    assert loaded["config_sha256"].tolist() == reference["config_sha256"].tolist()
    assert loaded["feature_schema"].tolist() == reference["feature_schema"].tolist()
    assert config == config_snapshot


def test_present_unused_descriptor_values_are_not_model_inputs(original_rows, tmp_path):
    config, original = original_rows
    frame = original.copy()
    unused = [column for column in config["features"]["columns"] if column != SELECTED]
    frame[unused] = "unused text"
    path = tmp_path / "unused.csv"
    frame.to_csv(path, index=False)
    loaded = _load(path, config, [SELECTED], source_columns=config["features"]["columns"])
    np.testing.assert_array_equal(loaded[SELECTED], original[SELECTED])


def test_default_loader_still_requires_its_frozen_feature_schema(original_rows, tmp_path):
    config, original = original_rows
    path = tmp_path / "single.csv"
    _selected_only(original, config).to_csv(path, index=False)
    with pytest.raises(ValueError, match="available feature schema"):
        _load(path, config, [SELECTED])
    with pytest.raises(ValueError, match="missing required columns"):
        _load(path, config, config["features"]["columns"])


@pytest.mark.parametrize("model_columns,source_columns", [
    ([], [SELECTED]), ([SELECTED, SELECTED], [SELECTED]),
    ([SELECTED], []), ([SELECTED], [SELECTED, SELECTED]),
    ([SELECTED], SELECTED), (SELECTED, [SELECTED]),
    ([" "], [SELECTED]), ([SELECTED], [None]),
])
def test_subset_contract_rejects_malformed_names(original_rows, tmp_path, model_columns, source_columns):
    config, _ = original_rows
    with pytest.raises(ValueError, match="unique nonempty names"):
        _load(tmp_path / "not-read.csv", config, model_columns, source_columns=source_columns)


@pytest.mark.parametrize("name", ["dockq_reduced", "dockq_full", "algorithm_ag", "complex_id"])
def test_subset_cannot_introduce_targets_or_other_non_descriptor_columns(original_rows, tmp_path, name):
    config, _ = original_rows
    with pytest.raises(ValueError, match="original context feature contract"):
        _load(tmp_path / "not-read.csv", config, [name], source_columns=config["features"]["columns"])


@pytest.mark.parametrize("column,value,match", [
    (SELECTED, np.nan, "finite"),
    ("config_sha256", "bad", "64 lowercase hexadecimal"),
    ("feature_schema", "wrong_schema", "incompatible"),
    ("graph_pose_protocol", "full", "graph-v2 contract"),
])
def test_subset_preserves_selected_value_and_provenance_checks(original_rows, tmp_path, column, value, match):
    config, original = original_rows
    frame = _selected_only(original, config)
    frame[column] = value
    path = tmp_path / "invalid.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match=match):
        _load(path, config, [SELECTED], source_columns=config["features"]["columns"])


def test_subset_requires_its_selected_descriptor(original_rows, tmp_path):
    config, original = original_rows
    path = tmp_path / "missing.csv"
    _selected_only(original, config).drop(columns=SELECTED).to_csv(path, index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        _load(path, config, [SELECTED], source_columns=config["features"]["columns"])


def test_paired_subset_accepts_ten_base_inputs_without_discarded_contact_flag(original_rows, tmp_path):
    config, original = original_rows
    source_columns = config["features"]["columns"]
    model_columns = [column for column in source_columns if column != "graph_has_interface_contact"]
    path = tmp_path / "paired_without_flag.csv"
    original.drop(columns="graph_has_interface_contact").to_csv(path, index=False)
    loaded = load_paired_dataset(
        path, model_columns, config["contact_graph"],
        docking_task=config["docking_task"], feature_schema=config["features"]["schema_version"],
        context_feature_columns=source_columns,
    )
    assert "graph_has_interface_contact" not in loaded
    assert len(set(model_columns) & set(loaded.columns)) == 10
    np.testing.assert_allclose(loaded[model_columns], original[model_columns])
    np.testing.assert_allclose(loaded[["dockq_reduced", "dockq_full"]], original[["dockq_reduced", "dockq_full"]])
    assert loaded["config_sha256"].tolist() == original["config_sha256"].tolist()
    assert loaded["scientific"].all()


@pytest.mark.parametrize("column", ["dockq_reduced", "dockq_full", "algorithm_ag", "complex_id"])
def test_paired_subset_cannot_treat_outcomes_or_identifiers_as_descriptors(original_rows, tmp_path, column):
    config, _ = original_rows
    with pytest.raises(ValueError, match="original context feature contract"):
        load_paired_dataset(
            tmp_path / "not-read.csv", [column], config["contact_graph"],
            docking_task=config["docking_task"], feature_schema=config["features"]["schema_version"],
            context_feature_columns=config["features"]["columns"],
        )


def test_paired_default_preserves_frozen_schema_and_requires_the_contact_flag(original_rows, tmp_path):
    config, original = original_rows
    source_columns = config["features"]["columns"]
    model_columns = [column for column in source_columns if column != "graph_has_interface_contact"]
    path = tmp_path / "without_flag.csv"
    original.drop(columns="graph_has_interface_contact").to_csv(path, index=False)
    kwargs = {"docking_task": config["docking_task"], "feature_schema": config["features"]["schema_version"]}
    with pytest.raises(ValueError, match="missing required columns"):
        load_paired_dataset(path, source_columns, config["contact_graph"], **kwargs)
    with pytest.raises(ValueError, match="available feature schema"):
        load_paired_dataset(path, model_columns, config["contact_graph"], **kwargs)
