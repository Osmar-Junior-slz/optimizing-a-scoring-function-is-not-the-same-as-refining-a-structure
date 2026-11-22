"""Fifteen fixed inputs are independent of algorithm identity and observed quality."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from docking_confidence.features.calibrator_descriptors import (
    FEATURE_COLUMNS, SCHEMA, SOURCE_CONTEXT_COLUMNS, model_features, transform_metadata,
)
from docking_confidence.features.contact_graph import (
    CONTACT_GRAPH_FEATURE_COLUMNS, CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS,
)
from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS


def _frame():
    return pd.DataFrame({column: [float(index + 1), float(index + 2)]
                         for index, column in enumerate(FEATURE_COLUMNS)}, index=[7, 2])


def test_registry_has_exactly_fifteen_fixed_descriptors_and_original_context():
    added = (
        "graph_receptor_residue_count", "graph_ligand_residue_count", "graph_degree_std",
        "graph_max_degree", "graph_connected_component_count",
    )
    expected = tuple(column for column in LOCAL_REFINEMENT_FEATURE_COLUMNS
                     if column != "graph_has_interface_contact") + added
    assert FEATURE_COLUMNS == expected and len(set(FEATURE_COLUMNS)) == 15
    assert SOURCE_CONTEXT_COLUMNS == LOCAL_REFINEMENT_FEATURE_COLUMNS
    assert len(SOURCE_CONTEXT_COLUMNS) == 11
    assert set(added).issubset(CONTACT_GRAPH_FEATURE_COLUMNS)
    assert not set(FEATURE_COLUMNS) & set(CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS)
    assert "graph_has_interface_contact" not in FEATURE_COLUMNS
    assert not any(column.startswith("algorithm") for column in FEATURE_COLUMNS)


def test_metadata_records_fixed_identity_transform_without_algorithm_predictor():
    metadata = transform_metadata()
    assert metadata["schema_version"] == SCHEMA == "gva_fixed_graph_descriptors_v1"
    assert metadata["source_context_columns"] == list(SOURCE_CONTEXT_COLUMNS)
    assert metadata["source_feature_columns"] == metadata["model_feature_columns"] == list(FEATURE_COLUMNS)
    assert metadata["algorithm_used_as_predictor"] is False
    assert metadata["missing_value_policy"] == "reject_without_imputation"
    assert metadata["transformation"] == "validated_identity"
    metadata["model_feature_columns"].clear()
    assert transform_metadata()["model_feature_columns"] == list(FEATURE_COLUMNS)


def test_features_preserve_row_order_and_use_registry_order_without_mutating_input():
    frame = _frame()
    frame = frame[list(reversed(FEATURE_COLUMNS))]
    original = deepcopy(frame)
    values, columns = model_features(frame)
    assert columns == list(FEATURE_COLUMNS) and values.shape == (2, 15)
    np.testing.assert_array_equal(values, original.loc[:, list(FEATURE_COLUMNS)].to_numpy())
    pd.testing.assert_frame_equal(frame, original)
    values[0, 0] = 999.
    pd.testing.assert_frame_equal(frame, original)


def test_algorithm_hashes_identifiers_and_observed_outcomes_are_ignored():
    frame = _frame()
    expected, _ = model_features(frame)
    frame["algorithm"] = [None, "unknown"]
    frame["algorithm_ag"] = [np.nan, np.inf]
    frame["complex_id"] = ["a", "b"]
    frame["complex_fingerprint_sha256"] = [None, "not a hash"]
    frame["dockq_full"] = [np.nan, np.inf]
    frame["dockq_reduced"] = [-20., 100.]
    frame["observed_margin"] = [None, "garbage"]
    frame["graph_has_interface_contact"] = [None, "ignored"]
    for column in CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS:
        frame[column] = np.nan
    actual, _ = model_features(frame)
    np.testing.assert_array_equal(actual, expected)


def test_zero_edge_graph_keeps_genuine_finite_descriptors_without_distance_quantiles():
    frame = _frame().iloc[:1].copy()
    frame.loc[:, ["graph_interface_edge_count", "graph_interface_density",
                  "graph_receptor_contact_fraction", "graph_ligand_contact_fraction",
                  "graph_clash_atom_pair_count", "graph_degree_std", "graph_max_degree"]] = 0.
    frame["graph_receptor_residue_count"] = 3.
    frame["graph_ligand_residue_count"] = 2.
    frame["graph_connected_component_count"] = 5.
    frame["graph_largest_component_fraction"] = .2
    frame["graph_min_cross_distance_angstrom"] = 20.
    values, columns = model_features(frame)
    assert np.isfinite(values).all() and values.shape == (1, 15)
    assert values[0, columns.index("graph_degree_std")] == 0.
    assert values[0, columns.index("graph_connected_component_count")] == 5.


@pytest.mark.parametrize("column", FEATURE_COLUMNS)
def test_every_fixed_descriptor_is_required_without_synthesizing_values(column):
    frame = _frame().drop(columns=[column])
    with pytest.raises(ValueError, match="missing fixed REDUCED descriptors"):
        model_features(frame)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, None, "bad"])
@pytest.mark.parametrize("column", [FEATURE_COLUMNS[0], "graph_receptor_residue_count", "graph_degree_std"])
def test_nonfinite_missing_and_nonnumeric_values_are_rejected(column, invalid):
    frame = _frame().astype(object)
    frame.loc[7, column] = invalid
    with pytest.raises(ValueError, match="finite numbers"):
        model_features(frame)


@pytest.mark.parametrize("column", [
    "uncertainty_count", "graph_interface_edge_count", "graph_clash_atom_pair_count",
    "graph_receptor_residue_count", "graph_ligand_residue_count",
    "graph_degree_std", "graph_max_degree", "graph_connected_component_count",
])
def test_negative_counts_and_degrees_are_rejected(column):
    frame = _frame()
    frame.loc[7, column] = -1.
    with pytest.raises(ValueError, match="nonnegative"):
        model_features(frame)


@pytest.mark.parametrize("column", ["graph_receptor_residue_count", "graph_ligand_residue_count"])
def test_receptor_and_ligand_sizes_must_be_positive(column):
    frame = _frame()
    frame.loc[7, column] = 0.
    with pytest.raises(ValueError, match="positive"):
        model_features(frame)


@pytest.mark.parametrize("frame", [None, np.ones((2, 15)), pd.DataFrame(), _frame().iloc[:0]])
def test_missing_nondataframe_and_empty_examples_are_rejected(frame):
    with pytest.raises(ValueError, match="nonempty DataFrame"):
        model_features(frame)


def test_duplicate_feature_columns_are_rejected():
    frame = _frame()
    frame = pd.concat([frame, frame[[FEATURE_COLUMNS[0]]]], axis=1)
    with pytest.raises(ValueError, match="unique column names"):
        model_features(frame)
