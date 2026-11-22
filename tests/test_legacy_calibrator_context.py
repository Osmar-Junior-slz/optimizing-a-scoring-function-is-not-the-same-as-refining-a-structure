"""Model input context is deterministic, outcome-free and replayable."""
import json

import numpy as np
import pandas as pd
import pytest

from docking_confidence.features.legacy_calibrator_context import (
    SCHEMA, model_features, transform_metadata,
)
from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS


SOURCE = list(LOCAL_REFINEMENT_FEATURE_COLUMNS)


def _frame():
    frame = pd.DataFrame({column: np.arange(5, dtype=float) + index / 10
                          for index, column in enumerate(SOURCE)})
    frame["algorithm"] = ["hybrid", "ag", "pso", "ag", "hybrid"]
    frame.index = ["last", "first", "middle", "another", "last"]
    frame["dockq_full"] = [.2, .4, .6, .8, 1.]
    frame["dockq_reduced"] = [.1, .4, .5, .7, .9]
    frame["complex_id"] = ["z", "a", "c", "b", "y"]
    return frame


def test_context_retains_rows_source_order_and_derives_algorithm_identity_without_mutation():
    frame = _frame()
    original = frame.copy(deep=True)
    X, columns = model_features(frame, SOURCE)
    assert X.shape == (5, 14)
    assert columns == SOURCE + ["algorithm_ag", "algorithm_pso", "algorithm_hybrid"]
    np.testing.assert_array_equal(X[:, :11], frame[SOURCE].to_numpy(float))
    np.testing.assert_array_equal(X[:, 11:], [[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1]])
    assert np.isfinite(X).all()
    pd.testing.assert_frame_equal(frame, original)
    X[0, 0] = -999.
    pd.testing.assert_frame_equal(frame, original)


def test_metadata_is_json_serializable_and_preserves_requested_descriptor_order():
    source = list(reversed(SOURCE))
    metadata = transform_metadata(source)
    assert metadata["schema_version"] == SCHEMA == "gva_algorithm_context_v1"
    assert json.loads(json.dumps(metadata)) == metadata
    assert metadata["source_feature_columns"] == source
    assert metadata["model_feature_columns"] == source + ["algorithm_ag", "algorithm_pso", "algorithm_hybrid"]
    source.reverse()
    assert metadata["source_feature_columns"] != source
    X, columns = model_features(_frame(), metadata["source_feature_columns"])
    np.testing.assert_array_equal(X[:, 0], _frame()[metadata["source_feature_columns"][0]])
    assert columns == metadata["model_feature_columns"]


@pytest.mark.parametrize("column", ["dockq_full", "dockq_reduced", "margin", "outcome", "label", "complex_id",
                                    "algorithm", "algorithm_ag", "prediction", "probability"])
def test_outcomes_identifiers_and_cached_indicators_cannot_be_requested_as_descriptors(column):
    source = [*SOURCE[:-1], column]
    with pytest.raises(ValueError, match="eleven original"):
        model_features(_frame(), source)


@pytest.mark.parametrize("source", [[], SOURCE[:-1], [*SOURCE, SOURCE[0]], "viable_fraction_mean", [*SOURCE[:-1], 1]])
def test_invalid_or_duplicate_source_names_are_rejected(source):
    with pytest.raises(ValueError):
        transform_metadata(source)


@pytest.mark.parametrize("algorithm", ["unknown", "AG", " ag", None, np.nan, 1])
def test_missing_or_unsupported_algorithm_is_rejected(algorithm):
    frame = _frame()
    frame["algorithm"] = frame["algorithm"].astype(object)
    frame.iloc[2, frame.columns.get_loc("algorithm")] = algorithm
    with pytest.raises(ValueError, match="algorithm ag, pso or hybrid"):
        model_features(frame, SOURCE)


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan, "invalid"])
def test_nonfinite_or_nonnumeric_descriptors_are_rejected(value):
    frame = _frame()
    frame[SOURCE[0]] = frame[SOURCE[0]].astype(object)
    frame.iloc[1, frame.columns.get_loc(SOURCE[0])] = value
    with pytest.raises(ValueError, match="finite numbers"):
        model_features(frame, SOURCE)


@pytest.mark.parametrize("cached", [2, .5, -1, np.nan, np.inf, "wrong"])
def test_forged_or_invalid_cached_onehot_values_are_rejected(cached):
    frame = _frame()
    frame["algorithm_ag"] = [0, 1, 0, 1, 0]
    frame["algorithm_ag"] = frame["algorithm_ag"].astype(object)
    frame.iloc[0, frame.columns.get_loc("algorithm_ag")] = cached
    with pytest.raises(ValueError, match="precomputed algorithm_ag disagrees"):
        model_features(frame, SOURCE)


def test_correct_cached_onehots_are_verified_but_matrix_is_derived_and_independent():
    frame = _frame()
    frame["algorithm_ag"] = ["0", "1", "0", "1", "0"]
    frame["algorithm_pso"] = [0., 0., 1., 0., 0.]
    frame["algorithm_hybrid"] = [1, 0, 0, 0, 1]
    original = frame.copy(deep=True)
    X, _ = model_features(frame, SOURCE)
    assert X.dtype == np.float64
    np.testing.assert_array_equal(X[:, 11:].sum(axis=1), np.ones(len(frame)))
    pd.testing.assert_frame_equal(frame, original)
    frame.iloc[0, frame.columns.get_loc("algorithm_ag")] = "1"
    assert X[0, 11] == 0.


def test_outcome_reference_and_identifier_changes_have_no_effect_on_model_features():
    frame = _frame()
    expected, names = model_features(frame, SOURCE)
    altered = frame.copy()
    altered["dockq_full"] = np.nan
    altered["dockq_reduced"] = "deliberately invalid: never read"
    altered["complex_id"] = "different"
    altered["reference_path"] = "unavailable.pdb"
    altered["outcome"] = -999.
    altered["probability"] = 1.
    actual, actual_names = model_features(altered, SOURCE)
    assert actual_names == names
    np.testing.assert_array_equal(actual, expected)


def test_missing_required_columns_duplicate_columns_and_empty_frame_are_rejected():
    for frame in [_frame().drop(columns=["algorithm"]), _frame().drop(columns=[SOURCE[0]]),
                  _frame().iloc[:0], pd.concat([_frame(), _frame()[[SOURCE[0]]]], axis=1)]:
        with pytest.raises(ValueError):
            model_features(frame, SOURCE)
