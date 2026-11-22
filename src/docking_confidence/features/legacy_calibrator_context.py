"""Historical V2 transform, used only to replay saved 14-input GVA artifacts.

This transform reads only the eleven original local-refinement descriptors and
the saved REDUCED algorithm. Experimental quality, FULL outcomes, identifiers and
previous model predictions never become explanatory variables here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from docking_confidence.features.execution_context import ALGORITHM_FEATURE_COLUMNS
from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS


SCHEMA = "gva_algorithm_context_v1"
ALGORITHMS = ("ag", "pso", "hybrid")


def _source_columns(source_feature_columns) -> list[str]:
    if not isinstance(source_feature_columns, (list, tuple)):
        raise ValueError("source_feature_columns must list the original REDUCED descriptors")
    columns = list(source_feature_columns)
    if any(not isinstance(column, str) for column in columns):
        raise ValueError("source_feature_columns must contain descriptor names")
    if len(columns) != len(set(columns)):
        raise ValueError("source_feature_columns must not contain duplicates")
    if set(columns) != set(LOCAL_REFINEMENT_FEATURE_COLUMNS):
        raise ValueError(
            "gva_algorithm_context_v1 requires exactly the eleven original local-refinement descriptors; "
            "outcomes, identifiers and precomputed algorithm columns are not source features"
        )
    return columns


def transform_metadata(source_feature_columns) -> dict:
    """Return a JSON-ready, ordered and replayable feature-transform contract."""
    source = _source_columns(source_feature_columns)
    return {
        "schema_version": SCHEMA,
        "source_feature_columns": source,
        "model_feature_columns": [*source, *ALGORITHM_FEATURE_COLUMNS],
        "algorithm_column": "algorithm",
        "algorithm_categories": list(ALGORITHMS),
        "precomputed_onehot_policy": "validate_then_derive",
    }


def model_features(frame: pd.DataFrame, source_feature_columns) -> tuple[np.ndarray, list[str]]:
    """Return a new finite matrix, preserving row order and the source column order.

    Algorithm indicators are always derived from ``algorithm``. If a source also
    contains cached indicator columns, verify their agreement first; none of their
    values is copied into the returned matrix. The input frame is never mutated.
    """
    metadata = transform_metadata(source_feature_columns)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("model features require a nonempty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("feature frame must not contain duplicate columns")
    required = ["algorithm", *metadata["source_feature_columns"]]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"missing model feature inputs: {missing}")
    algorithms = frame["algorithm"]
    if algorithms.isna().any() or not algorithms.isin(ALGORITHMS).all():
        raise ValueError("model features require algorithm ag, pso or hybrid")
    try:
        original = frame.loc[:, metadata["source_feature_columns"]].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("source descriptors must contain finite numbers") from exc
    if not np.isfinite(original).all():
        raise ValueError("source descriptors must contain finite numbers")
    derived = np.column_stack([(algorithms == algorithm).to_numpy(dtype=float) for algorithm in ALGORITHMS])
    for index, column in enumerate(ALGORITHM_FEATURE_COLUMNS):
        if column not in frame:
            continue
        try:
            cached = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"precomputed {column} disagrees with the saved algorithm") from exc
        if not np.isfinite(cached).all() or not np.array_equal(cached, derived[:, index]):
            raise ValueError(f"precomputed {column} disagrees with the saved algorithm")
    return np.column_stack((original, derived)), list(metadata["model_feature_columns"])
