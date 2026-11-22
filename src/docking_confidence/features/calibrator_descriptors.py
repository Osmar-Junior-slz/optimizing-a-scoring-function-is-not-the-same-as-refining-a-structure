"""Fixed fifteen REDUCED descriptors; no automatic selection or imputation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS


SCHEMA = "gva_fixed_graph_descriptors_v1"
SOURCE_CONTEXT_COLUMNS = LOCAL_REFINEMENT_FEATURE_COLUMNS
FEATURE_COLUMNS = (
    *(column for column in SOURCE_CONTEXT_COLUMNS if column != "graph_has_interface_contact"),
    "graph_receptor_residue_count",
    "graph_ligand_residue_count",
    "graph_degree_std",
    "graph_max_degree",
    "graph_connected_component_count",
)
_NONNEGATIVE_COLUMNS = (
    "uncertainty_count", "graph_interface_edge_count", "graph_clash_atom_pair_count",
    "graph_receptor_residue_count", "graph_ligand_residue_count",
    "graph_degree_std", "graph_max_degree", "graph_connected_component_count",
)
_POSITIVE_COLUMNS = ("graph_receptor_residue_count", "graph_ligand_residue_count")


def transform_metadata() -> dict:
    """Describe the validated identity transform independently of input outcomes."""
    return {
        "schema_version": SCHEMA,
        "source_context_columns": list(SOURCE_CONTEXT_COLUMNS),
        "source_feature_columns": list(FEATURE_COLUMNS),
        "model_feature_columns": list(FEATURE_COLUMNS),
        "algorithm_used_as_predictor": False,
        "transformation": "validated_identity",
        "missing_value_policy": "reject_without_imputation",
    }


def model_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Read exactly the fixed descriptors in order, preserving all input rows."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("model features require a nonempty DataFrame")
    if not frame.columns.is_unique:
        raise ValueError("feature frame must contain unique column names")
    missing = [column for column in FEATURE_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"missing fixed REDUCED descriptors: {missing}")
    try:
        values = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("fixed REDUCED descriptors must contain finite numbers") from exc
    if not np.isfinite(values).all():
        raise ValueError("fixed REDUCED descriptors must contain finite numbers; missing values are not imputed")
    for column in _NONNEGATIVE_COLUMNS:
        if (values[:, FEATURE_COLUMNS.index(column)] < 0).any():
            raise ValueError(f"{column} must be nonnegative")
    for column in _POSITIVE_COLUMNS:
        if (values[:, FEATURE_COLUMNS.index(column)] <= 0).any():
            raise ValueError(f"{column} must be positive")
    return values, list(FEATURE_COLUMNS)
