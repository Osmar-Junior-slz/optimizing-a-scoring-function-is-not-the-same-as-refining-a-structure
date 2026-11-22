from __future__ import annotations

import json
from pathlib import Path
import re
from typing import IO

import numpy as np
import pandas as pd

from docking_confidence.domain.structure import (
    CANONICAL_COMPLEX_FINGERPRINT_SCHEMA,
    STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA,
    CanonicalComplexFingerprint,
)
from docking_confidence.features.contact_graph import (
    CONTACT_GRAPH_METADATA_COLUMNS,
    CONTACT_GRAPH_NODE_UNIVERSE,
    CONTACT_GRAPH_SCHEMA_VERSION,
    ContactGraphConfig,
)


TRAINING_BASE_COLUMNS = [
    "complex_id",
    "run_id",
    "algorithm",
    "seed",
    "receptor_sequence",
    "ligand_sequence",
    "engine_reduced",
    "engine_full",
    "scientific_reduced",
    "scientific_full",
    "dockq_reduced",
    "dockq_full",
    "runtime_reduced_sec",
    "runtime_full_sec",
]

INFERENCE_BASE_COLUMNS = [
    "complex_id",
    "run_id",
    "algorithm",
    "seed",
    "receptor_sequence",
    "ligand_sequence",
    "engine_reduced",
    "scientific_reduced",
]

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_GRAPH_FIXED_STRINGS = {
    "graph_schema_version": CONTACT_GRAPH_SCHEMA_VERSION,
    "graph_coordinate_frame": "shared_complex_pose_v1",
    "graph_node_universe": CONTACT_GRAPH_NODE_UNIVERSE,
    "graph_pose_protocol": "reduced",
    "graph_complex_fingerprint_schema": CANONICAL_COMPLEX_FINGERPRINT_SCHEMA,
    "graph_structure_sequence_fingerprint_schema": (
        STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA
    ),
}


def _read_csv(source: str | Path | IO[bytes] | IO[str]) -> pd.DataFrame:
    # String-first parsing preserves identifiers such as "001". Conversions are
    # explicit below, so pandas cannot silently reinterpret scientific keys.
    return pd.read_csv(source, dtype=str, keep_default_na=True)


def _require_columns(df: pd.DataFrame, required: list[str] | tuple[str, ...]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _normalize_identifier(df: pd.DataFrame, column: str) -> None:
    if df[column].isna().any():
        raise ValueError(f"column {column} contains missing identifiers")
    normalized = df[column].astype(str).str.strip()
    if normalized.eq("").any():
        raise ValueError(f"column {column} contains blank identifiers")
    df[column] = normalized


def _coerce_finite_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    for column in dict.fromkeys(columns):
        df[column] = pd.to_numeric(df[column], errors="raise")
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"column {column} must contain only finite values")


def _coerce_integer(df: pd.DataFrame, column: str, *, minimum: int | None = None) -> None:
    _coerce_finite_numeric(df, [column])
    values = df[column].to_numpy(dtype=float)
    if not np.equal(values % 1, 0).all():
        raise ValueError(f"column {column} must contain integers")
    if minimum is not None and (values < minimum).any():
        raise ValueError(f"column {column} must contain integers >= {minimum}")
    df[column] = values.astype(np.int64)


def _parse_boolean(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "sim": True,
        "false": False,
        "0": False,
        "no": False,
        "nao": False,
        "não": False,
    }
    parsed = normalized.map(mapping)
    if series.isna().any() or parsed.isna().any():
        raise ValueError(f"{column} must contain explicit boolean values")
    return parsed.astype(bool)


def _normalize_hash_column(df: pd.DataFrame, column: str) -> None:
    _normalize_identifier(df, column)
    normalized = df[column].str.lower()
    if not normalized.str.fullmatch(_SHA256_RE).all():
        raise ValueError(f"{column} must contain 64 lowercase hexadecimal characters")
    df[column] = normalized


def _add_complex_fingerprints(df: pd.DataFrame) -> None:
    _require_columns(df, ["receptor_sequence", "ligand_sequence"])
    expected_hashes = (
        df["complex_fingerprint_sha256"].tolist()
        if "complex_fingerprint_sha256" in df.columns
        else [None] * len(df)
    )
    expected_schemas = (
        df["complex_fingerprint_schema"].tolist()
        if "complex_fingerprint_schema" in df.columns
        else [CANONICAL_COMPLEX_FINGERPRINT_SCHEMA] * len(df)
    )
    fingerprints: list[CanonicalComplexFingerprint] = []
    for receptor, ligand, expected_hash, expected_schema in zip(
        df["receptor_sequence"],
        df["ligand_sequence"],
        expected_hashes,
        expected_schemas,
    ):
        supplied_hash = None if pd.isna(expected_hash) else expected_hash
        schema = (
            CANONICAL_COMPLEX_FINGERPRINT_SCHEMA
            if pd.isna(expected_schema)
            else expected_schema
        )
        fingerprints.append(
            CanonicalComplexFingerprint.from_sequences(
                receptor,
                ligand,
                expected_sha256=supplied_hash,
                schema=schema,
            )
        )
    df["receptor_sequence"] = [item.receptor_sequence for item in fingerprints]
    df["ligand_sequence"] = [item.ligand_sequence for item in fingerprints]
    df["complex_fingerprint_schema"] = [item.schema for item in fingerprints]
    df["complex_fingerprint_sha256"] = [item.sha256 for item in fingerprints]

    by_id = df.groupby("complex_id", sort=False)["complex_fingerprint_sha256"].nunique()
    if (by_id != 1).any():
        raise ValueError("one complex_id maps to multiple canonical sequence fingerprints")
    by_fingerprint = df.groupby("complex_fingerprint_sha256", sort=False)["complex_id"].nunique()
    if (by_fingerprint != 1).any():
        raise ValueError(
            "one canonical sequence fingerprint maps to multiple complex_id aliases"
        )


def _parse_chain_json(value: object, *, column: str) -> tuple[str, ...]:
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{column} must be a JSON list of PDB chain IDs") from exc
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or len(item) != 1 for item in parsed)
        or len(parsed) != len(set(parsed))
        or parsed != sorted(parsed)
    ):
        raise ValueError(f"{column} must be a sorted non-empty JSON list of unique chain IDs")
    return tuple(parsed)


def _validate_graph_metadata(df: pd.DataFrame, cfg: dict | None) -> None:
    if not cfg or cfg.get("enabled") is not True:
        return
    if cfg.get("require_pose_provenance") is not True:
        raise ValueError("enabled contact graph requires pose provenance")
    graph_config = ContactGraphConfig.from_mapping(cfg)
    _require_columns(df, CONTACT_GRAPH_METADATA_COLUMNS)

    expected_strings = {
        **_GRAPH_FIXED_STRINGS,
        "graph_parser_schema_version": graph_config.parser_schema_version,
        "graph_altloc_policy": graph_config.altloc_policy,
    }
    for column, expected in expected_strings.items():
        _normalize_identifier(df, column)
        if not df[column].eq(expected).all():
            raise ValueError(f"{column} does not match the graph-v2 contract")

    _normalize_identifier(df, "graph_config_sha256")
    if not df["graph_config_sha256"].str.lower().eq(graph_config.digest).all():
        raise ValueError("graph descriptor configuration hash does not match configuration")
    df["graph_config_sha256"] = df["graph_config_sha256"].str.lower()

    _coerce_finite_numeric(
        df, ["graph_cutoff_angstrom", "graph_clash_cutoff_angstrom"]
    )
    for column, expected in (
        ("graph_cutoff_angstrom", graph_config.cutoff_angstrom),
        ("graph_clash_cutoff_angstrom", graph_config.clash_cutoff_angstrom),
    ):
        if not np.allclose(
            df[column].to_numpy(dtype=float), float(expected), rtol=0.0, atol=1e-12
        ):
            raise ValueError(f"{column} does not match graph configuration")

    _coerce_integer(df, "graph_model_id", minimum=1)
    if not df["graph_model_id"].eq(graph_config.model_id).all():
        raise ValueError("graph_model_id does not match graph configuration")
    for column, expected in (
        ("graph_include_hetero", graph_config.include_hetero),
        ("graph_heavy_atoms_only", graph_config.heavy_atoms_only),
        ("graph_exclude_waters", graph_config.exclude_waters),
    ):
        df[column] = _parse_boolean(df[column], column)
        if not df[column].eq(expected).all():
            raise ValueError(f"{column} does not match graph configuration")

    for column in (
        "graph_pose_source",
        "graph_pose_id",
        "graph_pose_selection_method",
        "graph_bound_complex_id",
        "graph_bound_run_id",
        "graph_bound_algorithm",
        "graph_bound_engine",
    ):
        _normalize_identifier(df, column)
    if not df["graph_pose_source"].isin(
        {"single_complex_pdb", "docking_engine", "explicit_transform"}
    ).all():
        raise ValueError("graph_pose_source contains an unsupported source")

    _coerce_integer(df, "graph_pose_rank", minimum=1)
    _coerce_integer(df, "graph_bound_seed", minimum=0)
    for graph_column, row_column in (
        ("graph_bound_complex_id", "complex_id"),
        ("graph_bound_run_id", "run_id"),
        ("graph_bound_algorithm", "algorithm"),
        ("graph_bound_seed", "seed"),
        ("graph_bound_engine", "engine_reduced"),
        ("graph_complex_fingerprint_schema", "complex_fingerprint_schema"),
        ("graph_complex_fingerprint_sha256", "complex_fingerprint_sha256"),
    ):
        if not df[graph_column].astype(str).eq(df[row_column].astype(str)).all():
            raise ValueError(f"{graph_column} is not bound to the same dataset row")

    for column in (
        "graph_source_file_sha256",
        "graph_pose_canonical_sha256",
        "graph_complex_fingerprint_sha256",
        "graph_structure_sequence_fingerprint_sha256",
    ):
        _normalize_hash_column(df, column)

    receptor_chains = [
        _parse_chain_json(value, column="graph_receptor_chains")
        for value in df["graph_receptor_chains"]
    ]
    ligand_chains = [
        _parse_chain_json(value, column="graph_ligand_chains")
        for value in df["graph_ligand_chains"]
    ]
    if any(set(left) & set(right) for left, right in zip(receptor_chains, ligand_chains)):
        raise ValueError("graph receptor and ligand chains must be disjoint")

    if "graph_transform_sha256" in df.columns:
        for source, value in zip(df["graph_pose_source"], df["graph_transform_sha256"]):
            missing = pd.isna(value) or not str(value).strip()
            if source == "explicit_transform" and missing:
                raise ValueError("explicit_transform graph poses require graph_transform_sha256")
            if not missing and _SHA256_RE.fullmatch(str(value).strip().lower()) is None:
                raise ValueError("graph_transform_sha256 must be a SHA-256 value")


def _validate_common_identity(df: pd.DataFrame) -> None:
    for column in ("complex_id", "run_id", "algorithm", "engine_reduced"):
        _normalize_identifier(df, column)
    _coerce_integer(df, "seed", minimum=0)
    if df.duplicated(["complex_id", "run_id"]).any():
        raise ValueError("duplicate complex_id/run_id keys detected")
    _add_complex_fingerprints(df)


def _validate_training_context(df, feature_columns, docking_task=None, feature_schema=None):
    from docking_confidence.io.config import (
        DOCKING_TASK_FEATURE_SCHEMAS, FEATURE_SCHEMA_VERSION,
        LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION, compatible_feature_schema, local_feature_columns,
    )
    columns = ("docking_task", "feature_schema", "config", "config_sha256")
    declared = any(column in df for column in columns)
    if not declared and docking_task is None and feature_schema in {None, FEATURE_SCHEMA_VERSION}:
        return
    _require_columns(df, columns)
    for column in columns:
        _normalize_identifier(df, column)
        if df[column].nunique() != 1:
            raise ValueError(f"mixed training context in {column}")
    task = df["docking_task"].iloc[0]
    schema = df["feature_schema"].iloc[0]
    if task not in DOCKING_TASK_FEATURE_SCHEMAS or not compatible_feature_schema(task, schema):
        raise ValueError("docking_task and feature_schema are incompatible")
    if docking_task is not None and task != docking_task:
        raise ValueError("dataset docking_task disagrees with configuration")
    if feature_schema is not None and schema != feature_schema:
        raise ValueError("dataset feature_schema disagrees with configuration")
    if task == "local_refinement" and tuple(feature_columns) != local_feature_columns(schema):
        raise ValueError("local_refinement requires its available feature schema")
    if schema == LOCAL_REFINEMENT_CONTEXT_SCHEMA_VERSION:
        from docking_confidence.features.execution_context import (
            ALGORITHM_FEATURE_COLUMNS, EXECUTION_CONTEXT_METADATA_COLUMNS, INPUT_POSE_REFERENCE_ROLE,
        )
        _require_columns(df, EXECUTION_CONTEXT_METADATA_COLUMNS)
        if not df["algorithm"].isin(["ag", "pso", "hybrid"]).all():
            raise ValueError("execution context requires a supported REDUCED algorithm")
        _coerce_finite_numeric(df, ["input_pose_retention", *ALGORITHM_FEATURE_COLUMNS])
        if not df["input_pose_retention"].between(0, 1).all():
            raise ValueError("input_pose_retention must be in [0, 1]")
        if not df["input_pose_retention_reference_role"].eq(INPUT_POSE_REFERENCE_ROLE).all():
            raise ValueError("input_pose_retention must not use the experimental reference")
        for column in EXECUTION_CONTEXT_METADATA_COLUMNS[1:]:
            _normalize_hash_column(df, column)
        for column in ALGORITHM_FEATURE_COLUMNS:
            expected = (df["algorithm"] == column.removeprefix("algorithm_")).astype(int)
            if not df[column].eq(expected).all():
                raise ValueError("algorithm context disagrees with its REDUCED run")
    _normalize_hash_column(df, "config_sha256")


def _validated_context_columns(feature_columns, context_feature_columns):
    if context_feature_columns is None:
        return feature_columns
    for columns in (feature_columns, context_feature_columns):
        if (not isinstance(columns, (list, tuple)) or not columns or
            any(not isinstance(name, str) or not name or name != name.strip() for name in columns) or
            len(set(columns)) != len(columns)):
            raise ValueError("model and context feature columns must be unique nonempty names")
    if not set(feature_columns).issubset(context_feature_columns):
        raise ValueError("model features must belong to the original context feature contract")
    return context_feature_columns


def load_paired_dataset(
    path: str | Path | IO[bytes] | IO[str],
    feature_columns: list[str],
    contact_graph_cfg: dict | None = None,
    *, docking_task: str | None = None, feature_schema: str | None = None,
    context_feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load paired outcomes and model inputs with an optional source context."""
    context_columns = _validated_context_columns(feature_columns, context_feature_columns)
    df = _read_csv(path)
    _require_columns(df, TRAINING_BASE_COLUMNS + list(feature_columns))
    if df.empty:
        raise ValueError("dataset is empty")
    _validate_training_context(df, context_columns, docking_task, feature_schema)
    _validate_common_identity(df)
    _normalize_identifier(df, "engine_full")
    for column in ("scientific_reduced", "scientific_full"):
        df[column] = _parse_boolean(df[column], column)
    df["scientific"] = df["scientific_reduced"] & df["scientific_full"]
    df["engine"] = [
        reduced if reduced == full else f"REDUCED={reduced};FULL={full}"
        for reduced, full in zip(df["engine_reduced"], df["engine_full"])
    ]
    _coerce_finite_numeric(
        df,
        [
            "dockq_reduced",
            "dockq_full",
            "runtime_reduced_sec",
            "runtime_full_sec",
            *feature_columns,
        ],
    )
    if ((df["dockq_reduced"] < 0) | (df["dockq_reduced"] > 1)).any():
        raise ValueError("dockq_reduced must be in [0, 1]")
    if ((df["dockq_full"] < 0) | (df["dockq_full"] > 1)).any():
        raise ValueError("dockq_full must be in [0, 1]")
    if (df[["runtime_reduced_sec", "runtime_full_sec"]] <= 0).any().any():
        raise ValueError("runtime values must be positive")
    _validate_graph_metadata(df, contact_graph_cfg)
    return df


def load_reduced_inference_dataset(
    path: str | Path | IO[bytes] | IO[str],
    feature_columns: list[str],
    contact_graph_cfg: dict | None = None,
    *, docking_task: str | None = None, feature_schema: str | None = None,
    context_feature_columns: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load model inputs while retaining the original acquisition context.

    An explicitly supplied context contract allows a calibrated model to use a
    strict subset of the acquired descriptors. Only its selected inputs are
    required and numerically parsed; the source schema and provenance retain
    their original validation. Existing callers keep the full-schema behavior.
    """
    context_columns = _validated_context_columns(feature_columns, context_feature_columns)
    df = _read_csv(path)
    _require_columns(df, INFERENCE_BASE_COLUMNS + list(feature_columns))
    if df.empty:
        raise ValueError("inference dataset is empty")
    _validate_training_context(df, context_columns, docking_task, feature_schema)
    _validate_common_identity(df)
    df["scientific_reduced"] = _parse_boolean(
        df["scientific_reduced"], "scientific_reduced"
    )
    _coerce_finite_numeric(df, list(feature_columns))
    _validate_graph_metadata(df, contact_graph_cfg)
    return df


def assess_dataset_provenance(df: pd.DataFrame, cfg: dict) -> dict:
    require_scientific = cfg.get("require_scientific_provenance") is True
    allowed = cfg.get("trusted_engine_allowlist", [])
    if not isinstance(allowed, list) or any(
        not isinstance(value, str) or not value.strip() for value in allowed
    ):
        raise ValueError("trusted_engine_allowlist must be a list of engine identifiers")
    allowlist = {value.strip() for value in allowed}
    scientific = df["scientific_reduced"] & df["scientific_full"]
    engine_allowed = df["engine_reduced"].isin(allowlist) & df["engine_full"].isin(
        allowlist
    )
    trusted = scientific & engine_allowed
    passed = bool(trusted.all()) if require_scientific else True
    if passed:
        status = "scientific_attested_and_engines_allowlisted"
    elif not bool(scientific.all()):
        status = "contains_non_scientific_rows"
    else:
        status = "engine_not_allowlisted"
    engines_reduced = sorted(df["engine_reduced"].unique().tolist())
    engines_full = sorted(df["engine_full"].unique().tolist())
    return {
        "status": status,
        "provenance_kind": "operator_attestation_not_cryptographic_authentication",
        "require_scientific_provenance": require_scientific,
        "trusted_engine_allowlist": sorted(allowlist),
        "passed": passed,
        "engines": sorted(set(engines_reduced) | set(engines_full)),
        "engines_reduced": engines_reduced,
        "engines_full": engines_full,
        "scientific_row_count": int(scientific.sum()),
        "allowlisted_engine_row_count": int(engine_allowed.sum()),
        "trusted_scientific_row_count": int(trusted.sum()),
        "total_row_count": int(len(df)),
    }
