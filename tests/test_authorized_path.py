from copy import deepcopy
import hashlib
from pathlib import Path

import pandas as pd

from docking_confidence.application.demo import make_demo
from docking_confidence.application.predict import predict
from docking_confidence.application.train import train
from docking_confidence.domain.structure import (
    STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA,
    CanonicalComplexFingerprint,
)
from docking_confidence.features.contact_graph import ContactGraphConfig
from docking_confidence.io.config import load_config, validate_config


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _graph_metadata(row: pd.Series, graph_cfg: dict) -> dict:
    fingerprint = CanonicalComplexFingerprint.from_sequences(
        row["receptor_sequence"], row["ligand_sequence"]
    )
    graph = ContactGraphConfig.from_mapping(graph_cfg)
    identity = f"{row['complex_id']}:{row['run_id']}"
    return {
        "complex_fingerprint_schema": fingerprint.schema,
        "complex_fingerprint_sha256": fingerprint.sha256,
        "graph_schema_version": graph.schema_version,
        "graph_config_sha256": graph.digest,
        "graph_parser_schema_version": graph.parser_schema_version,
        "graph_altloc_policy": graph.altloc_policy,
        "graph_node_universe": graph.node_universe,
        "graph_cutoff_angstrom": graph.cutoff_angstrom,
        "graph_clash_cutoff_angstrom": graph.clash_cutoff_angstrom,
        "graph_include_hetero": graph.include_hetero,
        "graph_heavy_atoms_only": graph.heavy_atoms_only,
        "graph_exclude_waters": graph.exclude_waters,
        "graph_model_id": graph.model_id,
        "graph_coordinate_frame": "shared_complex_pose_v1",
        "graph_pose_source": "docking_engine",
        "graph_pose_id": identity,
        "graph_pose_rank": 1,
        "graph_pose_selection_method": "trusted_test_fixture",
        "graph_source_file_sha256": _digest(f"source:{identity}"),
        "graph_pose_canonical_sha256": _digest(f"pose:{identity}"),
        "graph_receptor_chains": '["A"]',
        "graph_ligand_chains": '["P"]',
        "graph_bound_complex_id": row["complex_id"],
        "graph_bound_run_id": row["run_id"],
        "graph_bound_algorithm": row["algorithm"],
        "graph_bound_seed": int(row["seed"]),
        "graph_bound_engine": row["engine_reduced"],
        "graph_pose_protocol": "reduced",
        "graph_complex_fingerprint_schema": fingerprint.schema,
        "graph_complex_fingerprint_sha256": fingerprint.sha256,
        "graph_structure_sequence_fingerprint_schema": (
            STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA
        ),
        "graph_structure_sequence_fingerprint_sha256": _digest(
            f"structure:{identity}"
        ),
    }


def _trusted_config(root: Path) -> dict:
    config = deepcopy(load_config(root / "configs" / "default.yaml"))
    config["authorization"].update(
        {
            "allow_automatic_decision": True,
            "trusted_engine_allowlist": ["validated-test-engine"],
            "protocol_commitment_sha256": _digest("predeclared-test-protocol"),
        }
    )
    config["calibration"].update(
        {
            "minimum_independent_complexes": 4,
            "minimum_positive_class_complexes": 2,
            "minimum_negative_class_complexes": 2,
        }
    )
    config["policy"].update(
        {
            # This fixture deliberately exercises historical exact-binomial
            # authorization with perfectly separated, zero-error predictions.
            "method": "learn_then_test_bonferroni_binomial_iid",
            "risk_target": 0.9,
            "familywise_confidence_level": 0.8,
            "threshold_min": 0.0,
            "threshold_max": 0.8,
            "threshold_step": 0.4,
            "minimum_complex_coverage": 0.0,
            "minimum_evaluated_complexes": 4,
            "minimum_authorized_complexes": 4,
        }
    )
    return validate_config(config)


def _make_trusted_dataset(
    path: Path, config: dict, *, n_complexes: int = 25,
) -> pd.DataFrame:
    raw = path.with_name("raw_demo.csv")
    make_demo(str(raw), seed=7, n_complexes=n_complexes, runs_per_complex=2)
    frame = pd.read_csv(raw)
    for index in range(len(frame)):
        positive = index % 2 == 0
        frame.loc[index, "run_id"] = "POS" if positive else "NEG"
        frame.loc[index, "seed"] = 1 if positive else 2
        frame.loc[index, "engine_reduced"] = "validated-test-engine"
        frame.loc[index, "engine_full"] = "validated-test-engine"
        frame.loc[index, "scientific_reduced"] = True
        frame.loc[index, "scientific_full"] = True
        frame.loc[index, "scientific"] = True
        frame.loc[index, "engine"] = "validated-test-engine"
        frame.loc[index, "dockq_full"] = 0.8
        frame.loc[index, "dockq_reduced"] = 0.8 if positive else 0.1
        frame.loc[index, "runtime_reduced_sec"] = 50.0
        frame.loc[index, "runtime_full_sec"] = 100.0
        sign = 1.0 if positive else -1.0
        frame.loc[index, "scan_gap_norm"] = 3.0 * sign
        frame.loc[index, "viable_fraction_mean"] = 0.9 if positive else 0.1
        frame.loc[index, "elite_dispersion"] = 0.1 if positive else 1.0
        frame.loc[index, "uncertainty_count"] = 0.0 if positive else 8.0
        frame.loc[index, "energy_q20_norm"] = -6.0 if positive else -2.0
        frame.loc[index, "graph_has_interface_contact"] = 1.0 if positive else 0.0
        frame.loc[index, "graph_min_cross_distance_angstrom"] = 2.5 if positive else 9.0
        frame.loc[index, "graph_interface_edge_count"] = 8.0 if positive else 0.0
        frame.loc[index, "graph_interface_density"] = 0.5 if positive else 0.0
        frame.loc[index, "graph_receptor_contact_fraction"] = 0.8 if positive else 0.0
        frame.loc[index, "graph_ligand_contact_fraction"] = 1.0 if positive else 0.0
        frame.loc[index, "graph_largest_component_fraction"] = 0.7 if positive else 0.1
        frame.loc[index, "graph_clash_atom_pair_count"] = 0.0
        for key, value in _graph_metadata(frame.loc[index], config["contact_graph"]).items():
            frame.loc[index, key] = value
    frame.to_csv(path, index=False)
    return frame


def test_authorized_artifact_reaches_accept_reduced_on_trusted_unseen_complex(
    tmp_path: Path,
):
    root = Path(__file__).parents[1]
    config = _trusted_config(root)
    dataset = tmp_path / "trusted.csv"
    frame = _make_trusted_dataset(dataset, config)
    result = train(str(dataset), config, str(tmp_path / "run"))
    assert result["decision_capable"] is True

    source = frame.iloc[[0]].copy()
    source.loc[:, "complex_id"] = "NEW-COMPLEX"
    source.loc[:, "receptor_sequence"] = "ACDEFGHIKLMNPQRSTVWYACDEFGHIK"
    source.loc[:, "ligand_sequence"] = "WYVTSRQPN"
    for key, value in _graph_metadata(source.iloc[0], config["contact_graph"]).items():
        source.loc[:, key] = value
    artifact_features = config["features"]["columns"]
    inference_columns = [
        "complex_id",
        "run_id",
        "algorithm",
        "seed",
        "receptor_sequence",
        "ligand_sequence",
        "engine_reduced",
        "scientific_reduced",
        *artifact_features,
        *[column for column in source if column.startswith("graph_")],
    ]
    inference_columns = list(dict.fromkeys(inference_columns))
    inference = tmp_path / "new.csv"
    source[inference_columns].to_csv(inference, index=False)
    predictions = predict(result["artifact"], str(inference), str(tmp_path / "pred.csv"))
    assert predictions.loc[0, "action"] == "ACCEPT_REDUCED"
    assert bool(predictions.loc[0, "trusted_reduced_input_provenance"])
    assert bool(predictions.loc[0, "unseen_complex"])

    renamed = frame.iloc[[0]].copy()
    renamed.loc[:, "complex_id"] = "RENAMED-DEVELOPMENT-COMPLEX"
    for key, value in _graph_metadata(renamed.iloc[0], config["contact_graph"]).items():
        renamed.loc[:, key] = value
    renamed_input = tmp_path / "renamed.csv"
    renamed[inference_columns].to_csv(renamed_input, index=False)
    blocked = predict(
        result["artifact"],
        str(renamed_input),
        str(tmp_path / "renamed_predictions.csv"),
    )
    assert blocked.loc[0, "action"] == "RUN_FULL"
    assert blocked.loc[0, "reason"] == "complex_seen_during_artifact_development"
    assert not bool(blocked.loc[0, "unseen_complex"])
