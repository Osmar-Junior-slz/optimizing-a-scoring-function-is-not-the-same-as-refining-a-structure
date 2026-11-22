import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

from docking_confidence.domain.structure import (
    CanonicalComplexFingerprint,
    RunBinding,
)
from docking_confidence.features.contact_graph import (
    ContactGraphConfig,
    describe_interface_contact_graph,
)
from docking_confidence.io.config import load_config
from docking_confidence.io.dataset import load_reduced_inference_dataset
from docking_confidence.io.pdb import read_complex_pose, read_pdb
from docking_confidence.cli import main as cli_main


def _atom_line(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    number: int,
    x: float,
    y: float,
    z: float,
    *,
    altloc: str = "",
    occupancy: float = 1.0,
) -> str:
    return (
        f"ATOM  {serial:5d} {atom:^4s}{altloc:1s}{residue:>3s} {chain:1s}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{20.0:6.2f}          {atom[0]:>2s}\n"
    )


def _provenance(protocol: str = "reduced") -> tuple[RunBinding, CanonicalComplexFingerprint]:
    return (
        RunBinding(
            complex_id="NEW",
            run_id="R1",
            algorithm="pso",
            seed=7,
            protocol=protocol,
            engine="validated-engine",
        ),
        CanonicalComplexFingerprint.from_sequences(
            "ACDEFGHIKLMNPQRSTVWY", "ACDEFGHIK"
        ),
    )


def _pose(pdb: Path, *, protocol: str = "reduced", receptor=("A",), ligand=("B",)):
    run, fingerprint = _provenance(protocol)
    return read_complex_pose(
        pdb,
        receptor_chains=receptor,
        ligand_chains=ligand,
        run_binding=run,
        complex_fingerprint=fingerprint,
    )


def test_interface_graph_has_known_edge_and_strict_row_binding(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        "".join(
            [
                _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0),
                _atom_line(2, "CA", "GLY", "B", 1, 3.0, 0.0, 0.0),
                _atom_line(3, "CA", "SER", "B", 2, 10.0, 0.0, 0.0),
                "END\n",
            ]
        ),
        encoding="utf-8",
    )
    pose = _pose(pdb)
    config = ContactGraphConfig(cutoff_angstrom=4.5, clash_cutoff_angstrom=2.0)
    first = describe_interface_contact_graph(pose, config)
    second = describe_interface_contact_graph(pose, config)
    assert first == second
    assert first["graph_has_interface_contact"] == 1
    assert first["graph_interface_edge_count"] == 1.0
    assert first["graph_interface_density"] == pytest.approx(0.5)
    assert first["graph_min_cross_distance_angstrom"] == pytest.approx(3.0)
    assert first["graph_contact_distance_mean"] == pytest.approx(3.0)
    assert first["graph_source_file_sha256"] != ""
    assert first["graph_pose_canonical_sha256"] != ""

    training_config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    record = {
        "scientific_reduced": True,
        "engine_reduced": "validated-engine",
        "scan_gap_norm": 0.1,
        "viable_fraction_mean": 0.7,
        "elite_dispersion": 0.2,
        "uncertainty_count": 1,
        "energy_q20_norm": -4.0,
        **first,
    }
    inference = tmp_path / "graph_inference.csv"
    pd.DataFrame([record]).to_csv(inference, index=False)
    loaded = load_reduced_inference_dataset(
        inference,
        training_config["features"]["columns"],
        training_config["contact_graph"],
    )
    assert len(loaded) == 1

    tampered = dict(record)
    tampered["run_id"] = "OTHER"
    pd.DataFrame([tampered]).to_csv(inference, index=False)
    with pytest.raises(ValueError, match="graph_bound_run_id"):
        load_reduced_inference_dataset(
            inference,
            training_config["features"]["columns"],
            training_config["contact_graph"],
        )


def test_zero_contact_graph_is_valid_and_all_model_features_are_finite(tmp_path: Path):
    pdb = tmp_path / "separated.pdb"
    pdb.write_text(
        _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)
        + _atom_line(2, "CA", "GLY", "B", 1, 12.0, 0.0, 0.0),
        encoding="utf-8",
    )
    descriptors = describe_interface_contact_graph(_pose(pdb), ContactGraphConfig())
    assert descriptors["graph_has_interface_contact"] == 0
    assert descriptors["graph_interface_edge_count"] == 0.0
    assert descriptors["graph_min_cross_distance_angstrom"] == pytest.approx(12.0)
    assert descriptors["graph_contact_distance_mean"] is None
    numeric = [
        value
        for key, value in descriptors.items()
        if key.startswith("graph_") and isinstance(value, (int, float))
    ]
    assert np.isfinite(np.asarray(numeric, dtype=float)).all()


def test_pose_protocol_model_and_component_contracts_fail_closed(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)
        + _atom_line(2, "CA", "GLY", "B", 1, 3.0, 0.0, 0.0),
        encoding="utf-8",
    )
    run, fingerprint = _provenance()
    with pytest.raises(ValueError, match="disjoint"):
        read_complex_pose(
            pdb,
            receptor_chains=["A"],
            ligand_chains=["A"],
            run_binding=run,
            complex_fingerprint=fingerprint,
        )
    with pytest.raises(ValueError, match="model_id=99"):
        read_pdb(pdb, model_id=99)
    with pytest.raises(ValueError, match="finite"):
        ContactGraphConfig(cutoff_angstrom=float("inf")).validate()

    with pytest.raises(ValueError, match="protocol='reduced'"):
        describe_interface_contact_graph(
            _pose(pdb, protocol="full"), ContactGraphConfig()
        )
    full_descriptors = describe_interface_contact_graph(
        _pose(pdb, protocol="reduced"), ContactGraphConfig()
    )
    full_descriptors["graph_pose_protocol"] = "full"
    config = load_config(Path(__file__).parents[1] / "configs" / "default.yaml")
    row = {
        "scientific_reduced": True,
        "scan_gap_norm": 0.1,
        "viable_fraction_mean": 0.7,
        "elite_dispersion": 0.2,
        "uncertainty_count": 1,
        "energy_q20_norm": -4.0,
        **full_descriptors,
    }
    source = tmp_path / "full_as_reduced.csv"
    pd.DataFrame([row]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="graph_pose_protocol"):
        load_reduced_inference_dataset(
            source, config["features"]["columns"], config["contact_graph"]
        )


def test_swapping_components_changes_the_canonical_pose_hash(tmp_path: Path):
    pdb = tmp_path / "complex.pdb"
    pdb.write_text(
        _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)
        + _atom_line(2, "CA", "GLY", "B", 1, 3.0, 0.0, 0.0),
        encoding="utf-8",
    )
    first = _pose(pdb, receptor=("A",), ligand=("B",))
    second = _pose(pdb, receptor=("B",), ligand=("A",))
    assert (
        first.provenance.selection.canonical_pose_sha256
        != second.provenance.selection.canonical_pose_sha256
    )


def test_graph_describe_cli_emits_the_v2_contract(tmp_path: Path, monkeypatch):
    pdb = tmp_path / "complex.pdb"
    out = tmp_path / "graph.json"
    pdb.write_text(
        _atom_line(1, "CA", "ALA", "A", 1, 0.0, 0.0, 0.0)
        + _atom_line(2, "CA", "GLY", "B", 1, 3.0, 0.0, 0.0),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "docking-confidence",
            "graph-describe",
            "--reduced-complex-pdb",
            str(pdb),
            "--complex-id",
            "C1",
            "--run-id",
            "R1",
            "--algorithm",
            "pso",
            "--seed",
            "7",
            "--engine",
            "validated-engine",
            "--receptor-sequence",
            "ACDEFGHIK",
            "--ligand-sequence",
            "ACDEF",
            "--receptor-chains",
            "A",
            "--ligand-chains",
            "B",
            "--out",
            str(out),
        ],
    )
    cli_main()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["graph_schema_version"] == "residue_interface_contact_graph_v2"
    assert payload["graph_pose_protocol"] == "reduced"
    assert payload["graph_bound_run_id"] == "R1"
