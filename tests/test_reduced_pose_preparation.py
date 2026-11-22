from __future__ import annotations

import json
from pathlib import Path

import pytest

from docking_confidence.evaluation import reference_dockq


def _pdb_line(serial: int, chain: str, xyz: tuple[float, float, float]) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {'CA':^4s} ALA {chain}{serial:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}          C\n"
    )


@pytest.fixture
def reduced_inputs(tmp_path: Path) -> dict:
    receptor = tmp_path / "receptor.pdb"
    ligand = tmp_path / "ligand.pdb"
    pose = tmp_path / "best_pose.json"
    # Two receptor atoms deliberately cannot support a reference alignment.
    receptor.write_text(
        _pdb_line(1, "A", (10.0, 20.0, 30.0))
        + _pdb_line(2, "A", (11.0, 22.0, 33.0)) + "END\n",
        encoding="utf-8",
    )
    ligand.write_text(
        _pdb_line(1, "B", (-10.0, -20.0, -30.0))
        + _pdb_line(2, "B", (-11.0, -22.0, -33.0)) + "END\n",
        encoding="utf-8",
    )
    pose.write_text(json.dumps({
        "schema_version": "1.0", "pose_role": "best_computed_pose",
        "coords": [[13.0, 20.0, 30.0], [14.0, 22.0, 33.0]],
    }), encoding="utf-8")
    return dict(
        receptor_model_pdb=receptor, ligand_template_pdb=ligand,
        receptor_chain="A", ligand_chain="B", pose_json=pose,
        output_dir=tmp_path / "calibration",
    )


def test_prepares_computed_coordinates_without_reference_or_alignment(
    reduced_inputs: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_reader = reference_dockq.read_engine_order_atoms
    read_paths = []

    def record_read(path):
        read_paths.append(Path(path))
        return original_reader(path)

    def reject_reference(*args, **kwargs):
        raise AssertionError("inference must not evaluate or align to a reference")

    monkeypatch.setattr(reference_dockq, "read_engine_order_atoms", record_read)
    monkeypatch.setattr(reference_dockq, "evaluate_reference_pose", reject_reference)
    monkeypatch.setattr(reference_dockq, "_kabsch", reject_reference)
    source_bytes = {
        key: reduced_inputs[key].read_bytes()
        for key in ("receptor_model_pdb", "ligand_template_pdb", "pose_json")
    }

    prepared = reference_dockq.prepare_reduced_pose(**reduced_inputs)

    assert read_paths == [
        reduced_inputs["receptor_model_pdb"], reduced_inputs["ligand_template_pdb"],
    ]
    atoms = original_reader(prepared)
    assert [(atom.chain_id, atom.coord) for atom in atoms] == [
        ("A", (10.0, 20.0, 30.0)), ("A", (11.0, 22.0, 33.0)),
        ("B", (13.0, 20.0, 30.0)), ("B", (14.0, 22.0, 33.0)),
    ]
    assert {p.name for p in prepared.parent.iterdir()} == {"reduced_complex.pdb"}
    for key, original in source_bytes.items():
        assert reduced_inputs[key].read_bytes() == original


@pytest.mark.parametrize("field,value,match", [
    ("coords", [[13.0, 20.0, 30.0]], "coordinates do not match"),
    ("coords", [[float("nan"), 20.0, 30.0], [14.0, 22.0, 33.0]], "must be finite"),
    ("pose_role", "native_pose", "best_computed_pose"),
    ("schema_version", "0.0", "schema_version"),
])
def test_preparation_preserves_pose_integrity_checks(
    reduced_inputs: dict, field: str, value: object, match: str,
) -> None:
    pose_path = reduced_inputs["pose_json"]
    payload = json.loads(pose_path.read_text(encoding="utf-8"))
    payload[field] = value
    pose_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        reference_dockq.prepare_reduced_pose(**reduced_inputs)
    assert not reduced_inputs["output_dir"].exists()


@pytest.mark.parametrize("role", ["receptor_model_pdb", "ligand_template_pdb"])
def test_preparation_rejects_unexpected_chains(reduced_inputs: dict, role: str) -> None:
    with reduced_inputs[role].open("a", encoding="utf-8") as handle:
        handle.write(_pdb_line(3, "C", (0.0, 0.0, 0.0)))
    with pytest.raises(ValueError, match="unexpected chains"):
        reference_dockq.prepare_reduced_pose(**reduced_inputs)


def test_preparation_does_not_overwrite_receptor(reduced_inputs: dict) -> None:
    receptor_path = reduced_inputs["output_dir"] / "reduced_complex.pdb"
    receptor_path.parent.mkdir(parents=True)
    original = reduced_inputs["receptor_model_pdb"].read_bytes()
    receptor_path.write_bytes(original)
    reduced_inputs["receptor_model_pdb"] = receptor_path
    with pytest.raises(ValueError, match="must not overwrite"):
        reference_dockq.prepare_reduced_pose(**reduced_inputs)
    assert receptor_path.read_bytes() == original
