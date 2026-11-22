from __future__ import annotations

import json
from pathlib import Path

import pytest

from docking_confidence.evaluation.reference_dockq import evaluate_reference_pose


def _line(
    serial: int,
    atom: str,
    residue: str,
    chain: str,
    number: int,
    xyz: tuple[float, float, float],
) -> str:
    x, y, z = xyz
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:^4s} {residue:>3s} {chain}{number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {element:>2s}\n"
    )


def _atoms(chain: str, y: float) -> list[tuple[str, tuple[float, float, float]]]:
    return [
        ("N", (0.0, y, 0.0)),
        ("CA", (1.0, y, 0.0)),
        ("C", (1.0, y + 1.0, 0.0)),
        ("O", (0.0, y + 1.0, 0.0)),
        ("CB", (1.0, y, 1.0)),
    ]


def _write_pdb(path: Path, components: list[tuple[str, float]]) -> list[list[float]]:
    lines: list[str] = []
    coords: list[list[float]] = []
    serial = 1
    for chain, y in components:
        for atom, xyz in _atoms(chain, y):
            lines.append(_line(serial, atom, "ALA", chain, 1, xyz))
            coords.append(list(xyz))
            serial += 1
    path.write_text("".join(lines) + "END\n", encoding="utf-8")
    return coords


def _pose(path: Path, coords: list[list[float]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "pose_role": "best_computed_pose",
                "search_space_mode": "reduced",
                "engine_name": "pso",
                "coords": coords,
                "score_cheap": -1.0,
                "score_expensive": None,
                "final_rescore": None,
                "pocket_id": "p1",
                "generation": 2,
                "meta": {},
            }
        ),
        encoding="utf-8",
    )


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, list[list[float]]]:
    receptor = tmp_path / "A_model_st.pdb"
    ligand = tmp_path / "B_model_st.pdb"
    reference = tmp_path / "1abc_AB_st.pdb"
    _write_pdb(receptor, [("A", 0.0)])
    ligand_coords = _write_pdb(ligand, [("B", 3.0)])
    _write_pdb(reference, [("A", 0.0), ("B", 3.0)])
    return receptor, ligand, reference, ligand_coords


def test_perfect_pose_uses_reference_post_search_and_exports_complex(tmp_path: Path):
    receptor, ligand, reference, coords = _inputs(tmp_path)
    pose = tmp_path / "best_pose.json"
    _pose(pose, coords)

    result = evaluate_reference_pose(
        receptor_model_pdb=receptor,
        ligand_template_pdb=ligand,
        reference_complex_pdb=reference,
        receptor_chain="A",
        ligand_chain="B",
        pose_json=pose,
        output_dir=tmp_path / "postprocess",
    )

    assert result.dockq == pytest.approx(1.0, abs=1e-8)
    assert result.fnat == 1.0
    assert result.irmsd == pytest.approx(0.0, abs=1e-8)
    assert result.lrmsd == pytest.approx(0.0, abs=1e-8)
    assert result.complex_pdb_path.is_file()
    payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert payload["reference_used_for_search"] is False
    assert payload["metrics"]["dockq"] == pytest.approx(1.0)
    chains = {
        line[21:22]
        for line in result.complex_pdb_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("ATOM")
    }
    assert chains == {"A", "B"}


def test_displaced_pose_scores_worse_than_reference(tmp_path: Path):
    receptor, ligand, reference, coords = _inputs(tmp_path)
    displaced = [[x + 20.0, y, z] for x, y, z in coords]
    pose = tmp_path / "best_pose.json"
    _pose(pose, displaced)

    result = evaluate_reference_pose(
        receptor_model_pdb=receptor,
        ligand_template_pdb=ligand,
        reference_complex_pdb=reference,
        receptor_chain="A",
        ligand_chain="B",
        pose_json=pose,
        output_dir=tmp_path / "postprocess",
    )

    assert result.fnat == 0.0
    assert result.lrmsd > 10.0
    assert result.dockq < 0.5


def test_pose_coordinate_count_must_match_dockingpp_template_order(tmp_path: Path):
    receptor, ligand, reference, coords = _inputs(tmp_path)
    pose = tmp_path / "best_pose.json"
    _pose(pose, coords[:-1])

    with pytest.raises(ValueError, match="expected"):
        evaluate_reference_pose(
            receptor_model_pdb=receptor,
            ligand_template_pdb=ligand,
            reference_complex_pdb=reference,
            receptor_chain="A",
            ligand_chain="B",
            pose_json=pose,
            output_dir=tmp_path / "postprocess",
        )
