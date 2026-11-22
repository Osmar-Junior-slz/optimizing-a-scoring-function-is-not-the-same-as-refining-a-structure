from __future__ import annotations

import json
from pathlib import Path

import pytest

from docking_confidence.application.batch import run_batch
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import (
    Algorithm,
    BaseSearchConfig,
    DockingRunResult,
    ExecutionJob,
    ProtocolMode,
)
from docking_confidence.io.batch_discovery import discover_ppi4dock


def _line(
    serial: int,
    atom: str,
    chain: str,
    residue_number: int,
    xyz: tuple[float, float, float],
) -> str:
    x, y, z = xyz
    return (
        f"ATOM  {serial:5d} {atom:^4s} {'ALA':>3s} {chain}{residue_number:4d}    "
        f"{x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {atom[0]:>2s}\n"
    )


def _component(chain: str, y: float, *, serial_start: int = 1) -> tuple[str, list[list[float]]]:
    lines: list[str] = []
    coords: list[list[float]] = []
    serial = serial_start
    for residue in range(1, 6):
        x0 = float((residue - 1) * 3)
        for atom, offset in (
            ("N", (0.0, 0.0, 0.0)),
            ("CA", (1.0, 0.0, 0.0)),
            ("C", (1.0, 1.0, 0.0)),
            ("O", (0.0, 1.0, 0.0)),
            ("CB", (1.0, 0.0, 1.0)),
        ):
            xyz = (x0 + offset[0], y + offset[1], offset[2])
            lines.append(_line(serial, atom, chain, residue, xyz))
            coords.append(list(xyz))
            serial += 1
    return "".join(lines), coords


def _target(root: Path) -> tuple[Path, list[list[float]]]:
    directory = root / "1abc_AB"
    directory.mkdir(parents=True)
    receptor_text, _ = _component("A", 0.0)
    ligand_text, ligand_coords = _component("B", 3.0)
    (directory / "A_model_st.pdb").write_text(receptor_text + "END\n", encoding="utf-8")
    (directory / "B_model_st.pdb").write_text(ligand_text + "END\n", encoding="utf-8")
    (directory / "1abc_AB_st.pdb").write_text(
        receptor_text + ligand_text + "END\n", encoding="utf-8"
    )
    return directory, ligand_coords


class _ArtifactEngine:
    name = "artifact-scientific-test"
    scientific = True
    supports_artifact_work_dir = True
    requires_reference_postprocess = True

    def __init__(self, ligand_coords: list[list[float]]) -> None:
        self.ligand_coords = ligand_coords

    def audit_descriptor(self) -> dict[str, str]:
        return {"test_adapter": "v1"}

    def run(
        self,
        *,
        job: ExecutionJob,
        protein_path: Path,
        peptide_path: Path,
        work_dir: Path | None = None,
    ) -> DockingRunResult:
        assert work_dir is not None
        artifacts = work_dir / "artifacts"
        artifacts.mkdir(parents=True)
        (artifacts / "best_pose.json").write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "pose_role": "best_computed_pose",
                    "search_space_mode": job.protocol.value,
                    "engine_name": job.algorithm.value,
                    "coords": self.ligand_coords,
                    "score_cheap": -1.0,
                    "score_expensive": None,
                    "final_rescore": None,
                    "pocket_id": "p1",
                    "generation": 1,
                    "meta": {},
                }
            ),
            encoding="utf-8",
        )
        for name in ("summary.json", "result.json", "config.resolved.yaml"):
            (work_dir / name).write_text("{}\n", encoding="utf-8")
        return DockingRunResult(
            job_id=job.job_id,
            algorithm=job.algorithm.value,
            protocol=job.protocol.value,
            seed=job.seed,
            generations=job.profile.generations,
            population=job.profile.population,
            top_pockets=job.profile.top_pockets,
            evaluation_budget=job.profile.evaluation_budget,
            runtime_sec=1.0,
            dockq=None,
            pose_artifact_path="artifacts/best_pose.json",
            engine_summary_path="summary.json",
            engine_result_path="result.json",
            resolved_config_path="config.resolved.yaml",
            exit_code=0,
            engine=self.name,
            scientific=True,
        )


def test_scientific_artifact_is_evaluated_and_graphed_after_search(tmp_path: Path):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    input_root.mkdir()
    _, ligand_coords = _target(input_root)
    report = discover_ppi4dock(input_root, validate_contents=True)
    jobs = build_execution_plan(
        algorithms=[Algorithm.PSO],
        protocol_mode=ProtocolMode.REDUCED,
        base_config=BaseSearchConfig(2, 2, 1),
        seeds=[7],
    )

    summary = run_batch(
        input_root,
        output_root,
        jobs=jobs,
        engine=_ArtifactEngine(ligand_coords),
        report=report,
    )

    assert summary.state == "COMPLETED"
    receipt_path = (
        output_root
        / "targets"
        / "1abc_AB"
        / "jobs"
        / jobs[0].job_id
        / "receipt.json"
    )
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result = receipt["engine_result"]
    assert result["dockq"] == pytest.approx(1.0)
    assert result["fnat"] == 1.0
    for key in (
        "pose_artifact_path",
        "complex_pose_pdb_path",
        "reference_metrics_path",
        "graph_features_path",
    ):
        assert (receipt_path.parent / result[key]).is_file()
    metrics = json.loads(
        (receipt_path.parent / result["reference_metrics_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert metrics["reference_used_for_search"] is False
