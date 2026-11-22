import json

import pandas as pd
import pytest

from docking_confidence.application.batch import run_batch
from docking_confidence.application.experimental_protocol import prepare_experimental_protocol
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode
from docking_confidence.io.batch_discovery import discover_ppi4dock
from test_training_dataset import _FixturePoseEngine, _component


@pytest.mark.parametrize("initialization", ["uniform", "seeded"])
def test_training_selects_matched_initialization_before_reading_outcomes(tmp_path, initialization):
    source = tmp_path / "input"
    directory = source / "1abc_AB"
    directory.mkdir(parents=True)
    receptor = _component("A", "ALA", 0.0, 1)
    ligand = _component("B", "GLY", 3.0, 21)
    (directory / "A_model_st.pdb").write_text(receptor + "END\n")
    (directory / "B_model_st.pdb").write_text(ligand + "END\n")
    (directory / "1abc_AB_st.pdb").write_text(receptor + ligand + "END\n")
    report = discover_ppi4dock(source)
    jobs = build_execution_plan(algorithms=[Algorithm.AG], protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(2, 4, 1, experiment=FactorialExperimentConfig(evaluation_budget=12, pose_source="synthetic fixture")), seeds=[1])
    engine = _FixturePoseEngine(set())
    output = tmp_path / "batch"
    frozen = prepare_experimental_protocol(source, output, jobs=jobs, engine=engine, report=report)
    run_batch(source, output, jobs=jobs, engine=engine, report=report, experimental_protocol=frozen["path"])
    prepared = build_training_dataset(output, tmp_path / "prepared", factorial_initialization=initialization)
    assert prepared["valid"], prepared["issues"]
    assert prepared["counts"]["valid_pairs"] == 1
    assert prepared["experimental_initialization"] == initialization
    frame = pd.read_csv(prepared["dataset_path"])
    assert frame["run_id"].tolist() == [f"ag-reduced-{initialization}-seed1"]
    assert prepared["batch_manifest_sha256"] == json.loads((output / "batch_manifest.json").read_text())["manifest_sha256"]
