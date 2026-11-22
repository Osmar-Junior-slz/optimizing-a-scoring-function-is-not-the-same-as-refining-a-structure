"""Check the persistent evaluator against the actual, separately invoked paper endpoint."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import revision_random_pose_rate as rates
from revision_random_control import _persist_best_pose, evaluate_standard_complex, prepare_reduced_pose


@pytest.mark.skipif(not (rates.DEFAULT_STUDY / "planned_workers.json").is_file() or
                    not rates.DEFAULT_DEPS.is_dir(), reason="requires archived cohort and official DockQ runtime")
def test_persistent_worker_matches_the_official_endpoint_for_distinct_draws_and_replay(tmp_path):
    target, paths, configs = rates.inputs_for_study(rates.DEFAULT_STUDY.resolve(), 1)[0]
    backend = rates.official_dockq_preflight(dependency_path=rates.DEFAULT_DEPS)
    space = rates.space_for_trial(configs["expanded"], paths["ligand"])
    rng = np.random.default_rng(np.random.SeedSequence(1).spawn(2)[1])
    vectors = [rng.uniform(space.domain_low, space.domain_high, size=6) for _ in range(2)]
    poses = [space.solution_to_pose(v) for v in vectors]
    expected = []
    for index, pose in enumerate(poses):
        trial = tmp_path / str(index)
        saved = _persist_best_pose(out_dir=str(trial), best_pose=pose, engine_name="iid_uniform_control",
                                   search_space_mode="full")
        model = prepare_reduced_pose(receptor_model_pdb=paths["receptor"], ligand_template_pdb=paths["ligand"],
                                     receptor_chain=target["receptor_chain"], ligand_chain=target["ligand_chain"],
                                     pose_json=trial / saved["pose_output_path"], output_dir=trial / "model")
        score = evaluate_standard_complex(model, paths["reference"], receptor_chain=target["receptor_chain"],
                                          ligand_chain=target["ligand_chain"], output_dir=trial / "official",
                                          dependency_path=rates.DEFAULT_DEPS, timeout_seconds=180)["metrics"]
        expected.append([score[k] for k in rates.METRICS])
    assert abs(expected[0][0] - expected[1][0]) > .001
    with rates.scoring_worker(target, paths, rates.DEFAULT_DEPS, backend["implementation_sha256"], tmp_path) as score:
        for index in (0, 1, 0):
            assert score(poses[index].coords) == pytest.approx(expected[index], abs=1e-12, rel=0)
