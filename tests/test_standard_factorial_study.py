"""Regressions for complete quartets and genuine versus technical failures."""
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_standard_factorial import CONDITIONS, canonical_sha256, structural_failure, summarize
from audit_factorial_clashes import assess_gate, gate_counts


def test_effects_are_marginal_paired_differences_and_keep_no_pose_failures():
    plan = {
        "design": {"algorithms": ["example"], "attempts": 8, "seeds": [1]},
        "selection": {"number_of_complexes": 2},
        "analysis": {
            "cell_order": list(CONDITIONS),
            "contrast_weights": {"extent": [0.5, 0.5, -0.5, -0.5],
                                 "initialization": [-0.5, 0.5, -0.5, 0.5],
                                 "interaction": [-1, 1, 1, -1]},
            "uncertainty": {"repetitions": 100, "seed": 2026},
        },
    }
    rows = []
    for fp, values in (("first", [1, 1, 0, 0]), ("second", [0, 1, 0, 0])):
        for condition, value in zip(CONDITIONS, values):
            pose = int(condition != "expanded_seeded")
            rows.append({"complex_fingerprint_sha256": fp, "algorithm": "example",
                         "condition": condition, "valid_pose": pose, "primary": value,
                         "outcome_state": "scored" if pose else "no_valid_pose"})
    report = summarize(rows, plan)
    assert report["pooled_cells"]["expanded_seeded"] == {"planned": 2, "valid_poses": 0, "acceptable_poses": 0}
    effects = report["by_method"]["example"]["effects"]
    assert [effects[k]["estimate_pp"] for k in ("extent", "initialization", "interaction")] == [75, 25, 50]
    assert effects["extent"]["ci95_pp"] == [50, 100]
    rows[-1]["outcome_state"] = "unresolved"
    with pytest.raises(ValueError, match="Complete and verified"):
        summarize(rows, plan)


def test_genuine_no_pose_may_lack_count_but_timeout_is_unresolved(tmp_path):
    target = {"complex_id": "example"}
    job = {"job_id": "reduced", "protocol": "reduced"}
    obs = {"evaluation_budget": 7600}
    plan_hash = "example_plan"
    err = {"status": "FAILED", "complex_id": "example", "job_id": "reduced",
           "attempt": 1, "engine_status": "REDUCED_FAILED", "failure_reason": "no_valid_reduced_pose",
           "plan_sha256": plan_hash,
           "task_sha256": canonical_sha256({"plan_sha256": plan_hash, "target": target, "job": job})}
    (tmp_path / "error.json").write_text(json.dumps(err), encoding="utf-8")
    answer = structural_failure(tmp_path, obs, plan_hash, target, job, 1)
    assert answer["primary"] == 0 and answer["actual_eval_count"] is None and answer["dockq"] is None
    err.update(engine_status=None, failure_reason="timeout")
    (tmp_path / "error.json").write_text(json.dumps(err), encoding="utf-8")
    with pytest.raises(ValueError, match="Non-geometric"):
        structural_failure(tmp_path, obs, plan_hash, target, job, 1)


def test_common_physical_rule_uses_strict_cutoff_and_records_unresolved_values():
    gate = {"enabled": True, "threshold": 1.2, "min_distance": 1.2,
            "penalty_score": -1e6, "triggered": False}
    meta = {"pocket_id": "local_refinement", "docking_task": "local_refinement",
            "search_anchor_mode": "input_centroid", "input_relative_placement_used": True,
            "local_refinement_search_scale": 0.5, "final_hard_clash_gate": gate,
            "cheap_score_variant": "example", "final_score_variant": "example"}
    assert assess_gate(meta, "reduced_uniform")["gate_triggered"] == 0
    meta["local_refinement_search_scale"] = 1.0
    gate["min_distance"] = 1.0
    with pytest.raises(ValueError, match="disagrees"):
        assess_gate(meta, "expanded_seeded")
    gate["triggered"] = True
    assert assess_gate(meta, "expanded_seeded")["gate_triggered"] == 1
    gate.pop("min_distance")
    with pytest.raises(ValueError, match="Unverified distance"):
        assess_gate(meta, "expanded_seeded")
    gate.update(min_distance=None, triggered=False)
    meta["vina_pair_stats"] = {"n_pairs": 0, "min_distance": None, "cutoff_pair": 8.0}
    assert assess_gate(meta, "expanded_seeded")["distance_evidence"] == "zero_pairs_within_scoring_cutoff"
    rejected = {"gate_triggered": 1, "native_valid_pose": 1, "native_primary": 1,
                "primary": 0, "actual_eval_count": 7600}
    assert gate_counts([rejected])["primary_response_changed"] == 1
