"""Check initial-pose eligibility under the factorial study's final steric rule."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from revision_initial_pose import DEFAULT_STUDY, OUTPUT, _BatchFileLock, _atomic_write_json, _file_sha256
from revision_initial_pose import canonical_sha256, inputs_for_study, read, require, space_for_trial, write_csv
from revision_random_control import Config, load_config, load_peptide, load_receptor
from dockingpp.scoring.expensive import _apply_final_hard_clash_gate, debug_pose_final_rescore


def run(study, entry):
    study, entry = study.resolve(strict=True), entry.resolve(strict=True)
    scores_path = entry / "initial_scores.csv"
    plan = read(entry / "protocol.json")
    summary = read(entry / "summary.json")
    require(plan["schema"] == "article_unchanged_input_dockq_v1" and
            plan["selection_sha256"] == _file_sha256(study / "selection.json") and
            summary["initial_scores_sha256"] == _file_sha256(scores_path) and
            summary["protocol_sha256"] == canonical_sha256(plan), "Resultados iniciais incompatíveis.")
    with scores_path.open(encoding="utf-8-sig", newline="") as stream:
        scores = {r["complex_id"]: r for r in csv.DictReader(stream)}
    cases = inputs_for_study(study, None)
    require(len(cases) == len(scores) == 50 and {t["complex_id"] for t, _, _ in cases} == set(scores),
            "Os dados não contêm os mesmos 50 complexos.")
    rows = []
    for target, paths, configs in cases:
        cid = target["complex_id"]
        row = scores[cid]
        require(all(row[key+"_sha256"] == target[key+"_sha256"] for key in ("receptor","ligand","reference")),
                f"Dados de entrada incompatíveis: {cid}")
        receptor = load_receptor(str(paths["receptor"]))
        ligand = load_peptide(str(paths["ligand"]))
        pose = space_for_trial(configs["reduced"], paths["ligand"]).solution_to_pose(np.zeros(6))
        narrow = Config.model_validate(load_config(str(configs["reduced"])))
        wide = Config.model_validate(load_config(str(configs["expanded"])))
        evidence = debug_pose_final_rescore(pose, receptor, ligand, narrow)
        gate = evidence["hard_clash_gate"]
        gate_wide = _apply_final_hard_clash_gate({"pair_stats": {"min_distance": gate["min_distance"]}},
                                                wide)["hard_clash_gate"]
        distances, _ = cKDTree(receptor).query(ligand, k=1)
        minimum = float(np.min(distances))
        require(gate["enabled"] and gate_wide["enabled"] and gate["threshold"] == gate_wide["threshold"] == 1.2
                and gate["triggered"] == gate_wide["triggered"] == (minimum < 1.2)
                and (gate["min_distance"] is None or math.isclose(minimum, gate["min_distance"], abs_tol=1e-10)),
                f"Distância mínima ou veto físico incompatível: {cid}")
        good = float(row["initial_dockq"]) >= .23
        rows.append({"complex_id": cid, "initial_dockq": row["initial_dockq"],
                     "min_interpartner_distance_angstrom": minimum, "hard_clash_below_1_2": int(gate["triggered"]),
                     "acceptable_dockq": int(good), "acceptable_with_common_gate": int(good and not gate["triggered"])})
    with _BatchFileLock(entry / ".initial_sterics.lock"):
        out = entry / "initial_sterics.csv"
        write_csv(out, (r.values() for r in rows), list(rows[0]))
        result = {"script_sha256": _file_sha256(Path(__file__)), "input_scores_sha256": _file_sha256(scores_path),
                  "sterics_csv_sha256": _file_sha256(out), "protocol_sha256": summary["protocol_sha256"],
                  "n_complexes": len(rows), "threshold_angstrom_strictly_below": 1.2,
                  "n_hard_clash": sum(r["hard_clash_below_1_2"] for r in rows),
                  "n_acceptable_dockq": sum(r["acceptable_dockq"] for r in rows),
                  "n_acceptable_with_common_gate": sum(r["acceptable_with_common_gate"] for r in rows),
                  "min_interpartner_distance_angstrom": min(r["min_interpartner_distance_angstrom"] for r in rows),
                  "max_of_min_interpartner_distances_angstrom": max(r["min_interpartner_distance_angstrom"] for r in rows)}
        _atomic_write_json(entry / "initial_sterics_summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--entry", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.entry), indent=2))
