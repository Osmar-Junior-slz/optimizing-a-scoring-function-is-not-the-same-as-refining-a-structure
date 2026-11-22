"""Score the unchanged input pose of each complex in the published factorial cohort."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
from revision_random_control import (
    DEFAULT_DEPS, DEFAULT_STUDY, EXTENTS, ROOT, _BatchFileLock, _atomic_write_json, _file_sha256,
    canonical_sha256, engine_identity, evaluate_standard_complex, inputs_for_study,
    official_dockq_preflight, quality_metric_descriptor, read, require, verified_record,
)
from revision_random_pose_rate import space_for_trial, write_csv
from docking_confidence.evaluation.reference_dockq import _write_complex, read_engine_order_atoms

ORIGINAL = ROOT / "output/fatorial_dockq_20261119"
OUTPUT = ROOT / "output/revisao_professor/pose_inicial/entrada"
METRICS = ("dockq", "fnat", "irmsd", "lrmsd")
BACKEND_KEYS = ("implementation", "implementation_version", "implementation_sha256", "constants_sha256")


def original_evidence(cases, study, original, backend):
    """Verify the cohort, outcome ledger, provenance and explicit mappings of its scored runs."""
    result = read(original / "factorial_results.json")
    data = original / "factorial_observations.csv"
    require(result["complete"] and result["observations_sha256"] == _file_sha256(data)
            and result["selection_sha256"] == _file_sha256(study / "selection.json")
            and result["quality_metric"] == quality_metric_descriptor()
            and all(result["evaluator"][k] == backend[k] for k in BACKEND_KEYS),
            "A coorte ou a implementação do DockQ diverge do artigo.")
    targets = {target["complex_id"]: target for target, _, _ in cases}
    conditions = [f"{region}_{init}" for region in EXTENTS for init in ("seeded", "uniform")]
    expected = set(product(targets, ("ag", "pso", "hybrid"), conditions))
    require(len(targets) == 50, "A coorte do artigo deve conter 50 complexos.")
    scored = 0
    with data.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            cid = row["complex_id"]
            key = (cid, row["algorithm"], row["condition"])
            require(key in expected, f"Observação histórica repetida ou desconhecida: {key}")
            expected.remove(key)
            target = targets[cid]
            require(row["complex_fingerprint_sha256"] == target["complex_fingerprint_sha256"],
                    f"Identidade histórica divergente: {cid}")
            if row["outcome_state"] == "no_valid_pose":
                require(row["dockq"] == "" and row["valid_pose"] == "0", f"Rejeição histórica inconsistente: {key}")
                continue
            path = (original / row["official_record_relative_path"]).resolve(strict=True)
            rec, lig = target["receptor_chain"], target["ligand_chain"]
            require(path.is_relative_to(original) and _file_sha256(path) == row["official_record_sha256"],
                    f"Recibo histórico alterado: {key}")
            record = verified_record(path)
            require(row["outcome_state"] == "scored" and
                    row["reference_sha256"] == record["inputs"]["reference_sha256"] == target["reference_sha256"]
                    and record["native_to_model_chain_map"] == {rec: rec, lig: lig}
                    and record["quality_metric_sha256"] == result["quality_metric_sha256"]
                    and all(record["implementation"][k] == backend[k] for k in BACKEND_KEYS),
                    f"Referência, cadeias ou avaliador histórico divergente: {key}")
            scored += 1
    require(not expected and scored == 587, "Cobertura histórica incompatível com o artigo.")
    return result


def initial_model(target, paths, configs, directory):
    """Assemble the two supplied partners, checking the null transform in both archived regions."""
    receptor = read_engine_order_atoms(paths["receptor"])
    ligand = read_engine_order_atoms(paths["ligand"])
    rec, lig = target["receptor_chain"], target["ligand_chain"]
    require({a.chain_id for a in receptor} == {rec} and {a.chain_id for a in ligand} == {lig} and rec != lig,
            f"Cadeias de entrada divergentes: {target['complex_id']}")
    initial = np.asarray([a.coord for a in ligand])
    errors = {}
    for extent in EXTENTS:
        space = space_for_trial(configs[extent], paths["ligand"])
        require(space.anchor_mode == "input_centroid" and np.array_equal(space.base_coords, initial),
                "Incompatibilidade entre os leitores de coordenadas do motor e do DockQ.")
        identity = space.solution_to_pose(np.zeros(6)).coords
        errors[extent] = float(np.max(np.abs(identity - initial)))
        require(errors[extent] < 1e-10 and np.array_equal(np.round(identity, 3), initial),
                "A transformação nula alterou as coordenadas da pose fornecida.")
    model = directory / "initial_complex.pdb"
    _write_complex(model, receptor, ligand)
    combined = read_engine_order_atoms(model)
    source = receptor + ligand
    require(len(combined) == len(source) and all(
        a.match_key == b.match_key and a.residue_name == b.residue_name and a.altloc == b.altloc and a.coord == b.coord
        for a,b in zip(source, combined)), "A montagem PDB não conservou a ordem, identidade e coordenadas dos átomos.")
    return model, {"receptor_atoms": len(receptor), "ligand_atoms": len(ligand),
                   "maximum_abs_zero_transform_error_angstrom": errors}


def score(case, output, deps, plan, plan_sha):
    target, paths, configs = case
    cid = target["complex_id"]
    directory = output / "cases" / cid
    directory.mkdir(parents=True, exist_ok=True)
    model, geometry = initial_model(target, paths, configs, directory)
    model_sha = _file_sha256(model)
    binding = {"complex_id": cid, "protocol_sha256": plan_sha, "pose": "unchanged_supplied_input",
               "complex_fingerprint_sha256": target["complex_fingerprint_sha256"],
               "receptor_sha256": target["receptor_sha256"], "ligand_sha256": target["ligand_sha256"],
               "geometry_checks": geometry}
    artifact = directory / "standard_dockq.json"
    if not artifact.is_file():
        with TemporaryDirectory(dir=directory) as temporary:
            evaluate_standard_complex(model, paths["reference"], receptor_chain=target["receptor_chain"],
                ligand_chain=target["ligand_chain"], expected_model_sha256=model_sha,
                expected_reference_sha256=target["reference_sha256"], expected_metric_sha256=plan["quality_metric_sha256"],
                dependency_path=deps, source_binding=binding, output_dir=temporary, timeout_seconds=600)
            os.replace(Path(temporary) / artifact.name, artifact)
    record = verified_record(artifact)
    rec, lig = target["receptor_chain"], target["ligand_chain"]
    require(record["source_binding"] == binding and record["quality_metric_sha256"] == plan["quality_metric_sha256"]
            and record["inputs"]["model_sha256"] == model_sha
            and record["inputs"]["reference_sha256"] == target["reference_sha256"]
            and record["native_to_model_chain_map"] == {rec: rec, lig: lig}
            and all(record["implementation"][k] == plan["backend"][k] for k in BACKEND_KEYS),
            f"Recibo incompatível: {artifact}")
    return {"complex_id": cid, **{f"initial_{k}": record["metrics"][k] for k in METRICS},
            "receptor_chain": rec, "ligand_chain": lig,
            **{k: target[k] for k in ("complex_fingerprint_sha256","receptor_sha256","ligand_sha256","reference_sha256")},
            "initial_model_sha256": model_sha, "receipt": artifact.relative_to(output).as_posix(),
            "receipt_sha256": _file_sha256(artifact), "zero_transform_max_error_angstrom":
                max(geometry["maximum_abs_zero_transform_error_angstrom"].values())}


def run(study, original, output, deps):
    study, original, output = study.resolve(strict=True), original.resolve(strict=True), output.resolve()
    for source in (study, original):
        require(not output.is_relative_to(source) and not source.is_relative_to(output),
                "Escolha uma pasta separada da coleta e dos resultados históricos.")
    cases = inputs_for_study(study, None)
    backend = official_dockq_preflight(dependency_path=deps, timeout_seconds=180)
    require(backend.get("available") and backend["quality_metric"] == quality_metric_descriptor(),
            f"DockQ oficial 2.1.3 indisponível: {backend.get('error','')}")
    historical = original_evidence(cases, study, original, backend)
    plan = {"schema": "article_unchanged_input_dockq_v1", "selection_sha256": historical["selection_sha256"],
            "historical_observations_sha256": historical["observations_sha256"],
            "quality_metric": historical["quality_metric"], "quality_metric_sha256": historical["quality_metric_sha256"],
            "backend": {k: backend[k] for k in BACKEND_KEYS}, "engine_manifest_sha256": engine_identity(),
            "script_sha256": _file_sha256(Path(__file__)), "solution_vector": [0] * 6,
            "geometry": "supplied atomic coordinates in their shared frame; receptor fixed; no prior alignment",
            "input_orientation_provenance": "archived supplied placement; its origin relative to the native is undocumented",
            "endpoint": "one official DockQ per complex before search, without selecting candidates or steric filtering",
            "cases": [{**{k: t[k] for k in ("complex_id", "receptor_chain", "ligand_chain", "receptor_sha256",
                      "ligand_sha256", "reference_sha256")}, "geometry_config_sha256":
                      {extent: _file_sha256(cfg[extent]) for extent in EXTENTS}} for t, _, cfg in cases]}
    plan_sha = canonical_sha256(plan)
    output.mkdir(parents=True, exist_ok=True)
    with _BatchFileLock(output / ".initial_pose.lock"):
        protocol = output / "protocol.json"
        if protocol.exists():
            require(read(protocol) == plan, "A pasta contém resultados de outro protocolo.")
        else:
            require(all(p.name == ".initial_pose.lock" for p in output.iterdir()), "A pasta de saída deve estar vazia.")
            _atomic_write_json(protocol, plan)
        rows = []
        for index, case in enumerate(cases, start=1):
            rows.append(score(case, output, deps, plan, plan_sha))
            if index % 10 == 0:
                print(f"{index}/50", flush=True)
        require(len({r["complex_id"] for r in rows}) == len(rows) == 50, "Cobertura da entrada incompleta.")
        path = output / "initial_scores.csv"
        write_csv(path, (r.values() for r in rows), list(rows[0]))
        scores = np.asarray([r["initial_dockq"] for r in rows])
        summary = {"protocol_sha256": plan_sha, "initial_scores_sha256": _file_sha256(path), "n_complexes": len(rows),
                   "n_acceptable": int(sum(scores >= .23)), "mean": float(np.mean(scores)),
                   "median": float(np.median(scores)), "minimum": float(np.min(scores)), "maximum": float(np.max(scores)),
                   "max_zero_transform_error_angstrom": max(r["zero_transform_max_error_angstrom"] for r in rows)}
        _atomic_write_json(output / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--historical-results", type=Path, default=ORIGINAL)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--dockq-deps", type=Path, default=DEFAULT_DEPS if DEFAULT_DEPS.is_dir() else None)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.historical_results, args.output_dir, args.dockq_deps), indent=2))
