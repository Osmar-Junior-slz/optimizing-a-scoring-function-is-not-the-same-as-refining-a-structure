"""Freeze a new factorial cohort and engine protocols without running docking.

Selection uses archived input identities only. The primary endpoint and official
DockQ assessment are specified separately in the prespecified analysis plan.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.experimental_protocol import prepare_experimental_protocol
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, FactorialExperimentConfig, ProtocolMode
from docking_confidence.domain.structure import CanonicalComplexFingerprint
from prepare_frozen_decision_selection import component

RANK_PREFIX = "factorial-20261119:"
ALGORITHMS = ["ag", "pso", "hybrid"]
POSE = "Supplied partner models; their original placement relative to the experimental reference is undocumented."
ORDER_SEED = 20260925
BUDGET = 7600
TIMEOUT = 600
INPUT_ROLES = ("receptor", "ligand", "reference")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def read(path):
    return json.loads(Path(path).read_bytes())


def write_record(path, value):
    """Write once, allowing an interrupted preparation to resume with identical content."""
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Conflicting frozen record: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")


def sequence_hash(sequence):
    return digest(sequence.encode("ascii"))


def prepare_selection(out, count):
    inventory_path = ROOT / "output/quality_delivery_20261119/inventory.json"
    registry_path = ROOT / "configs/decision_data_registry.json"
    old_selection = ROOT / "output/decision_study_20261119/selection.json"
    fit_path = ROOT / "output/standard_fit_20261119/paired_dataset.csv"
    ledger = sorted((ROOT / "output/decision_exposure_ledger").glob("*.json"))
    inventory = read(inventory_path)
    excluded = set(read(registry_path)["known_fingerprints"])
    for path in ledger:
        excluded.update(read(path)["groups"])

    used_components = set()
    for target in read(old_selection)["targets"]:
        excluded.add(target["fingerprint"])
        used_components.update(target["component_sequence_sha256"])
    with fit_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            excluded.add(row["complex_fingerprint_sha256"])
            for name in ("receptor_sequence", "ligand_sequence"):
                used_components.add(sequence_hash(row[name]))
    official_components = set(used_components)
    source_root = Path(inventory["input_root"]).resolve(strict=True)
    targets = []
    fingerprints = set()
    def rank(target):
        return digest((RANK_PREFIX + target["fingerprint"]).encode("ascii")), target["complex_id"]

    for target in sorted(inventory["new_targets"], key=rank):
        fingerprint = target["fingerprint"]
        if fingerprint in excluded or fingerprint in fingerprints:
            continue
        target_id = target["complex_id"]
        paths = {r: Path(target[r + "_path"]).resolve(strict=True) for r in INPUT_ROLES}
        if any(p.parent != source_root / target_id for p in paths.values()):
            raise ValueError(f"Input path not recorded under target's source directory: {target_id}")
        receptor, receptor_sha = component(paths["receptor"])
        ligand, ligand_sha = component(paths["ligand"])
        if CanonicalComplexFingerprint.from_sequences(receptor, ligand).sha256 != fingerprint:
            raise ValueError(f"Inventory fingerprint changed: {target_id}")
        component_hashes = {sequence_hash(receptor), sequence_hash(ligand)}
        if used_components.intersection(component_hashes):
            continue
        used_components.update(component_hashes)
        fingerprints.add(fingerprint)
        atoms = {}
        for r in ("receptor", "ligand"):
            with paths[r].open("rb") as handle:
                atoms[r] = sum(line.startswith(b"ATOM  ") for line in handle)
        targets.append({
            **target, "complex_fingerprint_sha256": fingerprint, "rank_sha256": rank(target)[0],
            "receptor_sha256": receptor_sha, "ligand_sha256": ligand_sha,
            "reference_sha256": file_hash(paths["reference"]), "component_sequence_sha256": sorted(component_hashes),
            "partition_cost": atoms["receptor"] * atoms["ligand"],
        })
        if len(targets) == count:
            break
    if len(targets) != count:
        raise ValueError(f"Only {len(targets)} eligible groups for requested {count}")
    record = {
        "schema_version": "official_dockq_factorial_selection_v1",
        "input_root": str(source_root), "rank_key_prefix": RANK_PREFIX,
        "selection_rule": "ascending SHA256(prefix + lowercase canonical fingerprint), ties by complex_id; skip all previously exposed groups and all exact component repeats against fitting, calibration, validation, test and earlier selections",
        "eligibility_scope": "eligible inventory, exact sequence identity; family independence not established",
        "outcome_access": "only input identities, paths, hashes and sequence columns used",
        "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in
                          (inventory_path, registry_path, old_selection, fit_path, *ledger)},
        "excluded_group_count": len(excluded), "excluded_official_component_count": len(official_components),
        "algorithms": ALGORITHMS, "seeds": [1], "target_count": count, "targets": targets,
    }
    write_record(out / "selection.json", record)
    return record


def stage_pdbs(folder, target):
    dest = folder / target["complex_id"]
    dest.mkdir(parents=True, exist_ok=True)
    for r in INPUT_ROLES:
        src = Path(target[r + "_path"])
        copied = dest / src.name
        if not copied.exists():
            shutil.copyfile(src, copied)
        if file_hash(copied) != target[r + "_sha256"]:
            raise ValueError(f"Changed molecular input: {copied}")


def prepare(out, scientific_plan, workers):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    selection = prepare_selection(out, 50)
    parts = [[] for _ in range(workers)]
    loads = [0] * workers
    for target in sorted(selection["targets"], key=lambda t: (-t["partition_cost"], t["complex_id"])):
        k = min(range(workers), key=lambda i: (loads[i], i))
        loads[k] += target["partition_cost"]
        parts[k].append(target)
    config = BaseSearchConfig(population=100, generations=100, top_pockets=1,
                              experiment=FactorialExperimentConfig(evaluation_budget=BUDGET,
                              reduced_search_scale=0.5, max_translation=5., max_rotation_degrees=25.,
                              pose_source=POSE, execution_order_seed=ORDER_SEED))
    jobs = build_execution_plan(algorithms=[Algorithm(a) for a in ALGORITHMS],
                                protocol_mode=ProtocolMode.FACTORIAL, base_config=config, seeds=[1])
    engine = DockingPPLocalEngine(timeout_seconds=TIMEOUT)
    shards = []
    for k, part in enumerate(parts, 1):
        shard = out / f"shard_{k:02d}"
        inputs, batch = shard / "inputs", shard / "batch"
        for target in part:
            stage_pdbs(inputs, target)
        prepared = prepare_experimental_protocol(inputs, batch, jobs=jobs, engine=engine, analysis_role="confirmatory")
        protocol = prepared["protocol"]; plan = protocol["plan"]
        planned = {t["complex_id"]: t["complex_fingerprint_sha256"] for t in plan["targets"]}
        expected = {t["complex_id"]: t["fingerprint"] for t in part}
        if planned != expected or len(plan["jobs"]) != 12:
            raise ValueError(f"Engine plan differs from selection for {shard}")
        argv = ["-B", "-m", "docking_confidence", "batch-run", "--input-dir", str(inputs),
                "--output-dir", str(batch), "--protocol", "factorial_2x2", "--docking-task", "local_refinement",
                "--algorithms", *ALGORITHMS, "--seed", "1", "--generations", "100", "--pop-size", "100",
                "--top-pockets", "1", "--evaluation-budget", str(BUDGET), "--reduced-search-scale", "0.5",
                "--max-translation", "5", "--max-rotation-degrees", "25", "--execution-order-seed", str(ORDER_SEED),
                "--pose-source", POSE, "--timeout-seconds", str(TIMEOUT),
                "--experimental-protocol", prepared["path"]]
        shards.append({"index": k, "input_dir": str(inputs), "batch_dir": str(batch), "complex_ids": sorted(expected),
                       "experimental_protocol_path": prepared["path"], "protocol_file_sha256": prepared["sha256"],
                       "expected_tasks": 12 * len(part), "argv": argv})
        print(f"Prepared shard {k}/{workers}: {len(part)} complexes", flush=True)
    scientific_plan = Path(scientific_plan).resolve()
    if not scientific_plan.is_file():
        raise FileNotFoundError(f"Scientific analysis plan not yet present; rerun preparation to seal: {scientific_plan}")
    master_path = out / "planned_workers.json"
    master = {"schema_version": "official_dockq_factorial_workers_v1",
              "created_at_utc": (read(master_path)["created_at_utc"] if master_path.exists() else
                                 datetime.now(timezone.utc).isoformat()),
              "selection_path": str(out / "selection.json"), "selection_sha256": file_hash(out / "selection.json"),
              "analysis_plan_path": str(scientific_plan), "analysis_plan_sha256": file_hash(scientific_plan),
              "preparation_script_sha256": file_hash(__file__),
              "thread_limits": {var:"1" for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"]},
              "planned_complexes": 50, "planned_tasks": 600, "shards": shards}
    write_record(master_path, master)
    print(f"Prepared {master_path} | SHA256 {file_hash(master_path)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/fatorial_dockq_20261119")
    parser.add_argument("--scientific-plan", type=Path, default=ROOT / "output/fatorial_dockq_20261119/protocol/analysis_plan.json")
    parser.add_argument("--workers", type=int, choices=range(1, 51), default=8)
    args = parser.parse_args()
    prepare(args.out, args.scientific_plan, args.workers)
