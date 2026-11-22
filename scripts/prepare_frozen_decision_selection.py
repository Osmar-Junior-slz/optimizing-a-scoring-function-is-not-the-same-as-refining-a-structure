"""Freeze previously unrecorded molecular identities without reading outcomes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.domain.structure import CanonicalComplexFingerprint
from docking_confidence.io.batch_discovery import _STANDARD_RESIDUES


def digest(data):
    return hashlib.sha256(data).hexdigest()


def component(path):
    payload = Path(path).read_bytes()
    residues = {}
    for line in payload.decode("ascii").splitlines():
        if line.startswith("ENDMDL"):
            break
        if line.startswith("ATOM  "):
            key = (line[21:22], int(line[22:26]), line[26:27])
            amino = _STANDARD_RESIDUES[line[17:20].strip()]
            if key in residues and residues[key] != amino:
                raise ValueError("Conflicting residue identities")
            residues[key] = amino
    sequence = "".join(residues[k] for k in sorted(residues))
    if not sequence:
        raise ValueError("Empty molecular component")
    return sequence, digest(payload)


def prepare(inventory_path, registry_path, out_dir, *, count=50, seed=20261119):
    inventory_bytes = Path(inventory_path).read_bytes()
    registry_bytes = Path(registry_path).read_bytes()
    inventory, registry = json.loads(inventory_bytes), json.loads(registry_bytes)
    known = set(registry["known_fingerprints"])
    unique = {}
    for candidate in sorted(inventory["new_targets"], key=lambda t: t["complex_id"]):
        unique.setdefault(candidate["fingerprint"], candidate)
    ranked = sorted(unique.values(), key=lambda t: digest(f"{seed}:{t['fingerprint']}".encode()))
    selected, components, used = [], {"validation": set(), "test": set()}, set()
    counts = {"validation": 0, "test": 0}
    for role in ("validation", "test"):
        for candidate in ranked:
            fp = candidate["fingerprint"]
            if fp in used or fp in known:
                continue
            receptor, receptor_hash = component(candidate["receptor_path"])
            ligand, ligand_hash = component(candidate["ligand_path"])
            if CanonicalComplexFingerprint.from_sequences(receptor, ligand).sha256 != fp:
                raise ValueError(f"Inventory identity changed: {candidate['complex_id']}")
            sequences = {digest(receptor.encode()), digest(ligand.encode())}
            if role == "test" and sequences & components["validation"]:
                continue
            target = dict(candidate)
            target.update(role=role, path=str(Path(candidate["receptor_path"]).parent),
                          complex_fingerprint_sha256=fp,
                          receptor_sha256=receptor_hash, ligand_sha256=ligand_hash,
                          reference_sha256=digest(Path(candidate["reference_path"]).read_bytes()),
                          component_sequence_sha256=sorted(sequences))
            selected.append(target)
            used.add(fp)
            components[role].update(sequences)
            counts[role] += 1
            if counts[role] == count:
                break
        if counts[role] != count:
            raise ValueError(f"Insufficient fresh targets for {role}: {counts[role]}")
    selection = {
        "schema_version": "frozen_decision_study_selection_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": inventory["input_root"], "random_seed": seed,
        "selection_method": "SHA256(seed:fingerprint) rank; first validation groups, then test groups excluding exact shared component sequences",
        "inventory_sha256": digest(inventory_bytes), "registry_sha256": digest(registry_bytes),
        "eligible_fresh_unique_groups": len(unique), "counts": counts,
        "algorithms": ["ag", "pso", "hybrid"], "seeds": [1],
        "scope": "unseen in local saved evidence; exact molecular groups and exact components disjoint between validation and test; homology independence not established",
        "outcome_access": "no docking outcome, prediction or quality label read during selection",
        "targets": selected,
    }
    canonical = json.dumps(selection, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    selection["record_sha256"] = digest(canonical)
    split = {
        "schema_version": "decision_group_split_v1",
        "validation_groups": sorted(t["fingerprint"] for t in selected if t["role"] == "validation"),
        "test_groups": sorted(t["fingerprint"] for t in selected if t["role"] == "test"),
        "prior_exposure_groups": sorted(known), "algorithms": ["ag", "pso", "hybrid"], "seeds": [1],
        "selection_record_sha256": selection["record_sha256"],
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, record in (("selection.json", selection), ("split.json", split)):
        with (out / name).open("x", encoding="utf-8") as handle:
            json.dump(record, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
    return {"selection_path": str((out / "selection.json").resolve()), "counts": counts,
            "exact_component_overlap": len(components["validation"] & components["test"]),
            "record_sha256": selection["record_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--registry", default="configs/decision_data_registry.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--count-per-role", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(prepare(args.inventory, args.registry, args.out_dir, count=args.count_per_role), indent=2))
