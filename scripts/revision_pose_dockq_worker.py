"""Persistent, isolated DockQ 2.1.3 evaluator for a fixed two-chain trial."""
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.evaluation.official_dockq import _protein_chains
from docking_confidence.evaluation.official_dockq_runner import _backend, _sha
from docking_confidence.evaluation.reference_dockq import read_engine_order_atoms


def main():
    source = sys.stdin.buffer
    request = json.loads(source.readline())
    backend, version = _backend()
    if version["implementation_version"] != "2.1.3" or version["implementation_sha256"] != request["backend_sha256"]:
        raise ValueError("O DockQ difere da versão oficial usada no controle.")
    model_path, native_path = map(Path, (request["model"], request["reference"]))
    if _sha(native_path) != request["reference_sha256"]:
        raise ValueError("A referência difere do ensaio arquivado.")
    receptor, ligand = request["receptor_chain"], request["ligand_chain"]
    chains = {receptor: receptor, ligand: ligand}
    if len(chains) != 2 or any(_protein_chains(p) != set(chains) for p in (model_path, native_path)):
        raise ValueError("Modelo e referência devem conter as duas cadeias declaradas.")
    model = backend.load_PDB(str(model_path), chains=list(chains), small_molecule=False)
    native = backend.load_PDB(str(native_path), chains=list(chains), small_molecule=False)
    template = read_engine_order_atoms(model_path)
    mobile = [atom for atom in template if atom.chain_id == ligand]
    receptor_count = len(template) - len(mobile)
    # The article writer renumbers all ATOM/HETATM records consecutively. DockQ
    # can skip records. Map each atom it did parse back to the engine row.
    atoms = list(model[ligand].get_atoms())
    indices = np.array([atom.get_serial_number() - receptor_count - 1 for atom in atoms], dtype=int)
    if not atoms or len(set(indices)) != len(atoms) or min(indices) < 0 or max(indices) >= len(mobile):
        raise ValueError("Os átomos móveis do DockQ não correspondem aos do motor.")
    for atom, index in zip(atoms, indices):
        row = mobile[index]
        if row.atom_name != atom.get_name() or not np.array_equal(atom.coord, np.float32(row.coord)):
            raise ValueError("Ordem ou coordenadas divergentes na montagem do modelo.")
    print(json.dumps({"ready": True, "ligand_rows": len(mobile)}), flush=True)
    size = len(mobile) * 3 * 8
    while frame := source.read(size):
        if len(frame) != size:
            raise ValueError("Coordenadas incompletas.")
        coords = np.frombuffer(frame, dtype="<f8").reshape(-1, 3)[indices]
        # Match the actual PDB serializer (8.3f) and DockQ PDBParser's float32
        # conversion. Direct binary rounding differs at decimal ties.
        rounded = np.char.mod("%.3f", coords).astype(np.float64).astype(np.float32)
        for atom, xyz in zip(atoms, rounded):
            atom.coord[:] = xyz
        # DockQ's 2.1.3 caches use chain identity. Alignment and atom-count
        # caches depend only on topology. Distances, subsets and scores can
        # depend on the current coordinates and must be discarded every time.
        backend.get_residue_distances.cache_clear()
        backend.subset_atoms.cache_clear()
        backend.run_on_chains.cache_clear()
        mapping, _ = backend.run_on_all_native_interfaces(
            model, native, chain_map=chains, no_align=False, capri_peptide=False, low_memory=False)
        if len(mapping) != 1:
            raise ValueError("DockQ não identificou exatamente uma interface.")
        result = next(iter(mapping.values()))
        metrics = [float(result[name]) for name in ("DockQ", "fnat", "iRMSD", "LRMSD")]
        if not (0 <= metrics[0] <= 1 and 0 <= metrics[1] <= 1 and min(metrics[2:]) >= 0):
            raise ValueError("Métricas DockQ fora de faixa.")
        print(json.dumps(metrics, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
