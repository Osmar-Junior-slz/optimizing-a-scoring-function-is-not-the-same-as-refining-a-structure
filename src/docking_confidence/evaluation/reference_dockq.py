from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Iterable, Sequence

import numpy as np


_BACKBONE = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True, order=True)
class ResidueId:
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str


@dataclass(frozen=True)
class PDBAtom:
    record_name: str
    serial: int
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str
    coord: tuple[float, float, float]
    element: str
    source_line: str

    @property
    def residue_id(self) -> ResidueId:
        return ResidueId(
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )

    @property
    def match_key(self) -> tuple[str, int, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.atom_name,
        )

    @property
    def is_hydrogen(self) -> bool:
        return self.element.upper() in {"H", "D"}

    @property
    def is_backbone(self) -> bool:
        return self.atom_name.upper() in _BACKBONE


@dataclass(frozen=True)
class ReferenceDockQResult:
    dockq: float
    fnat: float
    irmsd: float
    lrmsd: float
    dockq_category: str
    native_contacts: int
    model_contacts: int
    recovered_native_contacts: int
    receptor_alignment_atoms: int
    ligand_rmsd_atoms: int
    interface_rmsd_atoms: int
    complex_pdb_path: Path
    metrics_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _guess_element(atom_name: str) -> str:
    letters = "".join(character for character in atom_name if character.isalpha())
    if not letters:
        return ""
    if letters[:2].upper() in {"CL", "BR"}:
        return letters[:2].upper()
    return letters[0].upper()


def read_engine_order_atoms(path: str | Path) -> list[PDBAtom]:
    """Mirror DockingPP's ATOM/HETATM coordinate-loader order exactly."""

    source = Path(path)
    atoms: list[PDBAtom] = []
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                coord = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except ValueError:
                continue
            try:
                residue_number = int(line[22:26].strip())
            except ValueError as exc:
                raise ValueError(
                    f"invalid residue number in {source.name} at accepted atom "
                    f"record {len(atoms) + 1}"
                ) from exc
            try:
                serial = int(line[6:11].strip())
            except ValueError:
                serial = len(atoms) + 1
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            atoms.append(
                PDBAtom(
                    record_name=line[:6].strip().upper(),
                    serial=serial,
                    atom_name=atom_name,
                    altloc=line[16:17].strip(),
                    residue_name=line[17:20].strip().upper(),
                    chain_id=line[21:22].strip() or "_",
                    residue_number=residue_number,
                    insertion_code=line[26:27].strip(),
                    coord=tuple(float(value) for value in coord),
                    element=(element or _guess_element(atom_name)).upper(),
                    source_line=line.rstrip("\r\n"),
                )
            )
    if not atoms:
        raise ValueError(f"no DockingPP-readable ATOM/HETATM records in {source}")
    return atoms


def _load_pose(path: Path, expected_atoms: int) -> tuple[np.ndarray, dict[str, object]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid best-pose JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("best-pose JSON must contain one object")
    if payload.get("schema_version") != "1.0":
        raise ValueError("best-pose schema_version must be '1.0'")
    if payload.get("pose_role") != "best_computed_pose":
        raise ValueError("best-pose pose_role must be 'best_computed_pose'")
    coords = np.asarray(payload.get("coords"), dtype=float)
    if coords.shape != (expected_atoms, 3):
        raise ValueError(
            "best-pose coordinates do not match the DockingPP ligand template "
            f"order: expected {(expected_atoms, 3)}, observed {coords.shape}"
        )
    if not np.all(np.isfinite(coords)):
        raise ValueError("best-pose coordinates must be finite")
    provenance = {
        key: payload.get(key)
        for key in (
            "schema_version",
            "pose_role",
            "search_space_mode",
            "engine_name",
            "score_cheap",
            "score_expensive",
            "final_rescore",
            "pocket_id",
            "generation",
            "meta",
        )
    }
    for key in ("score_cheap", "score_expensive", "final_rescore"):
        value = provenance[key]
        if value is not None and not math.isfinite(float(value)):
            raise ValueError(f"best-pose {key} must be finite or null")
    return coords, provenance


def _only_chain(atoms: Sequence[PDBAtom], chain: str, *, role: str) -> list[PDBAtom]:
    selected = [atom for atom in atoms if atom.chain_id == chain]
    if not selected:
        raise ValueError(f"{role} chain {chain!r} has no readable atoms")
    return selected


def _coords(atoms: Sequence[PDBAtom]) -> np.ndarray:
    return np.asarray([atom.coord for atom in atoms], dtype=float)


def _with_coords(atoms: Sequence[PDBAtom], coords: np.ndarray) -> list[PDBAtom]:
    if coords.shape != (len(atoms), 3):
        raise ValueError("replacement coordinates have the wrong shape")
    return [
        replace(atom, coord=tuple(float(value) for value in coords[index]))
        for index, atom in enumerate(atoms)
    ]


def _atom_map(
    atoms: Iterable[PDBAtom],
    *,
    backbone_only: bool,
    residue_filter: set[ResidueId] | None = None,
) -> dict[tuple[str, int, str, str], PDBAtom]:
    result: dict[tuple[str, int, str, str], PDBAtom] = {}
    for atom in atoms:
        if atom.is_hydrogen or atom.altloc not in {"", "A"}:
            continue
        if backbone_only and not atom.is_backbone:
            continue
        if residue_filter is not None and atom.residue_id not in residue_filter:
            continue
        result.setdefault(atom.match_key, atom)
    return result


def _matched(
    native: Sequence[PDBAtom],
    model: Sequence[PDBAtom],
    *,
    backbone_only: bool,
    residue_filter: set[ResidueId] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    native_map = _atom_map(
        native, backbone_only=backbone_only, residue_filter=residue_filter
    )
    model_map = _atom_map(
        model, backbone_only=backbone_only, residue_filter=residue_filter
    )
    keys = sorted(native_map.keys() & model_map.keys())
    return (
        np.asarray([native_map[key].coord for key in keys], dtype=float),
        np.asarray([model_map[key].coord for key in keys], dtype=float),
    )


def _kabsch(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Kabsch inputs must have equal shape (N, 3)")
    if moving.shape[0] < 3:
        raise ValueError("at least three receptor atoms are required for alignment")
    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    covariance = (moving - moving_center).T @ (reference - reference_center)
    left, _, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transposed) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transposed
    translation = reference_center - (moving_center @ rotation)
    return rotation, translation


def _transform(
    atoms: Sequence[PDBAtom], rotation: np.ndarray, translation: np.ndarray
) -> list[PDBAtom]:
    return _with_coords(atoms, (_coords(atoms) @ rotation) + translation.reshape(1, 3))


def _rmsd(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size == 0:
        raise ValueError("RMSD inputs must be non-empty and equally shaped")
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=1))))


def _heavy_by_residue(atoms: Sequence[PDBAtom]) -> dict[ResidueId, np.ndarray]:
    grouped: dict[ResidueId, list[tuple[float, float, float]]] = {}
    for atom in atoms:
        if atom.is_hydrogen or atom.altloc not in {"", "A"}:
            continue
        grouped.setdefault(atom.residue_id, []).append(atom.coord)
    return {
        residue: np.asarray(values, dtype=float)
        for residue, values in grouped.items()
        if values
    }


def _contacts(
    receptor: Sequence[PDBAtom], ligand: Sequence[PDBAtom], cutoff: float
) -> set[tuple[ResidueId, ResidueId]]:
    if cutoff <= 0:
        raise ValueError("contact cutoff must be positive")
    receptor_residues = _heavy_by_residue(receptor)
    ligand_residues = _heavy_by_residue(ligand)
    cutoff_squared = cutoff**2
    result: set[tuple[ResidueId, ResidueId]] = set()
    for receptor_id, receptor_coords in receptor_residues.items():
        for ligand_id, ligand_coords in ligand_residues.items():
            delta = receptor_coords[:, None, :] - ligand_coords[None, :, :]
            if bool(np.any(np.sum(delta * delta, axis=2) < cutoff_squared)):
                result.add((receptor_id, ligand_id))
    return result


def _interface_contacts(
    receptor: Sequence[PDBAtom], ligand: Sequence[PDBAtom], cutoff: float
) -> set[tuple[ResidueId, ResidueId]]:
    def representatives(atoms: Sequence[PDBAtom]) -> dict[ResidueId, np.ndarray]:
        residues: dict[ResidueId, dict[str, PDBAtom]] = {}
        for atom in atoms:
            if atom.is_hydrogen or atom.altloc not in {"", "A"}:
                continue
            residues.setdefault(atom.residue_id, {}).setdefault(
                atom.atom_name.upper(), atom
            )
        result: dict[ResidueId, np.ndarray] = {}
        for residue, atom_map in residues.items():
            representative = atom_map.get("CB") or atom_map.get("CA")
            if representative is not None:
                result[residue] = np.asarray(representative.coord, dtype=float)
        return result

    receptor_residues = representatives(receptor)
    ligand_residues = representatives(ligand)
    cutoff_squared = cutoff**2
    result: set[tuple[ResidueId, ResidueId]] = set()
    for receptor_id, receptor_coord in receptor_residues.items():
        for ligand_id, ligand_coord in ligand_residues.items():
            delta = receptor_coord - ligand_coord
            if float(delta @ delta) < cutoff_squared:
                result.add((receptor_id, ligand_id))
    return result


def _dockq(fnat: float, irmsd: float, lrmsd: float) -> float:
    bounded_fnat = min(1.0, max(0.0, float(fnat)))
    return float(
        (
            bounded_fnat
            + 1.0 / (1.0 + (float(irmsd) / 1.5) ** 2)
            + 1.0 / (1.0 + (float(lrmsd) / 8.5) ** 2)
        )
        / 3.0
    )


def _category(value: float) -> str:
    if value >= 0.80:
        return "high"
    if value >= 0.49:
        return "medium"
    if value >= 0.23:
        return "acceptable"
    return "incorrect"


def _pdb_line(atom: PDBAtom, serial: int) -> str:
    line = atom.source_line.ljust(80)
    x, y, z = atom.coord
    return (
        line[:6]
        + f"{serial:5d}"
        + line[11:30]
        + f"{x:8.3f}{y:8.3f}{z:8.3f}"
        + line[54:80]
    ).rstrip()


def _write_complex(
    path: Path, receptor: Sequence[PDBAtom], ligand: Sequence[PDBAtom]
) -> None:
    lines: list[str] = []
    serial = 1
    for atom in receptor:
        lines.append(_pdb_line(atom, serial))
        serial += 1
    lines.append("TER")
    for atom in ligand:
        lines.append(_pdb_line(atom, serial))
        serial += 1
    lines.extend(("TER", "END"))
    _atomic_text(path, "\n".join(lines) + "\n")


def prepare_reduced_pose(
    *,
    receptor_model_pdb: str | Path,
    ligand_template_pdb: str | Path,
    receptor_chain: str,
    ligand_chain: str,
    pose_json: str | Path,
    output_dir: str | Path,
) -> Path:
    """Persist the computed complex in the search coordinate frame.

    Only the receptor model, ligand template and computed pose are read. The
    ligand atom order is the engine loader order; no experimental reference,
    alignment, DockQ result or FULL execution participates in this operation.
    """

    if receptor_chain == ligand_chain:
        raise ValueError("receptor and ligand chains must be different")
    receptor_path = Path(receptor_model_pdb).resolve(strict=True)
    ligand_path = Path(ligand_template_pdb).resolve(strict=True)
    pose_path = Path(pose_json).resolve(strict=True)
    destination = Path(output_dir).resolve(strict=False)
    complex_path = destination / "reduced_complex.pdb"
    if complex_path.resolve(strict=False) in {receptor_path, ligand_path, pose_path}:
        raise ValueError("prepared complex must not overwrite its inputs")

    receptor_all = read_engine_order_atoms(receptor_path)
    ligand_all = read_engine_order_atoms(ligand_path)
    receptor = _only_chain(receptor_all, receptor_chain, role="model receptor")
    ligand = _only_chain(ligand_all, ligand_chain, role="ligand template")
    if len(receptor) != len(receptor_all):
        raise ValueError("model receptor PDB contains unexpected chains")
    if len(ligand) != len(ligand_all):
        raise ValueError("ligand template PDB contains unexpected chains")
    pose_coords, _ = _load_pose(pose_path, len(ligand))
    _write_complex(complex_path, receptor, _with_coords(ligand, pose_coords))
    return complex_path


def evaluate_reference_pose(
    *,
    receptor_model_pdb: str | Path,
    ligand_template_pdb: str | Path,
    reference_complex_pdb: str | Path,
    receptor_chain: str,
    ligand_chain: str,
    pose_json: str | Path,
    output_dir: str | Path,
    contact_cutoff_angstrom: float = 4.0,
    interface_cutoff_angstrom: float = 8.0,
) -> ReferenceDockQResult:
    """Evaluate a persisted pose against the untouched experimental complex.

    The reference is used only here, after the search has finished.
    """

    if receptor_chain == ligand_chain:
        raise ValueError("receptor and ligand chains must be different")
    receptor_path = Path(receptor_model_pdb).resolve(strict=True)
    ligand_path = Path(ligand_template_pdb).resolve(strict=True)
    reference_path = Path(reference_complex_pdb).resolve(strict=True)
    pose_path = Path(pose_json).resolve(strict=True)
    destination = Path(output_dir).resolve(strict=False)

    model_receptor_all = read_engine_order_atoms(receptor_path)
    ligand_template_all = read_engine_order_atoms(ligand_path)
    reference_all = read_engine_order_atoms(reference_path)
    model_receptor = _only_chain(
        model_receptor_all, receptor_chain, role="model receptor"
    )
    ligand_template = _only_chain(
        ligand_template_all, ligand_chain, role="ligand template"
    )
    native_receptor = _only_chain(
        reference_all, receptor_chain, role="native receptor"
    )
    native_ligand = _only_chain(reference_all, ligand_chain, role="native ligand")
    if len(model_receptor) != len(model_receptor_all):
        raise ValueError("model receptor PDB contains unexpected chains")
    if len(ligand_template) != len(ligand_template_all):
        raise ValueError("ligand template PDB contains unexpected chains")
    reference_chains = {atom.chain_id for atom in reference_all}
    if reference_chains != {receptor_chain, ligand_chain}:
        raise ValueError(
            "reference complex must contain exactly the receptor and ligand chains"
        )

    pose_coords, pose_provenance = _load_pose(pose_path, len(ligand_template))
    model_ligand = _with_coords(ligand_template, pose_coords)

    native_alignment, model_alignment = _matched(
        native_receptor, model_receptor, backbone_only=True
    )
    if native_alignment.shape[0] < 3:
        raise ValueError("insufficient matched receptor backbone atoms")
    rotation, translation = _kabsch(model_alignment, native_alignment)
    aligned_receptor = _transform(model_receptor, rotation, translation)
    aligned_ligand = _transform(model_ligand, rotation, translation)

    native_ligand_coords, model_ligand_coords = _matched(
        native_ligand, aligned_ligand, backbone_only=True
    )
    if native_ligand_coords.shape[0] == 0:
        raise ValueError("no matched ligand backbone atoms")
    lrmsd = _rmsd(model_ligand_coords, native_ligand_coords)

    native_contacts = _contacts(
        native_receptor, native_ligand, contact_cutoff_angstrom
    )
    if not native_contacts:
        raise ValueError("reference complex has no contacts at the selected cutoff")
    model_contacts = _contacts(
        aligned_receptor, aligned_ligand, contact_cutoff_angstrom
    )
    recovered = native_contacts & model_contacts
    fnat = float(len(recovered) / len(native_contacts))

    interface_pairs = _interface_contacts(
        native_receptor, native_ligand, interface_cutoff_angstrom
    )
    receptor_interface = {pair[0] for pair in interface_pairs}
    ligand_interface = {pair[1] for pair in interface_pairs}
    native_rec_interface, model_rec_interface = _matched(
        native_receptor,
        aligned_receptor,
        backbone_only=True,
        residue_filter=receptor_interface,
    )
    native_lig_interface, model_lig_interface = _matched(
        native_ligand,
        aligned_ligand,
        backbone_only=True,
        residue_filter=ligand_interface,
    )
    native_interface = np.vstack((native_rec_interface, native_lig_interface))
    model_interface = np.vstack((model_rec_interface, model_lig_interface))
    if native_interface.shape[0] < 3:
        raise ValueError("insufficient matched interface backbone atoms")
    interface_rotation, interface_translation = _kabsch(
        model_interface, native_interface
    )
    irmsd = _rmsd(
        (model_interface @ interface_rotation) + interface_translation,
        native_interface,
    )
    dockq = _dockq(fnat, irmsd, lrmsd)
    category = _category(dockq)

    complex_path = destination / "predicted_complex.pdb"
    metrics_path = destination / "reference_dockq.json"
    _write_complex(complex_path, aligned_receptor, aligned_ligand)
    payload = {
        "schema_version": "ppi4dock_reference_dockq_v1",
        "reference_used_for_search": False,
        "coordinate_frame": "experimental_reference_after_receptor_kabsch",
        "roles": {
            "receptor_chain": receptor_chain,
            "ligand_chain": ligand_chain,
        },
        "inputs": {
            "receptor_model_pdb": str(receptor_path),
            "receptor_model_sha256": _sha256(receptor_path),
            "ligand_template_pdb": str(ligand_path),
            "ligand_template_sha256": _sha256(ligand_path),
            "reference_complex_pdb": str(reference_path),
            "reference_complex_sha256": _sha256(reference_path),
            "pose_json": str(pose_path),
            "pose_json_sha256": _sha256(pose_path),
            "pose_provenance": pose_provenance,
        },
        "alignment": {
            "rotation": rotation.tolist(),
            "translation": translation.tolist(),
            "receptor_backbone_atoms": int(native_alignment.shape[0]),
        },
        "metrics": {
            "dockq": dockq,
            "dockq_category": category,
            "fnat": fnat,
            "irmsd": irmsd,
            "lrmsd": lrmsd,
            "native_contacts": len(native_contacts),
            "model_contacts": len(model_contacts),
            "recovered_native_contacts": len(recovered),
            "receptor_alignment_atoms": int(native_alignment.shape[0]),
            "ligand_rmsd_atoms": int(native_ligand_coords.shape[0]),
            "interface_rmsd_atoms": int(native_interface.shape[0]),
            "contact_cutoff_angstrom": float(contact_cutoff_angstrom),
            "interface_cutoff_angstrom": float(interface_cutoff_angstrom),
        },
        "outputs": {"predicted_complex_pdb": str(complex_path)},
    }
    _atomic_text(
        metrics_path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return ReferenceDockQResult(
        dockq=dockq,
        fnat=fnat,
        irmsd=irmsd,
        lrmsd=lrmsd,
        dockq_category=category,
        native_contacts=len(native_contacts),
        model_contacts=len(model_contacts),
        recovered_native_contacts=len(recovered),
        receptor_alignment_atoms=int(native_alignment.shape[0]),
        ligand_rmsd_atoms=int(native_ligand_coords.shape[0]),
        interface_rmsd_atoms=int(native_interface.shape[0]),
        complex_pdb_path=complex_path,
        metrics_path=metrics_path,
    )
