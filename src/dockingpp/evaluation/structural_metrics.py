"""Reference-based structural metrics for protein-peptide docking poses."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


BACKBONE_ATOMS = frozenset({"N", "CA", "C", "O"})


@dataclass(frozen=True, order=True)
class ResidueId:
    """Stable residue identifier inside one docking partner."""

    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str


@dataclass(frozen=True)
class PDBAtom:
    """Minimal atom record required by CAPRI-style structural metrics."""

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

    @property
    def residue_id(self) -> ResidueId:
        return ResidueId(
            chain_id=self.chain_id,
            residue_number=self.residue_number,
            insertion_code=self.insertion_code,
            residue_name=self.residue_name,
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
        return self.atom_name.upper() in BACKBONE_ATOMS


@dataclass(frozen=True)
class StructuralMetrics:
    """CAPRI-compatible measurements for one predicted complex."""

    ligand_rmsd: float
    interface_rmsd: float
    fnat: float
    dockq: float
    dockq_category: str
    native_contacts: int
    model_contacts: int
    recovered_native_contacts: int
    receptor_alignment_atoms: int
    ligand_rmsd_atoms: int
    interface_rmsd_atoms: int
    contact_cutoff_angstrom: float
    interface_cutoff_angstrom: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "ligand_rmsd": self.ligand_rmsd,
            "lrmsd": self.ligand_rmsd,
            "interface_rmsd": self.interface_rmsd,
            "irmsd": self.interface_rmsd,
            "fnat": self.fnat,
            "dockq": self.dockq,
            "dockq_category": self.dockq_category,
            "native_contacts": self.native_contacts,
            "model_contacts": self.model_contacts,
            "recovered_native_contacts": self.recovered_native_contacts,
            "receptor_alignment_atoms": self.receptor_alignment_atoms,
            "ligand_rmsd_atoms": self.ligand_rmsd_atoms,
            "interface_rmsd_atoms": self.interface_rmsd_atoms,
            "contact_cutoff_angstrom": self.contact_cutoff_angstrom,
            "interface_cutoff_angstrom": self.interface_cutoff_angstrom,
        }


def _guess_element(atom_name: str) -> str:
    name = "".join(char for char in atom_name.strip() if char.isalpha())
    if not name:
        return ""
    if len(name) >= 2 and name[:2].upper() in {"CL", "BR"}:
        return name[:2].upper()
    return name[0].upper()


def read_pdb_atoms(path: str | Path) -> list[PDBAtom]:
    """Read ATOM/HETATM records from the first model of a PDB file."""

    atoms: list[PDBAtom] = []
    in_first_model = True
    found_model = False

    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            record = line[:6].strip().upper()
            if record == "MODEL":
                if found_model:
                    break
                found_model = True
                in_first_model = True
                continue
            if record == "ENDMDL" and found_model:
                break
            if found_model and not in_first_model:
                continue
            if record not in {"ATOM", "HETATM"}:
                continue

            try:
                serial = int(line[6:11].strip() or len(atoms) + 1)
                residue_number = int(line[22:26].strip())
                coord = (
                    float(line[30:38]),
                    float(line[38:46]),
                    float(line[46:54]),
                )
            except (TypeError, ValueError):
                continue

            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) >= 78 else ""
            atoms.append(
                PDBAtom(
                    record_name=record,
                    serial=serial,
                    atom_name=atom_name,
                    altloc=line[16:17].strip(),
                    residue_name=line[17:20].strip(),
                    chain_id=line[21:22].strip(),
                    residue_number=residue_number,
                    insertion_code=line[26:27].strip(),
                    coord=coord,
                    element=(element or _guess_element(atom_name)).upper(),
                )
            )

    if not atoms:
        raise ValueError(f"No PDB atoms found in {path}")
    return atoms


def atoms_with_coordinates(
    template_atoms: Sequence[PDBAtom],
    coordinates: Sequence[Sequence[float]] | np.ndarray,
) -> list[PDBAtom]:
    """Copy atom metadata while replacing coordinates in the same atom order."""

    coords = np.asarray(coordinates, dtype=float)
    if coords.shape != (len(template_atoms), 3):
        raise ValueError(
            "Pose coordinates must match the peptide template atom order: "
            f"expected {(len(template_atoms), 3)}, got {coords.shape}"
        )
    return [
        replace(atom, coord=tuple(float(value) for value in coords[index]))
        for index, atom in enumerate(template_atoms)
    ]


def read_pose_json_coordinates(path: str | Path) -> np.ndarray:
    """Read the coordinate array persisted in artifacts/best_pose.json."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    coords = np.asarray(payload.get("coords"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or coords.shape[0] == 0:
        raise ValueError(f"Invalid pose coordinates in {path}: shape={coords.shape}")
    return coords


def _coords(atoms: Sequence[PDBAtom]) -> np.ndarray:
    return np.asarray([atom.coord for atom in atoms], dtype=float)


def _atom_map(
    atoms: Iterable[PDBAtom],
    *,
    backbone_only: bool,
    residue_filter: set[ResidueId] | None = None,
) -> dict[tuple[str, int, str, str], PDBAtom]:
    result: dict[tuple[str, int, str, str], PDBAtom] = {}
    for atom in atoms:
        if atom.is_hydrogen:
            continue
        if backbone_only and not atom.is_backbone:
            continue
        if residue_filter is not None and atom.residue_id not in residue_filter:
            continue
        if atom.altloc not in {"", "A"}:
            continue
        result.setdefault(atom.match_key, atom)
    return result


def _matched_coordinates(
    native_atoms: Sequence[PDBAtom],
    model_atoms: Sequence[PDBAtom],
    *,
    backbone_only: bool,
    residue_filter: set[ResidueId] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    native_map = _atom_map(
        native_atoms,
        backbone_only=backbone_only,
        residue_filter=residue_filter,
    )
    model_map = _atom_map(
        model_atoms,
        backbone_only=backbone_only,
        residue_filter=residue_filter,
    )
    keys = sorted(native_map.keys() & model_map.keys())
    native = np.asarray([native_map[key].coord for key in keys], dtype=float)
    model = np.asarray([model_map[key].coord for key in keys], dtype=float)
    return native, model


def _kabsch_transform(moving: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if moving.shape != reference.shape or moving.ndim != 2 or moving.shape[1] != 3:
        raise ValueError("Kabsch inputs must have equal shape (N, 3)")
    if moving.shape[0] < 3:
        raise ValueError("At least three matched atoms are required for structural alignment")

    moving_center = moving.mean(axis=0)
    reference_center = reference.mean(axis=0)
    moving_centered = moving - moving_center
    reference_centered = reference - reference_center
    covariance = moving_centered.T @ reference_centered
    left, _, right_transposed = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(left @ right_transposed) < 0:
        correction[-1, -1] = -1.0
    rotation = left @ correction @ right_transposed
    translation = reference_center - (moving_center @ rotation)
    return rotation, translation


def _transform_atoms(
    atoms: Sequence[PDBAtom],
    rotation: np.ndarray,
    translation: np.ndarray,
) -> list[PDBAtom]:
    transformed = (_coords(atoms) @ rotation) + translation.reshape(1, 3)
    return atoms_with_coordinates(atoms, transformed)


def _rmsd(first: np.ndarray, second: np.ndarray) -> float:
    if first.shape != second.shape or first.size == 0:
        raise ValueError("RMSD inputs must be non-empty and have equal shape")
    squared_distance = np.sum((first - second) ** 2, axis=1)
    return float(np.sqrt(np.mean(squared_distance)))


def _group_heavy_atoms(atoms: Sequence[PDBAtom]) -> dict[ResidueId, np.ndarray]:
    grouped: dict[ResidueId, list[tuple[float, float, float]]] = {}
    for atom in atoms:
        if atom.is_hydrogen or atom.altloc not in {"", "A"}:
            continue
        grouped.setdefault(atom.residue_id, []).append(atom.coord)
    return {
        residue: np.asarray(residue_coords, dtype=float)
        for residue, residue_coords in grouped.items()
        if residue_coords
    }


def residue_contacts(
    receptor_atoms: Sequence[PDBAtom],
    peptide_atoms: Sequence[PDBAtom],
    *,
    cutoff_angstrom: float = 5.0,
) -> set[tuple[ResidueId, ResidueId]]:
    """Return residue-residue contacts defined by any heavy-atom pair within cutoff."""

    if cutoff_angstrom <= 0:
        raise ValueError("Contact cutoff must be positive")
    receptor_residues = _group_heavy_atoms(receptor_atoms)
    peptide_residues = _group_heavy_atoms(peptide_atoms)
    cutoff_squared = float(cutoff_angstrom) ** 2
    contacts: set[tuple[ResidueId, ResidueId]] = set()

    for receptor_id, receptor_coords in receptor_residues.items():
        for peptide_id, peptide_coords in peptide_residues.items():
            deltas = receptor_coords[:, None, :] - peptide_coords[None, :, :]
            if bool(np.any(np.sum(deltas * deltas, axis=2) < cutoff_squared)):
                contacts.add((receptor_id, peptide_id))
    return contacts


def interface_residue_contacts(
    receptor_atoms: Sequence[PDBAtom],
    peptide_atoms: Sequence[PDBAtom],
    *,
    cutoff_angstrom: float = 8.0,
) -> set[tuple[ResidueId, ResidueId]]:
    """Return CAPRI peptide interface pairs using C-beta (C-alpha for glycine)."""

    if cutoff_angstrom <= 0:
        raise ValueError("Interface cutoff must be positive")

    def representative_atoms(atoms: Sequence[PDBAtom]) -> dict[ResidueId, np.ndarray]:
        residues: dict[ResidueId, dict[str, PDBAtom]] = {}
        for atom in atoms:
            if atom.is_hydrogen or atom.altloc not in {"", "A"}:
                continue
            residues.setdefault(atom.residue_id, {}).setdefault(atom.atom_name.upper(), atom)

        representatives: dict[ResidueId, np.ndarray] = {}
        for residue_id, atom_map in residues.items():
            representative = atom_map.get("CB") or atom_map.get("CA")
            if representative is not None:
                representatives[residue_id] = np.asarray(representative.coord, dtype=float)
        return representatives

    receptor_residues = representative_atoms(receptor_atoms)
    peptide_residues = representative_atoms(peptide_atoms)
    cutoff_squared = float(cutoff_angstrom) ** 2
    contacts: set[tuple[ResidueId, ResidueId]] = set()
    for receptor_id, receptor_coord in receptor_residues.items():
        for peptide_id, peptide_coord in peptide_residues.items():
            delta = receptor_coord - peptide_coord
            if float(delta @ delta) < cutoff_squared:
                contacts.add((receptor_id, peptide_id))
    return contacts


def dockq_score(fnat: float, interface_rmsd: float, ligand_rmsd: float) -> float:
    """Compute the original DockQ continuous score."""

    fnat_value = float(min(1.0, max(0.0, fnat)))
    irms_term = 1.0 / (1.0 + (float(interface_rmsd) / 1.5) ** 2)
    lrms_term = 1.0 / (1.0 + (float(ligand_rmsd) / 8.5) ** 2)
    return float((fnat_value + irms_term + lrms_term) / 3.0)


def dockq_category(score: float) -> str:
    """Map DockQ to the standard CAPRI-like quality categories."""

    value = float(score)
    if value >= 0.80:
        return "high"
    if value >= 0.49:
        return "medium"
    if value >= 0.23:
        return "acceptable"
    return "incorrect"


def evaluate_structures(
    *,
    native_receptor: Sequence[PDBAtom],
    native_peptide: Sequence[PDBAtom],
    model_receptor: Sequence[PDBAtom],
    model_peptide: Sequence[PDBAtom],
    contact_cutoff_angstrom: float = 4.0,
    interface_cutoff_angstrom: float = 8.0,
) -> StructuralMetrics:
    """Evaluate a model against a native protein-peptide complex."""

    native_receptor_alignment, model_receptor_alignment = _matched_coordinates(
        native_receptor,
        model_receptor,
        backbone_only=True,
    )
    if native_receptor_alignment.shape[0] < 3:
        raise ValueError("Insufficient matched receptor backbone atoms for alignment")
    rotation, translation = _kabsch_transform(
        model_receptor_alignment,
        native_receptor_alignment,
    )
    aligned_model_receptor = _transform_atoms(model_receptor, rotation, translation)
    aligned_model_peptide = _transform_atoms(model_peptide, rotation, translation)

    native_ligand_coords, model_ligand_coords = _matched_coordinates(
        native_peptide,
        aligned_model_peptide,
        backbone_only=True,
    )
    if native_ligand_coords.shape[0] == 0:
        raise ValueError("No matched peptide backbone atoms for ligand RMSD")
    ligand_rmsd = _rmsd(model_ligand_coords, native_ligand_coords)

    native_contacts = residue_contacts(
        native_receptor,
        native_peptide,
        cutoff_angstrom=contact_cutoff_angstrom,
    )
    if not native_contacts:
        raise ValueError("The native complex has no receptor-peptide contacts at the selected cutoff")
    model_contacts = residue_contacts(
        aligned_model_receptor,
        aligned_model_peptide,
        cutoff_angstrom=contact_cutoff_angstrom,
    )
    recovered_contacts = native_contacts & model_contacts
    fnat = float(len(recovered_contacts) / len(native_contacts))

    native_interface_contacts = interface_residue_contacts(
        native_receptor,
        native_peptide,
        cutoff_angstrom=interface_cutoff_angstrom,
    )
    receptor_interface = {contact[0] for contact in native_interface_contacts}
    peptide_interface = {contact[1] for contact in native_interface_contacts}
    native_rec_interface, model_rec_interface = _matched_coordinates(
        native_receptor,
        aligned_model_receptor,
        backbone_only=True,
        residue_filter=receptor_interface,
    )
    native_pep_interface, model_pep_interface = _matched_coordinates(
        native_peptide,
        aligned_model_peptide,
        backbone_only=True,
        residue_filter=peptide_interface,
    )
    native_interface = np.vstack((native_rec_interface, native_pep_interface))
    model_interface = np.vstack((model_rec_interface, model_pep_interface))
    if native_interface.shape[0] < 3:
        raise ValueError("Insufficient matched native interface backbone atoms")
    interface_rotation, interface_translation = _kabsch_transform(
        model_interface,
        native_interface,
    )
    aligned_interface = (model_interface @ interface_rotation) + interface_translation
    interface_rmsd = _rmsd(aligned_interface, native_interface)

    dockq = dockq_score(fnat, interface_rmsd, ligand_rmsd)
    return StructuralMetrics(
        ligand_rmsd=ligand_rmsd,
        interface_rmsd=interface_rmsd,
        fnat=fnat,
        dockq=dockq,
        dockq_category=dockq_category(dockq),
        native_contacts=len(native_contacts),
        model_contacts=len(model_contacts),
        recovered_native_contacts=len(recovered_contacts),
        receptor_alignment_atoms=int(native_receptor_alignment.shape[0]),
        ligand_rmsd_atoms=int(native_ligand_coords.shape[0]),
        interface_rmsd_atoms=int(native_interface.shape[0]),
        contact_cutoff_angstrom=float(contact_cutoff_angstrom),
        interface_cutoff_angstrom=float(interface_cutoff_angstrom),
    )


def evaluate_saved_pose(
    *,
    receptor_pdb: str | Path,
    native_peptide_pdb: str | Path,
    pose_json: str | Path,
    contact_cutoff_angstrom: float = 4.0,
    interface_cutoff_angstrom: float = 8.0,
) -> StructuralMetrics:
    """Evaluate one persisted best pose using the input complex as reference."""

    receptor_atoms = read_pdb_atoms(receptor_pdb)
    native_peptide_atoms = read_pdb_atoms(native_peptide_pdb)
    model_peptide_atoms = atoms_with_coordinates(
        native_peptide_atoms,
        read_pose_json_coordinates(pose_json),
    )
    return evaluate_structures(
        native_receptor=receptor_atoms,
        native_peptide=native_peptide_atoms,
        model_receptor=receptor_atoms,
        model_peptide=model_peptide_atoms,
        contact_cutoff_angstrom=contact_cutoff_angstrom,
        interface_cutoff_angstrom=interface_cutoff_angstrom,
    )


def write_complex_pdb(
    path: str | Path,
    receptor_atoms: Sequence[PDBAtom],
    peptide_atoms: Sequence[PDBAtom],
) -> None:
    """Write receptor and peptide atom records as one inspectable PDB complex."""

    output: list[str] = []
    serial = 1
    for partner_index, atoms in enumerate((receptor_atoms, peptide_atoms)):
        for atom in atoms:
            x, y, z = atom.coord
            output.append(
                f"{atom.record_name:<6}{serial:>5} {atom.atom_name:>4}{atom.altloc[:1]:1}"
                f"{atom.residue_name:>3} {atom.chain_id[:1]:1}{atom.residue_number:>4}"
                f"{atom.insertion_code[:1]:1}   {x:>8.3f}{y:>8.3f}{z:>8.3f}"
                f"{1.00:>6.2f}{0.00:>6.2f}          {atom.element:>2}"
            )
            serial += 1
        if partner_index == 0:
            output.append("TER")
    output.append("END")
    Path(path).write_text("\n".join(output) + "\n", encoding="ascii")


def export_saved_pose_complex(
    *,
    receptor_pdb: str | Path,
    native_peptide_pdb: str | Path,
    pose_json: str | Path,
    output_pdb: str | Path,
) -> None:
    """Export a persisted pose JSON as a complete receptor-peptide PDB."""

    receptor_atoms = read_pdb_atoms(receptor_pdb)
    peptide_template = read_pdb_atoms(native_peptide_pdb)
    model_peptide = atoms_with_coordinates(
        peptide_template,
        read_pose_json_coordinates(pose_json),
    )
    write_complex_pdb(output_pdb, receptor_atoms, model_peptide)
