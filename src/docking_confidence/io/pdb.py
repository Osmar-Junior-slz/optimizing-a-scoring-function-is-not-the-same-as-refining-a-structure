from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from docking_confidence.audit.manifest import sha256_file
from docking_confidence.domain.structure import (
    ALTLOC_POLICY_VERSION,
    PDB_PARSER_SCHEMA_VERSION,
    Atom,
    CanonicalComplexFingerprint,
    ComplexPose,
    PoseProvenance,
    PoseSelection,
    RunBinding,
    Structure,
    StructureSelection,
    canonical_pose_sha256,
    structure_sequence_fingerprint,
)


_WATER_RESIDUES = {"HOH", "WAT", "H2O", "DOD"}
_TWO_LETTER_ELEMENTS = {
    "AC", "AG", "AL", "AM", "AR", "AS", "AT", "AU", "BA", "BE", "BH",
    "BI", "BK", "BR", "CA", "CD", "CE", "CF", "CL", "CM", "CN", "CO",
    "CR", "CS", "CU", "DB", "DS", "DY", "ER", "ES", "EU", "FE", "FL",
    "FM", "FR", "GA", "GD", "GE", "HE", "HF", "HG", "HO", "HS", "IN",
    "IR", "KR", "LA", "LI", "LR", "LU", "LV", "MC", "MD", "MG", "MN",
    "MO", "MT", "NA", "NB", "ND", "NE", "NH", "NI", "NO", "NP", "OS",
    "PA", "PB", "PD", "PM", "PO", "PR", "PT", "PU", "RA", "RB", "RE",
    "RF", "RG", "RH", "RN", "RU", "SB", "SC", "SE", "SG", "SI", "SM",
    "SN", "SR", "TA", "TB", "TC", "TE", "TH", "TI", "TL", "TM", "TS",
    "XE", "YB", "ZN", "ZR",
}


def _field(line: str, start: int, end: int) -> str:
    return line[start:end] if len(line) > start else ""


def _parse_float(text: str, *, field: str, line_number: int) -> float:
    try:
        value = float(text.strip())
    except ValueError as exc:
        raise ValueError(f"invalid PDB {field} at line {line_number}") from exc
    if not np.isfinite(value):
        raise ValueError(f"non-finite PDB {field} at line {line_number}")
    return value


def _infer_element(raw_atom_name: str) -> str:
    """Infer an element using the PDB atom-name alignment convention."""

    padded = raw_atom_name.ljust(4)[:4]
    if padded[:1].isspace() or padded[:1].isdigit():
        letters = "".join(character for character in padded if character.isalpha())
        return letters[:1].upper()
    letters = "".join(character for character in padded[:2] if character.isalpha())
    candidate = letters.upper()
    return candidate if candidate in _TWO_LETTER_ELEMENTS else candidate[:1]


def _is_hydrogen(raw_atom_name: str, element: str) -> bool:
    return element.upper() in {"H", "D"} or _infer_element(raw_atom_name) in {
        "H",
        "D",
    }


def _parse_model_id(line: str, *, line_number: int) -> int:
    try:
        identifier = int(_field(line, 10, 14).strip())
    except ValueError as exc:
        raise ValueError(f"invalid MODEL identifier at line {line_number}") from exc
    if identifier < 1:
        raise ValueError(f"MODEL identifier must be positive at line {line_number}")
    return identifier


def _model_records(lines: list[str]) -> tuple[int, ...]:
    identifiers: list[int] = []
    active: int | None = None
    for line_number, line in enumerate(lines, start=1):
        record = _field(line, 0, 6)
        if record == "MODEL ":
            if active is not None:
                raise ValueError(f"nested MODEL record at line {line_number}")
            active = _parse_model_id(line, line_number=line_number)
            if active in identifiers:
                raise ValueError(f"duplicate MODEL identifier {active}")
            identifiers.append(active)
        elif record == "ENDMDL":
            if active is None:
                raise ValueError(f"ENDMDL without MODEL at line {line_number}")
            active = None
    if active is not None:
        raise ValueError(f"MODEL {active} is missing ENDMDL")
    return tuple(identifiers)


def _occupancy_for_selection(atom: Atom) -> float:
    return 0.0 if atom.occupancy is None else float(atom.occupancy)


def _select_residue_conformers(candidates: list[Atom]) -> tuple[Atom, ...]:
    """Choose one altLoc conformation per residue, with blank atoms shared."""

    sites: dict[tuple[str, int, str], list[Atom]] = {}
    for atom in candidates:
        sites.setdefault(
            (atom.chain_id, atom.residue_number, atom.insertion_code), []
        ).append(atom)

    selected: list[Atom] = []
    for site in sorted(sites):
        choices = sites[site]
        conformer_scores: dict[str, float] = {}
        for atom in choices:
            if atom.altloc:
                conformer_scores[atom.altloc] = conformer_scores.get(
                    atom.altloc, 0.0
                ) + _occupancy_for_selection(atom)
        chosen_altloc = ""
        if conformer_scores:
            chosen_altloc = min(
                conformer_scores,
                key=lambda code: (
                    -conformer_scores[code],
                    0 if code == "A" else 1,
                    code,
                ),
            )

        eligible = [
            atom for atom in choices if atom.altloc in {"", chosen_altloc}
        ]
        conformer_specific = [
            atom for atom in eligible if atom.altloc == chosen_altloc
        ] if chosen_altloc else eligible
        residue_name_scores: dict[str, float] = {}
        for atom in conformer_specific:
            residue_name_scores[atom.residue_name] = residue_name_scores.get(
                atom.residue_name, 0.0
            ) + _occupancy_for_selection(atom)
        if not residue_name_scores:
            raise ValueError(f"no selectable residue conformer at site {site}")
        chosen_residue_name = min(
            residue_name_scores,
            key=lambda name: (-residue_name_scores[name], name),
        )
        eligible = [
            atom for atom in eligible if atom.residue_name == chosen_residue_name
        ]

        by_atom_name: dict[str, list[Atom]] = {}
        for atom in eligible:
            by_atom_name.setdefault(atom.atom_name, []).append(atom)
        for atom_name in sorted(by_atom_name):
            atom_choices = by_atom_name[atom_name]
            chosen_atom = min(
                atom_choices,
                key=lambda atom: (
                    0 if chosen_altloc and atom.altloc == chosen_altloc else 1,
                    -_occupancy_for_selection(atom),
                    atom.altloc,
                    atom.element,
                    atom.xyz,
                    atom.is_hetero,
                ),
            )
            selected.append(chosen_atom)
    return tuple(selected)


def read_pdb(
    path: str | Path,
    *,
    include_hetero: bool = True,
    exclude_waters: bool = False,
    heavy_atoms_only: bool = False,
    model_id: int = 1,
) -> Structure:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() != ".pdb":
        raise ValueError(f"expected .pdb file: {source}")
    if isinstance(model_id, bool) or not isinstance(model_id, int) or model_id < 1:
        raise ValueError("model_id must be a positive integer")
    for field, value in (
        ("include_hetero", include_hetero),
        ("exclude_waters", exclude_waters),
        ("heavy_atoms_only", heavy_atoms_only),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"{field} must be boolean")

    lines = source.read_text(encoding="utf-8", errors="strict").splitlines(
        keepends=True
    )
    model_records = _model_records(lines)
    has_models = bool(model_records)
    if not has_models and model_id != 1:
        raise ValueError(
            f"requested model_id={model_id}, but the PDB has no MODEL records; "
            "an unmodeled PDB is model 1 only"
        )
    if has_models and model_id not in model_records:
        raise ValueError(
            f"requested model_id={model_id} is absent; available MODEL identifiers: "
            f"{list(model_records)}"
        )

    candidates: list[Atom] = []
    current_model: int | None = None
    for line_number, line in enumerate(lines, start=1):
        record = _field(line, 0, 6)
        if record == "MODEL ":
            current_model = _parse_model_id(line, line_number=line_number)
            continue
        if record == "ENDMDL":
            current_model = None
            continue
        if record not in {"ATOM  ", "HETATM"}:
            continue
        if has_models:
            if current_model is None:
                raise ValueError(
                    f"{record.strip()} record outside a MODEL block at line {line_number}"
                )
            if current_model != model_id:
                continue
        if record == "HETATM" and not include_hetero:
            continue

        raw_atom_name = _field(line, 12, 16)
        atom_name = raw_atom_name.strip()
        if not atom_name:
            raise ValueError(f"blank PDB atom name at line {line_number}")
        altloc = _field(line, 16, 17).strip().upper()
        residue_name = _field(line, 17, 20).strip().upper()
        if not residue_name:
            raise ValueError(f"blank PDB residue name at line {line_number}")
        chain_id = _field(line, 21, 22).strip() or "_"
        try:
            residue_number = int(_field(line, 22, 26).strip())
        except ValueError as exc:
            raise ValueError(
                f"invalid PDB residue number at line {line_number}"
            ) from exc
        insertion_code = _field(line, 26, 27).strip()
        x = _parse_float(
            _field(line, 30, 38), field="x coordinate", line_number=line_number
        )
        y = _parse_float(
            _field(line, 38, 46), field="y coordinate", line_number=line_number
        )
        z = _parse_float(
            _field(line, 46, 54), field="z coordinate", line_number=line_number
        )
        occupancy_text = _field(line, 54, 60).strip()
        occupancy = (
            _parse_float(
                occupancy_text, field="occupancy", line_number=line_number
            )
            if occupancy_text
            else None
        )
        element = _field(line, 76, 78).strip().upper() or _infer_element(
            raw_atom_name
        )
        if not element:
            raise ValueError(f"unable to determine element at line {line_number}")
        if exclude_waters and residue_name in _WATER_RESIDUES:
            continue
        if heavy_atoms_only and _is_hydrogen(raw_atom_name, element):
            continue

        candidates.append(
            Atom(
                component="unassigned",
                model_id=model_id,
                chain_id=chain_id,
                residue_number=residue_number,
                insertion_code=insertion_code,
                residue_name=residue_name,
                atom_name=atom_name,
                altloc=altloc,
                element=element,
                occupancy=occupancy,
                xyz=(x, y, z),
                is_hetero=record == "HETATM",
            )
        )

    selected = _select_residue_conformers(candidates)
    if not selected:
        raise ValueError(f"no selected ATOM/HETATM records found in {source.name}")
    selection = StructureSelection(
        model_id=model_id,
        include_hetero=include_hetero,
        exclude_waters=exclude_waters,
        heavy_atoms_only=heavy_atoms_only,
        parser_schema_version=PDB_PARSER_SCHEMA_VERSION,
        altloc_policy=ALTLOC_POLICY_VERSION,
    )
    return Structure(
        atoms=selected,
        source_file_sha256=sha256_file(source),
        selection=selection,
    )


def _normalize_chains(
    chains: set[str] | tuple[str, ...] | list[str], *, component: str
) -> tuple[str, ...]:
    normalized = tuple(sorted({str(chain).strip() or "_" for chain in chains}))
    if not normalized:
        raise ValueError(f"{component} chain set must be non-empty")
    if any(len(chain) != 1 for chain in normalized):
        raise ValueError(f"{component} chain IDs must be one PDB character")
    return normalized


def read_complex_pose(
    path: str | Path,
    *,
    receptor_chains: set[str] | tuple[str, ...] | list[str],
    ligand_chains: set[str] | tuple[str, ...] | list[str],
    run_binding: RunBinding,
    complex_fingerprint: CanonicalComplexFingerprint,
    include_hetero: bool = False,
    exclude_waters: bool = True,
    heavy_atoms_only: bool = True,
    model_id: int = 1,
    pose_id: str | None = None,
    pose_rank: int = 1,
    selection_method: str = "explicit_pose",
    pose_source: str = "single_complex_pdb",
    transform_digest: str | None = None,
) -> ComplexPose:
    receptor = _normalize_chains(receptor_chains, component="receptor")
    ligand = _normalize_chains(ligand_chains, component="ligand")
    if set(receptor) & set(ligand):
        raise ValueError("receptor and ligand chains must be disjoint")

    structure = read_pdb(
        path,
        include_hetero=include_hetero,
        exclude_waters=exclude_waters,
        heavy_atoms_only=heavy_atoms_only,
        model_id=model_id,
    )
    assigned: list[Atom] = []
    for atom in structure.atoms:
        if atom.chain_id in receptor:
            assigned.append(replace(atom, component="receptor"))
        elif atom.chain_id in ligand:
            assigned.append(replace(atom, component="ligand"))
    if not any(atom.component == "receptor" for atom in assigned):
        raise ValueError(f"no atoms found for receptor chains {receptor}")
    if not any(atom.component == "ligand" for atom in assigned):
        raise ValueError(f"no atoms found for ligand chains {ligand}")

    effective_pose_id = (
        str(pose_id).strip() if pose_id is not None else f"selected-pose-{pose_rank}"
    )
    structure_schema, structure_digest = structure_sequence_fingerprint(assigned)
    pose_digest = canonical_pose_sha256(
        assigned,
        structure_selection=structure.selection,
        receptor_chains=receptor,
        ligand_chains=ligand,
        pose_id=effective_pose_id,
        pose_rank=pose_rank,
        selection_method=selection_method,
        structure_sequence_fingerprint_sha256=structure_digest,
    )
    selection = PoseSelection(
        pose_id=effective_pose_id,
        pose_rank=pose_rank,
        selection_method=selection_method,
        receptor_chains=receptor,
        ligand_chains=ligand,
        structure_selection=structure.selection,
        source_file_sha256=structure.source_file_sha256,
        canonical_pose_sha256=pose_digest,
        structure_sequence_fingerprint_schema=structure_schema,
        structure_sequence_fingerprint_sha256=structure_digest,
    )
    return ComplexPose(
        atoms=tuple(assigned),
        receptor_chains=receptor,
        ligand_chains=ligand,
        provenance=PoseProvenance(
            source=pose_source,
            run=run_binding,
            selection=selection,
            complex_fingerprint=complex_fingerprint,
            transform_digest=transform_digest,
        ),
    )


def count_atoms(path: Path) -> int:
    return len(read_pdb(path).atoms)


def validate_pdb(path: Path) -> None:
    read_pdb(path)
