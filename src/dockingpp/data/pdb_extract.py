"""Utilities for extracting ligands from PDB text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from dockingpp.data.pdb_clean import ION_RESNAMES, WATER_RESNAMES
from dockingpp.data.pdb_validation import (
    CANONICAL_BACKBONE_ATOMS,
    DEFAULT_PEPTIDE_MAX_RESIDUES,
    DEFAULT_PEPTIDE_MIN_RESIDUES,
    STANDARD_AMINO_ACIDS,
    format_peptide_residue_range,
    validate_peptide_pdb_lines,
)

# Broader protein vocabulary retained only for the legacy ligand extractor:
# SEC/PYL are not emitted as ligands.  Peptide-chain candidacy below uses the
# stricter STANDARD_AMINO_ACIDS contract shared with pdb_validation.py.
PROTEIN_RESNAMES = {
    *STANDARD_AMINO_ACIDS,
    "SEC",
    "PYL",
}


@dataclass(frozen=True)
class ProteinChainInfo:
    """Auditable peptide-contract summary of one ATOM chain."""

    chain_id: str
    residue_count: int
    atom_count: int
    has_only_standard_residues: bool
    has_complete_backbone: bool
    is_peptide_candidate: bool


@dataclass(frozen=True)
class ReceptorPeptideSplit:
    """Canonical receptor/peptide PDB texts produced from a bound complex."""

    peptide_chain_id: str
    receptor_chain_ids: tuple[str, ...]
    receptor_pdb: str
    peptide_pdb: str
    chains: tuple[ProteinChainInfo, ...]
    validation: "ReceptorPeptideSplitValidation"


@dataclass(frozen=True)
class ReceptorPeptideSplitValidation:
    """Auditable evidence that the separated receptor no longer has the peptide."""

    is_valid: bool
    peptide_chain_id: str
    receptor_chain_ids: tuple[str, ...]
    receptor_atom_count: int
    receptor_hetatm_count: int
    peptide_atom_count: int
    removed_other_peptide_chain_ids: tuple[str, ...]
    removed_other_peptide_atom_count: int
    removed_other_peptide_chain_hetatm_count: int
    removed_peptide_chain_hetatm_count: int
    peptide_chain_atom_records_in_receptor: int
    peptide_chain_hetatm_records_in_receptor: int
    duplicate_atom_identity_count: int
    duplicate_coordinate_count: int
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "is_valid": self.is_valid,
            "peptide_chain_id": self.peptide_chain_id,
            "receptor_chain_ids": list(self.receptor_chain_ids),
            "receptor_atom_count": self.receptor_atom_count,
            "receptor_hetatm_count": self.receptor_hetatm_count,
            "peptide_atom_count": self.peptide_atom_count,
            "removed_other_peptide_chain_ids": list(
                self.removed_other_peptide_chain_ids
            ),
            "removed_other_peptide_atom_count": (
                self.removed_other_peptide_atom_count
            ),
            "removed_other_peptide_chain_hetatm_count": (
                self.removed_other_peptide_chain_hetatm_count
            ),
            "removed_peptide_chain_hetatm_count": (
                self.removed_peptide_chain_hetatm_count
            ),
            "peptide_chain_atom_records_in_receptor": (
                self.peptide_chain_atom_records_in_receptor
            ),
            "peptide_chain_hetatm_records_in_receptor": (
                self.peptide_chain_hetatm_records_in_receptor
            ),
            "duplicate_atom_identity_count": self.duplicate_atom_identity_count,
            "duplicate_coordinate_count": self.duplicate_coordinate_count,
            "message": self.message,
        }


def _parse_ligand_id(line: str) -> str:
    resname = line[17:20].strip()
    chain_id = line[21].strip()
    resseq = line[22:26].strip()
    icode = line[26].strip()
    return f"{resname}_{chain_id or 'X'}_{resseq}{icode or ''}"


def _should_include_resname(resname: str, include_waters: bool, include_ions: bool) -> bool:
    resname_upper = resname.upper()
    if not include_waters and resname_upper in WATER_RESNAMES:
        return False
    if not include_ions and resname_upper in ION_RESNAMES:
        return False
    return True


def _iter_model_lines(lines: Iterable[str]) -> tuple[list[str], str | None, str | None]:
    selected: list[str] = []
    model_header = None
    model_end = None
    found_model = False
    in_model = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        record = line[:6].strip().upper()

        if record == "MODEL":
            if not found_model:
                found_model = True
                in_model = True
                model_header = line
            else:
                break
            continue

        if record == "ENDMDL" and found_model and in_model:
            model_end = line
            in_model = False
            break

        if found_model and not in_model:
            continue

        selected.append(line)

    return selected, model_header, model_end


def _chain_id(line: str) -> str:
    return line[21:22].strip()


def _residue_key(line: str) -> tuple[str, str, str]:
    return (
        line[22:26].strip(),
        line[26:27].strip(),
        line[17:20].strip().upper(),
    )


def _atom_chain_lines(pdb_text: str) -> dict[str, list[str]]:
    """Group all first-model ATOM records by chain without hiding residues.

    Including non-standard ATOM residues here is deliberate: a chain
    containing SEC, PYL or another non-standard residue must be inspected and
    rejected by the shared peptide validator, rather than becoming a false
    candidate after the unsupported residue is silently dropped.
    """

    lines = [line.rstrip("\r\n") for line in pdb_text.splitlines()]
    model_lines, _, _ = _iter_model_lines(lines)
    chains: dict[str, list[str]] = {}
    for line in model_lines:
        if line[:6].strip().upper() != "ATOM":
            continue
        chains.setdefault(_chain_id(line), []).append(line)
    return chains


def _has_only_standard_residues(atom_lines: Iterable[str]) -> bool:
    residue_names = {
        line[17:20].strip().upper()
        for line in atom_lines
        if line[:6].strip().upper() == "ATOM"
    }
    return bool(residue_names) and residue_names <= STANDARD_AMINO_ACIDS


def _has_complete_backbone(atom_lines: Iterable[str]) -> bool:
    atoms_by_residue: dict[tuple[str, str, str], set[str]] = {}
    for line in atom_lines:
        if line[:6].strip().upper() != "ATOM":
            continue
        if line[16:17].strip().upper() not in {"", "A"}:
            continue
        atoms_by_residue.setdefault(_residue_key(line), set()).add(
            line[12:16].strip().upper()
        )
    required = set(CANONICAL_BACKBONE_ATOMS)
    return bool(atoms_by_residue) and all(
        required <= atom_names for atom_names in atoms_by_residue.values()
    )


def _first_model_hetatm_lines(
    pdb_text: str,
    *,
    exclude_chain_id: str | None = None,
    exclude_chain_ids: set[str] | None = None,
) -> list[str]:
    lines = [line.rstrip("\r\n") for line in pdb_text.splitlines()]
    model_lines, _, _ = _iter_model_lines(lines)
    excluded = set(exclude_chain_ids or set())
    if exclude_chain_id is not None:
        excluded.add(exclude_chain_id)
    return [
        line
        for line in model_lines
        if line[:6].strip().upper() == "HETATM"
        and _chain_id(line) not in excluded
    ]


def _count_first_model_hetatm_for_chain(pdb_text: str, chain_id: str) -> int:
    lines = [line.rstrip("\r\n") for line in pdb_text.splitlines()]
    model_lines, _, _ = _iter_model_lines(lines)
    return sum(
        1
        for line in model_lines
        if line[:6].strip().upper() == "HETATM" and _chain_id(line) == chain_id
    )


def _count_first_model_hetatm_for_chains(pdb_text: str, chain_ids: set[str]) -> int:
    lines = [line.rstrip("\r\n") for line in pdb_text.splitlines()]
    model_lines, _, _ = _iter_model_lines(lines)
    return sum(
        1
        for line in model_lines
        if line[:6].strip().upper() == "HETATM" and _chain_id(line) in chain_ids
    )


def _validate_peptide_limits(
    min_peptide_residues: int,
    max_peptide_residues: int,
) -> None:
    if min_peptide_residues < 1:
        raise ValueError("min_peptide_residues must be at least 1")
    if max_peptide_residues < min_peptide_residues:
        raise ValueError(
            "max_peptide_residues must be greater than or equal to "
            "min_peptide_residues"
        )


def inspect_protein_chains(
    pdb_text: str,
    *,
    min_peptide_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES,
    max_peptide_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES,
) -> tuple[ProteinChainInfo, ...]:
    """Inspect ATOM chains from the first PDB model.

    ``is_peptide_candidate`` is true only when the chain satisfies the exact
    standalone-peptide contract from :mod:`pdb_validation`: one chain, only
    standard-amino-acid ``ATOM`` records, complete N/CA/C/O backbone and an
    inclusive residue count inside the configured limits.  ``HETATM`` records
    never become peptide candidates.
    """

    _validate_peptide_limits(min_peptide_residues, max_peptide_residues)
    chains = _atom_chain_lines(pdb_text)
    result: list[ProteinChainInfo] = []
    for chain_id, atom_lines in chains.items():
        validation = validate_peptide_pdb_lines(
            atom_lines,
            source=f"<complex chain {chain_id or '<blank>'}>",
            min_residues=min_peptide_residues,
            max_residues=max_peptide_residues,
        )
        result.append(
            ProteinChainInfo(
                chain_id=chain_id,
                residue_count=validation.residue_count,
                atom_count=len(atom_lines),
                has_only_standard_residues=_has_only_standard_residues(atom_lines),
                has_complete_backbone=_has_complete_backbone(atom_lines),
                is_peptide_candidate=validation.is_valid,
            )
        )
    return tuple(result)


def _format_partner_pdb(chain_lines: Iterable[tuple[str, list[str]]]) -> str:
    output: list[str] = []
    for _, atom_lines in chain_lines:
        if not atom_lines:
            continue
        output.extend(atom_lines)
        output.append("TER")
    output.append("END")
    return "\n".join(output) + "\n"


def _format_receptor_pdb(
    chain_lines: Iterable[tuple[str, list[str]]],
    hetatm_lines: Iterable[str],
) -> str:
    """Format receptor ATOM chains while preserving first-model HETATM.

    Cofactors, crystallographic waters and ions are intentionally retained in
    the separated receptor so preparation does not silently destroy chemical
    context.  A later, explicit receptor-cleaning policy may remove selected
    HETATM classes.  HETATM records are never copied into the peptide partner,
    because that would violate its validation contract.
    """

    output: list[str] = []
    for _, atom_lines in chain_lines:
        if not atom_lines:
            continue
        output.extend(atom_lines)
        output.append("TER")
    output.extend(hetatm_lines)
    output.append("END")
    return "\n".join(output) + "\n"


def _atom_identity(line: str) -> tuple[str, str, str, str, str, str]:
    return (
        line[21:22].strip(),
        line[22:26].strip(),
        line[26:27].strip(),
        line[17:20].strip().upper(),
        line[12:16].strip().upper(),
        line[16:17].strip().upper(),
    )


def _atom_coordinate(line: str) -> tuple[float, float, float] | None:
    try:
        return (
            round(float(line[30:38]), 3),
            round(float(line[38:46]), 3),
            round(float(line[46:54]), 3),
        )
    except (TypeError, ValueError):
        return None


def _separation_validation(
    *,
    receptor_pdb: str,
    peptide_pdb: str,
    peptide_chain_id: str,
    receptor_chain_ids: tuple[str, ...],
    removed_other_peptide_chain_ids: tuple[str, ...],
    removed_other_peptide_atom_count: int,
    removed_other_peptide_chain_hetatm_count: int,
    removed_peptide_chain_hetatm_count: int,
) -> ReceptorPeptideSplitValidation:
    receptor_records = [
        line
        for line in receptor_pdb.splitlines()
        if line[:6].strip().upper() in {"ATOM", "HETATM"}
    ]
    peptide_records = [
        line
        for line in peptide_pdb.splitlines()
        if line[:6].strip().upper() in {"ATOM", "HETATM"}
    ]
    receptor_atom_records = [
        line for line in receptor_records if line[:6].strip().upper() == "ATOM"
    ]
    receptor_hetatm_records = [
        line for line in receptor_records if line[:6].strip().upper() == "HETATM"
    ]
    peptide_atom_records = [
        line for line in peptide_records if line[:6].strip().upper() == "ATOM"
    ]

    receptor_identities = {_atom_identity(line) for line in receptor_records}
    peptide_identities = {_atom_identity(line) for line in peptide_records}
    receptor_coords = {
        coord for coord in (_atom_coordinate(line) for line in receptor_records)
        if coord is not None
    }
    peptide_coords = {
        coord for coord in (_atom_coordinate(line) for line in peptide_records)
        if coord is not None
    }
    peptide_chain_atom_records_in_receptor = sum(
        1
        for line in receptor_atom_records
        if _chain_id(line) == peptide_chain_id
    )
    peptide_chain_hetatm_records_in_receptor = sum(
        1
        for line in receptor_hetatm_records
        if _chain_id(line) == peptide_chain_id
    )
    duplicate_atom_identity_count = len(receptor_identities & peptide_identities)
    duplicate_coordinate_count = len(receptor_coords & peptide_coords)
    is_valid = (
        peptide_chain_atom_records_in_receptor == 0
        and peptide_chain_hetatm_records_in_receptor == 0
        and duplicate_atom_identity_count == 0
        and duplicate_coordinate_count == 0
        and bool(receptor_atom_records)
        and bool(peptide_atom_records)
    )
    message = (
        "Separated receptor does not contain the selected peptide chain."
        if is_valid
        else "Separated receptor still overlaps with the peptide; do not use for docking."
    )
    return ReceptorPeptideSplitValidation(
        is_valid=is_valid,
        peptide_chain_id=peptide_chain_id,
        receptor_chain_ids=receptor_chain_ids,
        receptor_atom_count=len(receptor_atom_records),
        receptor_hetatm_count=len(receptor_hetatm_records),
        peptide_atom_count=len(peptide_atom_records),
        removed_other_peptide_chain_ids=removed_other_peptide_chain_ids,
        removed_other_peptide_atom_count=removed_other_peptide_atom_count,
        removed_other_peptide_chain_hetatm_count=(
            removed_other_peptide_chain_hetatm_count
        ),
        removed_peptide_chain_hetatm_count=removed_peptide_chain_hetatm_count,
        peptide_chain_atom_records_in_receptor=peptide_chain_atom_records_in_receptor,
        peptide_chain_hetatm_records_in_receptor=(
            peptide_chain_hetatm_records_in_receptor
        ),
        duplicate_atom_identity_count=duplicate_atom_identity_count,
        duplicate_coordinate_count=duplicate_coordinate_count,
        message=message,
    )


def split_receptor_peptide_chains(
    pdb_text: str,
    *,
    peptide_chain_id: str | None = None,
    min_peptide_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES,
    max_peptide_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES,
) -> ReceptorPeptideSplit:
    """Split a bound protein complex into a short peptide and its receptor.

    If ``peptide_chain_id`` is omitted, the shortest chain satisfying the
    shared peptide contract is selected deterministically (chain identifier
    breaks ties).  All other ATOM chains form the receptor.  First-model
    All peptide-candidate chains are removed from the receptor to avoid
    contamination when a complex contains repeated or symmetric peptides; only
    the selected chain is emitted as the peptide partner. HETATM records from
    other chains are preserved in the receptor, while HETATM records that
    share the selected peptide chain are removed. They are never copied into
    the peptide.
    Original atom serials, residue numbering and chain identifiers are
    preserved; only ``TER`` and one final ``END`` are normalized.
    """

    _validate_peptide_limits(min_peptide_residues, max_peptide_residues)

    chains = _atom_chain_lines(pdb_text)
    chain_info = inspect_protein_chains(
        pdb_text,
        min_peptide_residues=min_peptide_residues,
        max_peptide_residues=max_peptide_residues,
    )
    if len(chain_info) < 2:
        raise ValueError(
            "At least two ATOM chains are required to separate "
            "receptor and peptide"
        )

    by_id = {info.chain_id: info for info in chain_info}
    candidates = [
        info
        for info in chain_info
        if info.is_peptide_candidate
    ]
    if peptide_chain_id is None:
        if not candidates:
            raise ValueError(
                "No standard peptide chain found inside the configured residue limits"
            )
        selected = min(candidates, key=lambda info: (info.residue_count, info.chain_id))
    else:
        normalized_chain_id = str(peptide_chain_id).strip()
        if normalized_chain_id not in by_id:
            raise ValueError(
                f"Peptide chain {normalized_chain_id or '<blank>'!r} was not found"
            )
        selected = by_id[normalized_chain_id]
        if not selected.is_peptide_candidate:
            raise ValueError(
                f"Chain {normalized_chain_id or '<blank>'!r} does not satisfy "
                "the standard peptide contract (one ATOM chain, standard "
                "amino acids, complete N/CA/C/O backbone and configured "
                f"{format_peptide_residue_range(min_peptide_residues, max_peptide_residues)} "
                "residue range)"
            )

    removed_peptide_candidate_chain_ids = {
        info.chain_id
        for info in candidates
    }
    removed_other_peptide_chain_ids = tuple(
        info.chain_id
        for info in candidates
        if info.chain_id != selected.chain_id
    )
    receptor_items = [
        (chain_id, atom_lines)
        for chain_id, atom_lines in chains.items()
        if chain_id not in removed_peptide_candidate_chain_ids
    ]
    peptide_items = [(selected.chain_id, chains[selected.chain_id])]
    if not receptor_items:
        raise ValueError("No receptor chains remain after peptide separation")

    receptor_chain_ids = tuple(chain_id for chain_id, _ in receptor_items)
    removed_other_peptide_atom_count = sum(
        len(chains[chain_id])
        for chain_id in removed_other_peptide_chain_ids
        if chain_id in chains
    )
    removed_other_peptide_chain_hetatm_count = _count_first_model_hetatm_for_chains(
        pdb_text,
        set(removed_other_peptide_chain_ids),
    )
    removed_peptide_chain_hetatm_count = _count_first_model_hetatm_for_chain(
        pdb_text,
        selected.chain_id,
    )
    receptor_pdb = _format_receptor_pdb(
        receptor_items,
        _first_model_hetatm_lines(
            pdb_text,
            exclude_chain_ids=removed_peptide_candidate_chain_ids,
        ),
    )
    peptide_pdb = _format_partner_pdb(peptide_items)
    validation = _separation_validation(
        receptor_pdb=receptor_pdb,
        peptide_pdb=peptide_pdb,
        peptide_chain_id=selected.chain_id,
        receptor_chain_ids=receptor_chain_ids,
        removed_other_peptide_chain_ids=removed_other_peptide_chain_ids,
        removed_other_peptide_atom_count=removed_other_peptide_atom_count,
        removed_other_peptide_chain_hetatm_count=(
            removed_other_peptide_chain_hetatm_count
        ),
        removed_peptide_chain_hetatm_count=removed_peptide_chain_hetatm_count,
    )

    if not validation.is_valid:
        raise ValueError(validation.message)

    return ReceptorPeptideSplit(
        peptide_chain_id=selected.chain_id,
        receptor_chain_ids=receptor_chain_ids,
        receptor_pdb=receptor_pdb,
        peptide_pdb=peptide_pdb,
        chains=chain_info,
        validation=validation,
    )


def split_receptor_peptide_by_chain(
    pdb_text: str,
    peptide_chain: str,
    *,
    min_peptide_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES,
    max_peptide_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES,
) -> ReceptorPeptideSplit:
    """Split a complex using the peptide chain explicitly selected by the user."""

    return split_receptor_peptide_chains(
        pdb_text,
        peptide_chain_id=peptide_chain,
        min_peptide_residues=min_peptide_residues,
        max_peptide_residues=max_peptide_residues,
    )


def extract_ligands(
    pdb_text: str,
    include_waters: bool = False,
    include_ions: bool = False,
) -> dict[str, str]:
    """Extract ligands into separate PDB strings indexed by ligand_id."""

    lines = [line.rstrip("\n") for line in pdb_text.splitlines()]
    model_lines, model_header, model_end = _iter_model_lines(lines)
    ligand_lines: dict[str, list[str]] = {}

    for line in model_lines:
        record = line[:6].strip().upper()
        if record not in {"HETATM", "ATOM"}:
            continue

        resname = line[17:20].strip()
        resname_upper = resname.upper()
        if record == "ATOM" and resname_upper in PROTEIN_RESNAMES:
            continue
        if not _should_include_resname(resname, include_waters, include_ions):
            continue

        ligand_id = _parse_ligand_id(line)
        ligand_lines.setdefault(ligand_id, []).append(line)

    extracted: dict[str, str] = {}
    for ligand_id, lines_for_ligand in ligand_lines.items():
        output_lines: list[str] = []
        if model_header:
            output_lines.append(model_header)
        output_lines.extend(lines_for_ligand)
        if model_end:
            output_lines.append(model_end)
        output_lines.append("END")
        extracted[ligand_id] = "\n".join(output_lines) + "\n"

    return extracted
