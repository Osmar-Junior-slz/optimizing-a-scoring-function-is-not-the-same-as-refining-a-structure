"""Validation for receptor/peptide PDB pairs before docking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


RECEPTOR_PEPTIDE_OVERLAP = "receptor_peptide_overlap"
RECEPTOR_PEPTIDE_INVALID_PDB = "receptor_peptide_invalid_pdb"


@dataclass(frozen=True)
class ReceptorPeptidePairValidation:
    """Auditable preflight result for one receptor/peptide input pair."""

    receptor_source: str
    peptide_source: str
    is_valid: bool
    error_code: str | None
    message: str
    receptor_atom_count: int
    receptor_hetatm_count: int
    peptide_atom_count: int
    peptide_hetatm_count: int
    peptide_chain_ids: tuple[str, ...]
    receptor_chain_ids: tuple[str, ...]
    peptide_chain_atom_records_in_receptor: int
    peptide_chain_hetatm_records_in_receptor: int
    duplicate_atom_identity_count: int
    duplicate_coordinate_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "receptor_source": self.receptor_source,
            "peptide_source": self.peptide_source,
            "is_valid": self.is_valid,
            "error_code": self.error_code,
            "message": self.message,
            "receptor_atom_count": self.receptor_atom_count,
            "receptor_hetatm_count": self.receptor_hetatm_count,
            "peptide_atom_count": self.peptide_atom_count,
            "peptide_hetatm_count": self.peptide_hetatm_count,
            "peptide_chain_ids": list(self.peptide_chain_ids),
            "receptor_chain_ids": list(self.receptor_chain_ids),
            "peptide_chain_atom_records_in_receptor": (
                self.peptide_chain_atom_records_in_receptor
            ),
            "peptide_chain_hetatm_records_in_receptor": (
                self.peptide_chain_hetatm_records_in_receptor
            ),
            "duplicate_atom_identity_count": self.duplicate_atom_identity_count,
            "duplicate_coordinate_count": self.duplicate_coordinate_count,
        }


@dataclass(frozen=True)
class _PDBRecord:
    record_name: str
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: str
    insertion_code: str
    coord: tuple[float, float, float]

    @property
    def atom_identity(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
            self.atom_name,
            self.altloc,
        )

    @property
    def rounded_coord(self) -> tuple[float, float, float]:
        return (
            round(self.coord[0], 3),
            round(self.coord[1], 3),
            round(self.coord[2], 3),
        )


def _record_name(line: str) -> str:
    return line[:6].strip().upper()


def _first_model_lines(lines: list[str]) -> list[str]:
    first_model_index = next(
        (idx for idx, line in enumerate(lines) if _record_name(line) == "MODEL"),
        None,
    )
    if first_model_index is None:
        return lines

    selected: list[str] = []
    for line in lines[first_model_index + 1 :]:
        record = _record_name(line)
        if record in {"ENDMDL", "MODEL"}:
            break
        selected.append(line)
    return selected


def _parse_records(lines: Iterable[str]) -> list[_PDBRecord]:
    records: list[_PDBRecord] = []
    for line in _first_model_lines([str(line).rstrip("\r\n") for line in lines]):
        record = _record_name(line)
        if record == "END":
            break
        if record not in {"ATOM", "HETATM"}:
            continue
        try:
            coord = (
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            )
        except (TypeError, ValueError):
            continue
        records.append(
            _PDBRecord(
                record_name=record,
                atom_name=line[12:16].strip().upper(),
                altloc=line[16:17].strip().upper(),
                residue_name=line[17:20].strip().upper(),
                chain_id=line[21:22].strip(),
                residue_number=line[22:26].strip(),
                insertion_code=line[26:27].strip(),
                coord=coord,
            )
        )
    return records


def _invalid_result(
    *,
    receptor_source: str,
    peptide_source: str,
    message: str,
    receptor_records: list[_PDBRecord] | None = None,
    peptide_records: list[_PDBRecord] | None = None,
) -> ReceptorPeptidePairValidation:
    receptor_records = receptor_records or []
    peptide_records = peptide_records or []
    return ReceptorPeptidePairValidation(
        receptor_source=receptor_source,
        peptide_source=peptide_source,
        is_valid=False,
        error_code=RECEPTOR_PEPTIDE_INVALID_PDB,
        message=message,
        receptor_atom_count=sum(r.record_name == "ATOM" for r in receptor_records),
        receptor_hetatm_count=sum(r.record_name == "HETATM" for r in receptor_records),
        peptide_atom_count=sum(r.record_name == "ATOM" for r in peptide_records),
        peptide_hetatm_count=sum(r.record_name == "HETATM" for r in peptide_records),
        peptide_chain_ids=tuple(
            sorted({r.chain_id for r in peptide_records if r.record_name == "ATOM"})
        ),
        receptor_chain_ids=tuple(
            sorted({r.chain_id for r in receptor_records if r.record_name == "ATOM"})
        ),
        peptide_chain_atom_records_in_receptor=0,
        peptide_chain_hetatm_records_in_receptor=0,
        duplicate_atom_identity_count=0,
        duplicate_coordinate_count=0,
    )


def validate_receptor_peptide_pair_text(
    receptor_pdb_text: str,
    peptide_pdb_text: str,
    *,
    receptor_source: str = "<receptor>",
    peptide_source: str = "<peptide>",
) -> ReceptorPeptidePairValidation:
    """Validate that a receptor PDB does not still contain the peptide partner."""

    receptor_records = _parse_records(receptor_pdb_text.splitlines())
    peptide_records = _parse_records(peptide_pdb_text.splitlines())
    if not receptor_records:
        return _invalid_result(
            receptor_source=receptor_source,
            peptide_source=peptide_source,
            message="Receptor has no readable ATOM/HETATM records in the first model.",
            peptide_records=peptide_records,
        )
    if not peptide_records:
        return _invalid_result(
            receptor_source=receptor_source,
            peptide_source=peptide_source,
            message="Peptide has no readable ATOM/HETATM records in the first model.",
            receptor_records=receptor_records,
        )

    receptor_atom_records = [r for r in receptor_records if r.record_name == "ATOM"]
    receptor_hetatm_records = [r for r in receptor_records if r.record_name == "HETATM"]
    peptide_atom_records = [r for r in peptide_records if r.record_name == "ATOM"]
    peptide_hetatm_records = [r for r in peptide_records if r.record_name == "HETATM"]
    peptide_chain_ids = tuple(sorted({r.chain_id for r in peptide_atom_records}))
    receptor_chain_ids = tuple(sorted({r.chain_id for r in receptor_atom_records}))

    peptide_chain_atom_records_in_receptor = sum(
        1 for r in receptor_atom_records if r.chain_id in peptide_chain_ids
    )
    peptide_chain_hetatm_records_in_receptor = sum(
        1 for r in receptor_hetatm_records if r.chain_id in peptide_chain_ids
    )
    duplicate_atom_identity_count = len(
        {r.atom_identity for r in receptor_records}
        & {r.atom_identity for r in peptide_records}
    )
    duplicate_coordinate_count = len(
        {r.rounded_coord for r in receptor_records}
        & {r.rounded_coord for r in peptide_records}
    )
    is_valid = (
        peptide_chain_atom_records_in_receptor == 0
        and peptide_chain_hetatm_records_in_receptor == 0
        and duplicate_atom_identity_count == 0
        and duplicate_coordinate_count == 0
    )
    message = (
        "Receptor and peptide are separated: no peptide chain or duplicate atoms found in receptor."
        if is_valid
        else (
            "Receptor still contains the peptide or overlapping atoms. "
            "Use a receptor file with the peptide chain removed before docking."
        )
    )
    return ReceptorPeptidePairValidation(
        receptor_source=receptor_source,
        peptide_source=peptide_source,
        is_valid=is_valid,
        error_code=None if is_valid else RECEPTOR_PEPTIDE_OVERLAP,
        message=message,
        receptor_atom_count=len(receptor_atom_records),
        receptor_hetatm_count=len(receptor_hetatm_records),
        peptide_atom_count=len(peptide_atom_records),
        peptide_hetatm_count=len(peptide_hetatm_records),
        peptide_chain_ids=peptide_chain_ids,
        receptor_chain_ids=receptor_chain_ids,
        peptide_chain_atom_records_in_receptor=peptide_chain_atom_records_in_receptor,
        peptide_chain_hetatm_records_in_receptor=peptide_chain_hetatm_records_in_receptor,
        duplicate_atom_identity_count=duplicate_atom_identity_count,
        duplicate_coordinate_count=duplicate_coordinate_count,
    )


def validate_receptor_peptide_pair(
    receptor_path: str | Path,
    peptide_path: str | Path,
) -> ReceptorPeptidePairValidation:
    """Read and validate one receptor/peptide pair from disk."""

    receptor = Path(receptor_path)
    peptide = Path(peptide_path)
    try:
        receptor_text = receptor.read_text(encoding="utf-8", errors="ignore")
        peptide_text = peptide.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return _invalid_result(
            receptor_source=str(receptor),
            peptide_source=str(peptide),
            message=f"Could not read receptor/peptide PDB inputs: {exc}.",
        )
    return validate_receptor_peptide_pair_text(
        receptor_text,
        peptide_text,
        receptor_source=str(receptor),
        peptide_source=str(peptide),
    )


__all__ = [
    "RECEPTOR_PEPTIDE_INVALID_PDB",
    "RECEPTOR_PEPTIDE_OVERLAP",
    "ReceptorPeptidePairValidation",
    "validate_receptor_peptide_pair",
    "validate_receptor_peptide_pair_text",
]
