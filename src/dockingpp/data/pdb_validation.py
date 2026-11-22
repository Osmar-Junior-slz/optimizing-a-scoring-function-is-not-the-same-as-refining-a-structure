"""Validation of PDB partners intended to be standard peptide references.

The validator is intentionally independent from the docking pipeline.  It
classifies the first PDB model and reports whether the partner satisfies the
operational definition used by this project: a short chain of standard amino
acids with a complete canonical backbone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from dockingpp.domain.peptide_constraints import (
    DEFAULT_PEPTIDE_MAX_RESIDUES,
    DEFAULT_PEPTIDE_MIN_RESIDUES,
)

STANDARD_PEPTIDE = "standard_peptide"
NON_PEPTIDE = "small_molecule/non_peptide"
INVALID_PARTNER = "invalid"

REFERENCE_PARTNER_NOT_PEPTIDE = "reference_partner_not_peptide"
REFERENCE_PARTNER_INVALID_PDB = "reference_partner_invalid_pdb"
REFERENCE_PARTNER_INCOMPLETE_BACKBONE = "reference_partner_incomplete_backbone"
REFERENCE_PARTNER_MULTIPLE_CHAINS = "reference_partner_multiple_chains"
REFERENCE_PARTNER_RESIDUE_COUNT_OUT_OF_RANGE = (
    "reference_partner_residue_count_out_of_range"
)

PartnerClassification = Literal[
    "standard_peptide",
    "small_molecule/non_peptide",
    "invalid",
]

STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
CANONICAL_BACKBONE_ATOMS = ("N", "CA", "C", "O")
_ATOM_RECORDS = frozenset({"ATOM", "HETATM"})


@dataclass(frozen=True)
class BackboneIssue:
    """Canonical backbone atoms missing from one standard residue."""

    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    missing_atoms: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "residue_number": self.residue_number,
            "insertion_code": self.insertion_code,
            "residue_name": self.residue_name,
            "missing_atoms": list(self.missing_atoms),
        }


@dataclass(frozen=True)
class PeptidePDBValidation:
    """Auditable result of validating one reference partner."""

    source: str
    classification: PartnerClassification
    is_valid: bool
    error_code: str | None
    message: str
    atom_count: int
    atom_record_count: int
    hetatm_record_count: int
    residue_count: int
    chain_ids: tuple[str, ...]
    residue_names: tuple[str, ...]
    nonstandard_residue_names: tuple[str, ...]
    backbone_issues: tuple[BackboneIssue, ...]
    malformed_atom_records: int
    first_model_only: bool
    model_serial: int | None
    min_residues: int
    max_residues: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable audit payload."""

        return {
            "source": self.source,
            "classification": self.classification,
            "is_valid": self.is_valid,
            "error_code": self.error_code,
            "message": self.message,
            "atom_count": self.atom_count,
            "atom_record_count": self.atom_record_count,
            "hetatm_record_count": self.hetatm_record_count,
            "residue_count": self.residue_count,
            "chain_ids": list(self.chain_ids),
            "residue_names": list(self.residue_names),
            "nonstandard_residue_names": list(self.nonstandard_residue_names),
            "backbone_issues": [issue.as_dict() for issue in self.backbone_issues],
            "malformed_atom_records": self.malformed_atom_records,
            "first_model_only": self.first_model_only,
            "model_serial": self.model_serial,
            "operational_residue_range": {
                "min": self.min_residues,
                "max": self.max_residues,
            },
        }


@dataclass(frozen=True)
class _ParsedAtom:
    record_name: str
    atom_name: str
    altloc: str
    residue_name: str
    chain_id: str
    residue_number: int
    insertion_code: str

    @property
    def residue_key(self) -> tuple[str, int, str]:
        return (self.chain_id, self.residue_number, self.insertion_code)


def _validate_operational_range(min_residues: int, max_residues: int) -> None:
    if min_residues < 1:
        raise ValueError("min_residues must be at least 1")
    if max_residues < min_residues:
        raise ValueError("max_residues must be greater than or equal to min_residues")


def resolve_peptide_residue_limits(cfg: Any | None = None) -> tuple[int, int]:
    """Resolve the configured peptide residue-count range."""

    if isinstance(cfg, Mapping):
        min_residues = cfg.get("peptide_min_residues", DEFAULT_PEPTIDE_MIN_RESIDUES)
        max_residues = cfg.get("peptide_max_residues", DEFAULT_PEPTIDE_MAX_RESIDUES)
    else:
        min_residues = getattr(cfg, "peptide_min_residues", DEFAULT_PEPTIDE_MIN_RESIDUES)
        max_residues = getattr(cfg, "peptide_max_residues", DEFAULT_PEPTIDE_MAX_RESIDUES)

    if min_residues is None:
        min_residues = DEFAULT_PEPTIDE_MIN_RESIDUES
    if max_residues is None:
        max_residues = DEFAULT_PEPTIDE_MAX_RESIDUES
    min_residues = int(min_residues)
    max_residues = int(max_residues)
    _validate_operational_range(min_residues, max_residues)
    return min_residues, max_residues


def format_peptide_residue_range(
    min_residues: int,
    max_residues: int,
    *,
    separator: str = "..",
) -> str:
    """Format the peptide residue-count range consistently for messages/UI."""

    min_residues = int(min_residues)
    max_residues = int(max_residues)
    _validate_operational_range(min_residues, max_residues)
    return f"{min_residues}{separator}{max_residues}"


def _record_name(line: str) -> str:
    return line[:6].strip().upper()


def _first_model_lines(lines: list[str]) -> tuple[list[str], int | None]:
    """Select records from only the first explicit MODEL, when present."""

    first_model_index = next(
        (index for index, line in enumerate(lines) if _record_name(line) == "MODEL"),
        None,
    )
    if first_model_index is None:
        return lines, None

    model_line = lines[first_model_index]
    try:
        model_serial = int(model_line[10:14].strip())
    except (TypeError, ValueError):
        model_serial = None

    selected: list[str] = []
    for line in lines[first_model_index + 1 :]:
        record = _record_name(line)
        if record in {"ENDMDL", "MODEL"}:
            break
        selected.append(line)
    return selected, model_serial


def _parse_atom(line: str) -> _ParsedAtom:
    record = _record_name(line)
    if record not in _ATOM_RECORDS:
        raise ValueError("not an atom record")

    atom_name = line[12:16].strip().upper()
    residue_name = line[17:20].strip().upper()
    chain_id = line[21:22].strip()
    insertion_code = line[26:27].strip()
    residue_number = int(line[22:26].strip())

    # Coordinates are not retained, but parsing them prevents truncated or
    # column-shifted records from being accepted as a valid reference.
    float(line[30:38])
    float(line[38:46])
    float(line[46:54])
    if not atom_name or not residue_name:
        raise ValueError("atom name and residue name are required")

    return _ParsedAtom(
        record_name=record,
        atom_name=atom_name,
        altloc=line[16:17].strip().upper(),
        residue_name=residue_name,
        chain_id=chain_id,
        residue_number=residue_number,
        insertion_code=insertion_code,
    )


def _base_result(
    *,
    source: str,
    classification: PartnerClassification,
    is_valid: bool,
    error_code: str | None,
    message: str,
    atoms: Iterable[_ParsedAtom],
    malformed_atom_records: int,
    model_serial: int | None,
    min_residues: int,
    max_residues: int,
    backbone_issues: tuple[BackboneIssue, ...] = (),
) -> PeptidePDBValidation:
    atom_list = list(atoms)
    residue_names_by_key: dict[tuple[str, int, str], str] = {}
    for atom in atom_list:
        residue_names_by_key.setdefault(atom.residue_key, atom.residue_name)

    return PeptidePDBValidation(
        source=source,
        classification=classification,
        is_valid=is_valid,
        error_code=error_code,
        message=message,
        atom_count=len(atom_list),
        atom_record_count=sum(atom.record_name == "ATOM" for atom in atom_list),
        hetatm_record_count=sum(atom.record_name == "HETATM" for atom in atom_list),
        residue_count=len(residue_names_by_key),
        chain_ids=tuple(sorted({atom.chain_id for atom in atom_list})),
        residue_names=tuple(residue_names_by_key.values()),
        nonstandard_residue_names=tuple(
            sorted(
                {
                    residue_name
                    for residue_name in residue_names_by_key.values()
                    if residue_name not in STANDARD_AMINO_ACIDS
                }
            )
        ),
        backbone_issues=backbone_issues,
        malformed_atom_records=malformed_atom_records,
        first_model_only=True,
        model_serial=model_serial,
        min_residues=min_residues,
        max_residues=max_residues,
    )


def validate_peptide_pdb_text(
    pdb_text: str,
    *,
    source: str = "<memory>",
    min_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES,
    max_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES,
) -> PeptidePDBValidation:
    """Classify and validate a PDB partner intended to be a peptide.

    Only the first explicit MODEL is inspected.  When MODEL records are
    absent, the ordinary single-model PDB contents are inspected.
    """

    return validate_peptide_pdb_lines(
        pdb_text.splitlines(),
        source=source,
        min_residues=min_residues,
        max_residues=max_residues,
    )


def validate_peptide_pdb_lines(
    lines: Iterable[str],
    *,
    source: str = "<memory>",
    min_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES,
    max_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES,
) -> PeptidePDBValidation:
    """Validate PDB records supplied as an iterable, inspecting its first model."""

    _validate_operational_range(min_residues, max_residues)
    selected_lines, model_serial = _first_model_lines(
        [str(line).rstrip("\r\n") for line in lines]
    )

    atoms: list[_ParsedAtom] = []
    malformed_atom_records = 0
    for line in selected_lines:
        record = _record_name(line)
        if record == "END":
            break
        if record not in _ATOM_RECORDS:
            continue
        try:
            atoms.append(_parse_atom(line))
        except (TypeError, ValueError):
            malformed_atom_records += 1

    if malformed_atom_records:
        return _base_result(
            source=source,
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_INVALID_PDB,
            message=(
                f"Reference partner contains {malformed_atom_records} malformed "
                "ATOM/HETATM record(s) in the first model."
            ),
            atoms=atoms,
            malformed_atom_records=malformed_atom_records,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    if not atoms:
        return _base_result(
            source=source,
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_INVALID_PDB,
            message="Reference partner has no readable ATOM/HETATM records in the first model.",
            atoms=(),
            malformed_atom_records=0,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    residue_names_by_key: dict[tuple[str, int, str], set[str]] = {}
    residue_atoms: dict[tuple[str, int, str], set[str]] = {}
    for atom in atoms:
        residue_names_by_key.setdefault(atom.residue_key, set()).add(atom.residue_name)
        if atom.altloc in {"", "A"}:
            residue_atoms.setdefault(atom.residue_key, set()).add(atom.atom_name)

    conflicting_residues = [
        key for key, residue_names in residue_names_by_key.items() if len(residue_names) != 1
    ]
    if conflicting_residues:
        return _base_result(
            source=source,
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_INVALID_PDB,
            message="Reference partner has conflicting residue names for the same residue ID.",
            atoms=atoms,
            malformed_atom_records=0,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    residue_name_lookup = {
        key: next(iter(names)) for key, names in residue_names_by_key.items()
    }
    nonstandard_names = {
        residue_name
        for residue_name in residue_name_lookup.values()
        if residue_name not in STANDARD_AMINO_ACIDS
    }
    has_hetatm = any(atom.record_name == "HETATM" for atom in atoms)
    if has_hetatm or nonstandard_names:
        reasons: list[str] = []
        if has_hetatm:
            reasons.append("contains HETATM records")
        if nonstandard_names:
            reasons.append(
                "contains non-standard residue(s): " + ", ".join(sorted(nonstandard_names))
            )
        return _base_result(
            source=source,
            classification=NON_PEPTIDE,
            is_valid=False,
            error_code=REFERENCE_PARTNER_NOT_PEPTIDE,
            message="Reference partner is not a standard peptide: " + "; ".join(reasons) + ".",
            atoms=atoms,
            malformed_atom_records=0,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    chain_ids = {key[0] for key in residue_name_lookup}
    if len(chain_ids) != 1:
        return _base_result(
            source=source,
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_MULTIPLE_CHAINS,
            message=(
                "Reference peptide must contain exactly one ATOM chain; "
                f"found {len(chain_ids)} chains."
            ),
            atoms=atoms,
            malformed_atom_records=0,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    backbone_issues: list[BackboneIssue] = []
    for key, residue_name in residue_name_lookup.items():
        available = residue_atoms.get(key, set())
        missing = tuple(
            atom_name
            for atom_name in CANONICAL_BACKBONE_ATOMS
            if atom_name not in available
        )
        if not missing:
            continue
        chain_id, residue_number, insertion_code = key
        backbone_issues.append(
            BackboneIssue(
                chain_id=chain_id,
                residue_number=residue_number,
                insertion_code=insertion_code,
                residue_name=residue_name,
                missing_atoms=missing,
            )
        )

    if backbone_issues:
        return _base_result(
            source=source,
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_INCOMPLETE_BACKBONE,
            message=(
                f"Reference peptide has {len(backbone_issues)} residue(s) without "
                "the complete canonical N/CA/C/O backbone."
            ),
            atoms=atoms,
            malformed_atom_records=0,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
            backbone_issues=tuple(backbone_issues),
        )

    residue_count = len(residue_name_lookup)
    if not min_residues <= residue_count <= max_residues:
        return _base_result(
            source=source,
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_RESIDUE_COUNT_OUT_OF_RANGE,
            message=(
                f"Reference peptide has {residue_count} residue(s); the operational "
                f"range is {format_peptide_residue_range(min_residues, max_residues)}."
            ),
            atoms=atoms,
            malformed_atom_records=0,
            model_serial=model_serial,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    return _base_result(
        source=source,
        classification=STANDARD_PEPTIDE,
        is_valid=True,
        error_code=None,
        message=(
            f"Reference partner is a standard peptide with {residue_count} "
            "residue(s) and a complete canonical backbone."
        ),
        atoms=atoms,
        malformed_atom_records=0,
        model_serial=model_serial,
        min_residues=min_residues,
        max_residues=max_residues,
    )


def validate_peptide_pdb(
    path: str | Path,
    *,
    min_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES,
    max_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES,
) -> PeptidePDBValidation:
    """Read and validate a peptide-reference PDB without raising for I/O errors."""

    _validate_operational_range(min_residues, max_residues)
    pdb_path = Path(path)
    try:
        pdb_text = pdb_path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return _base_result(
            source=str(pdb_path),
            classification=INVALID_PARTNER,
            is_valid=False,
            error_code=REFERENCE_PARTNER_INVALID_PDB,
            message=f"Could not read reference partner PDB: {exc}.",
            atoms=(),
            malformed_atom_records=0,
            model_serial=None,
            min_residues=min_residues,
            max_residues=max_residues,
        )

    return validate_peptide_pdb_text(
        pdb_text,
        source=str(pdb_path),
        min_residues=min_residues,
        max_residues=max_residues,
    )


__all__ = [
    "BackboneIssue",
    "CANONICAL_BACKBONE_ATOMS",
    "DEFAULT_PEPTIDE_MAX_RESIDUES",
    "DEFAULT_PEPTIDE_MIN_RESIDUES",
    "INVALID_PARTNER",
    "NON_PEPTIDE",
    "PartnerClassification",
    "PeptidePDBValidation",
    "REFERENCE_PARTNER_INCOMPLETE_BACKBONE",
    "REFERENCE_PARTNER_INVALID_PDB",
    "REFERENCE_PARTNER_MULTIPLE_CHAINS",
    "REFERENCE_PARTNER_NOT_PEPTIDE",
    "REFERENCE_PARTNER_RESIDUE_COUNT_OUT_OF_RANGE",
    "STANDARD_AMINO_ACIDS",
    "STANDARD_PEPTIDE",
    "format_peptide_residue_range",
    "resolve_peptide_residue_limits",
    "validate_peptide_pdb",
    "validate_peptide_pdb_lines",
    "validate_peptide_pdb_text",
]
