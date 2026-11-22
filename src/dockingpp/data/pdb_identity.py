"""Canonical, provenance-aware identification of PDB docking inputs."""

from __future__ import annotations

import re
from pathlib import Path


_PDB_ID_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}$")
_PDB_FILENAME_RE = re.compile(
    r"^(?:case_)?(?P<pdb_id>[0-9][A-Za-z0-9]{3})(?:_(?:receptor|peptide))?$",
    re.IGNORECASE,
)


def pdb_id_from_header(path: str | Path) -> str | None:
    """Read the canonical four-character idCode from a PDB HEADER record."""

    candidate = Path(path)
    if not candidate.is_file():
        return None
    try:
        with candidate.open("r", encoding="utf-8", errors="replace") as handle:
            for raw_line in handle:
                record = raw_line[:6].strip().upper()
                if record == "HEADER":
                    pdb_id = raw_line[62:66].strip()
                    return pdb_id.upper() if _PDB_ID_RE.fullmatch(pdb_id) else None
                if record in {"ATOM", "HETATM", "MODEL"}:
                    break
    except OSError:
        return None
    return None


def pdb_id_from_filename(path: str | Path) -> str | None:
    """Infer an id only from an unambiguous PDB-style filename."""

    match = _PDB_FILENAME_RE.fullmatch(Path(path).stem)
    return match.group("pdb_id").upper() if match else None


def infer_complex_id(receptor_path: str | Path, peptide_path: str | Path) -> str | None:
    """Resolve one complex id without guessing from arbitrary directory names."""

    return resolve_complex_identity(receptor_path, peptide_path)["complex_id"]


def resolve_complex_identity(
    receptor_path: str | Path, peptide_path: str | Path
) -> dict[str, str | None]:
    """Resolve id, source and conflicts for audit metadata."""

    header_ids = {
        value
        for value in (
            pdb_id_from_header(receptor_path),
            pdb_id_from_header(peptide_path),
        )
        if value
    }
    if len(header_ids) == 1:
        return {
            "complex_id": next(iter(header_ids)),
            "complex_id_source": "pdb_header_idcode",
            "complex_id_conflict": None,
        }
    if len(header_ids) > 1:
        return {
            "complex_id": None,
            "complex_id_source": "unresolved_header_conflict",
            "complex_id_conflict": ",".join(sorted(header_ids)),
        }

    filename_ids = {
        value
        for value in (
            pdb_id_from_filename(receptor_path),
            pdb_id_from_filename(peptide_path),
        )
        if value
    }
    if len(filename_ids) == 1:
        return {
            "complex_id": next(iter(filename_ids)),
            "complex_id_source": "strict_pdb_filename",
            "complex_id_conflict": None,
        }
    if len(filename_ids) > 1:
        return {
            "complex_id": None,
            "complex_id_source": "unresolved_filename_conflict",
            "complex_id_conflict": ",".join(sorted(filename_ids)),
        }
    return {
        "complex_id": None,
        "complex_id_source": "unresolved",
        "complex_id_conflict": None,
    }


def build_input_id(receptor_path: str | Path, peptide_path: str | Path) -> str:
    """Build a stable non-scientific input identifier from the two filenames."""

    receptor_name = Path(receptor_path).name or str(receptor_path)
    peptide_name = Path(peptide_path).name or str(peptide_path)
    return f"{receptor_name}__{peptide_name}"


def normalize_pdb_id(value: str | None) -> str | None:
    """Normalize a user-supplied PDB id, returning ``None`` when it is not canonical."""

    candidate = str(value or "").strip()
    return candidate.upper() if _PDB_ID_RE.fullmatch(candidate) else None



__all__ = [
    "build_input_id",
    "infer_complex_id",
    "normalize_pdb_id",
    "pdb_id_from_filename",
    "pdb_id_from_header",
    "resolve_complex_identity",
]
