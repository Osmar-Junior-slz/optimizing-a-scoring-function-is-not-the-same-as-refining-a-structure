"""Utilities for cleaning PDB text."""

from __future__ import annotations

from typing import Iterable


WATER_RESNAMES = {"HOH", "WAT", "SOL", "H2O"}
ION_RESNAMES = {
    "NA",
    "CL",
    "K",
    "CA",
    "MG",
    "ZN",
    "MN",
    "FE",
    "CU",
    "CO",
    "NI",
    "CD",
    "HG",
    "BR",
    "I",
    "PO4",
    "SO4",
}
HEADER_RECORDS = {"HEADER", "TITLE", "REMARK", "MODEL", "ENDMDL", "TER", "END"}


def _get_record(line: str) -> str:
    return line[:6].strip().upper()


def _get_resname(line: str) -> str:
    return line[17:20].strip().upper()


def _first_model_lines(lines: Iterable[str]) -> list[str]:
    """Return the header and only the first coordinate model.

    PDB files without ``MODEL`` records are returned unchanged.  When models
    are present, records after the first ``ENDMDL`` (or the next ``MODEL`` in
    malformed input) are ignored.  ``clean_pdb_text`` appends the canonical
    final ``END`` record separately.
    """

    selected: list[str] = []
    found_model = False
    in_first_model = False

    for line in lines:
        record = _get_record(line)
        if record == "MODEL":
            if found_model:
                break
            found_model = True
            in_first_model = True
            selected.append(line)
            continue
        if record == "ENDMDL" and found_model:
            if in_first_model:
                selected.append(line)
            break
        if not found_model or in_first_model:
            selected.append(line)

    return selected


def _should_keep_atom(
    record: str,
    resname: str,
    remove_waters: bool,
    remove_hetatm: bool,
    remove_ions: bool,
    keep_het_resnames: set[str] | None,
) -> bool:
    if remove_waters and resname in WATER_RESNAMES:
        return False
    if remove_ions and resname in ION_RESNAMES:
        return False
    if record == "HETATM" and remove_hetatm:
        if keep_het_resnames and resname in keep_het_resnames:
            return True
        return False
    return True


def _iter_cleaned_lines(
    lines: Iterable[str],
    remove_waters: bool,
    remove_hetatm: bool,
    remove_ions: bool,
    keep_het_resnames: set[str] | None,
) -> list[str]:
    cleaned: list[str] = []
    for line in lines:
        if not line:
            continue
        record = _get_record(line)
        if record in {"ATOM", "HETATM"}:
            resname = _get_resname(line)
            if _should_keep_atom(
                record=record,
                resname=resname,
                remove_waters=remove_waters,
                remove_hetatm=remove_hetatm,
                remove_ions=remove_ions,
                keep_het_resnames=keep_het_resnames,
            ):
                cleaned.append(line)
            continue
        # END is normalized by clean_pdb_text so whitespace variants and
        # repeated records cannot produce duplicated terminators.
        if record in HEADER_RECORDS and record != "END":
            cleaned.append(line)
    return cleaned


def clean_pdb_text(
    pdb_text: str,
    remove_waters: bool = True,
    remove_hetatm: bool = True,
    remove_ions: bool = True,
    keep_het_resnames: set[str] | None = None,
) -> str:
    """Return cleaned PDB content based on simple deterministic filtering."""

    keep_het_resnames = {name.strip().upper() for name in keep_het_resnames or set() if name.strip()}
    lines = [line.rstrip("\r\n") for line in pdb_text.splitlines()]
    first_model = _first_model_lines(lines)
    cleaned = _iter_cleaned_lines(
        first_model,
        remove_waters=remove_waters,
        remove_hetatm=remove_hetatm,
        remove_ions=remove_ions,
        keep_het_resnames=keep_het_resnames or None,
    )
    cleaned.append("END")
    return "\n".join(cleaned) + "\n"
