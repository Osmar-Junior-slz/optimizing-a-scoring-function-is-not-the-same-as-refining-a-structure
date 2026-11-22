"""Preflight and canonical preparation for protein-peptide benchmark samples."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from dockingpp.evaluation.structural_metrics import (
    PDBAtom,
    evaluate_structures,
    interface_residue_contacts,
    read_pdb_atoms,
    residue_contacts,
    write_complex_pdb,
)


STANDARD_AMINO_ACIDS = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }
)


@dataclass(frozen=True)
class SampleAuditRow:
    sample: str
    status: str
    reason: str
    receptor_source: str | None
    peptide_source: str | None
    peptide_chain: str | None
    peptide_residues: int | None
    receptor_atoms: int | None
    peptide_atoms: int | None
    native_contacts_4a: int | None
    interface_pairs_8a: int | None
    identity_dockq: float | None
    receptor_sha256: str | None
    peptide_sha256: str | None
    prepared_receptor: str | None
    prepared_peptide: str | None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _residues(atoms: Sequence[PDBAtom]) -> set[tuple[str, int, str, str]]:
    return {
        (atom.chain_id, atom.residue_number, atom.insertion_code, atom.residue_name)
        for atom in atoms
    }


def _standard_residues(atoms: Sequence[PDBAtom]) -> set[tuple[str, int, str, str]]:
    return {residue for residue in _residues(atoms) if residue[3] in STANDARD_AMINO_ACIDS}


def _all_standard_peptide(atoms: Sequence[PDBAtom]) -> bool:
    residues = _residues(atoms)
    return bool(residues) and all(residue[3] in STANDARD_AMINO_ACIDS for residue in residues)


def _peptide_named_files(files: Sequence[Path]) -> list[Path]:
    return [path for path in files if "pept" in str(path).lower()]


def _receptor_named_files(files: Sequence[Path]) -> list[Path]:
    labels = ("protein", "proteina", "receptor")
    return [path for path in files if any(label in str(path).lower() for label in labels)]


def _find_receptor_for_peptide(files: Sequence[Path], peptide_path: Path) -> Path | None:
    candidates = [path for path in _receptor_named_files(files) if path != peptide_path]
    if candidates:
        return sorted(candidates, key=lambda path: path.stat().st_size, reverse=True)[0]
    remaining = [path for path in files if path != peptide_path]
    return sorted(remaining, key=lambda path: path.stat().st_size, reverse=True)[0] if remaining else None


def _recover_short_chain(
    files: Sequence[Path],
    *,
    min_peptide_residues: int,
    max_peptide_residues: int,
) -> tuple[Path, str, list[PDBAtom], list[PDBAtom]] | None:
    for path in sorted(files, key=lambda item: item.stat().st_size, reverse=True):
        atoms = read_pdb_atoms(path)
        chains: dict[str, list[PDBAtom]] = {}
        for atom in atoms:
            if atom.record_name == "ATOM" and atom.residue_name in STANDARD_AMINO_ACIDS:
                chains.setdefault(atom.chain_id, []).append(atom)
        chain_lengths = {
            chain: len(_standard_residues(chain_atoms)) for chain, chain_atoms in chains.items()
        }
        candidates = [
            chain
            for chain, length in chain_lengths.items()
            if min_peptide_residues <= length <= max_peptide_residues
        ]
        if not candidates or not any(length > max_peptide_residues for length in chain_lengths.values()):
            continue
        peptide_chain = min(candidates, key=lambda chain: (chain_lengths[chain], chain))
        peptide_atoms = chains[peptide_chain]
        receptor_atoms = [
            atom
            for chain, chain_atoms in chains.items()
            if chain != peptide_chain
            for atom in chain_atoms
        ]
        if receptor_atoms and peptide_atoms:
            return path, peptide_chain, receptor_atoms, peptide_atoms
    return None


def _validated_row(
    *,
    sample: str,
    status: str,
    reason: str,
    receptor_source: Path,
    peptide_source: Path,
    peptide_chain: str | None,
    receptor_atoms: Sequence[PDBAtom],
    peptide_atoms: Sequence[PDBAtom],
    output_dir: Path,
) -> SampleAuditRow:
    contacts = residue_contacts(receptor_atoms, peptide_atoms, cutoff_angstrom=4.0)
    interface = interface_residue_contacts(receptor_atoms, peptide_atoms, cutoff_angstrom=8.0)
    if not contacts or not interface:
        return SampleAuditRow(
            sample=sample,
            status="excluded_no_native_interface",
            reason="No native receptor-peptide interface under CAPRI cutoffs",
            receptor_source=str(receptor_source),
            peptide_source=str(peptide_source),
            peptide_chain=peptide_chain,
            peptide_residues=len(_standard_residues(peptide_atoms)),
            receptor_atoms=len(receptor_atoms),
            peptide_atoms=len(peptide_atoms),
            native_contacts_4a=len(contacts),
            interface_pairs_8a=len(interface),
            identity_dockq=None,
            receptor_sha256=_sha256(receptor_source),
            peptide_sha256=_sha256(peptide_source),
            prepared_receptor=None,
            prepared_peptide=None,
        )

    identity = evaluate_structures(
        native_receptor=receptor_atoms,
        native_peptide=peptide_atoms,
        model_receptor=receptor_atoms,
        model_peptide=peptide_atoms,
    )
    prepared_dir = output_dir / "prepared" / sample
    prepared_dir.mkdir(parents=True, exist_ok=True)
    receptor_out = prepared_dir / "receptor.pdb"
    peptide_out = prepared_dir / "peptide.pdb"
    native_out = prepared_dir / "native_complex.pdb"
    write_complex_pdb(receptor_out, receptor_atoms, [])
    write_complex_pdb(peptide_out, [], peptide_atoms)
    write_complex_pdb(native_out, receptor_atoms, peptide_atoms)
    return SampleAuditRow(
        sample=sample,
        status=status,
        reason=reason,
        receptor_source=str(receptor_source),
        peptide_source=str(peptide_source),
        peptide_chain=peptide_chain,
        peptide_residues=len(_standard_residues(peptide_atoms)),
        receptor_atoms=len(receptor_atoms),
        peptide_atoms=len(peptide_atoms),
        native_contacts_4a=len(contacts),
        interface_pairs_8a=len(interface),
        identity_dockq=identity.dockq,
        receptor_sha256=_sha256(receptor_source),
        peptide_sha256=_sha256(peptide_source),
        prepared_receptor=str(receptor_out),
        prepared_peptide=str(peptide_out),
    )


def audit_sample_directory(
    samples_root: str | Path,
    output_dir: str | Path,
    *,
    min_peptide_residues: int = 5,
    max_peptide_residues: int = 20,
) -> list[SampleAuditRow]:
    root = Path(samples_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[SampleAuditRow] = []

    for case_dir in sorted((path for path in root.iterdir() if path.is_dir()), key=lambda p: p.name.upper()):
        sample = case_dir.name.upper()
        files = sorted(case_dir.rglob("*.pdb"))
        parsed = {path: read_pdb_atoms(path) for path in files}
        peptide_candidates = _peptide_named_files(files)
        standard_candidate = next(
            (path for path in peptide_candidates if _all_standard_peptide(parsed[path])),
            None,
        )
        if standard_candidate is not None:
            peptide_atoms = parsed[standard_candidate]
            peptide_length = len(_standard_residues(peptide_atoms))
            receptor_path = _find_receptor_for_peptide(files, standard_candidate)
            if peptide_length > max_peptide_residues:
                rows.append(
                    SampleAuditRow(
                        sample, "excluded_peptide_length",
                        f"Standard chain has {peptide_length} residues; limit is {max_peptide_residues}",
                        str(receptor_path) if receptor_path else None, str(standard_candidate), None,
                        peptide_length, None, len(peptide_atoms), None, None, None,
                        _sha256(receptor_path) if receptor_path else None, _sha256(standard_candidate),
                        None, None,
                    )
                )
                continue
            if peptide_length < min_peptide_residues or receptor_path is None:
                rows.append(
                    SampleAuditRow(
                        sample, "excluded_invalid_pair", "Missing receptor or peptide below minimum length",
                        str(receptor_path) if receptor_path else None, str(standard_candidate), None,
                        peptide_length, None, len(peptide_atoms), None, None, None,
                        _sha256(receptor_path) if receptor_path else None, _sha256(standard_candidate),
                        None, None,
                    )
                )
                continue
            rows.append(
                _validated_row(
                    sample=sample,
                    status="ready_standard_peptide",
                    reason="Separate receptor and standard peptide are ready",
                    receptor_source=receptor_path,
                    peptide_source=standard_candidate,
                    peptide_chain=next(iter({atom.chain_id for atom in peptide_atoms}), None),
                    receptor_atoms=parsed[receptor_path],
                    peptide_atoms=peptide_atoms,
                    output_dir=output,
                )
            )
            continue

        recovered = _recover_short_chain(
            files,
            min_peptide_residues=min_peptide_residues,
            max_peptide_residues=max_peptide_residues,
        )
        if recovered is not None:
            source, chain, receptor_atoms, peptide_atoms = recovered
            rows.append(
                _validated_row(
                    sample=sample,
                    status="ready_recovered_chain",
                    reason=f"Recovered standard peptide chain {chain} from the bound complex",
                    receptor_source=source,
                    peptide_source=source,
                    peptide_chain=chain,
                    receptor_atoms=receptor_atoms,
                    peptide_atoms=peptide_atoms,
                    output_dir=output,
                )
            )
            continue

        nonstandard = sorted(
            {
                residue[3]
                for atoms in parsed.values()
                for residue in _residues(atoms)
                if residue[3] not in STANDARD_AMINO_ACIDS and residue[3] not in {"HOH", "WAT"}
            }
        )
        rows.append(
            SampleAuditRow(
                sample=sample,
                status="excluded_non_peptide_ligand",
                reason=(
                    "No standard peptide chain between "
                    f"{min_peptide_residues} and {max_peptide_residues} residues; "
                    f"non-standard ligands: {', '.join(nonstandard) or 'none'}"
                ),
                receptor_source=str(files[0]) if files else None,
                peptide_source=None,
                peptide_chain=None,
                peptide_residues=None,
                receptor_atoms=None,
                peptide_atoms=None,
                native_contacts_4a=None,
                interface_pairs_8a=None,
                identity_dockq=None,
                receptor_sha256=_sha256(files[0]) if files else None,
                peptide_sha256=None,
                prepared_receptor=None,
                prepared_peptide=None,
            )
        )

    payload = [asdict(row) for row in rows]
    (output / "sample_audit.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "sample_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(payload[0]) if payload else [])
        if payload:
            writer.writeheader()
            writer.writerows(payload)

    ready = [row for row in rows if row.status.startswith("ready_")]
    lines = [
        "# Auditoria das amostras proteina-peptideo",
        "",
        f"- Amostras recebidas: {len(rows)}",
        f"- Casos peptidicos prontos: {len(ready)}",
        f"- Casos excluidos: {len(rows) - len(ready)}",
        "",
        "| Amostra | Status | Residuos | Contatos 4 A | Interface 8 A | Motivo |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.sample} | {row.status} | {row.peptide_residues or '-'} | "
            f"{row.native_contacts_4a or '-'} | {row.interface_pairs_8a or '-'} | {row.reason} |"
        )
    (output / "sample_audit.md").write_text("\n".join(lines) + "\n", encoding="ascii")
    return rows
