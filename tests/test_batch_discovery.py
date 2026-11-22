from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from docking_confidence.application.batch_reporting import discovery_payload
from docking_confidence.domain.mobile_partner import (
    DEFAULT_MOBILE_PARTNER_DOMAIN,
    MobilePartnerResidueDomain,
)
from docking_confidence.domain.structure import (
    CANONICAL_COMPLEX_FINGERPRINT_SCHEMA,
)
from docking_confidence.io.batch_discovery import discover_ppi4dock


_ONE_TO_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}


def _atom_line(serial: int, residue: str, chain: str, number: int) -> str:
    return (
        f"ATOM  {serial:5d} {'CA':^4s} {residue:>3s} {chain:1s}{number:4d}    "
        f"{float(number):8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{20.0:6.2f}"
        f"          {'C':>2s}\n"
    )


def _pdb_text(chains: list[tuple[str, str]]) -> str:
    lines: list[str] = []
    serial = 1
    for chain, sequence in chains:
        for number, amino_acid in enumerate(sequence, start=1):
            lines.append(
                _atom_line(serial, _ONE_TO_THREE[amino_acid], chain, number)
            )
            serial += 1
    return "".join(lines) + "END\n"


def _write_target(
    root: Path,
    *,
    complex_id: str = "1abc_AB",
    receptor_sequence: str = "ACD",
    ligand_sequence: str = "GSKLT",
    reference_receptor_sequence: str | None = None,
    reference_ligand_sequence: str | None = None,
) -> Path:
    directory = root / complex_id
    directory.mkdir(parents=True)
    receptor_chain, ligand_chain = complex_id[-2], complex_id[-1]
    (directory / f"{receptor_chain}_model_st.pdb").write_text(
        _pdb_text([(receptor_chain, receptor_sequence)]), encoding="utf-8"
    )
    (directory / f"{ligand_chain}_model_st.pdb").write_text(
        _pdb_text([(ligand_chain, ligand_sequence)]), encoding="utf-8"
    )
    (directory / f"{complex_id}_st.pdb").write_text(
        _pdb_text(
            [
                (
                    receptor_chain,
                    reference_receptor_sequence or receptor_sequence,
                ),
                (ligand_chain, reference_ligand_sequence or ligand_sequence),
            ]
        ),
        encoding="utf-8",
    )
    return directory


def test_discovers_roles_hashes_sequences_counts_and_fingerprint(tmp_path: Path):
    target_dir = _write_target(tmp_path, complex_id="1abc_AB")

    report = discover_ppi4dock(tmp_path)

    assert report.input_root == tmp_path.resolve()
    assert report.validation_performed is True
    assert report.mobile_partner_domain == DEFAULT_MOBILE_PARTNER_DOMAIN
    assert report.issues == ()
    assert report.has_errors is False
    assert len(report.targets) == 1
    target = report.targets[0]
    assert target.complex_id == "1abc_AB"
    assert target.receptor_chain == "A"
    assert target.ligand_chain == "B"
    assert target.receptor_path == target_dir.resolve() / "A_model_st.pdb"
    assert target.ligand_path == target_dir.resolve() / "B_model_st.pdb"
    assert target.reference_path == target_dir.resolve() / "1abc_AB_st.pdb"
    assert target.valid is True
    assert target.validation_performed is True
    assert target.receptor_sequence == "ACD"
    assert target.ligand_sequence == "GSKLT"
    assert target.reference_receptor_sequence == "ACD"
    assert target.reference_ligand_sequence == "GSKLT"
    assert target.reference_sequence == "ACD|GSKLT"
    assert target.receptor_residue_count == 3
    assert target.ligand_residue_count == 5
    assert target.reference_residue_count == 8
    assert target.receptor_atom_count == 3
    assert target.ligand_atom_count == 5
    assert target.reference_atom_count == 8
    assert target.eligible is True
    assert "within_domain" in target.eligibility_reason
    assert target.complex_fingerprint_schema == CANONICAL_COMPLEX_FINGERPRINT_SCHEMA
    assert len(target.complex_fingerprint_sha256 or "") == 64
    expected_hash = hashlib.sha256(target.receptor_path.read_bytes()).hexdigest()
    assert target.receptor_sha256 == expected_hash
    assert report.valid_targets == (target,)
    assert report.eligible_targets == (target,)

    payload = discovery_payload(report)
    assert payload["schema_version"] == "ppi4dock_discovery_report_v2"
    assert payload["mobile_partner_domain"]["maximum_residues"] == 1200
    assert payload["counts"]["eligible_for_docking"] == 1
    assert payload["counts"]["outside_configured_domain"] == 0
    assert payload["targets"][0]["eligible_for_docking"] is True


def test_ligand_above_old_peptide_limit_is_eligible_in_unified_mode(
    tmp_path: Path,
):
    _write_target(tmp_path, ligand_sequence="A" * 31)

    report = discover_ppi4dock(tmp_path)

    target = report.targets[0]
    assert target.valid is True
    assert target.eligible is True
    assert target.ligand_residue_count == 31
    assert target in report.valid_targets
    assert target in report.eligible_targets
    assert target.eligibility_reason == (
        "mobile_partner_residue_count_within_domain: "
        "observed=31, domain=[5,1200]"
    )
    assert report.issues == ()


def test_local_corpus_maximum_is_eligible_but_hard_ceiling_remains(
    tmp_path: Path,
):
    _write_target(tmp_path, complex_id="1abc_AB", ligand_sequence="A" * 1150)
    _write_target(tmp_path, complex_id="2abc_CD", ligand_sequence="A" * 1201)

    report = discover_ppi4dock(tmp_path)

    accepted, rejected = report.targets
    assert accepted.ligand_residue_count == 1150
    assert accepted.eligible is True
    assert rejected.ligand_residue_count == 1201
    assert rejected.eligible is False
    assert rejected.eligibility_reason == (
        "mobile_partner_residue_count_above_1200: observed=1201"
    )


def test_explicit_legacy_ceiling_is_still_auditable(tmp_path: Path):
    _write_target(tmp_path, ligand_sequence="A" * 31)

    report = discover_ppi4dock(
        tmp_path,
        mobile_partner_domain=MobilePartnerResidueDomain(5, 30),
    )

    target = report.targets[0]
    assert target.valid is True
    assert target.eligible is False
    assert target.eligibility_reason == (
        "mobile_partner_residue_count_above_30: observed=31"
    )


def test_lightweight_discovery_does_not_read_or_hide_exact_empty_files(
    tmp_path: Path,
):
    directory = tmp_path / "2xyz_CD"
    directory.mkdir()
    for name in ("C_model_st.pdb", "D_model_st.pdb", "2xyz_CD_st.pdb"):
        (directory / name).write_bytes(b"")

    report = discover_ppi4dock(tmp_path, validate_contents=False)

    target = report.targets[0]
    assert target.valid is True
    assert target.validation_performed is False
    assert target.receptor_chain == "C"
    assert target.ligand_chain == "D"
    assert target.receptor_sha256 is None
    assert target.receptor_sequence is None
    assert target.eligible is None
    assert target.eligibility_reason == "content_validation_not_requested"


def test_exact_three_file_contract_reports_missing_and_unexpected_entries(
    tmp_path: Path,
):
    directory = _write_target(tmp_path)
    (directory / "B_model_st.pdb").unlink()
    (directory / "notes.txt").write_text("unexpected", encoding="utf-8")

    report = discover_ppi4dock(tmp_path, validate_contents=False)

    assert len(report.targets) == 1
    target = report.targets[0]
    assert target.valid is False
    assert target.eligibility_reason == "invalid_target"
    assert report.invalid_target_count == 1
    assert report.has_errors is True
    assert [issue.code for issue in report.issues] == ["invalid_target_file_set"]
    assert "B_model_st.pdb" in report.issues[0].message
    assert "notes.txt" in report.issues[0].message


def test_sequence_mismatch_is_reported_but_target_is_not_omitted(tmp_path: Path):
    _write_target(
        tmp_path,
        ligand_sequence="GSKLT",
        reference_ligand_sequence="GSKLA",
    )

    report = discover_ppi4dock(tmp_path)

    assert len(report.targets) == 1
    target = report.targets[0]
    assert target.valid is False
    assert target.validation_performed is True
    assert target.eligible is None
    assert "ligand model sequence differs" in target.errors[0]
    assert report.issues[0].code == "target_content_validation_failed"


def test_non_standard_amino_acid_is_rejected_explicitly(tmp_path: Path):
    directory = _write_target(tmp_path)
    receptor = directory / "A_model_st.pdb"
    receptor.write_text(
        receptor.read_text(encoding="utf-8").replace("ALA", "MSE", 1),
        encoding="utf-8",
    )

    report = discover_ppi4dock(tmp_path)

    target = report.targets[0]
    assert target.valid is False
    assert "non-standard residue 'MSE'" in target.errors[0]


def test_scanner_is_direct_strict_and_deterministic(tmp_path: Path):
    _write_target(tmp_path, complex_id="2xyz_CD")
    _write_target(tmp_path, complex_id="1abc_AB")
    nested_parent = tmp_path / "nested"
    _write_target(nested_parent, complex_id="3def_EF")
    (tmp_path / "README.txt").write_text("metadata", encoding="utf-8")

    report = discover_ppi4dock(tmp_path, validate_contents=False)

    assert [target.complex_id for target in report.targets] == ["1abc_AB", "2xyz_CD"]
    assert {issue.code for issue in report.issues} == {
        "invalid_target_directory_name",
        "unexpected_root_entry",
    }


@pytest.mark.parametrize(
    ("minimum", "maximum", "message"),
    [
        (0, 30, "positive"),
        (5, 0, "positive"),
        (31, 30, "cannot exceed"),
        (True, 30, "integer"),
    ],
)
def test_domain_bounds_are_strict(
    tmp_path: Path, minimum: int, maximum: int, message: str
):
    with pytest.raises(ValueError, match=message):
        discover_ppi4dock(
            tmp_path,
            mobile_partner_domain=MobilePartnerResidueDomain(
                minimum_residues=minimum,
                maximum_residues=maximum,
            ),
        )
