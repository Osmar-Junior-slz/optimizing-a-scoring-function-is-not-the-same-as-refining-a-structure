from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from docking_confidence.domain.mobile_partner import (
    DEFAULT_MOBILE_PARTNER_DOMAIN,
    MobilePartnerResidueDomain,
)
from docking_confidence.domain.structure import (
    CANONICAL_COMPLEX_FINGERPRINT_SCHEMA,
    Atom,
    CanonicalComplexFingerprint,
    Structure,
)
from docking_confidence.io.pdb import read_pdb


_TARGET_NAME = re.compile(
    r"(?P<pdb_id>[0-9A-Za-z]{4})_(?P<receptor>[0-9A-Za-z])(?P<ligand>[0-9A-Za-z])",
    flags=re.ASCII,
)

_STANDARD_RESIDUES = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


@dataclass(frozen=True)
class DiscoveryIssue:
    code: str
    message: str
    relative_path: str
    complex_id: str | None = None
    severity: str = "error"


@dataclass(frozen=True)
class PPI4DockTarget:
    complex_id: str
    directory: Path
    receptor_chain: str
    ligand_chain: str
    receptor_path: Path
    ligand_path: Path
    reference_path: Path
    valid: bool
    validation_performed: bool
    errors: tuple[str, ...] = ()
    receptor_sha256: str | None = None
    ligand_sha256: str | None = None
    reference_sha256: str | None = None
    receptor_sequence: str | None = None
    ligand_sequence: str | None = None
    reference_receptor_sequence: str | None = None
    reference_ligand_sequence: str | None = None
    reference_sequence: str | None = None
    complex_fingerprint_schema: str | None = None
    complex_fingerprint_sha256: str | None = None
    receptor_atom_count: int | None = None
    ligand_atom_count: int | None = None
    reference_atom_count: int | None = None
    receptor_residue_count: int | None = None
    ligand_residue_count: int | None = None
    reference_residue_count: int | None = None
    eligible: bool | None = None
    eligibility_reason: str = "content_validation_not_requested"


@dataclass(frozen=True)
class BatchDiscoveryReport:
    input_root: Path
    targets: tuple[PPI4DockTarget, ...]
    issues: tuple[DiscoveryIssue, ...]
    validation_performed: bool
    mobile_partner_domain: MobilePartnerResidueDomain

    @property
    def valid_targets(self) -> tuple[PPI4DockTarget, ...]:
        return tuple(target for target in self.targets if target.valid)

    @property
    def eligible_targets(self) -> tuple[PPI4DockTarget, ...]:
        return tuple(
            target
            for target in self.targets
            if target.valid and target.eligible is True
        )

    @property
    def invalid_target_count(self) -> int:
        return sum(not target.valid for target in self.targets)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)


def _residue_sequence(
    atoms: tuple[Atom, ...],
    *,
    expected_chains: tuple[str, ...],
    component: str,
) -> tuple[str, int, int]:
    actual_chains = tuple(sorted({atom.chain_id for atom in atoms}))
    if actual_chains != tuple(sorted(expected_chains)):
        raise ValueError(
            f"{component} chains are {list(actual_chains)}, expected "
            f"{list(sorted(expected_chains))}"
        )

    chain_sequences: list[str] = []
    residue_count = 0
    for chain in sorted(expected_chains):
        sites: dict[tuple[int, str], str] = {}
        for atom in atoms:
            if atom.chain_id != chain:
                continue
            site = (atom.residue_number, atom.insertion_code)
            previous = sites.setdefault(site, atom.residue_name)
            if previous != atom.residue_name:
                raise ValueError(
                    f"{component} contains conflicting residue names at "
                    f"chain {chain} site {site}"
                )
        if not sites:
            raise ValueError(f"{component} chain {chain} contains no residues")
        sequence: list[str] = []
        for site in sorted(sites):
            residue_name = sites[site]
            try:
                sequence.append(_STANDARD_RESIDUES[residue_name])
            except KeyError as exc:
                raise ValueError(
                    f"{component} contains non-standard residue "
                    f"{residue_name!r} at chain {chain} site {site}"
                ) from exc
        chain_sequences.append("".join(sequence))
        residue_count += len(sequence)
    return "|".join(chain_sequences), residue_count, len(atoms)


def _read_standard_structure(path: Path) -> Structure:
    return read_pdb(
        path,
        include_hetero=False,
        exclude_waters=True,
        heavy_atoms_only=False,
        model_id=1,
    )


def _validate_target_contents(
    target: PPI4DockTarget,
    *,
    mobile_partner_domain: MobilePartnerResidueDomain,
) -> PPI4DockTarget:
    receptor = _read_standard_structure(target.receptor_path)
    ligand = _read_standard_structure(target.ligand_path)
    reference = _read_standard_structure(target.reference_path)

    receptor_sequence, receptor_residues, receptor_atoms = _residue_sequence(
        receptor.atoms,
        expected_chains=(target.receptor_chain,),
        component="receptor model",
    )
    ligand_sequence, ligand_residues, ligand_atoms = _residue_sequence(
        ligand.atoms,
        expected_chains=(target.ligand_chain,),
        component="ligand model",
    )
    _, reference_residues, reference_atoms = _residue_sequence(
        reference.atoms,
        expected_chains=(target.receptor_chain, target.ligand_chain),
        component="reference complex",
    )
    reference_by_chain: dict[str, str] = {}
    for chain, component in (
        (target.receptor_chain, "reference receptor"),
        (target.ligand_chain, "reference ligand"),
    ):
        sequence, _, _ = _residue_sequence(
            tuple(atom for atom in reference.atoms if atom.chain_id == chain),
            expected_chains=(chain,),
            component=component,
        )
        reference_by_chain[chain] = sequence

    reference_receptor = reference_by_chain[target.receptor_chain]
    reference_ligand = reference_by_chain[target.ligand_chain]
    if receptor_sequence != reference_receptor:
        raise ValueError(
            "receptor model sequence differs from its reference-chain sequence"
        )
    if ligand_sequence != reference_ligand:
        raise ValueError(
            "ligand model sequence differs from its reference-chain sequence"
        )

    fingerprint = CanonicalComplexFingerprint.from_sequences(
        receptor_sequence,
        ligand_sequence,
    )
    if ligand_residues < mobile_partner_domain.minimum_residues:
        eligible = False
        eligibility_reason = (
            "mobile_partner_residue_count_below_"
            f"{mobile_partner_domain.minimum_residues}: "
            f"observed={ligand_residues}"
        )
    elif ligand_residues > mobile_partner_domain.maximum_residues:
        eligible = False
        eligibility_reason = (
            "mobile_partner_residue_count_above_"
            f"{mobile_partner_domain.maximum_residues}: "
            f"observed={ligand_residues}"
        )
    else:
        eligible = True
        eligibility_reason = (
            "mobile_partner_residue_count_within_domain: "
            f"observed={ligand_residues}, "
            "domain="
            f"[{mobile_partner_domain.minimum_residues},"
            f"{mobile_partner_domain.maximum_residues}]"
        )

    return replace(
        target,
        valid=True,
        validation_performed=True,
        errors=(),
        receptor_sha256=receptor.source_file_sha256,
        ligand_sha256=ligand.source_file_sha256,
        reference_sha256=reference.source_file_sha256,
        receptor_sequence=receptor_sequence,
        ligand_sequence=ligand_sequence,
        reference_receptor_sequence=reference_receptor,
        reference_ligand_sequence=reference_ligand,
        # Preserve operational component order even when chain identifiers are
        # not lexicographically ordered (for example, ``1abc_BA``).
        reference_sequence=f"{reference_receptor}|{reference_ligand}",
        complex_fingerprint_schema=CANONICAL_COMPLEX_FINGERPRINT_SCHEMA,
        complex_fingerprint_sha256=fingerprint.sha256,
        receptor_atom_count=receptor_atoms,
        ligand_atom_count=ligand_atoms,
        reference_atom_count=reference_atoms,
        receptor_residue_count=receptor_residues,
        ligand_residue_count=ligand_residues,
        reference_residue_count=reference_residues,
        eligible=eligible,
        eligibility_reason=eligibility_reason,
    )


def _target_from_directory(directory: Path, match: re.Match[str]) -> PPI4DockTarget:
    receptor_chain = match.group("receptor")
    ligand_chain = match.group("ligand")
    complex_id = directory.name
    return PPI4DockTarget(
        complex_id=complex_id,
        directory=directory,
        receptor_chain=receptor_chain,
        ligand_chain=ligand_chain,
        receptor_path=directory / f"{receptor_chain}_model_st.pdb",
        ligand_path=directory / f"{ligand_chain}_model_st.pdb",
        reference_path=directory / f"{complex_id}_st.pdb",
        valid=True,
        validation_performed=False,
    )


def discover_ppi4dock(
    input_root: str | Path,
    *,
    validate_contents: bool = True,
    mobile_partner_domain: MobilePartnerResidueDomain = (
        DEFAULT_MOBILE_PARTNER_DOMAIN
    ),
) -> BatchDiscoveryReport:
    """Discover direct PPI4DOCK children without guessing biological roles.

    A target named ``1a2y_AB`` binds the first suffix chain (``A``) to the
    operational receptor and the second (``B``) to the operational ligand.
    Content-invalid and operational-domain-ineligible targets remain visible in the
    report; callers must use the explicit flags rather than infer validity from
    presence in ``targets``.
    """

    if not isinstance(validate_contents, bool):
        raise ValueError("validate_contents must be boolean")
    if not isinstance(mobile_partner_domain, MobilePartnerResidueDomain):
        raise ValueError(
            "mobile_partner_domain must be a MobilePartnerResidueDomain"
        )
    root = Path(input_root).expanduser()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)
    root = root.resolve()

    targets: list[PPI4DockTarget] = []
    issues: list[DiscoveryIssue] = []
    for entry in sorted(root.iterdir(), key=lambda path: (path.name.casefold(), path.name)):
        relative = entry.name
        if entry.is_symlink():
            issues.append(
                DiscoveryIssue(
                    code="symlink_not_allowed",
                    message="symbolic links are not valid PPI4DOCK target entries",
                    relative_path=relative,
                )
            )
            continue
        if not entry.is_dir():
            issues.append(
                DiscoveryIssue(
                    code="unexpected_root_entry",
                    message="input root must contain target directories only",
                    relative_path=relative,
                )
            )
            continue
        match = _TARGET_NAME.fullmatch(entry.name)
        if match is None:
            issues.append(
                DiscoveryIssue(
                    code="invalid_target_directory_name",
                    message=(
                        "target directory must match <pdb4>_<XY>, for example "
                        "1a2y_AB"
                    ),
                    relative_path=relative,
                )
            )
            continue

        target = _target_from_directory(entry.resolve(), match)
        if target.receptor_chain == target.ligand_chain:
            message = "receptor and ligand chain identifiers must be distinct"
            targets.append(
                replace(
                    target,
                    valid=False,
                    validation_performed=validate_contents,
                    errors=(message,),
                    eligibility_reason="invalid_target",
                )
            )
            issues.append(
                DiscoveryIssue(
                    code="identical_component_chains",
                    message=message,
                    relative_path=relative,
                    complex_id=target.complex_id,
                )
            )
            continue

        expected_names = {
            target.receptor_path.name,
            target.ligand_path.name,
            target.reference_path.name,
        }
        try:
            children = tuple(entry.iterdir())
        except OSError as exc:
            message = f"cannot enumerate target directory: {exc}"
            targets.append(
                replace(
                    target,
                    valid=False,
                    validation_performed=validate_contents,
                    errors=(message,),
                    eligibility_reason="invalid_target",
                )
            )
            issues.append(
                DiscoveryIssue(
                    code="target_directory_unreadable",
                    message=message,
                    relative_path=relative,
                    complex_id=target.complex_id,
                )
            )
            continue

        actual_names = {child.name for child in children}
        invalid_files = [
            child.name
            for child in children
            if child.is_symlink() or not child.is_file()
        ]
        if actual_names != expected_names or invalid_files:
            missing = sorted(expected_names - actual_names)
            unexpected = sorted(actual_names - expected_names)
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if unexpected:
                parts.append(f"unexpected={unexpected}")
            if invalid_files:
                parts.append(f"not_regular_files={sorted(invalid_files)}")
            message = "target must contain exactly its three expected PDB files"
            if parts:
                message = f"{message}: {'; '.join(parts)}"
            targets.append(
                replace(
                    target,
                    valid=False,
                    validation_performed=validate_contents,
                    errors=(message,),
                    eligibility_reason="invalid_target",
                )
            )
            issues.append(
                DiscoveryIssue(
                    code="invalid_target_file_set",
                    message=message,
                    relative_path=relative,
                    complex_id=target.complex_id,
                )
            )
            continue

        if validate_contents:
            try:
                target = _validate_target_contents(
                    target,
                    mobile_partner_domain=mobile_partner_domain,
                )
            except (OSError, UnicodeError, ValueError) as exc:
                message = str(exc) or exc.__class__.__name__
                target = replace(
                    target,
                    valid=False,
                    validation_performed=True,
                    errors=(message,),
                    eligibility_reason="invalid_target",
                )
                issues.append(
                    DiscoveryIssue(
                        code="target_content_validation_failed",
                        message=message,
                        relative_path=relative,
                        complex_id=target.complex_id,
                    )
                )
        targets.append(target)

    return BatchDiscoveryReport(
        input_root=root,
        targets=tuple(targets),
        issues=tuple(issues),
        validation_performed=validate_contents,
        mobile_partner_domain=mobile_partner_domain,
    )
