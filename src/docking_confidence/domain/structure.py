from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable


CONTACT_GRAPH_SCHEMA_VERSION = "residue_interface_contact_graph_v2"
PDB_PARSER_SCHEMA_VERSION = "pdb_fixed_columns_v2"
ALTLOC_POLICY_VERSION = "residue_highest_total_occupancy_v1"
CANONICAL_COMPLEX_FINGERPRINT_SCHEMA = "canonical_component_sequences_v1"
STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA = "component_residue_sequences_v1"

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AA_ALPHABET = frozenset("ACDEFGHIKLMNPQRSTVWY")
_COMPONENTS = frozenset({"receptor", "ligand", "unassigned"})


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_sha256(value: str, *, field: str) -> str:
    normalized = str(value).strip().lower()
    if _SHA256_RE.fullmatch(normalized) is None:
        raise ValueError(f"{field} must contain 64 lowercase hexadecimal characters")
    return normalized


def _nonempty(value: object, *, field: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{field} cannot be blank")
    return normalized


def canonicalize_component_sequence(value: object, *, component: str) -> str:
    """Canonicalize a component as a sorted multiset of chain sequences.

    The input grammar is ``CHAIN[|CHAIN...]``. Chain order is deliberately
    discarded, while duplicate chains are retained. Only the 20 standard amino
    acid one-letter codes are accepted so that spelling differences cannot alter
    the development split silently.
    """

    if value is None:
        raise ValueError(f"{component}_sequence cannot be missing")
    raw = str(value).strip().upper()
    if not raw:
        raise ValueError(f"{component}_sequence cannot be blank")
    chains = [chain.strip() for chain in raw.split("|")]
    if any(not chain for chain in chains):
        raise ValueError(
            f"{component}_sequence must contain non-empty chains separated by '|'"
        )
    for chain in chains:
        invalid = sorted(set(chain) - _AA_ALPHABET)
        if invalid:
            raise ValueError(
                f"{component}_sequence contains unsupported amino-acid codes: "
                f"{invalid}"
            )
    return "|".join(sorted(chains))


@dataclass(frozen=True)
class CanonicalComplexFingerprint:
    receptor_sequence: str
    ligand_sequence: str
    schema: str = CANONICAL_COMPLEX_FINGERPRINT_SCHEMA
    sha256: str = ""

    def __post_init__(self) -> None:
        if self.schema != CANONICAL_COMPLEX_FINGERPRINT_SCHEMA:
            raise ValueError(
                "complex fingerprint schema must be "
                f"{CANONICAL_COMPLEX_FINGERPRINT_SCHEMA!r}"
            )
        receptor = canonicalize_component_sequence(
            self.receptor_sequence, component="receptor"
        )
        ligand = canonicalize_component_sequence(
            self.ligand_sequence, component="ligand"
        )
        expected = _canonical_sha256(
            {
                "schema": self.schema,
                "receptor_chains": receptor.split("|"),
                "ligand_chains": ligand.split("|"),
            }
        )
        if self.sha256:
            supplied = normalize_sha256(
                self.sha256, field="complex_fingerprint_sha256"
            )
            if supplied != expected:
                raise ValueError(
                    "complex_fingerprint_sha256 does not match the canonical "
                    "receptor/ligand sequences"
                )
        object.__setattr__(self, "receptor_sequence", receptor)
        object.__setattr__(self, "ligand_sequence", ligand)
        object.__setattr__(self, "sha256", expected)

    @classmethod
    def from_sequences(
        cls,
        receptor_sequence: object,
        ligand_sequence: object,
        *,
        expected_sha256: object | None = None,
        schema: object = CANONICAL_COMPLEX_FINGERPRINT_SCHEMA,
    ) -> CanonicalComplexFingerprint:
        return cls(
            receptor_sequence=str(receptor_sequence),
            ligand_sequence=str(ligand_sequence),
            schema=str(schema).strip(),
            sha256="" if expected_sha256 is None else str(expected_sha256),
        )


@dataclass(frozen=True)
class RunBinding:
    complex_id: str
    run_id: str
    algorithm: str
    seed: int
    protocol: str
    engine: str

    def __post_init__(self) -> None:
        for field in ("complex_id", "run_id", "algorithm", "engine"):
            object.__setattr__(
                self, field, _nonempty(getattr(self, field), field=field)
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        protocol = str(self.protocol).strip().lower()
        if protocol not in {"reduced", "full"}:
            raise ValueError("protocol must be 'reduced' or 'full'")
        object.__setattr__(self, "protocol", protocol)


@dataclass(frozen=True)
class Atom:
    component: str
    model_id: int
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    atom_name: str
    altloc: str
    element: str
    occupancy: float | None
    xyz: tuple[float, float, float]
    is_hetero: bool = False

    def __post_init__(self) -> None:
        if self.component not in _COMPONENTS:
            raise ValueError(f"unsupported atom component: {self.component!r}")
        if isinstance(self.model_id, bool) or not isinstance(self.model_id, int):
            raise ValueError("atom model_id must be an integer")
        if self.model_id < 1:
            raise ValueError("atom model_id must be positive")
        if isinstance(self.residue_number, bool) or not isinstance(
            self.residue_number, int
        ):
            raise ValueError("atom residue_number must be an integer")
        for field in ("chain_id", "residue_name", "atom_name", "element"):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"atom {field} cannot be blank")
        if len(self.insertion_code) > 1 or len(self.altloc) > 1:
            raise ValueError("PDB insertion_code and altloc must be at most one character")
        if len(self.xyz) != 3 or not all(math.isfinite(float(v)) for v in self.xyz):
            raise ValueError("atom coordinates must contain three finite values")
        if self.occupancy is not None:
            occupancy = float(self.occupancy)
            if not math.isfinite(occupancy) or not 0.0 <= occupancy <= 1.0:
                raise ValueError("atom occupancy must be finite and in [0, 1]")

    @property
    def residue_key(self) -> tuple[str, str, int, str, str]:
        return (
            self.component,
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
        )

    @property
    def canonical_record(self) -> tuple[object, ...]:
        return (
            self.component,
            self.model_id,
            self.chain_id,
            self.residue_number,
            self.insertion_code,
            self.residue_name,
            self.atom_name,
            self.altloc,
            self.element,
            None if self.occupancy is None else float(self.occupancy).hex(),
            *(float(value).hex() for value in self.xyz),
            self.is_hetero,
        )


@dataclass(frozen=True)
class StructureSelection:
    model_id: int = 1
    include_hetero: bool = False
    exclude_waters: bool = True
    heavy_atoms_only: bool = True
    parser_schema_version: str = PDB_PARSER_SCHEMA_VERSION
    altloc_policy: str = ALTLOC_POLICY_VERSION

    def __post_init__(self) -> None:
        if isinstance(self.model_id, bool) or not isinstance(self.model_id, int):
            raise ValueError("model_id must be an integer")
        if self.model_id < 1:
            raise ValueError("model_id must be positive")
        if self.parser_schema_version != PDB_PARSER_SCHEMA_VERSION:
            raise ValueError(
                f"parser_schema_version must be {PDB_PARSER_SCHEMA_VERSION!r}"
            )
        if self.altloc_policy != ALTLOC_POLICY_VERSION:
            raise ValueError(f"altloc_policy must be {ALTLOC_POLICY_VERSION!r}")
        for field in ("include_hetero", "exclude_waters", "heavy_atoms_only"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"{field} must be boolean")


@dataclass(frozen=True)
class Structure:
    atoms: tuple[Atom, ...]
    source_file_sha256: str
    selection: StructureSelection

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("structure must contain at least one selected atom")
        normalized = normalize_sha256(
            self.source_file_sha256, field="source_file_sha256"
        )
        object.__setattr__(self, "source_file_sha256", normalized)
        if any(atom.model_id != self.selection.model_id for atom in self.atoms):
            raise ValueError("all structure atoms must belong to the selected model")

    @property
    def source_sha256(self) -> str:
        """Compatibility alias; this is always the raw source-file digest."""

        return self.source_file_sha256

    @property
    def model_id(self) -> int:
        return self.selection.model_id


def structure_sequence_fingerprint(
    atoms: Iterable[Atom],
) -> tuple[str, str]:
    """Hash ordered residue-name sequences, independent of pose coordinates.

    This is an auditable fingerprint of residues actually represented in the
    selected structure, not a substitute for the canonical biological sequence.
    Chain labels and chain order are discarded; multiplicity is retained.
    """

    sites: dict[str, dict[str, dict[tuple[int, str], str]]] = {
        "receptor": {},
        "ligand": {},
    }
    for atom in atoms:
        if atom.component not in sites:
            raise ValueError(
                "structure sequence fingerprint requires assigned receptor/ligand atoms"
            )
        chain_sites = sites[atom.component].setdefault(atom.chain_id, {})
        site = (atom.residue_number, atom.insertion_code)
        previous = chain_sites.setdefault(site, atom.residue_name)
        if previous != atom.residue_name:
            raise ValueError(
                "selected structure contains multiple residue names at one residue site"
            )

    component_sequences: dict[str, list[list[str]]] = {}
    for component in ("receptor", "ligand"):
        if not sites[component]:
            raise ValueError(
                f"structure sequence fingerprint has no {component} residues"
            )
        sequences = [
            [residues[key] for key in sorted(residues)]
            for residues in sites[component].values()
        ]
        component_sequences[component] = sorted(sequences)
    payload = {
        "schema": STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA,
        "receptor": component_sequences["receptor"],
        "ligand": component_sequences["ligand"],
    }
    return STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA, _canonical_sha256(payload)


def canonical_pose_sha256(
    atoms: Iterable[Atom],
    *,
    structure_selection: StructureSelection,
    receptor_chains: tuple[str, ...],
    ligand_chains: tuple[str, ...],
    pose_id: str,
    pose_rank: int,
    selection_method: str,
    structure_sequence_fingerprint_sha256: str,
) -> str:
    payload = {
        "schema": "selected_complex_pose_v2",
        "structure_selection": {
            "parser_schema_version": structure_selection.parser_schema_version,
            "altloc_policy": structure_selection.altloc_policy,
            "model_id": structure_selection.model_id,
            "include_hetero": structure_selection.include_hetero,
            "exclude_waters": structure_selection.exclude_waters,
            "heavy_atoms_only": structure_selection.heavy_atoms_only,
        },
        "component_assignment": {
            "receptor_chains": list(receptor_chains),
            "ligand_chains": list(ligand_chains),
        },
        "pose_selection": {
            "pose_id": _nonempty(pose_id, field="pose_id"),
            "pose_rank": pose_rank,
            "selection_method": _nonempty(
                selection_method, field="selection_method"
            ),
        },
        "structure_sequence_fingerprint_sha256": normalize_sha256(
            structure_sequence_fingerprint_sha256,
            field="structure_sequence_fingerprint_sha256",
        ),
        "atoms": [list(atom.canonical_record) for atom in sorted(
            atoms, key=lambda item: item.canonical_record
        )],
    }
    return _canonical_sha256(payload)


@dataclass(frozen=True)
class PoseSelection:
    pose_id: str
    pose_rank: int
    selection_method: str
    receptor_chains: tuple[str, ...]
    ligand_chains: tuple[str, ...]
    structure_selection: StructureSelection
    source_file_sha256: str
    canonical_pose_sha256: str
    structure_sequence_fingerprint_schema: str
    structure_sequence_fingerprint_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "pose_id", _nonempty(self.pose_id, field="pose_id"))
        object.__setattr__(
            self,
            "selection_method",
            _nonempty(self.selection_method, field="selection_method"),
        )
        if isinstance(self.pose_rank, bool) or not isinstance(self.pose_rank, int):
            raise ValueError("pose_rank must be an integer")
        if self.pose_rank < 1:
            raise ValueError("pose_rank must be positive")
        receptor = tuple(sorted({_nonempty(v, field="receptor chain") for v in self.receptor_chains}))
        ligand = tuple(sorted({_nonempty(v, field="ligand chain") for v in self.ligand_chains}))
        if not receptor or not ligand:
            raise ValueError("receptor and ligand chain sets must be non-empty")
        if set(receptor) & set(ligand):
            raise ValueError("receptor and ligand chains must be disjoint")
        object.__setattr__(self, "receptor_chains", receptor)
        object.__setattr__(self, "ligand_chains", ligand)
        object.__setattr__(
            self,
            "source_file_sha256",
            normalize_sha256(self.source_file_sha256, field="source_file_sha256"),
        )
        object.__setattr__(
            self,
            "canonical_pose_sha256",
            normalize_sha256(
                self.canonical_pose_sha256, field="canonical_pose_sha256"
            ),
        )
        if (
            self.structure_sequence_fingerprint_schema
            != STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA
        ):
            raise ValueError(
                "structure sequence fingerprint schema must be "
                f"{STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA!r}"
            )
        object.__setattr__(
            self,
            "structure_sequence_fingerprint_sha256",
            normalize_sha256(
                self.structure_sequence_fingerprint_sha256,
                field="structure_sequence_fingerprint_sha256",
            ),
        )


@dataclass(frozen=True)
class PoseProvenance:
    source: str
    run: RunBinding
    selection: PoseSelection
    complex_fingerprint: CanonicalComplexFingerprint
    transform_digest: str | None = None

    def __post_init__(self) -> None:
        source = _nonempty(self.source, field="pose source")
        allowed = {"single_complex_pdb", "docking_engine", "explicit_transform"}
        if source not in allowed:
            raise ValueError(f"unsupported pose source: {source!r}")
        object.__setattr__(self, "source", source)
        if self.transform_digest is not None:
            object.__setattr__(
                self,
                "transform_digest",
                normalize_sha256(self.transform_digest, field="transform_digest"),
            )

    @property
    def pose_id(self) -> str:
        return self.selection.pose_id

    @property
    def source_sha256(self) -> str:
        return self.selection.source_file_sha256

    @property
    def engine(self) -> str:
        return self.run.engine

    @property
    def run_id(self) -> str:
        return self.run.run_id


@dataclass(frozen=True)
class ComplexPose:
    atoms: tuple[Atom, ...]
    receptor_chains: tuple[str, ...]
    ligand_chains: tuple[str, ...]
    provenance: PoseProvenance

    def __post_init__(self) -> None:
        if not self.atoms:
            raise ValueError("complex pose must contain selected atoms")
        selection = self.provenance.selection
        if tuple(self.receptor_chains) != selection.receptor_chains:
            raise ValueError("pose receptor chains disagree with provenance selection")
        if tuple(self.ligand_chains) != selection.ligand_chains:
            raise ValueError("pose ligand chains disagree with provenance selection")
        for atom in self.atoms:
            if atom.component == "receptor":
                expected_chains = set(self.receptor_chains)
            elif atom.component == "ligand":
                expected_chains = set(self.ligand_chains)
            else:
                raise ValueError("complex pose atoms must be assigned to a component")
            if atom.chain_id not in expected_chains:
                raise ValueError("atom component disagrees with selected component chains")
            if atom.model_id != selection.structure_selection.model_id:
                raise ValueError("atom model disagrees with pose selection")
        if not any(atom.component == "receptor" for atom in self.atoms):
            raise ValueError("complex pose has no receptor atoms")
        if not any(atom.component == "ligand" for atom in self.atoms):
            raise ValueError("complex pose has no ligand atoms")

        fingerprint_schema, fingerprint_sha256 = structure_sequence_fingerprint(
            self.atoms
        )
        if fingerprint_schema != selection.structure_sequence_fingerprint_schema:
            raise ValueError("pose structure fingerprint schema is inconsistent")
        if fingerprint_sha256 != selection.structure_sequence_fingerprint_sha256:
            raise ValueError("pose structure fingerprint is inconsistent")
        expected_pose_sha256 = canonical_pose_sha256(
            self.atoms,
            structure_selection=selection.structure_selection,
            receptor_chains=selection.receptor_chains,
            ligand_chains=selection.ligand_chains,
            pose_id=selection.pose_id,
            pose_rank=selection.pose_rank,
            selection_method=selection.selection_method,
            structure_sequence_fingerprint_sha256=fingerprint_sha256,
        )
        if expected_pose_sha256 != selection.canonical_pose_sha256:
            raise ValueError("canonical pose hash is inconsistent with selected atoms")
