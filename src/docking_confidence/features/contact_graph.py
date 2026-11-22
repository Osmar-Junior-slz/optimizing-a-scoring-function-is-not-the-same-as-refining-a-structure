from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Mapping

import numpy as np

from docking_confidence.domain.structure import (
    ALTLOC_POLICY_VERSION,
    CONTACT_GRAPH_SCHEMA_VERSION,
    PDB_PARSER_SCHEMA_VERSION,
    STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA,
    ComplexPose,
    structure_sequence_fingerprint,
)


CONTACT_GRAPH_FEATURE_COLUMNS = (
    "graph_has_interface_contact",
    "graph_min_cross_distance_angstrom",
    "graph_receptor_residue_count",
    "graph_ligand_residue_count",
    "graph_interface_edge_count",
    "graph_interface_density",
    "graph_receptor_contact_fraction",
    "graph_ligand_contact_fraction",
    "graph_mean_degree",
    "graph_degree_std",
    "graph_max_degree",
    "graph_receptor_mean_degree",
    "graph_ligand_mean_degree",
    "graph_isolated_residue_fraction",
    "graph_connected_component_count",
    "graph_largest_component_fraction",
    "graph_clash_atom_pair_count",
)

CONTACT_GRAPH_NODE_UNIVERSE = "all_selected_residues"

# These are useful audit summaries, but cannot be model features because they are
# undefined for the scientifically valid case in which the graph has zero edges.
CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS = (
    "graph_contact_distance_min",
    "graph_contact_distance_q25",
    "graph_contact_distance_median",
    "graph_contact_distance_q75",
    "graph_contact_distance_mean",
    "graph_contact_distance_max",
)

CONTACT_GRAPH_METADATA_COLUMNS = (
    "graph_schema_version",
    "graph_config_sha256",
    "graph_parser_schema_version",
    "graph_altloc_policy",
    "graph_node_universe",
    "graph_cutoff_angstrom",
    "graph_clash_cutoff_angstrom",
    "graph_include_hetero",
    "graph_heavy_atoms_only",
    "graph_exclude_waters",
    "graph_model_id",
    "graph_coordinate_frame",
    "graph_pose_source",
    "graph_pose_id",
    "graph_pose_rank",
    "graph_pose_selection_method",
    "graph_source_file_sha256",
    "graph_pose_canonical_sha256",
    "graph_receptor_chains",
    "graph_ligand_chains",
    "graph_bound_complex_id",
    "graph_bound_run_id",
    "graph_bound_algorithm",
    "graph_bound_seed",
    "graph_bound_engine",
    "graph_pose_protocol",
    "graph_complex_fingerprint_schema",
    "graph_complex_fingerprint_sha256",
    "graph_structure_sequence_fingerprint_schema",
    "graph_structure_sequence_fingerprint_sha256",
)

CONTACT_GRAPH_OPTIONAL_METADATA_COLUMNS = ("graph_transform_sha256",)


@dataclass(frozen=True)
class ContactGraphConfig:
    schema_version: str = CONTACT_GRAPH_SCHEMA_VERSION
    parser_schema_version: str = PDB_PARSER_SCHEMA_VERSION
    altloc_policy: str = ALTLOC_POLICY_VERSION
    node_universe: str = CONTACT_GRAPH_NODE_UNIVERSE
    model_id: int = 1
    cutoff_angstrom: float = 4.5
    clash_cutoff_angstrom: float = 2.0
    include_hetero: bool = False
    heavy_atoms_only: bool = True
    exclude_waters: bool = True

    def validate(self) -> None:
        if self.schema_version != CONTACT_GRAPH_SCHEMA_VERSION:
            raise ValueError(
                f"contact-graph schema_version must be {CONTACT_GRAPH_SCHEMA_VERSION!r}"
            )
        if self.parser_schema_version != PDB_PARSER_SCHEMA_VERSION:
            raise ValueError(
                f"contact-graph parser_schema_version must be {PDB_PARSER_SCHEMA_VERSION!r}"
            )
        if self.altloc_policy != ALTLOC_POLICY_VERSION:
            raise ValueError(
                f"contact-graph altloc_policy must be {ALTLOC_POLICY_VERSION!r}"
            )
        if self.node_universe != CONTACT_GRAPH_NODE_UNIVERSE:
            raise ValueError(
                "contact-graph node_universe must be "
                f"{CONTACT_GRAPH_NODE_UNIVERSE!r}"
            )
        if isinstance(self.model_id, bool) or not isinstance(self.model_id, int):
            raise ValueError("contact-graph model_id must be an integer")
        if self.model_id < 1:
            raise ValueError("contact-graph model_id must be positive")
        for field in ("include_hetero", "heavy_atoms_only", "exclude_waters"):
            if not isinstance(getattr(self, field), bool):
                raise ValueError(f"contact-graph {field} must be boolean")
        if not math.isfinite(float(self.cutoff_angstrom)):
            raise ValueError("contact cutoff must be finite")
        if not math.isfinite(float(self.clash_cutoff_angstrom)):
            raise ValueError("clash cutoff must be finite")
        if self.cutoff_angstrom <= 0:
            raise ValueError("contact cutoff must be positive")
        if not 0 < self.clash_cutoff_angstrom < self.cutoff_angstrom:
            raise ValueError(
                "clash cutoff must be positive and smaller than contact cutoff"
            )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> ContactGraphConfig:
        required = (
            "schema_version",
            "parser_schema_version",
            "altloc_policy",
            "node_universe",
            "model_id",
            "cutoff_angstrom",
            "clash_cutoff_angstrom",
            "include_hetero",
            "heavy_atoms_only",
            "exclude_waters",
        )
        missing = [field for field in required if field not in mapping]
        if missing:
            raise ValueError(
                f"contact-graph configuration is missing v2 fields: {missing}"
            )
        config = cls(
            schema_version=str(mapping["schema_version"]),
            parser_schema_version=str(mapping["parser_schema_version"]),
            altloc_policy=str(mapping["altloc_policy"]),
            node_universe=str(mapping["node_universe"]),
            model_id=mapping["model_id"],
            cutoff_angstrom=float(mapping["cutoff_angstrom"]),
            clash_cutoff_angstrom=float(mapping["clash_cutoff_angstrom"]),
            include_hetero=mapping["include_hetero"],
            heavy_atoms_only=mapping["heavy_atoms_only"],
            exclude_waters=mapping["exclude_waters"],
        )
        config.validate()
        return config

    @property
    def digest(self) -> str:
        self.validate()
        payload = json.dumps(
            asdict(self),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _connected_components(
    nodes: list[tuple], edges: list[tuple[tuple, tuple, float]]
) -> list[set[tuple]]:
    adjacency = {node: set() for node in nodes}
    for left, right, _ in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(nodes)
    components: list[set[tuple]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[tuple] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        remaining -= component
        components.append(component)
    return components


def _validate_pose_config_contract(
    pose: ComplexPose, config: ContactGraphConfig
) -> None:
    if pose.provenance.run.protocol != "reduced":
        raise ValueError(
            "contact-graph decision features require a pose bound to protocol='reduced'"
        )
    selection = pose.provenance.selection.structure_selection
    expected = {
        "parser_schema_version": config.parser_schema_version,
        "altloc_policy": config.altloc_policy,
        "model_id": config.model_id,
        "include_hetero": config.include_hetero,
        "heavy_atoms_only": config.heavy_atoms_only,
        "exclude_waters": config.exclude_waters,
    }
    observed = {
        "parser_schema_version": selection.parser_schema_version,
        "altloc_policy": selection.altloc_policy,
        "model_id": selection.model_id,
        "include_hetero": selection.include_hetero,
        "heavy_atoms_only": selection.heavy_atoms_only,
        "exclude_waters": selection.exclude_waters,
    }
    if observed != expected:
        raise ValueError(
            "complex pose selection does not match the contact-graph configuration: "
            f"expected {expected}, observed {observed}"
        )


def _finite_feature_contract(descriptors: Mapping[str, object]) -> None:
    missing = [
        feature for feature in CONTACT_GRAPH_FEATURE_COLUMNS if feature not in descriptors
    ]
    if missing:
        raise RuntimeError(f"contact-graph implementation omitted features: {missing}")
    for feature in CONTACT_GRAPH_FEATURE_COLUMNS:
        value = descriptors[feature]
        if isinstance(value, bool):
            continue
        try:
            numeric = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"contact-graph feature {feature} is not numeric") from exc
        if not math.isfinite(numeric):
            raise RuntimeError(f"contact-graph feature {feature} is non-finite")


def describe_interface_contact_graph(
    pose: ComplexPose,
    config: ContactGraphConfig,
) -> dict[str, object]:
    """Describe a deterministic residue-level bipartite contact graph.

    Receptor and ligand coordinates must already be expressed in the same
    Cartesian reference frame. ``ComplexPose`` enforces that both components
    came from one selected PDB model (or from one explicitly audited transform).
    A graph with zero interface edges is valid and is described without a
    numeric sentinel.
    """

    config.validate()
    _validate_pose_config_contract(pose, config)
    receptor: dict[tuple, list[tuple[float, float, float]]] = {}
    ligand: dict[tuple, list[tuple[float, float, float]]] = {}
    for atom in pose.atoms:
        if atom.component == "receptor":
            destination = receptor
        elif atom.component == "ligand":
            destination = ligand
        else:
            raise ValueError(f"unsupported pose component: {atom.component!r}")
        destination.setdefault(atom.residue_key, []).append(atom.xyz)
    if not receptor or not ligand:
        raise ValueError("complex pose must contain receptor and ligand atoms")

    contact_squared = float(config.cutoff_angstrom) ** 2
    clash_squared = float(config.clash_cutoff_angstrom) ** 2
    edges: list[tuple[tuple, tuple, float]] = []
    clash_atom_pairs = 0
    global_minimum_squared = math.inf
    for receptor_key in sorted(receptor):
        receptor_xyz = np.asarray(receptor[receptor_key], dtype=np.float64)
        for ligand_key in sorted(ligand):
            ligand_xyz = np.asarray(ligand[ligand_key], dtype=np.float64)
            delta = receptor_xyz[:, None, :] - ligand_xyz[None, :, :]
            distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
            if not np.isfinite(distance_squared).all():
                raise ValueError(
                    "cross-component distance calculation produced non-finite values"
                )
            minimum_squared = float(distance_squared.min())
            global_minimum_squared = min(global_minimum_squared, minimum_squared)
            clash_atom_pairs += int((distance_squared <= clash_squared).sum())
            if minimum_squared <= contact_squared:
                edges.append(
                    (receptor_key, ligand_key, float(np.sqrt(minimum_squared)))
                )
    if not math.isfinite(global_minimum_squared):
        raise RuntimeError("unable to calculate a finite cross-component distance")

    receptor_nodes = sorted(receptor)
    ligand_nodes = sorted(ligand)
    nodes = receptor_nodes + ligand_nodes
    degree = {node: 0 for node in nodes}
    for left, right, _ in edges:
        degree[left] += 1
        degree[right] += 1
    components = _connected_components(nodes, edges)
    degrees = np.asarray([degree[node] for node in nodes], dtype=np.float64)
    receptor_degrees = np.asarray(
        [degree[node] for node in receptor_nodes], dtype=np.float64
    )
    ligand_degrees = np.asarray(
        [degree[node] for node in ligand_nodes], dtype=np.float64
    )
    receptor_contacting = int((receptor_degrees > 0).sum())
    ligand_contacting = int((ligand_degrees > 0).sum())
    possible_edges = len(receptor_nodes) * len(ligand_nodes)
    distances = np.asarray([edge[2] for edge in edges], dtype=np.float64)

    run = pose.provenance.run
    selection = pose.provenance.selection
    complex_fingerprint = pose.provenance.complex_fingerprint
    structure_schema, structure_digest = structure_sequence_fingerprint(pose.atoms)
    if structure_schema != STRUCTURE_SEQUENCE_FINGERPRINT_SCHEMA:
        raise RuntimeError("unexpected structure sequence fingerprint schema")
    if structure_digest != selection.structure_sequence_fingerprint_sha256:
        raise ValueError("pose structure sequence fingerprint changed before graphing")

    descriptors: dict[str, object] = {
        # Join columns make the CLI output directly mergeable with a run row.
        "complex_id": run.complex_id,
        "run_id": run.run_id,
        "algorithm": run.algorithm,
        "seed": run.seed,
        f"engine_{run.protocol}": run.engine,
        "receptor_sequence": complex_fingerprint.receptor_sequence,
        "ligand_sequence": complex_fingerprint.ligand_sequence,
        "complex_fingerprint_schema": complex_fingerprint.schema,
        "complex_fingerprint_sha256": complex_fingerprint.sha256,
        # Closed graph/provenance contract.
        "graph_schema_version": config.schema_version,
        "graph_config_sha256": config.digest,
        "graph_parser_schema_version": config.parser_schema_version,
        "graph_altloc_policy": config.altloc_policy,
        "graph_node_universe": config.node_universe,
        "graph_cutoff_angstrom": float(config.cutoff_angstrom),
        "graph_clash_cutoff_angstrom": float(config.clash_cutoff_angstrom),
        "graph_include_hetero": config.include_hetero,
        "graph_heavy_atoms_only": config.heavy_atoms_only,
        "graph_exclude_waters": config.exclude_waters,
        "graph_model_id": config.model_id,
        "graph_coordinate_frame": "shared_complex_pose_v1",
        "graph_pose_source": pose.provenance.source,
        "graph_pose_id": selection.pose_id,
        "graph_pose_rank": selection.pose_rank,
        "graph_pose_selection_method": selection.selection_method,
        "graph_source_file_sha256": selection.source_file_sha256,
        "graph_pose_canonical_sha256": selection.canonical_pose_sha256,
        "graph_receptor_chains": json.dumps(
            list(selection.receptor_chains), separators=(",", ":")
        ),
        "graph_ligand_chains": json.dumps(
            list(selection.ligand_chains), separators=(",", ":")
        ),
        "graph_bound_complex_id": run.complex_id,
        "graph_bound_run_id": run.run_id,
        "graph_bound_algorithm": run.algorithm,
        "graph_bound_seed": run.seed,
        "graph_bound_engine": run.engine,
        "graph_pose_protocol": run.protocol,
        "graph_complex_fingerprint_schema": complex_fingerprint.schema,
        "graph_complex_fingerprint_sha256": complex_fingerprint.sha256,
        "graph_structure_sequence_fingerprint_schema": structure_schema,
        "graph_structure_sequence_fingerprint_sha256": structure_digest,
        "graph_transform_sha256": pose.provenance.transform_digest,
        # Always-finite model feature registry.
        "graph_has_interface_contact": int(bool(edges)),
        "graph_min_cross_distance_angstrom": float(
            np.sqrt(global_minimum_squared)
        ),
        "graph_receptor_residue_count": float(len(receptor_nodes)),
        "graph_ligand_residue_count": float(len(ligand_nodes)),
        "graph_interface_edge_count": float(len(edges)),
        "graph_interface_density": float(len(edges) / possible_edges),
        "graph_receptor_contact_fraction": float(
            receptor_contacting / len(receptor_nodes)
        ),
        "graph_ligand_contact_fraction": float(
            ligand_contacting / len(ligand_nodes)
        ),
        "graph_mean_degree": float(degrees.mean()),
        "graph_degree_std": float(degrees.std(ddof=0)),
        "graph_max_degree": float(degrees.max()),
        "graph_receptor_mean_degree": float(receptor_degrees.mean()),
        "graph_ligand_mean_degree": float(ligand_degrees.mean()),
        "graph_isolated_residue_fraction": float((degrees == 0).mean()),
        "graph_connected_component_count": float(len(components)),
        "graph_largest_component_fraction": float(
            max(len(component) for component in components) / len(nodes)
        ),
        "graph_clash_atom_pair_count": float(clash_atom_pairs),
    }
    if distances.size:
        descriptors.update(
            {
                "graph_contact_distance_min": float(distances.min()),
                "graph_contact_distance_q25": float(np.quantile(distances, 0.25)),
                "graph_contact_distance_median": float(
                    np.quantile(distances, 0.50)
                ),
                "graph_contact_distance_q75": float(np.quantile(distances, 0.75)),
                "graph_contact_distance_mean": float(distances.mean()),
                "graph_contact_distance_max": float(distances.max()),
            }
        )
    else:
        descriptors.update(
            {column: None for column in CONTACT_GRAPH_NULLABLE_AUDIT_COLUMNS}
        )
    _finite_feature_contract(descriptors)
    return descriptors
