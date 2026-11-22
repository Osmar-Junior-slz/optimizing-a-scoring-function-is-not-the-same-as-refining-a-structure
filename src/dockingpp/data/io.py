"""I/O helpers for dockingpp."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np

from dockingpp.domain.models import Pocket


class MolecularCoordinates(np.ndarray):
    """NumPy-compatible coordinates that retain source atom records for scoring."""

    atoms: list[dict[str, Any]]

    def __new__(
        cls,
        coordinates: list[list[float]] | np.ndarray,
        atoms: list[dict[str, Any]] | None = None,
    ) -> "MolecularCoordinates":
        instance = np.asarray(coordinates, dtype=float).view(cls)
        instance.atoms = list(atoms or [])
        return instance

    def __array_finalize__(self, source: Any) -> None:
        self.atoms = list(getattr(source, "atoms", []) or [])


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""

    import yaml

    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_receptor(path: str) -> Any:
    """Load receptor data from disk."""

    if path == "__dummy__":
        return {"dummy": True}
    return load_pdb_coords(path)


def load_peptide(path: str) -> Any:
    """Load peptide data from disk."""

    if path == "__dummy__":
        return {"dummy": True}
    return load_pdb_coords(path)


def load_pdb_coords(path: str) -> np.ndarray:
    """Load atomic coordinates from a PDB file.

    Reads ATOM and HETATM records using standard PDB column positions.
    """

    coords: list[list[float]] = []
    atoms: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            coord = [x, y, z]
            coords.append(coord)
            atoms.append({"coords": coord, "line": line.rstrip("\r\n")})

    if not coords:
        return np.zeros((0, 3), dtype=float)
    return MolecularCoordinates(coords, atoms=atoms)


def molecular_partner_size(partner: Any) -> tuple[int | None, int | None]:
    """Return atom/residue counts from exactly the records accepted by the loader.

    Keeping this helper beside :func:`load_pdb_coords` guarantees that feature
    backfilling during calibration and live inference use identical rules for
    malformed coordinate lines, HETATM records, and multi-model files.
    """

    try:
        atom_count = int(len(partner))
    except (TypeError, ValueError):
        atom_count = 0
    atom_count_value = atom_count if atom_count > 0 else None
    residue_keys: set[tuple[str, str, str]] = set()
    for atom in list(getattr(partner, "atoms", []) or []):
        if not isinstance(atom, dict):
            continue
        line = atom.get("line")
        if not isinstance(line, str) or len(line) < 27:
            continue
        if line[:6].strip().upper() not in {"ATOM", "HETATM"}:
            continue
        residue_keys.add((line[21:22], line[22:26], line[26:27]))
    residue_count = len(residue_keys)
    return atom_count_value, residue_count if residue_count > 0 else None


def _extract_coords(receptor: Any) -> np.ndarray:
    if isinstance(receptor, dict) and "coords" in receptor:
        return np.asarray(receptor["coords"], dtype=float)
    if isinstance(receptor, np.ndarray):
        return np.asarray(receptor, dtype=float)
    coords = getattr(receptor, "coords", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _get_cfg_value(cfg: Optional[Any], key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_debug_logger(cfg: Optional[Any]) -> Any:
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get("debug_logger")
    return getattr(cfg, "debug_logger", None)


def _surface_anchor_statistics(
    grid_points: np.ndarray,
    receptor_coords: np.ndarray,
    *,
    contact_radius: float,
    chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest distance and local receptor count for each grid point.

    This intentionally has a NumPy-only path so pocket generation remains
    reproducible when the optional SciPy dependency is unavailable.
    """

    points = np.asarray(grid_points, dtype=float)
    receptor = np.asarray(receptor_coords, dtype=float)
    nearest = np.full(points.shape[0], np.inf, dtype=float)
    nearby = np.zeros(points.shape[0], dtype=int)
    radius_sq = float(contact_radius) ** 2
    effective_chunk = max(1, int(chunk_size))

    for start in range(0, points.shape[0], effective_chunk):
        stop = min(points.shape[0], start + effective_chunk)
        deltas = points[start:stop, None, :] - receptor[None, :, :]
        dist_sq = np.sum(deltas * deltas, axis=2)
        nearest[start:stop] = np.sqrt(np.min(dist_sq, axis=1))
        nearby[start:stop] = np.sum(dist_sq <= radius_sq, axis=1)

    return nearest, nearby


def _select_separated_surface_anchors(
    candidate_coords: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    max_count: int,
    min_separation: float,
) -> list[int]:
    """Deterministically select high-scoring, spatially separated anchors."""

    coords = np.asarray(candidate_coords, dtype=float)
    scores = np.asarray(candidate_scores, dtype=float)
    if coords.size == 0 or max_count <= 0:
        return []

    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], -scores))
    selected: list[int] = []
    separation = max(0.0, float(min_separation))
    for raw_index in order:
        index = int(raw_index)
        if separation > 0.0 and any(
            float(np.linalg.norm(coords[index] - coords[previous])) < separation
            for previous in selected
        ):
            continue
        selected.append(index)
        if len(selected) >= int(max_count):
            break
    return selected


def load_pockets(
    receptor: Any,
    cfg: Optional[Any] = None,
    pockets_path: Optional[str] = None,
) -> List[Pocket]:
    """Build auditable surface-anchor candidates from a receptor.

    Automatic candidates are empty grid points near receptor atoms.  Local
    atoms describe the candidate interface, while ``receptor_coords_full`` is
    retained separately so every downstream collision/objective calculation
    can use the complete receptor.  This is a geometric candidate generator,
    not a learned cavity predictor.
    """

    receptor_coords = _extract_coords(receptor)
    debug_logger = _get_debug_logger(cfg)
    pocket_margin = float(_get_cfg_value(cfg, "pocket_margin", 2.0))
    if pockets_path and os.path.exists(pockets_path):
        with open(pockets_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle) or {}
        pockets_payload = payload.get("pockets", [])
        pockets: list[Pocket] = []
        for idx, entry in enumerate(pockets_payload):
            center = np.asarray(entry.get("center", [0.0, 0.0, 0.0]), dtype=float)
            radius = float(entry.get("radius", 0.0))
            pocket_id = str(entry.get("id") or f"pocket_{idx}")
            if receptor_coords.size:
                deltas = receptor_coords - center.reshape(1, 3)
                dists = np.linalg.norm(deltas, axis=1)
                mask = dists <= radius
                pocket_coords = receptor_coords[mask]
            else:
                pocket_coords = receptor_coords.copy()
            pockets.append(
                Pocket(
                    id=pocket_id,
                    center=center,
                    radius=radius,
                    coords=pocket_coords,
                    meta={
                        "coords": pocket_coords,
                        "local_coords": pocket_coords,
                        "receptor_coords_full": receptor_coords,
                        "detection_mode": "provided_file",
                    },
                )
            )
        if debug_logger is not None:
            debug_logger.log(
                {
                    "type": "pocket_generation",
                    "mode": "file",
                    "pockets_path": pockets_path,
                    "receptor_coords_n": int(receptor_coords.shape[0]),
                    "n_pockets": int(len(pockets)),
                }
            )
        return pockets

    def build_global_pocket() -> Pocket:
        coords = receptor_coords.copy()
        if coords.size:
            center = coords.mean(axis=0)
            deltas = coords - center.reshape(1, 3)
            max_dist = float(np.max(np.linalg.norm(deltas, axis=1)))
        else:
            center = np.zeros(3, dtype=float)
            max_dist = 0.0
        radius = max_dist + pocket_margin
        return Pocket(
            id="global",
            center=center,
            radius=radius,
            coords=coords,
            meta={
                "coords": coords,
                "local_coords": coords,
                "receptor_coords_full": receptor_coords,
                "detection_mode": "global_fallback",
            },
        )

    grid_size = float(_get_cfg_value(cfg, "pocket_grid_size", 4.0) or 0.0)
    grid_padding = float(_get_cfg_value(cfg, "pocket_grid_padding", 8.0) or 0.0)
    contact_radius = float(_get_cfg_value(cfg, "pocket_contact_radius", 8.0) or 0.0)
    min_clearance = float(_get_cfg_value(cfg, "pocket_anchor_min_clearance", 3.0) or 0.0)
    max_clearance = float(_get_cfg_value(cfg, "pocket_anchor_max_clearance", 6.0) or 0.0)
    target_clearance = float(_get_cfg_value(cfg, "pocket_anchor_target_clearance", 4.5) or 0.0)
    clearance_weight = float(_get_cfg_value(cfg, "pocket_anchor_clearance_weight", 2.0) or 0.0)
    min_contact_atoms = int(
        _get_cfg_value(
            cfg,
            "min_pocket_contact_atoms",
            _get_cfg_value(cfg, "min_pocket_atoms", 8),
        )
        or 0
    )
    max_pockets = max(1, int(_get_cfg_value(cfg, "auto_pocket_count", 32) or 32))
    anchor_separation = float(_get_cfg_value(cfg, "pocket_anchor_separation", 6.0) or 0.0)
    search_radius = float(_get_cfg_value(cfg, "pocket_search_radius", 5.0) or 0.0)
    chunk_size = max(1, int(_get_cfg_value(cfg, "pocket_grid_chunk_size", 128) or 128))
    max_grid_points = max(1, int(_get_cfg_value(cfg, "pocket_max_grid_points", 250_000) or 250_000))

    failure_reason: str | None = None
    grid_points = np.zeros((0, 3), dtype=float)
    candidate_coords = np.zeros((0, 3), dtype=float)
    candidate_nearest = np.zeros((0,), dtype=float)
    candidate_nearby = np.zeros((0,), dtype=int)
    pockets: list[Pocket] = []

    if receptor_coords.size == 0:
        failure_reason = "empty_receptor"
    elif grid_size <= 0.0:
        failure_reason = "invalid_grid_size"
    elif contact_radius <= 0.0:
        failure_reason = "invalid_contact_radius"
    elif max_clearance < min_clearance:
        failure_reason = "invalid_clearance_interval"
    else:
        bbox_min = receptor_coords.min(axis=0) - grid_padding
        bbox_max = receptor_coords.max(axis=0) + grid_padding
        axes = [
            np.arange(bbox_min[axis], bbox_max[axis] + (0.5 * grid_size), grid_size)
            for axis in range(3)
        ]
        n_grid_points = int(np.prod([len(axis) for axis in axes], dtype=np.int64))
        if n_grid_points <= 0:
            failure_reason = "empty_grid"
        elif n_grid_points > max_grid_points:
            failure_reason = "grid_point_limit_exceeded"
        else:
            grid_points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            nearest, nearby = _surface_anchor_statistics(
                grid_points,
                receptor_coords,
                contact_radius=contact_radius,
                chunk_size=chunk_size,
            )
            candidate_mask = (
                (nearest >= min_clearance)
                & (nearest <= max_clearance)
                & (nearby >= min_contact_atoms)
            )
            candidate_coords = grid_points[candidate_mask]
            candidate_nearest = nearest[candidate_mask]
            candidate_nearby = nearby[candidate_mask]
            candidate_scores = candidate_nearby.astype(float) - (
                clearance_weight * np.abs(candidate_nearest - target_clearance)
            )
            selected_indices = _select_separated_surface_anchors(
                candidate_coords,
                candidate_scores,
                max_count=max_pockets,
                min_separation=anchor_separation,
            )
            for pocket_index, selected_index in enumerate(selected_indices):
                center = np.asarray(candidate_coords[selected_index], dtype=float)
                local_distances = np.linalg.norm(
                    receptor_coords - center.reshape(1, 3),
                    axis=1,
                )
                local_coords = receptor_coords[local_distances <= contact_radius]
                pockets.append(
                    Pocket(
                        id=f"auto_surface_{pocket_index}",
                        center=center,
                        radius=search_radius,
                        coords=local_coords,
                        meta={
                            "coords": local_coords,
                            "local_coords": local_coords,
                            "receptor_coords_full": receptor_coords,
                            "detection_mode": "surface_clearance_grid",
                            "anchor_score": float(candidate_scores[selected_index]),
                            "nearest_receptor_distance": float(candidate_nearest[selected_index]),
                            "nearby_receptor_atoms": int(candidate_nearby[selected_index]),
                            "contact_radius": float(contact_radius),
                            "search_radius": float(search_radius),
                            "grid_size": float(grid_size),
                        },
                    )
                )
            if not pockets:
                failure_reason = "no_surface_anchor_passed_filters"

    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "pocket_generation",
                "mode": "surface_clearance_grid",
                "pockets_path": pockets_path,
                "receptor_coords_n": int(receptor_coords.shape[0]),
                "grid_size": float(grid_size),
                "grid_padding": float(grid_padding),
                "contact_radius": float(contact_radius),
                "min_clearance": float(min_clearance),
                "max_clearance": float(max_clearance),
                "min_contact_atoms": int(min_contact_atoms),
                "n_grid_points": int(grid_points.shape[0]),
                "n_candidates_before_separation": int(candidate_coords.shape[0]),
                "n_pockets": int(len(pockets)),
                "failure_reason": failure_reason,
            }
        )

    if pockets:
        return pockets

    search_space_mode = str(_get_cfg_value(cfg, "search_space_mode", "") or "").strip().lower()
    if debug_logger is not None:
        debug_logger.log(
            {
                "type": "pocket_fallback",
                "reason": failure_reason or "no_surface_anchor",
                "n_generated": 0,
                "fallback_target": "full_search" if search_space_mode == "reduced" else "global_pocket",
            }
        )
    if search_space_mode == "reduced":
        return []
    return [build_global_pocket()]
