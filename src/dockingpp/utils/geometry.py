"""Geometry helpers."""

from __future__ import annotations

import numpy as np


def pairwise_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""

    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff**2, axis=-1))


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Compute RMSD between two coordinate sets."""

    diff = a - b
    return float(np.sqrt(np.mean(diff**2)))


def apply_transform(
    coords: np.ndarray, rot_matrix: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    """Apply a rigid transform to coordinates."""

    rotated = np.asarray(coords, dtype=float) @ np.asarray(rot_matrix, dtype=float).T
    return rotated + np.asarray(translation, dtype=float)


def surface_anchor_normal(pocket) -> np.ndarray | None:
    """Outward receptor-only direction for automatically detected surface sites.

    Surface grid points represent an interface, not the mobile body's centroid.
    Explicit cavity pockets and the FULL global region retain their semantics.
    """
    meta = getattr(pocket, "meta", {}) or {}
    if meta.get("detection_mode") != "surface_clearance_grid":
        return None
    receptor = np.asarray(meta.get("receptor_coords_full", []), dtype=float)
    if receptor.ndim != 2 or receptor.shape[1] != 3 or not len(receptor):
        raise ValueError("surface pocket requires full receptor coordinates")
    center = np.asarray(pocket.center, dtype=float)
    normal = center - receptor.mean(axis=0)
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        nearest = np.argmin(np.sum((receptor - center) ** 2, axis=1))
        normal = center - receptor[nearest]
        norm = float(np.linalg.norm(normal))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("surface pocket has no defined outward direction")
    return normal / norm


def place_surface_contact(
    coords: np.ndarray, rotation: np.ndarray, contact_point: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Rigidly place the receptor-facing support atom at the local contact site.

    This changes translation only after rotation: internal distances are intact.
    Every atom lies on the outward side of the support plane. The full receptor
    collision checks still decide whether a placement is actually admissible.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not len(coords):
        raise ValueError("surface placement requires nonempty (N,3) coordinates")
    rotated = (coords - coords.mean(axis=0)) @ np.asarray(rotation, dtype=float).T
    support_index = int(np.argmin(rotated @ np.asarray(normal, dtype=float)))
    placed = rotated - rotated[support_index] + np.asarray(contact_point, dtype=float)
    return placed, support_index


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Generate a random rotation matrix."""

    quat = rng.normal(size=4)
    norm = np.linalg.norm(quat)
    if norm == 0:
        return np.eye(3, dtype=float)
    quat = quat / norm
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
        ],
        dtype=float,
    )
