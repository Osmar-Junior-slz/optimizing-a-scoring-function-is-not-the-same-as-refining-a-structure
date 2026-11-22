"""Final rescoring (Vina-inspired) used only at the end of the pipeline."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from dockingpp.domain.models import Pose


_VDW_RADII = {
    "H": 1.20,
    "HD": 1.20,
    "C": 1.90,
    "A": 1.90,
    "N": 1.80,
    "NA": 1.80,
    "O": 1.70,
    "OA": 1.70,
    "S": 2.00,
    "SA": 2.00,
    "P": 2.10,
    "F": 1.47,
    "CL": 1.75,
    "BR": 1.85,
    "I": 1.98,
    "ZN": 1.39,
    "MG": 1.73,
    "MN": 1.79,
    "FE": 1.72,
    "CA": 1.94,
    "CU": 1.40,
    "NI": 1.63,
}

_HYDROPHOBIC_TYPES = {"C", "A", "S"}
# PDB inputs carry elements rather than AutoDock atom types. The broad element
# fallback keeps the Vina-inspired score chemically aware until PDBQT typing is
# available; AutoDock-specific OA/NA/SA remain supported.
_ACCEPTOR_TYPES = {"O", "OA", "N", "NA", "S", "SA"}
_DONOR_TYPES = {"HD"}


def _last_token(value: str) -> str:
    tokens = str(value or "").strip().split()
    return tokens[-1] if tokens else ""


def _guess_element_from_name(name: str) -> str:
    normalized = str(name or "").strip()
    if not normalized:
        return "C"
    two = normalized[:2].upper()
    if two in {"CL", "BR"}:
        return two
    return normalized[0].upper()


def _parse_type_from_pdbqt(line: str) -> str:
    token = _last_token(line)
    if token:
        upper = token.upper()
        if upper in _VDW_RADII:
            return upper
        title = token.capitalize()
        if title in _VDW_RADII:
            return title
    atom_name = line[12:16] if len(line) >= 16 else ""
    return _guess_element_from_name(atom_name)


def _parse_charge_from_pdbqt(line: str) -> float:
    text = str(line or "")
    if not text:
        return 0.0
    # PDBQT stores the partial charge in columns 71-76. Standard PDB files
    # leave that field blank; scanning arbitrary numbers would mistake the
    # B-factor for a charge.
    charge_field = text[70:76].strip() if len(text) >= 76 else ""
    if not charge_field:
        return 0.0
    try:
        charge = float(charge_field)
    except (TypeError, ValueError):
        return 0.0
    return float(min(1.2, max(-1.2, charge)))


def _vdw_radius(atom_type: str) -> float:
    return float(_VDW_RADII.get(str(atom_type or "").upper(), 1.75))


def _is_hydrophobic(atom_type: str) -> bool:
    return str(atom_type or "").upper() in _HYDROPHOBIC_TYPES


def _is_acceptor(atom_type: str) -> bool:
    return str(atom_type or "").upper() in _ACCEPTOR_TYPES


def _is_donor(atom_type: str) -> bool:
    atom = str(atom_type or "").upper()
    return atom in _DONOR_TYPES or atom in {"N", "O", "S"}


def _extract_coords(source: Any) -> np.ndarray:
    if source is None:
        return np.zeros((0, 3), dtype=float)
    if isinstance(source, np.ndarray):
        return np.asarray(source, dtype=float)
    if isinstance(source, dict):
        coords = source.get("coords")
        if coords is None:
            coords = source.get("atoms")
        if coords is None:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(coords, dtype=float)
    coords = getattr(source, "coords", None)
    if coords is None:
        coords = getattr(source, "atoms", None)
    if coords is None:
        return np.zeros((0, 3), dtype=float)
    return np.asarray(coords, dtype=float)


def _extract_atoms(source: Any, *, default_type: str = "C") -> list[dict[str, Any]]:
    if source is None:
        return []

    if isinstance(source, dict):
        atoms_obj = source.get("atoms")
        if isinstance(atoms_obj, list):
            out: list[dict[str, Any]] = []
            for item in atoms_obj:
                if isinstance(item, dict) and item.get("coords") is not None:
                    out.append(
                        {
                            "coords": np.asarray(item.get("coords"), dtype=float),
                            "line": str(item.get("line", default_type)),
                        }
                    )
            if out:
                return out

    atoms_attr = getattr(source, "atoms", None)
    if isinstance(atoms_attr, list):
        out = []
        for item in atoms_attr:
            if isinstance(item, dict) and item.get("coords") is not None:
                out.append(
                    {
                        "coords": np.asarray(item.get("coords"), dtype=float),
                        "line": str(item.get("line", default_type)),
                    }
                )
        if out:
            return out

    coords = _extract_coords(source)
    if coords.size == 0:
        return []
    atoms: list[dict[str, Any]] = []
    for coord in np.asarray(coords, dtype=float):
        atoms.append({"coords": np.asarray(coord, dtype=float), "line": default_type})
    return atoms


def _build_ligand_atoms_for_pose(pose: Pose, peptide: Any) -> list[dict[str, Any]]:
    pose_coords = np.asarray(pose.coords, dtype=float)
    if pose_coords.size == 0:
        return []

    template_atoms = _extract_atoms(peptide, default_type="C")
    if len(template_atoms) == int(pose_coords.shape[0]):
        return [
            {"coords": np.asarray(pose_coords[idx], dtype=float), "line": template_atoms[idx].get("line", "C")}
            for idx in range(int(pose_coords.shape[0]))
        ]
    return [{"coords": np.asarray(coord, dtype=float), "line": "C"} for coord in pose_coords]


def _cfg_block(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    if isinstance(cfg, dict):
        if isinstance(cfg.get("rescoring"), dict):
            return dict(cfg["rescoring"])
        if isinstance(cfg.get("expensive"), dict):
            return dict(cfg["expensive"])
        return {}
    rescoring_cfg = getattr(cfg, "rescoring", None)
    if isinstance(rescoring_cfg, dict):
        return dict(rescoring_cfg)
    expensive_cfg = getattr(cfg, "expensive", None)
    if isinstance(expensive_cfg, dict):
        return dict(expensive_cfg)
    return {}


def _cfg_float(cfg: dict[str, Any], key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return float(default)


class VinaGaussian:
    def __init__(self, weight: float, mean: float, stddev: float) -> None:
        self.weight = float(weight)
        self.mean = float(mean)
        self.stddev = float(stddev)
        self._inv_sigma2 = 1.0 / (self.stddev * self.stddev)

    def calcular(self, r: float) -> float:
        x = (r - self.mean)
        return self.weight * float(np.exp(-x * x * self._inv_sigma2))


class VinaRepulsion:
    def __init__(self, weight: float, delta_cap: float = 0.30) -> None:
        self.weight = float(weight)
        self.delta_cap = float(delta_cap)

    def calcular(self, r: float, ri_vdw: float, rj_vdw: float) -> float:
        delta = (ri_vdw + rj_vdw) - r
        if delta <= 0.0:
            return 0.0
        clipped = min(delta, self.delta_cap)
        return self.weight * (clipped * clipped)


class VinaHydrophobic:
    def __init__(self, weight: float, r_min: float = 0.5, r_max: float = 5.5) -> None:
        self.weight = float(weight)
        self.r_min = float(r_min)
        self.r_max = float(r_max)

    def calcular(self, r: float, hyd_i: bool, hyd_j: bool) -> float:
        if hyd_i and hyd_j and self.r_min <= r <= self.r_max:
            return self.weight * (self.r_max - r)
        return 0.0


class VinaHydrogenBond:
    def __init__(self, weight: float, r0: float = 1.9, r_cut: float = 3.3) -> None:
        self.weight = float(weight)
        self.r0 = float(r0)
        self.r_cut = float(r_cut)
        self._width = self.r_cut - self.r0 if self.r_cut > self.r0 else 1.1

    def calcular(
        self,
        r: float,
        donor_i: bool,
        acc_i: bool,
        donor_j: bool,
        acc_j: bool,
    ) -> float:
        has_pair = (donor_i and acc_j) or (donor_j and acc_i)
        if not has_pair or r >= self.r_cut:
            return 0.0
        return self.weight * (1.0 - ((r - self.r0) / self._width) ** 2)


class DistanceDependentCoulomb:
    def __init__(self, weight: float = 1.0, eps0: float = 16.0, r_switch: float = 1.5) -> None:
        self.k_e = 332.0636
        self.eps0 = float(eps0)
        self.weight = float(weight)
        self.r_switch = float(r_switch)

    def calcular(self, r: float, qi: float, qj: float) -> float:
        r_eff = max(r, self.r_switch)
        energy = (self.k_e * qi * qj) / (self.eps0 * r_eff * r_eff)
        return self.weight * energy


class VinaScoringFunction:
    """Pairwise Vina-inspired scorer used as selective final rescoring."""

    def __init__(
        self,
        *,
        cutoff_pair: float = 8.0,
        w_g1: float = -0.035579,
        w_g2: float = -0.005156,
        w_rep: float = 0.840245,
        w_hyd: float = -0.035069,
        w_hb: float = -0.587439,
        w_elec: float = 0.20,
        eps0: float = 16.0,
        r_switch: float = 1.5,
        repulsion_delta_cap: float = 0.30,
        pair_clip_min: float = -20.0,
        pair_clip_max: float = 6.0,
        scale_a: float = 1.20,
        scale_b: float = 0.0,
    ) -> None:
        self.gauss1 = VinaGaussian(w_g1, 0.0, 0.5)
        self.gauss2 = VinaGaussian(w_g2, 3.0, 2.0)
        self.repulsion = VinaRepulsion(w_rep, delta_cap=repulsion_delta_cap)
        self.hydrophobic = VinaHydrophobic(w_hyd, r_min=0.5, r_max=5.5)
        self.hbond = VinaHydrogenBond(w_hb, r0=1.9, r_cut=3.3)
        self.elec = DistanceDependentCoulomb(weight=w_elec, eps0=eps0, r_switch=r_switch)
        self.cutoff_pair = float(cutoff_pair)
        self.pair_clip_min = float(pair_clip_min)
        self.pair_clip_max = float(pair_clip_max)
        self.scale_a = float(scale_a)
        self.scale_b = float(scale_b)

    def _prep_atom(self, atom: dict[str, Any]) -> tuple[np.ndarray, str, float]:
        coords = np.asarray(atom.get("coords"), dtype=float)
        line = str(atom.get("line", ""))
        atype = _parse_type_from_pdbqt(line)
        charge = _parse_charge_from_pdbqt(line)
        return coords, atype, charge

    def avaliar_pose_detalhado(
        self,
        ligante: list[dict[str, Any]],
        receptor: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Returns decomposed Vina-like terms for debug/audit without changing scoring."""

        gauss1_total = 0.0
        gauss2_total = 0.0
        repulsion_total = 0.0
        hydrophobic_total = 0.0
        hbond_total = 0.0
        electrostatic_total = 0.0

        energia_raw = 0.0
        energia_raw_unclipped = 0.0

        n_pairs = 0
        n_pairs_clipped_min = 0
        n_pairs_clipped_max = 0
        min_distance: float | None = None
        max_distance: float | None = None
        sum_distance = 0.0

        receptor_prepped = [self._prep_atom(atom) for atom in receptor]
        receptor_coords = (
            np.asarray([item[0] for item in receptor_prepped], dtype=float)
            if all(item[0].shape == (3,) for item in receptor_prepped)
            else None
        )
        for atom_l in ligante:
            pi, ti, qi = self._prep_atom(atom_l)
            ri_vdw = _vdw_radius(ti)
            hyd_i = _is_hydrophobic(ti)
            don_i = _is_donor(ti)
            acc_i = _is_acceptor(ti)

            for receptor_idx in _candidate_receptor_indices(pi, receptor_coords, self.cutoff_pair, len(receptor_prepped)):
                pj, tj, qj = receptor_prepped[receptor_idx]
                r = float(np.linalg.norm(pi - pj))
                if r > self.cutoff_pair:
                    continue

                rj_vdw = _vdw_radius(tj)
                hyd_j = _is_hydrophobic(tj)
                don_j = _is_donor(tj)
                acc_j = _is_acceptor(tj)

                t_gauss1 = self.gauss1.calcular(r)
                t_gauss2 = self.gauss2.calcular(r)
                t_repulsion = self.repulsion.calcular(r, ri_vdw, rj_vdw)
                t_hydrophobic = self.hydrophobic.calcular(r, hyd_i, hyd_j)
                t_hbond = self.hbond.calcular(r, don_i, acc_i, don_j, acc_j)
                t_electrostatic = 0.0
                if qi != 0.0 and qj != 0.0:
                    t_electrostatic = self.elec.calcular(r, qi, qj)

                pair_unclipped = (
                    t_gauss1
                    + t_gauss2
                    + t_repulsion
                    + t_hydrophobic
                    + t_hbond
                    + t_electrostatic
                )
                pair_clipped = float(np.clip(pair_unclipped, self.pair_clip_min, self.pair_clip_max))

                gauss1_total += t_gauss1
                gauss2_total += t_gauss2
                repulsion_total += t_repulsion
                hydrophobic_total += t_hydrophobic
                hbond_total += t_hbond
                electrostatic_total += t_electrostatic

                energia_raw_unclipped += pair_unclipped
                energia_raw += pair_clipped

                if pair_unclipped < self.pair_clip_min:
                    n_pairs_clipped_min += 1
                elif pair_unclipped > self.pair_clip_max:
                    n_pairs_clipped_max += 1

                n_pairs += 1
                sum_distance += r
                min_distance = r if min_distance is None else min(min_distance, r)
                max_distance = r if max_distance is None else max(max_distance, r)

        energia = self.scale_a * energia_raw + self.scale_b
        final_rescore = -float(energia)
        mean_distance = (sum_distance / float(n_pairs)) if n_pairs > 0 else None

        return {
            "final_rescore": float(final_rescore),
            "final_rescore_energy": float(energia),
            "vina_terms": {
                "gauss1": float(gauss1_total),
                "gauss2": float(gauss2_total),
                "repulsion": float(repulsion_total),
                "hydrophobic": float(hydrophobic_total),
                "hbond": float(hbond_total),
                "electrostatic": float(electrostatic_total),
                "raw_total": float(energia_raw),
                "scaled_total": float(energia),
                # Useful to diagnose how much pair clipping distorts raw term sums.
                "raw_unclipped_total": float(energia_raw_unclipped),
                "scale_a": float(self.scale_a),
                "scale_b": float(self.scale_b),
            },
            "pair_stats": {
                "n_pairs": int(n_pairs),
                "n_pairs_clipped_min": int(n_pairs_clipped_min),
                "n_pairs_clipped_max": int(n_pairs_clipped_max),
                "min_distance": None if min_distance is None else float(min_distance),
                "mean_distance": None if mean_distance is None else float(mean_distance),
                "max_distance": None if max_distance is None else float(max_distance),
                "cutoff_pair": float(self.cutoff_pair),
                "pair_clip_min": float(self.pair_clip_min),
                "pair_clip_max": float(self.pair_clip_max),
            },
        }

    def avaliar_pose(self, ligante: list[dict[str, Any]], receptor: list[dict[str, Any]]) -> float:
        payload = self.avaliar_pose_detalhado(ligante=ligante, receptor=receptor)
        return float(payload["final_rescore_energy"])


def _candidate_receptor_indices(
    point: np.ndarray,
    receptor_coords: np.ndarray | None,
    cutoff: float,
    receptor_count: int,
):
    """Prune only pairs outside a conservative enclosing cube.

    Retained pairs still use the original scalar norm, cutoff test, terms and
    accumulation order. The outward margin protects cutoff-boundary rounding;
    nonfinite coordinates and extreme cutoffs retain the original path.
    """

    if (
        receptor_coords is None
        or receptor_coords.shape != (receptor_count, 3)
        or np.asarray(point).shape != (3,)
        or not np.isfinite(cutoff)
        or cutoff < 1e-100
        or cutoff > 1e100
    ):
        return range(receptor_count)
    margin = max(1.0, abs(cutoff)) * (16.0 * np.finfo(float).eps)
    deltas = point - receptor_coords
    inside = np.all(np.abs(deltas) <= cutoff + margin, axis=1)
    # The scalar implementation propagates NaN distances; do not hide them.
    nonfinite = np.any(~np.isfinite(deltas), axis=1)
    return np.flatnonzero(inside | nonfinite)


def _log_expensive_event(
    cfg: Any,
    name: str,
    pose: Pose,
    reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    logger = getattr(cfg, "expensive_logger", None) or getattr(cfg, "logger", None)
    if logger is None:
        return
    step = int(getattr(cfg, "expensive_step", 0) or 0)
    extra_payload: dict[str, Any] = dict(extra or {})
    if reason is not None:
        extra_payload["reason"] = reason
    generation = pose.meta.get("generation")
    if generation is not None:
        extra_payload["generation"] = generation
    if extra_payload:
        logger.log_metric(name, 1.0, step=step, extra=extra_payload)
    else:
        logger.log_metric(name, 1.0, step=step)


def _log_expensive_skip(cfg: Any, pose: Pose, reason: str, error: str | None = None) -> None:
    extra = {"error": error} if error else None
    # Canonical naming for the new stage.
    _log_expensive_event(cfg, "final_rescore_skipped", pose, reason=reason, extra=extra)
    # Backward-compatible metric naming.
    _log_expensive_event(cfg, "expensive_skipped", pose, reason=reason, extra=extra)


def _build_vina_from_cfg(cfg: Any) -> VinaScoringFunction:
    cfg_block = _cfg_block(cfg)
    return VinaScoringFunction(
        cutoff_pair=_cfg_float(cfg_block, "cutoff_pair", 8.0),
        w_g1=_cfg_float(cfg_block, "w_g1", -0.035579),
        w_g2=_cfg_float(cfg_block, "w_g2", -0.005156),
        w_rep=_cfg_float(cfg_block, "w_rep", 0.840245),
        w_hyd=_cfg_float(cfg_block, "w_hyd", -0.035069),
        w_hb=_cfg_float(cfg_block, "w_hb", -0.587439),
        w_elec=_cfg_float(cfg_block, "w_elec", 0.20),
        eps0=_cfg_float(cfg_block, "eps0", 16.0),
        r_switch=_cfg_float(cfg_block, "r_switch", 1.5),
        repulsion_delta_cap=_cfg_float(cfg_block, "repulsion_delta_cap", 0.30),
        pair_clip_min=_cfg_float(cfg_block, "pair_clip_min", -20.0),
        pair_clip_max=_cfg_float(cfg_block, "pair_clip_max", 6.0),
        scale_a=_cfg_float(cfg_block, "scale_a", 1.20),
        scale_b=_cfg_float(cfg_block, "scale_b", 0.0),
    )


def _apply_final_hard_clash_gate(
    debug_payload: dict[str, Any],
    cfg: Any,
) -> dict[str, Any]:
    """Reject atom-center overlaps that the smooth Vina proxy cannot rescue."""

    cfg_block = _cfg_block(cfg)
    enabled = bool(cfg_block.get("hard_clash_gate_enabled", True))
    threshold = _cfg_float(cfg_block, "hard_clash_min_distance", 1.2)
    penalty_score = _cfg_float(cfg_block, "hard_clash_penalty_score", -1e6)
    pair_stats = debug_payload.get("pair_stats")
    if not isinstance(pair_stats, dict):
        pair_stats = {}
        debug_payload["pair_stats"] = pair_stats
    min_distance_raw = pair_stats.get("min_distance")
    try:
        min_distance = (
            None if min_distance_raw is None else float(min_distance_raw)
        )
    except (TypeError, ValueError):
        min_distance = None
    triggered = bool(
        enabled
        and min_distance is not None
        and np.isfinite(min_distance)
        and min_distance < threshold
    )
    gate_payload = {
        "enabled": bool(enabled),
        "triggered": bool(triggered),
        "min_distance": min_distance,
        "threshold": float(threshold),
        "penalty_score": float(penalty_score),
        "reason": "atom_center_distance_below_threshold" if triggered else None,
    }
    debug_payload["hard_clash_gate"] = gate_payload
    pair_stats["hard_clash_gate_triggered"] = bool(triggered)
    pair_stats["hard_clash_threshold"] = float(threshold)
    if not triggered:
        return debug_payload

    previous_score = float(debug_payload.get("final_rescore", 0.0))
    previous_energy = float(debug_payload.get("final_rescore_energy", -previous_score))
    gate_payload["pre_gate_score"] = previous_score
    gate_payload["pre_gate_energy"] = previous_energy
    vina_terms = debug_payload.get("vina_terms")
    if not isinstance(vina_terms, dict):
        vina_terms = {}
        debug_payload["vina_terms"] = vina_terms
    vina_terms["scaled_total_pre_hard_clash_gate"] = previous_energy
    vina_terms["scaled_total"] = float(-penalty_score)
    debug_payload["final_rescore_energy"] = float(-penalty_score)
    debug_payload["final_rescore"] = float(penalty_score)
    return debug_payload


def _score_pose_expensive_impl(pose: Pose, receptor: Any, peptide: Any, cfg: Any) -> float:
    """Compute selective final rescoring (higher score = better)."""

    ligante = _build_ligand_atoms_for_pose(pose, peptide=peptide)
    receptor_atoms = _extract_atoms(receptor, default_type="C")
    if not receptor_atoms:
        receptor_atoms = _extract_atoms(peptide, default_type="C")

    if not ligante or not receptor_atoms:
        debug_payload = {
            "final_rescore": 0.0,
            "final_rescore_energy": 0.0,
            "vina_terms": {
                "gauss1": 0.0,
                "gauss2": 0.0,
                "repulsion": 0.0,
                "hydrophobic": 0.0,
                "hbond": 0.0,
                "electrostatic": 0.0,
                "raw_total": 0.0,
                "scaled_total": 0.0,
            },
            "pair_stats": {
                "n_pairs": 0,
                "n_pairs_clipped_min": 0,
                "n_pairs_clipped_max": 0,
                "min_distance": None,
                "mean_distance": None,
                "max_distance": None,
            },
        }
        pose.meta["final_rescore_energy"] = 0.0
        pose.meta["final_rescore"] = 0.0
        pose.meta["vina_terms"] = debug_payload["vina_terms"]
        pose.meta["vina_pair_stats"] = debug_payload["pair_stats"]
        pose.meta["vina_rescore_debug"] = debug_payload
        pose.meta["final_score_variant"] = "vina_rescore"
        return 0.0

    vina = _build_vina_from_cfg(cfg)
    debug_payload = vina.avaliar_pose_detalhado(ligante=ligante, receptor=receptor_atoms)
    debug_payload = _apply_final_hard_clash_gate(debug_payload, cfg)
    energy = float(debug_payload["final_rescore_energy"])
    # Keep search/ranking direction as "maximize" in the current pipeline.
    validation_score = float(debug_payload["final_rescore"])
    pose.meta["final_rescore_energy"] = energy
    pose.meta["final_rescore"] = float(validation_score)
    pose.meta["vina_terms"] = debug_payload.get("vina_terms", {})
    pose.meta["vina_pair_stats"] = debug_payload.get("pair_stats", {})
    pose.meta["vina_rescore_debug"] = debug_payload
    pose.meta["final_hard_clash_gate"] = debug_payload.get("hard_clash_gate", {})
    pose.meta["final_score_variant"] = "vina_rescore"
    return float(validation_score)


def debug_pose_final_rescore(
    pose: Pose,
    receptor: Any,
    peptide: Any,
    cfg: Any,
) -> dict[str, Any]:
    """Debug helper that returns decomposed Vina-like final rescoring terms."""

    ligante = _build_ligand_atoms_for_pose(pose, peptide=peptide)
    receptor_atoms = _extract_atoms(receptor, default_type="C")
    if not receptor_atoms:
        receptor_atoms = _extract_atoms(peptide, default_type="C")

    if not ligante or not receptor_atoms:
        return {
            "final_rescore": 0.0,
            "final_rescore_energy": 0.0,
            "vina_terms": {
                "gauss1": 0.0,
                "gauss2": 0.0,
                "repulsion": 0.0,
                "hydrophobic": 0.0,
                "hbond": 0.0,
                "electrostatic": 0.0,
                "raw_total": 0.0,
                "scaled_total": 0.0,
            },
            "pair_stats": {
                "n_pairs": 0,
                "n_pairs_clipped_min": 0,
                "n_pairs_clipped_max": 0,
                "min_distance": None,
                "mean_distance": None,
                "max_distance": None,
            },
        }

    vina = _build_vina_from_cfg(cfg)
    return _apply_final_hard_clash_gate(
        vina.avaliar_pose_detalhado(ligante=ligante, receptor=receptor_atoms),
        cfg,
    )


def score_pose_final_rescore(
    pose: Pose,
    receptor: Any,
    peptide: Any,
    cfg: Any,
) -> Optional[float]:
    """Final selective rescoring (Vina-inspired) for already selected candidates."""

    try:
        score = _score_pose_expensive_impl(pose, receptor, peptide, cfg)
    except NotImplementedError:
        _log_expensive_skip(cfg, pose, reason="not_implemented")
        return None
    except Exception as exc:
        _log_expensive_skip(cfg, pose, reason="exception", error=type(exc).__name__)
        return None
    _log_expensive_event(cfg, "final_rescore_ran", pose)
    _log_expensive_event(cfg, "expensive_ran", pose)
    return score


def score_pose_expensive(
    pose: Pose,
    receptor: Any,
    peptide: Any,
    cfg: Any,
) -> Optional[float]:
    """Backward-compatible alias for final rescoring."""

    return score_pose_final_rescore(pose=pose, receptor=receptor, peptide=peptide, cfg=cfg)
