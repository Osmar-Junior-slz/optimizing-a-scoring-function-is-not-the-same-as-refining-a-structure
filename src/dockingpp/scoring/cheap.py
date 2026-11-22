"""Cheap geometric scoring used as fast triage proxy."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from dockingpp.domain.models import Pocket, Pose
from dockingpp.utils.kdtree import build_kdtree


def _coerce_float(
    payload: Dict[str, float] | None,
    keys: Iterable[str],
    default: float,
) -> float:
    if not isinstance(payload, dict):
        return float(default)

    for key in keys:
        if key not in payload:
            continue
        try:
            return float(payload[key])
        except (TypeError, ValueError):
            continue

    return float(default)


def _coerce_int(
    payload: Dict[str, float] | None,
    keys: Iterable[str],
    default: int,
) -> int:
    if not isinstance(payload, dict):
        return int(default)

    for key in keys:
        if key not in payload:
            continue
        try:
            return int(payload[key])
        except (TypeError, ValueError):
            continue

    return int(default)


class MonoObjetivoDockingKDTree:
    """
    Fast geometric objective for triage.

    Versão refatorada:
    - Usa KDTree para menor distância proteína-peptídeo.
    - Penaliza colisão severa.
    - Penaliza colisão moderada.
    - Substitui contato rígido por contato suave.
    - Mantém compatibilidade com calcular_energia(), avaliar_pose_meta()
      e score_pose_cheap().
    """

    __slots__ = (
        "receptor_coords",
        "kd",
        "centro_proteina",
        "limite_exposicao",
        "colisao_severa",
        "colisao_moderada",
        "contato_min",
        "contato_max",
        "contato_ideal",
        "contato_sigma",
        "W_COL",
        "W_EXP",
        "W_BONUS",
        "gate_min_contatos",
        "penalidade_colisao_severa",
    )

    def __init__(
        self,
        receptor_coords: np.ndarray,
        *,
        colisao_severa: float = 1.2,
        colisao_moderada: float = 1.6,
        contato_min: float = 2.0,
        contato_max: float = 4.5,
        contato_ideal: float = 3.0,
        contato_sigma: float = 1.0,
        margem_exposicao: float = 2.0,
        W_COL: float = 10.0,
        W_EXP: float = 1.0,
        W_BONUS: float = 2.1,
        gate_min_contatos: int = 5,
        penalidade_colisao_severa: float = 1e6,
    ) -> None:
        rc = np.asarray(receptor_coords, dtype=np.float32, order="C")

        if rc.ndim != 2 or rc.shape[1] != 3 or rc.shape[0] == 0:
            raise ValueError("receptor_coords must be non-empty and shaped as (N,3)")

        self.receptor_coords = rc
        self.kd = build_kdtree(rc)

        self.centro_proteina = rc.mean(axis=0, dtype=np.float32)

        dif = rc - self.centro_proteina
        raios = np.sqrt(np.sum(dif * dif, axis=1, dtype=np.float32))
        self.limite_exposicao = float(raios.max() + np.float32(margem_exposicao))

        self.colisao_severa = float(colisao_severa)
        self.colisao_moderada = float(colisao_moderada)

        self.contato_min = float(contato_min)
        self.contato_max = float(contato_max)

        # Segurança para não quebrar configurações antigas.
        # Se alguém passar contato_max antigo, por exemplo 3.0,
        # o contato_ideal é reposicionado para o meio da janela.
        contato_ideal_float = float(contato_ideal)
        if not (self.contato_min <= contato_ideal_float <= self.contato_max):
            contato_ideal_float = float((self.contato_min + self.contato_max) / 2.0)

        self.contato_ideal = contato_ideal_float
        self.contato_sigma = float(contato_sigma)

        self.W_COL = float(W_COL)
        self.W_EXP = float(W_EXP)
        self.W_BONUS = float(W_BONUS)

        self.gate_min_contatos = int(gate_min_contatos)
        self.penalidade_colisao_severa = float(penalidade_colisao_severa)

        self._validar_parametros()

    def _validar_parametros(self) -> None:
        if self.colisao_severa <= 0:
            raise ValueError("colisao_severa must be positive")

        if self.colisao_moderada <= self.colisao_severa:
            raise ValueError("colisao_moderada must be greater than colisao_severa")

        if self.contato_min <= self.colisao_moderada:
            raise ValueError("contato_min must be greater than colisao_moderada")

        if self.contato_max <= self.contato_min:
            raise ValueError("contato_max must be greater than contato_min")

        if not (self.contato_min <= self.contato_ideal <= self.contato_max):
            raise ValueError("contato_ideal must be between contato_min and contato_max")

        if self.contato_sigma <= 0:
            raise ValueError("contato_sigma must be positive")

    @staticmethod
    def _as_float32_coords(coords: np.ndarray, name: str) -> np.ndarray:
        arr = np.asarray(coords, dtype=np.float32, order="C")

        if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
            raise ValueError(f"{name} must be non-empty and shaped as (N,3)")

        return arr

    def _query_min(self, lig_pts: np.ndarray) -> np.ndarray:
        lig = self._as_float32_coords(lig_pts, "lig_pts")

        query = getattr(self.kd, "query", None)

        if callable(query):
            try:
                dists, _ = query(lig, k=1, distance_upper_bound=float(self.contato_max))
            except TypeError:
                dists, _ = query(lig, k=1)
            return np.asarray(dists, dtype=np.float32).reshape(-1)

        # Fallback para implementação sem KDTree real.
        best_sq = np.full(lig.shape[0], np.inf, dtype=np.float32)
        chunk_size = 2048
        for start in range(0, self.receptor_coords.shape[0], chunk_size):
            ref = self.receptor_coords[start : start + chunk_size]
            deltas = ref[None, :, :] - lig[:, None, :]
            dist_sq = np.sum(deltas * deltas, axis=2, dtype=np.float32)
            best_sq = np.minimum(best_sq, np.min(dist_sq, axis=1))
        return np.sqrt(best_sq).astype(np.float32, copy=False)

    def _calcular_penalidade_colisao(self, dist: np.ndarray) -> tuple[bool, float]:
        colisao_severa = bool(np.any(dist < self.colisao_severa))

        cm = self.colisao_moderada
        mcol = dist < cm

        if not bool(mcol.any()):
            return colisao_severa, 0.0

        delta = cm - dist[mcol]
        pen_col = float(np.sum(delta * delta, dtype=np.float32))

        return colisao_severa, pen_col

    def _calcular_bonus_contato_suave(self, dist: np.ndarray) -> tuple[float, int]:
        """
        Contato suave.

        Antes:
            contato = 1 se contato_min <= dist <= contato_max

        Agora:
            contato recebe peso maior perto de contato_ideal.
            Isso melhora a suavidade da função objetivo.
        """
        mascara = (dist >= self.contato_min) & (dist <= self.contato_max)
        contatos_binarios = int(np.sum(mascara))

        if not bool(mascara.any()):
            return 0.0, contatos_binarios

        d = dist[mascara]

        z = (d - self.contato_ideal) / self.contato_sigma
        bonus_por_atomo = np.exp(-0.5 * z * z)

        bonus_contato = float(np.sum(bonus_por_atomo, dtype=np.float32))

        return bonus_contato, contatos_binarios

    def _calcular_penalidade_exposicao(self, ligante_coords: np.ndarray) -> float:
        lig = self._as_float32_coords(ligante_coords, "ligante_coords")

        centro_lig = lig.mean(axis=0, dtype=np.float32)
        dx = centro_lig - self.centro_proteina

        dist_centro = float(np.sqrt(np.dot(dx, dx)))
        excess = dist_centro - self.limite_exposicao

        return float(excess * excess) if excess > 0.0 else 0.0

    def _calcular_rmsd_e_penalidade(
        self,
        ligante_coords: np.ndarray,
        coords_ref: np.ndarray | None,
        limite_rmsd: float,
        penalidade_rmsd: float,
        use_rmsd: bool,
    ) -> tuple[float | None, float]:
        if not use_rmsd or coords_ref is None:
            return None, 0.0

        rmsd = self.medir_rmsd(ligante_coords, coords_ref)

        if rmsd is None:
            return None, 0.0

        if rmsd <= limite_rmsd:
            return rmsd, 0.0

        return rmsd, float(penalidade_rmsd * (rmsd - limite_rmsd))

    def avaliar_pose(
        self,
        ligante_coords: np.ndarray,
        coords_ref: np.ndarray | None = None,
        limite_rmsd: float = 2.0,
        penalidade_rmsd: float = 1000.0,
        use_rmsd: bool = False,
    ) -> tuple[float, float | None]:
        lig = self._as_float32_coords(ligante_coords, "ligante_coords")

        dist = self._query_min(lig)

        colisao_severa, pen_col = self._calcular_penalidade_colisao(dist)

        if colisao_severa:
            return self.penalidade_colisao_severa, None

        bonus_contato, _contatos_binarios = self._calcular_bonus_contato_suave(dist)
        pen_exp = self._calcular_penalidade_exposicao(lig)

        rmsd, pen_rmsd = self._calcular_rmsd_e_penalidade(
            ligante_coords=lig,
            coords_ref=coords_ref,
            limite_rmsd=limite_rmsd,
            penalidade_rmsd=penalidade_rmsd,
            use_rmsd=use_rmsd,
        )

        energy = (
            self.W_COL * pen_col
            + self.W_EXP * pen_exp
            - self.W_BONUS * bonus_contato
            + pen_rmsd
        )

        return float(energy), rmsd

    def calcular_energia(
        self,
        ligante_coords: np.ndarray,
        coords_ref: np.ndarray | None = None,
        limite_rmsd: float = 2.0,
        penalidade_rmsd: float = 1000.0,
        use_rmsd: bool = False,
    ) -> tuple[float, float | None]:
        return self.avaliar_pose(
            ligante_coords=ligante_coords,
            coords_ref=coords_ref,
            limite_rmsd=limite_rmsd,
            penalidade_rmsd=penalidade_rmsd,
            use_rmsd=use_rmsd,
        )

    def avaliar_pose_meta(self, ligante_coords: np.ndarray) -> tuple[float, dict[str, float | int | bool]]:
        lig = self._as_float32_coords(ligante_coords, "ligante_coords")

        dist = self._query_min(lig)

        colisao_severa, pen_col = self._calcular_penalidade_colisao(dist)
        bonus_contato, contatos_binarios = self._calcular_bonus_contato_suave(dist)
        pen_exp = self._calcular_penalidade_exposicao(lig)

        if colisao_severa:
            energy = self.penalidade_colisao_severa
        else:
            energy = (
                self.W_COL * pen_col
                + self.W_EXP * pen_exp
                - self.W_BONUS * bonus_contato
            )

        meta = {
            "dmin_global": float(np.min(dist)) if dist.size else float("inf"),
            "mean_d": float(np.mean(dist)) if dist.size else float("inf"),
            "contatos": int(contatos_binarios),
            "contacts": float(contatos_binarios),
            "bonus_contato": float(bonus_contato),
            "contact_bonus": float(bonus_contato),
            "pen_col": float(pen_col),
            "pen_exp": float(pen_exp),
            "colisao_severa": bool(colisao_severa),
            "colisao_severa_limiar": float(self.colisao_severa),
            "colisao_moderada_limiar": float(self.colisao_moderada),
            "contato_min": float(self.contato_min),
            "contato_max": float(self.contato_max),
            "contato_ideal": float(self.contato_ideal),
            "contato_sigma": float(self.contato_sigma),
            "W_COL": float(self.W_COL),
            "W_EXP": float(self.W_EXP),
            "W_BONUS": float(self.W_BONUS),
        }

        return float(energy), meta

    def auto_tune(
        self,
        frac_longe: float,
        frac_colisao: float,
        frac_contatos_bons: float,
    ) -> None:
        """
        Autoajuste leve.

        Pode ser chamado pelo loop de busca a cada algumas gerações,
        usando estatísticas da população.
        """
        if frac_longe > 0.60:
            self.W_EXP = min(self.W_EXP * 1.10, 5.0)
            self.W_BONUS = min(self.W_BONUS * 1.05, 6.0)

        if frac_colisao > 0.15:
            self.W_COL = min(self.W_COL * 1.15, 30.0)

        if frac_contatos_bons < 0.10:
            self.contato_max = float(min(self.contato_max + 0.2, 6.0))
            self.contato_sigma = float(min(self.contato_sigma + 0.05, 1.2))
            self.W_BONUS = min(self.W_BONUS * 1.10, 6.0)

        if frac_contatos_bons > 0.50 and frac_colisao < 0.05:
            self.contato_max = float(max(self.contato_max - 0.1, 5.0))
            self.contato_sigma = float(max(self.contato_sigma - 0.03, 0.6))

        if not (self.contato_min <= self.contato_ideal <= self.contato_max):
            self.contato_ideal = float((self.contato_min + self.contato_max) / 2.0)

        self._validar_parametros()

    def coletar_stats_amostra(
        self,
        lista_ligs: list[np.ndarray],
        amostra: int = 64,
        rng: np.random.Generator | None = None,
    ) -> tuple[float, float, float]:
        if rng is None:
            rng = np.random.default_rng()

        n = len(lista_ligs)

        if n == 0:
            return 0.0, 0.0, 0.0

        idx = rng.choice(n, size=min(amostra, n), replace=False)

        longe = 0
        colisao = 0
        contatos_bons = 0

        for i in idx:
            _, meta = self.avaliar_pose_meta(lista_ligs[i])

            if float(meta["mean_d"]) > self.contato_max:
                longe += 1

            if bool(meta["colisao_severa"]):
                colisao += 1

            if int(meta["contatos"]) >= self.gate_min_contatos:
                contatos_bons += 1

        total = len(idx)

        return longe / total, colisao / total, contatos_bons / total

    def snapshot_parametros(self) -> dict[str, float | int]:
        return {
            "colisao_severa": float(self.colisao_severa),
            "colisao_moderada": float(self.colisao_moderada),
            "contato_min": float(self.contato_min),
            "contato_max": float(self.contato_max),
            "contato_ideal": float(self.contato_ideal),
            "contato_sigma": float(self.contato_sigma),
            "W_COL": float(self.W_COL),
            "W_EXP": float(self.W_EXP),
            "W_BONUS": float(self.W_BONUS),
            "gate_min_contatos": int(self.gate_min_contatos),
            "penalidade_colisao_severa": float(self.penalidade_colisao_severa),
            "limite_exposicao": float(self.limite_exposicao),
        }

    @staticmethod
    def medir_rmsd(ligante_coords: np.ndarray, coords_ref: np.ndarray) -> float | None:
        lig = np.asarray(ligante_coords, dtype=np.float32, order="C")
        ref = np.asarray(coords_ref, dtype=np.float32, order="C")

        if lig.shape != ref.shape:
            return None

        diff = lig - ref

        return float(np.sqrt(np.mean(diff * diff)))


def _resolve_pocket_coords(pocket: Pocket) -> tuple[np.ndarray, str]:
    source = "meta.receptor_coords_full"

    coords = (
        pocket.meta.get("receptor_coords_full")
        if isinstance(pocket.meta, dict)
        else None
    )

    if coords is None and isinstance(pocket.meta, dict):
        source = "meta.coords"
        coords = pocket.meta.get("coords")

    if coords is None and isinstance(pocket.meta, dict):
        source = "meta.atoms"
        coords = pocket.meta.get("atoms")

    if coords is None and getattr(pocket, "coords", None) is not None:
        source = "pocket.coords"
        coords = pocket.coords

    if coords is None:
        source = "center"
        coords = pocket.center.reshape(1, 3)

    return np.asarray(coords, dtype=float), source


def _cheap_geom_config(weights: Dict[str, float]) -> dict[str, float | int]:
    """
    Centraliza os parâmetros para evitar divergência entre cache e avaliador.
    """

    return {
        "colisao_severa": _coerce_float(
            weights,
            ("colisao_severa", "severe_clash_threshold"),
            1.2,
        ),
        "colisao_moderada": _coerce_float(
            weights,
            ("colisao_moderada", "moderate_clash_threshold"),
            1.6,
        ),
        "contato_min": _coerce_float(
            weights,
            ("contato_min", "contact_min"),
            2.0,
        ),
        "contato_max": _coerce_float(
            weights,
            ("contato_max", "contact_max"),
            4.5,
        ),
        "contato_ideal": _coerce_float(
            weights,
            ("contato_ideal", "contact_ideal", "ideal_contact"),
            3.0,
        ),
        "contato_sigma": _coerce_float(
            weights,
            ("contato_sigma", "contact_sigma", "smooth_contact_sigma"),
            1.0,
        ),
        "margem_exposicao": _coerce_float(
            weights,
            ("margem_exposicao", "exposure_margin"),
            2.1,
        ),
        "W_COL": _coerce_float(
            weights,
            ("w_col", "W_COL"),
            10.0,
        ),
        "W_EXP": _coerce_float(
            weights,
            ("w_exp", "W_EXP"),
            1.0,
        ),
        "W_BONUS": _coerce_float(
            weights,
            ("w_bonus", "W_BONUS"),
            2.0,
        ),
        "gate_min_contatos": _coerce_int(
            weights,
            ("gate_min_contatos",),
            5,
        ),
        "penalidade_colisao_severa": _coerce_float(
            weights,
            ("penalidade_colisao_severa", "severe_clash_penalty"),
            1e6,
        ),
    }


def _config_signature(
    *,
    config: dict[str, float | int],
    pocket_coords: np.ndarray,
) -> tuple:
    """
    Assinatura simples para cache.

    Importante:
    antes o cache guardava um avaliador por pocket e reutilizava sempre.
    Se você mudasse contato_max, W_BONUS etc., poderia continuar usando
    o avaliador antigo. Agora o cache só reaproveita se a configuração
    for a mesma.
    """
    return (
        tuple(sorted(config.items())),
        tuple(pocket_coords.shape),
    )


def _build_or_get_cheap_evaluator(
    *,
    pocket: Pocket,
    weights: Dict[str, float],
) -> MonoObjetivoDockingKDTree | None:
    pocket_coords, _source = _resolve_pocket_coords(pocket)

    if pocket_coords.size == 0:
        return None

    config = _cheap_geom_config(weights)

    cache_key = "_cheap_geom_kdtree_evaluator"
    signature_key = "_cheap_geom_kdtree_evaluator_signature"

    signature = _config_signature(config=config, pocket_coords=pocket_coords)

    cached = pocket.meta.get(cache_key) if isinstance(pocket.meta, dict) else None
    cached_signature = pocket.meta.get(signature_key) if isinstance(pocket.meta, dict) else None

    if isinstance(cached, MonoObjetivoDockingKDTree) and cached_signature == signature:
        return cached

    evaluator = MonoObjetivoDockingKDTree(
        receptor_coords=pocket_coords,
        colisao_severa=float(config["colisao_severa"]),
        colisao_moderada=float(config["colisao_moderada"]),
        contato_min=float(config["contato_min"]),
        contato_max=float(config["contato_max"]),
        contato_ideal=float(config["contato_ideal"]),
        contato_sigma=float(config["contato_sigma"]),
        margem_exposicao=float(config["margem_exposicao"]),
        W_COL=float(config["W_COL"]),
        W_EXP=float(config["W_EXP"]),
        W_BONUS=float(config["W_BONUS"]),
        gate_min_contatos=int(config["gate_min_contatos"]),
        penalidade_colisao_severa=float(config["penalidade_colisao_severa"]),
    )

    if isinstance(pocket.meta, dict):
        pocket.meta[cache_key] = evaluator
        pocket.meta[signature_key] = signature

    return evaluator


def score_pose_cheap(pose: Pose, pocket: Pocket, weights: Dict[str, float]) -> float:
    """
    Compute the cheap triage score.

    Observação:
    o loop de busca maximiza score.
    A função geométrica retorna energia, onde menor é melhor.
    Por isso aqui usamos score = -energy.
    """

    pose_coords = np.asarray(pose.coords, dtype=float)
    pocket_coords, pocket_coords_source = _resolve_pocket_coords(pocket)

    if pose_coords.size == 0 or pocket_coords.size == 0:
        debug_logger = pocket.meta.get("debug_logger") if isinstance(pocket.meta, dict) else None

        if debug_logger is not None:
            reason = "empty_pose_coords" if pose_coords.size == 0 else "empty_pocket_coords"
            debug_logger.log(
                {
                    "type": "cheap_score_zero",
                    "reason": reason,
                    "pose_coords_n": int(pose_coords.shape[0]),
                    "pocket_coords_n": int(pocket_coords.shape[0]),
                    "pocket_id": getattr(pocket, "id", None),
                    "pocket_coords_source": pocket_coords_source,
                }
            )

        pose.meta["cheap_score_variant"] = "geom_kdtree_smooth_contact"
        pose.meta["cheap_energy"] = None

        return 0.0

    evaluator = _build_or_get_cheap_evaluator(pocket=pocket, weights=weights)

    if evaluator is None:
        pose.meta["cheap_score_variant"] = "geom_kdtree_smooth_contact"
        pose.meta["cheap_energy"] = None

        return 0.0

    energy, meta = evaluator.avaliar_pose_meta(pose_coords)

    score = -float(energy)

    contatos = int(meta.get("contatos", 0))
    colisao_severa = bool(meta.get("colisao_severa", False))

    pose.meta["dmin_global"] = float(meta.get("dmin_global", float("inf")))
    pose.meta["mean_d"] = float(meta.get("mean_d", float("inf")))

    # Compatibilidade com código antigo
    pose.meta["contatos"] = contatos
    pose.meta["contacts"] = float(contatos)
    pose.meta["colisao_severa"] = colisao_severa
    pose.meta["clashes"] = 1.0 if colisao_severa else 0.0

    # Novos metadados úteis para debug e gráfico
    pose.meta["bonus_contato"] = float(meta.get("bonus_contato", 0.0))
    pose.meta["contact_bonus"] = float(meta.get("contact_bonus", 0.0))
    pose.meta["pen_col"] = float(meta.get("pen_col", 0.0))
    pose.meta["pen_exp"] = float(meta.get("pen_exp", 0.0))

    pose.meta["cheap_energy"] = float(energy)
    pose.meta["cheap_score_variant"] = "geom_kdtree_smooth_contact"
    pose.meta["cheap_receptor_coords_source"] = pocket_coords_source
    pose.meta["cheap_receptor_coords_n"] = int(pocket_coords.shape[0])

    pose.meta["cheap_geom_params"] = evaluator.snapshot_parametros()

    return float(score)
