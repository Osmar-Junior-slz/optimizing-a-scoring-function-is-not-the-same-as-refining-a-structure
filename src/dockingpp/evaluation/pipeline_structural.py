"""Persistencia tolerante a falhas das metricas estruturais do pipeline.

As metricas deste modulo sao exclusivamente pos-hoc: elas comparam a pose
final persistida com o complexo de referencia e nunca participam da selecao ou
da otimizacao da busca.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dockingpp.data.pdb_validation import NON_PEPTIDE, validate_peptide_pdb
from dockingpp.evaluation.structural_metrics import evaluate_saved_pose


STRUCTURAL_METRICS_REL_PATH = "artifacts/structural_metrics.json"

STRUCTURAL_METRICS_SUMMARY_KEYS = (
    "structural_metrics_enabled",
    "structural_metrics_available",
    "structural_metrics_status",
    "structural_metrics_error_code",
    "structural_metrics_partner_kind",
    "structural_metrics_output_path",
    "structural_metrics_error",
    "structural_metrics_error_type",
    "structural_metrics_reference_role",
    "structural_metrics_evaluation_role",
    "structural_metrics_pose_output_path",
    "structural_metrics",
    "dockq",
    "dockq_category",
    "fnat",
    "irmsd",
    "lrmsd",
)


def _cfg_value(cfg: Any, key: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact_path(out_dir: Path, relative_or_absolute: str | os.PathLike[str]) -> Path:
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return out_dir / candidate


def _summary_fields(
    *,
    enabled: bool,
    status: str,
    output_path: str | None,
    pose_output_path: str | None,
    metrics: Mapping[str, Any] | None = None,
    error: Exception | None = None,
    error_code: str | None = None,
    partner_kind: str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    metrics_payload = dict(metrics) if isinstance(metrics, Mapping) else None
    error_message = str(error) if error is not None else message
    return {
        "structural_metrics_enabled": bool(enabled),
        "structural_metrics_available": bool(status == "available"),
        "structural_metrics_status": str(status),
        "structural_metrics_error_code": error_code,
        "structural_metrics_partner_kind": partner_kind,
        "structural_metrics_output_path": output_path,
        "structural_metrics_error": error_message,
        "structural_metrics_error_type": (
            None if error is None else type(error).__name__
        ),
        "structural_metrics_reference_role": "input_peptide_as_native_reference",
        "structural_metrics_evaluation_role": "post_hoc_reference_based_validation",
        "structural_metrics_pose_output_path": pose_output_path,
        "structural_metrics": metrics_payload,
        "dockq": None if metrics_payload is None else metrics_payload.get("dockq"),
        "dockq_category": (
            None if metrics_payload is None else metrics_payload.get("dockq_category")
        ),
        "fnat": None if metrics_payload is None else metrics_payload.get("fnat"),
        "irmsd": None if metrics_payload is None else metrics_payload.get("irmsd"),
        "lrmsd": None if metrics_payload is None else metrics_payload.get("lrmsd"),
    }


def evaluate_and_persist_structural_metrics(
    *,
    cfg: Any,
    out_dir: str | os.PathLike[str],
    receptor_path: str | os.PathLike[str],
    peptide_path: str | os.PathLike[str],
    pose_output_path: str | None,
) -> dict[str, Any]:
    """Calcula metricas da pose final e persiste um artefato auditavel.

    Qualquer incompatibilidade de entrada e registrada no artefato e nos
    summaries, sem interromper a execucao de docking.
    """

    enabled = bool(_cfg_value(cfg, "structural_metrics_enabled", True))
    if not enabled:
        return _summary_fields(
            enabled=False,
            status="disabled",
            output_path=None,
            pose_output_path=pose_output_path,
        )

    output_root = Path(out_dir)
    artifact_path = output_root / STRUCTURAL_METRICS_REL_PATH
    try:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return _summary_fields(
            enabled=True,
            status="unavailable",
            output_path=None,
            pose_output_path=pose_output_path,
            error=exc,
        )
    receptor = Path(receptor_path)
    peptide = Path(peptide_path)
    pose = (
        _resolve_artifact_path(output_root, pose_output_path)
        if pose_output_path
        else None
    )
    contact_cutoff = float(
        _cfg_value(cfg, "structural_contact_cutoff_angstrom", 4.0)
    )
    interface_cutoff = float(
        _cfg_value(cfg, "structural_interface_cutoff_angstrom", 8.0)
    )

    metrics_payload: dict[str, Any] | None = None
    error: Exception | None = None
    error_code: str | None = None
    partner_kind: str | None = None
    message: str | None = None
    partner_validation: dict[str, Any] | None = None
    try:
        if pose is None:
            raise ValueError("Final pose artifact is unavailable")
        validation = validate_peptide_pdb(
            peptide,
            # Structural backfills may contain valid peptide references outside
            # the configured operational residue-count search protocol.  Here we validate
            # molecular kind/backbone only; the execution preflight enforces
            # the configured experimental length range.
            min_residues=1,
            max_residues=100_000,
        )
        partner_validation = validation.as_dict()
        if validation.classification == NON_PEPTIDE:
            partner_kind = validation.classification
            status = "not_applicable"
            error_code = "reference_partner_not_peptide"
            message = (
                "DockQ, Fnat, iRMSD and LRMSD are not applicable because the "
                "reference partner does not satisfy the standard-peptide "
                "definition (one chain of ATOM standard amino-acid residues "
                "with a complete N/CA/C/O backbone). "
                f"{validation.message}"
            )
        elif not validation.is_valid:
            partner_kind = validation.classification
            error_code = validation.error_code
            raise ValueError(validation.message)
        else:
            partner_kind = validation.classification
            metrics_payload = evaluate_saved_pose(
                receptor_pdb=receptor,
                native_peptide_pdb=peptide,
                pose_json=pose,
                contact_cutoff_angstrom=contact_cutoff,
                interface_cutoff_angstrom=interface_cutoff,
            ).as_dict()
            status = "available"
    except Exception as exc:  # noqa: BLE001 - falha cientifica nao derruba o docking
        error = exc
        status = "unavailable"

    provenance: dict[str, Any] = {
        "receptor_path": str(receptor),
        "native_peptide_path": str(peptide),
        "pose_path": None if pose is None else str(pose),
        "pose_output_path": pose_output_path,
        "reference_role": "input_peptide_as_native_reference",
        "evaluation_role": "post_hoc_reference_based_validation",
        "reference_partner_validation": partner_validation,
    }
    for key, path in (
        ("receptor_sha256", receptor),
        ("native_peptide_sha256", peptide),
        ("pose_sha256", pose),
    ):
        try:
            provenance[key] = None if path is None else _sha256(path)
        except OSError:
            provenance[key] = None

    artifact_payload = {
        "schema_version": "1.0",
        "status": status,
        "available": bool(status == "available"),
        "error_code": error_code,
        "partner_kind": partner_kind,
        "metrics": metrics_payload,
        "error": (
            None
            if error is None and message is None
            else {
                "type": None if error is None else type(error).__name__,
                "code": error_code,
                "message": str(error) if error is not None else message,
            }
        ),
        "parameters": {
            "contact_cutoff_angstrom": contact_cutoff,
            "interface_cutoff_angstrom": interface_cutoff,
        },
        "provenance": provenance,
    }
    try:
        artifact_path.write_text(
            json.dumps(artifact_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError as exc:
        return _summary_fields(
            enabled=True,
            status="unavailable",
            output_path=None,
            pose_output_path=pose_output_path,
            error=exc,
        )
    return _summary_fields(
        enabled=True,
        status=status,
        output_path=STRUCTURAL_METRICS_REL_PATH,
        pose_output_path=pose_output_path,
        metrics=metrics_payload,
        error=error,
        error_code=error_code,
        partner_kind=partner_kind,
        message=message,
    )


def copy_structural_metrics_fields(
    payload: Mapping[str, Any] | None,
    *,
    path_prefix: str | None = None,
) -> dict[str, Any]:
    """Extrai os campos estruturais e ajusta caminhos para um summary pai."""

    source = payload if isinstance(payload, Mapping) else {}
    fields = {
        key: source.get(key)
        for key in STRUCTURAL_METRICS_SUMMARY_KEYS
        if key in source
    }
    if not fields:
        return _summary_fields(
            enabled=False,
            status="unavailable",
            output_path=None,
            pose_output_path=None,
            error=ValueError("Structural metrics metadata is missing"),
        )
    if path_prefix:
        for key in (
            "structural_metrics_output_path",
            "structural_metrics_pose_output_path",
        ):
            value = fields.get(key)
            if value:
                fields[key] = os.path.join(path_prefix, str(value)).replace("\\", "/")
    return fields


def inject_structural_metrics_into_json(
    path: str | os.PathLike[str],
    structural_fields: Mapping[str, Any],
) -> None:
    """Acrescenta o contrato estrutural a um JSON ja criado pelo pipeline."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload.update(dict(structural_fields))
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = [
    "STRUCTURAL_METRICS_REL_PATH",
    "STRUCTURAL_METRICS_SUMMARY_KEYS",
    "copy_structural_metrics_fields",
    "evaluate_and_persist_structural_metrics",
    "inject_structural_metrics_into_json",
]
