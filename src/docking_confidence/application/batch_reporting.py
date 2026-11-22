from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from typing import Any

from docking_confidence.io.batch_discovery import BatchDiscoveryReport


def discovery_rows(report: BatchDiscoveryReport) -> list[dict[str, Any]]:
    """Return one explicit, user-facing row per recognized target."""

    rows: list[dict[str, Any]] = []
    for target in report.targets:
        rows.append(
            {
                "complex_id": target.complex_id,
                "valid": target.valid,
                "eligible_for_docking": target.eligible,
                "eligibility_reason": target.eligibility_reason,
                "receptor_chain": target.receptor_chain,
                "ligand_chain": target.ligand_chain,
                "receptor_residues": target.receptor_residue_count,
                "ligand_residues": target.ligand_residue_count,
                "receptor_pdb": str(target.receptor_path),
                "ligand_pdb": str(target.ligand_path),
                "reference_pdb": str(target.reference_path),
                "receptor_sha256": target.receptor_sha256,
                "ligand_sha256": target.ligand_sha256,
                "reference_sha256": target.reference_sha256,
                "complex_fingerprint_sha256": target.complex_fingerprint_sha256,
                "errors": " | ".join(target.errors),
            }
        )
    return rows


def discovery_payload(report: BatchDiscoveryReport) -> dict[str, Any]:
    valid = len(report.valid_targets)
    eligible = len(report.eligible_targets)
    return {
        "schema_version": "ppi4dock_discovery_report_v2",
        "input_root": str(report.input_root),
        "validation_performed": report.validation_performed,
        "mobile_partner_domain": report.mobile_partner_domain.as_dict(),
        "counts": {
            "recognized_targets": len(report.targets),
            "valid_targets": valid,
            "invalid_targets": report.invalid_target_count,
            "eligible_for_docking": eligible,
            "outside_configured_domain": sum(
                target.valid and target.eligible is False
                for target in report.targets
            ),
            "root_or_target_issues": len(report.issues),
        },
        "role_rule": (
            "first suffix character = operational receptor; "
            "second suffix character = operational ligand"
        ),
        "targets": discovery_rows(report),
        "issues": [asdict(issue) for issue in report.issues],
    }


def write_discovery_reports(
    report: BatchDiscoveryReport,
    *,
    json_path: str | Path | None = None,
    csv_path: str | Path | None = None,
) -> None:
    """Persist optional JSON and CSV views of the same discovery snapshot."""

    if json_path is not None:
        destination = Path(json_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(
                discovery_payload(report),
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    if csv_path is not None:
        destination = Path(csv_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        rows = discovery_rows(report)
        fieldnames = list(rows[0]) if rows else ["complex_id"]
        with destination.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
