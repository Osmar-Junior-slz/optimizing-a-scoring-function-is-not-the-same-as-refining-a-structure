"""Auditable batch evaluation of persisted docking poses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from dockingpp.data.pdb_validation import (
    NON_PEPTIDE,
    PeptidePDBValidation,
    validate_peptide_pdb,
)
from dockingpp.evaluation.structural_metrics import (
    evaluate_saved_pose,
    export_saved_pose_complex,
    read_pose_json_coordinates,
)
from dockingpp.evaluation.pipeline_structural import (
    copy_structural_metrics_fields,
    evaluate_and_persist_structural_metrics,
    inject_structural_metrics_into_json,
)
from dockingpp.data.io import load_pdb_coords
from dockingpp.domain.models import Pose
from dockingpp.scoring.expensive import debug_pose_final_rescore


DEFAULT_COMPLETE_PROTEINS = ("1A1N", "1BE9", "2BBA", "2BVS")


@dataclass(frozen=True)
class PoseJob:
    protein: str
    experiment: str
    round_name: str
    mode: str
    heuristic: str
    algorithm: str
    child_run: str | None
    experiment_dir: Path
    receptor_pdb: Path
    peptide_pdb: Path
    pose_json: Path
    result_dir: Path

    @property
    def source_key(self) -> str:
        parts = (
            self.protein,
            self.experiment,
            self.round_name,
            self.mode,
            self.heuristic,
            self.algorithm,
        )
        if self.child_run:
            parts = (*parts, self.child_run)
        return "/".join(parts)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_result_pose_path(
    relative: Path,
) -> tuple[str, str, str, str, str | None] | None:
    """Parse both the current mode-first tree and the legacy round-first tree."""

    if relative.parts[-2:] != ("artifacts", "best_pose.json"):
        return None
    run_parts = relative.parts[:-2]
    if len(run_parts) not in {4, 5}:
        return None

    first, second, heuristic, algorithm = run_parts[:4]
    if first.lower().startswith("rodada_"):
        round_name, mode = first, second
    elif second.lower().startswith("rodada_"):
        mode, round_name = first, second
    else:
        return None

    mode = mode.lower()
    if mode not in {"full", "reduced"}:
        return None
    child_run = run_parts[4] if len(run_parts) == 5 and mode == "reduced" else None
    if len(run_parts) == 5 and child_run is None:
        return None
    return round_name, mode, heuristic, algorithm, child_run


def discover_pose_jobs(
    runs_root: str | Path,
    proteins: Sequence[str] = DEFAULT_COMPLETE_PROTEINS,
) -> list[PoseJob]:
    """Discover only canonical result-tree poses for explicitly selected proteins."""

    root = Path(runs_root)
    jobs: list[PoseJob] = []
    for protein in proteins:
        protein_dir = root / protein
        if not protein_dir.is_dir():
            continue
        for experiment_dir in sorted(path for path in protein_dir.iterdir() if path.is_dir()):
            receptor = experiment_dir / "inputs" / "receptor.pdb"
            peptide = experiment_dir / "inputs" / "peptide.pdb"
            results_dir = experiment_dir / "results"
            if not results_dir.is_dir():
                continue
            for pose in sorted(results_dir.rglob("artifacts/best_pose.json")):
                relative = pose.relative_to(results_dir)
                parsed = _parse_result_pose_path(relative)
                if parsed is None:
                    continue
                round_name, mode, heuristic, algorithm, child_run = parsed
                jobs.append(
                    PoseJob(
                        protein=protein,
                        experiment=experiment_dir.name,
                        round_name=round_name,
                        mode=mode,
                        heuristic=heuristic,
                        algorithm=algorithm,
                        child_run=child_run,
                        experiment_dir=experiment_dir,
                        receptor_pdb=receptor,
                        peptide_pdb=peptide,
                        pose_json=pose,
                        result_dir=pose.parent.parent,
                    )
                )
    return jobs


def _load_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


@lru_cache(maxsize=32)
def _load_molecule(path: str):
    return load_pdb_coords(path)


def evaluate_pose_job(
    job: PoseJob,
    *,
    export_pdb_dir: Path | None = None,
    include_element_rescore: bool = False,
) -> dict:
    if not job.receptor_pdb.is_file():
        raise FileNotFoundError(f"Missing receptor: {job.receptor_pdb}")
    if not job.peptide_pdb.is_file():
        raise FileNotFoundError(f"Missing peptide: {job.peptide_pdb}")

    metrics = evaluate_saved_pose(
        receptor_pdb=job.receptor_pdb,
        native_peptide_pdb=job.peptide_pdb,
        pose_json=job.pose_json,
    )
    result = _load_json(job.result_dir / "result.json")
    summary = _load_json(job.result_dir / "summary.json")
    config = _load_json(job.result_dir / "config.resolved.json")

    row = {
        "source_key": job.source_key,
        "protein": job.protein,
        "experiment": job.experiment,
        "round": job.round_name,
        "mode": job.mode,
        "heuristic": job.heuristic,
        "algorithm": job.algorithm,
        "run_id": summary.get("run_id") or result.get("run_id"),
        "seed": summary.get("seed", config.get("seed")),
        "runtime_sec": summary.get("runtime_sec"),
        "n_eval_total": summary.get("n_eval_total"),
        "best_score_cheap": summary.get("best_score_cheap", result.get("best_score_cheap")),
        "final_rescore": summary.get("final_rescore", result.get("final_rescore")),
        "generations_saved": summary.get("generations_saved_by_confidence_stop"),
        "pose_sha256": sha256_file(job.pose_json),
        "receptor_sha256": sha256_file(job.receptor_pdb),
        "peptide_sha256": sha256_file(job.peptide_pdb),
        "pose_path": str(job.pose_json),
        **metrics.as_dict(),
    }
    if include_element_rescore:
        receptor_with_types = _load_molecule(str(job.receptor_pdb.resolve()))
        peptide_with_types = _load_molecule(str(job.peptide_pdb.resolve()))
        element_rescore = debug_pose_final_rescore(
            Pose(coords=read_pose_json_coordinates(job.pose_json)),
            receptor_with_types,
            peptide_with_types,
            config,
        )
        element_terms = element_rescore.get("vina_terms", {})
        row.update(
            {
                "pdb_element_rescore": element_rescore.get("final_rescore"),
                "pdb_element_rescore_energy": element_rescore.get("final_rescore_energy"),
                "pdb_element_hbond": element_terms.get("hbond"),
                "pdb_element_hydrophobic": element_terms.get("hydrophobic"),
                "pdb_element_repulsion": element_terms.get("repulsion"),
            }
        )

    if export_pdb_dir is not None:
        output_pdb = export_pdb_dir / f"{job.source_key.replace('/', '__')}.pdb"
        output_pdb.parent.mkdir(parents=True, exist_ok=True)
        export_saved_pose_complex(
            receptor_pdb=job.receptor_pdb,
            native_peptide_pdb=job.peptide_pdb,
            pose_json=job.pose_json,
            output_pdb=output_pdb,
        )
        row["exported_complex_pdb"] = str(output_pdb)
    return row


def persist_structural_metrics_for_job(job: PoseJob) -> dict:
    """Persist structural metric artifacts and update legacy summaries."""

    fields = evaluate_and_persist_structural_metrics(
        cfg={},
        out_dir=job.result_dir,
        receptor_path=job.receptor_pdb,
        peptide_path=job.peptide_pdb,
        pose_output_path="artifacts/best_pose.json",
    )
    updated_files: list[str] = []
    for filename in ("summary.json", "result.json"):
        target = job.result_dir / filename
        if target.is_file():
            inject_structural_metrics_into_json(target, fields)
            updated_files.append(str(target))

    parent_fields: dict | None = None
    parent_dir = job.result_dir.parent
    parent_summary = _load_json(parent_dir / "summary.json")
    parent_pose_reference = parent_summary.get("pose_output_path")
    parent_references_job_pose = False
    if isinstance(parent_pose_reference, str) and parent_pose_reference.strip():
        normalized_reference = parent_pose_reference.replace("\\", os.sep).replace(
            "/", os.sep
        )
        referenced_pose = parent_dir / normalized_reference
        parent_references_job_pose = os.path.normcase(
            str(referenced_pose.resolve(strict=False))
        ) == os.path.normcase(str(job.pose_json.resolve(strict=False)))

    is_reduced_child = (
        job.mode == "reduced"
        and (
            job.result_dir.name.startswith(("auto_grid_", "auto_surface_"))
            or job.result_dir.name == "fallback_full"
        )
    )
    if is_reduced_child and parent_references_job_pose:
        parent_fields = copy_structural_metrics_fields(
            fields,
            path_prefix=job.result_dir.name,
        )
        for filename in ("summary.json", "result.json"):
            target = parent_dir / filename
            if target.is_file():
                inject_structural_metrics_into_json(target, parent_fields)
                updated_files.append(str(target))

    return {
        "source_key": job.source_key,
        "available": fields.get("structural_metrics_available"),
        "status": fields.get("structural_metrics_status"),
        "dockq": fields.get("dockq"),
        "fnat": fields.get("fnat"),
        "irmsd": fields.get("irmsd"),
        "lrmsd": fields.get("lrmsd"),
        "artifact_path": str(job.result_dir / "artifacts" / "structural_metrics.json"),
        "updated_files": updated_files,
        "parent_updated": parent_fields is not None,
    }


def _validate_job_reference(job: PoseJob) -> PeptidePDBValidation:
    """Validate the reference before any structural metric is calculated."""

    return validate_peptide_pdb(
        job.peptide_pdb,
        # The batch evaluator validates molecular kind/backbone.  Peptide
        # length limits belong to the execution protocol, not to backfills.
        min_residues=1,
        max_residues=100_000,
    )


def _not_applicable_job(
    job: PoseJob,
    validation: PeptidePDBValidation,
) -> dict:
    """Build the audit record for a molecular non-peptide reference."""

    return {
        "source_key": job.source_key,
        "status": "not_applicable",
        "available": False,
        "error_code": validation.error_code,
        "partner_kind": validation.classification,
        "message": validation.message,
        "peptide_path": str(job.peptide_pdb),
    }


def _duplicate_groups(rows: Iterable[dict], field: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return [
        {
            field: value,
            "count": len(group),
            "proteins": sorted({str(item["protein"]) for item in group}),
            "experiments": sorted({str(item["experiment"]) for item in group}),
            "rounds": sorted({str(item["round"]) for item in group}),
            "modes": sorted({str(item["mode"]) for item in group}),
            "heuristics": sorted({str(item["heuristic"]) for item in group}),
            "algorithms": sorted({str(item["algorithm"]) for item in group}),
            "source_keys": sorted(str(item["source_key"]) for item in group),
        }
        for value, group in sorted(groups.items())
        if len(group) > 1
    ]


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    cleaned = [float(value) for value in values]
    return {
        "n": len(cleaned),
        "mean": statistics.fmean(cleaned),
        "median": statistics.median(cleaned),
        "stdev": statistics.stdev(cleaned) if len(cleaned) > 1 else 0.0,
        "min": min(cleaned),
        "max": max(cleaned),
    }


def _numeric_summary_from_rows(rows: Sequence[dict], name: str) -> dict[str, float | int | None]:
    values = [
        float(row[name])
        for row in rows
        if row.get(name) is not None
    ]
    if not values:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "stdev": None,
            "min": None,
            "max": None,
        }
    return _numeric_summary(values)


def summarize_rows(rows: Sequence[dict]) -> dict:
    by_protein = Counter(str(row["protein"]) for row in rows)
    by_algorithm = Counter(str(row["algorithm"]) for row in rows)
    by_heuristic = Counter(str(row["heuristic"]) for row in rows)
    categories = Counter(str(row["dockq_category"]) for row in rows)
    grouped_metrics: dict[str, dict] = {}
    for fields in (("algorithm",), ("protein", "algorithm"), ("algorithm", "heuristic")):
        groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
        for row in rows:
            groups[tuple(str(row[field]) for field in fields)].append(row)
        label = "_by_" + "_and_".join(fields)
        grouped_metrics[label] = {
            "/".join(key): {
                name: _numeric_summary_from_rows(group, name)
                for name in ("lrmsd", "irmsd", "fnat", "dockq", "runtime_sec")
            }
            for key, group in sorted(groups.items())
        }

    return {
        "rows": len(rows),
        "counts_by_protein": dict(sorted(by_protein.items())),
        "counts_by_algorithm": dict(sorted(by_algorithm.items())),
        "counts_by_heuristic": dict(sorted(by_heuristic.items())),
        "dockq_categories": dict(sorted(categories.items())),
        "metrics": {
            name: _numeric_summary_from_rows(rows, name)
            for name in ("lrmsd", "irmsd", "fnat", "dockq")
        },
        **grouped_metrics,
    }


def write_batch_outputs(
    *,
    rows: Sequence[dict],
    errors: Sequence[dict],
    jobs: Sequence[PoseJob],
    proteins: Sequence[str],
    output_dir: str | Path,
    expected_per_protein: int | None,
    persisted: Sequence[dict] = (),
    not_applicable: Sequence[dict] = (),
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0]) if rows else []
    with (output / "structural_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(rows)

    duplicate_sources = _duplicate_groups(rows, "source_key")
    duplicate_poses = _duplicate_groups(rows, "pose_sha256")
    duplicate_run_ids = _duplicate_groups(
        [row for row in rows if row.get("run_id")], "run_id"
    )
    cross_protein_duplicates = [
        group for group in duplicate_poses if len(group["proteins"]) > 1
    ]
    paired_condition_duplicates = [
        group
        for group in duplicate_poses
        if group["count"] == 2
        and len(group["proteins"]) == 1
        and len(group["experiments"]) == 1
        and len(group["rounds"]) == 1
        and len(group["modes"]) == 1
        and len(group["algorithms"]) == 1
        and group["heuristics"] == ["com_heuristica", "sem_heuristica"]
    ]
    paired_hashes = {group["pose_sha256"] for group in paired_condition_duplicates}
    unexpected_pose_duplicates = [
        group for group in duplicate_poses if group["pose_sha256"] not in paired_hashes
    ]
    discovered_counts = Counter(job.protein for job in jobs)
    reference_applicable_jobs = max(len(jobs) - len(not_applicable), 0)
    unsuccessful_applicable_jobs = max(reference_applicable_jobs - len(rows), 0)
    count_mismatches = {
        protein: {
            "expected": expected_per_protein,
            "discovered": discovered_counts.get(protein, 0),
        }
        for protein in proteins
        if expected_per_protein is not None
        and discovered_counts.get(protein, 0) != expected_per_protein
    }
    audit = {
        "selected_proteins": list(proteins),
        "excluded_historical_runs": {
            "1A4W": "QWE is a small molecule and peptide chain I remained in the receptor",
            "1B57": "PGH is a small molecule, not a peptide reference",
        },
        "expected_per_protein": expected_per_protein,
        "expected_total": (
            expected_per_protein * len(proteins)
            if expected_per_protein is not None
            else None
        ),
        "discovered_jobs": len(jobs),
        "successful_rows": len(rows),
        "successful_structural_metric_rows": len(rows),
        "failed_rows": len(errors),
        "not_applicable_rows": len(not_applicable),
        "reference_applicable_jobs": reference_applicable_jobs,
        "unsuccessful_applicable_jobs": unsuccessful_applicable_jobs,
        "has_successful_structural_metrics": bool(rows),
        "scientifically_complete": not bool(not_applicable or errors),
        "persisted_structural_metrics": len(persisted),
        "persisted_structural_metrics_available": sum(
            1 for item in persisted if item.get("available")
        ),
        "discovered_counts": dict(sorted(discovered_counts.items())),
        "count_mismatches": count_mismatches,
        "duplicate_source_keys": duplicate_sources,
        "duplicate_run_ids": duplicate_run_ids,
        "duplicate_pose_groups": duplicate_poses,
        "paired_condition_duplicate_pose_groups": paired_condition_duplicates,
        "unexpected_duplicate_pose_groups": unexpected_pose_duplicates,
        "cross_protein_duplicate_pose_groups": cross_protein_duplicates,
        "errors": list(errors),
        "not_applicable": list(not_applicable),
        "persisted": list(persisted),
        "passed": not (
            not rows
            or not_applicable
            or errors
            or duplicate_sources
            or duplicate_run_ids
            or unexpected_pose_duplicates
            or count_mismatches
        ),
    }
    (output / "audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    summary = summarize_rows(rows) if rows else {"rows": 0}
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def run_batch(
    *,
    runs_root: str | Path,
    proteins: Sequence[str] = DEFAULT_COMPLETE_PROTEINS,
    output_dir: str | Path,
    expected_per_protein: int | None = 60,
    export_pdb: bool = False,
    include_element_rescore: bool = False,
    persist_to_runs: bool = False,
) -> dict:
    jobs = discover_pose_jobs(runs_root, proteins)
    rows: list[dict] = []
    errors: list[dict] = []
    persisted: list[dict] = []
    not_applicable: list[dict] = []
    export_dir = Path(output_dir) / "complexes" if export_pdb else None
    for job in jobs:
        reference_validation = _validate_job_reference(job)
        if reference_validation.classification == NON_PEPTIDE:
            not_applicable.append(
                _not_applicable_job(job, reference_validation)
            )
            if persist_to_runs:
                try:
                    persisted.append(persist_structural_metrics_for_job(job))
                except Exception as exc:  # noqa: BLE001 - audit persistence failure
                    persisted.append(
                        {
                            "source_key": job.source_key,
                            "available": False,
                            "status": "failed_to_persist",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                    errors.append(
                        {
                            "source_key": job.source_key,
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                            "stage": "persist_not_applicable",
                        }
                    )
            continue
        if not reference_validation.is_valid:
            errors.append(
                {
                    "source_key": job.source_key,
                    "error_type": "InvalidPeptideReference",
                    "error_code": reference_validation.error_code,
                    "message": reference_validation.message,
                    "stage": "reference_validation",
                }
            )
            if persist_to_runs:
                try:
                    persisted.append(persist_structural_metrics_for_job(job))
                except Exception as exc:  # noqa: BLE001 - audit persistence failure
                    persisted.append(
                        {
                            "source_key": job.source_key,
                            "available": False,
                            "status": "failed_to_persist",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
            continue
        try:
            rows.append(
                evaluate_pose_job(
                    job,
                    export_pdb_dir=export_dir,
                    include_element_rescore=include_element_rescore,
                )
            )
            if persist_to_runs:
                persisted.append(persist_structural_metrics_for_job(job))
        except Exception as exc:  # Preserve technical failures in the audit report.
            persisted_result: dict | None = None
            if persist_to_runs:
                try:
                    persisted_result = persist_structural_metrics_for_job(job)
                    persisted.append(persisted_result)
                except Exception as persist_exc:  # noqa: BLE001 - audit all failures
                    persisted.append(
                        {
                            "source_key": job.source_key,
                            "available": False,
                            "status": "failed_to_persist",
                            "error_type": type(persist_exc).__name__,
                            "message": str(persist_exc),
                        }
                    )
            if (
                persisted_result is not None
                and persisted_result.get("status") == "not_applicable"
            ):
                continue
            errors.append(
                {
                    "source_key": job.source_key,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
    rows.sort(key=lambda row: str(row["source_key"]))
    return write_batch_outputs(
        rows=rows,
        errors=errors,
        jobs=jobs,
        proteins=proteins,
        output_dir=output_dir,
        expected_per_protein=expected_per_protein,
        persisted=persisted,
        not_applicable=not_applicable,
    )
