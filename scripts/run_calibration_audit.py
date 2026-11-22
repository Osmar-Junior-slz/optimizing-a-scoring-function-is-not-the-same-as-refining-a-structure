"""Audit saved docking only; never run a docking engine or install a model."""
from pathlib import Path
import argparse
import hashlib
from io import BytesIO
import json
import os
import sys
import time

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from docking_confidence.application.batch import _validate_manifest
from docking_confidence.application.training_dataset import _read_receipt, _read_failure
from docking_confidence.application.experimental_protocol import bind_experimental_protocol
from docking_confidence.application.calibration_comparison import (
    METHODS, ComparisonOptions, compare_calibrators,
)
from docking_confidence.application.probability_reporting import _resampled_metrics
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.io.config import LOCAL_REFINEMENT_FEATURE_COLUMNS, config_sha256
from docking_confidence.io.dataset import load_paired_dataset


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def source_jobs(root):
    manifests = [(folder, read(folder / "batch_manifest.json")) for folder in sorted(root.glob("shard_*")) if folder.is_dir()]
    representatives = {}
    for _, manifest in manifests:
        _validate_manifest(manifest)
        for target in manifest["plan"]["targets"]:
            fingerprint = target["complex_fingerprint_sha256"]
            if fingerprint not in representatives or target["complex_id"] < representatives[fingerprint]["complex_id"]:
                representatives[fingerprint] = target
    canonical = {target["complex_id"] for target in representatives.values()}
    records = {"uniform": [], "seeded": []}
    provenance = []
    checked = 0
    for folder, manifest in manifests:
        status = read(folder / "batch_status.json")
        if status["state"] not in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
            raise ValueError("audit requires completed saved batches")
        plan = manifest["plan"]
        binding = plan["experimental_protocol"]
        if bind_experimental_protocol(binding["path"], {k: v for k, v in plan.items() if k != "experimental_protocol"}) != binding:
            raise ValueError("experimental protocol changed")
        provenance.append({"manifest": str(folder / "batch_manifest.json"),
                           "manifest_sha256": manifest["manifest_sha256"],
                           "status_file_sha256": hashlib.sha256((folder / "batch_status.json").read_bytes()).hexdigest()})
        jobs = {job["job_id"]: job for job in plan["jobs"]}
        for target in plan["targets"]:
            results = {}
            for job_id, job in jobs.items():
                directory = folder / "targets" / target["complex_id"] / "jobs" / job_id
                receipt = _read_receipt(directory, manifest, target, job)
                failure = _read_failure(directory, manifest, target, job) if receipt is None else None
                if receipt is None and failure is None:
                    raise ValueError("unaccounted saved job")
                result = receipt["engine_result"] if receipt else None
                if result and (result["actual_eval_count"] != 7600 or result["stop_reason"] != "objective_evaluation_ceiling"):
                    raise ValueError("effective evaluation budget mismatch")
                task = status["tasks"][target["complex_id"] + "::" + job_id]
                duration = task.get("duration_seconds")
                if not isinstance(duration, (int, float)) or duration < 0:
                    raise ValueError("saved task duration is unavailable")
                results[job_id] = (result, duration)
                checked += 1
            if target["complex_id"] not in canonical:
                continue
            for initialization in records:
                for algorithm in ("ag", "pso", "hybrid"):
                    reduced, reduced_task = results[f"{algorithm}-reduced-{initialization}-seed1"]
                    full, full_task = results[f"{algorithm}-expanded-{initialization}-seed1"]
                    if full is None:
                        raise ValueError("FULL failures prevent complete operational reconstruction")
                    records[initialization].append({
                        "complex_id": target["complex_id"], "fingerprint": target["complex_fingerprint_sha256"],
                        "algorithm": algorithm, "seed": 1,
                        "reduced_dockq": reduced["dockq"] if reduced else None, "full_dockq": full["dockq"],
                        "reduced_engine_sec": reduced["runtime_sec"] if reduced else None,
                        "full_engine_sec": full["runtime_sec"],
                        "reduced_task_sec": reduced_task, "full_task_sec": full_task,
                        "reduced_failed": reduced is None,
                    })
    if len({manifest["plan"]["input_root"] for _, manifest in manifests}) != 1:
        raise ValueError("follow-up preparation requires one verified input root")
    return records, {"validated_jobs": checked, "canonical_targets": len(canonical),
                     "input_root": manifests[0][1]["plan"]["input_root"],
                     "engine_snapshot": manifests[0][1]["plan"]["engine"],
                     "representatives": representatives, "provenance": provenance}


def paired_intervals(frame, options):
    common = np.logical_and.reduce([np.isfinite(frame[name + "__score"].to_numpy(float)) for name in METHODS])
    f = frame.loc[common]
    groups = f.complex_fingerprint_sha256.to_numpy(str)
    y = f.preserved.to_numpy(int)
    probabilities = {name: f[name + "__score"].to_numpy(float) for name in METHODS}
    probabilities["prevalence"] = f.training_prevalence.to_numpy(float)
    comparisons = [("venn_abers", "uncalibrated"), ("isotonic", "uncalibrated"),
                   ("venn_abers", "isotonic"), ("venn_abers", "prevalence")]
    metric_names = ("brier_score", "log_loss", "expected_calibration_error_10bin", "roc_auc")
    samples = {a + "_minus_" + b: {metric: [] for metric in metric_names} for a, b in comparisons}
    unique = np.unique(groups)
    if len(unique) < 2:
        return {"unavailable_reason": "fewer_than_two_common_groups"}
    members = [np.flatnonzero(groups == group) for group in unique]
    rng = np.random.default_rng(options.seed)
    for _ in range(options.bootstrap_repetitions):
        indices = np.concatenate([members[i] for i in rng.integers(0, len(unique), len(unique))])
        metrics = {name: _resampled_metrics(
            y[indices], p[indices],
            f.base_raw_score.to_numpy(float)[indices] if name == "uncalibrated" else None)
            for name, p in probabilities.items()}
        for a, b in comparisons:
            for metric in metric_names:
                first, second = metrics[a][metric], metrics[b][metric]
                if first is not None and second is not None:
                    samples[a + "_minus_" + b][metric].append(first - second)
    return {key: {metric: {"ci95": np.quantile(values, [.025, .975]).tolist() if len(values) >= .8 * options.bootstrap_repetitions else None,
                           "valid_repetitions": len(values)} for metric, values in result.items()}
            for key, result in samples.items()}


def operational_outcome(records, accepted, run_reduced, rule):
    baseline_engine = sum(row["full_engine_sec"] for row in records)
    baseline_task = sum(row["full_task_sec"] for row in records)
    engine_cost = task_cost = 0.
    engine_missing = false_accepted = avoided = 0
    qualities = []; baseline_quality = []; loss = []
    for row, accept, reduced_enabled in zip(records, accepted, run_reduced):
        if accept and (not reduced_enabled or row["reduced_failed"]):
            raise ValueError("cannot accept a failed or unexecuted REDUCED")
        if reduced_enabled:
            task_cost += row["reduced_task_sec"]
            if row["reduced_engine_sec"] is None:
                engine_missing += 1
            else:
                engine_cost += row["reduced_engine_sec"]
        if not accept:
            engine_cost += row["full_engine_sec"]; task_cost += row["full_task_sec"]
        q = row["reduced_dockq"] if accept else row["full_dockq"]
        if accept:
            avoided += 1
            false_accepted += int(not rule.label(row["reduced_dockq"], row["full_dockq"]))
        qualities.append(q); baseline_quality.append(row["full_dockq"])
        loss.append(row["full_dockq"] - q)
    return {"planned_pairs": len(records), "hypothetical_full_avoided": avoided,
            "actually_avoided_in_saved_experiment": 0, "hypothetical_incorrect_acceptances": false_accepted,
            "empirical_acceptance_risk": false_accepted / avoided if avoided else None,
            "coverage": avoided / len(records), "full_runs_remaining": len(records) - avoided,
            "reduced_runs_used": int(sum(run_reduced)),
            "structural_successes": int(sum(q >= rule.minimum_reduced_dockq for q in qualities)),
            "full_only_structural_successes": int(sum(q >= rule.minimum_reduced_dockq for q in baseline_quality)),
            "mean_quality_loss_vs_full": float(np.mean(loss)),
            "baseline_engine_sec": baseline_engine, "cascade_engine_known_lower_bound_sec": engine_cost,
            "missing_failed_reduced_engine_times": engine_missing,
            "engine_saving_upper_bound_sec": baseline_engine - engine_cost,
            "baseline_sum_task_durations_sec": baseline_task,
            "cascade_sum_task_durations_without_calibrator_sec": task_cost,
            "sum_task_duration_saving_upper_bound_sec": baseline_task - task_cost,
            "complete_calibrator_overhead_measured": False,
            "actual_net_saving_sec": None,
            "scope": "counterfactual replay of measured costs; task durations include failed jobs and benchmark postprocessing; sums are not parallel wall time"}


def policies_on_all_jobs(result, records, rule):
    frame = pd.DataFrame(result["predictions"])
    lookup = {(row.complex_id, row.algorithm, row.seed): row for row in frame.itertuples()}
    group_fold = {group: fold for fold in result["folds"] for group in fold["groups"]["outer_test"]}
    output = {}
    for name in (*METHODS, "venn_abers_lower", "cascade_margin"):
        accept, economic_accept, economic_use = [], [], []
        for row in records:
            prediction = lookup.get((row["complex_id"], row["algorithm"], row["seed"]))
            current = bool(getattr(prediction, name + "__accept")) if prediction is not None else False
            fold = group_fold[row["fingerprint"]]
            enabled = fold.get("policies", {}).get(name, {}).get("economic_engine_only_enabled", False)
            accept.append(current)
            economic_use.append(enabled)
            economic_accept.append(current and enabled)
        output[name] = {"reduced_first": operational_outcome(records, accept, [True] * len(records), rule),
                        "validation_economic_switch_engine_only": operational_outcome(records, economic_accept, economic_use, rule)}
    valid = [not row["reduced_failed"] for row in records]
    preserved = [valid[i] and bool(rule.label(row["reduced_dockq"], row["full_dockq"])) for i, row in enumerate(records)]
    output["bounds_and_baselines"] = {
        "always_valid_reduced_with_failure_fallback": {
            **operational_outcome(records, valid, [True] * len(records), rule),
            "uses_outcomes_for_decision": False,
            "interpretation": "applicable unconstrained baseline, not a calibrated policy or an oracle"},
        "perfect_preservation_oracle": {
            **operational_outcome(records, preserved, [True] * len(records), rule),
            "uses_outcomes_for_decision": True,
            "interpretation": "unavailable future labels; retrospective oracle only"},
    }
    qr = frame.dockq_reduced.to_numpy(float); qf = frame.dockq_full.to_numpy(float)
    finite = np.isfinite(frame.cascade_quality.to_numpy(float))
    output["cascade_regression"] = {"evaluated_pairs": int(finite.sum()),
                                     "reduced_quality_mae": float(np.mean(np.abs(frame.cascade_quality.to_numpy(float)[finite] - qr[finite]))) if finite.any() else None,
                                     "full_gain_mae": float(np.mean(np.abs(frame.cascade_gain.to_numpy(float)[finite] - (qf - qr)[finite]))) if finite.any() else None,
                                     "probability_metrics": None, "reason": "regression outputs are not calibrated probabilities"}
    return output


def plot_comparison(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    labels = {"uncalibrated": "Sem calibração", "isotonic": "Isotônica", "venn_abers": "Venn–Abers"}
    for method in METHODS:
        row = result["metrics"][method]["common_test_rows"]
        curves = row["curves"]; bins = [b for b in curves["reliability_bins"] if b["count"]]
        axes[0].plot([b["mean_probability"] for b in bins], [b["observed_positive_fraction"] for b in bins], "o-", label=labels[method])
        roc = curves["roc"]
        axes[1].plot([b["false_positive_rate"] for b in roc], [b["true_positive_rate"] for b in roc], label=labels[method])
        pr = row["violation_curves"]["precision_recall"]
        axes[2].plot([b["recall"] for b in pr], [b["precision"] for b in pr], label=labels[method])
    axes[0].plot([0, 1], [0, 1], "--", color="gray", alpha=.5)
    axes[1].plot([0, 1], [0, 1], "--", color="gray", alpha=.5)
    for ax, title, xlabel, ylabel in zip(axes, ["Confiabilidade: preservação", "ROC: preservação", "PR: violações"],
        ["Probabilidade estimada", "Falsos positivos", "Sensibilidade"], ["Frequência observada", "Sensibilidade", "Precisão"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1.02))
        ax.grid(alpha=.15); ax.legend(fontsize=8)
    fig.suptitle(f"Mesmas {result['counts']['common_test_rows']} previsões externas; estudo exploratório por grupo")
    fig.savefig(path, dpi=170); plt.close(fig)


def prepare_five_complex_followup(audit, output):
    representatives = audit["representatives"]
    selected = [representatives[key] for key in sorted(representatives)[:5]]
    jobs = []
    for target in selected:
        for algorithm in ("ag", "pso", "hybrid"):
            for extent, budgets in (("reduced", (1900, 3800, 7600)), ("expanded", (7600,))):
                for budget in budgets:
                    jobs.append({"complex_id": target["complex_id"], "algorithm": algorithm,
                                 "seed": 1, "initialization": "uniform", "extent": extent,
                                 "effective_evaluation_budget": budget, "population": 100})
    write(output / "followup_5_spec.json", {
        "scope": "planned_budget_probe_not_executed_or_scientifically_validated",
        "selection": "first five canonical molecular fingerprints in lexicographic SHA256 order; no outcome criterion",
        "targets": selected, "jobs": jobs, "planned_jobs": len(jobs),
        "input_root": audit.get("input_root"), "engine_snapshot": audit.get("engine_snapshot"),
        "required_measurements": ["actual_eval_count", "stop_reason", "reference_dockq_after_search",
                                  "engine_and_task_wall_seconds", "failed_job_cost", "confidence_overhead_sec",
                                  "timing_scope_and_parallelism", "pose_and_input_hashes"],
        "execution_note": "Specification only. A dedicated auditable mixed-budget cascade protocol is required; do not bypass the existing all-four-conditions factorial guard.",
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", default="runs/calibrator_pilot50_fixed_parallel_20261115")
    parser.add_argument("--preparation-root", default="output/calibrator_pilot_50_20261115/fixed_preparations")
    parser.add_argument("--output-dir", default="output/calibration_audit_20261115")
    args = parser.parse_args()
    output = Path(args.output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output / "audit_summary.json").exists():
        raise FileExistsError("audit results already exist; choose a new output directory")
    records, audit = source_jobs(Path(args.study_root).resolve())
    write(output / "saved_job_audit.json", audit)
    write(output / "saved_cost_records.json", records)
    prepare_five_complex_followup(audit, output)
    summary = {"scope": "retrospective_exploratory", "docking_executed": False,
               "automatic_authorization": False, "cases": {}}
    options = ComparisonOptions()
    for initialization in ("uniform", "seeded"):
        source = Path(args.preparation_root) / ("merged_" + initialization)
        preparation, cfg = read(source / "dataset_build_report.json"), read(source / "dataset_config.json")
        snapshot = Path(preparation["dataset_path"]).read_bytes()
        if not preparation["valid"] or hashlib.sha256(snapshot).hexdigest() != preparation["dataset_sha256"]:
            raise ValueError("prepared dataset changed or invalid")
        if config_sha256(cfg) != preparation["config_sha256"]:
            raise ValueError("frozen configuration changed")
        frame = load_paired_dataset(BytesIO(snapshot), cfg["features"]["columns"], cfg["contact_graph"],
                                    docking_task=cfg["docking_task"], feature_schema=cfg["features"]["schema_version"])
        rule = PreservationRule(**cfg["preservation"])
        planned_groups = sorted({row["fingerprint"] for row in records[initialization]})
        for family, columns in (("original11", LOCAL_REFINEMENT_FEATURE_COLUMNS), ("context15", cfg["features"]["columns"])):
            name = initialization + "_" + family
            destination = output / name; destination.mkdir(exist_ok=False)
            result = compare_calibrators(frame, columns, rule=rule, options=options, planned_groups=planned_groups)
            predictions = pd.DataFrame(result["predictions"])
            predictions.to_csv(destination / "predictions.csv", index=False)
            result["paired_group_bootstrap"] = paired_intervals(predictions, options)
            result["operational"] = policies_on_all_jobs(result, records[initialization], rule)
            result["dataset_sha256"] = preparation["dataset_sha256"]
            result["preparation_dir"] = str(source.resolve())
            write(destination / "comparison.json", result)
            plot_comparison(result, destination / "diagnostics.png")
            summary["cases"][name] = {key: result[key] for key in ("counts", "metrics", "prevalence_reference_common", "paired_group_bootstrap", "operational")}
            print(json.dumps({"case": name, "counts": result["counts"],
                              "brier": {m: result["metrics"][m]["common_test_rows"]["preservation"]["brier_score"] for m in METHODS}}), flush=True)
    write(output / "audit_summary.json", summary)
    print("Saved calibration audit complete; no docking or model installation performed.", flush=True)


if __name__ == "__main__":
    main()
