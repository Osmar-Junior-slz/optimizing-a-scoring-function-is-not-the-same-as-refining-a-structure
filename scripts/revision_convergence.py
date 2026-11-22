"""Convergence of the recorded factorial searches on an objective-evaluation axis."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import median
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from docking_confidence.application.factorial_reporting import CONDITIONS, _inside, _json

METHOD_NAMES = {"ag": "Algoritmo genético", "pso": "Enxame de partículas", "hybrid": "Busca híbrida"}
CONDITION_STYLES = {
    "reduced_uniform": ("Estreita, uniforme", "#0072B2", "-"),
    "reduced_seeded": ("Estreita, com pose", "#0072B2", "--"),
    "expanded_uniform": ("Ampla, uniforme", "#B44812", "-"),
    "expanded_seeded": ("Ampla, com pose", "#B44812", "--"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_trace(study, row):
    """Resolve the exact run referenced by the primary observation (also on rejection)."""
    job = _inside(study, f"{row['batch_shard']}/batch/targets/{row['complex_id']}/jobs/{row['job_id']}")
    if row["outcome_state"] == "scored":
        receipt_path = job / "receipt.json"
        if digest(receipt_path) != row["receipt_file_sha256"]:
            raise ValueError(f"Receipt hash mismatch: {receipt_path}")
        receipt = _json(receipt_path)
        result = _inside(job, receipt["engine_result"]["engine_result_path"])
    elif row["outcome_state"] == "no_valid_pose":
        if digest(job / "error.json") != row["error_file_sha256"]:
            raise ValueError(f"Failure record hash mismatch: {job}")
        matches = [p for p in job.glob("engine/dockingpp_runs/*/output/result.json")
                   if digest(p) == row["engine_result_sha256"]]
        if len(matches) != 1:
            raise ValueError(f"Missing or ambiguous rejected attempt: {job}")
        result = matches[0]
    else:
        raise ValueError(f"Unresolved observation: {job}")
    # Narrow search has a single refinement pocket. Its wrapper logs only aggregate metrics.
    directory = result.parent
    if row["condition"].startswith("reduced_"):
        directory = directory / "local_refinement"
    if not result.is_file():
        raise FileNotFoundError(result)
    return directory / "metrics.jsonl"


def checkpoints(contents, budget):
    """Read completed generations and maximize over their recorded best cheap scores."""
    generations = defaultdict(dict)
    for line in contents:
        # Avoid decoding unrelated events and the much larger initial population record.
        if b'"best_score"' not in line and b'"n_eval"' not in line:
            continue
        event = json.loads(line)
        key = event.get("name")
        if key not in ("best_score", "n_eval"):
            continue
        generation = event.get("generation")
        if type(generation) is not int or generation < 0 or key in generations[generation]:
            raise ValueError("Invalid or duplicated generation metric")
        value = event["value"]
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"Invalid {key} in generation {generation}")
        generations[generation][key] = value
    if not generations or sorted(generations) != list(range(len(generations))):
        raise ValueError("Missing generation or empty search trace")
    evaluations, best = 0, -math.inf
    result = []
    for generation in range(len(generations)):
        metrics = generations[generation]
        if metrics.keys() != {"best_score", "n_eval"}:
            raise ValueError(f"Incomplete generation {generation}")
        calls = metrics["n_eval"]
        if int(calls) != calls or calls < 0:
            raise ValueError(f"Invalid objective evaluation count in generation {generation}")
        evaluations += int(calls)
        best = max(best, metrics["best_score"])
        result.append((generation, evaluations, metrics["best_score"], best))
    if evaluations != budget:
        raise ValueError(f"Trace accounts for {evaluations} evaluations, expected {budget}")
    return result


def align_curve(points, grid):
    """At each budget use only checkpoints already observed, with no extrapolation."""
    counts = [point[1] for point in points]
    if not grid or grid[0] < counts[0] or grid[-1] > counts[-1]:
        raise ValueError("Evaluation grid extends beyond observed checkpoints")
    return [points[bisect_right(counts, n) - 1][3] for n in grid]


def write_plot(rows, destination):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, axes = plt.subplots(1, len(METHOD_NAMES), figsize=(10.5, 4.05), sharex=True, sharey=True)
    for ax, (algorithm, name) in zip(axes, METHOD_NAMES.items()):
        for condition in CONDITIONS:
            group = [r for r in rows if r[0] == algorithm and r[1] == condition]
            label, color, linestyle = CONDITION_STYLES[condition]
            ax.step([r[2] for r in group], [r[4] for r in group], where="post",
                    label=label, color=color, linestyle=linestyle, linewidth=1.7)
        ax.set_title(name)
        ax.grid(axis="y", alpha=.23)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mediana do melhor valor observado")
    fig.supxlabel("Avaliações acumuladas da função objetivo", y=.15)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False)
    fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.29, wspace=.12)
    fig.savefig(destination / "convergence.pdf", bbox_inches="tight")
    fig.savefig(destination / "convergence.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def analyze(study, observations, output, grid_step):
    study, observations, output = (path.resolve() for path in (study, observations, output))
    if (output.is_relative_to(study) or study.is_relative_to(output)
            or observations.is_relative_to(output) or output.is_relative_to(observations.parent)):
        raise ValueError("Choose an output directory separate from the archived inputs and primary analysis")
    if grid_step <= 0:
        raise ValueError("--grid-step must be positive")
    collection = _json(study / "collection_summary.json")
    budget = collection["actual_evaluations_each"]
    selection = _json(study / "selection.json")["targets"]
    planned = {(t["complex_id"], method, c) for t in selection for method in METHOD_NAMES for c in CONDITIONS}
    with observations.open(encoding="utf-8-sig", newline="") as f:
        trials = list(csv.DictReader(f))
    keys = {(row["complex_id"], row["algorithm"], row["condition"]) for row in trials}
    if keys != planned or len(trials) != len(keys) or len(trials) != collection["planned_runs"]:
        raise ValueError("Observations do not match the planned factorial cohort")
    outcome_counts = Counter(row["outcome_state"] for row in trials)
    if (outcome_counts["scored"] != collection["counts"]["SUCCEEDED"]
            or outcome_counts["no_valid_pose"] != collection["physical_reduced_no_pose"]):
        raise ValueError("Terminal outcomes differ from the archived collection")

    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output, prefix=".convergence-") as temporary:
        staging = Path(temporary)
        grouped = defaultdict(list)
        manifests = []
        with (staging / "checkpoints.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("complex_id", "algorithm", "condition", "outcome", "generation",
                             "cumulative_evaluations", "generation_best_score", "best_so_far"))
            for row in trials:
                if int(row["actual_eval_count"]) != budget or int(row["evaluation_budget"]) != budget:
                    raise ValueError(f"Unexpected budget: {row['complex_id']} {row['job_id']}")
                path = checked_trace(study, row)
                contents = path.read_bytes()
                try:
                    series = checkpoints(contents.splitlines(), budget)
                except ValueError as exc:
                    raise ValueError(f"{path}: {exc}") from exc
                key = (row["algorithm"], row["condition"])
                grouped[key].append(series)
                writer.writerows((row["complex_id"], *key, row["outcome_state"], *p) for p in series)
                manifests.append({"complex_id": row["complex_id"], "job_id": row["job_id"],
                                  "outcome": row["outcome_state"], "trace": path.relative_to(study).as_posix(),
                                  "sha256": hashlib.sha256(contents).hexdigest(), "checkpoints": len(series)})

        start = max(s[0][1] for group in grouped.values() for s in group)
        grid = sorted({start, budget, *range(((start + grid_step - 1) // grid_step) * grid_step, budget, grid_step)})
        rows = []
        for (method, condition), series in sorted(grouped.items()):
            aligned = [align_curve(s, grid) for s in series]
            for n, values in zip(grid, zip(*aligned)):
                rows.append((method, condition, n, len(values), median(values)))
        with (staging / "median_curves.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("algorithm", "condition", "cumulative_evaluations", "n_complexes", "median_best_so_far"))
            writer.writerows(rows)
        write_plot(rows, staging)
        summary = {
            "objective": "maximum recorded search best_score; larger is better",
            "scope": "Post-hoc descriptive objective trajectories; includes runs without a valid final pose.",
            "alignment": "Maximum through completed generation checkpoints, carried forward to common budgets. Median across complexes.",
            "budget": budget, "grid_evaluations": grid, "counts": dict(outcome_counts),
            "n_complexes": len(selection), "n_runs": len(trials), "n_groups": len(grouped),
            "observations_sha256": digest(observations), "sources": manifests,
        }
        (staging / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        for name in ("checkpoints.csv", "median_curves.csv", "convergence.pdf", "convergence.png", "manifest.json"):
            os.replace(staging / name, output / name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=ROOT / "runs" / "fatorial_dockq_20261119")
    parser.add_argument("--observations", type=Path, help="Official factorial_observations.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "revisao_professor" / "convergencia")
    parser.add_argument("--grid-step", type=int, default=100, help="Evaluation spacing for median curves")
    args = parser.parse_args()
    source = args.observations or ROOT / "output" / args.study.resolve().name / "factorial_observations.csv"
    result = analyze(args.study, source, args.output_dir, args.grid_step)
    print(f"{result['n_runs']} buscas, {result['n_groups']} curvas: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
