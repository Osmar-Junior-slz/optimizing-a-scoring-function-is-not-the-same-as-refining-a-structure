"""Read an existing calibrated batch and write only scientific_summary.json/.md.

Uses only the standard library; never imports or executes the docking pipeline.
"""
import argparse
from decimal import Decimal
import json
import math
from pathlib import Path
from statistics import mean, median


ALGORITHMS = {"ag": "AG", "pso": "PSO", "hybrid": "Hybrid"}


def dockq_value(value):
    """Missing scores stay missing; invalid scores must not enter statistics."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Invalid DockQ: {value!r}")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"Invalid DockQ: {value!r}")
    return value


def preservation(reduced, full):
    reduced, full = dockq_value(reduced), dockq_value(full)
    if reduced is None or full is None:
        missing = ", ".join(name for name, value in (("REDUCED", reduced), ("FULL", full))
                            if value is None)
        return {"dockq_reduced": reduced, "dockq_full": full,
                "dockq_full_minus_reduced": None, "Y": None,
                "preservation_status": f"Preservação completa não pode ser determinada: DockQ {missing} indisponível."}
    # Decimal avoids classifying the exact 0.05 boundary by binary rounding.
    difference = Decimal(str(full)) - Decimal(str(reduced))
    return {"dockq_reduced": reduced, "dockq_full": full,
            "dockq_full_minus_reduced": float(difference),
            "Y": int(reduced >= 0.23 and difference <= Decimal("0.05")),
            "preservation_status": "Determinada"}


def calculate_metrics(cases):
    accepted = [case for case in cases if case["action"] == "ACCEPT_REDUCED"]
    scores = [dockq_value(case["dockq_reduced"]) for case in accepted]
    available = [score for score in scores if score is not None]
    good = sum(score >= 0.23 for score in available)
    return {
        "total_decisions": len(cases),
        "accepted_reduced": len(accepted),
        "coverage_percent": 100 * len(accepted) / len(cases) if cases else None,
        "failed_reduced": sum(case["reduced_state"] == "FAILED" for case in cases),
        "accepted_dockq_mean": mean(available) if available else None,
        "accepted_dockq_median": median(available) if available else None,
        "accepted_dockq_ge_0_23": good,
        "accepted_dockq_ge_0_23_percent": 100 * good / len(accepted) if accepted else None,
        "accepted_dockq_lt_0_23": sum(score < 0.23 for score in available),
        "accepted_dockq_missing": len(scores) - len(available),
    }


def consolidate(cases):
    return {"overall": calculate_metrics(cases),
            "by_algorithm": {label: calculate_metrics([c for c in cases if c["algorithm"] == label])
                             for label in ALGORITHMS.values()},
            "cases": cases}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_score(batch, task):
    if task is None or task["state"] != "SUCCEEDED":
        return None
    path = batch / "targets" / task["complex_id"] / "jobs" / task["job_id"] / "postprocess" / "reference_dockq.json"
    if not path.is_file():
        return None
    return dockq_value(read_json(path)["metrics"]["dockq"])


def read_cases(batch):
    status = read_json(batch / "batch_status.json")
    tasks = status["tasks"]
    cases, seen = [], set()
    for task in tasks.values():
        if task["protocol"] != "reduced":
            continue
        decision = task.get("confidence_decision")
        if decision is None:
            raise ValueError(f"Missing persisted decision: {task['complex_id']}/{task['job_id']}")
        path = batch / "targets" / task["complex_id"] / "jobs" / task["job_id"] / "confidence_decision.json"
        persisted = read_json(path)
        for key in ("action", "reason", "complex_id", "reduced_job_id", "full_job_id", "decision_sha256"):
            if persisted[key] != decision[key]:
                raise ValueError(f"Decision disagrees with batch status: {path} ({key})")
        if decision["action"] not in {"ACCEPT_REDUCED", "RUN_FULL"}:
            raise ValueError(f"Unknown decision: {path}")
        if decision["complex_id"] != task["complex_id"] or decision["reduced_job_id"] != task["job_id"]:
            raise ValueError(f"Decision belongs to another case: {path}")
        key = (task["complex_id"], task["algorithm"], task["seed"])
        if key in seen:
            raise ValueError(f"Duplicate case: {key}")
        seen.add(key)
        full = tasks.get(f"{task['complex_id']}::{decision['full_job_id']}")
        if full is not None and (full["complex_id"], full["algorithm"], full["seed"], full["protocol"]) != (*key, "full"):
            raise ValueError(f"FULL does not match REDUCED: {key}")
        cases.append({"complex_id": key[0], "algorithm": ALGORITHMS[key[1]], "seed": key[2],
                      "action": decision["action"], "reduced_state": task["state"],
                      **preservation(read_score(batch, task), read_score(batch, full))})
    if not cases:
        raise ValueError("No persisted decisions found in the batch.")
    return sorted(cases, key=lambda c: (c["complex_id"], c["algorithm"], c["seed"]))


def markdown(report):
    def fmt(value):
        if value is None:
            return "Indisponível"
        return f"{value:.6f}" if isinstance(value, float) else str(value)

    lines = ["# Relatório científico", "", f"Fonte: `{report['source_batch']}`", "",
             "Uma decisão por proteína, algoritmo e seed. Aceitação é a ação persistida ACCEPT_REDUCED. "
             "Falha é o estado FAILED de REDUCED, não uma rejeição pelo calibrador.", "",
             "Cobertura = aceitos / decisões × 100. Percentual com DockQ ≥ 0.23 = quantidade / todos os aceitos × 100. "
             "Média e mediana usam somente DockQ disponível dos aceitos; ausências são informadas e nunca substituídas por zero. "
             "Denominadores vazios são indisponíveis.", "",
             "| Métrica | Total | AG | PSO | Hybrid |", "|---|---:|---:|---:|---:|"]
    labels = ["Total de decisões", "REDUCED aceitos", "Cobertura (%)", "REDUCED falharam",
              "DockQ médio dos aceitos", "DockQ mediano dos aceitos", "Aceitos DockQ ≥ 0.23",
              "Aceitos DockQ ≥ 0.23 (%)", "Aceitos DockQ < 0.23", "Aceitos sem DockQ"]
    for key, label in zip(report["overall"], labels):
        values = [report["overall"][key]] + [report["by_algorithm"][a][key] for a in ALGORITHMS.values()]
        lines.append(f"| {label} | " + " | ".join(map(fmt, values)) + " |")
    lines += ["", "## Preservação por caso", "",
              "Y = 1 se DockQ_REDUCED ≥ 0.23 e DockQ_FULL − DockQ_REDUCED ≤ 0.05; caso contrário, Y = 0. "
              "Sem ambos os DockQ, Y é indisponível. FULL é pareado pela mesma proteína, algoritmo e seed, "
              "exclusivamente neste experimento. A preservação completa não pode ser determinada quando FULL não existe.", "",
              "| Proteína | Algoritmo | Seed | DockQ REDUCED | DockQ FULL | FULL − REDUCED | Y | Situação |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        lines.append("| " + " | ".join(fmt(case[k]) for k in (
            "complex_id", "algorithm", "seed", "dockq_reduced", "dockq_full",
            "dockq_full_minus_reduced", "Y", "preservation_status")) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path, help="Existing calibrated batch directory")
    parser.add_argument("--output-dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    report = {"source_batch": str(args.batch.resolve()), **consolidate(read_cases(args.batch))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "scientific_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.output_dir / "scientific_summary.md").write_text(markdown(report), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
