"""Joint report from verified controls and confidence recorded before FULL."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import json
import tempfile

import numpy as np
import pandas as pd

from docking_confidence.application.batch import _atomic_write_json, _file_sha256, _load_json_object, _payload_sha256, _validate_manifest
from docking_confidence.application.article_study import BLOCKS, load_article_protocol, study_software
from docking_confidence.application.calibrated_batch import CalibratedBatchGate
from docking_confidence.application.factorial_reporting import analyze_factorial_batch, factorial_markdown, write_factorial_report
from docking_confidence.application.probability_reporting import DiagnosticOptions, frozen_prediction_report
from docking_confidence.application.runtime import operational_cost_assessment
from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.domain.preservation import PreservationRule
from docking_confidence.validation.metrics import sigmoid_raw_score
from docking_confidence.risk.policy import audit_locked_policy
from docking_confidence.risk.scientific import risk_coverage_envelope, zero_error_sample_requirements
from docking_confidence.application.reduced_evidence import _artifact_path, read_reduced_graph


def _source_hashes(root):
    return {str(path.relative_to(root)): _file_sha256(path)
            for name in BLOCKS for path in sorted((root / name).rglob("*"))
            if path.is_file() and not path.name.endswith((".lock", ".tmp"))}


def _verified_live_graph(directory, receipt, session):
    """Verify the graph actually scored, before reference alignment/PDB rounding."""
    path = directory / "confidence/evidence.json"
    if _file_sha256(path) != receipt.get("confidence_evidence_sha256"):
        raise ValueError("A evidência de inferência difere do recibo REDUCED.")
    evidence = _load_json_object(path)
    target, job, result = receipt["target"], receipt["job"], receipt["engine_result"]
    expected = {"schema_version": "reference_free_reduced_evidence_v1",
        "complex_id": target["complex_id"], "run_id": job["job_id"],
        "complex_fingerprint_sha256": target["complex_fingerprint_sha256"],
        "artifact_sha256": session.artifact_sha256, "engine_pose_json": result["pose_artifact_path"]}
    if any(evidence.get(key) != value for key, value in expected.items()):
        raise ValueError("A evidência de inferência identifica outra execução ou modelo.")
    for path_key, hash_key in (("engine_pose_json", "engine_pose_sha256"),
                              ("pose_path", "pose_sha256"), ("graph_path", "graph_sha256")):
        if _file_sha256(_artifact_path(directory, evidence[path_key])) != evidence[hash_key]:
            raise ValueError("O hash da pose ou do grafo de inferência mudou.")
    graph_receipt = deepcopy(receipt)
    graph_receipt["engine_result"].update(complex_pose_pdb_path=evidence["pose_path"],
                                         graph_features_path=evidence["graph_path"])
    return read_reduced_graph(directory, graph_receipt, session.artifact["resolved_config"])


def _recorded_predictions(batch, frame, session):
    rows = []
    for row in frame.to_dict("records"):
        folder = batch / "targets" / row["complex_id"] / "jobs" / row["run_id"]
        decision = _load_json_object(folder / "confidence_decision.json")
        core = {key: value for key, value in decision.items() if key != "decision_sha256"}
        if decision.get("decision_sha256") != _payload_sha256(core):
            raise ValueError("A decisão de confiança foi alterada.")
        if (decision.get("artifact_sha256") != session.artifact_sha256
                or decision.get("validation_mode") is not True or decision["action"] != "RUN_FULL"
                or decision["reduced_receipt_sha256"] != _file_sha256(folder / "receipt.json")):
            raise ValueError("O resultado não foi avaliado pelo calibrador congelado em modo de validação.")
        receipt = _load_json_object(folder / "receipt.json")
        if (receipt["target"]["complex_fingerprint_sha256"] != row["complex_fingerprint_sha256"]
                or receipt["job"]["job_id"] != row["run_id"]):
            raise ValueError("A previsão e o recibo identificam execuções diferentes.")
        graph = _verified_live_graph(folder, receipt, session)
        input_path = folder / "confidence/inference_input.csv"
        replay = session.predict_bytes(input_path.read_bytes()).iloc[0]
        saved = pd.read_csv(folder / "confidence/prediction.csv", float_precision="round_trip").iloc[0]
        if replay["input_snapshot_sha256"] != saved["input_snapshot_sha256"]:
            raise ValueError("A entrada de inferência difere da entrada registrada ao vivo.")
        if (str(replay["complex_fingerprint_sha256"]) != row["complex_fingerprint_sha256"]
                or str(replay["run_id"]) != row["run_id"]):
            raise ValueError("A previsão pertence a outro complexo ou execução.")
        graph_differences = {}
        for feature in session.artifact["feature_columns"]:
            # Dataset preparation audits a reference-aligned, rounded PDB. Its
            # graph may differ from the independently verified raw pose used
            # by inference. Evaluate the actual input without relaxing tolerance.
            expected = graph[feature] if feature.startswith("graph_") else row[feature]
            if not np.isclose(float(replay[feature]), float(expected), atol=1e-10, rtol=1e-10):
                raise ValueError("As características ao vivo e as características auditadas discordam.")
            if feature.startswith("graph_") and not np.isclose(float(row[feature]), float(expected), atol=1e-10, rtol=1e-10):
                graph_differences[feature] = {"post_reference": float(row[feature]), "live": float(expected)}
        for column in ("base_raw_score", "probability", "probability_lower", "probability_upper"):
            if not np.isclose(float(replay[column]), float(saved[column]), atol=1e-12, rtol=1e-12):
                raise ValueError("A previsão salva difere da reprodução com o modelo congelado.")
        if not np.isclose(float(replay["probability"]), float(decision["probability"]), atol=1e-12, rtol=1e-12):
            raise ValueError("A probabilidade registrada na decisão difere da previsão.")
        rows.append({**row, **graph, "audited_feature_source": "reference_free_reduced_evidence_v1",
            "post_reference_graph_differences": json.dumps(graph_differences, sort_keys=True),
            "dockq_reduced": receipt["engine_result"].get("dockq"), **{key: replay[key] for key in (
            "base_raw_score", "probability", "probability_lower", "probability_upper",
            "in_applicability_domain", "unseen_complex", "trusted_reduced_input_provenance")}})
    return pd.DataFrame(rows) if rows else frame.assign(**{key: pd.Series(dtype=float) for key in (
        "base_raw_score", "probability", "probability_lower", "probability_upper",
        "in_applicability_domain", "unseen_complex", "trusted_reduced_input_provenance")})


def compare_frozen_methods(frame, comparators, *, options=None, risk_frame=None, all_complexes=None):
    """Same test rows, original calibration rows, original risk-based thresholds."""
    raw = frame["base_raw_score"].to_numpy(dtype=float)
    iso = np.interp(raw, comparators["isotonic_x"], comparators["isotonic_y"])
    va_report = frozen_prediction_report(frame, options=options)
    iso_report = frozen_prediction_report(frame.assign(probability=iso), options=options)
    evaluations = {"original": va_report["evaluations"]["before_venn_abers"],
                   "isotonic": iso_report["evaluations"]["after_venn_abers"],
                   "venn_abers": va_report["evaluations"]["after_venn_abers"]}
    probabilities = {"original": sigmoid_raw_score(raw), "isotonic": iso,
                     "venn_abers": frame["probability"].to_numpy(dtype=float)}
    eligible = np.ones(len(frame), dtype=bool)
    for column in ("in_applicability_domain", "unseen_complex", "trusted_reduced_input_provenance"):
        eligible &= frame[column].to_numpy(dtype=bool)
    y = frame["preserved"].to_numpy(dtype=int)
    decisions = {}
    for method, probability in probabilities.items():
        policy = comparators["policies"][method]
        threshold = policy.get("threshold")
        score = frame["probability_lower"].to_numpy(dtype=float) if method == "venn_abers" else probability
        accepted = eligible & (score >= threshold) if policy.get("supported") and threshold is not None else np.zeros(len(y), dtype=bool)
        measured_overhead = frame["confidence_overhead_sec"].to_numpy(dtype=float) if "confidence_overhead_sec" in frame else np.array([np.nan])
        # Only VA has measured live extraction/inference overhead. Alternative
        # methods retain an explicit unknown-overhead bound, never a speed claim.
        overhead = float(measured_overhead.sum()) if method == "venn_abers" and np.isfinite(measured_overhead).all() else None
        decisions[method] = {"threshold": threshold, "risk_policy_supported": policy.get("supported", False),
            "hypothetical_accepted_reduced": int(accepted.sum()),
            "hypothetical_false_accepts": int((accepted & (y == 0)).sum()),
            "hypothetical_coverage": float(accepted.mean()) if len(accepted) else None,
            "hypothetical_false_acceptance_rate": float((accepted & (y == 0)).sum() / accepted.sum()) if accepted.any() else None,
            "hypothetical_full_calls": int((~accepted).sum()),
            "frozen_policy_test_audit": audit_locked_policy(frame, score, eligible, threshold,
                comparators["policy_config"]) if "policy_config" in comparators else None,
            "cost": operational_cost_assessment(frame, accepted, overhead_sec=overhead)}
        frame[f"probability_{method}"] = probability
    result = {"evaluations": evaluations, "decisions": decisions,
            "paired_venn_minus_original": va_report["paired_after_minus_before"],
            "paired_isotonic_minus_original": iso_report["paired_after_minus_before"],
            "bootstrap": va_report["bootstrap"],
            "scope": "same verified paired test rows; thresholds frozen on original risk role; counterfactual decisions only"}
    if "risk_protocol" in comparators:
        rf = frame if risk_frame is None else risk_frame
        risk_raw = rf["base_raw_score"].to_numpy(dtype=float)
        risk_scores = {"original": sigmoid_raw_score(risk_raw),
            "isotonic": np.interp(risk_raw, comparators["isotonic_x"], comparators["isotonic_y"]),
            "venn_abers": rf["probability_lower"].to_numpy(dtype=float)}
        risk_eligible = np.ones(len(rf), dtype=bool)
        for column in ("in_applicability_domain", "unseen_complex", "trusted_reduced_input_provenance"):
            risk_eligible &= rf[column].to_numpy(dtype=bool)
        result["structural_risk_profile"] = risk_coverage_envelope(rf, risk_scores, risk_eligible,
            comparators["risk_protocol"], stage="article_test", all_complexes=all_complexes)
        result["scope"] = "frozen models; historical training-target probability metrics; exact structural risk curves with no approved acceptance tolerance"
    return result


def write_risk_outputs(profile, destination, name):
    rows = [{"method": method, **row} for method, curve in profile["curves"].items() for row in curve]
    pd.DataFrame(rows).to_csv(destination / f"risco_cobertura_{name}.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, curve in profile["curves"].items():
        available = [row for row in curve if row["simultaneous_complex_risk_upper_bound"] is not None]
        x = [row["empirical_complex_coverage"] for row in available]
        line, = ax.plot(x, [row["empirical_worst_case_complex_risk"] for row in available], label=method + " observado")
        ax.plot(x, [row["simultaneous_complex_risk_upper_bound"] for row in available],
                linestyle="--", color=line.get_color(), label=method + " limite simultâneo")
    ax.set(xlabel="Cobertura empírica de complexos", ylabel="Risco por complexo aceito",
           xlim=(0, 1), ylim=(0, 1), title="Risco–cobertura: limites binomiais exatos")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / f"risco_cobertura_{name}.png", dpi=180)
    plt.close(fig)


def _calibration_plot(evaluations, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for name, evaluation in evaluations.items():
        curves = evaluation["curves"]
        bins = [row for row in curves["reliability_bins"] if row["count"]]
        axes[0].plot([row["mean_probability"] for row in bins], [row["observed_positive_fraction"] for row in bins], marker="o", label=name)
        axes[1].plot([row["false_positive_rate"] for row in curves["roc"]], [row["true_positive_rate"] for row in curves["roc"]], label=name)
        axes[2].plot([row["recall"] for row in curves["precision_recall"]], [row["precision"] for row in curves["precision_recall"]], label=name)
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    for axis, title, xlabel, ylabel in zip(axes, ("Calibração", "ROC", "Precisão–revocação"),
            ("Probabilidade estimada", "Taxa de falsos positivos", "Revocação"), ("Frequência observada", "Sensibilidade", "Precisão")):
        axis.set(title=title, xlabel=xlabel, ylabel=ylabel, xlim=(0, 1), ylim=(0, 1))
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_article_report(output_dir):
    root = Path(output_dir).resolve(strict=True)
    protocol = load_article_protocol(root)
    if study_software() != protocol["software"]:
        raise ValueError("O código ou as bibliotecas mudaram depois de preparar o estudo.")
    statuses = {name: _load_json_object(root / name / "batch_status.json") for name in BLOCKS}
    if any(status["state"] not in {"COMPLETED", "COMPLETED_WITH_ERRORS"} for status in statuses.values()):
        raise ValueError("Conclua os dois blocos antes de gerar o relatório final.")
    if statuses[BLOCKS[1]]["counts"]["skipped_calibrator"]:
        raise ValueError("A validação precisa conservar todos os FULL como referência de qualidade.")
    gate = CalibratedBatchGate(protocol["artifact_path"])
    if (gate.session.artifact_sha256 != protocol["artifact_sha256"]
            or gate.session.manifest_sha256 != protocol["model_manifest_sha256"]):
        raise ValueError("O modelo difere do protocolo integrado.")
    for name in BLOCKS:
        manifest = _load_json_object(root / name / "batch_manifest.json")
        _validate_manifest(manifest)
        base_plan = {key: value for key, value in manifest["plan"].items()
                     if key not in {"experimental_protocol", "confidence_gate"}}
        if base_plan != protocol["block_plans"][name]:
            raise ValueError("Um bloco usa dados ou parâmetros diferentes do protocolo do artigo.")
        for path in (root / name).glob("targets/*/jobs/*/receipt.json"):
            receipt = _load_json_object(path)
            result = receipt["engine_result"]
            if (result.get("actual_eval_count") != receipt["job"]["profile"]["evaluation_budget"]
                    or result.get("stop_reason") != "objective_evaluation_ceiling"):
                raise ValueError("Uma execução não cumpriu o orçamento efetivo congelado.")
    hashes = _source_hashes(root)
    destination = root / "analise_artigo"
    report_path = destination / "article_report.json"
    if report_path.exists():
        saved = _load_json_object(report_path)
        if saved["source_hashes"] != hashes or saved["protocol_sha256"] != protocol["protocol_sha256"]:
            raise ValueError("As fontes mudaram depois de gerar o relatório integrado.")
        return {"json": str(report_path), "markdown": str(destination / "RELATORIO_ARTIGO.md")}
    factorial = analyze_factorial_batch(root / BLOCKS[0])
    if factorial["diagnostics"] or factorial["csv_audit"]["discrepancies"]:
        raise ValueError("A auditoria dos controles encontrou divergências; confira os resultados antes de analisar.")
    # Temporary preparation is inside this verified workspace root; it never
    # replaces historical data or creates a second fitted model.
    with tempfile.TemporaryDirectory(prefix=".article-audit-", dir=root) as temporary:
        assert root in Path(temporary).resolve().parents
        preparation = build_training_dataset(root / BLOCKS[1], temporary, profile="operational",
            config_path=gate.artifact_path.parent / "resolved_config.json", fingerprint_alias_policy="first_complex_id")
        if not preparation["valid"] and (preparation["counts"]["candidate_rows"] or
                any(issue["code"] != "official_loader" for issue in preparation["issues"])):
            raise ValueError(f"Os pares da avaliação não passaram na auditoria: {preparation['issues']}")
        frame = pd.read_csv(preparation["candidate_dataset_path"], float_precision="round_trip")
        preparation_summary = {key: preparation[key] for key in ("counts", "failures", "unmatched", "issues")}
    frame = _recorded_predictions(root / BLOCKS[1], frame, gate.session)
    live_inputs = []
    for task in statuses[BLOCKS[1]]["tasks"].values():
        if task["protocol"] == "reduced" and task["state"] == "SUCCEEDED":
            path = root / BLOCKS[1] / "targets" / task["complex_id"] / "jobs" / task["job_id"] / "confidence/inference_input.csv"
            live_inputs.append(pd.read_csv(path, float_precision="round_trip"))
    all_predictions = _recorded_predictions(root / BLOCKS[1], pd.concat(live_inputs, ignore_index=True), gate.session) if live_inputs else frame.copy()
    paired_keys = set(zip(frame["complex_id"], frame["run_id"]))
    all_predictions["quality_pair_available"] = pd.Series(
        [(cid, run_id) in paired_keys for cid, run_id in zip(all_predictions.complex_id, all_predictions.run_id)],
        index=all_predictions.index, dtype=bool)
    rule = PreservationRule(**gate.artifact["preservation_rule"])
    frame["preserved"] = [rule.label(reduced, full) for reduced, full in zip(frame.dockq_reduced, frame.dockq_full)]
    risk_frame = all_predictions.drop(columns=["dockq_full"], errors="ignore").merge(
        frame[["complex_id", "run_id", "dockq_full"]], on=["complex_id", "run_id"], how="left", validate="one_to_one")
    universe = [target["complex_fingerprint_sha256"] for target in protocol["block_plans"][BLOCKS[1]]["targets"]]
    comparison = compare_frozen_methods(frame, protocol["comparators"], options=DiagnosticOptions(
        bootstrap_repetitions=protocol["analysis"]["bootstrap_repetitions"], bootstrap_seed=protocol["analysis"]["bootstrap_seed"]),
        risk_frame=risk_frame, all_complexes=universe)
    task_seconds = {name: float(sum(task.get("duration_seconds") or 0 for task in status["tasks"].values())) for name, status in statuses.items()}
    calibration = {"paired_rows": len(frame), "independent_paired_complexes": int(frame.complex_fingerprint_sha256.nunique()),
        "audited_feature_source": "reference_free_reduced_evidence_v1",
        "pairs_with_post_reference_graph_rounding_differences": int(frame["post_reference_graph_differences"].ne("{}").sum()) if len(frame) else 0,
        "paired_rows_in_applicability_domain": int(frame.in_applicability_domain.sum()),
        "scored_reduced_runs": len(all_predictions), "scored_runs_without_quality_pair": int((~all_predictions.quality_pair_available).sum()),
        "preserved_pairs": int(frame.preserved.sum()), "mean_dockq_reduced": float(frame.dockq_reduced.mean()) if len(frame) else None,
        "mean_dockq_full": float(frame.dockq_full.mean()) if len(frame) else None,
        "mean_dockq_full_minus_reduced": float((frame.dockq_full - frame.dockq_reduced).mean()) if len(frame) else None,
        "median_dockq_full_minus_reduced": float((frame.dockq_full - frame.dockq_reduced).median()) if len(frame) else None,
        "probability_training_target": gate.artifact["preservation_rule"],
        "target_interpretation": "legacy experimental training target; maximum_dockq_loss=0.05 is not a published universal noninferiority margin; structural risk is evaluated on a distinct predeclared DockQ-class endpoint",
        "preparation": preparation_summary, "comparison": comparison,
        "automatic_acceptance_authorized": False,
        "historical_artifact_decision_capable_audit_only": gate.artifact["decision_capable"],
        "actually_avoided_full": 0, "actually_saved_full_runtime_sec": 0,
        "measured_latest_attempt_task_seconds": task_seconds,
        "timing_scope": "job timers include preparation/reference processing; latest attempt only; excludes orchestration and prior retries; counterfactual costs are paired engine-time sums, not observed savings"}
    report = {"schema_version": "joint_article_report_v1", "protocol_sha256": protocol["protocol_sha256"],
        "artifact_sha256": protocol["artifact_sha256"], "source_hashes": hashes, "counts": {name: status["counts"] for name, status in statuses.items()},
        "controls": factorial, "calibration": calibration,
        "risk_methodology": protocol["comparators"]["risk_protocol"],
        "validation_risk_profile": protocol["comparators"]["validation_risk_profile"],
        "sample_size_planning_illustrative_only": zero_error_sample_requirements(protocol["comparators"]["risk_protocol"]),
        "limitations": ["All controls and FULL validation outcomes are retained; actual avoided FULL is zero.",
            "The model is used only on its 5700/7600 uniform regime; factorial arms are causal controls, not additional training examples.",
            "Test outcomes never refit the base model, calibrators or risk thresholds; comparison does not automatically select a replacement.",
            "Counterfactual costs exclude unpaired failures and unknown overhead; validation cost is reported separately.",
            "Independent molecular inputs and a local frozen hash do not establish prospective preregistration or prove generalization."]}
    report["limitations"].append("Exact canonical-sequence separation does not automatically eliminate homology between different sequences.")
    report["limitations"].append("The primary structural risk endpoint differs from the historical probability-training target. LTT permits frozen arbitrary scores; this does not recalibrate those scores as probabilities of the new endpoint.")
    destination.mkdir(exist_ok=True)
    write_factorial_report(factorial, destination / "controles", plots=True)
    frame.to_csv(destination / "calibrador_previsoes_e_qualidade.csv", index=False)
    all_predictions.to_csv(destination / "calibrador_todas_previsoes.csv", index=False)
    _calibration_plot(comparison["evaluations"], destination / "calibracao_roc_pr.png")
    write_risk_outputs(report["validation_risk_profile"], destination, "validacao")
    write_risk_outputs(comparison["structural_risk_profile"], destination, "teste")
    lines = ["# Artigo: controles do orientador e Venn–Abers", "",
        f"Modelo congelado: `{protocol['artifact_sha256']}`. Complexos selecionados: {protocol['selection']['selected_count']}.", "",
        "## Confiança calibrada e qualidade", "",
        f"{len(all_predictions)} REDUCED receberam probabilidades; {len(frame)} pares têm qualidade auditada; "
        f"{calibration['preserved_pairs']} preservaram a regra estrutural. "
        "Os três métodos usam os mesmos pares. Isotônica foi ajustada somente na calibração original. A grade de limiares foi fixada antes deste teste; nenhum limiar de aceitação foi aprovado.", "",
        "As métricas probabilísticas abaixo usam o alvo histórico do modelo: DockQ REDUCED ≥ 0,23 e perda ≤ 0,05. "
        "A margem 0,05 é uma escolha experimental anterior, não uma constante estabelecida pelo artigo do DockQ. "
        "A análise de risco estrutural usa um desfecho distinto, descrito abaixo; as probabilidades não foram renomeadas como probabilidades desse novo desfecho.", "",
        "| Método | Brier | ECE (10 intervalos) | ROC-AUC | AUC-PR | AP |", "|---|---:|---:|---:|---:|---:|"]
    def fmt(value):
        return "indisponível" if value is None else f"{value:.6g}"
    for name, evaluation in comparison["evaluations"].items():
        metrics = evaluation["metrics"]
        lines.append("| " + name + " | " + " | ".join(fmt(metrics[key]) for key in (
            "brier_score", "expected_calibration_error_10bin", "roc_auc", "pr_auc_trapezoidal", "average_precision")) + " |")
    lines += ["", "Intervalos de confiança por reamostragem de complexos e curvas completas estão no JSON.", "",
        "![Calibração, ROC e precisão–revocação](calibracao_roc_pr.png)", "", "## Decisões e processamento", "",
        "FULL efetivamente evitados nesta validação: **0**. Tempo economizado por dispensa de FULL: **0 s**. "
        "FULL foi executado para conferir a qualidade. As decisões e custos abaixo são simulações separadas, com limiares congelados.", "",
        "| Método | Limiar da validação | Política apoiada | Aceitações hipotéticas | Aceitações incorretas |", "|---|---:|---|---:|---:|"]
    for name, decision in comparison["decisions"].items():
        lines.append(f"| {name} | {fmt(decision['threshold'])} | {decision['risk_policy_supported']} | "
                     f"{decision['hypothetical_accepted_reduced']} | {decision['hypothetical_false_accepts']} |")
    lines += ["", f"Autorização automática do modelo instalado: **{gate.artifact['decision_capable']}**. "
        "Uma probabilidade disponível não equivale à aprovação para dispensar FULL.", "",
        "Quando não há aceitação do REDUCED, a cobertura é zero e a taxa de erro entre aceitações é indisponível; isso não demonstra decisões perfeitas. "
        "Cobertura, erros e auditoria da política congelada no teste estão no JSON.", "",
        "## Regra de decisão: risco e cobertura sem limite de erro aprovado", "",
        "Conforme definido para este estudo, a taxa máxima tolerável de aceitação incorreta ainda não foi aprovada. "
        "Nenhum ponto da curva é convertido automaticamente em regra de aceitação.", "",
        "Aplicamos a inversão do teste binomial exato do Learn then Test (seção 3.2), com Bonferroni (seção 2.3), "
        "a todos os métodos, limiares e às duas etapas declaradas. A unidade independente é o complexo: "
        "um único erro entre suas execuções aceitas torna o complexo adverso. Isso controla risco condicional por complexo, não risco médio por execução.", "",
        "Desfecho operacional do estudo: REDUCED ao menos aceitável e sem queda de classe DockQ em relação ao FULL. "
        "Usamos os cortes publicados 0,23/0,49/0,80; a comparação entre classes é uma definição deste estudo, não uma política de docking publicada pronta. "
        "DockQ contínuo e suas diferenças permanecem no relatório: igualdade de classe não prova equivalência estrutural ou biológica. "
        "Qualidade desconhecida entre resultados aceitos é tratada como adversa para o limite superior, sem inventar DockQ.", "",
        "Os limites exigem complexos independentes e representativos e protocolo fixo antes dos resultados. "
        "A revisão da partição de risco já observada é exploratória; homologia, reutilização adaptativa ou mudança de distribuição impedem transferir automaticamente a garantia.", "",
        "![Curva na validação original](risco_cobertura_validacao.png)", "",
        "![Curva no teste do artigo](risco_cobertura_teste.png)", "",
        "Referências: [Learn then Test](https://arxiv.org/abs/2110.01052), "
        "[DockQ](https://doi.org/10.1371/journal.pone.0161879), "
        "[Venn–Abers indutivo](https://arxiv.org/abs/1511.00213).", "",
        "Custos medidos, falhas, pares ausentes e estimativas contrafactuais estão separados no JSON. "
        "Reduzir avaliações em 25% não implica reduzir o tempo em 25%.", "", "## Controles exigidos pelo orientador", "",
        factorial_markdown(factorial).replace("# Análise fatorial 2 × 2", "### Resultados fatoriais"), "", "## Limites de interpretação", ""]
    lines += ["- " + limitation for limitation in report["limitations"]]
    (destination / "RELATORIO_ARTIGO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if _source_hashes(root) != hashes:
        raise ValueError("Os resultados mudaram durante a análise; o relatório não foi finalizado.")
    _atomic_write_json(report_path, report)
    return {"json": str(report_path), "markdown": str(destination / "RELATORIO_ARTIGO.md")}
