from __future__ import annotations

from pathlib import Path
from time import monotonic
from uuid import uuid4

import pandas as pd
import streamlit as st
import altair as alt

from docking_confidence.application.training_dataset import build_training_dataset
from docking_confidence.application.train import TrainingEvidenceError
from docking_confidence.application.training_results import (
    existing_training_outputs,
    load_training_result,
    train_prepared_dataset,
)


_PREPARATION_KEY = "training_preparation"
_SIGNATURE_KEY = "training_preparation_inputs"
_RESULT_KEY = "training_verified_result"
_COUNT_LABELS = {
    "complexes_found": "Complexos encontrados",
    "valid_pairs": "Pares FULL/REDUCED válidos",
    "reduced_succeeded": "REDUCED bem-sucedidos",
    "reduced_failed": "REDUCED_FAILED",
    "unmatched_pairs": "Pares sem correspondência",
    "usable_rows": "Linhas utilizáveis no treinamento",
    "independent_complexes": "Complexos independentes",
    "missing_features": "Features ausentes",
    "provenance_errors": "Erros de proveniência",
    "ambiguous_pairs": "Pares ambíguos",
    "candidate_rows": "Linhas candidatas para diagnóstico",
    "resolved_alias_groups": "Grupos de sequências repetidas resolvidos",
    "excluded_alias_targets": "Nomes repetidos excluídos do dataset",
    "excluded_alias_planned_runs": "Execuções planejadas dos nomes excluídos",
}


def _yes_no(value: object) -> str:
    return "SIM" if value is True else "NÃO"


def _article_dataset_settings(batch_dir: str) -> tuple[str, str | None, bool]:
    """Choose the scientific recipe from the immutable batch, never a UI preset."""
    from docking_confidence.application.batch import _load_json_object, _validate_manifest
    batch_dir = batch_dir.strip()
    path = Path(batch_dir) / "batch_manifest.json"
    if not batch_dir.strip() or not path.is_file():
        return "operational", None, False
    manifest = _load_json_object(path)
    _validate_manifest(manifest)
    plan = manifest["plan"]
    if plan.get("experimental_design") == "factorial_2x2":
        return "default", manifest["manifest_sha256"], True
    if plan.get("execution_strategy") == "operational_cascade_v1":
        return "operational", manifest["manifest_sha256"], False
    raise ValueError("Este lote usa um protocolo histórico. A tela do artigo analisa os controles fatoriais ou os pares 5.700/7.600; a reprodução histórica permanece disponível pelos scripts.")


def _show_fields(fields: dict) -> None:
    st.table(pd.DataFrame([
        {"Indicador": key, "Resultado": "Não disponível" if value is None else str(value)}
        for key, value in fields.items()
    ]))


def _progress_display():
    bar = st.progress(0.0, text="Iniciando…")
    st.caption("Progresso por etapas e itens processados; o percentual não estima o tempo restante.")
    started = monotonic()
    last_update = 0.0
    last_fraction = -1.0

    def update(fraction: float, message: str) -> None:
        nonlocal last_update, last_fraction
        now = monotonic()
        # Large batches can emit thousands of receipt updates. Keep the browser
        # responsive while delivering start, completion and regular increments.
        if fraction not in (0.0, 1.0) and now - last_update < 0.10 and fraction - last_fraction < 0.01:
            return
        elapsed = int(now - started)
        bar.progress(fraction, text=f"{message} · {elapsed // 60:02d}:{elapsed % 60:02d} decorridos")
        last_update, last_fraction = now, fraction

    return update


def _render_preparation(prepared: dict) -> None:
    st.subheader("Análise do batch")
    counts = prepared.get("counts", {})
    _show_fields({label: counts.get(key) for key, label in _COUNT_LABELS.items()})
    if prepared.get("valid") is True:
        st.success("Dataset válido pelo loader oficial. Revise o resumo antes de treinar.")
    else:
        st.error("Dataset bloqueado. Resolva os problemas registrados antes de treinar.")
    for issue in prepared.get("issues", []):
        st.warning(f"{issue.get('code', 'diagnóstico')}: {issue.get('message', issue)}")
        if issue.get("code") == "fingerprint_alias":
            st.write("Mesmas sequências:", ", ".join(issue.get("complex_ids", [])))
    if any(issue.get("code") == "fingerprint_alias" for issue in prepared.get("issues", [])):
        st.info("Para preparar um representante por sequência, marque 'Resolver sequências repetidas' e analise o batch novamente.")
    resolutions = prepared.get("alias_resolutions", [])
    if resolutions:
        st.warning("Sequências repetidas resolvidas pelo primeiro nome em ordem alfabética. A escolha usa apenas o manifesto, sem consultar sucesso, DockQ ou tempo. Os arquivos originais foram preservados.")
        st.table(pd.DataFrame([
            {"Nome mantido": item["selected_complex_id"],
             "Nomes excluídos": ", ".join(item["excluded_complex_ids"])}
            for item in resolutions
        ]))
    failures = prepared.get("failures", [])
    if failures:
        st.info("Execuções com falha ficam fora das linhas pareadas. REDUCED_FAILED indica retorno obrigatório ao FULL; essas falhas, por si só, não bloqueiam os demais pares válidos.")
        with st.expander("Falhas de execução e REDUCED_FAILED"):
            st.dataframe(pd.DataFrame([
                {"Complexo": item.get("complex_id"), "Execução": item.get("run_id"),
                 "Categoria": item.get("category"), "Motivo": item.get("message"),
                 "Nome excluído por sequência repetida": _yes_no(item.get("excluded_by_alias_policy", False))}
                for item in failures
            ]), hide_index=True)
    preview = prepared.get("preview", {})
    st.subheader("Preview antes de treinar")
    _show_fields({
        "Número de linhas": preview.get("row_count"),
        "Complexos independentes": preview.get("independent_complex_count"),
        "Algoritmos": preview.get("algorithm_count"),
        "Seeds presentes": preview.get("seeds"),
        "Preservados": preview.get("preserved_count"),
        "Não preservados": preview.get("non_preserved_count"),
        "REDUCED_FAILED fora do dataset": preview.get("reduced_failed_count"),
        "Protocolo": prepared.get("docking_task"),
        "Feature schema": prepared.get("feature_schema"),
        "Configuração utilizada": prepared.get("config_path"),
        "SHA-256 do dataset": prepared.get("dataset_sha256"),
    })
    st.write("Features utilizadas:", prepared.get("feature_columns", []))
    requirements = prepared.get("evidence_requirements", {})
    if requirements:
        with st.expander("Requisitos de evidência do perfil"):
            st.json(requirements)
            st.caption("São requisitos predeclarados. Nenhum gate ou locked test foi avaliado nesta preparação.")
    if prepared.get("candidate_dataset_path") and not prepared.get("valid"):
        st.caption("O CSV candidato é diagnóstico; ele não autoriza treinamento.")
    st.write("Relatório de preparação:", prepared.get("report_path"))
    with st.expander("Relatório completo de preparação"):
        st.json(prepared)


def _render_training_result(result: dict) -> None:
    artifact = result["artifact"]
    calibration = artifact["calibration"]
    policy = artifact["policy"]
    locked = artifact["locked_test_audit"]
    runtime = artifact["runtime_gate"]
    config = artifact["resolved_config"]
    run_risk = config["policy"]["method"] == "cluster_bootstrap_run_risk"
    risk_key = "empirical_run_risk" if run_risk else "empirical_conditional_complex_risk"
    st.subheader("Resultado do treinamento")
    _show_fields({
        "Modelo treinado": "SIM",
        "Probabilidade calibrada / Venn-Abers concluído": "SIM",
        "Controle de risco suportado": _yes_no(policy.get("supported")),
        "Locked test aprovado": _yes_no(locked.get("supported")),
        "Autorização automática / decision_capable": _yes_no(artifact["decision_capable"]),
        "Autorização permitida pelo perfil": _yes_no(config["authorization"]["allow_automatic_decision"]),
        "Protocolo": artifact.get("docking_task", config.get("docking_task")),
        "Feature schema": artifact.get("feature_schema", config["features"]["schema_version"]),
    })
    st.subheader("Calibração probabilística")
    _show_fields({
        "Venn-Abers concluído": "SIM",
        "Complexos usados": calibration["independent_complex_count"],
        "Complexos positivos": calibration["positive_class_complex_count"],
        "Complexos negativos": calibration["negative_class_complex_count"],
    })
    st.subheader("Calibração de risco")
    if run_risk:
        st.caption(
            "Erro por execução aceita; a incerteza é estimada com bootstrap de complexos "
            "inteiros. O intervalo é aproximado. Reanalisar dados já usados para escolher "
            "o modelo, limiar ou método não constitui uma nova confirmação independente."
        )
    risk_evidence = policy if policy.get("supported") else policy.get("best_observed_candidate", {})
    _show_fields({
        "Supported": _yes_no(policy.get("supported")),
        "Limiar selecionado": policy.get("threshold"),
        "Limiar avaliado": risk_evidence.get("threshold"),
        "Risk target": config["policy"]["risk_target"],
        "Erro observado por execução" if run_risk else "Erro observado por complexo": risk_evidence.get(risk_key),
        "Limite superior aproximado" if run_risk else "Limite superior": risk_evidence.get("simultaneous_risk_upper_bound"),
        "Execuções candidatas aceitas": risk_evidence.get("accepted_row_count"),
        "Aceites incorretos": risk_evidence.get("false_acceptance_row_count"),
        "Cobertura de complexos": risk_evidence.get("complex_coverage"),
        "Complexos avaliados": risk_evidence.get("total_complex_count"),
        "Complexos com execução candidata aceita": risk_evidence.get("accepted_complex_count"),
        "Motivo": policy.get("reason"),
    })
    if policy.get("best_observed_candidate"):
        with st.expander("Melhor candidato diagnóstico — não selecionado"):
            st.json(policy["best_observed_candidate"])
    st.subheader("Locked test")
    _show_fields({
        "Supported": _yes_no(locked.get("supported")),
        "Erro observado por execução" if run_risk else "Risco observado": locked.get(risk_key),
        "Limite superior aproximado" if run_risk else "Limite superior": locked.get("simultaneous_risk_upper_bound"),
        "Cobertura": locked.get("complex_coverage"),
        "Motivo": locked.get("reason"),
    })
    st.write("Métricas probabilísticas do locked test")
    st.json(artifact["development_summary"]["locked_test_probability_metrics"])
    if result.get("diagnostics"):
        _render_probability_diagnostics(result["diagnostics"])
    st.subheader("Runtime gate")
    _show_fields({
        "Passou": _yes_no(runtime.get("passed")),
        "Speedup FULL / REDUCED": runtime.get("speedup_full_over_reduced"),
        "Escopo": runtime.get("assessment_scope"),
        "Motivo": runtime.get("reason"),
    })
    st.subheader("Autorização — cada gate")
    for name, passed in artifact["authorization"]["gates"].items():
        st.write(f"{name}: {_yes_no(passed)}")
    if artifact["decision_capable"] is not True:
        st.warning("O modelo foi treinado, mas a autorização automática está desabilitada ou algum gate não passou.")
    st.subheader("Arquivos verificados")
    for name, path in result["paths"].items():
        st.write(f"{name}: {path}")
    for name in ("artifact.json", "validation_report.json", "audit_manifest.json"):
        st.download_button(
            f"Baixar {name}", data=result["downloads"][name], file_name=name,
            mime="application/json", key=f"training_download_{name}",
        )


def _render_probability_diagnostics(diagnostics: dict) -> None:
    probability = diagnostics["probability"]
    st.subheader("Discriminação e calibração no teste")
    st.caption(
        "Classe positiva: qualidade estrutural REDUCED preservada segundo a regra DockQ. "
        "Antes: sigmoid do score logístico; depois: probabilidade Venn–Abers. "
        "PR-AUC integra a curva por trapézios; AP usa incrementos de recall, por isso os valores podem diferir. "
        "O diagnóstico retrospectivo não escolhe o modelo final nem autoriza decisões."
    )
    bootstrap = probability["bootstrap"]
    st.caption(
        f"{probability['n_complexes']} complexos; prevalência positiva / referência PR: "
        f"{probability['positive_prevalence_pr_baseline']}. IC aproximado de "
        f"{100 * bootstrap['confidence_level']:g}% com {bootstrap['bootstrap_repetitions']} "
        "reamostragens de complexos inteiros; os modelos permanecem congelados."
    )
    labels = {
        "roc_auc": "ROC-AUC", "pr_auc_trapezoidal": "PR-AUC (trapézios)",
        "average_precision": "AP", "brier_score": "Brier", "log_loss": "Log loss",
        "expected_calibration_error_10bin": "ECE (10 bins)",
    }
    evaluations = probability["evaluations"]
    rows = []
    for name, evaluation in evaluations.items():
        for metric, label in labels.items():
            interval = evaluation["confidence_intervals"][metric]
            rows.append({"Modelo": "Antes de Venn–Abers" if name == "before_venn_abers" else "Depois de Venn–Abers",
                         "Métrica": label, "Valor": evaluation["metrics"][metric],
                         "IC inferior": interval["lower"], "IC superior": interval["upper"],
                         "Motivo de IC indisponível": interval["unavailable_reason"]})
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    curve_names = (("roc", "false_positive_rate", "true_positive_rate", "Curva ROC"),
                   ("precision_recall", "recall", "precision", "Curva precisão–recall"),
                   ("reliability_bins", "mean_probability", "observed_positive_fraction", "Confiabilidade"))
    for key, x, y, title in curve_names:
        points = []
        for name, evaluation in evaluations.items():
            for index, point in enumerate(evaluation["curves"][key]):
                if point[x] is not None and point[y] is not None:
                    points.append({**point, "Modelo": "Antes" if name == "before_venn_abers" else "Depois", "Ordem": index})
        if not points:
            st.info(f"{title}: indisponível sem linhas ou sem ambas as classes.")
            continue
        axis_labels = {"false_positive_rate": "Taxa de falsos positivos", "true_positive_rate": "Taxa de verdadeiros positivos",
                       "recall": "Recall", "precision": "Precisão", "mean_probability": "Probabilidade prevista média",
                       "observed_positive_fraction": "Fração preservada observada"}
        chart = alt.Chart(pd.DataFrame(points)).mark_line(point=True).encode(
            x=alt.X(f"{x}:Q", scale=alt.Scale(domain=[0, 1]), title=axis_labels[x]),
            y=alt.Y(f"{y}:Q", scale=alt.Scale(domain=[0, 1]), title=axis_labels[y]),
            color="Modelo:N", order="Ordem:Q", tooltip=["Modelo:N", f"{x}:Q", f"{y}:Q"],
        )
        if key in ("roc", "reliability_bins"):
            reference = alt.Chart(pd.DataFrame({x: [0, 1], y: [0, 1]})).mark_line(strokeDash=[4, 4], color="gray").encode(x=f"{x}:Q", y=f"{y}:Q")
            chart = chart + reference
        elif probability["positive_prevalence_pr_baseline"] is not None:
            reference = alt.Chart(pd.DataFrame({y: [probability["positive_prevalence_pr_baseline"]]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y=f"{y}:Q")
            chart = chart + reference
        st.write(title)
        st.altair_chart(chart, width="stretch")
    risk_curve = diagnostics.get("coverage_risk", {})
    if risk_curve.get("available"):
        risk_points = pd.DataFrame(risk_curve["points"]).dropna(subset=["empirical_run_risk"])
        if not risk_points.empty:
            st.write("Cobertura e risco observado")
            st.altair_chart(alt.Chart(risk_points).mark_line(point=True).encode(
                x=alt.X("run_coverage:Q", title="Fração de execuções aceitas", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("empirical_run_risk:Q", title="Erro observado entre aceites", scale=alt.Scale(domain=[0, 1])),
                order="threshold:Q", tooltip=["threshold:Q", "accepted_count:Q", "false_accepted_count:Q"],
            ), width="stretch")
            st.caption("Curva de uma grade fixa para diagnóstico; não substitui o gate de risco nem escolhe um novo limiar no teste.")
    comparison = diagnostics.get("descriptor_comparison")
    if comparison:
        st.write("Comparação de descritores nas mesmas partições")
        st.caption("População, interface e combinação usam os mesmos complexos de treino e calibração. O teste é usado só para avaliar; nenhuma variante é escolhida automaticamente.")
        variants = []
        for name, variant in comparison["variants"].items():
            variants.append({"Descritores": name, "Disponível": variant["available"],
                "Features": ", ".join(variant["features"]),
                **(variant["diagnostics"]["evaluations"]["after_venn_abers"]["metrics"] if variant["available"] else {"Motivo": variant["reason"]})})
        st.dataframe(pd.DataFrame(variants), hide_index=True)
    cost = diagnostics.get("operational_cost", {})
    if cost.get("available"):
        st.write("Custo simulado com fallback")
        _show_fields({"FULL de todos (s)": cost["baseline_full_sec"],
                      "REDUCED de todos (s)": cost["all_reduced_sec"],
                      "FULL dos fallbacks (s)": cost["fallback_full_sec"],
                      "Estratégia pareada (s)": cost["paired_strategy_sec"],
                      "Economia (s)": cost["saved_sec"],
                      "Speedup FULL / estratégia": cost["speedup_full_over_strategy"]})
        st.caption("Simulação da política nos pares completos, somando tempos do motor por execução. Overhead não medido e execuções falhas ou sem par requerem contabilização adicional; o runtime gate histórico continua separado.")
    authorized_cost = diagnostics.get("operational_cost_authorization_applied", {})
    if authorized_cost.get("available"):
        st.caption(f"Aplicando também a autorização atual do modelo, o custo pareado simulado é {authorized_cost['paired_strategy_sec']} s, com {authorized_cost['fallback_row_count']} fallbacks.")


def render_training_page() -> None:
    st.header("Análise e calibração para o artigo")
    st.caption("Batch FULL/REDUCED → dataset pareado → treinamento → Venn-Abers → risco por execução → teste separado → artifact.json")
    # Inherit the docking output until the user supplies a distinct training source.
    source = str(st.session_state.get("ppi4dock_output_dir", ""))
    previous = st.session_state.get("training_inherited_batch_dir", "")
    if st.session_state.get("training_batch_dir", "") in ("", previous):
        st.session_state["training_batch_dir"] = source
    st.session_state["training_inherited_batch_dir"] = source
    batch_dir = st.text_input("Pasta dos resultados do batch", key="training_batch_dir")
    settings_error = None
    try:
        profile, manifest_sha256, factorial = _article_dataset_settings(batch_dir)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        profile, manifest_sha256, factorial = "operational", None, False
        settings_error = str(exc)
        st.error(settings_error)
    st.caption("A configuração de análise é identificada pelo protocolo salvo. Os estudos de orçamento igual e reduzido permanecem separados.")
    out_dir = st.text_input("Pasta de saída do treinamento", key="training_output_dir")
    st.caption("Sequências repetidas usam o primeiro identificador em ordem alfabética, antes de considerar qualidade. As exclusões ficam no relatório; nenhum resultado é apagado.")
    factorial_initialization = (st.selectbox(
        "Inicialização do par fatorial", ["uniform", "seeded"], key="training_factorial_initialization",
        format_func=lambda value: "Uniforme" if value == "uniform" else "Semeada",
        help="Analisa pares da mesma inicialização, mantendo os outros resultados do controle.",
    ) if factorial else "uniform")
    alias_policy = "first_complex_id"
    batch_dir, out_dir = batch_dir.strip(), out_dir.strip()
    signature = (batch_dir, profile, out_dir, alias_policy, factorial_initialization, manifest_sha256)
    if st.session_state.get(_SIGNATURE_KEY) != signature:
        st.session_state.pop(_PREPARATION_KEY, None)
        st.session_state.pop(_RESULT_KEY, None)
    valid_inputs = bool(batch_dir.strip() and out_dir.strip() and settings_error is None)
    if st.button("1 - Analisar resultados / preparar dataset", key="training_prepare", disabled=not valid_inputs):
        try:
            if not valid_inputs:
                raise ValueError("Informe as pastas e um perfil válido.")
            with st.status("Preparando dataset", expanded=True) as status:
                progress = _progress_display()
                preparation_dir = Path(out_dir).resolve() / "dataset_preparation" / uuid4().hex
                try:
                    prepared = build_training_dataset(
                        batch_dir, preparation_dir, profile=profile,
                        fingerprint_alias_policy=alias_policy,
                        factorial_initialization=factorial_initialization,
                        progress_callback=progress,
                    )
                except Exception:
                    status.update(label="Preparação interrompida por erro", state="error")
                    raise
                st.session_state[_PREPARATION_KEY] = prepared
                st.session_state[_SIGNATURE_KEY] = signature
                status.update(label="Preparação concluída" if prepared.get("valid") else "Preparação bloqueada", state="complete" if prepared.get("valid") else "error")
        except Exception as exc:
            st.session_state.pop(_PREPARATION_KEY, None)
            st.error(f"Não foi possível preparar o dataset: {exc}")
    prepared = st.session_state.get(_PREPARATION_KEY)
    if prepared:
        _render_preparation(prepared)
    collisions = existing_training_outputs(out_dir) if out_dir.strip() else []
    if collisions:
        st.info("Treinamento já existe nesta pasta.")
        try:
            # Verify once per bundle byte-state, retaining only verified bytes in session.
            stamp = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in collisions)
            cached = st.session_state.get(_RESULT_KEY)
            if not cached or cached["stamp"] != stamp:
                cached = {"stamp": stamp, "result": load_training_result(out_dir)}
                st.session_state[_RESULT_KEY] = cached
            _render_training_result(cached["result"])
        except Exception as exc:
            st.session_state.pop(_RESULT_KEY, None)
            st.error(f"O treinamento existente não pôde ser verificado: {exc}")
    ready = bool(valid_inputs and prepared and prepared.get("valid") is True and not collisions)
    if st.button("2 - Treinar e calibrar", key="training_train", disabled=not ready):
        try:
            if not ready or st.session_state.get(_SIGNATURE_KEY) != signature:
                raise ValueError("Prepare e valide o dataset para os parâmetros atuais.")
            with st.status("Executando treinamento, calibração e avaliação", expanded=True) as status:
                progress = _progress_display()
                try:
                    result = train_prepared_dataset(prepared, out_dir, progress_callback=progress)
                except Exception:
                    status.update(label="Treinamento interrompido; confira o diagnóstico", state="error")
                    raise
                status.update(label="Treinamento concluído; confira os gates", state="complete")
            _render_training_result(result)
        except TrainingEvidenceError as exc:
            st.error(f"Treinamento interrompido por evidência insuficiente: {exc}")
            st.write("Etapa:", exc.stage)
            evidence = exc.evidence
            if exc.stage == "probability_calibration":
                _show_fields({
                    "Complexos independentes: observados / mínimo": f"{evidence['independent_complex_count']} / {evidence['minimum_independent_complexes']}",
                    "Complexos positivos: observados / mínimo": f"{evidence['positive_class_complex_count']} / {evidence['minimum_positive_class_complexes']}",
                    "Complexos negativos: observados / mínimo": f"{evidence['negative_class_complex_count']} / {evidence['minimum_negative_class_complexes']}",
                })
            st.json(evidence)
        except Exception as exc:
            st.error(f"Treinamento não concluído: {exc}")
