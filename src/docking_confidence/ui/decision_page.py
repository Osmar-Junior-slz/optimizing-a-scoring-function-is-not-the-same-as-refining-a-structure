"""Guided preparation, freezing and independent evaluation of one decision rule.

No model fit, threshold search, docking or evaluation starts on a rerun. The
scientific service owns validation and immutable artifacts; this module only
collects explicit inputs and renders the resulting stage status.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import streamlit as st


def prepare_decision_protocol(*args, **kwargs):
    from docking_confidence.application.decision_validation import prepare_decision_protocol as prepare
    return prepare(*args, **kwargs)


def select_decision_policy(*args, **kwargs):
    from docking_confidence.application.decision_validation import select_decision_policy as select
    return select(*args, **kwargs)


def evaluate_frozen_policy(*args, **kwargs):
    from docking_confidence.application.decision_validation import evaluate_frozen_policy as evaluate
    return evaluate(*args, **kwargs)


def _read_split(path):
    split = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(split, dict) or split.get("schema_version") != "decision_group_split_v1":
        raise ValueError("A divisão precisa usar o formato decision_group_split_v1.")
    for key in ("validation_groups", "test_groups"):
        if not isinstance(split.get(key), list) or not split[key]:
            raise ValueError(f"A divisão precisa conter uma lista não vazia em {key}.")
    return split


def _default_model_path():
    created = st.session_state.get("gva_result") or {}
    path = created.get("paths", {}).get("artifact.json")
    if path:
        return str(path)
    binding_path = Path(__file__).resolve().parents[3] / "configs/calibrator.yaml"
    if binding_path.is_file():
        # Read the binding, rather than loading or fitting a potentially large model.
        import yaml
        binding = yaml.safe_load(binding_path.read_text(encoding="utf-8")) or {}
        path = binding.get("artifact_path")
        if path:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = binding_path.parent.parent / candidate
            return str(candidate.resolve())
    return ""


def _json_download(label, payload, filename, key):
    st.download_button(label, json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False,
                                       default=str).encode("utf-8"),
                       file_name=filename, mime="application/json", key=key, on_click="ignore")


def _number(value):
    if value is None:
        return "Indisponível"
    return str(value)


def _percentage(value):
    if value is None or not isinstance(value, (float, int)) or not math.isfinite(value):
        return "Indisponível"
    return f"{value:.2%}".replace(".", ",")


def _render_comparison(report):
    selected, baseline = report.get("selected"), report.get("baseline")
    if not isinstance(selected, dict) or not isinstance(baseline, dict):
        return
    rows = []
    for label, metrics in (("Regra congelada", selected), ("Aceitar todo REDUCED válido", baseline)):
        rows.append({
            "Regra": label,
            "Execuções planejadas": _number(metrics.get("planned_runs")),
            "Pares válidos": _number(metrics.get("valid_runs")),
            "Aceitos": _number(metrics.get("accepted_runs")),
            "Perdas entre aceitos": _number(metrics.get("adverse_accepted_runs")),
            "Preservação entre aceitos": _percentage(metrics.get("preservation")),
            "Cobertura das execuções": _percentage(metrics.get("coverage")),
            "Falhas ou ausências": _number(metrics.get("failed_or_missing_runs")),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Preservação usa as aceitações no denominador. Cobertura usa todas as execuções planejadas. Sem aceitações, a preservação fica indisponível.")
    difference = report.get("difference", {})
    if difference.get("baseline_equal"):
        st.warning("A regra produziu as mesmas aceitações que aceitar todo REDUCED válido. Essa comparação não demonstra ganho de seleção do calibrador.")
    if report.get("criteria"):
        labels = {
            "preservation_met": "Preservação exigida",
            "coverage_met": "Cobertura mínima",
            "minimum_complexes_met": "Quantidade mínima de complexos aceitos",
            "selective_rule": "Seleção de um subconjunto dos casos válidos",
            "empirical_baseline_improvement": "Melhora empírica frente a aceitar todos",
        }
        st.table(pd.DataFrame([
            {"Critério": labels.get(key, key), "Resultado": "Atendido" if value else "Não atendido"}
            for key, value in report["criteria"].items() if key in labels
        ]))
    if not selected.get("runtime_complete", False):
        st.caption("O custo integral da política não foi demonstrado com os tempos disponíveis.")


def _saved_result(key, signature):
    saved = st.session_state.get(key)
    return saved["result"] if saved and saved["signature"] == signature else None


def _service_error(prefix, exc):
    detail = str(exc)
    translations = (
        ("Model is not ready for prospective decisions", "O modelo ainda não registra uma medida estrutural padronizada e um contrato de execução compatíveis com uma avaliação nova. Reavalie os exemplos e crie um modelo compatível antes de preparar esse teste."),
        ("Configuration differs from the frozen model", "A configuração informada difere da configuração usada no ajuste do modelo. Use a configuração de origem para um diagnóstico ou crie um modelo no novo protocolo."),
        ("Previously exposed molecular groups", "A divisão inclui complexos cujos resultados já foram examinados. Use outros complexos para o teste novo ou mantenha esta análise como exploratória."),
        ("Declared validation/test groups overlap model fitting", "A validação ou o teste inclui complexos usados no ajuste ou na calibração. Prepare uma divisão com complexos separados."),
        ("Policy-validation and final-test molecular groups overlap", "Há complexos repetidos entre a validação da regra e o teste final. Cada complexo precisa permanecer em um único papel."),
    )
    friendly = next((message for marker, message in translations if marker in detail), detail)
    st.error(f"{prefix}: {friendly}")
    if friendly != detail:
        with st.expander("Detalhe da incompatibilidade"):
            st.code(detail, language=None)


def render_decision_page():
    st.subheader("Validar a decisão antes do uso")
    st.write("Prepare o protocolo, congele uma regra usando somente a validação e confira essa regra em complexos separados no teste final.")
    st.caption("O critério de mais de 90% é uma exigência empírica do estudo. A margem prevista pelo modelo não é uma porcentagem de confiança.")
    st.info("Resultados já examinados continuam exploratórios. Um teste novo exige fixar o protocolo antes de examinar os resultados finais.")

    project = Path(__file__).resolve().parents[3]
    st.markdown("**1. Preparar o protocolo antes do teste**")
    with st.form("decision_prepare_form", enter_to_submit=False):
        artifact = st.text_input("Modelo salvo (artifact.json)", value=_default_model_path(), key="decision_artifact").strip()
        config = st.text_input("Configuração dos exemplos do modelo",
                               value=str(project / "configs/local_refinement_standard_decision.yaml"),
                               key="decision_config").strip()
        split_path = st.text_input("Arquivo da divisão entre validação e teste (JSON)", key="decision_split",
                                   help="Use uma divisão registrada por fingerprints, com validation_groups, test_groups e prior_exposure_groups.").strip()
        prepare_output = st.text_input("Nova pasta para salvar o protocolo", key="decision_prepare_output").strip()
        preservation = st.number_input("Preservação exigida entre aceitos (%)", min_value=0.0, max_value=100.0,
                                        value=90.0, step=1.0, key="decision_min_preservation")
        st.caption("A taxa observada precisa ser estritamente maior que esse valor; não basta ser igual.")
        with st.expander("Mínimos de cobertura e tamanho da avaliação"):
            coverage = st.number_input("Cobertura mínima das execuções planejadas (%)", min_value=0.0,
                                       max_value=100.0, value=75.0, step=5.0, key="decision_min_coverage")
            minimum_complexes = st.number_input("Mínimo de complexos aceitos", min_value=1, value=20,
                                                step=1, key="decision_min_complexes")
            st.caption("Esses mínimos são escolhas explícitas do protocolo. O tamanho da amostra e a incerteza continuam sendo apresentados no relatório.")
        prospective = st.checkbox("Avaliação nova com resultados finais ainda não examinados",
                                  value=False, key="decision_prospective")
        st.caption("O JSON fixa os algoritmos e as seeds. Na ausência desses campos, usa AG, PSO e híbrido com seed 1. Nenhum docking é iniciado por este formulário.")
        prepare = st.form_submit_button("Preparar protocolo de decisão", key="decision_prepare", type="primary")
    prepare_signature = (artifact, config, split_path, prepare_output, preservation, coverage, minimum_complexes, prospective)
    if prepare:
        st.session_state.pop("decision_selected_result", None)
        st.session_state.pop("decision_evaluation_result", None)
        st.session_state["decision_protocol_path"] = ""
        st.session_state["decision_policy_path"] = ""
        try:
            if not all((artifact, config, split_path, prepare_output)):
                raise ValueError("Informe o modelo, sua configuração, a divisão dos complexos e uma nova pasta de saída.")
            split = _read_split(split_path)
            with st.spinner("Conferindo o modelo e registrando o protocolo"):
                prepared = prepare_decision_protocol(
                    artifact, split["validation_groups"], split["test_groups"], prepare_output,
                    config_path=config, minimum_preservation=float(preservation)/100,
                    minimum_coverage=float(coverage)/100, minimum_accepted_complexes=int(minimum_complexes),
                    prior_exposure_groups=split.get("prior_exposure_groups", ()),
                    analysis_role="prospective" if prospective else "exploratory",
                    algorithms=tuple(split.get("algorithms", ("ag", "pso", "hybrid"))),
                    seeds=tuple(split.get("seeds", (1,))), planned_runs=split.get("planned_runs"),
                )
            st.session_state["decision_prepared_result"] = {"signature": prepare_signature, "result": prepared}
            st.session_state["decision_protocol_path"] = str(prepared["protocol_path"])
        except Exception as exc:
            st.session_state.pop("decision_prepared_result", None)
            _service_error("Não foi possível preparar o protocolo", exc)
            st.caption("Se o modelo usa a medida estrutural histórica ou outro orçamento, reavalie os exemplos com a medida padronizada e crie um novo modelo compatível antes do teste novo.")
    prepared = _saved_result("decision_prepared_result", prepare_signature)
    if prepared:
        st.success("Protocolo registrado. A regra e o teste final ainda estão pendentes.")
        st.write("Protocolo salvo em:", prepared["protocol_path"])
        if prepared.get("protocol", {}).get("analysis_role") == "exploratory" or not prospective:
            st.warning("Este protocolo está marcado como exploratório e não autoriza decisões automáticas.")
        _json_download("Baixar protocolo", prepared["protocol"], "protocolo_decisao.json", "decision_download_protocol")

    st.markdown("**2. Selecionar e congelar a regra usando a validação**")
    with st.form("decision_select_form", enter_to_submit=False):
        protocol_path = st.text_input("Protocolo de decisão salvo", key="decision_protocol_path").strip()
        validation_path = st.text_input("CSV pareado da validação da regra", key="decision_validation_dataset").strip()
        select_output = st.text_input("Nova pasta para salvar a regra e o diagnóstico da validação",
                                      key="decision_select_output").strip()
        select = st.form_submit_button("Selecionar e congelar regra", key="decision_select")
    select_signature = (protocol_path, validation_path, select_output)
    if select:
        st.session_state.pop("decision_evaluation_result", None)
        st.session_state["decision_policy_path"] = ""
        try:
            if not all(select_signature):
                raise ValueError("Informe o protocolo, o CSV de validação e uma nova pasta de saída.")
            with st.spinner("Avaliando os candidatos somente na validação"):
                selected = select_decision_policy(protocol_path, validation_path, select_output)
            st.session_state["decision_selected_result"] = {"signature": select_signature, "result": selected}
            st.session_state["decision_policy_path"] = str(selected["policy_path"])
        except Exception as exc:
            st.session_state.pop("decision_selected_result", None)
            _service_error("Não foi possível congelar a regra", exc)
    selected = _saved_result("decision_selected_result", select_signature)
    if selected:
        policy = selected["policy"]
        if policy.get("status") == "selected":
            st.success("Regra congelada para o teste separado. A validação ainda não autoriza uso automático.")
        else:
            st.warning("Nenhuma regra útil foi encontrada nos critérios registrados. FULL continua necessário; o teste final não deve ser usado para escolher outro corte.")
        st.write("Regra salva em:", selected["policy_path"])
        _render_comparison(selected.get("report") or policy.get("validation_report", {}))
        _json_download("Baixar regra e diagnóstico", policy, "regra_congelada.json", "decision_download_policy")

    st.markdown("**3. Avaliar a regra congelada no teste final**")
    with st.form("decision_evaluate_form", enter_to_submit=False):
        policy_path = st.text_input("Regra congelada salva", key="decision_policy_path").strip()
        test_path = st.text_input("CSV pareado do teste final", key="decision_test_dataset").strip()
        evaluate_output = st.text_input("Nova pasta para salvar a avaliação final", key="decision_evaluate_output").strip()
        evaluate = st.form_submit_button("Avaliar regra no teste final", key="decision_evaluate")
    evaluate_signature = (policy_path, test_path, evaluate_output)
    if evaluate:
        try:
            if not all(evaluate_signature):
                raise ValueError("Informe a regra congelada, o CSV de teste e uma nova pasta de saída.")
            with st.spinner("Conferindo a independência e avaliando a regra congelada"):
                evaluated = evaluate_frozen_policy(policy_path, test_path, evaluate_output)
            st.session_state["decision_evaluation_result"] = {"signature": evaluate_signature, "result": evaluated}
        except Exception as exc:
            st.session_state.pop("decision_evaluation_result", None)
            _service_error("Não foi possível avaliar o teste final", exc)
    evaluated = _saved_result("decision_evaluation_result", evaluate_signature)
    if evaluated:
        report = evaluated["report"]
        if report.get("decision_capable") is True:
            st.success("A regra atendeu aos critérios registrados no teste separado. O relatório verificado deve acompanhar seu uso.")
        elif report.get("status") == "exploratory":
            st.warning("Avaliação exploratória concluída. Ela não habilita decisões automáticas nem constitui um novo teste confirmatório.")
        else:
            st.error("A avaliação final não habilitou a regra para dispensar FULL. Confira os critérios e o comparador abaixo.")
        _render_comparison(report)
        st.write("Relatório salvo em:", evaluated["report_path"])
        _json_download("Baixar avaliação final", report, "avaliacao_final_decisao.json", "decision_download_evaluation")
    else:
        st.caption("Teste final pendente. Criar o modelo, preparar o protocolo ou congelar a regra não habilita automaticamente a dispensa de FULL.")
