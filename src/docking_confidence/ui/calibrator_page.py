"""Create one Generalized Venn–Abers (2025) quality-margin calibrator."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from docking_confidence.application.generalized_calibrator import SCHEMA, create_from_source, load_calibrator
from docking_confidence.application.portable_calibrator_export import export_calibrator_zip


def render_calibrator_page():
    st.write("Carregue resultados de docking para estimar quanto a busca reduzida preserva a qualidade.")
    st.caption("Método: Generalized Venn–Abers, artigo de 2025, algoritmo 2 com perda quadrática.")
    st.caption("Novos calibradores usam 15 descritores fixos dos resultados REDUCED.")
    source = st.text_input(
        "Pasta de resultados, CSV pareado ou ZIP de resultados", key="gva_source",
        help="Aceita a pasta avaliacao_calibrador, a pasta do estudo ou o ZIP que contém as previsões e a qualidade.",
    ).strip()
    output = st.text_input("Nova pasta para salvar o calibrador", key="gva_output").strip()
    with st.expander("Configuração dos exemplos"):
        config = st.text_input("Arquivo de configuração (opcional)", key="gva_config",
            help="Quando o CSV registra a configuração original, ela é localizada automaticamente.").strip()
        st.caption("Treino e calibração são separados por complexo: 70% para o modelo-base e 30% para calibração, com divisão fixa.")
    signature = (source, output, config, SCHEMA)
    if st.session_state.get("gva_result_inputs") != signature:
        st.session_state.pop("gva_result", None)
    ready = bool(source and output)
    created_now = False
    if st.button("Criar Generalized Venn–Abers", key="gva_create", type="primary", disabled=not ready):
        st.session_state.pop("gva_result", None)
        try:
            with st.status("Criando calibrador", expanded=True) as status:
                bar = st.progress(0.0)
                result = create_from_source(source, output, config_path=config or None,
                    progress_callback=lambda fraction, text: bar.progress(fraction, text=text))
                status.update(label="Calibrador criado", state="complete")
            st.session_state["gva_result"] = result
            st.session_state["gva_result_inputs"] = signature
            created_now = True
        except Exception as exc:
            status.update(label="Confira os dados de entrada", state="error")
            st.error(f"Não foi possível criar o calibrador: {exc}")
    if output and (Path(output) / "artifact.json").is_file() and "gva_result" not in st.session_state:
        try:
            st.session_state["gva_result"] = load_calibrator(Path(output) / "artifact.json")
            st.session_state["gva_result_inputs"] = signature
        except Exception as exc:
            st.error(f"O calibrador salvo não pôde ser verificado: {exc}")
    result = st.session_state.get("gva_result")
    if result is None:
        return
    if result["artifact"]["schema_version"] != SCHEMA:
        st.info("Modelo histórico carregado. Para criar a versão ajustada, informe uma nova pasta de saída.")
    if created_now:
        st.success("Um calibrador Generalized Venn–Abers de qualidade foi criado e salvo nesta ação.")
    else:
        st.info("Calibrador salvo carregado e verificado.")
    st.caption("Origem do ajuste registrada no artefato: " + result["artifact"]["training_dataset_snapshot"]["path"])
    if result["artifact"]["schema_version"] == SCHEMA:
        features = result["artifact"]["feature_columns"]
        st.write(f"Descritores utilizados: {len(features)}")
        st.caption("Preditor híbrido: 50% linear + 50% árvores; uma única calibração Venn–Abers ao final.")
        with st.expander("Ver descritores do modelo"):
            st.write(", ".join(features))
    st.table(pd.DataFrame([
        {"Conjunto": "Treino do modelo-base" if role == "proper_training" else "Calibração Venn–Abers",
         "Complexos": values["complex_count"], "Exemplos": values["row_count"]}
        for role, values in result["artifact"]["partition"].items()
    ]))
    validation = result["report"].get("development_validation", {})
    if validation.get("available"):
        st.write(f"Erro médio na validação interna por complexo: {validation['group_mae']:.3f} de margem DockQ.")
        if result["artifact"]["schema_version"] == SCHEMA:
            st.caption("A validação separa complexos somente dentro do treino; a padronização e os dois componentes são ajustados em cada divisão. Os dados de calibração ficam reservados.")
        skill = validation.get("skill_against_fold_training_mean")
        if skill is not None and skill <= 0:
            st.warning("A validação interna ainda não mostrou ganho sobre a média do treino. A criação do arquivo não significa que o modelo esteja validado para uso automático.")
    diagnostics = result["report"].get("calibration_diagnostics", {})
    if diagnostics:
        st.caption(f"A calibração contém {diagnostics['negative_row_count']} exemplos de perda, de {diagnostics['negative_complex_count']} complexos.")
        if not diagnostics["point_can_be_negative"]:
            st.warning("A função pontual aprendida não contém margens negativas. Seu sinal não consegue identificar perdas individuais neste ajuste.")
    st.write("A saída estima a margem média de qualidade. Nos resultados observados, zero marca o critério de preservação e valores negativos indicam perda além desse critério. Uma média estimada positiva pode coexistir com perdas individuais.")
    st.caption("A margem estimada e a faixa de calibração estão em unidades de DockQ. Não representam porcentagem de confiança nem intervalo de cobertura de 90%.")
    portable_zip = export_calibrator_zip(result["paths"]["artifact.json"])
    st.download_button("Baixar calibrador", portable_zip, file_name="calibrador_generalized_venn_abers_2025.zip",
        mime="application/zip", key="gva_download", on_click="ignore")
    st.write("Calibrador salvo em:", result["paths"]["artifact.json"])
