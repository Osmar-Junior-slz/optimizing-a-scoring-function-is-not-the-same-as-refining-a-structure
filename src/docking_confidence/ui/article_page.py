"""The article has one workflow; controls and confidence evaluation are mandatory."""
from __future__ import annotations

from pathlib import Path
import io
import zipfile

import pandas as pd
import streamlit as st

from docking_confidence.application.article_study import (
    BLOCKS, PROTOCOL, STATUS, launch_article_study, load_article_protocol,
    prepare_article_study, request_article_cancel,
)
from docking_confidence.application.batch import _load_json_object, load_batch_status
from docking_confidence.application.batch_monitor import process_alive
from docking_confidence.application.calibrated_batch import configured_calibrator_path
from docking_confidence.domain.mobile_partner import DEFAULT_MOBILE_PARTNER_DOMAIN


def _restore(protocol):
    st.session_state["ppi4dock_input_dir"] = protocol["input_dir"]
    st.session_state["article_pose_source"] = protocol["pose_source"]


def _worker_alive(root):
    path = root / "article_worker.json"
    if not path.exists():
        return False
    info = _load_json_object(path)
    return process_alive(info["pid"], info.get("started_at_utc"))


@st.cache_data(ttl=60, max_entries=2)
def _report_zip(folder, report_mtime):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(Path(folder).rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(folder).as_posix())
    return buffer.getvalue()


@st.fragment(run_every="5s")
def _progress(output):
    root = Path(output)
    state_path = root / STATUS
    state = _load_json_object(state_path) if state_path.exists() else {"state": "PREPARED"}
    labels = {"PREPARED": "Preparado", "VALIDATING": "Conferindo protocolo e entradas", "RUNNING": "Executando", "ANALYZING": "Gerando relatório conjunto",
              "COMPLETED": "Concluído", "CANCELLED": "Interrompido", "FAILED": "Falhou"}
    st.subheader("Andamento do estudo")
    st.write(labels.get(state["state"], state["state"]))
    for name, label in zip(BLOCKS, ("Controles do orientador", "Avaliação do Venn–Abers")):
        status = load_batch_status(root / name)
        if status:
            counts = status["counts"]
            done = counts["succeeded"] + counts["failed"]
            st.progress(done / counts["planned"] if counts["planned"] else 0,
                        text=f"{label}: {done}/{counts['planned']} · {counts['failed']} falhas registradas")
    if state.get("error"):
        st.error(state["error"])
    alive = _worker_alive(root)
    if alive is False and state["state"] in {"VALIDATING", "RUNNING", "ANALYZING"}:
        st.warning("O processo não está ativo. Retome o estudo para aproveitar os resultados já gravados.")
    if alive is not False and st.button("Solicitar interrupção", key="article_cancel"):
        request_article_cancel(root)
        st.info("Interrupção solicitada; a execução atual termina antes de parar.")
    report_path = root / "analise_artigo/article_report.json"
    if state["state"] == "COMPLETED" and report_path.is_file():
        report = _load_json_object(report_path)
        st.subheader("Resultado conjunto para o artigo")
        st.dataframe(pd.DataFrame([
            {"Método": name, **evaluation["metrics"]}
            for name, evaluation in report["calibration"]["comparison"]["evaluations"].items()
        ]), hide_index=True)
        st.image(str(root / "analise_artigo/calibracao_roc_pr.png"))
        st.image(str(root / "analise_artigo/risco_cobertura_teste.png"))
        st.caption("FULL foram mantidos para conferir a qualidade. Decisões e custos simulados estão identificados no relatório; não representam economia já realizada.")
        folder = root / "analise_artigo"
        st.download_button("Baixar relatório e tabelas do artigo", _report_zip(str(folder), report_path.stat().st_mtime_ns),
                           file_name="resultados_artigo.zip", mime="application/zip", key="article_download")
        st.write(f"Resultados completos preservados em: {root}")


def render_article_page():
    st.header("Estudo do artigo: orientador e calibrador")
    st.write("Uma única execução reúne os controles de amplitude e inicialização, a avaliação do Venn–Abers e o relatório conjunto.")
    st.caption("Tarefa: refinamento local da pose fornecida. A origem da pose e sua relação com a referência devem ser descritas.")
    st.caption("Motor DockingPP integrado ao projeto. Cadeias e sequências canônicas são extraídas dos PDBs. "
               f"Parceiro móvel elegível: {DEFAULT_MOBILE_PARTNER_DOMAIN.label}.")
    columns = st.columns(2)
    source = columns[0].text_input("Pasta das proteínas de teste", key="ppi4dock_input_dir", bind="query-params")
    output = columns[1].text_input("Nova pasta de resultados do artigo", key="ppi4dock_output_dir", bind="query-params")
    pose_source = st.text_area("Origem da pose fornecida e relação com a referência", key="article_pose_source")
    model_path = configured_calibrator_path()
    if model_path is None or not model_path.is_file():
        st.error("O calibrador instalado não foi encontrado. Prepare o modelo antes de iniciar o artigo.")
        return
    model = _load_json_object(model_path)
    st.info("O Venn–Abers instalado será usado após cada REDUCED de 5.700 avaliações. "
            "O FULL também executará para conferir as decisões. Os quatro controles do orientador usam 7.600 avaliações cada.")
    st.caption("Análise atual: curvas de risco e cobertura com limites binomiais exatos por complexo. "
               "A taxa máxima de erro ainda não foi definida; nenhum limiar de aceitação será aprovado nesta execução.")
    st.caption(f"Modelo: {model_path.name} · {model_path.parent.name} · AG, PSO e híbrido · população 100 · seed 1.")
    if not model["decision_capable"]:
        st.warning("O modelo gera probabilidades, mas ainda não tem validação suficiente para dispensar FULL automaticamente. O estudo medirá seu desempenho sem alterar essa restrição.")
    st.dataframe(pd.DataFrame([
        {"Bloco": "Orientador", "Espaço": extent, "Inicialização": initialization, "Avaliações": 7600}
        for extent in ("Reduzido", "Ampliado") for initialization in ("Uniforme", "Semeada")
    ] + [
        {"Bloco": "Calibrador", "Espaço": "Reduzido", "Inicialização": "Uniforme", "Avaliações": 5700},
        {"Bloco": "Calibrador", "Espaço": "Ampliado", "Inicialização": "Uniforme", "Avaliações": 7600},
    ]), hide_index=True)
    st.caption("São seis condições por algoritmo e complexo, com controles e avaliação armazenados separadamente no mesmo estudo. As proteínas do desenvolvimento serão bloqueadas no teste.")
    if st.button("Validar e preparar estudo do artigo", type="primary", key="article_prepare",
                 disabled=not (source.strip() and output.strip() and pose_source.strip())):
        try:
            with st.status("Validando proteínas, calibrador e protocolo..."):
                prepare_article_study(source, output, pose_source=pose_source)
            st.success("Estudo preparado. Nenhum docking foi iniciado.")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if not output.strip() or not (Path(output) / PROTOCOL).is_file():
        return
    try:
        root = Path(output).resolve()
        protocol = load_article_protocol(root)
        st.caption(f"Modelo congelado neste estudo: {Path(protocol['artifact_path']).parent.name} · SHA-256 {protocol['artifact_sha256'][:16]}…")
        st.button("Carregar dados do estudo salvo", key="article_restore", on_click=_restore, args=(protocol,))
        count = protocol["selection"]["selected_count"]
        total = sum(len(plan["jobs"]) * len(plan["targets"]) for plan in protocol["block_plans"].values())
        st.write(f"Protocolo preparado: **{count} complexos distintos · {total} execuções**. O calibrador e a grade de limiares ficam congelados antes do teste; o estudo gera curvas, sem aprovar uma regra de aceitação.")
        if count != 100:
            st.caption(f"Esta seleção contém {count} complexos distintos; os resultados serão apresentados com esse número real.")
        inputs_match = (Path(source).expanduser().resolve() == Path(protocol["input_dir"])
                        and pose_source.strip() == protocol["pose_source"])
        if not inputs_match:
            st.warning("Os campos diferem do protocolo salvo. Carregue os dados salvos ou prepare uma nova pasta.")
        if st.button("Iniciar / retomar estudo completo", type="primary", key="article_start",
                     disabled=not inputs_match or _worker_alive(root) is not False):
            worker = launch_article_study(root, resume=True)
            st.success(f"Estudo iniciado. Registro de execução: {worker.log_path}")
        _progress(str(root))
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
