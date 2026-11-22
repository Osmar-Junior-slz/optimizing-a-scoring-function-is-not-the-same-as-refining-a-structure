from __future__ import annotations

from pathlib import Path
import time

import pytest
from streamlit.testing.v1 import AppTest

from docking_confidence.ui import batch_page

from docking_confidence.application.batch import _write_status
from docking_confidence.application.batch_telemetry import WorkerHeartbeat
from test_batch_monitoring import manifest_fixture, iso




def test_article_component_exposes_folder_only_real_flow_without_running_it(
    monkeypatch, tmp_path,
) -> None:
    from docking_confidence.ui import article_page
    model = tmp_path / "artifact.json"
    model.write_text('{"decision_capable": false}')
    monkeypatch.setattr(article_page, "configured_calibrator_path", lambda: model)
    def forbidden(*args, **kwargs):
        raise AssertionError("Rendering the article component must not prepare or launch a study")
    monkeypatch.setattr(article_page, "prepare_article_study", forbidden)
    monkeypatch.setattr(article_page, "launch_article_study", forbidden)
    # The main entrypoint now creates GVA and validates a separate decision.
    # Keep the legacy article component's guarantees scoped to that component.
    app = AppTest.from_string(
        "from docking_confidence.ui.article_page import render_article_page\n"
        "render_article_page()"
    ).run(timeout=20)
    assert not app.exception
    assert any(item.value == "Estudo do artigo: orientador e calibrador" for item in app.header)
    labels = {item.label for item in app.text_input}
    assert "Pasta das proteínas de teste" in labels
    assert "Nova pasta de resultados do artigo" in labels
    assert "Motor DockingPP (.zip)" not in labels
    assert "Identificador do complexo" not in labels
    assert "Sequência canônica do receptor" not in labels
    assert "Sequência canônica do parceiro móvel" not in labels
    assert not app.file_uploader
    assert not app.sidebar.selectbox
    visible_text = [str(item.value) for item in app.markdown]
    visible_text.extend(str(item.value) for item in app.caption)
    visible_text.extend(str(item.value) for item in app.info)
    assert any(
        "Tarefa: refinamento local da pose fornecida" in value
        for value in visible_text
    )
    assert any(
        "sequências canônicas são extraídas dos PDBs" in value
        for value in visible_text
    )
    assert any(
        "Motor DockingPP integrado ao projeto" in value
        for value in visible_text
    )
    assert any("5–1200 aa" in value for value in visible_text)
    assert not any("5–30" in value for value in visible_text)
    assert not app.selectbox  # Both mandatory blocks are one study, not alternatives.
    assert set(app.dataframe[0].value["Bloco"]) == {"Orientador", "Calibrador"}
    assert len(app.dataframe[0].value) == 6
    assert not any(item.label == "Tipo de busca" for item in app.selectbox)
    assert app.button(key="article_prepare").disabled
    app.run(timeout=20)
    assert not app.exception
    assert app.button(key="article_prepare").disabled


def test_batch_page_uses_integrated_motor_without_archive_discovery(
    monkeypatch,
) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("The integrated UI must not discover an external engine")
    monkeypatch.setattr("docking_confidence.application.batch_launcher.discover_dockingpp_archive", forbidden)
    app_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "docking_confidence"
        / "ui"
        / "app.py"
    )
    app = AppTest.from_file(str(app_path)).run(timeout=20)

    assert not app.exception
    assert "Motor DockingPP (.zip)" not in {item.label for item in app.text_input}


def test_ui_restores_from_url_and_only_reads_persisted_state_on_refresh(tmp_path, monkeypatch):
    manifest, status = manifest_fixture(tmp_path, 216)
    tasks = list(status["tasks"].values())
    for i, task in enumerate(tasks):
        task["state"] = "SUCCEEDED" if i < 488 else "FAILED" if i < 503 else "PENDING"
        if i < 503:
            task.update(started_at_utc=iso(time.time()-100), completed_at_utc=iso(time.time()-10))
    tasks[503].update(state="RUNNING", started_at_utc=iso(time.time()-20), attempt_count=1)
    _write_status(tmp_path / "batch_status.json", status)
    heartbeat = WorkerHeartbeat(tmp_path, manifest["batch_id"])
    job = next(j for j in manifest["plan"]["jobs"] if j["job_id"] == tasks[503]["job_id"])
    heartbeat.start_job(tasks[503], job, tmp_path / "engine")
    heartbeat.data.update(generation_current=47, generation_total=75, evaluations_current=4782, evaluations_total=7600)
    heartbeat.pulse()
    def forbidden(*args, **kwargs):
        raise AssertionError("Refresh must never launch/discover/cancel a batch")
    monkeypatch.setattr(batch_page, "launch_real_batch", forbidden)
    monkeypatch.setattr(batch_page, "discover_ppi4dock", forbidden)
    monkeypatch.setattr(batch_page, "request_batch_cancel", forbidden)
    before = {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    for _ in range(2):
        # New session, no session_state carried over; URL alone restores selection.
        app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
        app.query_params["ppi4dock_output_dir"] = str(tmp_path)
        app.run(timeout=20)
        assert not app.exception
        assert app.text_input(key="ppi4dock_output_dir").value == str(tmp_path)
        values = {item.label: item.value for item in app.metric}
        assert values["Total de jobs"] == "1296"
        assert values["Concluídos"] == "488"
        assert values["Falhas"] == "15"
        assert values["Pendentes"] == "792"
        assert values["EM EXECUÇÃO AGORA"] == "1"
        assert any("216 × 3 × 2 × 1 = 1296" in item.value for item in app.markdown)
        assert any("Geração 47 / 75" in item.value for item in app.markdown)
        assert any("PROCESSANDO NORMALMENTE" in item.value for item in app.success)
        for _ in range(2):
            app.run(timeout=20)
            assert not app.exception
            assert len(app.code) == 1  # Events don't accumulate on reruns.
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()} == before


def test_batch_launch_requires_submit_and_reruns_never_duplicate_it(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from test_batch_discovery import _write_target
    from docking_confidence.io.batch_discovery import discover_ppi4dock
    source = tmp_path / "input"
    _write_target(source)
    report = discover_ppi4dock(source)
    calls = []
    def launch(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(pid=123, log_path=tmp_path / "worker.log")
    monkeypatch.setattr(batch_page, "launch_real_batch", launch)
    app = AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_input_dir"] = str(source)
    app.session_state["ppi4dock_output_dir"] = str(tmp_path / "output")
    app.session_state["ppi4dock_validated_report"] = report
    app.run(timeout=20)
    assert not app.exception
    assert calls == []
    next(b for b in app.button if b.label == "Executar pares do artigo").click().run(timeout=20)
    assert not app.exception
    assert len(calls) == 1
    for _ in range(3):
        app.run(timeout=20)
        assert not app.exception
    assert len(calls) == 1


def test_monitor_fragment_registers_five_second_refresh_and_has_no_mutations(tmp_path, monkeypatch):
    # Exercise the real fragment body independently of the form, as timer ticks do.
    manifest_fixture(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("Timer invoked a batch mutation")
    monkeypatch.setattr(batch_page, "launch_real_batch", forbidden)
    monkeypatch.setattr(batch_page, "request_batch_cancel", forbidden)
    from streamlit.testing.v1.local_script_runner import LocalScriptRunner
    emitted = []
    original = LocalScriptRunner.forward_msgs
    def capture_messages(runner):
        messages = original(runner)
        emitted.extend(messages)
        return messages
    monkeypatch.setattr(LocalScriptRunner, "forward_msgs", capture_messages)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from docking_confidence.ui.batch_page import _render_durable_status\n"
        "_render_durable_status(st.session_state['output'])\n"
    )
    app.session_state["output"] = str(tmp_path)
    for _ in range(3):
        app.run(timeout=20)
        assert not app.exception
        assert any(item.label == "Total de jobs" for item in app.metric)
    # Check the timer registration in Streamlit's actual emitted messages.
    assert any(message.HasField("auto_rerun") and message.auto_rerun.interval == 5
               for message in emitted)


@pytest.mark.parametrize("alive", [True, False])
def test_monitor_shows_worker_startup_or_early_failure_before_manifest(tmp_path, monkeypatch, alive):
    monkeypatch.setattr(batch_page, "process_alive", lambda *args: alive)
    app = AppTest.from_string(
        "from docking_confidence.ui.batch_page import _render_durable_status\n"
        "import streamlit as st\n"
        "_render_durable_status(st.session_state['output'])\n"
    )
    app.session_state["output"] = str(tmp_path)
    app.session_state[batch_page._WORKER_LAUNCH_KEY] = {
        "output": str(tmp_path.resolve()), "pid": 123,
        "started_at_utc": iso(time.time()), "log_path": str(tmp_path / "worker.log"),
    }
    app.run(timeout=20)
    assert not app.exception
    if alive:
        assert any("aguardando o pré-voo" in item.value for item in app.info)
    else:
        assert any("encerrou antes de registrar" in item.value for item in app.error)
    assert not app.get("progress")  # No plan exists yet to define a percentage.
    # Selecting another output must not display a worker from the previous one.
    app.session_state["output"] = str(tmp_path / "other")
    app.run(timeout=20)
    assert not app.exception and not app.error
    assert any("Aguardando o início do lote" in item.value for item in app.info)


def test_monitor_shows_percentage_and_preliminary_eta_after_first_success(tmp_path, monkeypatch):
    from docking_confidence.application.batch_monitor import load_batch_monitor
    manifest, status = manifest_fixture(tmp_path)
    task = next(iter(status["tasks"].values()))
    task.update(state="SUCCEEDED", duration_seconds=20, started_at_utc=iso(900), completed_at_utc=iso(920))
    _write_status(tmp_path / "batch_status.json", status)
    heartbeat = WorkerHeartbeat(tmp_path, manifest["batch_id"])
    heartbeat.data["worker_started_at_utc"] = iso(900)
    heartbeat.pulse()
    view = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: True)
    monkeypatch.setattr(batch_page, "load_batch_monitor", lambda output: view)
    app = AppTest.from_string(
        "from docking_confidence.ui.batch_page import _render_durable_status\n"
        f"_render_durable_status({str(tmp_path)!r})"
    ).run(timeout=20)
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Progresso do lote"] == "8.3%"
    assert metrics["Tempo decorrido nesta execução"] == "1m 40s"
    assert metrics["Tempo restante estimado"] == "3m 40s"
    assert metrics["Término previsto"] == batch_page._local_time(iso(1220))
    assert any("Estimativa preliminar" in item.value for item in app.info)
    assert any("média geral" in item.value for item in app.caption)
    # A lost worker must remove the old prediction when the panel refreshes.
    view.update(eta=None, health={**view["health"], "kind": "stopped", "process_alive": False})
    app.run(timeout=20)
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Tempo restante estimado"] == "Indisponível"
    assert metrics["Término previsto"] == "Aguardando estimativa"
    assert not any("Estimativa preliminar" in item.value for item in app.info)


def test_monitor_explains_initial_eta_wait_and_overdue_job(tmp_path, monkeypatch):
    from docking_confidence.application.batch_monitor import load_batch_monitor
    manifest_fixture(tmp_path)
    view = load_batch_monitor(tmp_path, now=1000, probe=lambda *args: True)
    view["health"]["kind"] = "healthy"
    monkeypatch.setattr(batch_page, "load_batch_monitor", lambda output: view)
    app = AppTest.from_string(
        "from docking_confidence.ui.batch_page import _render_durable_status\n"
        f"_render_durable_status({str(tmp_path)!r})"
    ).run(timeout=20)
    assert not app.exception
    assert any("aguardando a primeira execução" in item.value for item in app.caption)
    view["eta"] = {"seconds": 0, "overdue_jobs": 1, "finish_at_utc": None}
    app.run(timeout=20)
    assert not app.exception
    assert any("ultrapassou a duração média" in item.value for item in app.caption)
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["Tempo restante estimado"] == "Calculando"
    assert metrics["Término previsto"] == "Aguardando estimativa"












def test_batch_monitor_displays_calibrator_decision_and_saved_full(tmp_path, monkeypatch):
    from docking_confidence.application.batch_monitor import load_batch_monitor

    manifest_fixture(tmp_path)
    status = load_batch_monitor(tmp_path)
    status["counts"]["skipped_calibrator"] = 1
    status["confidence_decisions"] = [{
        "complex_id": "1abc_AB", "reduced_job_id": "pso_reduced_1",
        "action": "ACCEPT_REDUCED", "reason": "accepted_by_calibrated_policy",
        "probability_lower": 0.94, "probability_upper": 0.97, "threshold": 0.9,
    }]
    monkeypatch.setattr(batch_page, "load_batch_monitor", lambda output: status)
    app = AppTest.from_string(
        "import streamlit as st\n"
        "from docking_confidence.ui.batch_page import _render_durable_status\n"
        "_render_durable_status(st.session_state['output'])"
    )
    app.session_state["output"] = str(tmp_path)
    app.run(timeout=20)
    assert not app.exception
    metrics = {item.label: item.value for item in app.metric}
    assert metrics["FULL evitados pelo calibrador"] == "1"
    decision_frame = next(item.value for item in app.dataframe if "action" in item.value.columns)
    assert decision_frame.iloc[0]["action"] == "ACCEPT_REDUCED"
    assert decision_frame.iloc[0]["reason"] == "accepted_by_calibrated_policy"






@pytest.mark.parametrize("old_mode", ["reduced_vs_full", "reduced", "full", "calibrated_reduced"])
def test_article_page_migrates_old_browser_modes_without_launching(old_mode, monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("Opening or migrating the article page must not launch docking")
    monkeypatch.setattr(batch_page, "launch_real_batch", forbidden)
    app=AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_real_protocol"]=old_mode
    app.session_state["ppi4dock_real_algorithms"]=["pso"]
    app.session_state["ppi4dock_real_seed_count"]=5
    app.run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="ppi4dock_real_protocol").value == "operational_cascade"
    assert app.multiselect(key="ppi4dock_real_algorithms").value == ["ag","pso","hybrid"]
    assert app.number_input(key="ppi4dock_real_seed_count").value == 1
    assert not any("calibrador" in item.label.lower() for item in app.text_input)
    assert not any("Verificar calibrador" in item.label for item in app.button)


def test_article_collection_always_requests_complete_pairs(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_batch_discovery import _write_target
    from docking_confidence.io.batch_discovery import discover_ppi4dock
    source=tmp_path/"input";_write_target(source);calls=[]
    monkeypatch.setattr(batch_page,"launch_real_batch",lambda **kw:calls.append(kw) or SimpleNamespace(pid=123,log_path=tmp_path/"worker.log"))
    app=AppTest.from_string("from docking_confidence.ui.batch_page import render_batch_page\nrender_batch_page()")
    app.session_state["ppi4dock_input_dir"]=str(source)
    app.session_state["ppi4dock_output_dir"]=str(tmp_path/"output")
    app.session_state["ppi4dock_validated_report"]=discover_ppi4dock(source)
    app.run(timeout=30)
    next(b for b in app.button if b.label=="Executar pares do artigo").click().run(timeout=30)
    assert not app.exception
    assert len(calls)==1
    assert calls[0]["collect_operational_pairs"] is True
    assert calls[0]["confidence_artifact"] is None
    assert calls[0]["protocol"]=="operational_cascade"
    assert calls[0]["algorithms"]==["ag","pso","hybrid"]
