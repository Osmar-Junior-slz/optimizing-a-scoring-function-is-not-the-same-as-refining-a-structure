from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import time
import sys

import pandas as pd
import streamlit as st

from docking_confidence.application.batch import (
    BatchError, _job_snapshot, request_batch_cancel,
)
from docking_confidence.application.batch_monitor import load_batch_monitor, process_alive, task_duration
from docking_confidence.application.batch_telemetry import timestamp
from docking_confidence.application.batch_launcher import (
    launch_real_batch,
)
from docking_confidence.application.batch_reporting import discovery_rows
from docking_confidence.application.planning import build_execution_plan
from docking_confidence.domain.execution import Algorithm, BaseSearchConfig, DockingTask, ProtocolMode
from docking_confidence.domain.execution import (
    FACTORIAL_EVALUATION_BUDGET, FACTORIAL_PRIMARY_OUTCOME, FIXED_EVALUATION_BUDGET, FactorialExperimentConfig,
)
from docking_confidence.application.experimental_protocol import (
    ExperimentalProtocolPlanConflictError, PROTOCOL_FILENAME,
    load_experimental_protocol, prepare_experimental_protocol,
)
from docking_confidence.domain.mobile_partner import DEFAULT_MOBILE_PARTNER_DOMAIN
from docking_confidence.io.batch_discovery import (
    BatchDiscoveryReport,
    discover_ppi4dock,
)


_REPORT_KEY = "ppi4dock_validated_report"
_WORKER_LAUNCH_KEY = "ppi4dock_worker_launch"


def _factorial_form_values(protocol: dict) -> dict:
    """Restore only saved settings that this form can reproduce exactly."""
    plan = protocol["plan"]
    jobs = plan["jobs"]
    profile = next(job["profile"] for job in jobs if job["profile"]["search_extent"] == "reduced")
    seeds = sorted({job["seed"] for job in jobs})
    if seeds != list(range(seeds[0], seeds[-1] + 1)):
        raise ValueError("As seeds salvas não formam a sequência contínua usada nesta tela.")
    if (profile["budget_policy"] != FIXED_EVALUATION_BUDGET
        or profile["evaluation_budget"] != FACTORIAL_EVALUATION_BUDGET
        or profile["generations"] != BaseSearchConfig().generations
        or profile["top_pockets"] != 1):
        raise ValueError("Este protocolo usa um orçamento ou perfil antigo que esta tela não consegue reproduzir.")
    algorithms = list(dict.fromkeys(job["algorithm"] for job in jobs))
    controls = FactorialExperimentConfig(
        profile["evaluation_budget"], profile["search_scale"], profile["max_translation"],
        profile["max_rotation_degrees"], profile["pose_source"], profile["execution_order_seed"],
    )
    restored = build_execution_plan(
        algorithms=[Algorithm(value) for value in algorithms], protocol_mode=ProtocolMode.FACTORIAL,
        base_config=BaseSearchConfig(population=profile["population"], top_pockets=1,
                                     docking_task=DockingTask.LOCAL_REFINEMENT, experiment=controls),
        seeds=seeds,
    )
    if [_job_snapshot(job) for job in restored] != jobs:
        raise ValueError("Esta tela não consegue reproduzir exatamente todas as condições salvas.")
    return {
        "ppi4dock_input_dir": plan["input_root"],
        "ppi4dock_real_protocol": "factorial_2x2",
        "ppi4dock_real_algorithms": algorithms,
        "ppi4dock_real_docking_task": "local_refinement",
        "ppi4dock_real_seed_start": seeds[0], "ppi4dock_real_seed_count": len(seeds),
        "ppi4dock_real_timeout": plan.get("execution_limits", {}).get("timeout_seconds") or 86400.0,
        "factorial_population": profile["population"], "factorial_scale": profile["search_scale"],
        "factorial_translation": profile["max_translation"], "factorial_rotation": profile["max_rotation_degrees"],
        "factorial_pose_source": profile["pose_source"], "factorial_order_seed": profile["execution_order_seed"],
        "factorial_analysis_role": protocol["analysis"]["analysis_role"],
    }


def _apply_factorial_form_values(values: dict) -> None:
    for key, value in values.items():
        st.session_state[key] = value
    st.session_state.pop("factorial_prepared", None)


def _render_saved_factorial_protocol(output_text: str) -> None:
    path = Path(output_text).expanduser() / PROTOCOL_FILENAME
    if not output_text.strip() or not path.is_file():
        return
    try:
        protocol = load_experimental_protocol(path)
        values = _factorial_form_values(protocol)
    except (BatchError, KeyError, TypeError, ValueError, OSError) as exc:
        st.warning(f"O protocolo salvo não pôde ser carregado para o formulário: {exc}")
        return
    plan = protocol["plan"]
    saved_runtime = plan["engine"]["descriptor"].get("python_runtime_version")
    saved_python = plan["engine"]["descriptor"].get("python_executable")
    runtime_changed = saved_runtime is not None and saved_runtime != sys.version
    runtime_changed |= saved_python is not None and saved_python != sys.executable
    with st.expander("Protocolo já salvo nesta pasta", expanded=True):
        st.write(f"{len(plan['targets'])} complexos · {', '.join(value.upper() for value in values['ppi4dock_real_algorithms'])} · {len(plan['targets']) * len(plan['jobs'])} execuções planejadas.")
        if st.button("Carregar configuração do protocolo salvo", key="factorial_load_saved",
                     on_click=_apply_factorial_form_values, args=(values,)):
            st.success("Configuração salva carregada. Valide a entrada e prepare o protocolo para conferir a compatibilidade.")
        if runtime_changed:
            old_version = saved_runtime.split()[0] if saved_runtime else "de outro ambiente"
            st.warning(f"O protocolo salvo usa Python {old_version}; esta interface usa Python {sys.version.split()[0]}. O ambiente de execução mudou. Prepare em outra pasta para usar o ambiente atual.")
        st.caption("A configuração pode ser recuperada. O protocolo existente permanece vinculado aos dados, ao motor e ao ambiente em que foi preparado.")


def _render_report(report: BatchDiscoveryReport) -> None:
    total = len(report.targets)
    valid = len(report.valid_targets)
    eligible = len(report.eligible_targets)
    outside = sum(
        target.valid and target.eligible is False for target in report.targets
    )
    columns = st.columns(5)
    columns[0].metric("Reconhecidos", total)
    columns[1].metric("PDBs válidos", valid)
    columns[2].metric(
        f"Elegíveis {report.mobile_partner_domain.label}", eligible
    )
    columns[3].metric("Fora do domínio", outside)
    columns[4].metric("Erros", len(report.issues))

    if eligible < 30:
        st.error(
            f"Há somente {eligible} alvos na faixa operacional configurada. "
            "Isso não sustenta sozinho uma calibração legítima."
        )
    if outside:
        st.warning(
            f"{outside} complexos são estruturalmente legíveis, mas o parceiro "
            "móvel está fora da faixa operacional configurada de "
            f"{report.mobile_partner_domain.label}."
        )
    if report.has_errors:
        st.error(
            "O pré-voo encontrou entradas inválidas. A execução fica bloqueada "
            "até os erros serem resolvidos."
        )

    rows = discovery_rows(report)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        st.dataframe(
            frame[
                [
                    "complex_id",
                    "valid",
                    "eligible_for_docking",
                    "receptor_chain",
                    "ligand_chain",
                    "receptor_residues",
                    "ligand_residues",
                    "eligibility_reason",
                    "errors",
                ]
            ],
            hide_index=True,
        )
        st.download_button(
            "Baixar relatório de pré-voo CSV",
            data=frame.to_csv(index=False).encode("utf-8"),
            file_name="ppi4dock_preflight.csv",
            mime="text/csv",
            icon=":material/download:",
        )
    if report.issues:
        st.subheader("Erros de descoberta/validação")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "código": issue.code,
                        "complexo": issue.complex_id,
                        "entrada": issue.relative_path,
                        "mensagem": issue.message,
                    }
                    for issue in report.issues
                ]
            ),
            hide_index=True,
        )


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return "indisponível"
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}h {minutes:02d}m {seconds:02d}s" if hours else f"{minutes}m {seconds:02d}s"


def _local_time(value, *, short=False) -> str:
    instant = timestamp(value)
    if instant is None:
        return "indisponível"
    return datetime.fromtimestamp(instant, timezone.utc).astimezone().strftime(
        "%H:%M:%S" if short else "%d/%m/%Y %H:%M:%S %Z"
    )


def _remember_worker_launch(launched, output_text: str) -> None:
    st.session_state["ppi4dock_worker_pid"] = launched.pid
    st.session_state["ppi4dock_worker_kind"] = "real"
    st.session_state[_WORKER_LAUNCH_KEY] = {
        "output": str(Path(output_text).expanduser().resolve(strict=False)),
        "pid": launched.pid, "log_path": str(launched.log_path),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }


@st.fragment(run_every="5s")
def _render_durable_status(output_text: str) -> None:
    """Read-only fragment: no launch, cancellation, discovery, or engine calls."""
    if not output_text.strip():
        return
    try:
        status = load_batch_monitor(output_text)
    except Exception as exc:  # noqa: BLE001 - local status is user-visible
        st.error(f"Não foi possível ler o estado persistido: {exc}")
        return
    if status is None:
        st.subheader("Monitoramento do lote")
        st.caption("Atualização automática a cada 5 segundos.")
        launch = st.session_state.get(_WORKER_LAUNCH_KEY, {})
        if launch.get("output") == str(Path(output_text).expanduser().resolve(strict=False)):
            alive = process_alive(launch["pid"], launch["started_at_utc"])
            if alive is False:
                st.error("O worker encerrou antes de registrar o estado do lote. Consulte o log de inicialização.")
            else:
                st.info("Worker iniciado: aguardando o pré-voo e o registro do plano. O percentual e a previsão aparecerão aqui automaticamente.")
            st.caption(f"PID: {launch['pid']} · Início: {_local_time(launch['started_at_utc'])} · Log: {launch['log_path']}")
        else:
            st.info("Aguardando o início do lote nesta pasta de saída. O monitoramento acompanhará a execução automaticamente.")
        return
    st.subheader("Estado persistido do lote")
    st.caption(f"Estado: {status['state']} · Atualização automática a cada 5 segundos · PID: {status['worker_pid'] or 'não registrado'}")
    counts = status["counts"]
    metrics = st.columns(5)
    for column, label, key in zip(metrics,
        ["Total de jobs", "Concluídos", "Falhas", "Pendentes", "EM EXECUÇÃO AGORA"],
        ["planned", "succeeded", "failed", "pending", "running"]):
        column.metric(label, counts[key])
    if counts.get("skipped_calibrator") or status.get("confidence_decisions"):
        st.metric("FULL evitados pelo calibrador", counts.get("skipped_calibrator", 0))
        st.caption("FULL evitados contam como decisões concluídas no progresso; não são execuções realizadas do motor.")
    st.markdown("**Progresso global**")
    st.progress(min(1.0, max(0.0, status["global_fraction"])), text=(
        f"{status['processed']} / {counts['planned']} jobs processados · {status['global_fraction']:.1%}"
    ))
    st.caption("O percentual global conta jobs finalizados, incluindo falhas e FULL evitados. O avanço da busca atual aparece abaixo.")
    eta = status["eta"]
    eta_ready = eta is not None and not eta.get("overdue_jobs")
    completed = status["state"] in {"COMPLETED", "COMPLETED_WITH_ERRORS"} and status["processed"] == counts["planned"]
    eta_placeholder = "Calculando" if status["state"] == "RUNNING" and status["health"]["kind"] == "healthy" else "Indisponível"
    with st.container(horizontal=True):
        st.metric("Progresso do lote", f"{status['global_fraction']:.1%}", border=True)
        st.metric("Tempo decorrido nesta execução", _duration(status.get("elapsed_seconds")), border=True)
        st.metric("Tempo restante estimado", _duration(eta["seconds"]) if eta_ready else "Concluído" if completed else eta_placeholder, border=True)
        st.metric("Término previsto", _local_time(eta["finish_at_utc"]) if eta_ready else _local_time(status.get("completed_at_utc")) if completed else "Aguardando estimativa", border=True)
    if eta_ready:
        qualifier = " preliminar" if eta["preliminary"] else ""
        st.info(f"Estimativa{qualifier} de tempo restante: {_duration(eta['seconds'])} · Término previsto: {_local_time(eta['finish_at_utc'])} · Base: {eta['sample_count']} execuções concluídas com sucesso.")
        if eta["preliminary"]:
            st.caption("Poucas amostras: a previsão pode variar bastante. Usa tempos da mesma combinação de algoritmo, protocolo e condição; combinações ainda sem amostras usam a média geral.")
        else:
            st.caption("Média dos tempos reais por algoritmo, protocolo e condição, com pelo menos 5 sucessos no lote e 3 por combinação restante. O tamanho dos complexos pode variar.")
        if eta["fallback_jobs"]:
            st.caption(f"{eta['fallback_jobs']} jobs restantes ainda usam a média geral por falta de amostras da própria combinação.")
    elif eta and eta.get("overdue_jobs"):
        st.caption("Previsão em recálculo: a execução atual ultrapassou a duração média observada. Uma nova conclusão atualizará a estimativa.")
    elif status["state"] == "RUNNING" and status["health"]["kind"] == "healthy":
        st.caption("Estimativa em cálculo: aguardando a primeira execução concluída com sucesso e duração registrada. O percentual já acompanha o trabalho realizado.")
    elif not completed:
        st.caption("Previsão suspensa enquanto o worker estiver parado, sem atividade recente ou com o lote interrompido.")
    if status["count_mismatch"]:
        st.warning("A contagem do estado diverge do manifesto. O total exibido vem do plano persistido; nenhum job foi alterado.")

    health = status["health"]
    age = health["age_seconds"]
    if age is not None:
        st.caption(f"Última atividade: há {int(age)} segundos (heartbeat do worker)")
    else:
        st.caption(f"Heartbeat ainda não disponível. Último estado gravado: {_local_time(status['status_updated_at_utc'])}")
    if health["kind"] == "finished":
        st.info(f"Worker finalizado: {status['state']}.")
    elif health["kind"] == "healthy":
        st.success("🟢 PROCESSANDO NORMALMENTE")
    elif health["kind"] == "stopped":
        st.warning("🟡 PROCESSO DO WORKER NÃO ESTÁ ATIVO — verificado no sistema operacional. O estado persistido foi preservado.")
    elif age is not None:
        st.warning(f"🟡 SEM ATUALIZAÇÃO HÁ {age / 60:.1f} MINUTOS")
    else:
        st.warning("🟡 SEM HEARTBEAT — este lote pode ter sido iniciado com uma versão anterior.")
    if health["kind"] in {"stale", "unknown"}:
        st.caption("Processo presente; atraso de telemetria não confirma travamento." if health["process_alive"] is True
                   else "Não foi possível confirmar o processo; ausência de heartbeat não prova que ele morreu.")
    if status["orphaned_running"]:
        st.warning(f"{status['orphaned_running']} job(s) ainda marcado(s) RUNNING no arquivo, sem worker ativo. Esses registros foram preservados e não são contados como execução atual.")

    st.subheader("Processando agora")
    if not status["active_jobs"]:
        st.write("Nenhum job ativo registrado.")
    for task in status["active_jobs"]:
        with st.container(border=True):
            if health["kind"] == "stopped":
                st.caption("Último job registrado como ativo; o processo do worker não está presente.")
            st.write(f"**{task['complex_id']} · {str(task.get('algorithm', '')).upper()} · {str(task.get('protocol', '')).upper()} · seed {task.get('seed')}**")
            st.caption(f"job_id: {task['job_id']} · Início: {_local_time(task.get('started_at_utc'))} · Tempo decorrido: {_duration(task['elapsed_seconds'])}")
            st.write(task.get("phase", "Aguardando telemetria deste job"))
            st.markdown("**Progresso do job atual**")
            if task.get("monitor_warning"):
                st.warning(task["monitor_warning"])
            fixed = task.get("progress_basis") == "evaluations"
            if task.get("progress_fraction") is not None:
                if fixed:
                    st.write(f"Geração observada: {task['generation_current']}")
                else:
                    st.write(f"Geração {task['generation_current']} / {task['generation_total']}")
                st.progress(task["progress_fraction"], text=f"Progresso real da busca: {task['progress_fraction']:.1%}")
                if fixed:
                    st.caption("Percentual das avaliações efetivas da busca. A avaliação estrutural acontece em seguida.")
                else:
                    st.caption(f"Percentual por gerações do bolsão atual (índice {task.get('pocket_index')}). Preparação e pós-processamento não têm percentual medido.")
            else:
                st.caption("Progresso interno ainda não disponível; nenhum percentual é simulado.")
            evaluations = task.get("evaluations_current")
            if evaluations is not None:
                st.write(f"Avaliações {evaluations} / {task.get('evaluations_total', 'indisponível')}")
                if not fixed:
                    st.caption("Avaliações reais acumuladas das gerações do bolsão atual / orçamento planejado por bolsão. FULL pode ultrapassar o orçamento nominal.")
            if task.get("metrics_caught_up") is False:
                st.caption("Lendo o histórico de métricas; contadores disponíveis até a geração mostrada.")
            metric_time = timestamp(task.get("metrics_updated_at_utc"))
            if metric_time is not None:
                metric_age = max(0, time.time() - metric_time)
                st.caption(f"Última métrica do motor: há {int(metric_age)} segundos. O heartbeat indica atividade do worker; esta métrica indica avanço da busca.")
                if metric_age > 60:
                    st.warning("A busca está sem novas métricas há mais de 1 minuto; pode haver uma geração longa ou pós-processamento em andamento.")

    last = status["last_completed"]
    st.markdown("**Último concluído:**")
    if last:
        st.write(f"{last['complex_id']} · {str(last.get('algorithm', '')).upper()} · {str(last.get('protocol', '')).upper()} · tempo {_duration(task_duration(last))} · {_local_time(last.get('completed_at_utc'))}")
    else:
        st.write("Nenhum sucesso com horário de conclusão registrado.")
    plan = status["plan"]
    with st.expander("Decomposição dos jobs planejados", expanded=True):
        st.write(f"Complexos elegíveis: {plan['eligible_complexes']} · Complexos no plano: {plan['complexes']}")
        st.write(f"Algoritmos: {len(plan['algorithms'])} ({', '.join(plan['algorithms'])})")
        st.write(f"Protocolos: {len(plan['protocols'])} ({', '.join(p.upper() for p in plan['protocols'])})")
        st.write(f"Seeds: {len(plan['seeds'])} ({', '.join(str(s) for s in plan['seeds'])})")
        if plan["cartesian"]:
            arm_count = len(plan.get("conditions", [])) or len(plan["protocols"])
            st.write(f"**Jobs planejados: {plan['complexes']} × {len(plan['algorithms'])} × {arm_count} × {len(plan['seeds'])} = {plan['planned_jobs']}**")
            if plan.get("conditions"):
                st.write("Condições fatoriais: " + ", ".join(plan["conditions"]))
        else:
            st.warning(f"Plano não cartesiano: {plan['jobs_per_complex']} configurações por complexo; total real {plan['planned_jobs']}. O produto dos fatores seria {plan['cartesian_jobs']}.")
        if plan["folder_count_difference"]:
            st.warning(f"A pasta tem '{plan['folder_label_count']}' no nome, mas o manifesto contém {plan['complexes']} complexos: diferença de {plan['folder_count_difference']:+d}. O nome da pasta não define um limite; o pré-voo seleciona todos os complexos elegíveis.")
        st.caption(f"PDB IDs distintos: {plan['unique_pdb_ids']} · IDs de complexo duplicados: {plan['duplicate_complex_ids']} · Excluídos: {plan['excluded_complexes']}. Cada par de cadeias é um complexo independente.")
        st.caption(f"Entrada registrada no manifesto: {plan['input_root']}")
    st.subheader("Eventos recentes")
    st.code("\n".join(f"{_local_time(event.get('at_utc'), short=True)}  {event.get('message', '')}" for event in status["events"][-20:]) or "Nenhum evento registrado.", language=None)
    st.caption("Até 20 eventos na tela; horários no fuso do servidor. Fechar a página não interrompe o worker.")
    if status["failed_tasks"]:
        with st.expander("Falhas recentes (até 20)"):
            st.dataframe(pd.DataFrame([{k: row.get(k) for k in ("complex_id", "job_id", "completed_at_utc", "last_error")} for row in status["failed_tasks"]]), hide_index=True)
    if status.get("confidence_decisions"):
        with st.expander("Decisões do calibrador", expanded=True):
            st.dataframe(pd.DataFrame([
                {key: row.get(key) for key in (
                    "complex_id", "reduced_job_id", "action", "reason",
                    "probability_lower", "probability_upper", "threshold",
                )}
                for row in status["confidence_decisions"]
            ]), hide_index=True)


def render_batch_page() -> None:
    for key, value in {
        "ppi4dock_real_algorithms": ["ag", "pso", "hybrid"],
        "ppi4dock_real_generations": 100, "ppi4dock_real_population": 100,
        "ppi4dock_real_top_pockets": 32,
        "ppi4dock_real_seed_start": 1, "ppi4dock_real_seed_count": 1,
        "ppi4dock_real_timeout": 86400.0,
        "factorial_scale": 0.5, "factorial_order_seed": 20260925,
        "factorial_translation": 5.0, "factorial_rotation": 25.0,
        "factorial_population": 100,
    }.items():
        st.session_state.setdefault(key, value)
    st.header("Docking de proteínas em lote")
    st.write(
        "A pasta é lida por uma regra auditável: em `1a2y_AB`, "
        "`A_model_st.pdb` é o receptor, `B_model_st.pdb` é o parceiro móvel e "
        "`1a2y_AB_st.pdb` é somente a referência experimental."
    )
    st.info(
        "Informe os caminhos completos das pastas. O identificador, as cadeias e as sequências canônicas "
        "são extraídos automaticamente dos PDBs; nenhum arquivo é alterado na "
        "pasta de entrada."
    )

    path_columns = st.columns(2)
    with path_columns[0]:
        input_text = st.text_input(
            "Pasta de entrada PPI4DOCK",
            key="ppi4dock_input_dir",
            bind="query-params",
            placeholder=r"C:\caminho\PPI4DOCK_PDB",
        )
    with path_columns[1]:
        output_text = st.text_input(
            "Pasta de saída do lote",
            key="ppi4dock_output_dir",
            bind="query-params",
            placeholder=r"C:\caminho\resultados_docking",
        )
    _render_durable_status(output_text)
    _render_saved_factorial_protocol(output_text)
    validate_clicked = st.button(
        "Validar lote",
        type="primary",
        icon=":material/fact_check:",
    )

    if validate_clicked:
        if not input_text.strip():
            st.error("Informe a pasta de entrada.")
        else:
            try:
                with st.status("Validando nomes, cadeias, sequências e hashes..."):
                    report = discover_ppi4dock(input_text, validate_contents=True)
                st.session_state[_REPORT_KEY] = report
            except Exception as exc:  # noqa: BLE001 - validation boundary
                st.session_state.pop(_REPORT_KEY, None)
                st.error(f"Falha no pré-voo: {exc}")

    report = st.session_state.get(_REPORT_KEY)
    if isinstance(report, BatchDiscoveryReport):
        if str(report.input_root) != str(Path(input_text).expanduser().resolve(strict=False)):
            st.warning("A pasta de entrada mudou; valide novamente antes de executar.")
            st.session_state.pop(_REPORT_KEY, None)
            report = None
        else:
            _render_report(report)

    eligible_count = (
        len(report.eligible_targets)
        if isinstance(report, BatchDiscoveryReport)
        else 0
    )
    st.subheader("Experimentos para o artigo")
    st.write(
        "O motor DockingPP integrado executa somente os alvos "
        f"elegíveis na faixa de {DEFAULT_MOBILE_PARTNER_DOMAIN.label}. "
        "A referência experimental é usada apenas "
        "depois da busca, para DockQ e para os artefatos auditáveis."
    )
    st.caption(
        "Os limites de elegibilidade são registrados no protocolo. Os resultados se aplicam ao domínio e à tarefa de refinamento local avaliados."
    )
    if isinstance(report, BatchDiscoveryReport):
        st.info(
            f"Este plano real processará {eligible_count} de "
            f"{len(report.valid_targets)} complexos válidos."
        )

    study_labels = {
        "operational_cascade": "Custo e calibração — REDUCED 5.700 / FULL 7.600",
        "factorial_2x2": "Controle do orientador — amplitude e inicialização",
    }
    # Old browser sessions must not retain a removed production/legacy mode.
    if st.session_state.get("ppi4dock_real_protocol") not in study_labels:
        st.session_state["ppi4dock_real_protocol"] = "operational_cascade"
        st.session_state["ppi4dock_real_algorithms"] = ["ag", "pso", "hybrid"]
        st.session_state["ppi4dock_real_seed_start"] = 1
        st.session_state["ppi4dock_real_seed_count"] = 1
    real_protocol = st.selectbox(
        "Experimento do artigo", options=list(study_labels),
        format_func=study_labels.__getitem__, key="ppi4dock_real_protocol",
    )
    factorial_mode = real_protocol == "factorial_2x2"
    operational_mode = not factorial_mode
    real_docking_task = "local_refinement"
    confidence_artifact = None
    if operational_mode:
        from docking_confidence.application.operational import load_operational_config
        operational_base = load_operational_config()
        controls = operational_base.operational
        st.info(f"Coleta de pares para o artigo: REDUCED {controls.reduced_evaluation_budget} / FULL {controls.full_evaluation_budget} avaliações. "
                "Os dois são sempre executados para medir qualidade e custo e produzir os rótulos do calibrador. "
                "O Venn–Abers será avaliado depois, com separação por complexo; nenhum FULL será dispensado nesta coleta.")
    else:
        st.info("Controle do orientador: executar as quatro condições de amplitude e inicialização, "
                "com 7.600 avaliações em cada uma: reduzido/uniforme, reduzido/semeada, ampliado/uniforme e ampliado/semeada.")

    with st.form("ppi4dock_real_worker_form"):
        st.caption("Motor DockingPP integrado ao projeto. AG, PSO e Hybrid disponíveis no mesmo sistema.")
        st.caption("Refinamento local da pose fornecida. REDUCED permite movimentos menores; FULL permite movimentos maiores. "
                   "A referência experimental é usada somente na avaliação posterior de qualidade.")
        if factorial_mode:
            st.subheader("Controles independentes")
            factor_columns = st.columns(3)
            common_budget = FACTORIAL_EVALUATION_BUDGET
            factor_columns[0].number_input("Avaliações efetivas por execução", value=common_budget, disabled=True)
            reduced_scale = factor_columns[1].number_input("Escala do espaço reduzido", min_value=0.01, max_value=0.99, step=0.05, key="factorial_scale")
            order_seed = factor_columns[2].number_input("Seed da ordem de execução", min_value=0, step=1, key="factorial_order_seed")
            bound_columns = st.columns(2)
            max_translation = bound_columns[0].number_input("Limite de translação ampliado (± Å)", min_value=0.01, step=0.5, key="factorial_translation")
            max_rotation = bound_columns[1].number_input("Limite de rotação ampliado (± graus)", min_value=0.01, max_value=180.0, step=5.0, key="factorial_rotation")
            pose_source = st.text_area("Origem da pose fornecida e relação com a referência", key="factorial_pose_source", placeholder="Descreva como a pose foi obtida, se usa a referência e quais perturbações foram aplicadas.")
            analysis_role = st.selectbox("Finalidade da análise", options=["exploratory", "confirmatory"], format_func=lambda value: "Exploratória — dados já vistos ou histórico desconhecido" if value == "exploratory" else "Confirmatória — procedimento fixado antes de observar dados novos", key="factorial_analysis_role")
            st.caption("Uniforme: população amostrada na caixa de parâmetros de translação/rotação. Semeada: primeiro candidato é a pose fornecida; os demais são uniformes. A seed aleatória controla a repetição, não a semeadura da pose.")
            st.caption("A busca para ao completar 7.600 avaliações novas. O resultado principal é pose válida com DockQ ≥ 0,23; falhas contam no total planejado.")

        with st.expander("Configuração avançada do experimento", expanded=False):
            real_algorithms = st.multiselect(
                "Algoritmos",
                options=["ag", "pso", "hybrid"],
                key="ppi4dock_real_algorithms",
            )
            if operational_mode:
                real_generations = operational_base.generations
                real_population = operational_base.population
                real_top_pockets = 1
                st.caption(f"População {real_population}; um bolsão; espaço reduzido com escala {controls.reduced_search_scale}. "
                           "Parada antecipada permanece desativada até validação independente. "
                           "As gerações dependem do consumo de avaliações.")
            elif factorial_mode:
                real_generations = BaseSearchConfig().generations
                real_top_pockets = 1
                real_population = st.number_input("População", min_value=1, max_value=common_budget, step=1, key="factorial_population")
                st.caption("Refinamento local usa um bolsão. O número de gerações depende do consumo de avaliações.")
            run_columns = st.columns(3)
            real_seed_start = run_columns[0].number_input(
                "Seed inicial",
                min_value=0,
                step=1,
                key="ppi4dock_real_seed_start",
            )
            real_seed_count = run_columns[1].number_input(
                "Seeds",
                min_value=1,
                step=1,
                key="ppi4dock_real_seed_count",
                help=(
                    "Quantidade de sementes consecutivas a executar, a partir da Seed inicial. "
                    "Exemplo: Seed inicial = 1 e Seeds = 5 executa 1, 2, 3, 4, 5."
                ),
            )
            real_timeout = run_columns[2].number_input(
                "Limite por job (segundos)",
                min_value=60.0,
                step=60.0,
                key="ppi4dock_real_timeout",
            )
            real_resume = st.checkbox(
                "Retomar após um cancelamento anterior",
                value=False,
                key="ppi4dock_real_resume",
            )
        launch_real_clicked = st.form_submit_button(
            "Preparar protocolo fatorial" if factorial_mode else "Executar pares do artigo",
            type="primary",
            disabled=(
                report is None
                or bool(getattr(report, "has_errors", True))
                or eligible_count == 0
                or not real_algorithms
                or not output_text.strip()
            ),
            icon=":material/play_arrow:",
        )
    if launch_real_clicked:
        if factorial_mode:
            st.session_state.pop("factorial_prepared", None)
        try:
            launch_options = dict(
                input_dir=input_text,
                output_dir=output_text,
                algorithms=real_algorithms,
                protocol=real_protocol,
                docking_task=real_docking_task,
                generations=int(real_generations),
                population=int(real_population),
                top_pockets=int(real_top_pockets),
                seed_start=int(real_seed_start),
                seed_count=int(real_seed_count),
                timeout_seconds=float(real_timeout),
                resume_after_cancel=bool(real_resume),
                confidence_artifact=confidence_artifact,
                collect_operational_pairs=operational_mode,
            )
            if factorial_mode:
                if not pose_source.strip():
                    raise ValueError("Preencha o campo Origem da pose fornecida e relação com a referência.")
                controls = FactorialExperimentConfig(int(common_budget), float(reduced_scale), float(max_translation), float(max_rotation), pose_source, int(order_seed))
                jobs = build_execution_plan(algorithms=[Algorithm(value) for value in real_algorithms], protocol_mode=ProtocolMode.FACTORIAL,
                    base_config=BaseSearchConfig(int(real_generations), int(real_population), int(real_top_pockets), DockingTask.LOCAL_REFINEMENT, controls),
                    seeds=range(int(real_seed_start), int(real_seed_start) + int(real_seed_count)))
                from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
                prepared = prepare_experimental_protocol(input_text, output_text, jobs=jobs, engine=DockingPPLocalEngine(timeout_seconds=float(real_timeout)), report=report, analysis_role=analysis_role)
                launch_options.update(evaluation_budget=int(common_budget), reduced_search_scale=float(reduced_scale), max_translation=float(max_translation),
                    max_rotation_degrees=float(max_rotation), pose_source=pose_source, execution_order_seed=int(order_seed), experimental_protocol=prepared["path"])
                st.session_state["factorial_prepared"] = {"prepared": prepared, "launch_options": launch_options}
                st.success("Protocolo salvo. Confira o plano abaixo e inicie quando estiver pronto.")
            else:
                launched = launch_real_batch(**launch_options)
                _remember_worker_launch(launched, output_text)
                st.success(f"Worker real iniciado (PID {launched.pid}). Log: {launched.log_path}")
        except ExperimentalProtocolPlanConflictError as exc:
            st.error("Não foi possível preparar o protocolo fatorial: a configuração atual difere do protocolo já salvo nesta pasta.")
            field_labels = {
                "engine.descriptor.python_executable": "Executável do Python",
                "engine.descriptor.python_runtime_version": "Versão do Python",
                "engine.descriptor.source_manifest_sha256": "Versão do código do motor",
                "execution_limits.timeout_seconds": "Limite de tempo por execução",
                "input_root": "Pasta de entrada", "jobs": "Configurações planejadas",
            }
            st.dataframe(pd.DataFrame([
                {"Campo alterado": field_labels.get(item["field"], item["field"]),
                 "Protocolo salvo": item["saved"], "Configuração atual": item["current"]}
                for item in exc.differences
            ]), hide_index=True)
            st.info("Para manter o experimento original, carregue a configuração salva e use o mesmo ambiente. Para usar a configuração atual, informe uma pasta de saída nova e prepare novamente.")
        except Exception as exc:  # noqa: BLE001 - process launch boundary
            action = "preparar o protocolo fatorial" if factorial_mode else "iniciar o motor real"
            st.error(f"Não foi possível {action}: {exc}")

    prepared_state = st.session_state.get("factorial_prepared")
    if factorial_mode and prepared_state and prepared_state["prepared"]["protocol"]["analysis"].get("budget_policy") == FIXED_EVALUATION_BUDGET:
        saved_options = prepared_state["launch_options"]
        if saved_options["input_dir"] == input_text and saved_options["output_dir"] == output_text:
            frozen = prepared_state["prepared"]
            plan = frozen["protocol"]["plan"]
            st.subheader("Protocolo preparado")
            st.write(f"{len(plan['targets'])} complexos × {len(plan['jobs'])} configurações = {len(plan['targets']) * len(plan['jobs'])} jobs")
            st.dataframe(pd.DataFrame([{"Algoritmo": job["algorithm"], "Seed": job["seed"], "Amplitude": job["profile"]["search_extent"], "Inicialização": job["profile"]["initialization"], "Avaliações efetivas exigidas": job["profile"]["evaluation_budget"]} for job in plan["jobs"]]), hide_index=True)
            st.caption("A execução usa os valores salvos neste protocolo. Para mudar controles, prepare outro protocolo em uma pasta nova.")
            import json
            st.download_button("Baixar protocolo", data=json.dumps(frozen["protocol"], indent=2, ensure_ascii=False).encode("utf-8"), file_name="experimental_protocol.json", mime="application/json")
            resume_prepared = st.checkbox("Retomar após cancelamento (protocolo preparado)", value=saved_options["resume_after_cancel"], key="factorial_resume_prepared")
            if st.button("Iniciar/retomar as quatro condições", key="factorial_launch", type="primary"):
                try:
                    launched = launch_real_batch(**{**saved_options, "resume_after_cancel": resume_prepared})
                    _remember_worker_launch(launched, output_text)
                    st.success(f"Worker fatorial iniciado (PID {launched.pid}). Log: {launched.log_path}")
                except Exception as exc:
                    st.error(f"Falha ao iniciar: {exc}")

    if factorial_mode and output_text.strip() and (Path(output_text) / "batch_manifest.json").is_file():
        if st.button("Analisar resultados fatoriais salvos", key="factorial_report"):
            try:
                from docking_confidence.application.factorial_reporting import analyze_factorial_batch, write_factorial_report
                with st.status("Analisando quartetos e reamostrando complexos..."):
                    analysis = analyze_factorial_batch(output_text)
                    paths = write_factorial_report(analysis, Path(output_text) / "factorial_analysis", plots=True)
                st.session_state["factorial_report_result"] = {"output": output_text, "paths": paths, "analysis": analysis}
            except Exception as exc:
                st.error(f"Falha na análise: {exc}")
        result = st.session_state.get("factorial_report_result")
        if result and result["output"] == output_text:
            st.write("Relatórios fatoriais:", result["paths"])
            import json
            saved_report = json.loads(Path(result["paths"]["json"]).read_text(encoding="utf-8"))
            st.caption(saved_report["interpretation"])
            st.write("Finalidade declarada:", saved_report["analysis_role"])
            if saved_report.get("primary_outcome") == FACTORIAL_PRIMARY_OUTCOME:
                st.subheader("Desfecho principal: pose válida com DockQ ≥ 0,23")
                st.caption("As falhas entram no total planejado. Se houver pendências, a proporção é provisória.")
                rows = []
                for algorithm, entry in saved_report["by_algorithm"].items():
                    for condition, info in entry["conditions"].items():
                        primary = info["primary_outcome"]
                        rows.append({"Algoritmo": algorithm, "Condição": condition, "Sucessos": primary["successes"],
                                     "Insucessos": primary["failures_or_below_threshold"], "Pendências": primary["unresolved"],
                                     "Planejados": primary["denominator"], "Proporção": f"{primary['proportion_of_planned']:.1%}"})
                st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.dataframe(pd.read_csv(result["paths"]["effects"]), hide_index=True)
            with st.expander("Falhas e quartetos incompletos"):
                st.json(saved_report["diagnostics"])
            for key, path in result["paths"].items():
                if key.startswith("plot_"):
                    st.image(path)
            st.download_button("Baixar relatório fatorial", data=Path(result["paths"]["markdown"]).read_bytes(), file_name="factorial_analysis.md")

    action_columns = st.columns(2)
    if action_columns[0].button("Atualizar estado", icon=":material/refresh:"):
        st.rerun()
    if action_columns[1].button(
        "Solicitar cancelamento",
        disabled=not output_text.strip(),
        icon=":material/stop_circle:",
    ):
        marker = request_batch_cancel(output_text)
        st.warning(f"Cancelamento solicitado em {marker}; vale entre jobs.")
