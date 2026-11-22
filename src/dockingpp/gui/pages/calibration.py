"""Confidence calibrator training page."""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import streamlit as st

from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.state import AppState, PROJECT_ROOT, StateKeys
from dockingpp.pipeline.confianca.artifact_calibrator import (
    load_confidence_calibrator,
)


TRAINING_GUIDANCE = [
    {
        "Total disponivel": "26 proteinas",
        "Calibracao V5 grouped OOF": "21",
        "Teste externo reservado": "5",
    },
    {
        "Total disponivel": "30 proteinas",
        "Calibracao V5 grouped OOF": "25",
        "Teste externo reservado": "5",
    },
    {
        "Total disponivel": "40 proteinas",
        "Calibracao V5 grouped OOF": "30 a 32",
        "Teste externo reservado": "8 a 10",
    },
    {
        "Total disponivel": "50 proteinas",
        "Calibracao V5 grouped OOF": "35 a 40",
        "Teste externo reservado": "10 a 15",
    },
]

MINIMUM_COHORT_NOTICE = (
    "21 complexos independentes sao o minimo congelado da V5, nao uma "
    "garantia de aprovacao. O candidato ainda precisa superar o quality gate."
)
GROUPED_OOF_NOTICE = (
    "A validacao interna V5 usa grouped out-of-fold (OOF): todas as observacoes "
    "de um complexo ficam juntas e cada previsao OOF e produzida sem treinar "
    "naquele complexo. Nao e um unico holdout escolhido por seed."
)
OPERATING_THRESHOLD_NOTICE = (
    "A V5 aprende a probabilidade de preservacao e autoriza apenas quando o "
    "limite superior de risco, estimado por complexo, respeita o alvo congelado. "
    "Sem suporte estatistico, o artefato permanece somente observacional."
)
QUALITY_GATE_FAILURE_MARKER = "calibration quality gate failed"
MISSING_PAIRED_CONTROLS_MARKER = (
    "preservation calibration requires one independent canonical full control"
)
MINIMUM_VALIDATION_COMPLEXES_PER_FOLD = 3
_REJECTED_ARTIFACT_TOKEN = re.compile(
    r"rejected_artifact(?:_path)?\s*=\s*[\"']?"
    r"(?P<path>.+?\.json)[\"']?(?=[\s;,]|$)",
    flags=re.IGNORECASE,
)


def calibration_diagnostics_from_artifact(
    path: str | Path | None,
) -> dict[str, Any] | None:
    """Read the compact V5 nested-validation evidence shown by the GUI."""

    if path is None or not str(path).strip():
        return None
    artifact_path = Path(str(path)).expanduser()
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    if not isinstance(payload, Mapping) or str(payload.get("schema_version")) != "5":
        return None

    evidence = payload.get("calibration_evidence")
    provenance = payload.get("training_provenance")
    if not isinstance(evidence, Mapping) or not isinstance(provenance, Mapping):
        return None
    quality_gate = evidence.get("calibration_quality_gate")
    operating = evidence.get("operating_point_assessment")
    if not isinstance(quality_gate, Mapping) or not isinstance(operating, Mapping):
        return None
    validation = quality_gate.get("model_metrics")
    baseline = quality_gate.get("baseline_metrics")
    if not isinstance(validation, Mapping) or not isinstance(baseline, Mapping):
        return None

    def finite(mapping: Mapping[str, Any], key: str) -> float | None:
        try:
            value = float(mapping.get(key))
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    brier = finite(validation, "brier_score")
    baseline_brier = finite(baseline, "brier_score")
    log_loss = finite(validation, "log_loss")
    baseline_log_loss = finite(baseline, "log_loss")
    threshold = finite(operating, "minimum_probability")
    if threshold is None:
        policy = payload.get("decision_policy")
        threshold = (
            finite(policy, "minimum_probability")
            if isinstance(policy, Mapping)
            else None
        )
    operating_display = {
        "threshold": threshold,
        "above_threshold_weight_fraction": finite(operating, "row_coverage"),
        "above_threshold_complex_count": operating.get(
            "accepted_complex_count"
        ),
        "observed_success_fraction_above_threshold": (
            1.0 - float(operating["observed_risk"])
            if operating.get("observed_risk") is not None
            else None
        ),
        "operating_threshold_reached": operating.get("supported") is True,
    }
    return {
        "artifact_path": str(artifact_path),
        "schema_version": 5,
        "calibrated": payload.get("calibrated") is True,
        "decision_capable": payload.get("decision_capable") is True,
        "target_definition": payload.get("target_definition"),
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_delta": (
            brier - baseline_brier
            if brier is not None and baseline_brier is not None
            else None
        ),
        "log_loss": log_loss,
        "baseline_log_loss": baseline_log_loss,
        "log_loss_delta": (
            log_loss - baseline_log_loss
            if log_loss is not None and baseline_log_loss is not None
            else None
        ),
        "oof_complex_count": provenance.get("n_complexes"),
        "oof_unique_sample_count": provenance.get("n_rows"),
        "quality_gate_passed": quality_gate.get("passed") is True,
        "quality_gate_reasons": [
            str(value) for value in quality_gate.get("reasons", [])
        ],
        "operating_threshold_metrics": operating_display,
        "operating_point_assessment": dict(operating),
        "risk_control": None,
    }

def _format_metric(value: Any, *, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "indisponivel"
    return f"{number:.{digits}f}" if math.isfinite(number) else "indisponivel"


def resolve_project_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve(strict=False)


def build_training_command(
    *,
    run_root: Path,
    output_path: Path,
    dockq_threshold: float,
    cv_folds: int,
    seed: int,
    min_complexes: int,
    required_top_pockets: int = 32,
    full_run_root: Path | None = None,
    max_dockq_loss: float = 0.05,
    v5_risk_target: float = 0.20,
    risk_confidence_level: float = 0.95,
) -> list[str]:
    script_path = PROJECT_ROOT / "scripts" / "train_confidence_calibrator.py"
    command = [
        sys.executable,
        str(script_path),
        str(run_root),
        str(output_path),
        "--target",
        "preservation",
        "--calibrator-version",
        "v5",
        "--full-run-root",
        str(full_run_root or run_root),
        "--dockq-threshold",
        str(float(dockq_threshold)),
        "--max-dockq-loss",
        str(float(max_dockq_loss)),
        "--v5-risk-target",
        str(float(v5_risk_target)),
        "--risk-confidence-level",
        str(float(risk_confidence_level)),
        "--cv-folds",
        str(int(cv_folds)),
        "--seed",
        str(int(seed)),
        "--required-top-pockets",
        str(int(required_top_pockets)),
        "--min-complexes",
        str(int(min_complexes)),
    ]
    return command


def validate_training_request(
    *,
    run_root: Path,
    dockq_threshold: float,
    cv_folds: int,
    min_complexes: int,
    full_run_root: Path | None = None,
    max_dockq_loss: float = 0.05,
    v5_risk_target: float = 0.20,
    risk_confidence_level: float = 0.95,
) -> list[str]:
    errors: list[str] = []
    if not run_root.exists() or not run_root.is_dir():
        errors.append(f"Pasta de runs nao encontrada: {run_root}")
    if full_run_root is not None and (
        not full_run_root.exists() or not full_run_root.is_dir()
    ):
        errors.append(f"Pasta de controles FULL nao encontrada: {full_run_root}")
    if not 0.0 < float(dockq_threshold) <= 1.0:
        errors.append("dockq_threshold precisa estar em (0, 1].")
    if not 0.0 <= float(max_dockq_loss) <= 1.0:
        errors.append("max_dockq_loss precisa estar em [0, 1].")
    if not 0.0 < float(v5_risk_target) < 1.0:
        errors.append("v5_risk_target precisa estar em (0, 1).")
    if not 0.5 < float(risk_confidence_level) < 1.0:
        errors.append("risk_confidence_level precisa estar em (0.5, 1).")
    if int(cv_folds) < 2:
        errors.append("cv_folds precisa ser pelo menos 2 para produzir OOF.")
    if int(cv_folds) > int(min_complexes):
        errors.append("cv_folds nao pode exceder min_complexes.")
    if (
        int(cv_folds) >= 2
        and int(min_complexes) >= 21
        and int(min_complexes) // int(cv_folds)
        < MINIMUM_VALIDATION_COMPLEXES_PER_FOLD
    ):
        errors.append(
            "A configuracao minima precisa deixar pelo menos 3 complexos em "
            "cada fold OOF; reduza cv_folds ou aumente min_complexes."
        )
    if int(min_complexes) < 21:
        errors.append(
            "min_complexes precisa ser pelo menos 21 no protocolo V5. Esse "
            "minimo nao garante aprovacao no quality gate."
        )
    return errors


def run_training_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "command": command,
    }


def _combined_training_output(result: Mapping[str, Any]) -> str:
    return "\n".join(
        str(result.get(field) or "") for field in ("stdout", "stderr")
    )


def is_quality_gate_rejection(result: Mapping[str, Any]) -> bool:
    """Distinguish a scientifically rejected candidate from an execution error."""

    try:
        failed = int(result.get("returncode", 0)) != 0
    except (TypeError, ValueError):
        failed = True
    return failed and QUALITY_GATE_FAILURE_MARKER in _combined_training_output(
        result
    ).lower()


def is_missing_paired_controls_error(result: Mapping[str, Any]) -> bool:
    """Identify a data-design failure rather than a model/exec failure."""

    try:
        failed = int(result.get("returncode", 0)) != 0
    except (TypeError, ValueError):
        failed = True
    return failed and MISSING_PAIRED_CONTROLS_MARKER in _combined_training_output(
        result
    ).lower()


def rejected_artifact_path(result: Mapping[str, Any]) -> str | None:
    """Recover the optional audit-artifact path emitted by the trainer."""

    output = _combined_training_output(result)
    for line in output.splitlines():
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, Mapping):
            value = str(
                payload.get("rejected_artifact")
                or payload.get("rejected_artifact_path")
                or ""
            ).strip()
            if value:
                return value
    match = _REJECTED_ARTIFACT_TOKEN.search(output)
    if match is None:
        return None
    return match.group("path").strip().strip("\"'") or None


def is_approved_calibrator_artifact(path: str | Path) -> bool:
    """Return true only when the runtime loader accepts a calibrated artifact."""

    try:
        calibrator = load_confidence_calibrator(Path(path))
    except (OSError, ValueError, TypeError):
        return False
    return (
        calibrator.calibrated is True
        and calibrator.decision_capable is True
    )


def quality_gate_rejection_message(
    result: Mapping[str, Any],
    *,
    official_output_path: str | Path,
    previous_calibrator_present: bool,
) -> str:
    """Build an actionable, non-generic explanation for a rejected candidate."""

    lines = [
        "O candidato foi treinado, mas reprovou no quality gate grouped OOF e "
        "nao foi ativado. Isso indica desempenho fora da amostra insuficiente, "
        "nao uma falha de execucao."
    ]
    if previous_calibrator_present:
        lines.append(
            "O calibrador anterior foi preservado no caminho oficial: "
            f"{official_output_path}."
        )
    else:
        lines.append(
            "Nenhum calibrador foi ativado e o caminho oficial permaneceu sem "
            f"um novo artefato aprovado: {official_output_path}."
        )
    rejected_path = rejected_artifact_path(result)
    if rejected_path:
        lines.append(f"Evidencia do candidato rejeitado: {rejected_path}.")
        diagnostics = calibration_diagnostics_from_artifact(rejected_path)
        if diagnostics is not None:
            lines.append(
                "Comparacao OOF: Brier "
                f"{_format_metric(diagnostics.get('brier_score'))} versus "
                f"{_format_metric(diagnostics.get('baseline_brier_score'))} "
                "no baseline; log-loss "
                f"{_format_metric(diagnostics.get('log_loss'))} versus "
                f"{_format_metric(diagnostics.get('baseline_log_loss'))}."
            )
    return " ".join(lines)


def render_calibration_diagnostics(path: str | Path | None) -> None:
    """Render a compact panel separating calibration from decision utility."""

    diagnostics = calibration_diagnostics_from_artifact(path)
    if diagnostics is None:
        return
    st.subheader("Diagnostico grouped OOF")
    if (
        diagnostics.get("calibrated") is True
        and diagnostics.get("decision_capable") is not True
    ):
        st.warning(
            "O modelo superou o gate probabilistico, mas ficou marcado como "
            "apenas observacional: o ponto operacional nao tem suporte OOF "
            "suficiente para aceitar automaticamente a busca reduzida."
        )
    columns = st.columns(4)
    metric_rows = (
        (
            "Brier OOF",
            diagnostics.get("brier_score"),
            diagnostics.get("brier_delta"),
        ),
        (
            "Brier baseline",
            diagnostics.get("baseline_brier_score"),
            None,
        ),
        (
            "Log-loss OOF",
            diagnostics.get("log_loss"),
            diagnostics.get("log_loss_delta"),
        ),
        (
            "Complexos independentes",
            diagnostics.get("oof_complex_count"),
            None,
        ),
    )
    for column, (label, value, delta) in zip(columns, metric_rows):
        with column:
            display_value = (
                str(int(value))
                if label == "Complexos independentes" and value is not None
                else _format_metric(value)
            )
            display_delta = (
                f"{float(delta):+.4f} vs baseline" if delta is not None else None
            )
            st.metric(label, display_value, display_delta)

    operating = diagnostics.get("operating_threshold_metrics")
    if not isinstance(operating, Mapping):
        st.caption(
            "O artefato V5 nao contem a avaliacao aninhada do ponto operacional; "
            "retreine pelo protocolo V5 atual."
        )
        return
    threshold = _format_metric(operating.get("threshold"), digits=2)
    coverage = _format_metric(
        100.0 * float(operating.get("above_threshold_weight_fraction") or 0.0),
        digits=2,
    )
    selected_complexes = int(operating.get("above_threshold_complex_count") or 0)
    observed = operating.get("observed_success_fraction_above_threshold")
    if operating.get("operating_threshold_reached") is not True:
        st.warning(
            f"O limiar operacional {threshold} nao foi alcancado nas previsoes "
            "OOF. Mesmo um modelo calibrado nesse cenario faria fallback em "
            "100% dos casos."
        )
    else:
        st.info(
            f"No limiar {threshold}, a cobertura OOF foi {coverage}% em "
            f"{selected_complexes} complexos; fracao observada de sucesso: "
            f"{_format_metric(observed, digits=3)}."
        )
        st.caption(
            "A V5 combina probabilidade estatistica, suporte de dominio e "
            "concordancia com o grafo. Antes da confirmacao externa, produz "
            "somente telemetria e nao comanda decisao automatica."
        )
    risk_control = diagnostics.get("risk_control")
    if isinstance(risk_control, Mapping):
        selected_risk = risk_control.get("selected_operating_point")
        if risk_control.get("supported") is True and isinstance(
            selected_risk, Mapping
        ):
            st.success(
                "Controle de risco OOF aprovado: limite superior unilateral "
                f"{_format_metric(selected_risk.get('risk_upper_confidence_bound'), digits=3)} "
                "para risco maximo "
                f"{_format_metric(risk_control.get('max_authorized_risk'), digits=3)}; "
                "cobertura de complexos "
                f"{_format_metric(100.0 * float(selected_risk.get('authorized_complex_coverage') or 0.0), digits=1)}%."
            )
        else:
            st.warning(
                "Nenhum limiar OOF controlou o risco por complexo no nivel de "
                "confianca solicitado. O calibrador pode produzir telemetria, "
                "mas nao autoriza REDUCED automaticamente."
            )


class CalibrationPage(BasePage):
    id = "Calibracao"
    title = "Calibracao"

    def render(self, state: AppState) -> None:
        st.header("Calibracao do calibrador de confianca")
        st.info(
            "Este calibrador nao e LLM nem rede neural. Ele e uma regressao "
            "arquitetura auditavel: regressao logistica mais grafo de evidencias "
            "por complexo. O JSON grava coeficientes, escalas, vizinhos, hashes "
            "dos pares FULL/REDUCED e metricas de validacao. "
            "O alvo e a probabilidade de REDUCED preservar qualidade em relacao "
            "ao controle FULL, nao apenas a probabilidade de obter uma pose boa."
        )

        st.subheader("Como usar")
        st.markdown(
            """
1. Para cada seed/motor/complexo, rode REDUCED e um controle FULL independente com o mesmo protocolo.
2. O FULL deve ser um braco planejado; `fallback_full` seletivo nao vale como controle de treino.
3. Reserve complexos PDB diferentes para o teste externo e gere os mesmos pares neles.
4. Treine a V5; ela permanece em observe ate passar a confirmacao externa intocada, sem refit.
5. Depois da aprovacao, informe `confidence_calibrator.json` em `confidence_calibrator_path` e use
   `assist_fallback` ou `assist_all`.
"""
        )
        st.info(MINIMUM_COHORT_NOTICE)
        st.info(GROUPED_OOF_NOTICE)
        st.info(OPERATING_THRESHOLD_NOTICE)

        st.subheader("Separacao recomendada")
        st.table(TRAINING_GUIDANCE)
        st.warning(
            "Nao misture o mesmo PDB em treino e teste final. Seeds diferentes "
            "do mesmo PDB nao contam como proteinas independentes. Os folds OOF "
            "mantem todos os pares do mesmo complexo juntos. O treinador usa "
            "todos os complexos elegiveis da pasta; para tentar com exatamente "
            "15, mantenha somente essa coorte na pasta selecionada."
        )

        st.subheader("Treinar artefato")
        run_root_text = st.text_input(
            "Pasta com runs REDUCED para calibracao",
            key=StateKeys.CALIBRATION_RUN_ROOT,
            help=(
                "Use runs gerados em modo reduzido, preferencialmente com "
                "confidence_policy=observe, e somente a coorte destinada a "
                "calibracao."
            ),
        )
        full_run_root_text = st.text_input(
            "Pasta com controles FULL pareados",
            key=StateKeys.CALIBRATION_FULL_RUN_ROOT,
            help=(
                "Pode ser a mesma raiz quando os resultados possuem diretorios "
                "irmaos full/ e reduced/. Cada REDUCED precisa de FULL com mesmo "
                "complexo, entradas, rodada, motor, seed, populacao e geracoes."
            ),
        )
        output_text = st.text_input(
            "Arquivo de saida do calibrador",
            key=StateKeys.CALIBRATION_OUTPUT_PATH,
            help=(
                "Exemplo: artifacts/confidence_calibrator.json. Um candidato "
                "reprovado nao sobrescreve o calibrador aprovado anterior."
            ),
        )
        col_left, col_right = st.columns(2)
        with col_left:
            dockq_threshold = st.number_input(
                "Qmin: DockQ minimo do REDUCED",
                min_value=0.01,
                max_value=1.0,
                step=0.01,
                key=StateKeys.CALIBRATION_DOCKQ_THRESHOLD,
            )
            max_dockq_loss = st.number_input(
                "Delta: perda maxima de DockQ versus FULL",
                min_value=0.0,
                max_value=1.0,
                step=0.01,
                key=StateKeys.CALIBRATION_MAX_DOCKQ_LOSS,
            )
            cv_folds = st.number_input(
                "Numero de folds grouped OOF",
                min_value=2,
                step=1,
                key=StateKeys.CALIBRATION_CV_FOLDS,
                help=(
                    "Cada complexo fica fora do treino uma vez por repeticao. "
                    "Com 21 complexos, 5 folds usam cerca de 4 por fold."
                ),
            )
        with col_right:
            v5_risk_target = st.number_input(
                "Alvo de risco da V5",
                min_value=0.01,
                max_value=0.50,
                step=0.01,
                key=StateKeys.CALIBRATION_V5_RISK_TARGET,
                help=(
                    "O protocolo congelado usa 0,20. A V5 valida o limite superior "
                    "do risco por complexo sem escolher o melhor ponto depois."
                ),
            )
            risk_confidence_level = st.number_input(
                "Nivel de confianca do limite de risco",
                min_value=0.80,
                max_value=0.99,
                step=0.01,
                key=StateKeys.CALIBRATION_RISK_CONFIDENCE_LEVEL,
            )
        seed = st.number_input(
            "Seed reproduzivel da construcao dos folds",
            min_value=1,
            step=1,
            key=StateKeys.CALIBRATION_SEED,
        )
        min_complexes = st.number_input(
            "Minimo de complexos para permitir treinamento",
            min_value=21,
            step=1,
            key=StateKeys.CALIBRATION_MIN_COMPLEXES,
            help=(
                "21 permite apenas tentar o treino V5. A aprovacao depende das "
                "metricas grouped OOF; prefira uma coorte maior."
            ),
        )

        run_root = resolve_project_path(str(run_root_text or ""))
        full_run_root = resolve_project_path(str(full_run_root_text or ""))
        output_path = resolve_project_path(str(output_text or ""))
        command = build_training_command(
            run_root=run_root,
            full_run_root=full_run_root,
            output_path=output_path,
            dockq_threshold=float(dockq_threshold),
            max_dockq_loss=float(max_dockq_loss),
            v5_risk_target=float(v5_risk_target),
            risk_confidence_level=float(risk_confidence_level),
            cv_folds=int(cv_folds),
            seed=int(seed),
            min_complexes=int(min_complexes),
        )
        st.caption("Comando auditavel que sera executado:")
        st.code(" ".join(command), language="powershell")

        if st.button("Treinar candidato a calibrador"):
            errors = validate_training_request(
                run_root=run_root,
                full_run_root=full_run_root,
                dockq_threshold=float(dockq_threshold),
                max_dockq_loss=float(max_dockq_loss),
                v5_risk_target=float(v5_risk_target),
                risk_confidence_level=float(risk_confidence_level),
                cv_folds=int(cv_folds),
                min_complexes=int(min_complexes),
            )
            if errors:
                for error in errors:
                    st.error(error)
                return
            previous_calibrator_present = is_approved_calibrator_artifact(
                output_path
            )
            result = run_training_command(command)
            result["official_output_path"] = str(output_path)
            result["previous_calibrator_present"] = previous_calibrator_present
            st.session_state[StateKeys.CALIBRATION_LAST_RESULT] = result

        last_result = st.session_state.get(StateKeys.CALIBRATION_LAST_RESULT)
        if isinstance(last_result, dict):
            st.subheader("Ultima saida")
            result_output_path = str(
                last_result.get("official_output_path") or output_path
            )
            if int(last_result.get("returncode", 1)) == 0:
                approved_diagnostics = calibration_diagnostics_from_artifact(
                    result_output_path
                )
                if (
                    approved_diagnostics is not None
                    and approved_diagnostics.get("decision_capable") is not True
                ):
                    st.warning(
                        "Candidato calibrado para observacao, mas sem suporte "
                        "OOF suficiente no limiar operacional. O artefato foi "
                        f"atualizado em {result_output_path}, porem nao habilita "
                        "aceitacao reduzida automatica."
                    )
                else:
                    st.success(
                        "Candidato aprovado no quality gate grouped OOF e no "
                        "ponto operacional. "
                        f"Calibrador atualizado: {result_output_path}"
                    )
                    st.caption(
                        "Agora use este caminho em confidence_calibrator_path "
                        "antes de rodar assist_fallback ou assist_all."
                    )
            elif is_missing_paired_controls_error(last_result):
                st.error(
                    "Treinamento bloqueado corretamente: faltam controles FULL "
                    "independentes pareados para os runs REDUCED. Resultados "
                    "fallback_full seletivos nao podem preencher essa lacuna. "
                    "Nenhum modelo foi treinado e o calibrador oficial foi "
                    "preservado. Veja abaixo quantos pares estao ausentes."
                )
            elif is_quality_gate_rejection(last_result):
                st.warning(
                    quality_gate_rejection_message(
                        last_result,
                        official_output_path=result_output_path,
                        previous_calibrator_present=bool(
                            last_result.get("previous_calibrator_present", False)
                        ),
                    )
                )
            else:
                st.error(
                    "O treinamento nao foi concluido por um erro de execucao. "
                    "Leia a saida tecnica abaixo."
                )
            diagnostic_artifact_path = (
                result_output_path
                if int(last_result.get("returncode", 1)) == 0
                else rejected_artifact_path(last_result)
            )
            render_calibration_diagnostics(diagnostic_artifact_path)
            if last_result.get("stdout"):
                st.code(str(last_result.get("stdout")), language="json")
            if last_result.get("stderr"):
                st.code(str(last_result.get("stderr")), language="text")
