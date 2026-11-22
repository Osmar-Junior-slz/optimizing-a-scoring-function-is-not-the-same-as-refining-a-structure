"""Finalize the 50-complex development study without training or installing a gate."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.application.batch import _validate_manifest
from docking_confidence.application.experimental_protocol import bind_experimental_protocol
from docking_confidence.application.training_dataset import _read_receipt, _read_failure


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def verify_runs(root):
    counts = Counter()
    conditions = defaultdict(Counter)
    errors = Counter()
    targets = {}
    fingerprints = defaultdict(list)
    bindings = []
    engine = None
    checked_inputs = set()
    for batch in sorted(root.glob("shard_*")):
        if not batch.is_dir():
            continue
        manifest = read(batch / "batch_manifest.json")
        _validate_manifest(manifest)
        status = read(batch / "batch_status.json")
        if status["state"] not in {"COMPLETED", "COMPLETED_WITH_ERRORS"}:
            raise ValueError(f"unfinished batch: {batch}")
        counts.update(status["counts"])
        plan = manifest["plan"]
        if engine is None:
            engine = plan["engine"]
        if engine != plan["engine"]:
            raise ValueError("different engines across shards")
        binding = plan["experimental_protocol"]
        unbound = {key: value for key, value in plan.items() if key != "experimental_protocol"}
        if bind_experimental_protocol(binding["path"], unbound) != binding:
            raise ValueError("factorial protocol changed")
        bindings.append(binding)
        for target in plan["targets"]:
            cid = target["complex_id"]
            if cid in targets:
                raise ValueError("a target appears in multiple shards")
            targets[cid] = target
            fingerprints[target["complex_fingerprint_sha256"]].append(cid)
            for role in ("receptor", "ligand", "reference"):
                path = Path(plan["input_root"]) / target[role + "_path"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != target[role + "_sha256"]:
                    raise ValueError(f"input changed: {path}")
                checked_inputs.add(str(path.resolve()))
            for job in plan["jobs"]:
                condition = job["profile"]["search_extent"] + "_" + job["profile"]["initialization"]
                bucket = conditions[condition]
                bucket["planned"] += 1
                directory = batch / "targets" / cid / "jobs" / job["job_id"]
                receipt = _read_receipt(directory, manifest, target, job)
                if receipt is not None:
                    result = receipt["engine_result"]
                    if result["actual_eval_count"] != 7600 or result["stop_reason"] != "objective_evaluation_ceiling":
                        raise ValueError(f"effective budget mismatch: {directory}")
                    bucket["succeeded"] += 1
                    bucket["structural_success"] += int(result["dockq"] >= .23)
                else:
                    failure = _read_failure(directory, manifest, target, job)
                    if failure is None:
                        raise ValueError(f"unaccounted job: {directory}")
                    bucket["failed"] += 1
                    errors[failure["reason"] + ": " + str(failure["failure_reason"])] += 1
    if len(bindings) != 8 or len(targets) != 50 or len(fingerprints) != 49 or counts["planned"] != 600:
        raise ValueError("the pilot does not match the frozen 50-target plan")
    from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
    if DockingPPLocalEngine().audit_descriptor() != engine["descriptor"]:
        raise ValueError("current docking engine differs from frozen pilot engine")
    return {
        "counts": dict(counts), "conditions": {key: dict(value) for key, value in conditions.items()},
        "target_count": len(targets), "molecular_fingerprints": len(fingerprints),
        "aliases": [sorted(ids) for ids in fingerprints.values() if len(ids) > 1],
        "verified_input_files": len(checked_inputs), "failure_categories": dict(errors),
        "successful_receipts_verified_at_7600_effective_evaluations": True,
        "current_engine_matches_frozen_engine": True,
        "experimental_protocol_bindings": bindings,
    }


def policy_detail(directory, suffix="primary"):
    old = pd.DataFrame(read(directory / f"{suffix}_legacy.json")["predictions"])
    new = pd.DataFrame(read(directory / f"{suffix}_context.json")["predictions"])
    if not old[["complex_id", "algorithm", "seed"]].equals(new[["complex_id", "algorithm", "seed"]]):
        raise ValueError("paired predictions disagree")
    group_column = "pilot_source_cluster" if suffix == "source_blocked" else "complex_fingerprint_sha256"
    y = old.preserved.to_numpy(dtype=int)
    groups = old[group_column].to_numpy(dtype=str)
    keys = np.unique(groups)
    members = [np.flatnonzero(groups == key) for key in keys]
    result = {}
    for name, predictions in (("legacy", old), ("context", new)):
        column = name + "_balanced_C1__"
        probability = predictions[column + "after"].to_numpy(dtype=float)
        lower = predictions[column + "lower"].to_numpy(dtype=float)
        eligible = predictions[column + "in_domain"].to_numpy(dtype=bool)
        available = np.isfinite(probability)
        accepted = available & eligible & (lower >= .9)
        count = int(accepted.sum())
        wrong = accepted & (y == 0)
        samples = []
        rng = np.random.default_rng(20261115)
        for _ in range(2000):
            idx = np.concatenate([members[i] for i in rng.integers(0, len(keys), len(keys))])
            denominator = int(accepted[idx].sum())
            if denominator:
                samples.append(float(wrong[idx].sum() / denominator))
        result[name] = {
            "total_valid_pairs": len(y), "calibration_available_pairs": int(available.sum()),
            "calibration_unavailable_violations": int(((~available) & (y == 0)).sum()),
            "in_domain_and_calibration_available": int((eligible & available).sum()),
            "accepted": count, "accepted_molecular_groups": len(np.unique(groups[accepted])),
            "false_accepted": int(wrong.sum()),
            "coverage_of_all_valid_pairs": count / len(y),
            "risk": float(wrong.sum() / count) if count else None,
            "conditional_cluster_percentile_risk_ci95": np.quantile(samples, [.025, .975]).tolist() if samples and wrong.any() else None,
            "risk_interval_note": "Exploratory, conditional on fitted fold models; not reported when no violations are observed because resampling cannot establish an upper safety bound then.",
            "false_acceptance_ids": predictions.loc[wrong, "complex_id"].tolist(),
            "accept_all_valid_risk": float((y == 0).mean()),
        }
    return result


def assessment(directory):
    result = {"primary": read(directory / "primary_summary.json"),
              "primary_full_denominator_policies": policy_detail(directory),
              "source_cluster_audit": read(directory / "source_cluster_audit.json")}
    if (directory / "source_blocked_summary.json").exists():
        result["source_blocked"] = read(directory / "source_blocked_summary.json")
        result["source_blocked_full_denominator_policies"] = policy_detail(directory, "source_blocked")
    else:
        result["source_blocked_unavailable"] = read(directory / "source_blocked_unavailable.json")
    return result


def number(value):
    return "indisponível" if value is None else f"{value:.3f}".replace(".", ",")


def percentage(value):
    return "indisponível" if value is None else f"{100 * value:.1f}%".replace(".", ",")


def report_text(data):
    rows = []
    for key, title in (("legacy", "Histórico: orçamento/inicialização diferentes"),
                       ("fixed_uniform", "Novo piloto: uniforme (principal)"),
                       ("fixed_seeded", "Novo piloto: semeada (sensibilidade)")):
        s = data[key]["primary"]
        rows.append(f"| {title} | {s['common_calibrated_pairs']}/{s['total_pairs']} | {number(s['legacy']['roc_auc'])} → {number(s['context']['roc_auc'])} | {number(s['legacy']['brier_score'])} → {number(s['context']['brier_score'])} | {number(s['prevalence_reference']['brier_score'])} |")
    primary = data["fixed_uniform"]["primary"]
    old, new = primary["legacy"], primary["context"]
    available = all(value is not None for value in (
        new["roc_auc"], old["roc_auc"], new["brier_score"], old["brier_score"]))
    better = available and new["roc_auc"] > old["roc_auc"] and new["brier_score"] < old["brier_score"]
    conclusion = ("A variante ainda não demonstrou melhora suficientemente consistente para avançar ao calibrador definitivo. Houve ganho pontual de AUC e Brier no novo piloto principal."
                  if better else
                  "A alteração não melhorou conjuntamente AUC e Brier no novo piloto principal; não há base para promovê-la a calibrador definitivo.")
    if not available:
        conclusion = "A evidência disponível não permite calcular todas as métricas principais; não há base para promover a variante a calibrador definitivo."
    elif better:
        auc_ci = primary["paired_bootstrap"]["auc_vs_legacy"]["ci95"]
        brier_ci = primary["paired_bootstrap"]["brier_vs_legacy"]["ci95"]
        if auc_ci and brier_ci and (auc_ci[0] <= 0 or brier_ci[1] >= 0):
            conclusion += " Os intervalos exploratórios ainda são compatíveis com ausência de ganho."
        if new["brier_score"] >= primary["prevalence_reference"]["brier_score"]:
            conclusion += " As probabilidades da variante não superaram a referência de prevalência no Brier."
    if (primary["context_violation"]["pr_auc_trapezoidal"] is not None
            and primary["legacy_violation"]["pr_auc_trapezoidal"] is not None
            and primary["context_violation"]["pr_auc_trapezoidal"] < primary["legacy_violation"]["pr_auc_trapezoidal"]):
        conclusion += " A AUC-PR de violações piorou."
    lines = [
        "# Diagnóstico do calibrador — piloto com 50 complexos", "", conclusion, "",
        "Este é desenvolvimento na amostra de 50 fornecida pelo usuário. O novo docking usa as mesmas moléculas do histórico; não constitui um teste final independente. O conjunto de 200 não foi executado, e nenhum calibrador automático foi instalado.", "",
        "## O que foi ajustado", "",
        "A variante opcional acrescenta a identidade do algoritmo e a retenção da pose de entrada aos 11 descritores existentes. Mantém regressão logística balanceada C=1, Venn–Abers e os critérios originais de autorização. A evidência experimental e o resultado ampliado não entram no cálculo dos novos descritores. A calibração Venn–Abers foi conferida contra uma implementação isotônica independente.", "",
        "## Comparação nas mesmas previsões fora do ajuste", "",
        "As duas representações são reajustadas nas mesmas partições; esta tabela não compara diretamente com a AUC do artefato histórico do artigo. AUC maior e Brier menor são melhores. A referência de prevalência é calculada somente nos grupos de desenvolvimento de cada fold. O numerador da coluna pares contém as linhas com probabilidade disponível nos dois candidatos. Os folds produzem probabilidades diagnósticas, inclusive quando ficam abaixo dos mínimos de evidência da rotina oficial; nenhum artefato autorizado é criado com isso.", "",
        "| Experimento | Pares avaliados/válidos | AUC anterior → variante | Brier anterior → variante | Brier da prevalência |",
        "|---|---:|---:|---:|---:|", *rows, "",
        f"No piloto uniforme, a AUC-PR de violações por integração trapezoidal passou de {number(primary['legacy_violation']['pr_auc_trapezoidal'])} para {number(primary['context_violation']['pr_auc_trapezoidal'])}; a precisão média (AP, medida diferente) passou de {number(primary['legacy_violation']['average_precision'])} para {number(primary['context_violation']['average_precision'])}. Há {primary['violations']} violações nas {primary['common_calibrated_pairs']} linhas comuns.", "",
        f"A variante tem Brier {number(new['brier_score'])}, frente a {number(primary['prevalence_reference']['brier_score'])} da referência de prevalência. As diferenças devem ser interpretadas com seus intervalos e a disponibilidade da calibração; melhora de ordenação não demonstra, por si só, probabilidades mais úteis que uma previsão simples.", "",
        "### Incerteza da diferença no piloto principal", "",
    ]
    for key, title in (("auc_vs_legacy", "AUC variante − anterior"),
                       ("brier_vs_legacy", "Brier variante − anterior"),
                       ("brier_vs_prevalence", "Brier variante − prevalência")):
        ci = primary["paired_bootstrap"][key]["ci95"]
        lines.append(f"- {title}: IC exploratório de 95% [{number(ci[0])}; {number(ci[1])}]." if ci is not None
                     else f"- {title}: intervalo indisponível com as classes/grupos observados.")
    lines.extend(["", "Bootstrap por grupo molecular, condicional aos modelos ajustados; não inclui a incerteza da seleção de candidatos. A sensibilidade por origem PDB/componente compartilhado consta no JSON e pode ter menos linhas disponíveis. Não se trocou seed ou fold para obter melhora.", "",
                  "## Política hipotética, limiar inferior 0,90", "",
                  "| Modelo | Aceites/pares válidos | Violações entre aceites | Risco observado | Cobertura |",
                  "|---|---:|---:|---:|---:|"])
    for key, title in (("legacy", "Anterior"), ("context", "Variante")):
        p = data["fixed_uniform"]["primary_full_denominator_policies"][key]
        lines.append(f"| {title} | {p['accepted']}/{p['total_valid_pairs']} | {p['false_accepted']} | {percentage(p['risk'])} | {percentage(p['coverage_of_all_valid_pairs'])} |")
    lines.extend(["", "Essa tabela é uma simulação com previsões de desenvolvimento. Não libera o descarte do FULL. Menor risco com menor cobertura não demonstra dominância, e ausência de erros não é garantia. Falhas do REDUCED exigem fallback e permanecem no denominador das execuções planejadas, embora não formem pares para treinar o classificador.", "",
                  "## Sensibilidade ao agrupamento por origem", ""])
    previous_policy = data["fixed_uniform"]["primary_full_denominator_policies"]["legacy"]
    current_policy = data["fixed_uniform"]["primary_full_denominator_policies"]["context"]
    lines[-2:-2] = [f"A calibração da representação anterior ficou disponível em {previous_policy['calibration_available_pairs']}/{previous_policy['total_valid_pairs']} pares; na variante, em {current_policy['calibration_available_pairs']}/{current_policy['total_valid_pairs']}. Nos demais casos é necessário fallback. A variante deixou sem probabilidade {current_policy['calibration_unavailable_violations']} violações. Isso limita a comparação e explica parte da cobertura menor; não é correto tratar apenas os casos com probabilidade como se fossem toda a amostra.", ""]
    sensitivity = data["fixed_uniform"].get("source_blocked")
    cluster_count = data["fixed_uniform"]["source_cluster_audit"]["source_clusters"]
    if sensitivity:
        lines.extend([f"O bloqueio por origem PDB ou sequência exata de um componente produz {cluster_count} grupos. A comparação tem {sensitivity['common_calibrated_pairs']}/{sensitivity['total_pairs']} pares e {sensitivity['violations']} violações disponíveis. AUC: {number(sensitivity['legacy']['roc_auc'])} → {number(sensitivity['context']['roc_auc'])}; Brier: {number(sensitivity['legacy']['brier_score'])} → {number(sensitivity['context']['brier_score'])}. A disponibilidade menor deve ser considerada junto das métricas; linhas sem ambas as classes na calibração não foram preenchidas.", ""])
        if cluster_count == data["fixed_uniform"]["source_cluster_audit"]["complex_fingerprints"]:
            lines.extend(["Nesta preparação uniforme, o bloqueio não uniu grupos adicionais: ele reproduz a análise principal e não conta como nova evidência de robustez. A preparação semeada e o histórico têm agrupamentos e disponibilidades diferentes, registrados no JSON.", ""])
    else:
        lines.extend(["A sensibilidade por origem ficou indisponível sem trocar as partições; a causa exata consta no JSON.", ""])
    lines.extend(["## Integridade e dados de entrada", ""])
    runs = data["runs"]
    lines.extend([f"- {runs['target_count']} identificadores; {runs['molecular_fingerprints']} fingerprints distintos. Aliases: {runs['aliases']}.",
                  f"- {runs['counts']['planned']} execuções: {runs['counts']['succeeded']} concluídas com resultado e {runs['counts']['failed']} falhas registradas.",
                  f"- {runs['verified_input_files']} arquivos originais conferidos por SHA-256. Protocolos, recibos e orçamento de 7.600 avaliações efetivas conferidos. Motor atual igual ao congelado.", ""])
    if "preparations" in data:
        uniform = data["preparations"]["fixed_uniform"]["counts"]
        seeded = data["preparations"]["fixed_seeded"]["counts"]
        lines.extend([f"Após exigir pares completos e excluir um alias, a preparação uniforme contém {uniform['usable_rows']} pares de {uniform['independent_complexes']} grupos; a semeada contém {seeded['usable_rows']} pares de {seeded['independent_complexes']} grupos. Um grupo sem REDUCED válido não é substituído por outro complexo.", ""])
    lines.extend(["| Condição | Execuções com resultado | Falhas | Poses com DockQ ≥0,23 / planejadas |",
                  "|---|---:|---:|---:|"])
    for key in ("reduced_uniform", "expanded_uniform", "reduced_seeded", "expanded_seeded"):
        condition = runs["conditions"][key]
        lines.append(f"| {key} | {condition.get('succeeded', 0)} | {condition.get('failed', 0)} | {condition.get('structural_success', 0)}/{condition['planned']} |")
    lines.extend(["", "Essas contagens incluem os 50 identificadores e as falhas no denominador; o treinamento remove um alias por regra anterior aos resultados e exige pares completos. Qualidade estrutural acima de 0,23 e preservação relativa ao ampliado são desfechos diferentes.", ""])
    audit = data["input_pose_audit"]
    lines.extend([f"Na auditoria posterior, {audit['input_above_023']}/{audit['n']} poses fornecidas já tinham DockQ experimental ≥0,23, com média {number(audit['mean_input_dockq'])}. Isso caracteriza refinamento local de entradas em sua maioria próximas da referência. A origem e a independência da preparação dessas poses ainda precisam ser documentadas; esses números não são descritores do modelo.", "",
                  "## Decisão para a etapa de 200", "",
                  "A variante permanece experimental. Antes de uma execução definitiva, congelar descritores e protocolo, auditar duplicatas/homologia entre conjuntos, reservar grupos independentes para ajuste, calibração de probabilidade, calibração de risco e avaliação, e exigir os gates originais de evidência. A melhora pontual de desenvolvimento, quando presente, justifica uma candidata para essa etapa, não a alegação de um calibrador já validado.", "",
                  "A preparação mesclada é inelegível para autorização em produção. O experimento fatorial mantém todas as condições. Uma futura política adaptativa com orçamento fixo precisa de contrato próprio para os pares escolhidos; nenhum desses limites foi relaxado neste piloto. O benefício operacional também precisa ser medido com o custo total dos descritores e dos fallbacks, além das métricas de probabilidade; este piloto não demonstra economia real de uma política adaptativa.", "",
                  "Os números completos, intervalos, disponibilidade por fold, erros por complexo e hashes estão em `resumo_piloto.json`. Os gráficos estão nas pastas `fixed_uniform_assessment`, `fixed_seeded_assessment` e `legacy_v2_assessment`. A implementação e a reprodução estão documentadas em `docs/PILOTO_CALIBRADOR_50.md`.", ""])
    check = data.get("official_training_check")
    if check:
        lines.extend(["## Verificação do treinamento oficial", ""])
        if check["status"] == "insufficient_evidence":
            support = check["evidence"]
            lines.extend([f"A tentativa única com a divisão oficial foi interrompida por evidência insuficiente na etapa `{check['stage']}`. Grupos disponíveis/requeridos: {support.get('independent_complex_count')}/{support.get('minimum_independent_complexes')}; grupos com classe positiva: {support.get('positive_class_complex_count')}/{support.get('minimum_positive_class_complexes')}; grupos com classe negativa: {support.get('negative_class_complex_count')}/{support.get('minimum_negative_class_complexes')}. Não se alteraram os mínimos nem a seed para produzir um artefato.", ""])
        else:
            lines.extend(["O treinamento produziu somente um artefato diagnóstico, com decisão automática desabilitada; ele não foi instalado no sistema.", ""])
        lines.extend(["Evidência completa em `fixed_uniform_strict_training_evidence.json`. Testes da implementação documentados em `VALIDACAO.md`.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()
    root = Path(args.output_root).resolve()
    destination = root / "resumo_piloto.json"
    if destination.exists():
        raise FileExistsError("final pilot summary already exists")
    names = ["legacy_v2_assessment", "fixed_uniform_assessment", "fixed_seeded_assessment"]
    while not all((root / name / "primary_summary.json").exists() and (
        (root / name / "source_blocked_summary.json").exists()
        or (root / name / "source_blocked_unavailable.json").exists()) for name in names):
        if not args.wait:
            raise ValueError("pilot assessments are not yet complete")
        time.sleep(5)
    preparations = {}
    for name, directory in (("legacy", "legacy_context_dataset"),
                            ("fixed_uniform", "fixed_preparations/merged_uniform"),
                            ("fixed_seeded", "fixed_preparations/merged_seeded")):
        preparation = read(root / directory / "dataset_build_report.json")
        payload = Path(preparation["dataset_path"]).read_bytes()
        if not preparation["valid"] or hashlib.sha256(payload).hexdigest() != preparation["dataset_sha256"]:
            raise ValueError("prepared dataset changed")
        preparations[name] = preparation
    commitment = read(root / "development_comparison_commitment.json")
    for source, expected in commitment["source_sha256"].items():
        if hashlib.sha256(Path(source).read_bytes()).hexdigest() != expected:
            raise ValueError(f"comparison implementation changed: {source}")
    data = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "exploratory_development_on_50_complexes_only", "automatic_authorization": False,
        "definitive_200_training_started": False,
        "runs": verify_runs(Path(args.study_root).resolve()),
        "legacy": assessment(root / "legacy_v2_assessment"),
        "fixed_uniform": assessment(root / "fixed_uniform_assessment"),
        "fixed_seeded": assessment(root / "fixed_seeded_assessment"),
        "input_pose_audit": read(root / "input_pose_audit/input_quality_summary.json"),
        "development_comparison_commitment": commitment,
        "preparations": preparations,
    }
    check_path = root / "fixed_uniform_strict_training_evidence.json"
    if check_path.exists():
        data["official_training_check"] = read(check_path)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (root / "RELATORIO_PILOTO_50.md").write_text(report_text(data), encoding="utf-8")
    print(json.dumps({"summary": str(destination), "run_counts": data["runs"]["counts"]}))


if __name__ == "__main__":
    main()
