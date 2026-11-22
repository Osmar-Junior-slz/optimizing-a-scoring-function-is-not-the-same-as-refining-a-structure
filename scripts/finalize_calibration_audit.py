"""Replay frozen comparisons, add support audits, and write a readable report."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys
import argparse
from datetime import datetime, timezone

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(name, "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import pandas as pd
from docking_confidence.application.calibration_comparison import METHODS, metric_bundle, ComparisonOptions
from docking_confidence.calibration.venn_abers import VennAbersCalibrator
from docking_confidence.risk.policy import select_policy


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/calibration_audit_20261115"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def enrich_case(name):
    path = OUTPUT / name / "comparison.json"
    result = read(path)
    old_bounds = result["operational"].pop("oracles_not_applicable_policies", None)
    if old_bounds:
        result["operational"]["bounds_and_baselines"] = {
            "always_valid_reduced_with_failure_fallback": {
                **old_bounds["accept_every_valid_pose_even_if_wrong"], "uses_outcomes_for_decision": False,
                "interpretation": "applicable unconstrained baseline, not a calibrated policy or an oracle"},
            "perfect_preservation_oracle": {
                **old_bounds["accept_only_truly_preserved"], "uses_outcomes_for_decision": True,
                "interpretation": "unavailable future labels; retrospective oracle only"}}
    prediction_path = path.parent / "predictions.csv"
    frame = pd.DataFrame(result["predictions"])
    initialization = name.split("_")[0]
    source = Path(result.get("preparation_dir", ROOT / "output/calibrator_pilot_50_20261115/fixed_preparations" / ("merged_" + initialization)))
    cfg = read(source / "dataset_config.json")
    original = pd.read_csv(source / "paired_dataset.csv").sort_values(
        ["complex_fingerprint_sha256", "complex_id", "algorithm", "seed"], kind="stable").reset_index(drop=True)
    if not frame[["complex_id", "algorithm", "seed"]].equals(original[["complex_id", "algorithm", "seed"]]):
        raise ValueError("prediction identities disagree with frozen input")
    y = frame.preserved.to_numpy(int)
    common = np.logical_and.reduce([np.isfinite(frame[m + "__score"].to_numpy(float)) for m in METHODS])
    official = []
    for method in METHODS:
        p = frame[method + "__score"].to_numpy(float)
        ranking = frame.base_raw_score.to_numpy(float) if method == "uncalibrated" else p
        available = np.isfinite(p)
        result["metrics"][method].update(
            common_test_rows=metric_bundle(y[common], p[common], ranking[common]),
            own_available_rows=metric_bundle(y[available], p[available], ranking[available]))
        result["metrics"][method]["unavailable_violations"] = int(((~available) & (y == 0)).sum())
        result["metrics"][method]["per_fold"] = {
            str(fold): metric_bundle(y[mask], p[mask], ranking[mask])
            for fold in sorted(frame.fold.unique()) if (mask := common & frame.fold.eq(fold).to_numpy()).any()}
    for fold in result["folds"]:
        if "model" not in fold:
            official.append({"fold": fold["fold"], "probability_evidence_passed": False,
                             "reason": fold["unavailable_reason"]})
            continue
        model = fold["model"]
        X = original[result["features"]].to_numpy(float)
        raw = ((X - model["scaler_mean"]) / model["scaler_scale"]) @ np.asarray(model["coef"]) + model["intercept"]
        domain = fold["domain"]
        inside = ((X >= domain["lower"]) & (X <= domain["upper"])).all(axis=1)
        cal = original.complex_fingerprint_sha256.isin(fold["groups"]["probability_calibration"]).to_numpy() & inside
        cal_groups = original.complex_fingerprint_sha256.to_numpy(str)
        support = {"independent_complex_count": len(np.unique(cal_groups[cal])),
                   "positive_class_complex_count": len(np.unique(cal_groups[cal & (y == 1)])),
                   "negative_class_complex_count": len(np.unique(cal_groups[cal & (y == 0)]))}
        expected = cfg["calibration"]
        supported = (support["independent_complex_count"] >= expected["minimum_independent_complexes"]
                     and support["positive_class_complex_count"] >= expected["minimum_positive_class_complexes"]
                     and support["negative_class_complex_count"] >= expected["minimum_negative_class_complexes"])
        probe = {"fold": fold["fold"], "probability_support": support,
                 "probability_evidence_passed": bool(supported), "threshold": .9}
        test = frame.fold.eq(fold["fold"]).to_numpy()
        if not np.allclose(frame.base_raw_score.to_numpy(float)[test], raw[test], rtol=0, atol=1e-12):
            raise ValueError("saved base scores do not replay")
        if "venn_abers" in fold["calibrators"]:
            calibrator = VennAbersCalibrator.from_export(fold["calibrators"]["venn_abers"])
            _, _, probability = calibrator.predict_interval(frame.base_raw_score.to_numpy(float)[test])
            if not np.array_equal(probability, frame.venn_abers__score.to_numpy(float)[test]):
                raise ValueError("saved Venn-Abers predictions do not replay")
            val = original.complex_fingerprint_sha256.isin(fold["groups"]["threshold_validation"]).to_numpy()
            lower, _, _ = calibrator.predict_interval(raw[val])
            validation = original.loc[val].assign(preserved=y[val])
            policy, _ = select_policy(validation, lower, inside[val], cfg["policy"])
            probe["fixed_0_90_risk_gate"] = policy
        else:
            probe["reason"] = "no_two_class_venn_abers_calibrator"
        probe["automatic_authorization"] = False
        official.append(probe)
    result["official_gate_probes"] = official
    specification = importlib.util.spec_from_file_location("frozen_bootstrap_driver", ROOT / "scripts/run_calibration_audit.py")
    driver = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(driver)
    result["paired_group_bootstrap"] = driver.paired_intervals(frame, ComparisonOptions(**result["options"]))
    result["prediction_file_sha256"] = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    result["point_metrics_scope"] = "same available outer rows; missing methods excluded only from probability metrics, never from operational fallback costs"
    write(path, result)
    return result


def num(value, digits=3):
    return "indisponível" if value is None else f"{value:.{digits}f}".replace(".", ",")


def generate_report(summary):
    primary = summary["cases"]["uniform_original11"]
    seeded = summary["cases"]["seeded_original11"]
    lines = ["# Auditoria comparativa do calibrador — 15/11/2026", "",
        "O Venn–Abers melhora as probabilidades em relação à sigmoid do modelo original nesta comparação exploratória. Isso não se traduziu em economia de processamento nem demonstra superioridade global sobre a isotônica. A recomendação é manter o Venn–Abers, não ativar a cascata no orçamento atual e investigar um REDUCED efetivamente mais barato antes de trocar o calibrador.", "",
        "Os dockings das coleções não foram reexecutados. Os resultados são reanálises dos lotes salvos das 50 proteínas; a suíte de código usa exemplos sintéticos. Nenhum modelo foi instalado ou autorizado automaticamente; as 200 proteínas não foram utilizadas.", "",
        "## Dados, rótulos e objetivo", "",
        "Os 50 identificadores correspondem a 49 fingerprints moleculares. O alias 1bui_BC é excluído pela regra anterior aos desfechos. Cada inicialização tem 147 pares planejados canônicos (49 × três algoritmos). Há 137 pares completos uniformes e 138 semeados; os 10 e nove REDUCED falhos permanecem na reconstrução operacional e exigem FULL. Os 600 registros originais foram verificados, incluindo condições excluídas do treinamento.", "",
        "O rótulo 1 exige simultaneamente DockQ REDUCED ≥0,23 e perda FULL−REDUCED ≤0,05. O rótulo 0 é o complemento. A direção está correta para preservar qualidade estrutural, e a implementação numérica do Venn–Abers não apresentou erro nos testes independentes. Entretanto, esse alvo não é equivalente a prever benefício de executar FULL: quando ambos são ruins, o rótulo continua 0 mesmo que FULL não resolva o problema. Não se deve retirar a qualidade mínima para melhorar artificialmente os números.", "",
        "Os preditores são exclusivamente REDUCED. A análise principal mantém os 11 descritores originais, regressão logística C=1 e pesos balanceados. Os 15 descritores são sensibilidade separada. Referência experimental, DockQ e tempo FULL não são características de inferência. A retenção da entrada da variante de 15 compara com a pose fornecida, não com a estrutura experimental.", "",
        "## Desenho da comparação", "",
        "Cinco folds externos por grupos planejados, seed 20261115, sem procura de divisões favoráveis. Cada desenvolvimento externo tem papéis disjuntos para modelo/domínio, calibração probabilística e validação de limiares. Uma única regressão por fold fornece os mesmos escores para sigmoid original, isotônica convencional e Venn–Abers. O teste externo não ajusta nenhum componente. O limiar de cada alternativa é escolhido na validação para maximizar cobertura com risco empírico ≤10%, pelo menos três grupos aceitos e ambas as classes na validação. São políticas diagnósticas, sem garantia de risco ou autorização de produção. A política com limite inferior Venn–Abers é registrada adicionalmente.", "",
        "Os dados já foram examinados em desenvolvimento anterior; não constituem um teste final prospectivo. A separação por fingerprint evita aliases exatos entre papéis, mas não certifica ausência de homologia ou dependência entre complexos de mesma origem. Os três algoritmos por proteína também são correlacionados: separar proteínas entre papéis não torna as linhas do Venn–Abers i.i.d. A validade probabilística teórica não deve ser transferida automaticamente a essa população. O risco operacional usa agrupamento por complexo e permanece aproximado.", "",
        "## Probabilidades: comparação principal", "",
        f"Os três métodos têm probabilidade disponível nos mesmos {primary['counts']['common_test_rows']}/137 pares, de {primary['counts']['common_test_groups']} grupos. Nesse subconjunto há somente cinco violações. Falta de classe no treino ou na calibração deixou outros folds indisponíveis; isso não foi preenchido com probabilidades inventadas. As métricas abaixo são condicionais à disponibilidade, não estimativas de toda a amostra.", "",
        "| Método | Brier ↓ | ECE 10 bins ↓ | Log-loss ↓ | AUC-ROC ↑ |",
        "|---|---:|---:|---:|---:|"]
    labels = {"uncalibrated": "Original sem calibração", "isotonic": "Isotônica", "venn_abers": "Venn–Abers"}
    for method in METHODS:
        m = primary["metrics"][method]["common_test_rows"]["preservation"]
        lines.append(f"| {labels[method]} | {num(m['brier_score'])} | {num(m['expected_calibration_error_10bin'])} | {num(m['log_loss'])} | {num(m['roc_auc'])} |")
    base = primary["prevalence_reference_common"]["preservation"]
    lines.extend([f"| Prevalência do desenvolvimento | {num(base['brier_score'])} | {num(base['expected_calibration_error_10bin'])} | {num(base['log_loss'])} | — |", "",
        "A isotônica teve menor Brier/ECE; o Venn–Abers teve menor log-loss entre os três modelos. A isotônica produz probabilidades exatas 0/1 e alguns desses extremos erram; a log-loss foi calculada com clipping em 10⁻¹², documentado no JSON. Não selecionar uma vencedora olhando apenas o indicador favorável. ECE depende de bins e é instável com poucos exemplos; também estão registrados ECE com cinco bins, contagens por bin, médias previstas, frequências observadas, erros extremos e métricas por fold.", "",
        "O Brier da prevalência foi melhor que o dos três métodos. Isso mostra que melhorar a sigmoid balanceada não basta para demonstrar informação útil além da alta frequência de preservação. Pesos balanceados mudam a interpretação da sigmoid como probabilidade populacional; parte do ganho de calibração corrige essa distorção de prevalência. Uma transformação monotônica tampouco cria discriminação estrutural nova; diferenças agregadas de AUC também refletem empates e escalas entre folds.", "",
        "### Incerteza pareada", "",
        "Bootstrap de 2.000 amostras por complexo, condicional aos modelos ajustados, sem corrigir a exploração anterior:", ""])
    for comparison in ("venn_abers_minus_uncalibrated", "venn_abers_minus_isotonic", "venn_abers_minus_prevalence"):
        ci = primary["paired_group_bootstrap"][comparison]["brier_score"]["ci95"]
        lines.append(f"- Δ Brier {comparison}: [{num(ci[0])}; {num(ci[1])}]." if ci else f"- {comparison}: intervalo indisponível.")
    lines.extend(["", "### Sensibilidade semeada e com 15 descritores", "",
        "| Conjunto | Brier original | Brier isotônica | Brier Venn–Abers | Pares comuns |",
        "|---|---:|---:|---:|---:|"])
    for name, case in summary["cases"].items():
        scores = [case["metrics"][m]["common_test_rows"]["preservation"]["brier_score"] for m in METHODS]
        lines.append(f"| {name} | {num(scores[0])} | {num(scores[1])} | {num(scores[2])} | {case['counts']['common_test_rows']} |")
    lines.extend(["", "Inicializações são ensaios correlacionados sobre as mesmas moléculas. A sensibilidade semeada não substitui o resultado principal. Nela, a diferença de Brier entre isotônica e Venn–Abers tem intervalo que inclui zero, enquanto o Venn–Abers é menos vulnerável aos erros extremos em log-loss. Não há evidência de superioridade geral que justifique substituí-lo.", "",
        "## Decisões e custo: todos os 147 pares planejados", "",
        "A tabela utiliza limiares escolhidos apenas na validação e todos os jobs externos, incluindo indisponibilidade do modelo e falhas REDUCED. Os resultados são simulações sobre tempos e poses salvos. No experimento observado, nenhum FULL foi evitado: todos foram executados pelo desenho fatorial.", "",
        "| Política reduzido-primeiro | FULL dispensados (simulados) | Aceites incorretos | Economia máxima do motor (s) |",
        "|---|---:|---:|---:|"])
    for method in (*METHODS, "cascade_margin"):
        row = primary["operational"][method]["reduced_first"]
        label = labels.get(method, "Hipótese linear de cascata")
        lines.append(f"| {label} | {row['hypothetical_full_avoided']}/147 | {row['hypothetical_incorrect_acceptances']} | {num(row['engine_saving_upper_bound_sec'], 1)} |")
    bounds = primary["operational"]["bounds_and_baselines"]
    perfect = bounds["perfect_preservation_oracle"]
    every = bounds["always_valid_reduced_with_failure_fallback"]
    lines.extend(["", "Valores negativos significam custo adicional mesmo antes de todo o overhead. Os tempos do motor dos REDUCED falhos não foram registrados separadamente; eles não foram inventados. As durações de todas as tarefas, incluindo falhas, foram contabilizadas em uma segunda análise no JSON, mas essas durações incluem pós-processamento experimental e não equivalem a CPU ou ao tempo de parede de uma política serial.", "",
        f"Nos 137 pares válidos uniformes, REDUCED soma 8.333,934 s e FULL 8.351,828 s. Mesmo dispensando todos os FULL com REDUCED válido, inclusive os casos incorretos, o teto otimista é {num(every['engine_saving_upper_bound_sec'], 1)} s antes do tempo do motor das dez falhas e do calibrador. Nas durações completas das tarefas, até esse cenário idealizado perde {num(-every['sum_task_duration_saving_upper_bound_sec'], 1)} s.", "",
        f"Um oracle que conhecesse os rótulos e aceitasse todos os {perfect['hypothetical_full_avoided']} reduzidos realmente preservados ainda teria economia do motor de, no máximo, {num(perfect['engine_saving_upper_bound_sec'], 1)} s. É um limite retrospectivo, não uma política aplicável. Assim, exigir preservação perfeita já elimina o ganho computacional neste regime, independentemente de escolher Venn–Abers ou isotônica.", "",
        "Aceitar qualquer REDUCED válido com fallback apenas nas falhas é um baseline aplicável, não um oracle: ele não usa rótulos para decidir. Neste conjunto, porém, tem 13 aceites não preservados e também aumenta a soma das durações das tarefas. O oracle acima é diferente: ele consulta os rótulos verdadeiros, que não estariam disponíveis em uso.", "",
        "A alternativa de escolher na validação se vale a pena habilitar globalmente a cascata também está registrada. No regime principal, todos os métodos optaram por FULL-only em todos os folds, pois a validação não mostrou economia da cascata. Esse roteamento simulado apenas igualou o custo do baseline; não foi instalado e o guard do experimento fatorial permanece intacto.", "",
        "Não é possível declarar economia real de produção: faltam execuções adaptativas com temporização integral, overhead de extração/inferência, custo das falhas, paralelismo controlado e medição de CPU/parede apropriada. O código agora registra preparação de características e decisão nas futuras execuções com calibrador; o escopo exclui o motor, a referência experimental e a gravação final do próprio registro.", "",
        "## Por que o benefício não estava demonstrado", "",
        "1. Comparações anteriores de representações com Venn–Abers nos dois lados não isolavam o efeito de calibrar o mesmo modelo-base.",
        "2. Não há exemplos independentes suficientes de não preservação em vários papéis. A caixa de domínio 1%–99% pode remover justamente os negativos que restavam. Os detalhes por característica e fold foram preservados; os mínimos não foram reduzidos para gerar aprovação.",
        "3. O esquema com quatro papéis usa muitos grupos. Com 49, a calibração oficial piloto costuma ficar abaixo de dez grupos após o domínio; o perfil padrão exige ainda mais. Repetir algoritmos não resolve o número de proteínas independentes.",
        "4. A sigmoid de um modelo balanceado não é uma boa referência de probabilidade populacional. A prevalência precisa constar como baseline.",
        "5. O critério de tempo legado analisava somente os aceites, omitindo o custo REDUCED dos fallbacks. Um bom resultado nesse critério podia coexistir com cascata mais lenta.",
        "6. Orçamento efetivo igual de 7.600 avaliações torna REDUCED quase tão caro quanto FULL. O experimento fatorial responde sobre amplitude/inicialização, mas não oferece automaticamente uma etapa barata para uma cascata.",
        "7. Os exemplos já foram vistos no desenvolvimento e têm possíveis relações de origem/homologia; falta confirmação em complexos independentes. As poses fornecidas também são, em sua maioria, próximas da referência, caracterizando refinamento local. A origem e a preparação precisam ser documentadas.", "",
        "## Hipótese Learning to Cascade", "",
        "Foi implementada apenas uma hipótese linear inspirada no objetivo do artigo, sem alegar reprodução da rede original: Ridge com alpha=1 estima separadamente DockQ REDUCED e ganho FULL−REDUCED, usando somente descritores REDUCED e os mesmos grupos de treino. A margem prevista combina qualidade mínima e ganho tolerável; o limiar usa a mesma validação dos demais métodos. As saídas são regressões, não probabilidades calibradas.", ""])
    cascade = primary["operational"]["cascade_regression"]
    lines.extend([f"Na análise principal, MAE da qualidade reduzida = {num(cascade['reduced_quality_mae'])}; MAE do ganho FULL = {num(cascade['full_gain_mae'])}, maior que a tolerância de perda de 0,05. A política também aumentou o custo. Portanto, a hipótese não foi promovida e não substituiu Venn–Abers.", "",
        "## Correções implementadas", "",
        "- `application/calibration_comparison.py`: comparação com modelo compartilhado, quatro papéis por grupo, limiares ajustados só na validação, fallback para métodos indisponíveis e hipótese de cascata separada.",
        "- `application/runtime.py`: novo escopo de custo total dos pares, incluindo todos os REDUCED, FULL de fallback e overhead medido. Overhead ausente permanece desconhecido e bloqueia o critério econômico.",
        "- `io/config.py` e três perfis locais: validação e ativação explícita do novo escopo. Os critérios estatísticos de risco/calibração e a autorização automática desabilitada foram preservados.",
        "- `artifact/schema.py`: validação das identidades de custo e do novo escopo; artefatos sem a opção nova continuam usando o cálculo legado, preservando reprodução e compatibilidade.",
        "- `application/batch.py`: medição futura da preparação e da avaliação do calibrador, vinculada aos recibos/decisões por hashes. Nenhum algoritmo de docking foi alterado.",
        "- `application/training_dataset.py`: transporte apenas de overhead medido e de decisão íntegra ligada ao REDUCED. Valores não medidos não são preenchidos com zero.",
        "- `application/pilot_diagnostics.py`: rótulos pela regra compartilhada e parâmetros do snapshot, em vez de constantes embutidas na comparação.",
        "- Scripts, protocolo, relatórios e testes de isolamento do teste, domínio, pareamento, custo, compatibilidade e integridade temporal.", "",
        "As configurações locais novas têm hashes diferentes. Preparações antigas permanecem intocadas e esta análise lê seus snapshots congelados; para treinar com as configurações atuais é necessário preparar um novo snapshot. Não reescrever o hash das linhas antigas nem reaproveitar o conjunto como se tivesse sido preparado com outro protocolo.", "",
        "## Recomendação e próximo teste", "",
        "Manter Venn–Abers na arquitetura. A isotônica é uma candidata comparativa com melhor Brier em parte deste estudo, mas tem erros extremos, disponibilidade e risco operacional que impedem concluir superioridade geral. Nenhuma política reduzido-primeiro demonstrou economia neste orçamento; não ativar a cascata como otimização de tempo nessas condições.", "",
        "O próximo ensaio útil é primeiro reduzir o custo real do REDUCED. Foi preparada a especificação `followup_5_spec.json`: cinco fingerprints escolhidos deterministicamente sem usar os resultados, REDUCED com 1.900/3.800/7.600 avaliações e FULL com 7.600, três algoritmos, seed 1 e inicialização uniforme: 60 jobs planejados, não executados. É necessário um protocolo auditável próprio de orçamento misto; não desativar o guard que exige todas as condições do experimento fatorial. Cinco complexos avaliam o mecanismo, sem fornecer comprovação estatística final.", "",
        "Após encontrar uma etapa barata que preserve qualidade suficiente, congelar descritores, divisão por grupos/homologia, objetivo de perda, calibrador e limiares de validação. Somente então testar uma política inteira em dados de grupos novos, com exemplos de ambas as classes e custo integral medido. Esse é o experimento que pode demonstrar benefício científico; repetir o calibrador no mesmo orçamento não resolve o gargalo econômico.", "",
        "## Reproduzir e conferir", "",
        "Executar com o Python 3.12 de `.venv`: `python scripts/run_calibration_audit.py --output-dir output/nova_auditoria` (usar uma pasta nova). O script lê e verifica resultados salvos; não chama o motor. A finalização deste estudo está em `scripts/finalize_calibration_audit.py`. Os CSVs preservam todas as previsões externas, e os JSONs preservam modelos, mapas de calibração, divisões, limiares, bins, intervalos, falhas e custos. Os gráficos estão em cada subpasta de comparação.", "",
        "Testes e manifesto dos arquivos modificados constam nos registros desta pasta. As limitações acima fazem parte do resultado e não devem ser retiradas ao transportar os números para o artigo.", "",
        "## Referências metodológicas", "",
        "- [Vovk, Petej e Fedorova, 2015 — Venn–Abers indutivo/cruzado e distinção entre probabilidades imprecisas e ponto agregado](https://papers.nips.cc/paper/2015/file/a9a1d5317a33ae8cef33961c34144f84-Paper.pdf).",
        "- [Enomoto e Eda, AAAI 2021 — Learning to Cascade e custo/qualidade do sistema inteiro](https://cdn.aaai.org/ojs/16900/16900-13-20394-1-2-20210518.pdf).",
        "- [Documentação oficial de IsotonicRegression](https://scikit-learn.org/stable/modules/generated/sklearn.isotonic.IsotonicRegression.html).", ""])
    va_quality = primary["operational"]["venn_abers"]["reduced_first"]
    quality_note = (f"Na simulação Venn–Abers, {va_quality['structural_successes']}/147 saídas finais têm DockQ ≥0,23, contra "
                    f"{va_quality['full_only_structural_successes']}/147 do FULL-only. Ainda houve "
                    f"{va_quality['hypothetical_incorrect_acceptances']} aceite que violou o rótulo combinado de preservação. "
                    "Taxa de qualidade absoluta, perda em relação ao FULL e risco entre aceites são critérios distintos; "
                    "esses ganhos descritivos não garantem preservação por caso nem resolvem o custo adicional.")
    cost_position = next(i for i, value in enumerate(lines) if value.startswith("Valores negativos significam"))
    lines[cost_position:cost_position] = [quality_note, ""]
    gate_lines = ["## Evidência exigida pelos gates atuais", "",
                  "A comparação acima admite ajustes diagnósticos com poucos exemplos; isso não libera produção. Os probes abaixo preservam os parâmetros oficiais do snapshot piloto: dez grupos na calibração probabilística, pelo menos dois grupos de cada classe, corte 0,90 e os critérios de risco originais.", "",
                  "| Fold | Grupos na calibração dentro do domínio | Grupos com não preservação | Evidência probabilística aprovada | Regra oficial de risco aprovada |",
                  "|---|---:|---:|---|---|"]
    for probe in primary["official_gate_probes"]:
        support = probe.get("probability_support")
        if support is None:
            gate_lines.append(f"| {probe['fold']} | — | — | Não: treino sem ambas as classes | — |")
        else:
            policy = probe.get("fixed_0_90_risk_gate", {})
            gate_lines.append(f"| {probe['fold']} | {support['independent_complex_count']} | {support['negative_class_complex_count']} | {'Sim' if probe['probability_evidence_passed'] else 'Não'} | {'Sim' if policy.get('supported') else 'Não'} |")
    gate_lines.extend(["", "Nenhum probe autoriza o uso automático. O domínio excluiu negativos por quantidade/densidade de contatos, fração de contato do ligante, dispersão e colisões em diferentes folds; os grupos e as características exatas estão nos JSONs. Esses registros diagnosticam a escassez sem escolher outra seed ou ampliar os limites para forçar aprovação.", ""])
    position = lines.index("## Correções implementadas")
    lines[position:position] = gate_lines
    lines.extend(["Para finalizar um novo diretório da mesma comparação, executar `python scripts/finalize_calibration_audit.py --output-dir output/nova_auditoria` após o script de cálculo.", ""])
    tests_path = OUTPUT / "validation_evidence.json"
    if tests_path.exists():
        evidence = read(tests_path)
        lines.extend(["## Testes concluídos", "",
                      f"Suíte completa: {evidence['full_suite_passed']} aprovados e {evidence['full_suite_skipped']} opcionais pulados, em {num(evidence['full_suite_seconds'], 2)} s. Suíte focada: 144 aprovados. Teste complementar de reprodução de artefato local legado: aprovado. Os testes pulados dependem de um ZIP histórico ou do código original externo; as verificações escalares e do motor incorporado foram executadas.", "",
                      "Esses testes validam o código e a compatibilidade, não substituem uma confirmação científica em dados novos.", ""])
    input_baseline = summary.get("unchanged_input_baseline")
    if input_baseline:
        lines.extend(["## Baseline das poses fornecidas sem nova busca", "",
            f"A auditoria anterior das entradas, usada apenas como avaliação, encontrou {input_baseline['canonical_targets_above_threshold']}/{input_baseline['canonical_targets']} poses canônicas com DockQ ≥0,23, antes de nova busca. Repetindo a mesma pose como referência descritiva para os três algoritmos, isso equivale a {input_baseline['run_weighted_successes']}/147, contra 111/147 do FULL uniforme. Não são 147 observações independentes, e o custo de produzir essas entradas é desconhecido. Isso reforça que o estudo é de refinamento local de entradas frequentemente boas, não demonstra descoberta de encaixes a partir de poses arbitrárias. FULL não deve ser tratado como qualidade perfeita.", ""])
    manifest = read(OUTPUT / "implementation_manifest.json") if (OUTPUT / "implementation_manifest.json").exists() else None
    if manifest:
        lines.extend(["## Arquivos desta auditoria", ""])
        for relative in manifest["files"]:
            lines.append(f"- [{relative}](<{(ROOT / relative).as_posix()}>).")
        lines.append("")
    return "\n".join(lines)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUTPUT))
    args = parser.parse_args()
    OUTPUT = Path(args.output_dir).resolve()
    summary = read(OUTPUT / "audit_summary.json")
    for name in summary["cases"]:
        result = enrich_case(name)
        summary["cases"][name].update({key: result[key] for key in ("metrics", "official_gate_probes", "prediction_file_sha256", "operational", "paired_group_bootstrap")})
    audit_path = OUTPUT / "saved_job_audit.json"
    audit = read(audit_path)
    manifest = read(audit["provenance"][0]["manifest"])
    audit["input_root"] = manifest["plan"]["input_root"]
    audit["engine_snapshot"] = manifest["plan"]["engine"]
    write(audit_path, audit)
    spec_path = OUTPUT / "followup_5_spec.json"
    spec = read(spec_path)
    spec["input_root"], spec["engine_snapshot"] = audit["input_root"], audit["engine_snapshot"]
    write(spec_path, spec)
    baseline_path = ROOT / "output/calibrator_pilot_50_20261115/input_pose_audit/input_quality_summary.json"
    if baseline_path.exists():
        baseline = read(baseline_path)
        canonical = {value["complex_id"] for value in audit["representatives"].values()}
        rows = [row for row in baseline["rows"] if row["complex_id"] in canonical]
        count = sum(row["input_experimental_dockq"] >= .23 for row in rows)
        summary["unchanged_input_baseline"] = {
            "canonical_targets": len(rows), "canonical_targets_above_threshold": count,
            "run_weighted_successes": 3 * count, "used_as_predictor": False,
            "scope": "descriptive no-search baseline; input preparation cost unknown",
            "source_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest()}
    import platform
    from importlib.metadata import version
    summary["runtime_versions"] = {"python": platform.python_version(),
        **{package: version(package) for package in ("numpy", "pandas", "scikit-learn", "scipy")}}
    summary["reviewed_at_utc"] = datetime.now(timezone.utc).isoformat()
    write(OUTPUT / "audit_summary.json", summary)
    paths = ["src/docking_confidence/application/calibration_comparison.py", "src/docking_confidence/application/runtime.py",
             "src/docking_confidence/application/pilot_diagnostics.py", "src/docking_confidence/application/batch.py",
             "src/docking_confidence/application/training_dataset.py", "src/docking_confidence/artifact/schema.py",
             "src/docking_confidence/io/config.py", "configs/local_refinement_default.yaml",
             "configs/local_refinement_pilot.yaml", "configs/local_refinement_context_pilot.yaml",
             "scripts/run_calibration_audit.py", "scripts/finalize_calibration_audit.py",
             "tests/test_calibration_comparison.py", "tests/test_total_cascade_runtime.py", "tests/test_training_context.py",
             "docs/PROTOCOLO_AUDITORIA_CALIBRACAO.md", "docs/COMO_FUNCIONA_O_CALIBRADOR_HOJE.md", "README.md"]
    write(OUTPUT / "implementation_manifest.json", {"files": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in paths},
           "scope": "audit-specific files; repository already had unrelated changes",
           "prior_protocol_commitment": read(OUTPUT / "protocol_commitment.json"),
           "protocol_clarification": "planned groups include the input group with no valid reduced features; clarified before comparison execution; retrospective study, not preregistration"})
    (OUTPUT / "RELATORIO_COMPARATIVO.md").write_text(generate_report(summary), encoding="utf-8")
    print("Comparison report and frozen-prediction replay verified.")


if __name__ == "__main__":
    main()
