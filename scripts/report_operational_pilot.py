"""Analyze saved operational receipts only; never executes docking or trains models."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from docking_confidence.application.calibrated_batch import configured_calibrator_path
from run_operational_pilot import summarize, protected_snapshot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study",required=True)
    parser.add_argument("--output-dir",required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    study=Path(args.study).resolve();out=Path(args.output_dir).resolve();out.mkdir(parents=True,exist_ok=True)
    data=summarize(study)
    paired=[p for p in data["pairs"] if p["valid_pair"]]
    by_algorithm={}
    for algorithm in ("ag","pso","hybrid"):
        pp=[p for p in paired if p["algorithm"]==algorithm]
        ids={p["complex_id"] for p in pp}
        r=sum(p["reduced_wall_sec"] for p in pp);f=sum(p["full_wall_sec"] for p in pp)
        metrics={name:{mode:statistics.mean(row[name] for row in data["rows"]
            if row["algorithm"]==algorithm and row["complex_id"] in ids and row["protocol"]==mode)
            for mode in ("reduced","full")} for name in ("dockq","fnat","irmsd","lrmsd")}
        by_algorithm[algorithm]={"valid_pairs":len(pp),"reduced_wall_sec":r,"full_wall_sec":f,
            "saving_fraction":1-r/f,"metrics":metrics}
    full_cost=data["totals"]["full"]["all_engine_call_wall_sec"]
    reduced_cost=data["totals"]["reduced"]["all_engine_call_wall_sec"]
    oracle=sum(p["reduced_wall_sec"]+(0 if p["preserved"] else p["full_wall_sec"]) for p in data["pairs"])
    model_path=configured_calibrator_path()
    before=json.loads((study / "protected_before.json").read_text())
    after=protected_snapshot(root)
    changed=[name for name,digest in before.items() if after.get(name)!=digest]
    prep=json.loads((out / "training_preparation/dataset_build_report.json").read_text(encoding="utf-8"))
    assessment={"by_algorithm":by_algorithm,"paired_mean_dockq_reduced":statistics.mean(p["reduced_dockq"] for p in paired),
        "paired_mean_dockq_full":statistics.mean(p["full_dockq"] for p in paired),
        "all_attempts_reduced_time_saving_fraction":1-reduced_cost/full_cost,
        "observed_engine_cascade_increase_fraction":(reduced_cost+full_cost)/full_cost-1,
        "oracle_engine_cost_sec":oracle,"oracle_saving_sec":full_cost-oracle,
        "oracle_scope":"retrospective zero-violation acceptance of all preserved REDUCED; unavailable in deployment; excludes overhead",
        "configured_artifact_path":str(model_path),"configured_artifact_exists":model_path.is_file() if model_path else False,
        "calibrator_validated_for_new_regime":False,"new_regime_brier":None,"new_regime_ece":None,
        "missing_probability_metrics_reason":"no installed compatible model or independent new-regime calibration/test data",
        "training_preparation":prep["preview"],"protected_files":{"count":len(before),"changed":changed}}
    (out / "analysis.json").write_text(json.dumps(assessment,indent=2,ensure_ascii=False),encoding="utf-8")
    for name,rows in (("all_jobs.csv",data["rows"]),("matched_pairs.csv",data["pairs"])):
        fields=list(dict.fromkeys(k for row in rows for k in row))
        with (out/name).open("w",encoding="utf-8-sig",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    algorithm_table="\n".join(f"| {alg.upper()} | {v['valid_pairs']} | {v['reduced_wall_sec']:.2f} | {v['full_wall_sec']:.2f} | {100*v['saving_fraction']:.1f}% | {v['metrics']['dockq']['reduced']:.4f} | {v['metrics']['dockq']['full']:.4f} |" for alg,v in by_algorithm.items())
    structural_table="\n".join(f"| {alg.upper()} | {v['metrics']['fnat']['reduced']:.4f} / {v['metrics']['fnat']['full']:.4f} | {v['metrics']['irmsd']['reduced']:.3f} / {v['metrics']['irmsd']['full']:.3f} | {v['metrics']['lrmsd']['reduced']:.3f} / {v['metrics']['lrmsd']['full']:.3f} |" for alg,v in by_algorithm.items())
    pairs_table="\n".join(f"| {p['complex_id']} | {p['algorithm']} | {p['reduced_dockq']:.4f} | {p['full_dockq']:.4f} | {'Sim' if p['preserved'] else 'Não'} |" if p['valid_pair'] else f"| {p['complex_id']} | {p['algorithm']} | Falha de geometria | {p['full_dockq']:.4f} | Não |" for p in data['pairs'])
    text=f'''# Implementação da cascata operacional e resultados medidos

A cascata operacional agora usa REDUCED com 5.700 avaliações e FULL com 7.600,
população 100 e os limites de movimento reduzidos existentes. O Venn–Abers foi
preservado como decisor. A configuração científica do orientador continua
separada, com quatro condições e orçamento igual. Nenhum resultado histórico
foi substituído. A integridade dos {len(before)} arquivos protegidos foi conferida:
{len(changed)} alterações.

## Leitura simples dos resultados

O REDUCED fez 25% menos avaliações. Na soma dos 12 pares com poses válidas nos
dois modos, sua chamada ao motor foi {100*data['paired_reduced_time_saving_fraction']:.1f}% mais curta.
Isso não aconteceu em todos os algoritmos: o híbrido foi mais lento na soma de
seus pares válidos. A cascata completa ainda não mostrou economia, pois falta
um calibrador compatível e validado que possa dispensar FULL.

| Configuração | REDUCED | FULL |
|---|---:|---:|
| Operacional | 5.700 avaliações | 7.600 avaliações |
| Experimento do orientador | 7.600 avaliações | 7.600 avaliações |
| População no novo perfil | 100 | 100 |
| Translação por eixo | ±2,5 Å | ±5 Å |
| Rotação por ângulo | ±12,5° | ±25° |

## Desenho executado

Foram usados os cinco complexos já selecionados por hash no plano anterior:
1axi_BD, 1bgx_AC, 1d4v_BF, 1dj7_AB e 1e96_AB. Os hashes de entrada foram
conferidos antes de executar. Este piloto novo não executou nem alterou o
plano anterior de 1.900/3.800/7.600 avaliações.

Trinta jobs novos: cinco complexos × AG/PSO/híbrido × REDUCED/FULL, seed 1,
inicialização uniforme e execução serial REDUCED antes de FULL. Não executei
novamente as 50 proteínas nem iniciei o calibrador definitivo de 200.
Não houve ajuste de algoritmo, pontuação, regra estrutural ou limiar após os
resultados. Tempos são de parede, sujeitos a carga, ordem de execução e caches
da máquina; não são segundos de CPU nem estimativas de ganho populacional.
Uma seed e cinco complexos oferecem evidência exploratória, sem teste
estatístico confirmatório. Os três algoritmos por complexo são correlacionados.

## Avaliações e custo real

Todos os 15 REDUCED completaram 5.700 avaliações, inclusive os três cuja pose
final foi rejeitada. Todos os 15 FULL completaram 7.600. O contador inclui novas
chamadas da função objetivo geométrica, inclusive chamadas internas do híbrido;
não inclui cache, rescore final, importações nem tarefas de validação.

Houve 27 sucessos e três falhas REDUCED de geometria, todas em 1bgx_AC. Os FULL
correspondentes rodaram normalmente. Falha nunca foi convertida em pose válida.
Pose válida significa que passou nas verificações de saída e geometria; não
significa automaticamente bom DockQ.

| Medida | REDUCED | FULL |
|---|---:|---:|
| Jobs planejados | 15 | 15 |
| Poses válidas | 12 | 15 |
| Poses válidas com DockQ ≥0,23 sobre o total planejado | 9/15 | 8/15 |
| Tempo de todas as chamadas ao motor, incluindo falhas | {reduced_cost:.2f} s | {full_cost:.2f} s |
| Tempo dos mesmos 12 pares válidos | {data['paired_reduced_wall_sec']:.2f} s | {data['paired_full_wall_sec']:.2f} s |
| DockQ médio nesses mesmos pares | {assessment['paired_mean_dockq_reduced']:.4f} | {assessment['paired_mean_dockq_full']:.4f} |

Incluindo todas as tentativas, a soma de tempo REDUCED foi
{100*assessment['all_attempts_reduced_time_saving_fraction']:.1f}% menor. Comparando somente
os pares válidos, foi {100*data['paired_reduced_time_saving_fraction']:.1f}% menor. Os dois
números usam denominadores diferentes e não são intercambiáveis.

## Diferenças por algoritmo

Cada linha abaixo usa os mesmos quatro complexos com resultados válidos nos
dois modos. Economia negativa significa REDUCED mais lento.

| Algoritmo | Pares válidos | REDUCED s | FULL s | Economia de tempo | DockQ REDUCED | DockQ FULL |
|---|---:|---:|---:|---:|---:|---:|
{algorithm_table}

Médias estruturais nos mesmos pares; cada célula mostra REDUCED / FULL.
Fnat maior é melhor; iRMSD e lRMSD menores são melhores, em angstrom.

| Algoritmo | Fnat | iRMSD Å | lRMSD Å |
|---|---:|---:|---:|
{structural_table}

A média de DockQ favorável não demonstra que todas as poses foram preservadas.
A regra continua DockQ REDUCED ≥0,23 e perda FULL−REDUCED ≤0,05: oito dos 15
casos planejados foram preservados. Aceitar cegamente os 12 REDUCED válidos
produziria quatro violações dessa regra; as três falhas exigem FULL.

## Decisões e situação do calibrador

O vínculo instalado aponta para `{model_path}`. Esse arquivo não está
presente. O novo regime não possui modelo validado instalado. Foram registradas
12 decisões RUN_FULL por ausência de calibrador e três por falha REDUCED.
Nenhuma probabilidade foi inventada. Nenhum FULL foi evitado.

O custo observado das chamadas de REDUCED mais FULL foi
{data['all_engine_cascade_sec']:.2f} s, contra {full_cost:.2f} s na soma das chamadas FULL
do mesmo piloto. Não foi executado outro lote FULL-only separado; essa soma é
a referência de custo pareada, sujeita ao efeito da ordem e dos caches.
É um acréscimo de {100*assessment['observed_engine_cascade_increase_fraction']:.1f}%
antes de contabilizar overhead adicional. Não é ganho de uma cascata calibrada.
Os recibos registram preparação e decisão separadamente; o conjunto ainda
inclui avaliação posterior contra referência, que não deve ser confundida com
tempo de uma produção sem referência.

Como diagnóstico retrospectivo, aceitar todos os oito casos realmente
preservados, conhecendo seus rótulos, ainda custaria {oracle:.2f} s de chamadas
ao motor: {oracle-full_cost:.2f} s a mais que FULL. Esse oráculo não é aplicável
em produção e não constitui prova de impossibilidade em outras proteínas ou
com tolerância a erro diferente. Mostra que este piloto não comprova a economia
final, mesmo que algumas buscas individuais sejam mais rápidas.

O contrato do calibrador compara os perfis imutáveis de treinamento com os de
execução. REDUCED treinado em 7.600 não autoriza automaticamente REDUCED em
5.700. Incompatibilidade explícita de orçamento implica RUN_FULL e impede
pontuar o modelo antigo na cascata nova. Modelo compatível continua passando
pelo `CalibratedBatchGate`, Venn–Abers, domínio e regras de autorização existentes.

Foi preparado um dataset novo, com hashes e proveniência válidos: 12 pares,
quatro complexos independentes com pares, oito rótulos positivos e quatro
negativos. Isso valida a preparação dos arquivos, não a calibração. Não treinei
nem recalibrei um modelo com esses poucos dados. A receita exige, entre outros
critérios, 20 complexos independentes na calibração probabilística, cinco
complexos positivos e quatro negativos nessa etapa, e 30 avaliados/autorizados
na avaliação da política. As divisões precisam ser disjuntas por complexo e
considerar homologia. São necessários novos dados nesse mesmo regime,
treinamento e Venn–Abers novos, validação dos limiares e teste independente.
Brier e ECE no novo regime permanecem indisponíveis, por falta de modelo e
teste compatíveis; não foram reaproveitados números do orçamento antigo.

## Parada antecipada e diagnóstico do híbrido

Não ativei limite arbitrário de score nem estagnação. O score geométrico não é
o DockQ experimental. O Venn–Abers existente avalia poses finais; uma política
que consulte poses intermediárias precisa de dados e controle do risco da
seleção sequencial. Até haver evidência, o encerramento bem-sucedido é por orçamento.

Para investigar o híbrido mais lento, foram feitas duas reexecuções diagnósticas
instrumentadas de 1axi_BD, fora das 30 medições principais. O perfil confirmou
5.700/7.600 chamadas reais de score barato, cache de avaliador construído uma
vez em cada braço e quatro/duas chamadas de rescore final. Custos de importação
e consultas geométricas também diferiram. Esses registros mostram que uma
avaliação não tem custo constante, mas não isolam uma causa única da diferença
de tempo: instrumentação, ordem e caches da máquina interferem. Não misturei
esses tempos com os do piloto nem limitei o rescore para melhorar artificialmente
a velocidade. Evidências em `profiling/profile_summary.json` e arquivos `.prof`.

## Arquivos e testes

Arquivos novos:
- `configs/operational_cascade.yaml`: receita operacional.
- `configs/local_refinement_operational.yaml`: receita de treinamento independente.
- `src/docking_confidence/application/operational.py`: validação, contrato de treinamento e fallback conservador.
- `scripts/run_operational_pilot.py`: seleção, execução serial e auditoria de custos.
- `scripts/report_operational_pilot.py`: análise reproduzível dos resultados salvos.
- `tests/test_operational_cascade.py`: testes específicos do regime novo.
- `docs/CASCATA_OPERACIONAL.md`: funcionamento, reprodução e limites.

Arquivos existentes ajustados nesta entrega:
- `domain/execution.py`, `application/protocols.py` e `application/planning.py`: novo perfil, sem alterar a receita histórica/fatorial.
- `application/batch.py` e `adapters/dockingpp_archive_engine.py`: distinguir regimes e encaminhar o orçamento ao motor existente.
- `application/calibrated_batch.py`: diagnóstico explícito de orçamento diferente do treinamento.
- `application/training_dataset.py`: impedir mistura de regimes de treinamento.
- `application/batch_launcher.py`, `cli.py`, `ui/batch_page.py` e `ui/training_page.py`: expor operação e treinamento separados.
- `README.md`: indicar a documentação nova.

Os caminhos de código abreviados acima ficam em `src/docking_confidence`.
O workspace já tinha alterações anteriores; esta lista não atribui todo o diff
acumulado desta pasta à presente entrega. Nenhum arquivo do motor nativo
`src/dockingpp` foi modificado neste ajuste.

| Verificação executada | Resultado |
|---|---|
| Suíte completa `python -m pytest -q` | 578 aprovados, 2 opcionais ignorados |
| Operacional, launcher e interface após ajustes finais | 49 aprovados |
| Regressão adicional de contagem das falhas | 1 aprovado |
| `git diff --check` | Sem erros |
| Medições reais em cinco complexos | 30 jobs, 27 sucessos e 3 rejeições geométricas REDUCED |
| Preservação de arquivos científicos e registros históricos | {len(before)} conferidos, {len(changed)} alterados |

As contagens de testes de linhas diferentes se sobrepõem; não devem ser somadas.
Os dois opcionais dependem de fontes/ZIPs externos antigos; o motor integrado
foi testado. Os testes cobrem contadores dos três algoritmos, última geração
parcial, isolamento do fatorial, bloqueio do modelo antigo, fallback sem modelo,
aceitação somente autorizada, retomada e interface. JUnit e manifestos ficam
nesta pasta. A limpeza removeu somente três imagens temporárias da revisão de
um PDF já entregue, total de 743.831 bytes; o PDF final foi preservado. Não
removi dados, experimentos, fontes, testes nem documentação relevante.

## Recomendação

Manter os dois protocolos separados. Usar este perfil operacional para coletar
novos pares e validar um calibrador próprio antes de autorizar dispensas de FULL.
O requisito de 25% menos avaliações foi implementado; economia de tempo não é
uniforme entre algoritmos e a economia da cascata completa não foi demonstrada.
Enquanto faltar um modelo autorizado, FULL isolado evita o custo adicional de
uma cascata que sempre faz fallback. Não substituir Venn–Abers, relaxar qualidade
ou ativar parada antecipada sem evidência adicional.

## Resultados por par

| Complexo | Algoritmo | DockQ REDUCED | DockQ FULL | Preservado |
|---|---|---:|---:|---|
{pairs_table}

As entradas são poses fornecidas em refinamento local. Os resultados não
demonstram docking cego a partir de parceiros não ligados, nem generalização
para 200 proteínas novas. A seleção já havia sido registrada antes deste piloto,
mas os mesmos complexos já foram vistos no desenvolvimento anterior.

## Reprodução e fontes

Execução: `python scripts/run_operational_pilot.py --input-dir PASTA --output-dir PASTA_NOVA`.
Relatório sem docking: `python scripts/report_operational_pilot.py --study PASTA_DO_PILOTO --output-dir PASTA_DO_RELATORIO`.
O relatório pressupõe `training_preparation/dataset_build_report.json`, gerado
com `build_training_dataset(..., profile="operational")`.
Receitas e comandos completos estão em `docs/CASCATA_OPERACIONAL.md`.

Dados brutos: `runs/operational_budget_pilot5_20261115/batch`.
Consolidação: `runs/operational_budget_pilot5_20261115/comparison.json`.
Dados por job e por par: `all_jobs.csv` e `matched_pairs.csv` nesta pasta.
Limpeza: `cleanup_manifest.json`. Preparação de treinamento: subpasta `training_preparation`.

Referências: [Vovk e Petej 2014](https://www.auai.org/uai2014/proceedings/individuals/166.pdf)
fundamentam a calibração Venn–Abers sob as hipóteses de distribuição do trabalho.
[Jazbec e colaboradores 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ea5a63f7ddb82e58623693fd1f4933f7-Abstract-Conference.html)
estudam controle de risco em saídas antecipadas de redes neurais. A transferência
dessas ideias ao docking é uma proposta para validação, não resultado desses
artigos sobre este sistema.
'''
    (out/'RELATORIO_IMPLEMENTACAO.md').write_text(text,encoding='utf-8')
    print(out/'RELATORIO_IMPLEMENTACAO.md')


if __name__=="__main__":main()
