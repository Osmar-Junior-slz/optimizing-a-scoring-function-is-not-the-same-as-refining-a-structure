# Optimizing a scoring function is not the same as refining a structure

Código preservado do estudo **Optimizing a scoring function is not the same as
refining a structure: a random-sampling control in local refinement of protein
complexes**, submetido ao *Journal of Computer-Aided Molecular Design*.

Esta cópia registra o código existente em 22/11/2026. O estudo compara regiões
de refinamento local de complexos proteína-proteína, com algoritmo genético,
PSO, busca híbrida e controles aleatórios. FULL e REDUCED neste estudo são
regiões locais ao redor da pose fornecida.

## Conteúdo

- `src/`: motor DockingPP, aplicação, calibração, avaliação e interface.
- `scripts/`: coleta, controles, análises e reprodução das execuções.
- `tests/`: testes automatizados com exemplos sintéticos criados durante os testes.
- `configs/`: receitas e parâmetros de execução.
- `requirements-*.txt` e `pyproject.toml`: dependências.

O arquivo interno `factorial_v5.bundle` contém somente fontes/configurações
históricas usadas pelo motor para preservar a reprodução do estudo. Seus
arquivos são conferidos por SHA-256 durante o carregamento.

Dados moleculares, modelos treinados, registro de exposição dos dados,
resultados, manuscritos, PDFs, ambientes Python e caches não estão incluídos.
As análises e predições que dependem desses materiais exigem fornecê-los
separadamente. Referências a caminhos de estudos nos scripts são preservadas.
Este repositório não contém o pacote completo de reprodução dos dados do artigo.

## Instalação no Windows

Ambiente de origem: Python 3.12. Os fontes originais são preservados sem
normalização de finais de linha, pois o motor verifica hashes dos arquivos.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -c requirements-baseline.txt -e ".[dev]"
.\.venv\Scripts\python.exe iniciar_sistema.py --check
.\iniciar_sistema.cmd
```

O avaliador DockQ oficial usa um ambiente separado, conforme
`requirements-dockq-standard.txt`:

```powershell
.\.venv\Scripts\python.exe -m pip install --target .local/dockq_standard_runtime -r requirements-dockq-standard.txt
```

## Verificação e comandos

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m docking_confidence --help
.\.venv\Scripts\python.exe scripts/revision_random_control.py --help
.\.venv\Scripts\python.exe scripts/revision_convergence.py --help
.\.venv\Scripts\python.exe scripts/revision_seed_replay.py --help
```

As entradas científicas devem ser fornecidas separadamente. O repositório
preserva as implementações e não inicia coletas ao ser instalado.
