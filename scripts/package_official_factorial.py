"""Build a verified article delivery from the frozen decision study and its factorial extension.

Packaging preserves the original archive and its scientific observations.
Independent commands verify the package or reestimate the factorial effects
from its portable observations table.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / "output/entrega_final_titulo_original_20261119"
STUDY = ROOT / "runs/fatorial_dockq_20261119"
RESULTS = ROOT / "output/fatorial_dockq_20261119"
PREFIX = "evidencias/fatorial_dockq_20261119/"
ARCHIVE = "entrega_estudo_dockq_com_fatorial.zip"
OFFICIAL = "calibrador_gva_2025_hibrido_15_standard_20261119"
SCRIPTS = ("prepare_official_factorial_cohort.py", "collect_official_factorial.py",
           "analyze_standard_factorial.py", "audit_factorial_clashes.py", "package_official_factorial.py")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def encoded(record):
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8")


def read(path):
    return json.loads(Path(path).read_bytes())


def safe_member(name):
    p = PurePosixPath(name)
    require(bool(p.parts) and p.as_posix() == name and not p.is_absolute() and
            ".." not in p.parts and "\\" not in name and ":" not in name, f"Unsafe member: {name}")
    return name


def canonical_file(path, root):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(root.resolve(strict=True)) and path.is_file(), f"Unexpected source: {path}")
    return path


class Contents:
    def __init__(self, archive):
        self.archive = archive
        self.files = {}
        self.seen = set()

    def add(self, name, data, expected=None):
        safe_member(name)
        require(name.casefold() not in self.seen, f"Duplicate member: {name}")
        self.seen.add(name.casefold())
        digest = sha(data)
        require(expected is None or digest == expected, f"Changed bytes: {name}")
        # Source dates and local commitment times are recorded inside the records.
        # A stable container timestamp also allows a byte-for-byte retry.
        entry = zipfile.ZipInfo(name, (2026, 11, 19, 0, 0, 0))
        self.archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=4)
        self.files[name] = {"sha256": digest, "size_bytes": len(data)}

    def take(self, path, name, root, expected=None):
        self.add(name, canonical_file(path, root).read_bytes(), expected)


def add_decision_study(contents, delivery):
    original = delivery / "entrega_decisao.zip"
    v = read(delivery / "package_verification.json")
    require(file_sha(original) == v["archive_sha256"] and v["valid"], "Original archive integrity mismatch")
    stages = {"standard_fit_20261119", "decision_study_20261119", "decision_validation_stage_20261119",
              "decision_test_stage_20261119", "coletas", "verificacao_reduced", "decision_data_registry.json"}
    with zipfile.ZipFile(original) as source:
        names = source.namelist()
        require(len(names) == len(set(names)) == v["verified_members"], "Invalid original archive inventory")
        for name in sorted(names):
            parts = PurePosixPath(name).parts
            if parts[0] == "sistema":
                if len(parts) > 2 and parts[1] == "models" and parts[2] != OFFICIAL:
                    continue
            elif parts[0] != "evidencias" or len(parts) < 2 or parts[1] not in stages:
                continue
            contents.add(name, source.read(name), v["member_sha256"][name])
        proto = json.loads(source.read("evidencias/decision_study_20261119/protocol/decision_protocol.json"))
        require(contents.files[f"sistema/models/{OFFICIAL}/artifact.json"]["sha256"] ==
                proto["model"]["artifact_sha256"], "New model differs from the study commitment")
    for script in SCRIPTS:
        contents.take(ROOT/"scripts"/script, "sistema/scripts/"+script, ROOT/"scripts")
    return v["archive_sha256"]


def reviewed_title(source, qa):
    title = qa.get("title")
    require(isinstance(title, str) and title.strip(), "Article QA must identify the reviewed title")
    title_in_source = re.search(r"\\title\s*\{([^{}]+)\}", source)
    require(title_in_source and " ".join(title_in_source.group(1).split()) == title,
            "The title in the manuscript differs from the reviewed edition")
    return title


def add_article(contents, article, qa_path):
    qa = read(qa_path)
    require(qa.get("status") == "passed" or qa.get("valid") is True, "Final article QA has not passed")
    tex = article/"artigo_academico.tex"
    pdf = article/"artigo_academico.pdf"
    require(file_sha(tex) == qa["source_tex_sha256"] and file_sha(pdf) == qa["pdf_sha256"],
            "The supplied final QA record does not identify this TEX/PDF")
    source = tex.read_text(encoding="utf-8-sig")
    title = reviewed_title(source, qa)
    generator = article / "gerar_figuras.py"
    paths = {tex, pdf, article/"sbc-template.sty", article/"sbc.bst", article/"referencias.bib", generator}
    figures = article / "figuras"
    paths.update(p for p in figures.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".png"})
    for item in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source):
        safe_member(item)
        candidates = (article/item, article/"figuras"/item)
        graphic = next((p for p in candidates if p.is_file()), None)
        require(graphic is not None, f"Unresolved graphic: {item}")
        paths.add(graphic)
    todo = [source]
    while todo:
        for item in re.findall(r"\\(?:input|include)\{([^}]+)\}", todo.pop()):
            safe_member(item)
            path = article/item
            if not path.suffix:
                path = path.with_suffix(".tex")
            if path not in paths:
                paths.add(path)
                todo.append(canonical_file(path, article).read_text(encoding="utf-8-sig"))
    for p in sorted(paths):
        contents.take(p, "artigo/"+p.relative_to(article).as_posix(), article)
    provenance = article / "dados/proveniencia_figuras.json"
    contents.take(provenance, "artigo/dados/proveniencia_figuras.json", article)
    decision_sources = {
        "validation_decisions.csv": "decision_validation_stage_20261119/decision/validation_decisions.csv",
        "decision_policy.json": "decision_validation_stage_20261119/decision/decision_policy.json",
        "policy_candidates.json": "decision_validation_stage_20261119/decision/policy_candidates.json",
        "test_decisions.csv": "decision_test_stage_20261119/decision/test_decisions.csv",
        "final_evaluation.json": "decision_test_stage_20261119/decision/final_evaluation.json",
        "final_stage_summary.json": "decision_test_stage_20261119/stage_summary.json",
    }
    figure_inputs = read(provenance)["inputs"]
    seen = set()
    for item in figure_inputs:
        name = safe_member(item["local_path"].replace("\\", "/"))
        require(name.startswith("dados/"), f"Unexpected figure input: {name}")
        contents.take(article / name, "artigo/" + name, article, item["sha256"])
        basename = PurePosixPath(name).name
        if basename in decision_sources:
            require(contents.files["evidencias/" + decision_sources[basename]]["sha256"] == item["sha256"],
                    f"Figure input disagrees with the official decision stage: {basename}")
        seen.add(basename)
    require(set(decision_sources) | {"harmonized_results.json", "calibracao_validacao_bins.csv"} <= seen,
            "Figure provenance lacks inputs for the complete article")
    require(contents.files["artigo/dados/harmonized_results.json"]["sha256"] ==
            file_sha(RESULTS / "harmonized_results.json"), "Figure data differ from the current factorial study")
    audit_dir = ROOT / "tmp/pdfs/manuscrito_fatorial_dockq_20261119"
    audit_name = "auditoria_predicoes_teste_20261119.json"
    audit = audit_dir / audit_name
    report = read(audit)
    for relative, expected in report["source_files"].items():
        require(contents.files["evidencias/" + safe_member(relative)]["sha256"] == expected,
                f"Source for the before/after independent test comparison differs: {relative}")
    require(report["decision"]["valid"] == 140 and report["decision"]["threshold"] == 0.0906,
            "Independent comparison does not identify the frozen test rule")
    audit_member = "artigo/dados/" + audit_name
    if audit_member in contents.files:
        require(contents.files[audit_member]["sha256"] == file_sha(audit), "Two different independent-test audits")
    else:
        contents.take(audit, audit_member, audit_dir)
    audit_script = "auditar_predicoes_teste.py"
    contents.take(audit_dir / audit_script, "artigo/" + audit_script, audit_dir)
    contents.take(qa_path, "artigo/qa_final.json", ROOT)
    return {"title": title, "tex_sha256": file_sha(tex), "pdf_sha256": file_sha(pdf),
            "qa_sha256": file_sha(qa_path), "independent_test_audit": audit_member,
            "independent_test_audit_sha256": file_sha(audit)}


def add_harmonized_files(contents, primary_results, primary_file_hash):
    results = read(RESULTS / "harmonized_results.json")
    require(results["complete"] is True and results["primary_results_sha256"] == primary_file_hash and
            results["primary_observations_sha256"] == primary_results["observations_sha256"] and
            results["analysis_plan_sha256"] == primary_results["analysis_plan_sha256"],
            "Common eligibility and primary analysis do not refer to the same collection")
    for name, key in (("protocol/common_eligibility_addendum.json", "addendum_sha256"),
                      ("harmonized_observations.csv", "observations_sha256")):
        contents.take(RESULTS / name, PREFIX + name, RESULTS, results[key])
    contents.take(RESULTS / "harmonized_results.json", PREFIX + "harmonized_results.json", RESULTS)
    by_job = {}
    with (RESULTS / "harmonized_observations.csv").open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = row["complex_id"], row["job_id"]
            require(key not in by_job, f"Repeated harmonized observation: {key}")
            by_job[key] = row
    return by_job


def add_factorial(contents):
    results = read(RESULTS/"factorial_results.json")
    progress = read(RESULTS/"scoring_progress.json")
    design = read(RESULTS/"protocol"/"analysis_plan.json")["design"]
    expected = design["attempts"]
    require(results["complete"] is True and progress["planned"] == expected == 600 and
            progress["pending"] == 0 and not progress["issues"], "Incomplete factorial results")
    require(results["quality_metric"]["implementation_version"] == "2.1.3", "Unexpected DockQ")
    for name, hashkey in (("protocol/analysis_plan.json","analysis_plan_sha256"),
                          ("factorial_observations.csv","observations_sha256")):
        contents.take(RESULTS/name, PREFIX+name, RESULTS, results[hashkey])
    for name in ("scoring_progress.json","factorial_results.json"):
        contents.take(RESULTS/name, PREFIX+name, RESULTS)
    for name, hashkey in (("selection.json","selection_sha256"), ("planned_workers.json","planned_workers_sha256")):
        contents.take(STUDY/name, PREFIX+name, STUDY, results[hashkey])
    selection = read(STUDY/"selection.json")
    workers = read(STUDY/"planned_workers.json")
    collection = read(STUDY / "collection_summary.json")
    map_name = "collection_receipt_hashes.json"
    require(collection["evidence_map"] == map_name and collection["planned_runs"] == expected
            and collection["unresolved"] == collection["technical_failures"] == 0
            and all(collection[key] == results[key] for key in
                    ("analysis_plan_sha256", "selection_sha256", "planned_workers_sha256")),
            "Final native collection audit differs from the scientific results")
    contents.take(STUDY / "collection_summary.json", PREFIX + "collection_summary.json", STUDY)
    contents.take(STUDY / map_name, PREFIX + map_name, STUDY, collection["evidence_map_sha256"])
    receipt_map = read(STUDY / map_name)
    require(len(receipt_map) == expected, "Incomplete inventory of final native receipts and failures")
    audited_shards = {r["shard"]: r for r in collection["shards"]}
    require(workers["selection_sha256"] == results["selection_sha256"] and
            workers["analysis_plan_sha256"] == results["analysis_plan_sha256"], "Worker commitment mismatch")
    require(contents.files["sistema/scripts/prepare_official_factorial_cohort.py"]["sha256"] ==
            workers["preparation_script_sha256"], "Selection program differs from its original commitment")
    targets = {t["complex_id"]: t for t in selection["targets"]}
    for rel, expected_sha in selection["source_hashes"].items():
        relative = Path(rel).as_posix()
        destination = "sistema/" + safe_member(relative)
        if destination in contents.files:
            require(contents.files[destination]["sha256"] == expected_sha,
                    f"Base archive disagrees with selection input: {relative}")
        else:
            contents.take(ROOT/relative, destination, ROOT, expected_sha)
    batches = {}
    inputs = {}
    for shard in workers["shards"]:
        root = Path(shard["batch_dir"]).resolve(strict=True)
        require(root.is_relative_to(STUDY) and root.name == "batch", "Batch outside frozen collection")
        native = root.parent.name
        for f in ("experimental_protocol.json","batch_manifest.json","batch_status.json"):
            digest = audited_shards[native][Path(f).stem + "_sha256"]
            if f == "experimental_protocol.json":
                require(digest == shard["protocol_file_sha256"], f"Final shard commitment changed: {native}")
            contents.take(root/f, PREFIX+"coleta/"+native+"/"+f, STUDY, digest)
        for cid in shard["complex_ids"]:
            require(cid not in batches, "Duplicate complex in shards")
            batches[cid] = root
            for role in ("reference","receptor","ligand"):
                orig = Path(targets[cid][role+"_path"])
                staged = Path(shard["input_dir"])/cid/orig.name
                dest = PREFIX+"entradas/"+cid+"/"+orig.name
                contents.take(staged, dest, STUDY, targets[cid][role+"_sha256"])
                inputs[cid,role] = dest
    require(set(targets) == set(batches) and len(targets)==50, "Incomplete cohort")
    observations = list(csv.DictReader(io.StringIO((RESULTS/"factorial_observations.csv").read_text(encoding="utf-8-sig"))))
    require(len(observations) == expected, "Unexpected observation count")
    primary_file_hash = contents.files[PREFIX + "factorial_results.json"]["sha256"]
    harmonized = add_harmonized_files(contents, results, primary_file_hash)
    require(set(harmonized) == {(row["complex_id"], row["job_id"]) for row in observations},
            "Harmonized study does not cover all planned primary observations")
    index = []
    outcomes = Counter()
    cells = Counter()
    for row in observations:
        cid, jid = row["complex_id"], row["job_id"]
        state = row["outcome_state"]
        job = batches[cid]/"targets"/cid/"jobs"/jid
        prefix = PREFIX+"execucoes/"+cid+"/"+jid+"/"
        eligibility = harmonized[cid, jid]
        for native, harmonized_field in (("condition", "condition"), ("algorithm", "algorithm"),
                                         ("outcome_state", "native_outcome_state"),
                                         ("primary", "native_primary"), ("valid_pose", "native_valid_pose")):
            require(row[native] == eligibility[harmonized_field], f"Primary and common analyses differ: {cid}/{jid}/{native}")
        raw_gate_path = safe_member(eligibility["gate_evidence_path_relative_to_job"].replace("\\", "/"))
        gate_source = canonical_file(job / raw_gate_path, job)
        gate_hash = eligibility["gate_evidence_sha256"]
        require(file_sha(gate_source) == gate_hash, f"Gate evidence differs: {cid}/{jid}")
        native_file, hash_field = (("receipt.json", "receipt_file_sha256") if state == "scored"
                                  else ("error.json", "error_file_sha256"))
        native_relative = (job / native_file).relative_to(STUDY).as_posix()
        require(receipt_map[native_relative] == row[hash_field], f"Native collection receipt changed: {cid}/{jid}")
        info = {"complex_id": cid, "job_id": jid, "outcome": state,
                "reference": inputs[cid,"reference"], "reference_sha256": targets[cid]["reference_sha256"]}
        if state == "scored":
            receipt = job/"receipt.json"
            contents.take(receipt,prefix+"receipt.json",STUDY,row["receipt_file_sha256"])
            engine_result = read(receipt)["engine_result"]
            rel = safe_member(PurePosixPath(engine_result["complex_pose_pdb_path"].replace("\\","/")).as_posix())
            pose = job/rel
            contents.take(pose,prefix+rel,STUDY,row["pose_sha256"])
            official = RESULTS/"quality"/cid/jid/"standard_dockq.json"
            member = PREFIX + official.relative_to(RESULTS).as_posix()
            contents.take(official,member,RESULTS,row["official_record_sha256"])
            record = read(official)
            require(record["inputs"]["model_sha256"] == row["pose_sha256"] and
                    record["inputs"]["reference_sha256"] == info["reference_sha256"] and
                    record["quality_metric_sha256"] == results["quality_metric_sha256"],
                    f"Official quality record mismatch: {cid}/{jid}")
            require(canonical_file(job / engine_result["pose_artifact_path"], job) == gate_source,
                    f"Common eligibility is not bound to the winner in the receipt: {cid}/{jid}")
            gate_member = prefix + "best_pose.json"
            contents.take(gate_source, gate_member, job, gate_hash)
            info.update(receipt=prefix+"receipt.json", pose=prefix+rel, dockq=member)
        elif state == "no_valid_pose":
            contents.take(job/"error.json",prefix+"error.json",STUDY,row["error_file_sha256"])
            info.update(error=prefix+"error.json")
            if row.get("engine_result_sha256"):
                matching = [p for p in job.glob("engine/dockingpp_runs/*/output/result.json")
                            if file_sha(p) == row["engine_result_sha256"]]
                require(len(matching) == 1, f"Failure evidence unavailable: {cid}/{jid}")
                contents.take(matching[0],prefix+"engine_result.json",STUDY,row["engine_result_sha256"])
                info.update(engine_result=prefix+"engine_result.json")
            if gate_source == (job / "error.json").resolve():
                gate_member = info["error"]
            else:
                require(raw_gate_path.startswith("engine/dockingpp_runs/") and gate_source.name == "result.json"
                        and info.get("engine_result") and row["engine_result_sha256"] == gate_hash,
                        f"Missing common eligibility evidence for unsuccessful run: {cid}/{jid}")
                gate_member = info["engine_result"]
        else:
            raise ValueError(f"Unresolved observation: {cid}/{jid}")
        info.update(gate_evidence=gate_member, gate_evidence_sha256=gate_hash)
        index.append(info)
        cells[cid, row["algorithm"], row["condition"]] += 1
        outcomes[state] += 1
    expected_cells = {(cid,alg,cond) for cid in targets for alg in design["algorithms"] for cond in design["conditions"]}
    require(set(cells)==expected_cells and set(cells.values())=={1} and
            dict(outcomes) == {k:v for k,v in progress["states"].items() if v},
            "Cohort multiplicity or terminal counts differ from certified analysis")
    contents.add(PREFIX+"indice_arquivos.json", encoded(index))
    return {"planned": expected,"states":dict(outcomes),"selection_sha256":results["selection_sha256"],
            "analysis_plan_sha256":results["analysis_plan_sha256"], "results_sha256":primary_file_hash,
            "harmonized_results_sha256":contents.files[PREFIX + "harmonized_results.json"]["sha256"]}


GUIDE = """# {title}

O manuscrito está em `artigo/artigo_academico.pdf`. O mesmo diretório reúne o
arquivo LaTeX, a bibliografia, os estilos, as figuras citadas e a verificação
editorial vinculada ao PDF. O manifesto na raiz identifica o conteúdo de cada
arquivo por tamanho e SHA-256.

## Experimento fatorial com orçamento fixo

Em `evidencias/fatorial_dockq_20261119/`, `protocol/analysis_plan.json`
registra o delineamento e os contrastes. A seleção dos 50 complexos
está em `selection.json`. As quatro combinações de amplitude e inicialização
foram avaliadas por três métodos, com 7.600 avaliações da função objetivo em
cada uma das 600 buscas. Os registros `factorial_observations.csv` e
`factorial_results.json` apresentam, respectivamente, os resultados por busca
e os efeitos estimados.

Durante a coleta, a auditoria do motor identificou que a busca reduzida podia
rejeitar a pose vencedora por colisão, enquanto a ampliada podia conservá-la.
O adendo `protocol/common_eligibility_addendum.json` documenta quando se
definiu a análise complementar com o mesmo critério físico para as quatro
condições. Os dados por busca estão em `harmonized_observations.csv` e as
estimativas, em `harmonized_results.json`. O índice fornece o arquivo da
pose vencedora ou da falha que sustenta a classificação física de cada busca.

O índice `indice_arquivos.json` associa cada busca aos arquivos do pacote.
Para cada pose válida, contém os caminhos relativos do recibo, da pose final,
da estrutura experimental de referência e da avaliação oficial. As buscas
sem pose válida têm um registro de falha. As entradas estruturais congeladas
estão em `entradas/`. Os protocolos, manifestos e estados dos oito lotes
estão em `coleta/`. Os escores estruturais usados na análise são os dos
registros DockQ 2.1.3 em `quality/`. O encerramento da coleta está documentado
em `collection_summary.json`. O inventário `collection_receipt_hashes.json`
identifica por SHA-256 o recibo ou a falha registrada em cada uma das 600 buscas.

## Análise secundária da regra de seleção

Em `evidencias/`, a seleção e o protocolo estão em `decision_study_20261119`.
Os dados de desenvolvimento estão em `standard_fit_20261119`. A validação e o teste
independente têm diretórios próprios. Cada fase contém os escores DockQ e as
decisões por comparação. O modelo de 19 de novembro encontra-se em
`sistema/models/calibrador_gva_2025_hibrido_15_standard_20261119`. O limiar
de 0,0906 foi fixado na validação. No teste da regra, a preservação entre
aceitos foi de 94,55% e a cobertura, de 73,33%, abaixo da meta de 75%.

A comparação antes e depois da calibração usa os mesmos 140 pares no teste
independente. As estimativas de erro, ordenação e calibração por grupos
estão em `artigo/dados/auditoria_predicoes_teste_20261119.json`. Esse
arquivo identifica as seis fontes originais por caminho relativo e SHA-256.
Para reproduzir as estimativas em um novo arquivo, execute na raiz da
extração com NumPy, pandas e scikit-learn instalados.

```
python artigo/auditar_predicoes_teste.py --evidence-root evidencias --out conferido_teste.json
```

## Verificação e reprodução

Instale as dependências indicadas em `sistema/README.md` e
`sistema/requirements-dockq-standard.txt`. Os programas de preparação,
execução e análise do experimento fatorial estão em `sistema/scripts/`. A partir da
raiz do pacote extraído, este comando recalcula os três efeitos por método e
os intervalos de confiança nos dois critérios de elegibilidade com os dados
das mesmas 600 buscas.

```
python sistema/scripts/package_official_factorial.py --recompute-effects evidencias/fatorial_dockq_20261119
```

O programa confirma a correspondência entre os valores recalculados e os
publicados. Os registros
congelados conservam os caminhos absolutos da execução original. Para uma
instalação em outro computador, o índice fornece os arquivos correspondentes
dentro deste pacote. A avaliação de uma pose do experimento fatorial pode ser repetida
com a função `rescore_receipt_standard` de
`docking_confidence.evaluation.official_dockq`, informando o recibo e a
referência indicados no índice, um diretório novo de saída e o ambiente
compatível com DockQ 2.1.3. O arquivo do modelo estrutural está na posição
relativa que o recibo registra.

A descoberta completa de candidatos para uma nova amostragem depende do
acervo de origem, inclusive de estruturas que ficaram fora da seleção.
No estudo da regra, os arquivos estruturais brutos de desenvolvimento,
validação e teste devem ser obtidos da instalação original pelos identificadores
e SHA-256 preservados. O material distribuído inclui os escores e os dados
pareados dessas etapas. As estruturas selecionadas e poses finais válidas do
experimento fatorial estão incluídas neste ZIP.
"""


def build_archive(temp_archive, article, qa_path, delivery):
    with zipfile.ZipFile(temp_archive, "x", allowZip64=True) as archive:
        contents = Contents(archive)
        old_sha = add_decision_study(contents, delivery)
        article_record = add_article(contents,article,qa_path)
        factorial_record = add_factorial(contents)
        contents.add("README.md", GUIDE.format(title=article_record["title"]).encode("utf-8"))
        manifest = {
            "schema_version": "official_decision_and_factorial_delivery_v1",
            "parent_decision_archive_sha256": old_sha,
            "article": article_record, "factorial": factorial_record,
            "prior_validation_and_test": "decision stage results previously frozen; original models and scores are not refitted, recomputed or replaced by packaging",
            "source_pose_scope": "factorial input partners, references and final scorable poses included; original 19 November decision-stage raw structures remain external",
            "inventory_excludes": ["manifesto.json"], "files": dict(sorted(contents.files.items())),
        }
        contents.add("manifesto.json", encoded(manifest))


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        require(len(names) == len({safe_member(n).casefold() for n in names}), "Repeated or unsafe ZIP path")
        data = archive.read("manifesto.json")
        manifest = json.loads(data)
        require(manifest["schema_version"] == "official_decision_and_factorial_delivery_v1", "Unexpected archive schema")
        files = manifest["files"]
        require(set(names) == set(files) | {"manifesto.json"}, "Archive member inventory is incomplete")
        for name, expected in files.items():
            with archive.open(name) as reader:
                digest = hashlib.file_digest(reader, "sha256").hexdigest()
            require(digest == expected["sha256"] and archive.getinfo(name).file_size == expected["size_bytes"],
                    f"ZIP member differs from inventory: {name}")
        main = files["artigo/artigo_academico.pdf"]
        for member, field in (("artigo_academico.pdf", "pdf_sha256"),
                              ("artigo_academico.tex", "tex_sha256"), ("qa_final.json", "qa_sha256")):
            require(files["artigo/" + member]["sha256"] == manifest["article"][field],
                    f"Article hash differs: {member}")
        summary = manifest["article"]
        qa = json.loads(archive.read("artigo/qa_final.json"))
        source = archive.read("artigo/artigo_academico.tex").decode("utf-8-sig")
        require(reviewed_title(source, qa) == summary["title"], "Manifest title differs from the article")
        require(qa["source_tex_sha256"] == summary["tex_sha256"] and qa["pdf_sha256"] == summary["pdf_sha256"],
                "Article QA metadata identifies different files")
        require(files[safe_member(summary["independent_test_audit"])]["sha256"] == summary["independent_test_audit_sha256"],
                "Independent before/after calibration audit hash differs")
        for member, field in (("factorial_results.json", "results_sha256"),
                              ("harmonized_results.json", "harmonized_results_sha256")):
            require(files[PREFIX + member]["sha256"] == manifest["factorial"][field],
                    f"Scientific result hash differs: {member}")
    return {"schema_version":"official_decision_and_factorial_verification_v1",
            "verified": True, "archive_sha256":file_sha(path), "manifest_sha256":sha(data),
            "archive_size_bytes":path.stat().st_size, "verified_members":len(names),
            "article_pdf_sha256":main["sha256"], "results_sha256":manifest["factorial"]["results_sha256"]}


def verify_portable_gate(supplement, entry, row):
    from audit_factorial_clashes import assess_gate

    relative = PurePosixPath(safe_member(entry["gate_evidence"])).relative_to(PREFIX)
    artifact = canonical_file(supplement / Path(relative), supplement)
    require(file_sha(artifact) == row["gate_evidence_sha256"] == entry["gate_evidence_sha256"],
            "Structural evidence does not match the archived eligibility assessment")
    record = read(artifact)
    if row["native_outcome_state"] == "scored":
        meta = record["meta"]
        native_valid = 1
        native_primary = int(float(row["dockq"]) >= 0.23)
    else:
        require(row["native_outcome_state"] == "no_valid_pose" and record["final_pose_available"] is False
                and record["failure_reason"] == "no_valid_reduced_pose"
                and record["executed_pockets"] == ["local_refinement"] and len(record["per_pocket_results"]) == 1,
                "Cannot identify the recorded geometrical rejection")
        meta = record["per_pocket_results"][0]["best_pose_meta"]
        native_valid = native_primary = 0
    triggered = assess_gate(meta, row["condition"])["gate_triggered"]
    eligible = 1 - triggered
    require(triggered == int(row["gate_triggered"]) and (native_valid or triggered) and
            row["valid_pose"] == eligible * native_valid and row["primary"] == eligible * native_primary,
            f"Published eligibility differs from the native structural evidence: {row['complex_id']}/{row['job_id']}")


def recompute_effects(supplement):
    from analyze_standard_factorial import summarize

    plan_path = supplement / "protocol/analysis_plan.json"
    output = {"verified": True}
    for criterion in ("factorial", "harmonized"):
        results = read(supplement / f"{criterion}_results.json")
        table_path = supplement / f"{criterion}_observations.csv"
        require(results["complete"] is True and file_sha(table_path) == results["observations_sha256"] and
                file_sha(plan_path) == results["analysis_plan_sha256"], "Results, observations or plan differ")
        if criterion == "harmonized":
            require(file_sha(supplement / "protocol/common_eligibility_addendum.json") == results["addendum_sha256"]
                    and file_sha(supplement / "factorial_results.json") == results["primary_results_sha256"] and
                    file_sha(supplement / "factorial_observations.csv") == results["primary_observations_sha256"],
                    "Common eligibility is not bound to these data and this addendum")
            evidence = {(e["complex_id"], e["job_id"]): e for e in read(supplement / "indice_arquivos.json")}
        with table_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            row["valid_pose"] = int(row["valid_pose"])
            row["primary"] = int(row["primary"])
            if criterion == "harmonized":
                row["outcome_state"] = row["native_outcome_state"]
                verify_portable_gate(supplement, evidence[row["complex_id"], row["job_id"]], row)
        regenerated = summarize(rows, read(plan_path))
        require(all(results[k] == value for k, value in regenerated.items()),
                f"Recomputed {criterion} counts, effects or intervals differ from publication")
        output[criterion] = {"trials": len(rows), **regenerated}
    return output


def extract_supplement(archive_path, work_folder, final):
    """Install new supplement as a unit, validating a previous partial publication."""
    final.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entries = {k[len(PREFIX):]: v for k,v in json.loads(archive.read("manifesto.json"))["files"].items()
                   if k.startswith(PREFIX)}
        require(bool(entries), "Missing supplement")
        if final.exists():
            actual = {p.relative_to(final).as_posix():file_sha(p) for p in final.rglob("*") if p.is_file()}
            require(actual == {k:v["sha256"] for k,v in entries.items()}, "Existing loose supplement differs")
            return
        work_folder.mkdir()
        for relative, expected in entries.items():
            safe_member(relative)
            path = work_folder/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = archive.read(PREFIX+relative)
            require(sha(data)==expected["sha256"], f"Extraction mismatch: {relative}")
            path.write_bytes(data)
    work_folder.rename(final)


def install_new_file(source, destination, checksum):
    if destination.exists():
        require(file_sha(destination) == checksum, f"Another release already exists: {destination}")
        return
    # Hard-link the complete temp file at a new name. NTFS rejects collisions;
    # an interruption cannot expose a zero-length or partially written release.
    try:
        os.link(source, destination)
    except FileExistsError:
        require(file_sha(destination)==checksum, f"Competing release differs: {destination}")
    except OSError as exc:
        # Some Windows sync folders lack hard links. On Windows rename to an
        # existing path fails atomically, which preserves the same contract.
        if os.name != "nt" or exc.winerror not in {1, 5, 50}:
            raise
        try:
            os.rename(source, destination)
        except FileExistsError:
            require(file_sha(destination) == checksum, f"Competing release differs: {destination}")


def publish(article, qa_path, delivery):
    article,qa_path,delivery=(Path(p).resolve(strict=True) for p in (article,qa_path,delivery))
    loose_article=delivery/"conteudo/artigo"
    for name in ("artigo_academico.pdf","artigo_academico.tex","referencias.bib"):
        require(file_sha(article/name)==file_sha(loose_article/name), f"Synchronize the final article first: {name}")
    archive_path=delivery/ARCHIVE
    verification_path=delivery/"verificacao_estudo_dockq_com_fatorial.json"
    with tempfile.TemporaryDirectory(dir=delivery, prefix=".staging-fatorial-") as folder:
        stage = Path(folder)
        temp_archive = stage / "final.zip"
        build_archive(temp_archive,article,qa_path,delivery)
        verification=verify_archive(temp_archive)
        item = stage / "verification.json"
        item.write_bytes(encoded(verification))
        # Check preexisting destinations before installing any member of the release.
        for p,digest in ((archive_path,verification["archive_sha256"]),(verification_path,file_sha(item))):
            require(not p.exists() or file_sha(p)==digest, f"Another release already exists: {p}")
        extract_supplement(temp_archive,stage/"supplement",delivery/"conteudo"/PREFIX.rstrip("/"))
        install_new_file(temp_archive,archive_path,verification["archive_sha256"])
        install_new_file(item,verification_path,file_sha(item))
    return {"archive":str(archive_path), "verification":str(verification_path), **verification}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-only",type=Path,metavar="ZIP")
    mode.add_argument("--recompute-effects", type=Path, metavar="SUPPLEMENT")
    mode.add_argument("--publish",action="store_true")
    parser.add_argument("--destination",type=Path,default=DELIVERY)
    parser.add_argument("--article-dir",type=Path,default=DELIVERY/"conteudo/artigo")
    parser.add_argument("--article-qa",type=Path)
    args=parser.parse_args()
    if args.verify_only:
        result=verify_archive(args.verify_only.resolve(strict=True))
        external=args.verify_only.parent/"verificacao_estudo_dockq_com_fatorial.json"
        if external.exists():
            require(read(external)==result,"External archive verification record differs")
    elif args.recompute_effects:
        result = recompute_effects(args.recompute_effects.resolve(strict=True))
    else:
        qa_path = args.article_qa or args.article_dir / "qa_final.json"
        result=publish(args.article_dir,qa_path,args.destination)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
