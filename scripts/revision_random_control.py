"""Uniform random control on the archived article cohort, with official winner DockQ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import _atomic_write_json, _BatchFileLock, _file_sha256
from docking_confidence.evaluation.official_dockq import (
    canonical_sha256, evaluate_standard_complex, official_dockq_preflight, quality_metric_descriptor,
)
from docking_confidence.evaluation.reference_dockq import prepare_reduced_pose
from dockingpp.application.config import Config
from dockingpp.data.io import load_config, load_peptide, load_receptor
from dockingpp.domain.models import RunResult
from dockingpp.pipeline.execucao.busca_execucao._rescoring_final import ServicoRescoringFinal
from dockingpp.pipeline.execucao.busca_execucao.rescoring_artefatos import _persist_best_pose
from dockingpp.pipeline.logging import RunLogger
from dockingpp.pipeline.run import _build_global_pocket
from dockingpp.scoring.cheap import score_pose_cheap
from dockingpp.search.algoritmos.espaco_rigido import RigidSearchSpace

EXTENTS = ("reduced", "expanded")
BUDGET = 7600
DEFAULT_STUDY = ROOT / "runs/fatorial_dockq_20261119"
DEFAULT_DEPS = ROOT / ".local/dockq_standard_runtime"


def read(path):
    return json.loads(path.read_bytes())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verified_record(path):
    value = read(path)
    digest = value.pop("record_sha256")
    require(canonical_sha256(value) == digest, f"Registro com integridade inválida: {path}")
    return value


def save_record(path, record):
    _atomic_write_json(path, {**record, "record_sha256": canonical_sha256(record)})


def engine_identity():
    # This script imports dockingpp directly from src. Validate that very tree;
    # the default engine adapter executes the separately frozen v5 snapshot.
    engine = DockingPPLocalEngine(seed_lineage=True)
    descriptor = engine.audit_descriptor()
    require(engine.module_root.resolve() == (ROOT / "src").resolve(), "Origem do motor divergente.")
    return descriptor["source_manifest_sha256"]


def inputs_for_study(study, limit):
    """Recover archived model copies by ID, in the registered selection order."""
    selection = read(study / "selection.json")
    workers = read(study / "planned_workers.json")
    require(workers["selection_sha256"] == _file_sha256(study / "selection.json"),
            "A seleção difere do plano de coleta.")
    targets = selection["targets"]
    require(len(targets) == selection["target_count"] == len({t["complex_id"] for t in targets}),
            "Seleção incompleta ou com identificadores repetidos.")
    require(selection["seeds"] == [1], "O controle se destina ao fatorial do artigo com seed 1.")
    if limit is not None:
        require(0 < limit <= len(targets), "Limite de complexos fora da seleção.")
        targets = targets[:limit]
    selected = {t["complex_id"]: t for t in targets}
    cases = {}
    for shard in workers["shards"]:
        # Relocate a frozen study using its shard names, even when its original
        # input_root path belongs to a different installation.
        batch = study / Path(shard["batch_dir"]).parent.name / "batch"
        require(batch.resolve().is_relative_to(study), "Diretório de coleta inválido.")
        manifest = read(batch / "batch_manifest.json")
        require(canonical_sha256(manifest["plan"]) == manifest["plan_sha256"], "Plano de lote inconsistente.")
        for target in manifest["plan"]["targets"]:
            cid = target["complex_id"]
            if cid not in selected:
                continue
            require(Path(cid).name == cid and cid not in cases, "Identificador de complexo inválido ou repetido.")
            paths = {}
            input_dir = (batch.parent / "inputs").resolve()
            for kind in ("receptor", "ligand", "reference"):
                key = kind + "_sha256"
                path = (input_dir / target[kind + "_path"]).resolve(strict=True)
                require(path.is_relative_to(input_dir) and target[key] == selected[cid][key] == _file_sha256(path),
                        f"Entrada divergente: {cid}/{kind}")
                paths[kind] = path
            configs = {}
            for extent in EXTENTS:
                directory = batch / "targets" / cid / "jobs" / f"ag-{extent}-uniform-seed1"
                receipt = directory / "receipt.json"
                if receipt.exists():
                    evidence = read(receipt)
                    require(evidence["record_sha256"] == canonical_sha256(
                                {k: v for k, v in evidence.items() if k != "record_sha256"}),
                            f"Recibo alterado: {receipt}")
                    config = (directory / evidence["engine_result"]["resolved_config_path"]).resolve(strict=True)
                    require(config.is_relative_to(directory.resolve()), f"Configuração fora do ensaio: {config}")
                else:
                    # A structural rejection has no successful receipt.
                    error = read(directory / "error.json")
                    require(error.get("failure_reason") == "no_valid_reduced_pose", f"Falha técnica: {directory}")
                    configs_found = list(directory.glob("engine/dockingpp_runs/*/resolved_config.yaml"))
                    require(configs_found and len({_file_sha256(p) for p in configs_found}) == 1,
                            f"Configuração ausente ou ambígua: {directory}")
                    config = configs_found[0]
                configs[extent] = config
            cases[cid] = (target, paths, configs)
    require(set(cases) == set(selected), "A coorte selecionada não corresponde aos lotes arquivados.")
    return [cases[t["complex_id"]] for t in targets]


def uniform_pool(space, count, keep, draws=None):
    """Retain the highest cheap scores across independent draws, earliest tie first."""
    require(count > 0 and 0 < keep <= count, "Número de candidatos inválido.")
    pool = []
    for index in range(count):
        pose = space.random_pose()
        score = space.evaluate_pose_score(pose)
        require(math.isfinite(score), "Avaliação geométrica não finita.")
        pose.score_cheap = score
        pose.meta.update(pocket_id="local_refinement", draw=index + 1)
        if draws is not None:
            draws["vectors"][index] = pose.meta["solution_vector"]
            draws["cheap_scores"][index] = score
        entry = (score, -index, pose)
        if len(pool) < keep:
            heapq.heappush(pool, entry)
        elif entry[:2] > pool[0][:2]:
            heapq.heapreplace(pool, entry)
    return [entry[2] for entry in sorted(pool, reverse=True)]


def draw_winner(cfg, receptor, ligand, count, draws=None):
    pocket = _build_global_pocket(receptor, cfg)
    pocket.id = "local_refinement"
    pocket.meta.update(detection_mode="supplied_pose_local_refinement", input_relative_placement_used=True)
    # Same generator and first pocket child used by the original population engine.
    random = np.random.default_rng(np.random.SeedSequence(cfg.seed).spawn(2)[1])
    space = RigidSearchSpace(cfg=cfg, rng=random, pocket=pocket, base_coords=ligand,
                             score_cheap_fn=score_pose_cheap, max_trans=cfg.max_trans, max_rot_deg=cfg.max_rot_deg)
    pool = uniform_pool(space, count, min(count, cfg.rescoring_candidate_pool), draws)
    result = RunResult(best_pose=pool[0], population=None, metrics={"rescoring_candidates": pool})
    logger = RunLogger()
    cfg.expensive_logger = logger
    rescoring = ServicoRescoringFinal().aplicar(cfg=cfg, receptor=receptor, peptide=ligand, result=result, logger=logger)
    require(space.actual_eval_count == count and rescoring["rescoring_applied"], "Avaliações ou rescoring incompletos.")
    return result.best_pose, rescoring


def thresholds(extent, clash, dockq):
    native_valid = not (extent == "reduced" and clash)
    if native_valid:
        require(dockq is not None and math.isfinite(dockq) and 0 <= dockq <= 1, "Falta DockQ oficial.")
    primary = int(native_valid and dockq >= 0.23)
    return {"valid_pose": int(native_valid), "success": primary, "success_common_gate": int(primary and not clash)}


def draw_and_save(case, extent, output):
    target, paths, configs = case
    cfg = Config.model_validate(load_config(str(configs[extent])))
    audit = cfg.docking_confidence_audit
    require(cfg.initialization == "uniform" and cfg.docking_task == "local_refinement" and cfg.seed == 1
            and cfg.search_extent == extent and cfg.search_space_mode == ("reduced" if extent == "reduced" else "full")
            and cfg.max_trans == 5 and cfg.max_rot_deg == 25
            and cfg.local_refinement_reduced_scale == (.5 if extent == "reduced" else 1.)
            and cfg.objective_evaluation_budget_per_pocket == BUDGET
            and audit["receptor_sha256"] == target["receptor_sha256"] and audit["peptide_sha256"] == target["ligand_sha256"],
            "A configuração arquivada difere do desenho do artigo.")
    draws = {"vectors": np.empty((BUDGET, 6), dtype="<f8"), "cheap_scores": np.empty(BUDGET, dtype="<f8")}
    winner, rescore = draw_winner(cfg, load_receptor(str(paths["receptor"])), load_peptide(str(paths["ligand"])),
                                  BUDGET, draws)
    gate = winner.meta["final_hard_clash_gate"]
    require(gate["enabled"] is True and gate["threshold"] == 1.2 and type(gate["triggered"]) is bool,
            "Regra estérica final divergente.")
    artifact = _persist_best_pose(out_dir=str(output), best_pose=winner, engine_name="iid_uniform_control",
                                  search_space_mode=cfg.search_space_mode)
    require(artifact["final_pose_available"], "Falha ao salvar a vencedora pré-avaliação.")
    require(artifact["pose_output_path"] == "artifacts/best_pose.json", "Caminho da pose inesperado.")
    temporary = output / ".draws.npz.tmp"
    with temporary.open("wb") as handle:
        np.savez(handle, **draws)
    os.replace(temporary, output / "draws.npz")
    row = {"complex_id": target["complex_id"], "extent": extent, "seed": cfg.seed, "objective_evaluations": BUDGET,
            "draw_selected": winner.meta["draw"], "score_cheap": winner.score_cheap, "final_rescore": winner.final_rescore,
            "candidates_rescored": rescore["rescored_candidates_count"], "hard_clash": gate["triggered"]}
    return row, hashlib.sha256(draws["vectors"].tobytes(order="C")).hexdigest()


def trial(case, extent, output, deps, protocol_hash):
    target, paths, _ = case
    pose_file = output / "artifacts/best_pose.json"
    checkpoint = output / "selected.json"
    if not checkpoint.exists():
        row, vectors_sha256 = draw_and_save(case, extent, output)
        save_record(checkpoint, {"row": row, "winner_sha256": _file_sha256(pose_file), "protocol_sha256": protocol_hash,
                                 "draws_sha256": _file_sha256(output / "draws.npz"), "vectors_sha256": vectors_sha256})
    selected = verified_record(checkpoint)
    row = selected["row"]
    require(selected["protocol_sha256"] == protocol_hash and selected["winner_sha256"] == _file_sha256(pose_file)
            and selected["draws_sha256"] == _file_sha256(output / "draws.npz")
            and row["complex_id"] == target["complex_id"] and row["extent"] == extent,
            f"Pose selecionada não corresponde a esta execução: {checkpoint}")
    dockq = None
    if extent == "expanded" or not row["hard_clash"]:
        complex_pdb = prepare_reduced_pose(receptor_model_pdb=paths["receptor"], ligand_template_pdb=paths["ligand"],
                    receptor_chain=target["receptor_chain"], ligand_chain=target["ligand_chain"],
                    pose_json=pose_file, output_dir=output / "model")
        official = output / "standard_dockq.json"
        binding = {"complex_id": target["complex_id"], "extent": extent,
                   "winner_sha256": selected["winner_sha256"], "protocol_sha256": protocol_hash}
        # A crash before publication leaves only a scratch directory. A crash
        # after publication reuses the validated score of this exact winner.
        if not official.exists():
            with TemporaryDirectory(dir=output) as temporary:
                evaluate_standard_complex(complex_pdb, paths["reference"],
                         receptor_chain=target["receptor_chain"], ligand_chain=target["ligand_chain"],
                         expected_reference_sha256=target["reference_sha256"], dependency_path=deps, output_dir=temporary,
                         source_binding=binding, timeout_seconds=600)
                os.replace(Path(temporary) / official.name, official)
        result = verified_record(official)
        require(result["source_binding"] == binding and result["quality_metric"] == quality_metric_descriptor()
                and result["inputs"]["model_sha256"] == _file_sha256(complex_pdb)
                and result["inputs"]["reference_sha256"] == target["reference_sha256"], f"DockQ incompatível: {official}")
        dockq = result["metrics"]["dockq"]
    return {**row, "dockq": dockq, **thresholds(extent, row["hard_clash"], dockq)}


def run(study, output, limit, deps):
    study, output = study.resolve(strict=True), output.resolve()
    require(not output.is_relative_to(study) and not study.is_relative_to(output), "Escolha uma saída separada da coleta.")
    require(not output.is_relative_to(ROOT / "output/fatorial_dockq_20261119"), "Preserve a análise original.")
    for directory in (output, *output.parents):
        require(not any((directory / marker).exists() for marker in (
                    "batch_manifest.json", "experimental_protocol.json", "planned_workers.json", "factorial_results.json")),
                f"Preserve a coleta ou análise arquivada em {directory}.")
    if output.exists() and any(p.name != ".control.lock" for p in output.iterdir()):
        require((output / "protocol.json").is_file() and read(output / "protocol.json").get("schema") == "article_uniform_control_v2",
                "Escolha uma saída vazia ou uma saída deste controle.")
    cases = inputs_for_study(study, limit)
    preflight = official_dockq_preflight(dependency_path=deps, timeout_seconds=180)
    require(preflight.get("available") and preflight["quality_metric"] == quality_metric_descriptor(),
            f"DockQ oficial 2.1.3 indisponível: {preflight.get('error', '')}")
    plan = {"schema": "article_uniform_control_v2", "cohort_sha256": _file_sha256(study / "selection.json"),
            "script_sha256": _file_sha256(Path(__file__)), "engine_manifest_sha256": engine_identity(),
            "numpy_version": np.__version__,
            "dockq_implementation_sha256": preflight["implementation_sha256"], "budget": BUDGET,
            "complexes": [{"id": t["complex_id"], "configs": {e: _file_sha256(cfg[e]) for e in EXTENTS}}
                          for t, _, cfg in cases],
            "sampling": "independent uniform in six coordinates; seed 1, common normalized draws across extents",
            "selection": "retain best cheap scores, apply archived final rescoring and winner steric gate",
            "unit": "one winner per complex and extent; no optimizer"}
    output.mkdir(parents=True, exist_ok=True)
    with _BatchFileLock(output / ".control.lock"):
        protocol = output / "protocol.json"
        if protocol.is_file():
            require(read(protocol) == plan, "Saída pertence a outro protocolo. Escolha uma nova pasta.")
        else:
            require(all(p.name == ".control.lock" for p in output.iterdir()), "A pasta de saída deve estar vazia.")
            _atomic_write_json(protocol, plan)
        plan_hash = canonical_sha256(plan)
        rows = []
        for case in cases:
            cid = case[0]["complex_id"]
            for extent in EXTENTS:
                directory = output / "trials" / cid / extent
                record = directory / "trial.json"
                if record.exists():
                    saved = verified_record(record)
                    require(saved["protocol_sha256"] == plan_hash and
                            saved["row"]["complex_id"] == cid and saved["row"]["extent"] == extent and
                            all(_file_sha256(directory / key) == value for key, value in saved["files"].items()),
                            f"Registro incompatível na retomada: {record}")
                    row = saved["row"]
                else:
                    print(f"{cid} — {extent} ({BUDGET} avaliações)", flush=True)
                    directory.mkdir(parents=True, exist_ok=True)
                    row = trial(case, extent, directory, deps, plan_hash)
                    files = ["artifacts/best_pose.json", "selected.json", "draws.npz"]
                    if row["dockq"] is not None:
                        files += ["model/reduced_complex.pdb", "standard_dockq.json"]
                    saved = {"protocol_sha256": plan_hash, "row": row,
                             "files": {key: _file_sha256(directory / key) for key in files}}
                    save_record(record, saved)
                rows.append(row)
        results = {}
        for extent in EXTENTS:
            group = [r for r in rows if r["extent"] == extent]
            successes = sum(r["success"] for r in group)
            results[extent] = {"planned": len(cases), "valid_poses": sum(r["valid_pose"] for r in group),
                               "successes": successes, "success_rate": successes / len(cases),
                               "successes_common_gate": sum(r["success_common_gate"] for r in group)}
        with (output / "observations.csv.tmp").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(output / "observations.csv.tmp", output / "observations.csv")
        _atomic_write_json(output / "summary.json", {"protocol_sha256": plan_hash, "results": results})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY, help="Pasta da coleta fatorial original.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Pasta nova ou do mesmo controle para retomar.")
    parser.add_argument("--limit-complexes", type=int, help="Piloto com os primeiros N complexos na ordem da seleção.")
    parser.add_argument("--dockq-deps", type=Path, default=DEFAULT_DEPS if DEFAULT_DEPS.exists() else None,
                        help="Diretório de dependências isoladas do DockQ oficial 2.1.3.")
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.output_dir, args.limit_complexes, args.dockq_deps), indent=2))


if __name__ == "__main__":
    main()
