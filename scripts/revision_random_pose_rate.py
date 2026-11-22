"""Measure official DockQ success across individual draws of the frozen random control."""
from __future__ import annotations

import argparse
from concurrent.futures import CancelledError, FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager, ExitStack
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from revision_random_control import (
    BUDGET, DEFAULT_DEPS, DEFAULT_STUDY, EXTENTS, Config, RigidSearchSpace, _BatchFileLock,
    _atomic_write_json, _file_sha256, canonical_sha256, inputs_for_study, load_config, load_peptide,
    official_dockq_preflight, quality_metric_descriptor, read, require, save_record, verified_record,
)
from docking_confidence.evaluation.reference_dockq import _write_complex, read_engine_order_atoms

RESULTS = ROOT / "output/revisao_professor"
WORKER = Path(__file__).with_name("revision_pose_dockq_worker.py")
CHUNK = 100
METRICS = ("dockq", "fnat", "irmsd", "lrmsd")


def space_for_trial(config, ligand):
    cfg = Config.model_validate(load_config(str(config)))
    require(cfg.docking_task == "local_refinement" and cfg.initialization == "uniform",
            "Esperava-se uma configuração fatorial uniforme com âncora no centróide inicial.")
    # Geometry only. No sampling, objective evaluation or reference is used here.
    return RigidSearchSpace(cfg=cfg, rng=None, pocket=None, base_coords=load_peptide(str(ligand)),
                            score_cheap_fn=None, max_trans=cfg.max_trans, max_rot_deg=cfg.max_rot_deg)


@contextmanager
def scoring_worker(target, paths, deps, backend_sha, scratch):
    model = scratch / "template.pdb"
    rec = read_engine_order_atoms(paths["receptor"])
    lig = read_engine_order_atoms(paths["ligand"])
    require(all(a.chain_id == target["receptor_chain"] for a in rec) and
            all(a.chain_id == target["ligand_chain"] for a in lig), "Cadeia inesperada nos modelos do controle.")
    _write_complex(model, rec, lig)
    request = {"model": str(model), "reference": str(paths["reference"]), "backend_sha256": backend_sha,
               **{key: target[key] for key in ("receptor_chain", "ligand_chain", "reference_sha256")}}
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    if deps:
        env["PYTHONPATH"] = str(Path(deps).resolve(strict=True))
    cmd = [sys.executable, "-B", "-X", "utf8", str(WORKER)]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env,
                          creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
        def response():
            line = process.stdout.readline()
            require(line, f"O avaliador DockQ terminou inesperadamente (código {process.poll()}).")
            return json.loads(line)

        try:
            process.stdin.write((json.dumps(request) + "\n").encode())
            process.stdin.flush()
            require(response() == {"ready": True, "ligand_rows": len(lig)}, "Modelo incompatível com o motor.")

            def score(coords):
                frame = np.asarray(coords, dtype="<f8")
                require(frame.shape == (len(lig), 3) and np.isfinite(frame).all(), "Pose inválida.")
                process.stdin.write(frame.tobytes(order="C"))
                process.stdin.flush()
                return response()

            yield score
        finally:
            process.stdin.close()
        require(process.wait() == 0, "O processo DockQ terminou com erro.")


def write_csv(path, rows, fields):
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)
    os.replace(temporary, path)


def score_trial(case, extent, control, out, limit, deps, backend_sha, protocol_sha, control_sha, cancel):
    target, paths, configs = case
    cid = target["complex_id"]
    original = control / "trials" / cid / extent
    trial = verified_record(original / "trial.json")
    selected = verified_record(original / "selected.json")
    winner = trial["row"]
    archive = original / "draws.npz"
    require(trial["protocol_sha256"] == selected["protocol_sha256"] == control_sha and
            trial["files"]["draws.npz"] == selected["draws_sha256"] == _file_sha256(archive) and
            trial["files"]["selected.json"] == _file_sha256(original / "selected.json") and
            (winner["complex_id"], winner["extent"], winner["seed"], winner["objective_evaluations"]) ==
            (cid, extent, 1, BUDGET), f"Artefatos incompatíveis para {cid}/{extent}.")
    with np.load(archive, allow_pickle=False) as pool:
        vectors = pool["vectors"]
    require(vectors.dtype == np.dtype("<f8") and vectors.shape == (BUDGET, 6) and
            hashlib.sha256(vectors.tobytes(order="C")).hexdigest() == selected["vectors_sha256"],
            "Matriz de vetores incompatível.")
    space = space_for_trial(configs[extent], paths["ligand"])
    require(np.all((vectors >= space.domain_low) & (vectors <= space.domain_high)), "Sorteio fora dos limites.")
    chosen = winner["draw_selected"] - 1
    require(0 <= chosen < BUDGET, "Índice inválido de vencedora.")
    winner_path = original / "artifacts/best_pose.json"
    require(selected["winner_sha256"] == trial["files"]["artifacts/best_pose.json"] == _file_sha256(winner_path)
            and np.array_equal(space.solution_to_pose(vectors[chosen]).coords, read(winner_path)["coords"]),
            "A transformação do sorteio vencedor não reproduz as coordenadas gravadas pelo controle.")
    binding = {"protocol_sha256": protocol_sha, "trial_sha256": _file_sha256(original / "trial.json")}
    directory = out / "trials" / cid / extent
    directory.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    with ExitStack() as stack:
        score = None
        for start in range(0, limit, CHUNK):
            if cancel.is_set():
                raise CancelledError()
            stop = min(start + CHUNK, limit)
            checkpoint = directory / "chunks" / f"{start + 1:05d}.json"
            if checkpoint.is_file():
                saved = verified_record(checkpoint)
                require(saved["binding"] == binding and saved["first_draw"] == start + 1 and
                        len(saved["metrics"]) == stop - start, f"Bloco de outra avaliação: {checkpoint}")
                metrics = saved["metrics"]
            else:
                if score is None:
                    scratch = stack.enter_context(TemporaryDirectory(dir=directory))
                    score = stack.enter_context(scoring_worker(target, paths, deps, backend_sha, Path(scratch)))
                metrics = []
                for vector in vectors[start:stop]:
                    if cancel.is_set():
                        raise CancelledError()
                    metrics.append(score(space.solution_to_pose(vector).coords))
                save_record(checkpoint, {"binding": binding, "first_draw": start + 1, "metrics": metrics})
                print(f"{cid}/{extent}: {stop}/{limit} poses", flush=True)
            all_metrics.extend(metrics)
    if winner["dockq"] is not None and chosen < limit:
        require(math.isclose(all_metrics[chosen][0], winner["dockq"], abs_tol=1e-12, rel_tol=0),
                "A vencedora não reproduziu o DockQ da avaliação individual do controle.")
    successes = sum(m[0] >= 0.23 for m in all_metrics)
    write_csv(directory / "poses.csv", ((i, *m, int(m[0] >= 0.23)) for i, m in enumerate(all_metrics, 1)),
              ("draw", *METRICS, "success"))
    return {"complex_id": cid, "extent": extent, "pose_successes": successes, "poses_scored": limit,
            "pose_success_rate": successes / limit, "winner_draw": chosen + 1,
            "winner_dockq": winner["dockq"], "winner_success": winner["success"]}


def run(study, control, out, complex_limit, draw_limit, deps, workers=1):
    study, control, out = study.resolve(strict=True), control.resolve(strict=True), out.resolve()
    require(type(workers) is int and workers >= 1, "O número de avaliadores deve ser um inteiro positivo.")
    require(1 <= draw_limit <= BUDGET, "O número de poses deve estar entre 1 e 7600.")
    frozen = read(control / "protocol.json")
    control_sha = canonical_sha256(frozen)
    require(frozen["schema"] == "article_uniform_control_v2" and frozen["budget"] == BUDGET and
            read(control / "summary.json")["protocol_sha256"] == control_sha and
            _file_sha256(study / "selection.json") == frozen["cohort_sha256"],
            "Execute primeiro o controle aleatório completo para a seleção desejada.")
    for original in (study, control, ROOT / "output/fatorial_dockq_20261119"):
        require(not out.is_relative_to(original) and not original.is_relative_to(out), "Use uma saída separada.")
    for directory in (out, *out.parents):
        require(not any((directory / name).is_file() for name in ("batch_manifest.json", "planned_workers.json",
                    "experimental_protocol.json", "factorial_results.json")), f"Preserve a coleta em {directory}.")
    count = len(frozen["complexes"]) if complex_limit is None else complex_limit
    require(0 < count <= len(frozen["complexes"]), "Limite de complexos fora do controle.")
    cases = inputs_for_study(study, count)
    require([{"id": t["complex_id"], "configs": {e: _file_sha256(c[e]) for e in EXTENTS}}
             for t, _, c in cases] == frozen["complexes"][:count], "Coorte ou configurações divergentes.")
    backend = official_dockq_preflight(dependency_path=deps, timeout_seconds=180)
    require(backend.get("quality_metric") == quality_metric_descriptor() and
            backend["implementation_sha256"] == frozen["dockq_implementation_sha256"], "DockQ diverge do controle.")
    plan = {"schema": "article_uniform_dockq_draws_v1", "control_sha256": control_sha,
            "implementation_sha256": backend["implementation_sha256"], "metric": backend["quality_metric"],
            "script_sha256": _file_sha256(Path(__file__)), "worker_sha256": _file_sha256(WORKER),
            "complexes": [t["complex_id"] for t, _, _ in cases], "draws_per_trial": draw_limit,
            "threshold_inclusive": 0.23, "pose_denominator": "all draws in order, without steric filtering",
            "winner_denominator": "one scheduled winner per complex; same steric policy as article",
            "full_study": count == read(study / "selection.json")["target_count"] and draw_limit == BUDGET}
    protocol_sha = canonical_sha256(plan)
    if out.exists() and any(p.name != ".pose_rate.lock" for p in out.iterdir()):
        require((out / "protocol.json").is_file() and read(out / "protocol.json") == plan,
                "Escolha uma pasta vazia ou a saída da mesma avaliação.")
    with _BatchFileLock(out / ".pose_rate.lock"):
        protocol = out / "protocol.json"
        if protocol.exists():
            require(read(protocol) == plan, "A saída pertence a outro protocolo.")
        else:
            require(all(p.name == ".pose_rate.lock" for p in out.iterdir()), "A saída deve estar vazia.")
            _atomic_write_json(protocol, plan)
        cancel = Event()
        with ThreadPoolExecutor(max_workers=min(workers, count * len(EXTENTS))) as pool:
            futures = [pool.submit(score_trial, case, extent, control, out, draw_limit, deps,
                                   backend["implementation_sha256"], protocol_sha, control_sha, cancel)
                       for case in cases for extent in EXTENTS]
            try:
                pending = set(futures)
                while pending:
                    completed, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                    for future in completed:
                        future.result()
            except BaseException:
                cancel.set()
                for future in futures:
                    future.cancel()
                raise
            # Submission order is the frozen cohort and extent order, irrespective
            # of when individual workers completed or resumed their checkpoints.
            rows = [f.result() for f in futures]
        rates = {}
        for extent in EXTENTS:
            group = [r for r in rows if r["extent"] == extent]
            hits = sum(r["pose_successes"] for r in group)
            total = sum(r["poses_scored"] for r in group)
            winners = sum(r["winner_success"] for r in group)
            rates[extent] = {"pose_successes": hits, "poses_scored": total, "pose_success_rate": hits / total,
                             "winner_successes": winners, "complexes": len(group),
                             "winner_success_rate": winners / len(group)}
        write_csv(out / "observations.csv", (r.values() for r in rows), rows[0].keys())
        result = {"protocol_sha256": protocol_sha, "full_study": plan["full_study"], "results": rates}
        _atomic_write_json(out / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--control-dir", type=Path, default=RESULTS / "controle", help="Controle já concluído.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "taxa_poses", help="Saída desta avaliação.")
    parser.add_argument("--limit-complexes", type=int, help="Avaliar os primeiros N complexos do controle.")
    parser.add_argument("--draw-limit", type=int, default=BUDGET, help="Avaliar os primeiros N sorteios por ensaio.")
    parser.add_argument("--workers", type=int, default=1, help="Número de ensaios avaliados em paralelo (padrão 1).")
    parser.add_argument("--dockq-deps", type=Path, default=DEFAULT_DEPS if DEFAULT_DEPS.exists() else None)
    args = parser.parse_args()
    print(json.dumps(run(args.study, args.control_dir, args.output_dir, args.limit_complexes, args.draw_limit,
                         args.dockq_deps, workers=args.workers), indent=2))


if __name__ == "__main__":
    main()
