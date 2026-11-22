"""Repetir as condições semeadas do fatorial e conferir a linhagem da pose fornecida."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from functools import lru_cache
import hashlib
import inspect
import io
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from revision_convergence import ROOT, checked_trace, checkpoints, digest
from docking_confidence.adapters.dockingpp_local_engine import DockingPPLocalEngine
from docking_confidence.application.batch import _atomic_write_json, _BatchFileLock
from docking_confidence.application.factorial_reporting import _inside, _json
from docking_confidence.evaluation.official_dockq import canonical_sha256

DEFAULT_STUDY = ROOT / 'runs/fatorial_dockq_20261119'
OBSERVATIONS = ROOT / 'output/fatorial_dockq_20261119/factorial_observations.csv'
DEFAULT_OUT = ROOT / 'output/revisao_professor/linhagem'

# The original serial driver used by the six-run validation. Only its scheduling
# changed; preserve its receipts while binding the exception to this replay code.
_SERIAL_SCRIPT_HASH = '649fba7edc4fa756ff4d527ab6c94881005d03fe1d67abfcf2c82d7616171333'
_COMPATIBLE_REPLAY_CORE = '06d8a3bd1797a337f9ca81a1d3ce417fbcfe421bf803e417ccbcfa1f3ff46ec1'


@lru_cache(maxsize=1)
def replay_core_hash():
    functions = (engine_files, inspect_run, compare_replay, replay_run)
    source = '\n'.join(inspect.getsource(f) for f in functions)
    return hashlib.sha256(source.encode('utf-8')).hexdigest()


def binding_matches(saved, current):
    if saved == current:
        return True
    return (saved == {**current, 'script': _SERIAL_SCRIPT_HASH}
            and replay_core_hash() == _COMPATIBLE_REPLAY_CORE)


def selected_rows(observations, target=None, method=None, region=None):
    with observations.open(encoding='utf-8-sig', newline='') as f:
        rows = [row for row in csv.DictReader(f) if row['condition'].endswith('_seeded')]
    rows = [r for r in rows if (not target or r['complex_id'] == target)
            and (not method or r['algorithm'] == method)
            and (not region or r['condition'] == region + '_seeded')]
    identities = {(r['complex_id'], r['job_id']) for r in rows}
    if not rows or len(identities) != len(rows):
        raise ValueError('Seleção vazia ou execuções semeadas duplicadas.')
    return rows


def protocol_inputs(study, rows):
    """Verify the two search inputs against the protocol. The reference is not needed."""
    workers = _json(study / 'planned_workers.json')
    inputs = {}
    selected = {(r['batch_shard'], r['complex_id']) for r in rows}
    for part in workers['shards']:
        shard = _inside(study, 'shard_%02d' % part['index'])
        protocol = shard / 'batch/experimental_protocol.json'
        if digest(protocol) != part['protocol_file_sha256']:
            raise ValueError(f'Protocolo histórico alterado: {shard.name}')
        for target in _json(protocol)['plan']['targets']:
            key = (shard.name, target['complex_id'])
            if key not in selected:
                continue
            paths = []
            for role in ('receptor', 'ligand'):
                path = _inside(shard / 'inputs', target[role + '_path'])
                if digest(path) != target[role + '_sha256']:
                    raise ValueError(f'Entrada diferente da registrada: {path}')
                paths.append(path)
            inputs[key] = paths
    if inputs.keys() != selected:
        raise ValueError('Há complexos sem entradas verificadas.')
    return inputs


def engine_files(study, row):
    logs = checked_trace(study, row).parent
    output = logs.parent if row['condition'] == 'reduced_seeded' else logs
    return output, output.parent / 'resolved_config.yaml'


def inspect_run(output, reduced):
    result = _json(output / 'result.json')
    folder = output / 'local_refinement' if reduced else output
    pocket, = _json(folder / 'summary.json')['population_confidence_by_pocket']
    valid = result['final_pose_available']
    if not valid and (not reduced or result.get('failure_reason') != 'no_valid_reduced_pose'):
        raise ValueError('Execução sem desfecho estrutural reconhecido.')
    meta = result['best_pose_meta'] if valid else result['per_pocket_results'][0]['best_pose_meta']
    pose = meta['solution_vector']
    if len(pose) != 6 or not all(math.isfinite(v) for v in pose):
        raise ValueError('Vetor da pose inválido.')
    audit = pocket['execution_audit']
    if audit['objective_evaluations_actual'] != 7600 or pocket['initial_population_audit']['initialization'] != 'seeded':
        raise ValueError('Orçamento ou inicialização incompatível com o estudo.')
    curve = checkpoints((folder / 'metrics.jsonl').read_bytes().splitlines(), 7600)
    comparison = {'valid': valid, 'vector': pose, 'score': result.get('final_score'),
                  'initial_population': audit['initial_population_sha256'],
                  'eval_by_generation': audit['objective_evaluations_by_generation'],
                  'objective_trajectory': curve}
    return comparison, meta, pocket.get('seed_lineage')


def compare_replay(original, replay, row):
    reduced = row['condition'] == 'reduced_seeded'
    history, _, _ = inspect_run(original, reduced)
    current, meta, trace = inspect_run(replay, reduced)
    if not trace or trace.get('schema') != 'seed_lineage_v1' or not trace.get('generations'):
        raise ValueError('O motor não produziu o histórico de linhagem esperado.')
    if current['valid'] and any(type(meta.get(k)) != bool for k in ('seed_lineage', 'seed_unchanged')):
        raise ValueError('A pose final não tem identificação de linhagem.')
    mismatch = [key for key in history if history[key] != current[key]]
    return {'complex_id': row['complex_id'], 'algorithm': row['algorithm'], 'condition': row['condition'],
            'job_id': row['job_id'], 'matches_original': not mismatch, 'differences': mismatch,
            'valid_final_pose': current['valid'], 'historical_dockq': row['dockq'] or None,
            'final_seed_lineage': meta['seed_lineage'] if current['valid'] else None,
            'final_unchanged_seed': meta['seed_unchanged'] if current['valid'] else None,
            'trace': trace}


def replay_run(study, row, input_paths, out, engine_identity):
    original, config = engine_files(study, row)
    reduced = row['condition'] == 'reduced_seeded'
    pocket_folder = original / 'local_refinement' if reduced else original
    binding = {'engine_manifest': engine_identity, 'script': digest(Path(__file__)),
               'original_result': digest(original / 'result.json'), 'config': digest(config),
               'original_trace': digest(pocket_folder / 'metrics.jsonl'),
               'original_summary': digest(pocket_folder / 'summary.json'),
               'reference_observation': canonical_sha256(row)}
    directory = _inside(out, f"jobs/{row['complex_id']}/{row['job_id']}")
    directory.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'replay.json'
    if receipt.exists():
        saved = _json(receipt)
        if saved.pop('sha256') != canonical_sha256(saved) or not binding_matches(saved['binding'], binding):
            raise ValueError(f'Registro incompatível. Use outra pasta de saída: {receipt}')
        return saved['result']

    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
    # Match the original worker settings for numerical libraries.
    env.update({k: '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    with TemporaryDirectory(dir=directory, prefix='run-') as tmp:
        work = Path(tmp)
        cmd = [sys.executable, '-B', '-m', 'dockingpp.adapters.inbound.cli', 'run', '--config', str(config),
               '--receptor', str(input_paths[0]), '--peptide', str(input_paths[1]), '--out', str(work / 'output')]
        with (directory / 'engine.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(cmd, env=env, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            # The original motor exits with code 1 on a completed steric rejection.
            path = work / 'output/result.json'
            status = _json(path) if path.is_file() else {}
            if not (completed.returncode == 1 and reduced and status.get('status') == 'REDUCED_FAILED'
                    and status.get('failure_reason') == 'no_valid_reduced_pose'
                    and status.get('final_pose_available') is False):
                raise RuntimeError(f"Falha do motor. Consulte {directory / 'engine.log'}")
        result = compare_replay(original, work / 'output', row)
    record = {'binding': binding, 'result': result}
    _atomic_write_json(receipt, {**record, 'sha256': canonical_sha256(record)})
    return result


def write_csv(path, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    temp = path.with_suffix('.tmp')
    temp.write_text(buffer.getvalue(), encoding='utf-8', newline='')
    temp.replace(path)


def export_results(out, results):
    individuals, generations, groups = [], [], defaultdict(list)
    survival = defaultdict(list)
    for result in results:
        fixed = {k:result[k] for k in ('complex_id', 'job_id', 'algorithm', 'condition')}
        groups[result['algorithm'], result['condition']].append(result)
        individuals.append({**fixed, **{k:result[k] for k in ('matches_original','valid_final_pose',
                            'final_seed_lineage','final_unchanged_seed','historical_dockq')},
                            'differences': ','.join(result['differences'])})
        for generation in result['trace']['generations']:
            if result['matches_original']:
                survival[result['algorithm'], result['condition'], generation['generation']].append(generation)
            generations.append({**fixed, 'matches_original':result['matches_original'],
                **{k:generation.get(k) for k in ('generation','n_eval_cumulative','population_size',
                'unchanged_seed_count','seed_lineage_count','pso_global_best_from_seed_particle')}})
    summary = []
    for (method, condition), runs in sorted(groups.items()):
        matches = [r for r in runs if r['matches_original']]
        valid = [r for r in matches if r['valid_final_pose']]
        summary.append({'algorithm':method,'condition':condition,'replayed':len(runs),
                        'matched':len(matches),'matched_valid_final_poses':len(valid),
                        'final_from_seed':sum(r['final_seed_lineage'] for r in valid),
                        'final_unchanged':sum(r['final_unchanged_seed'] for r in valid)})
    # A run contributes only to generations it actually reached. Survival means that
    # at least one unchanged copy of the supplied pose is still in the population.
    survival_rows = []
    for (method, condition, generation), reached in sorted(survival.items()):
        unchanged = sum(g['unchanged_seed_count'] > 0 for g in reached)
        descendants = sum(g['seed_lineage_count'] > 0 for g in reached)
        survival_rows.append({'algorithm': method, 'condition': condition, 'generation': generation,
            'verified_runs_reaching_generation': len(reached), 'with_unchanged_seed': unchanged,
            'fraction_with_unchanged_seed': unchanged / len(reached), 'with_seed_lineage': descendants,
            'fraction_with_seed_lineage': descendants / len(reached)})
    if survival_rows:
        write_csv(out / 'sobrevivencia.csv', survival_rows)
    write_csv(out / 'por_execucao.csv', individuals)
    write_csv(out / 'por_geracao.csv', generations)
    write_csv(out / 'resumo.csv', summary)


def replay_selection(rows, inputs, study, out, identity, workers):
    # The output lock covers the entire selection. Each subprocess has its own
    # directory, and aggregation always follows the original observation order.
    ordered = [None] * len(rows)
    with ThreadPoolExecutor(max_workers=min(workers, len(rows))) as pool:
        pending = {
            pool.submit(replay_run, study, row, inputs[row['batch_shard'], row['complex_id']],
                        out, identity): index for index, row in enumerate(rows)
        }
        try:
            for finished, future in enumerate(as_completed(pending), 1):
                index = pending[future]
                ordered[index] = future.result()
                row = rows[index]
                print(f"[{finished}/{len(rows)}] {row['complex_id']} {row['job_id']}", flush=True)
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    return ordered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=DEFAULT_STUDY)
    parser.add_argument('--observations', type=Path, default=OBSERVATIONS)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--complex', dest='target', help='Apenas este complexo da coorte')
    parser.add_argument('--method', choices=('ag','pso','hybrid'))
    parser.add_argument('--region', choices=('reduced','expanded'))
    parser.add_argument('--workers', type=int, default=1, metavar='N',
                        help='Máximo de processos de docking simultâneos (padrão: 1)')
    parser.add_argument('--plan', action='store_true', help='Listar o tamanho da seleção sem iniciar as buscas')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers deve ser um inteiro positivo.')
    study, out = args.study.resolve(), args.output_dir.resolve()
    observed = args.observations.resolve().parent
    if any(out.is_relative_to(path) or path.is_relative_to(out) for path in (study, observed)):
        parser.error('Escolha uma pasta de saída separada dos resultados históricos.')
    rows = selected_rows(args.observations, args.target, args.method, args.region)
    inputs = protocol_inputs(study, rows)
    print(f'{len(rows)} buscas semeadas em {len(inputs)} complexos. Saída: {out}', flush=True)
    if args.plan:
        return 0
    engine = DockingPPLocalEngine(seed_lineage=True)
    if engine.module_root.resolve() != (ROOT / 'src').resolve():
        raise ValueError('O motor verificado difere do pacote executado na repetição.')
    identity = engine.audit_descriptor()['source_manifest_sha256']
    with _BatchFileLock(out / 'replay.lock'):
        # Prevent a pilot or other selection silently replacing the aggregate CSVs of a full study.
        selection = {'schema':'seed_replay_selection_v1','jobs': [[r['complex_id'],r['job_id']] for r in rows],
                     'engine_manifest': identity, 'observations_sha256':digest(args.observations)}
        plan_path = out / 'selection.json'
        if plan_path.exists() and _json(plan_path) != selection:
            raise ValueError('Esta pasta pertence a outra seleção. Use --output-dir com uma pasta nova.')
        if not plan_path.exists():
            if any(path.name != 'replay.lock' for path in out.iterdir()):
                raise ValueError('Escolha uma pasta vazia para iniciar a análise de linhagem.')
            _atomic_write_json(plan_path, selection)
        results = replay_selection(rows, inputs, study, out, identity, args.workers)
        export_results(out, results)
    mismatches = sum(not r['matches_original'] for r in results)
    print(f'{len(results)} repetições processadas. Divergências em relação ao histórico: {mismatches}.')
    return int(bool(mismatches))


if __name__ == '__main__':
    raise SystemExit(main())
