"""Small, hash-bound correction to the immutable upstream V5 archive.

The original ZIP/config pins remain intact. Protocol failures, surface placement
and score-zero ranking corrections are reproduced from exact source preimages.
Algorithm operators and scoring formulas are preserved; REDUCED surface anchoring
is versioned separately from the unchanged FULL coordinate transformation.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

ARCHIVE_SHA256 = "55cda28c370294b4019f74167f434f543273db0ba5a7961f24c54f9e57cfc576"
MEMBER = "dockingpp/dockingpp/pipeline/execucao/busca_execucao/executor_busca.py"
SOURCE_SHA256 = "b893a205e0ce145c9acce12b924293e5e81ac747e2c2cf393c678d07f69f681c"
POPULATION_MEMBER = "dockingpp/dockingpp/search/algoritmos/base.py"
POPULATION_SHA256 = "35b90d5fe9aadabac34a2d26d889a4096bf7fcc06954db0acebedb86414f093c"
PATCH_VERSION = "factorial_seed_lineage_v6"

_FAILURE_HELPER = '''
def _fail_reduced(*, cfg, out_dir, run_id, status, reason, evidence=None):
    """Persist a failed REDUCED arm; candidates remain diagnostic artifacts."""
    evidence = evidence or {}
    payload = {
        **evidence,
        "schema_version": "2.0",
        "mode": "reduced",
        "search_space_mode": "reduced",
        "requested_protocol": "reduced",
        "executed_mode": "reduced",
        "run_id": run_id,
        "status": status,
        "failure_reason": reason,
        "next_action": "RUN_FULL_REQUIRED",
        "search_strategy": cfg.search_engine,
        "seed": cfg.seed,
        "generations": cfg.generations,
        "pop_size": cfg.pop_size,
        "fallback_to_full": False,
        "fallback_executed": False,
        "final_pose_available": False,
        "pose_output_path": None,
    }
    for name in ("summary.json", "result.json"):
        atomic_write_json(os.path.join(out_dir, name), payload)
    with open(os.path.join(out_dir, "metrics.jsonl"), "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"name": "run.status", "status": status,
                                 "reason": reason}) + "\\n")
    raise RuntimeError(f"{status}: {reason}; RUN_FULL_REQUIRED")


'''


def patch_source(source: bytes) -> bytes:
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("V5 executor source does not match the patch preimage")
    text = source.decode("utf-8")
    # Do not reclassify the first rejected pose as a valid parent winner.
    start = text.index("        if best_result is None:\n            best_result = fallback_result")
    end = text.index("        for item in per_pocket_results:", start)
    text = text[:start] + '''        if best_result is None:
            _fail_reduced(
                cfg=cfg, out_dir=out_dir, run_id=run_id,
                status="REDUCED_FAILED", reason="no_valid_reduced_pose",
                evidence={"per_pocket_results": per_pocket_results,
                          "executed_pockets": feasible_pocket_ids,
                          "search_engine_name": engine_name_reduced,
                          "total_n_eval": total_n_eval},
            )

''' + text[end:]
    # The old no-pocket branch ran FULL unconditionally, ignoring both flags.
    start = text.index("        if fallback_triggered:\n", text.index("        fallback_triggered = not feasible_pockets"))
    end = text.index("        selection_stage_policy_metadata =", start)
    text = text[:start] + '''        if fallback_triggered:
            _fail_reduced(
                cfg=cfg, out_dir=out_dir, run_id=run_id,
                status="NO_FEASIBLE_POCKET", reason="no_feasible_pocket",
                evidence={"candidate_pockets": candidate_pocket_ids,
                          "rejected_pockets": rejected,
                          "scan_results": scan_results,
                          "executed_pockets": [], "total_n_eval": 0},
            )

''' + text[end:]
    # Also forbid confidence/geometric fallbacks, even if a future config
    # enables their recommendation. A FULL arm must be scheduled separately.
    start = text.index('        """Persist reduced evidence, execute full search, and publish both states."""')
    end = text.index("    def executar(\n", start)
    text = text[:start] + '''        """A separate FULL job is required after a REDUCED failure."""
        _fail_reduced(
            cfg=cfg, out_dir=out_dir, run_id=run_id,
            status="RUN_FULL_REQUIRED",
            reason=fallback_reason_override or "confidence_policy_requested_full",
            evidence={"reduced_evidence": parent_summary},
        )

''' + text[end:]
    text = text.replace("class ExecutorBuscaPipeline:", _FAILURE_HELPER + "class ExecutorBuscaPipeline:", 1)
    compile(text, MEMBER, "exec")
    return text.encode("utf-8")


# Exact line replacements, applied only after the full preimage hash matches.
_SOURCE_CORRECTIONS = {'dockingpp/dockingpp/application/config.py': {'after_sha256': '1b0c503b83cab23cecfb0fcbae3b6fafd75230534923a38b9dc6c8cb5478f722',
                                               'before_sha256': '048b1dd07aac40749dc514df8ffa26c1ab8e2b443df1afbed9ebb1b10e6ff83f',
                                               'spans': [(8,
                                                          9,
                                                          'from typing import Any, Dict, Literal, '
                                                          'Optional\n'),
                                                         (33,
                                                          33,
                                                          '    docking_task: '
                                                          'Literal["surface_discovery", '
                                                          '"local_refinement"] = '
                                                          '"surface_discovery"\n'
                                                          '    local_refinement_reduced_scale: '
                                                          'float = Field(default=0.5, gt=0.0, '
                                                          'le=1.0)\n'
                                                          '    # None preserves the historical '
                                                          'mode-dependent initialization '
                                                          'contract.\n'
                                                          '    initialization: '
                                                          'Optional[Literal["uniform", "seeded"]] '
                                                          '= None\n'
                                                          '    search_extent: '
                                                          'Optional[Literal["reduced", '
                                                          '"expanded"]] = None\n'
                                                          '    experiment_budget_policy: '
                                                          'Optional[Literal["common_ceiling", '
                                                          '"fixed_evaluations"]] = None\n'
                                                          '    pose_source: Optional[str] = '
                                                          'None\n')]},
 'dockingpp/dockingpp/pipeline/execucao/busca_execucao/_execucao_unica_coordenador.py': {'after_sha256': 'e2ff4c4d064b32e882c665824a72100c99e3d6e95c278c57f0bd3b1ce8fbd30c',
                                                                                         'before_sha256': 'd443b94f1a581815ca02fdd41c88319415c22a923861ada23031adf4926f54e4',
                                                                                         'spans': [(179,
                                                                                                    179,
                                                                                                    '            '
                                                                                                    '"docking_task": '
                                                                                                    'getattr(cfg, '
                                                                                                    '"docking_task", '
                                                                                                    '"surface_discovery"),\n'
                                                                                                    '            '
                                                                                                    '"local_refinement_reduced_scale": '
                                                                                                    'getattr(cfg, '
                                                                                                    '"local_refinement_reduced_scale", '
                                                                                                    '0.5),\n'
                                                                                                    '            '
                                                                                                    '"max_trans": '
                                                                                                    'cfg.max_trans,\n'
                                                                                                    '            '
                                                                                                    '"max_rot_deg": '
                                                                                                    'cfg.max_rot_deg,\n')]},
 'dockingpp/dockingpp/pipeline/execucao/busca_execucao/_rescoring_final.py': {'after_sha256': '0e527f382dff28f35d2cd5bd5ae38e550c2889f9239259d61bb0de5358b6dc2b',
                                                                              'before_sha256': '2f19263c56fa9d5be9d79df28ccb039bffc5d92665fa10aefe6e923bfe985a05',
                                                                              'spans': [(231,
                                                                                         232,
                                                                                         '            '
                                                                                         'key=lambda '
                                                                                         'pose: '
                                                                                         'float(pose.final_rescore '
                                                                                         'if '
                                                                                         'pose.final_rescore '
                                                                                         'is not '
                                                                                         'None '
                                                                                         'else '
                                                                                         'float("-inf")),\n')]},
 'dockingpp/dockingpp/pipeline/execucao/busca_execucao/rescoring_artefatos.py': {'after_sha256': 'a42b0e51346dff1ffba3b1b15cab0c92b70819e5f503d307e4d45195b0031035',
                                                                                 'before_sha256': 'fa23226d2fb0e121075c9a7f998ab70badccd663c7e9310acd01b18dadc10ae3',
                                                                                 'spans': [(153,
                                                                                            154,
                                                                                            '    '
                                                                                            'value '
                                                                                            '= '
                                                                                            'getattr(pose, '
                                                                                            '"score_cheap", '
                                                                                            'None)\n'
                                                                                            '    '
                                                                                            'return '
                                                                                            'float(value '
                                                                                            'if '
                                                                                            'value '
                                                                                            'is '
                                                                                            'not '
                                                                                            'None '
                                                                                            'else '
                                                                                            'float("-inf"))\n')]},
 'dockingpp/dockingpp/pipeline/execucao/selecao_bolsoes.py': {'after_sha256': '3a47e143d26c8f8326c287ecfe52d8e0e9b828e52d2f1b8bf60f63d8c2996ed0',
                                                              'before_sha256': '7ed721ec85a642a51e77637e875517938716fa0bc8422dd4f87d0a1f39fd1db4',
                                                              'spans': [(152,
                                                                         153,
                                                                         '        local_refinement '
                                                                         '= getattr(cfg, '
                                                                         '"docking_task", '
                                                                         '"surface_discovery") == '
                                                                         '"local_refinement"\n'
                                                                         '        if '
                                                                         'local_refinement:\n'
                                                                         '            # The '
                                                                         'supplied complex defines '
                                                                         'an explicit local '
                                                                         'refinement task.\n'
                                                                         '            # Use the '
                                                                         'complete receptor for '
                                                                         'scoring in both arms, '
                                                                         'and let the\n'
                                                                         '            # rigid '
                                                                         'search space apply the '
                                                                         'audited nested motion '
                                                                         'bounds.\n'
                                                                         '            from '
                                                                         'dockingpp.pipeline '
                                                                         'import run as run_mod\n'
                                                                         '\n'
                                                                         '            local_pocket '
                                                                         '= '
                                                                         'run_mod._build_global_pocket(receptor, '
                                                                         'cfg)\n'
                                                                         '            '
                                                                         'local_pocket.id = '
                                                                         '"local_refinement"\n'
                                                                         '            '
                                                                         'local_pocket.meta["detection_mode"] '
                                                                         '= '
                                                                         '"supplied_pose_local_refinement"\n'
                                                                         '            '
                                                                         'local_pocket.meta["input_relative_placement_used"] '
                                                                         '= True\n'
                                                                         '            '
                                                                         'loaded_pockets = '
                                                                         '[local_pocket]\n'
                                                                         '        elif '
                                                                         'search_space_mode == '
                                                                         '"full" and not '
                                                                         'modo_legado_pockets:\n'),
                                                                        (176,
                                                                         177,
                                                                         '        if '
                                                                         'local_refinement or '
                                                                         '(search_space_mode == '
                                                                         '"full" and not '
                                                                         'modo_legado_pockets):\n'),
                                                                        (228,
                                                                         229,
                                                                         '        # Surface '
                                                                         'viability must use the '
                                                                         'same final collision '
                                                                         'veto as search.\n'
                                                                         '        # Keep the '
                                                                         'original fractional '
                                                                         'proxy as an additional '
                                                                         'filter, not as a\n'
                                                                         '        # replacement '
                                                                         'for the whole-pose hard '
                                                                         'gate.\n'
                                                                         '        final_gate_cfg = '
                                                                         'self._obter_valor_cfg(cfg, '
                                                                         '"rescoring", {}) or {}\n'
                                                                         '        scan_cfg = '
                                                                         '_ScanConfigOverlay(scan_cfg, '
                                                                         '{\n'
                                                                         '            '
                                                                         '"surface_anchor_enabled": '
                                                                         'str(getattr(cfg, '
                                                                         '"search_anchor_mode", '
                                                                         '"auto") or '
                                                                         '"auto").lower() == '
                                                                         '"auto",\n'
                                                                         '            '
                                                                         '"hard_clash_gate_enabled": '
                                                                         'self._obter_valor_cfg(final_gate_cfg, '
                                                                         '"hard_clash_gate_enabled", '
                                                                         'True),\n'
                                                                         '            '
                                                                         '"hard_clash_min_distance": '
                                                                         'self._obter_valor_cfg(final_gate_cfg, '
                                                                         '"hard_clash_min_distance", '
                                                                         '1.2),\n'
                                                                         '        })\n'
                                                                         '        scan_enabled = '
                                                                         'bool(self._obter_valor_cfg(scan_cfg, '
                                                                         '"enabled", False)) and '
                                                                         'search_space_mode == '
                                                                         '"reduced" and not '
                                                                         'local_refinement\n'),
                                                                        (285,
                                                                         285,
                                                                         '            '
                                                                         '"docking_task": '
                                                                         'getattr(cfg, '
                                                                         '"docking_task", '
                                                                         '"surface_discovery"),\n'
                                                                         '            '
                                                                         '"not_applicable_reason": '
                                                                         '"supplied_pose_local_refinement" '
                                                                         'if local_refinement else '
                                                                         'None,\n'),
                                                                        (421,
                                                                         422,
                                                                         '        if '
                                                                         'search_space_mode == '
                                                                         '"reduced" and not '
                                                                         'local_refinement and '
                                                                         'bool(getattr(cfg, '
                                                                         '"usar_reducao_condicionada_ao_peptideo", '
                                                                         'False)):\n')]},
 'dockingpp/dockingpp/pipeline/logging.py': {'after_sha256': '309da535f7e9c438dfb583c49e2c0848327ceaebf89db78f0cf34d9c2669d8bb',
                                             'before_sha256': 'ff682465dfb3d410014ab3319763957ac789441d30268fb4032e97d9b82e4e2d',
                                             'spans': [(12,
                                                        13,
                                                        'from typing import Any, Dict, List, '
                                                        'Optional, TextIO\n'),
                                                       (20,
                                                        22,
                                                        '    out_dir: str | None = None\n'
                                                        '    live_write: bool = False\n'
                                                        '    _live_handle: TextIO | None = '
                                                        'field(default=None, init=False, '
                                                        'repr=False, compare=False)\n'
                                                        '    _live_path: str | None = '
                                                        'field(default=None, init=False, '
                                                        'repr=False, compare=False)\n'),
                                                       (42,
                                                        48,
                                                        '    def _append_record(self, payload: '
                                                        'Dict[str, Any]) -> None:\n'
                                                        '        if not self.live_write or not '
                                                        'self.out_dir:\n'
                                                        '            self.close()\n'
                                                        '            return\n'
                                                        '        path = '
                                                        'f"{self.out_dir}/metrics.jsonl"\n'
                                                        '        try:\n'
                                                        '            line = json.dumps(payload) + '
                                                        '"\\n"\n'
                                                        '            if self._live_handle is None '
                                                        'or self._live_path != path:\n'
                                                        '                self.close()\n'
                                                        '                self._live_handle = '
                                                        'open(path, "a", encoding="utf-8")\n'
                                                        '                self._live_path = path\n'
                                                        '            '
                                                        'self._live_handle.write(line)\n'
                                                        '            # Keep each completed metric '
                                                        'visible to live readers while avoiding\n'
                                                        '            # a file open/close (and '
                                                        'cloud-folder notification) for every '
                                                        'row.\n'
                                                        '            self._live_handle.flush()\n'
                                                        '        except BaseException:\n'
                                                        '            try:\n'
                                                        '                self.close()\n'
                                                        '            except Exception:\n'
                                                        '                pass\n'
                                                        '            raise\n'
                                                        '\n'
                                                        '    def close(self) -> None:\n'
                                                        '        handle, self._live_handle = '
                                                        'self._live_handle, None\n'
                                                        '        self._live_path = None\n'
                                                        '        if handle is not None:\n'
                                                        '            handle.close()\n'
                                                        '\n'
                                                        '    def __del__(self) -> None:\n'
                                                        '        try:\n'
                                                        '            self.close()\n'
                                                        '        except Exception:\n'
                                                        '            pass\n'),
                                                       (57,
                                                        59,
                                                        '    def flush(self, out_dir: str) -> '
                                                        'None:\n'
                                                        '        # Windows must release the live '
                                                        'append handle before the final rewrite.\n'
                                                        '        self.close()\n'
                                                        '        path = '
                                                        'f"{out_dir}/metrics.jsonl"\n')]},
 'dockingpp/dockingpp/pipeline/scan.py': {'after_sha256': '63ec761a86b2ee640f06434822716b81680862927b2aa77d8bd1d67c420f2b6f',
                                          'before_sha256': '04166d625c5f721fcf2e07beb4146a7e13b485d2fe15767b2afd431df641ef68',
                                          'spans': [(276,
                                                     276,
                                                     '    contact_point: np.ndarray | None = '
                                                     'None,\n'),
                                                    (339,
                                                     340,
                                                     '    # For a surface site, the region '
                                                     'constrains the actual contacting support\n'
                                                     "    # atom; a large mobile partner's "
                                                     'centroid is naturally outside that region.\n'
                                                     '    centroid = coords.mean(axis=0) if '
                                                     'contact_point is None else '
                                                     'np.asarray(contact_point, dtype=float)\n'),
                                                    (367,
                                                     367,
                                                     '    sample_proxy_is_feasible = '
                                                     'sample_is_feasible\n'
                                                     '    hard_gate_enabled = contact_point is not '
                                                     'None and bool(_cfg_value(cfg_scan, '
                                                     '"hard_clash_gate_enabled", True))\n'
                                                     '    hard_gate_threshold = '
                                                     '_cfg_float(cfg_scan, '
                                                     '"hard_clash_min_distance", 1.2)\n'
                                                     '    sample_hard_clash_gate_passed = not '
                                                     'hard_gate_enabled or min_d >= '
                                                     'hard_gate_threshold\n'
                                                     '    sample_is_feasible = sample_is_feasible '
                                                     'and sample_hard_clash_gate_passed\n'),
                                                    (385,
                                                     385,
                                                     '        "sample_proxy_is_feasible": '
                                                     'sample_proxy_is_feasible,\n'
                                                     '        "sample_hard_clash_gate_passed": '
                                                     'sample_hard_clash_gate_passed,\n'
                                                     '        "sample_hard_clash_threshold": '
                                                     'hard_gate_threshold if hard_gate_enabled '
                                                     'else None,\n'),
                                                    (419,
                                                     419,
                                                     '\n'
                                                     '\n'
                                                     'def _feasible_pose_statistics(sample_stats: '
                                                     'list[dict[str, Any]]) -> dict[str, Any]:\n'
                                                     '    """Describe usable poses without '
                                                     'rewarding contacts from rejected samples.\n'
                                                     '\n'
                                                     '    ``sample_is_feasible`` already includes '
                                                     'the surface hard-clash veto. Keep\n'
                                                     '    the statistics of all attempts '
                                                     'separately for acceptance and diagnostics.\n'
                                                     '    """\n'
                                                     '\n'
                                                     '    accepted = [item for item in '
                                                     'sample_stats if '
                                                     'bool(item.get("sample_is_feasible", '
                                                     'False))]\n'
                                                     '    if not accepted:\n'
                                                     '        return {\n'
                                                     '            "n_samples": 0,\n'
                                                     '            "energy_best": 100.0,\n'
                                                     '            "energy_mean": 100.0,\n'
                                                     '            "energy_std": 0.0,\n'
                                                     '            "energy_q20": 100.0,\n'
                                                     '            "mean_contacts": 0.0,\n'
                                                     '            "contact_fraction_mean": 0.0,\n'
                                                     '            "best_contact_count": 0.0,\n'
                                                     '            "moderate_clash_mean": 1.0,\n'
                                                     '            "mean_exposure_penalty": 10.0,\n'
                                                     '            "best_sample_meta": None,\n'
                                                     '        }\n'
                                                     '    energies = '
                                                     'np.asarray([float(item.get("energy_sample", '
                                                     '100.0)) for item in accepted])\n'
                                                     '    contacts = '
                                                     'np.asarray([float(item.get("raw_contact_count", '
                                                     '0.0)) for item in accepted])\n'
                                                     '    return {\n'
                                                     '        "n_samples": len(accepted),\n'
                                                     '        "energy_best": '
                                                     'float(np.min(energies)),\n'
                                                     '        "energy_mean": '
                                                     'float(np.mean(energies)),\n'
                                                     '        "energy_std": '
                                                     'float(np.std(energies)),\n'
                                                     '        "energy_q20": '
                                                     'float(np.quantile(energies, 0.20)),\n'
                                                     '        "mean_contacts": '
                                                     'float(np.mean(contacts)),\n'
                                                     '        "contact_fraction_mean": '
                                                     'float(np.mean([float(item.get("frac_contact", '
                                                     '0.0)) for item in accepted])),\n'
                                                     '        "best_contact_count": '
                                                     'float(np.max(contacts)),\n'
                                                     '        "moderate_clash_mean": '
                                                     'float(np.mean([float(item.get("frac_moderate", '
                                                     '1.0)) for item in accepted])),\n'
                                                     '        "mean_exposure_penalty": '
                                                     'float(np.mean([float(item.get("pen_exp", '
                                                     '0.0)) for item in accepted])),\n'
                                                     '        "best_sample_meta": '
                                                     'dict(accepted[int(np.argmin(energies))]),\n'
                                                     '    }\n'),
                                                    (460,
                                                     460,
                                                     '            "feasible_pose_statistics": '
                                                     '_feasible_pose_statistics([]),\n'
                                                     '            "ranking_statistics_scope": '
                                                     '"feasible_samples_v1",\n'),
                                                    (520,
                                                     520,
                                                     '        "feasible_pose_statistics": '
                                                     '_feasible_pose_statistics(sample_stats),\n'
                                                     '        "ranking_statistics_scope": '
                                                     '"feasible_samples_v1",\n'),
                                                    (550,
                                                     551,
                                                     '    # Older scan tables have only aggregate '
                                                     'fields; retain that read contract.\n'
                                                     '    pose_stats = '
                                                     'core_stats.get("feasible_pose_statistics")\n'
                                                     '    if not isinstance(pose_stats, dict):\n'
                                                     '        pose_stats = core_stats\n'
                                                     '    normalized_best_contact = '
                                                     'float(pose_stats.get("best_contact_count", '
                                                     '0.0)) / float(n_atoms)\n'),
                                                    (553,
                                                     556,
                                                     '        "energy_q20": '
                                                     'float(pose_stats.get("energy_q20", 100.0)),\n'
                                                     '        "energy_mean": '
                                                     'float(pose_stats.get("energy_mean", '
                                                     '100.0)),\n'
                                                     '        "energy_std": '
                                                     'float(pose_stats.get("energy_std", 0.0)),\n'),
                                                    (557,
                                                     558,
                                                     '        "contact_fraction_mean": '
                                                     'float(pose_stats.get("contact_fraction_mean", '
                                                     '0.0)),\n'),
                                                    (560,
                                                     562,
                                                     '        "moderate_clash_mean": '
                                                     'float(pose_stats.get("moderate_clash_mean", '
                                                     '1.0)),\n'
                                                     '        "mean_exposure_penalty": '
                                                     'float(pose_stats.get("mean_exposure_penalty", '
                                                     '10.0)),\n'),
                                                    (581,
                                                     581,
                                                     '        "pose_statistics_scope": '
                                                     'core_stats.get("ranking_statistics_scope", '
                                                     '"all_samples_legacy"),\n'),
                                                    (626,
                                                     626,
                                                     '        from dockingpp.utils.geometry import '
                                                     'place_surface_contact, '
                                                     'surface_anchor_normal\n'
                                                     '        surface_normal = '
                                                     'surface_anchor_normal(pocket) if '
                                                     '_cfg_value(cfg_scan, '
                                                     '"surface_anchor_enabled", True) else None\n'),
                                                    (631,
                                                     631,
                                                     '            contact_point = None\n'
                                                     '            support_index = None\n'
                                                     '            if surface_normal is not None:\n'
                                                     '                contact_point = '
                                                     'np.asarray(transform["translation"], '
                                                     'dtype=float)\n'
                                                     '                placed, support_index = '
                                                     'place_surface_contact(\n'
                                                     '                    coords, '
                                                     'transform["rotation"], contact_point, '
                                                     'surface_normal,\n'
                                                     '                )\n'),
                                                    (637,
                                                     637,
                                                     '                '
                                                     'contact_point=contact_point,\n'),
                                                    (643,
                                                     643,
                                                     '            if surface_normal is not None:\n'
                                                     '                '
                                                     'stats["surface_anchor_version"] = '
                                                     '"surface_contact_v1"\n'
                                                     '                '
                                                     'stats["surface_support_atom_index"] = '
                                                     'support_index\n'
                                                     '                '
                                                     'stats["surface_contact_point"] = '
                                                     'contact_point.tolist()\n'
                                                     '                stats["placed_centroid"] = '
                                                     'placed.mean(axis=0).tolist()\n'),
                                                    (699,
                                                     699,
                                                     '        "feasible_pose_statistics": '
                                                     'core_stats.get("feasible_pose_statistics"),\n'
                                                     '        "ranking_statistics_scope": '
                                                     'core_stats.get("ranking_statistics_scope", '
                                                     '"all_samples_legacy"),\n'),
                                                    (748,
                                                     748,
                                                     '        "feasible_pose_statistics": '
                                                     'core_stats.get("feasible_pose_statistics"),\n'
                                                     '        "ranking_statistics_scope": '
                                                     'core_stats.get("ranking_statistics_scope", '
                                                     '"all_samples_legacy"),\n'),
                                                    (836,
                                                     837,
                                                     '    2. Mean receptor-contact fraction of '
                                                     'feasible samples descending.\n'),
                                                    (839,
                                                     840,
                                                     '    5. Robust energy quantile of feasible '
                                                     'samples ascending.\n'),
                                                    (851,
                                                     852,
                                                     '    pose_stats = '
                                                     'metrics.get("feasible_pose_statistics")\n'
                                                     '    if not isinstance(pose_stats, dict):\n'
                                                     '        pose_stats = metrics\n'
                                                     '    contact = '
                                                     '_safe_sort_float(pose_stats.get("contact_fraction_mean", '
                                                     '0.0), 0.0)\n'),
                                                    (853,
                                                     854,
                                                     '    energy_q20 = '
                                                     '_safe_sort_float(pose_stats.get("energy_q20", '
                                                     'float("inf")), float("inf"))\n'),
                                                    (877, 880, ''),
                                                    (893,
                                                     894,
                                                     '    # The selection count must not change '
                                                     'the ranking contract. This order\n'
                                                     '    # also controls which pocket wins under '
                                                     'the scan_priority policy.\n'
                                                     '    limit = len(scored) if top_k is None or '
                                                     'int(top_k) <= 0 else int(top_k)\n'
                                                     '    return [item[1] for item in '
                                                     'scored[:limit]]\n')]},
 'dockingpp/dockingpp/scoring/expensive.py': {'after_sha256': 'a16fbc2591493a12be6555671df0828ea64529cb10c6e24f3d96c7ac5eaf9a5e',
                                              'before_sha256': 'add10f2c07a6f83c228c47b5f35bcd4b910ba4c23d07a8af7ead9f5e8296b4d1',
                                              'spans': [(343,
                                                         343,
                                                         '        receptor_coords = (\n'
                                                         '            np.asarray([item[0] for item '
                                                         'in receptor_prepped], dtype=float)\n'
                                                         '            if all(item[0].shape == (3,) '
                                                         'for item in receptor_prepped)\n'
                                                         '            else None\n'
                                                         '        )\n'),
                                                        (350,
                                                         351,
                                                         '            for receptor_idx in '
                                                         '_candidate_receptor_indices(pi, '
                                                         'receptor_coords, self.cutoff_pair, '
                                                         'len(receptor_prepped)):\n'
                                                         '                pj, tj, qj = '
                                                         'receptor_prepped[receptor_idx]\n'),
                                                        (436,
                                                         436,
                                                         '\n'
                                                         '\n'
                                                         'def _candidate_receptor_indices(\n'
                                                         '    point: np.ndarray,\n'
                                                         '    receptor_coords: np.ndarray | None,\n'
                                                         '    cutoff: float,\n'
                                                         '    receptor_count: int,\n'
                                                         '):\n'
                                                         '    """Prune only pairs outside a '
                                                         'conservative enclosing cube.\n'
                                                         '\n'
                                                         '    Retained pairs still use the '
                                                         'original scalar norm, cutoff test, terms '
                                                         'and\n'
                                                         '    accumulation order. The outward '
                                                         'margin protects cutoff-boundary '
                                                         'rounding;\n'
                                                         '    nonfinite coordinates and extreme '
                                                         'cutoffs retain the original path.\n'
                                                         '    """\n'
                                                         '\n'
                                                         '    if (\n'
                                                         '        receptor_coords is None\n'
                                                         '        or receptor_coords.shape != '
                                                         '(receptor_count, 3)\n'
                                                         '        or np.asarray(point).shape != '
                                                         '(3,)\n'
                                                         '        or not np.isfinite(cutoff)\n'
                                                         '        or cutoff < 1e-100\n'
                                                         '        or cutoff > 1e100\n'
                                                         '    ):\n'
                                                         '        return range(receptor_count)\n'
                                                         '    margin = max(1.0, abs(cutoff)) * '
                                                         '(16.0 * np.finfo(float).eps)\n'
                                                         '    deltas = point - receptor_coords\n'
                                                         '    inside = np.all(np.abs(deltas) <= '
                                                         'cutoff + margin, axis=1)\n'
                                                         '    # The scalar implementation '
                                                         'propagates NaN distances; do not hide '
                                                         'them.\n'
                                                         '    nonfinite = '
                                                         'np.any(~np.isfinite(deltas), axis=1)\n'
                                                         '    return np.flatnonzero(inside | '
                                                         'nonfinite)\n')]},
 'dockingpp/dockingpp/search/algoritmos/_ciclo_populacional.py': {'after_sha256': '25cd88eb993b7d663d73c3a6fc32f9cb2dd1ad66c5a9ce7a43b90485357ed682',
                                                                  'before_sha256': 'f6f12e81f3524755886317861ffe7ae489dff3a44c7c61e77eb6ff188faa2f1e',
                                                                  'spans': [(32,
                                                                             32,
                                                                             '    '
                                                                             'initialize_population,\n'
                                                                             '    '
                                                                             'population_generations,\n'
                                                                             '    '
                                                                             'require_fixed_evaluation_completion,\n'
                                                                             '    '
                                                                             'uses_fixed_evaluations,\n'),
                                                                            (170,
                                                                             171,
                                                                             '        poses, '
                                                                             'initial_audit = '
                                                                             'initialize_population(search_space, '
                                                                             'settings.pop_size, '
                                                                             'cfg)\n'),
                                                                            (172,
                                                                             172,
                                                                             '        '
                                                                             'runtime.initial_population_audit '
                                                                             '= initial_audit\n'
                                                                             '        if '
                                                                             'initial_audit is not '
                                                                             'None:\n'
                                                                             '            '
                                                                             'logger.log_metric("search.initial_population", '
                                                                             'settings.pop_size, '
                                                                             'step=step_offset,\n'
                                                                             '                              '
                                                                             'extra={"pocket_id": '
                                                                             'str(pocket.id), '
                                                                             '"initial_population_audit": '
                                                                             'initial_audit})\n'),
                                                                            (203,
                                                                             204,
                                                                             '        for '
                                                                             'generation in '
                                                                             'population_generations(search_space, '
                                                                             'settings.generations):\n'),
                                                                            (261,
                                                                             261,
                                                                             '        '
                                                                             'require_fixed_evaluation_completion(search_space)\n'),
                                                                            (776,
                                                                             777,
                                                                             '                not '
                                                                             'uses_fixed_evaluations(environment.cfg)\n'
                                                                             '                and '
                                                                             'hybrid_stop.patience '
                                                                             '> 0\n'),
                                                                            (881,
                                                                             882,
                                                                             '        if '
                                                                             '(confidence.search_control_enabled '
                                                                             'and '
                                                                             'confidence.early_stop_enabled\n'
                                                                             '                and '
                                                                             'not '
                                                                             'uses_fixed_evaluations(environment.cfg)):\n')]},
 'dockingpp/dockingpp/search/algoritmos/_configuracao_populacional.py': {'after_sha256': 'c0c62c065868433aff1110923331413d2601aeccfe9f8436c7adea1c9043ae2a',
                                                                         'before_sha256': '5fc78fe519ddb6600973a04f9df695af11db60a02e4b927dc102350c92758207',
                                                                         'spans': [(10,
                                                                                    11,
                                                                                    'import '
                                                                                    'hashlib\n'
                                                                                    'from '
                                                                                    'itertools '
                                                                                    'import count\n'
                                                                                    'import json\n'
                                                                                    'from typing '
                                                                                    'import Any, '
                                                                                    'Iterator\n'),
                                                                                   (26,
                                                                                    26,
                                                                                    '\n'
                                                                                    '\n'
                                                                                    'def '
                                                                                    'audit_sha256(payload: '
                                                                                    'Any) -> str:\n'
                                                                                    '    """Hash '
                                                                                    'deterministic '
                                                                                    'scientific '
                                                                                    'records '
                                                                                    'without '
                                                                                    'consuming the '
                                                                                    'RNG."""\n'
                                                                                    '\n'
                                                                                    '    data = '
                                                                                    'json.dumps(payload, '
                                                                                    'sort_keys=True, '
                                                                                    'separators=(",", '
                                                                                    '":"),\n'
                                                                                    '                      '
                                                                                    'ensure_ascii=False, '
                                                                                    'allow_nan=False).encode("utf-8")\n'
                                                                                    '    return '
                                                                                    'hashlib.sha256(data).hexdigest()\n'
                                                                                    '\n'
                                                                                    '\n'
                                                                                    'MAX_GENERATIONS_WITHOUT_NEW_EVALUATIONS '
                                                                                    '= 50\n'
                                                                                    '\n'
                                                                                    '\n'
                                                                                    'def '
                                                                                    'uses_fixed_evaluations(cfg: '
                                                                                    'Any) -> '
                                                                                    'bool:\n'
                                                                                    '    return '
                                                                                    'getattr(cfg, '
                                                                                    '"experiment_budget_policy", '
                                                                                    'None) == '
                                                                                    '"fixed_evaluations"\n'
                                                                                    '\n'
                                                                                    '\n'
                                                                                    'def '
                                                                                    'population_generations(search_space: '
                                                                                    'Any, '
                                                                                    'generations: '
                                                                                    'int) -> '
                                                                                    'Iterator[int]:\n'
                                                                                    '    """Keep '
                                                                                    'native '
                                                                                    'operators '
                                                                                    'running until '
                                                                                    'the real '
                                                                                    'objective '
                                                                                    'budget is '
                                                                                    'consumed."""\n'
                                                                                    '    if not '
                                                                                    'uses_fixed_evaluations(search_space.cfg):\n'
                                                                                    '        yield '
                                                                                    'from '
                                                                                    'range(generations '
                                                                                    '+ 1)\n'
                                                                                    '        '
                                                                                    'return\n'
                                                                                    '    if '
                                                                                    'search_space.evaluation_budget '
                                                                                    'is None:\n'
                                                                                    '        raise '
                                                                                    'ValueError("fixed_evaluations '
                                                                                    'requires an '
                                                                                    'objective '
                                                                                    'evaluation '
                                                                                    'budget")\n'
                                                                                    '    '
                                                                                    'without_progress '
                                                                                    '= 0\n'
                                                                                    '    for '
                                                                                    'generation in '
                                                                                    'count():\n'
                                                                                    '        if '
                                                                                    'search_space.evaluation_budget_exhausted:\n'
                                                                                    '            '
                                                                                    'return\n'
                                                                                    '        '
                                                                                    'before = '
                                                                                    'search_space.actual_eval_count\n'
                                                                                    '        yield '
                                                                                    'generation\n'
                                                                                    '        '
                                                                                    'without_progress '
                                                                                    '= '
                                                                                    'without_progress '
                                                                                    '+ 1 if before '
                                                                                    '== '
                                                                                    'search_space.actual_eval_count '
                                                                                    'else 0\n'
                                                                                    '        if '
                                                                                    'without_progress '
                                                                                    '>= '
                                                                                    'MAX_GENERATIONS_WITHOUT_NEW_EVALUATIONS:\n'
                                                                                    '            '
                                                                                    'raise '
                                                                                    'RuntimeError(\n'
                                                                                    '                '
                                                                                    'f"Orçamento '
                                                                                    'fixo não '
                                                                                    'atingido: '
                                                                                    '{search_space.actual_eval_count}/"\n'
                                                                                    '                '
                                                                                    'f"{search_space.evaluation_budget} '
                                                                                    'avaliações '
                                                                                    'efetivas após '
                                                                                    '"\n'
                                                                                    '                '
                                                                                    'f"{without_progress} '
                                                                                    'gerações sem '
                                                                                    'novas '
                                                                                    'avaliações."\n'
                                                                                    '            '
                                                                                    ')\n'
                                                                                    '\n'
                                                                                    '\n'
                                                                                    'def '
                                                                                    'require_fixed_evaluation_completion(search_space: '
                                                                                    'Any) -> '
                                                                                    'None:\n'
                                                                                    '    if '
                                                                                    'uses_fixed_evaluations(search_space.cfg) '
                                                                                    'and not '
                                                                                    'search_space.evaluation_budget_exhausted:\n'
                                                                                    '        raise '
                                                                                    'RuntimeError(\n'
                                                                                    '            '
                                                                                    'f"Orçamento '
                                                                                    'fixo não '
                                                                                    'atingido: '
                                                                                    '{search_space.actual_eval_count}/"\n'
                                                                                    '            '
                                                                                    'f"{search_space.evaluation_budget} '
                                                                                    'avaliações '
                                                                                    'efetivas."\n'
                                                                                    '        )\n'
                                                                                    '\n'
                                                                                    '\n'
                                                                                    'def '
                                                                                    'initialize_population(\n'
                                                                                    '    '
                                                                                    'search_space: '
                                                                                    'Any, '
                                                                                    'pop_size: '
                                                                                    'int, cfg: '
                                                                                    'Any, *, '
                                                                                    'seeded_by_default: '
                                                                                    'bool = '
                                                                                    'False,\n'
                                                                                    ') -> '
                                                                                    'tuple[list[Any], '
                                                                                    'dict[str, '
                                                                                    'Any] | '
                                                                                    'None]:\n'
                                                                                    '    """Sample '
                                                                                    'identically, '
                                                                                    'replacing '
                                                                                    'only '
                                                                                    'candidate '
                                                                                    'zero for '
                                                                                    'seeded runs.\n'
                                                                                    '\n'
                                                                                    '    Sampling '
                                                                                    'the same '
                                                                                    'number of '
                                                                                    'vectors in '
                                                                                    'both arms '
                                                                                    'preserves '
                                                                                    'paired RNG\n'
                                                                                    '    streams. '
                                                                                    'The supplied '
                                                                                    'pose is an '
                                                                                    'ordinary '
                                                                                    'candidate, '
                                                                                    'evaluated '
                                                                                    'normally.\n'
                                                                                    '    """\n'
                                                                                    '\n'
                                                                                    '    '
                                                                                    'initialization '
                                                                                    '= '
                                                                                    'getattr(cfg, '
                                                                                    '"initialization", '
                                                                                    'None)\n'
                                                                                    '    if '
                                                                                    'initialization '
                                                                                    'is not None:\n'
                                                                                    '        if '
                                                                                    'initialization '
                                                                                    'not in '
                                                                                    '{"uniform", '
                                                                                    '"seeded"}:\n'
                                                                                    '            '
                                                                                    'raise '
                                                                                    'ValueError("initialization '
                                                                                    'must be '
                                                                                    'uniform or '
                                                                                    'seeded")\n'
                                                                                    '        if '
                                                                                    'not '
                                                                                    'search_space.local_refinement:\n'
                                                                                    '            '
                                                                                    'raise '
                                                                                    'ValueError("independent '
                                                                                    'initialization '
                                                                                    'requires '
                                                                                    'local_refinement")\n'
                                                                                    '    seeded = '
                                                                                    'initialization '
                                                                                    '== "seeded" '
                                                                                    'or (\n'
                                                                                    '        '
                                                                                    'initialization '
                                                                                    'is None and '
                                                                                    'seeded_by_default\n'
                                                                                    '    )\n'
                                                                                    '    poses = '
                                                                                    '[search_space.random_pose() '
                                                                                    'for _ in '
                                                                                    'range(pop_size)]\n'
                                                                                    '    if '
                                                                                    'seeded:\n'
                                                                                    '        '
                                                                                    'poses[0] = '
                                                                                    'search_space.solution_to_pose(np.zeros(search_space.dim, '
                                                                                    'dtype=float))\n'
                                                                                    '        '
                                                                                    'poses[0].meta["initialization_source"] '
                                                                                    '= '
                                                                                    '"supplied_pose_identity"\n'
                                                                                    '    if '
                                                                                    'initialization '
                                                                                    'is None:\n'
                                                                                    '        '
                                                                                    'return poses, '
                                                                                    'None\n'
                                                                                    '    record = '
                                                                                    '{\n'
                                                                                    '        '
                                                                                    '"schema": '
                                                                                    '"factorial_initial_population_v1",\n'
                                                                                    '        '
                                                                                    '"seed": '
                                                                                    'int(cfg.seed),\n'
                                                                                    '        '
                                                                                    '"initialization": '
                                                                                    'initialization,\n'
                                                                                    '        '
                                                                                    '"search_extent": '
                                                                                    'getattr(cfg, '
                                                                                    '"search_extent", '
                                                                                    'None),\n'
                                                                                    '        '
                                                                                    '"pose_source": '
                                                                                    'getattr(cfg, '
                                                                                    '"pose_source", '
                                                                                    'None),\n'
                                                                                    '        '
                                                                                    '"input_pose_coordinates_sha256": '
                                                                                    'audit_sha256(search_space.base_coords.tolist()),\n'
                                                                                    '        '
                                                                                    '"domain_low": '
                                                                                    'search_space.domain_low.tolist(),\n'
                                                                                    '        '
                                                                                    '"domain_high": '
                                                                                    'search_space.domain_high.tolist(),\n'
                                                                                    '        '
                                                                                    '"vector_order": '
                                                                                    '["tx", "ty", '
                                                                                    '"tz", '
                                                                                    '"rx_degrees", '
                                                                                    '"ry_degrees", '
                                                                                    '"rz_degrees"],\n'
                                                                                    '        '
                                                                                    '"sampling": '
                                                                                    '"independent_uniform_coordinates_within_rigid_domain",\n'
                                                                                    '        '
                                                                                    '"population_size": '
                                                                                    'pop_size,\n'
                                                                                    '        '
                                                                                    '"seeded_candidate_count": '
                                                                                    '1 if seeded '
                                                                                    'else 0,\n'
                                                                                    '        '
                                                                                    '"solution_vectors": '
                                                                                    '[list(pose.meta["solution_vector"]) '
                                                                                    'for pose in '
                                                                                    'poses],\n'
                                                                                    '        '
                                                                                    '"recorded_before_objective_evaluations": '
                                                                                    'int(search_space.actual_eval_count),\n'
                                                                                    '        '
                                                                                    '"stop_rules": '
                                                                                    '{\n'
                                                                                    '            '
                                                                                    '"generations": '
                                                                                    'int(cfg.generations),\n'
                                                                                    '            '
                                                                                    '"objective_evaluation_ceiling": '
                                                                                    'search_space.evaluation_budget,\n'
                                                                                    '            '
                                                                                    '"budget_policy": '
                                                                                    'getattr(cfg, '
                                                                                    '"experiment_budget_policy", '
                                                                                    'None),\n'
                                                                                    '            '
                                                                                    '"hybrid_plateau_patience": '
                                                                                    'int(getattr(cfg, '
                                                                                    '"hybrid_plateau_patience", '
                                                                                    '80)),\n'
                                                                                    '            '
                                                                                    '"hybrid_plateau_eps": '
                                                                                    'float(getattr(cfg, '
                                                                                    '"hybrid_plateau_eps", '
                                                                                    '1e-6)),\n'
                                                                                    '            '
                                                                                    '"confidence_search_control_enabled": '
                                                                                    'confidence_search_control_enabled(cfg),\n'
                                                                                    '        },\n'
                                                                                    '    }\n'
                                                                                    '    if '
                                                                                    'uses_fixed_evaluations(cfg):\n'
                                                                                    '        '
                                                                                    'record["stop_rules"].update(\n'
                                                                                    '            '
                                                                                    'generation_limit_enabled=False,\n'
                                                                                    '            '
                                                                                    'max_generations_without_new_evaluations=MAX_GENERATIONS_WITHOUT_NEW_EVALUATIONS,\n'
                                                                                    '        )\n'
                                                                                    '    return '
                                                                                    'poses, '
                                                                                    '{**record, '
                                                                                    '"sha256": '
                                                                                    'audit_sha256(record)}\n')]},
 'dockingpp/dockingpp/search/algoritmos/_estado_populacional.py': {'after_sha256': 'fba0770deff59a7455a650dde1e962253001f263ef68587acc2710ea7c878727',
                                                                   'before_sha256': '823a1f473f192b6474a91581da51fb69b87b4f3c264c70ede41c68496c974083',
                                                                   'spans': [(59,
                                                                              59,
                                                                              '    '
                                                                              'initial_population_audit: '
                                                                              'dict[str, Any] | '
                                                                              'None = None\n')]},
 'dockingpp/dockingpp/search/algoritmos/_resumo_populacional.py': {'after_sha256': '2e83313c7d907a2eb092a8a93507b97d1d872377fd5191659344a0764c45d627',
                                                                   'before_sha256': 'e35cf17743bf57e9d7b959c105d0eee23e935ac4f30629b4acef813360069fa0',
                                                                   'spans': [(16,
                                                                              16,
                                                                              '    '
                                                                              'audit_sha256,\n'),
                                                                             (386,
                                                                              386,
                                                                              '    if '
                                                                              'runtime.initial_population_audit '
                                                                              'is not None:\n'
                                                                              '        '
                                                                              'pocket_summary["initial_population_audit"] '
                                                                              '= '
                                                                              'runtime.initial_population_audit\n'
                                                                              '        if '
                                                                              'runtime.confidence_stop_applied:\n'
                                                                              '            '
                                                                              'stop_reason = '
                                                                              '"confidence_early_stop"\n'
                                                                              '        elif '
                                                                              'runtime.objective_evaluation_budget_exhausted:\n'
                                                                              '            '
                                                                              'stop_reason = '
                                                                              '"objective_evaluation_ceiling"\n'
                                                                              '        elif '
                                                                              'runtime.hybrid_plateau_stop_applied:\n'
                                                                              '            '
                                                                              'stop_reason = '
                                                                              '"hybrid_plateau"\n'
                                                                              '        else:\n'
                                                                              '            '
                                                                              'stop_reason = '
                                                                              '"generation_limit"\n'
                                                                              '        execution = '
                                                                              '{\n'
                                                                              '            '
                                                                              '"schema": '
                                                                              '"factorial_execution_audit_v1",\n'
                                                                              '            '
                                                                              '"initial_population_sha256": '
                                                                              'runtime.initial_population_audit["sha256"],\n'
                                                                              '            "seed": '
                                                                              'int(context.cfg.seed),\n'
                                                                              '            '
                                                                              '"algorithm": '
                                                                              'context.strategy_name,\n'
                                                                              '            '
                                                                              '"budget_policy": '
                                                                              'getattr(context.cfg, '
                                                                              '"experiment_budget_policy", '
                                                                              'None),\n'
                                                                              '            '
                                                                              '"objective_evaluation_ceiling": '
                                                                              'context.search_space.evaluation_budget,\n'
                                                                              '            '
                                                                              '"objective_evaluations_actual": '
                                                                              'int(context.search_space.actual_eval_count),\n'
                                                                              '            '
                                                                              '"objective_evaluations_by_generation": '
                                                                              'actual_eval_series,\n'
                                                                              '            '
                                                                              '"evaluation_count_basis": '
                                                                              '"new_cheap_objective_calls_including_internal_operators_excluding_cache_hits_and_final_rescoring",\n'
                                                                              '            '
                                                                              '"completed_generations": '
                                                                              'completed_generations,\n'
                                                                              '            '
                                                                              '"stop_reason": '
                                                                              'stop_reason,\n'
                                                                              '            '
                                                                              '"stop_rules": '
                                                                              'runtime.initial_population_audit["stop_rules"],\n'
                                                                              '        }\n'
                                                                              '        '
                                                                              'pocket_summary["execution_audit"] '
                                                                              '= {**execution, '
                                                                              '"sha256": '
                                                                              'audit_sha256(execution)}\n')]},
 'dockingpp/dockingpp/search/algoritmos/ag.py': {'after_sha256': 'eaf0ba6b22da1e93992f9cf110c8aced7f6dc8d1a976d001656cde1740489dea',
                                                 'before_sha256': '8d8a2eff24a00c0039b7a8a236a9be8ea6712a47bc682b5e293c920a0ece8719',
                                                 'spans': [(15,
                                                            16,
                                                            'from .contexto import '
                                                            'GenerationContext, inherit_seed, '
                                                            'mark_seed_origin, seed_origin\n'),
                                                           (176,
                                                            180,
                                                            '        elite = '
                                                            'np.argsort(fitness)[:elite_count]\n'
                                                            '        next_solutions = '
                                                            '[solutions[int(index)].copy() for '
                                                            'index in elite]\n'
                                                            '        origins = [seed_origin(pose) '
                                                            'for pose in context.previous]\n'
                                                            '        next_origins = '
                                                            '[origins[int(index)] for index in '
                                                            'elite]\n'),
                                                           (186,
                                                            186,
                                                            '                parents = '
                                                            '[(origins[first], origins[second])] * '
                                                            '2\n'),
                                                           (196,
                                                            197,
                                                            '                parents = '
                                                            '[(origins[first],), '
                                                            '(origins[second],)]\n'
                                                            '            for child, child_parents '
                                                            'in zip((child_a, child_b), '
                                                            'parents):\n'),
                                                           (206,
                                                            206,
                                                            '                '
                                                            'next_origins.append(inherit_seed(mutated, '
                                                            '*child_parents))\n'),
                                                           (210,
                                                            211,
                                                            '        for solution, origin in '
                                                            'zip(next_solutions, next_origins):\n'),
                                                           (212,
                                                            212,
                                                            '            mark_seed_origin(pose, '
                                                            'origin)\n')]},
 'dockingpp/dockingpp/search/algoritmos/base.py': {'after_sha256': '7e469063290520763b0eeb393be1b6c222048dafcadcb79dfa06119c5229ea77',
                                                   'before_sha256': '35b90d5fe9aadabac34a2d26d889a4096bf7fcc06954db0acebedb86414f093c',
                                                   'spans': [(32,
                                                              32,
                                                              '    initialize_population,\n'
                                                              '    population_generations,\n'
                                                              '    '
                                                              'require_fixed_evaluation_completion,\n'
                                                              '    uses_fixed_evaluations,\n'),
                                                             (48,
                                                              49,
                                                              'from .contexto import '
                                                              'GenerationContext, SeedOrigin, '
                                                              'mark_seed_origin\n'),
                                                             (189,
                                                              189,
                                                              '        fixed_evaluations = '
                                                              'uses_fixed_evaluations(cfg)\n'),
                                                             (201,
                                                              202,
                                                              '        '
                                                              'confidence_early_stop_enabled = '
                                                              'confidence.early_stop_enabled and '
                                                              'not fixed_evaluations\n'),
                                                             (285,
                                                              286,
                                                              '            next_poses = '
                                                              'self._next_generation(context)\n'
                                                              '            logger.log_metric(\n'
                                                              '                '
                                                              '"search.operator_completed", 1, '
                                                              'step=generation_index,\n'
                                                              '                '
                                                              'extra={"engine_name": '
                                                              'type(self).__name__,\n'
                                                              '                       '
                                                              '"search_strategy": '
                                                              'self.strategy_name,\n'
                                                              '                       "seed": '
                                                              'cfg.seed, "pop_size": pop_size,\n'
                                                              '                       '
                                                              '"generations": cfg.generations,\n'
                                                              '                       "pocket_id": '
                                                              'str(pocket.id)},\n'
                                                              '            )\n'
                                                              '            return next_poses\n'),
                                                             (290,
                                                              291,
                                                              '        poses, initial_audit = '
                                                              'initialize_population(\n'
                                                              '            search_space, pop_size, '
                                                              'cfg,\n'
                                                              '            '
                                                              'seeded_by_default=(search_space.local_refinement\n'
                                                              '                               and '
                                                              'getattr(cfg, "search_space_mode", '
                                                              '"full") == "reduced"),\n'
                                                              '        )\n'),
                                                             (292,
                                                              293,
                                                              '        '
                                                              'runtime.initial_population_audit = '
                                                              'initial_audit\n'
                                                              '        track_seed = (\n'
                                                              '            initial_audit is not '
                                                              'None and '
                                                              'initial_audit["initialization"] == '
                                                              '"seeded"\n'
                                                              '            and self.strategy_name '
                                                              'in {"ag", "hybrid", "pso"}\n'
                                                              '        )\n'
                                                              '        seed_generations = []\n'
                                                              '        if track_seed:\n'
                                                              '            for index, pose in '
                                                              'enumerate(poses):\n'
                                                              '                '
                                                              'mark_seed_origin(pose, '
                                                              'SeedOrigin.UNCHANGED if index == 0 '
                                                              'else SeedOrigin.UNRELATED)\n'
                                                              '        if initial_audit is not '
                                                              'None:\n'
                                                              '            '
                                                              'logger.log_metric("search.initial_population", '
                                                              'pop_size, step=step_offset,\n'
                                                              '                              '
                                                              'extra={"pocket_id": str(pocket.id), '
                                                              '"initial_population_audit": '
                                                              'initial_audit})\n'
                                                              '        hybrid_plateau_patience = 0 '
                                                              'if fixed_evaluations else '
                                                              'hybrid_stop.patience\n'),
                                                             (309,
                                                              310,
                                                              '        for generation in '
                                                              'population_generations(search_space, '
                                                              'generations):\n'),
                                                             (386,
                                                              386,
                                                              '            if track_seed:\n'
                                                              '                '
                                                              'seed_generations.append({\n'
                                                              '                    "generation": '
                                                              'generation,\n'
                                                              '                    '
                                                              '"n_eval_cumulative": '
                                                              'int(search_space.actual_eval_count),\n'
                                                              '                    '
                                                              '"population_size": len(poses),\n'
                                                              '                    '
                                                              '"unchanged_seed_count": '
                                                              'sum(p.meta["seed_unchanged"] for p '
                                                              'in poses),\n'
                                                              '                    '
                                                              '"seed_lineage_count": '
                                                              'sum(p.meta["seed_lineage"] for p in '
                                                              'poses),\n'
                                                              '                    '
                                                              '"pso_global_best_from_seed_particle": '
                                                              'runtime.hybrid_operator_state.get(\n'
                                                              '                        '
                                                              '"pso_global_best_from_seed_particle"\n'
                                                              '                    ),\n'
                                                              '                })\n'),
                                                             (739,
                                                              739,
                                                              '        '
                                                              'require_fixed_evaluation_completion(search_space)\n'),
                                                             (780,
                                                              780,
                                                              '        if track_seed:\n'
                                                              '            lineage_audit = {\n'
                                                              '                "schema": '
                                                              '"seed_lineage_v1",\n'
                                                              '                "algorithm": '
                                                              'self.strategy_name,\n'
                                                              '                '
                                                              '"lineage_semantics": '
                                                              '("same_particle_as_supplied_initial_pose"\n'
                                                              '                                      '
                                                              'if self.strategy_name == "pso"\n'
                                                              '                                      '
                                                              'else '
                                                              '"operator_pedigree_from_supplied_initial_pose"),\n'
                                                              '                "generations": '
                                                              'seed_generations,\n'
                                                              '                '
                                                              '"search_best_seed_lineage": '
                                                              'runtime.best_pose.meta["seed_lineage"],\n'
                                                              '                '
                                                              '"search_best_unchanged_seed": '
                                                              'runtime.best_pose.meta["seed_unchanged"],\n'
                                                              '            }\n'
                                                              '            '
                                                              'pocket_summary["seed_lineage"] = '
                                                              'lineage_audit\n'
                                                              '            '
                                                              'logger.log_metric("search.seed_lineage", '
                                                              '1, step=step_offset + generation,\n'
                                                              '                              '
                                                              'extra={"pocket_id": str(pocket.id), '
                                                              '"seed_lineage": lineage_audit})\n'
                                                              '        if initial_audit is not '
                                                              'None:\n'
                                                              '            '
                                                              'logger.log_metric("search.execution_audit", '
                                                              'search_space.actual_eval_count,\n'
                                                              '                              '
                                                              'step=step_offset + '
                                                              'runtime.best_generation,\n'
                                                              '                              '
                                                              'extra={"pocket_id": '
                                                              'str(pocket.id),\n'
                                                              '                                     '
                                                              '"execution_audit": '
                                                              'pocket_summary["execution_audit"]})\n'),
                                                             (855,
                                                              856,
                                                              '            key=lambda pose: '
                                                              'float(pose.score_cheap if '
                                                              'pose.score_cheap is not None else '
                                                              'float("-inf")),\n')]},
 'dockingpp/dockingpp/search/algoritmos/contexto.py': {'after_sha256': 'ceba273f9d1c478956512f71b7a7a892bc44e17a9d3a66a5871813b059dc81c4',
                                                       'before_sha256': 'fe247f3c9944d608808d291318d1235e8d061b8be54880a171c5c78583705a01',
                                                       'spans': [(10,
                                                                  10,
                                                                  'from enum import IntEnum\n'),
                                                                 (15,
                                                                  15,
                                                                  '\n'
                                                                  '\n'
                                                                  'class SeedOrigin(IntEnum):\n'
                                                                  '    """A path from the supplied '
                                                                  'pose through the realized '
                                                                  'search operators."""\n'
                                                                  '\n'
                                                                  '    UNRELATED = 0\n'
                                                                  '    DESCENDANT = 1\n'
                                                                  '    UNCHANGED = 2\n'
                                                                  '\n'
                                                                  '\n'
                                                                  'def seed_origin(pose: Pose) -> '
                                                                  'SeedOrigin | None:\n'
                                                                  '    if not '
                                                                  'isinstance(pose.meta, dict) or '
                                                                  '"seed_lineage" not in '
                                                                  'pose.meta:\n'
                                                                  '        return None\n'
                                                                  '    if '
                                                                  'pose.meta["seed_unchanged"]:\n'
                                                                  '        return '
                                                                  'SeedOrigin.UNCHANGED\n'
                                                                  '    return '
                                                                  'SeedOrigin.DESCENDANT if '
                                                                  'pose.meta["seed_lineage"] else '
                                                                  'SeedOrigin.UNRELATED\n'
                                                                  '\n'
                                                                  '\n'
                                                                  'def inherit_seed(solution: '
                                                                  'np.ndarray, *parents: '
                                                                  'SeedOrigin | None) -> '
                                                                  'SeedOrigin | None:\n'
                                                                  '    """Retain exact survival '
                                                                  'only along an unbroken path of '
                                                                  'identity transforms."""\n'
                                                                  '    if all(parent is None for '
                                                                  'parent in parents):\n'
                                                                  '        return None\n'
                                                                  '    if SeedOrigin.UNCHANGED in '
                                                                  'parents and not '
                                                                  'np.any(solution):\n'
                                                                  '        return '
                                                                  'SeedOrigin.UNCHANGED\n'
                                                                  '    if any(parent in '
                                                                  '(SeedOrigin.UNCHANGED, '
                                                                  'SeedOrigin.DESCENDANT) for '
                                                                  'parent in parents):\n'
                                                                  '        return '
                                                                  'SeedOrigin.DESCENDANT\n'
                                                                  '    return '
                                                                  'SeedOrigin.UNRELATED\n'
                                                                  '\n'
                                                                  '\n'
                                                                  'def mark_seed_origin(pose: '
                                                                  'Pose, origin: SeedOrigin | '
                                                                  'None) -> None:\n'
                                                                  '    if origin is not None:\n'
                                                                  '        '
                                                                  'pose.meta["seed_lineage"] = '
                                                                  'origin != SeedOrigin.UNRELATED\n'
                                                                  '        '
                                                                  'pose.meta["seed_unchanged"] = '
                                                                  'origin == '
                                                                  'SeedOrigin.UNCHANGED\n')]},
 'dockingpp/dockingpp/search/algoritmos/espaco_rigido.py': {'after_sha256': 'a46a485b7d99f1f5dbe3c4457d4f1195f412718f8e10f4babdcecab8c75c8ba6',
                                                            'before_sha256': 'd3911935b91d30e84c84dc1cdb0764b8625e6a262175bf3a728ebd802a9f0e9b',
                                                            'spans': [(11,
                                                                       12,
                                                                       'from '
                                                                       'dockingpp.utils.geometry '
                                                                       'import apply_transform, '
                                                                       'place_surface_contact, '
                                                                       'surface_anchor_normal\n'),
                                                                      (45,
                                                                       45,
                                                                       '        self.docking_task '
                                                                       '= str(getattr(cfg, '
                                                                       '"docking_task", '
                                                                       '"surface_discovery"))\n'
                                                                       '        if '
                                                                       'self.docking_task not in '
                                                                       '{"surface_discovery", '
                                                                       '"local_refinement"}:\n'
                                                                       '            raise '
                                                                       'ValueError("unsupported '
                                                                       'docking_task")\n'
                                                                       '        '
                                                                       'self.local_refinement = '
                                                                       'self.docking_task == '
                                                                       '"local_refinement"\n'
                                                                       '        self.search_scale '
                                                                       '= 1.0\n'
                                                                       '        if '
                                                                       'self.local_refinement and '
                                                                       'getattr(cfg, '
                                                                       '"search_space_mode", '
                                                                       '"full") == "reduced":\n'
                                                                       '            '
                                                                       'self.search_scale = '
                                                                       'float(getattr(cfg, '
                                                                       '"local_refinement_reduced_scale", '
                                                                       '0.5))\n'
                                                                       '            if not 0.0 < '
                                                                       'self.search_scale <= 1.0:\n'
                                                                       '                raise '
                                                                       'ValueError("local_refinement_reduced_scale '
                                                                       'must be in (0, 1]")\n'
                                                                       '        max_trans *= '
                                                                       'self.search_scale\n'
                                                                       '        max_rot_deg *= '
                                                                       'self.search_scale\n'),
                                                                      (57,
                                                                       58,
                                                                       '        if '
                                                                       'self.local_refinement:\n'
                                                                       '            if '
                                                                       'requested_anchor_mode not '
                                                                       'in {"auto", '
                                                                       '"input_centroid"}:\n'
                                                                       '                raise '
                                                                       'ValueError("local_refinement '
                                                                       'requires auto or '
                                                                       'input_centroid anchor")\n'
                                                                       '            '
                                                                       'self.anchor_mode = '
                                                                       '"input_centroid"\n'
                                                                       '        elif '
                                                                       'requested_anchor_mode == '
                                                                       '"auto":\n'),
                                                                      (78,
                                                                       78,
                                                                       '        '
                                                                       'self.surface_normal = '
                                                                       'None\n'
                                                                       '        if '
                                                                       'requested_anchor_mode == '
                                                                       '"auto" and '
                                                                       'self.anchor_mode == '
                                                                       '"pocket_center":\n'
                                                                       '            '
                                                                       'self.surface_normal = '
                                                                       'surface_anchor_normal(pocket)\n'
                                                                       '            if '
                                                                       'self.surface_normal is not '
                                                                       'None:\n'
                                                                       '                '
                                                                       'self.anchor_mode = '
                                                                       '"surface_contact"\n'),
                                                                      (150,
                                                                       150,
                                                                       '            '
                                                                       '"docking_task": '
                                                                       'self.docking_task,\n'
                                                                       '            '
                                                                       '"input_relative_placement_used": '
                                                                       'bool(self.local_refinement '
                                                                       'or self.anchor_mode == '
                                                                       '"input_centroid"),\n'
                                                                       '            '
                                                                       '"search_domain_low": '
                                                                       'self.domain_low.tolist(),\n'
                                                                       '            '
                                                                       '"search_domain_high": '
                                                                       'self.domain_high.tolist(),\n'
                                                                       '            '
                                                                       '"local_refinement_search_scale": '
                                                                       'self.search_scale,\n'),
                                                                      (163,
                                                                       163,
                                                                       '        if '
                                                                       'self.surface_normal is not '
                                                                       'None:\n'
                                                                       '            contact_point '
                                                                       '= self.anchor_center + '
                                                                       'solution[:3]\n'
                                                                       '            coords, '
                                                                       'support_index = '
                                                                       'place_surface_contact(\n'
                                                                       '                '
                                                                       'self.base_coords, '
                                                                       'rotation, contact_point, '
                                                                       'self.surface_normal,\n'
                                                                       '            )\n'
                                                                       '            meta.update(\n'
                                                                       '                '
                                                                       'surface_anchor_version="surface_contact_v1",\n'
                                                                       '                '
                                                                       'surface_contact_point=contact_point.tolist(),\n'
                                                                       '                '
                                                                       'surface_support_atom_index=support_index,\n'
                                                                       '                '
                                                                       'surface_normal=self.surface_normal.tolist(),\n'
                                                                       '                '
                                                                       'rotation_center=coords.mean(axis=0).tolist(),\n'
                                                                       '                '
                                                                       'transform_semantics="rotate_input_then_place_surface_support_at_pocket_anchor_plus_offset",\n'
                                                                       '            )\n')]},
 'dockingpp/dockingpp/search/algoritmos/hibrido.py': {'after_sha256': 'c53fa1f0f4e938be80adb7c88281b279702664130d3048b8a32f6db845281368',
                                                      'before_sha256': 'e344a34773d6934ae7890e327f77f2f3740f19f86d875172b13d9e6fecdf1e6f',
                                                      'spans': [(18,
                                                                 19,
                                                                 'from .contexto import '
                                                                 'GenerationContext, SeedOrigin, '
                                                                 'inherit_seed, mark_seed_origin, '
                                                                 'seed_origin\n'),
                                                                (237,
                                                                 237,
                                                                 '            '
                                                                 'origins=[seed_origin(pose) for '
                                                                 'pose in context.previous],\n'),
                                                                (270,
                                                                 270,
                                                                 '        donor: int | None = '
                                                                 'None,\n'
                                                                 '        scout: bool = False,\n'),
                                                                (272,
                                                                 272,
                                                                 '        source = '
                                                                 'state.origins[index]\n'
                                                                 '        if source is not None:\n'
                                                                 '            if scout:\n'
                                                                 '                '
                                                                 'state.origins[index] = '
                                                                 'SeedOrigin.UNRELATED\n'
                                                                 '            else:\n'
                                                                 '                donor_origin = '
                                                                 'state.origins[donor] if donor is '
                                                                 'not None else None\n'
                                                                 '                '
                                                                 'state.origins[index] = '
                                                                 'inherit_seed(state.solutions[index], '
                                                                 'source, donor_origin)\n'),
                                                                (289,
                                                                 289,
                                                                 '            '
                                                                 'mark_seed_origin(pose, '
                                                                 'state.origins[index])\n'),
                                                                (317,
                                                                 318,
                                                                 '                '
                                                                 'self._set_solution(state, index, '
                                                                 'candidate, candidate_fit, '
                                                                 'donor=peer)\n'),
                                                                (347,
                                                                 348,
                                                                 '                '
                                                                 'self._set_solution(state, index, '
                                                                 'candidate, candidate_fit, '
                                                                 'donor=peer)\n'),
                                                                (370,
                                                                 370,
                                                                 '                scout=True,\n'),
                                                                (483,
                                                                 483,
                                                                 '        new_origins = '
                                                                 'list(state.origins)\n'
                                                                 '        new_origins[0] = '
                                                                 'state.origins[elite]\n'),
                                                                (561,
                                                                 561,
                                                                 '                '
                                                                 'new_origins[position] = '
                                                                 'inherit_seed(child, '
                                                                 'state.origins[first], '
                                                                 'state.origins[second])\n'),
                                                                (565,
                                                                 565,
                                                                 '                '
                                                                 'new_origins[position] = '
                                                                 'state.origins[base_index]\n'),
                                                                (569,
                                                                 569,
                                                                 '        state.origins = '
                                                                 'new_origins\n')]},
 'dockingpp/dockingpp/search/algoritmos/pso.py': {'after_sha256': '6189be170ebbd4c6ee531ac4ebbcfd500a47fbb22943ec868c5fb2d84277800c',
                                                  'before_sha256': 'b1aaf3b3eddb684f7d784934788dfedf84ee7de09cc63ca8985d54610654b3be',
                                                  'spans': [(17,
                                                             18,
                                                             'from .contexto import '
                                                             'GenerationContext, SeedOrigin, '
                                                             'inherit_seed, mark_seed_origin, '
                                                             'seed_origin\n'),
                                                            (31,
                                                             31,
                                                             '    seed_origin: SeedOrigin | None = '
                                                             'None\n'),
                                                            (187,
                                                             187,
                                                             '                    '
                                                             'seed_origin=seed_origin(pose),\n'),
                                                            (214,
                                                             214,
                                                             '        if '
                                                             'global_best_particle.seed_origin is '
                                                             'not None:\n'
                                                             '            '
                                                             'context.hybrid_operator_state["pso_global_best_from_seed_particle"] '
                                                             '= (\n'
                                                             '                '
                                                             'global_best_particle.seed_origin != '
                                                             'SeedOrigin.UNRELATED\n'
                                                             '            )\n'),
                                                            (238,
                                                             238,
                                                             '            mark_seed_origin(pose, '
                                                             'inherit_seed(position, '
                                                             'particle.seed_origin))\n')]},
 'dockingpp/dockingpp/utils/geometry.py': {'after_sha256': '3ed0169ea5389efb71e054baff38189a678ccda376d62e9b5dd36d8b09453ace',
                                           'before_sha256': 'b2efd32095abed30e8d7eb0ebdcc3d4a6b6d3f86f7ba588ee4dac20767fe5934',
                                           'spans': [(21, 22, 'def apply_transform(\n'),
                                                     (27,
                                                      28,
                                                      '    return rotated + '
                                                      'np.asarray(translation, dtype=float)\n'
                                                      '\n'
                                                      '\n'
                                                      'def surface_anchor_normal(pocket) -> '
                                                      'np.ndarray | None:\n'
                                                      '    """Outward receptor-only direction for '
                                                      'automatically detected surface sites.\n'
                                                      '\n'
                                                      '    Surface grid points represent an '
                                                      "interface, not the mobile body's centroid.\n"
                                                      '    Explicit cavity pockets and the FULL '
                                                      'global region retain their semantics.\n'
                                                      '    """\n'
                                                      '    meta = getattr(pocket, "meta", {}) or '
                                                      '{}\n'
                                                      '    if meta.get("detection_mode") != '
                                                      '"surface_clearance_grid":\n'
                                                      '        return None\n'
                                                      '    receptor = '
                                                      'np.asarray(meta.get("receptor_coords_full", '
                                                      '[]), dtype=float)\n'
                                                      '    if receptor.ndim != 2 or '
                                                      'receptor.shape[1] != 3 or not '
                                                      'len(receptor):\n'
                                                      '        raise ValueError("surface pocket '
                                                      'requires full receptor coordinates")\n'
                                                      '    center = np.asarray(pocket.center, '
                                                      'dtype=float)\n'
                                                      '    normal = center - '
                                                      'receptor.mean(axis=0)\n'
                                                      '    norm = float(np.linalg.norm(normal))\n'
                                                      '    if norm <= 1e-12:\n'
                                                      '        nearest = '
                                                      'np.argmin(np.sum((receptor - center) ** 2, '
                                                      'axis=1))\n'
                                                      '        normal = center - '
                                                      'receptor[nearest]\n'
                                                      '        norm = '
                                                      'float(np.linalg.norm(normal))\n'
                                                      '    if norm <= 1e-12 or not '
                                                      'np.isfinite(norm):\n'
                                                      '        raise ValueError("surface pocket '
                                                      'has no defined outward direction")\n'
                                                      '    return normal / norm\n'
                                                      '\n'
                                                      '\n'
                                                      'def place_surface_contact(\n'
                                                      '    coords: np.ndarray, rotation: '
                                                      'np.ndarray, contact_point: np.ndarray,\n'
                                                      '    normal: np.ndarray,\n'
                                                      ') -> tuple[np.ndarray, int]:\n'
                                                      '    """Rigidly place the receptor-facing '
                                                      'support atom at the local contact site.\n'
                                                      '\n'
                                                      '    This changes translation only after '
                                                      'rotation: internal distances are intact.\n'
                                                      '    Every atom lies on the outward side of '
                                                      'the support plane. The full receptor\n'
                                                      '    collision checks still decide whether a '
                                                      'placement is actually admissible.\n'
                                                      '    """\n'
                                                      '    coords = np.asarray(coords, '
                                                      'dtype=float)\n'
                                                      '    if coords.ndim != 2 or coords.shape[1] '
                                                      '!= 3 or not len(coords):\n'
                                                      '        raise ValueError("surface placement '
                                                      'requires nonempty (N,3) coordinates")\n'
                                                      '    rotated = (coords - '
                                                      'coords.mean(axis=0)) @ np.asarray(rotation, '
                                                      'dtype=float).T\n'
                                                      '    support_index = int(np.argmin(rotated @ '
                                                      'np.asarray(normal, dtype=float)))\n'
                                                      '    placed = rotated - '
                                                      'rotated[support_index] + '
                                                      'np.asarray(contact_point, dtype=float)\n'
                                                      '    return placed, support_index\n')]}}


def descriptor(archive_sha256: str) -> dict[str, object] | None:
    if archive_sha256 != ARCHIVE_SHA256:
        return None
    return {
        "version": PATCH_VERSION,
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sources": {MEMBER: SOURCE_SHA256, **{name: item["before_sha256"] for name, item in _SOURCE_CORRECTIONS.items()}},
        "outputs": {name: item["after_sha256"] for name, item in _SOURCE_CORRECTIONS.items()},
    }


def patch_member(member: str, source: bytes) -> bytes:
    if member == MEMBER:
        return patch_source(source)
    correction = _SOURCE_CORRECTIONS.get(member)
    if correction is None or hashlib.sha256(source).hexdigest() != correction["before_sha256"]:
        raise ValueError("V5 source does not match the patch preimage")
    lines = source.decode("utf-8").splitlines(keepends=True)
    for start, end, replacement in reversed(correction["spans"]):
        lines[start:end] = replacement.splitlines(keepends=True)
    updated = "".join(lines).encode("utf-8")
    if hashlib.sha256(updated).hexdigest() != correction["after_sha256"]:
        raise ValueError("V5 correction produced an unexpected source hash")
    compile(updated, member, "exec")
    return updated
