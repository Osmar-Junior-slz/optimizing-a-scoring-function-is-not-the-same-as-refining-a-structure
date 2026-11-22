"""Configuracao do caso de uso de docking.

O modelo fica na camada de aplicacao porque descreve politicas de execucao. O
caminho historico ``dockingpp.pipeline.run.Config`` continua sendo reexportado.
"""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from dockingpp.domain.peptide_constraints import (
    DEFAULT_PEPTIDE_MAX_RESIDUES,
    DEFAULT_PEPTIDE_MIN_RESIDUES,
)


class Config(BaseModel):
    """Configuracao validada do DockingPP."""

    model_config = ConfigDict(extra="allow")

    seed: int = 7
    device: str = "cpu"
    search_engine: str = "hybrid"
    generations: int = 5
    pop_size: int = 20
    topk: int = 5
    num_atoms: int = 10
    max_trans: float = 5.0
    max_rot_deg: float = 25.0
    search_anchor_mode: str = "auto"
    docking_task: Literal["surface_discovery", "local_refinement"] = "surface_discovery"
    local_refinement_reduced_scale: float = Field(default=0.5, gt=0.0, le=1.0)
    # None preserves the historical mode-dependent initialization contract.
    initialization: Optional[Literal["uniform", "seeded"]] = None
    search_extent: Optional[Literal["reduced", "expanded"]] = None
    experiment_budget_policy: Optional[Literal["common_ceiling", "fixed_evaluations"]] = None
    pose_source: Optional[str] = None
    hybrid_operator_mode: str = "canonical"
    alpha: float = 0.8
    onlooker_frac: float = 0.8
    elite_size: int = 1
    tournament_k: int = 3
    limit_no_improve: int = 30
    ga_interval: int = 20
    p_crossover_ari: float = 0.5
    p_mutation_gauss: float = 0.7
    adaptive_operator_probabilities: bool = False
    operator_adaptation_rate: float = 0.25
    operator_probability_min: float = 0.1
    operator_probability_max: float = 0.9
    sw_interval: int = 3
    sw_max_iter: int = 50
    sw_elite_iter: int = 15
    sw_heavy_iter: int = 20
    sw_patience: int = 8
    sw_iter_budget: int = 2000
    top_frac_sw: float = 0.1
    ag_elite_size: int = 1
    ag_tournament_k: int = 3
    ag_crossover_probability: float = 0.9
    ag_mutation_probability: Optional[float] = None
    ag_sbx_eta: float = 15.0
    ag_mutation_eta: float = 20.0
    pso_inertia_weight: float = 1.0
    pso_cognitive_coefficient: float = 2.0
    pso_social_coefficient: float = 2.0
    pso_velocity_clamp_fraction: float = 0.2
    objective_evaluation_budget_per_pocket: Optional[int] = None
    hybrid_plateau_patience: int = 80
    hybrid_plateau_eps: float = 1e-6
    hybrid_memory_size: int = 15
    hybrid_memory_inject_count: int = 2
    hybrid_memory_warmup_generations: int = 1
    hybrid_memory_replace_only_if_better: bool = True
    hybrid_memory_rounding_decimals: int = 8
    hybrid_memory_operator_mode: Optional[str] = None
    hybrid_memory_alpha: Optional[float] = 0.8
    hybrid_memory_onlooker_frac: Optional[float] = 0.8
    hybrid_memory_elite_size: Optional[int] = None
    hybrid_memory_tournament_k: Optional[int] = None
    hybrid_memory_limit_no_improve: Optional[int] = 30
    hybrid_memory_ga_interval: Optional[int] = 20
    hybrid_memory_p_crossover_ari: Optional[float] = 0.5
    hybrid_memory_p_mutation_gauss: Optional[float] = 0.7
    hybrid_memory_adaptive_operator_probabilities: Optional[bool] = None
    hybrid_memory_operator_adaptation_rate: Optional[float] = None
    hybrid_memory_operator_probability_min: Optional[float] = None
    hybrid_memory_operator_probability_max: Optional[float] = None
    hybrid_memory_top_frac_sw: Optional[float] = 0.1
    hybrid_memory_sw_interval: Optional[int] = 3
    hybrid_memory_sw_max_iter: Optional[int] = None
    hybrid_memory_sw_elite_iter: Optional[int] = 15
    hybrid_memory_sw_heavy_iter: Optional[int] = 20
    hybrid_memory_sw_patience: Optional[int] = 8
    hybrid_memory_sw_iter_budget: Optional[int] = None
    cheap_weights: Dict[str, float] = Field(default_factory=dict)
    rescoring_enabled: bool = True
    rescoring_top_k: int = 1
    rescoring_scope: str = "global"
    cheap_score_variant: str = "geom_kdtree"
    final_score_variant: str = "vina_rescore"
    structural_metrics_enabled: bool = True
    structural_contact_cutoff_angstrom: float = 4.0
    structural_interface_cutoff_angstrom: float = 8.0
    complex_id: Optional[str] = None
    complex_id_source: Optional[str] = None
    complex_id_conflict: Optional[str] = None
    input_id: Optional[str] = None
    confidence_calibrator_path: Optional[str] = None
    confidence_calibrator_required: bool = False
    confidence_calibrator: Any = Field(default=None, exclude=True)
    confidence_calibrator_load_error: Optional[str] = None
    peptide_input_validation_enabled: bool = True
    receptor_peptide_pair_validation_enabled: bool = True
    peptide_min_residues: int = DEFAULT_PEPTIDE_MIN_RESIDUES
    peptide_max_residues: int = DEFAULT_PEPTIDE_MAX_RESIDUES
    expensive_every: int = 0
    expensive_topk: Optional[int] = None
    top_pockets: int = 32
    full_search: bool = True
    max_pockets_used: int = 8
    search_space_mode: str = "full"
    budget_policy: str = "split"
    cross_pocket_selection_policy: str = "scan_priority"
    reduced_geometric_fallback_enabled: bool = True
    pockets_path: Optional[str] = None
    pocket_margin: float = 2.0
    pocket_grid_size: float = 4.0
    pocket_grid_padding: float = 8.0
    pocket_contact_radius: float = 8.0
    pocket_anchor_min_clearance: float = 3.0
    pocket_anchor_max_clearance: float = 6.0
    pocket_anchor_target_clearance: float = 4.5
    pocket_anchor_clearance_weight: float = 2.0
    pocket_anchor_separation: float = 6.0
    pocket_search_radius: float = 5.0
    pocket_grid_chunk_size: int = 128
    pocket_max_grid_points: int = 250_000
    min_pocket_contact_atoms: int = 8
    auto_pocket_count: int = 32
    allow_peptide_coordinate_prior: bool = False
    pocket_rank_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "w_size": 1.0,
            "w_compact": 1.0,
            "w_depth": 0.5,
            "w_proximity": 0.0,
        }
    )
    scan: Dict[str, Any] = Field(default_factory=dict)
    rescoring: Dict[str, Any] = Field(default_factory=dict)
    confidence_policy: Dict[str, Any] = Field(default_factory=dict)
    reducao: Dict[str, Any] = Field(default_factory=dict)
    debug_log_enabled: bool = False
    debug_log_path: Optional[str] = None
    debug_log_level: str = "INFO"
    debug_enabled: bool = True
    debug_level: str = "AUDIT"
    debug_dirname: str = "debug"
    trace_generation_interval: int = 1
    trace_score_interval: int = 1
    trace_write_jsonl: bool = True
    usar_reducao_condicionada_ao_peptideo: bool = False

__all__ = ["Config"]
