"""Session state helpers for the Streamlit GUI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from dockingpp.data.io import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
CALIBRATION_MIN_COMPLEXES_POLICY_VERSION = 2


class StateKeys:
    PAGE = "page"
    DEFAULT_OUT_DIR = "default_out_dir"
    LOADED_CONFIG = "loaded_config"
    CONFIG_SOURCE_LABEL = "config_source_label"
    CONFIG_OVERRIDES = "config_overrides"
    LAST_OUT_DIR = "last_out_dir"
    PREPARED_RECEPTOR_PATH = "prepared_receptor_path"
    PREPARED_PEPTIDE_PATH = "prepared_peptide_path"
    PREPARED_PDB_TEXT = "prepared_pdb_text"
    PREPARED_PDB_NAME = "prepared_pdb_name"
    PDB_CLEAN_OUTDIR = "pdb_clean_outdir"
    PDB_CLEAN_OUTDIR_SELECTED = "pdb_clean_outdir_selected"
    PDB_EXTRACT_LIGANDS = "pdb_extract_ligands"
    PDB_EXTRACT_FILES = "pdb_extract_files"
    REPORTS_ROOT = "reports_root"
    REPORTS_ROOT_PENDING = "reports_root_pending"
    REPORT_RUNS = "report_runs"
    OUT_DIR = "out_dir"
    PROTEIN_NAME = "protein_name"
    RECENT_OUT_DIRS = "recent_out_dirs"
    CONFIG_CHOICE = "config_choice"
    CONFIG_UPLOAD = "config_upload"
    RUN_MODE = "run_mode"
    COMPARE_TOP_POCKETS = "compare_top_pockets"
    OVERRIDE_SEED = "override_seed"
    OVERRIDE_GENERATIONS = "override_generations"
    OVERRIDE_POP_SIZE = "override_pop_size"
    OVERRIDE_N_ROUNDS = "override_n_rounds"
    EXPERIMENT_PROTOCOL = "experiment_protocol"
    OVERRIDE_TOPK = "override_topk"
    OVERRIDE_PEPTIDE_MIN_RESIDUES = "override_peptide_min_residues"
    OVERRIDE_PEPTIDE_MAX_RESIDUES = "override_peptide_max_residues"
    OVERRIDE_FULL_SEARCH = "override_full_search"
    OVERRIDE_TOP_POCKETS = "override_top_pockets"
    OVERRIDE_DEBUG_ENABLED = "override_debug_enabled"
    OVERRIDE_DEBUG_PATH = "override_debug_path"
    OVERRIDE_DEBUG_LEVEL = "override_debug_level"
    OVERRIDE_RESCORING_ENABLED = "override_rescoring_enabled"
    OVERRIDE_RESCORING_TOP_K = "override_rescoring_top_k"
    OVERRIDE_RESCORING_SCOPE = "override_rescoring_scope"
    OVERRIDE_CONFIDENCE_POLICY_ENABLED = "override_confidence_policy_enabled"
    OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL = "override_confidence_policy_mode_label"
    OVERRIDE_CONFIDENCE_CALIBRATOR_PATH = "override_confidence_calibrator_path"
    CALIBRATION_RUN_ROOT = "calibration_run_root"
    CALIBRATION_FULL_RUN_ROOT = "calibration_full_run_root"
    CALIBRATION_OUTPUT_PATH = "calibration_output_path"
    CALIBRATION_DOCKQ_THRESHOLD = "calibration_dockq_threshold"
    CALIBRATION_MAX_DOCKQ_LOSS = "calibration_max_dockq_loss"
    CALIBRATION_V5_RISK_TARGET = "calibration_v5_risk_target"
    CALIBRATION_RISK_CONFIDENCE_LEVEL = "calibration_risk_confidence_level"
    CALIBRATION_CV_FOLDS = "calibration_cv_folds"
    CALIBRATION_SEED = "calibration_seed"
    CALIBRATION_MIN_COMPLEXES = "calibration_min_complexes"
    CALIBRATION_MIN_COMPLEXES_POLICY_VERSION = (
        "calibration_min_complexes_policy_version"
    )
    CALIBRATION_LAST_RESULT = "calibration_last_result"


@dataclass
class AppState:
    page: str
    cfg_source: str
    cfg_dict: dict[str, Any]
    config_overrides: dict[str, Any]
    last_out_dir: str | None
    prepared_receptor_path: str
    prepared_peptide_path: str
    prepared_pdb_text: str
    prepared_pdb_name: str
    pdb_prep_outdir: str
    default_out_dir: str
    protein_name: str
    reports_root: str
    report_runs: list[Path]


def init_state_defaults() -> None:
    """Initialize Streamlit session state defaults."""

    if StateKeys.PAGE not in st.session_state:
        st.session_state[StateKeys.PAGE] = "Início"
    if StateKeys.DEFAULT_OUT_DIR not in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state[StateKeys.DEFAULT_OUT_DIR] = f"runs/run_{timestamp}"
    if StateKeys.LOADED_CONFIG not in st.session_state:
        st.session_state[StateKeys.LOADED_CONFIG] = load_config(str(DEFAULT_CONFIG_PATH))
    if StateKeys.CONFIG_SOURCE_LABEL not in st.session_state:
        st.session_state[StateKeys.CONFIG_SOURCE_LABEL] = "Padrão (configs/default.yaml)"
    if StateKeys.CONFIG_OVERRIDES not in st.session_state:
        st.session_state[StateKeys.CONFIG_OVERRIDES] = {}
    if StateKeys.LAST_OUT_DIR not in st.session_state:
        st.session_state[StateKeys.LAST_OUT_DIR] = ""
    if StateKeys.PREPARED_RECEPTOR_PATH not in st.session_state:
        st.session_state[StateKeys.PREPARED_RECEPTOR_PATH] = ""
    if StateKeys.PREPARED_PEPTIDE_PATH not in st.session_state:
        st.session_state[StateKeys.PREPARED_PEPTIDE_PATH] = ""
    if StateKeys.PREPARED_PDB_TEXT not in st.session_state:
        st.session_state[StateKeys.PREPARED_PDB_TEXT] = ""
    if StateKeys.PREPARED_PDB_NAME not in st.session_state:
        st.session_state[StateKeys.PREPARED_PDB_NAME] = ""
    if StateKeys.PDB_CLEAN_OUTDIR not in st.session_state:
        st.session_state[StateKeys.PDB_CLEAN_OUTDIR] = "datasets/cleaned"
    if StateKeys.REPORTS_ROOT not in st.session_state:
        st.session_state[StateKeys.REPORTS_ROOT] = "runs_experimentos" if Path("runs_experimentos").exists() else "runs"
    if StateKeys.REPORT_RUNS not in st.session_state:
        st.session_state[StateKeys.REPORT_RUNS] = []
    if StateKeys.OUT_DIR not in st.session_state:
        st.session_state[StateKeys.OUT_DIR] = st.session_state[StateKeys.DEFAULT_OUT_DIR]
    if StateKeys.PROTEIN_NAME not in st.session_state:
        st.session_state[StateKeys.PROTEIN_NAME] = ""
    if StateKeys.RECENT_OUT_DIRS not in st.session_state:
        st.session_state[StateKeys.RECENT_OUT_DIRS] = [st.session_state[StateKeys.DEFAULT_OUT_DIR]]
    if StateKeys.CALIBRATION_RUN_ROOT not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_RUN_ROOT] = "runs_experimentos"
    if StateKeys.CALIBRATION_FULL_RUN_ROOT not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_FULL_RUN_ROOT] = "runs_experimentos"
    if StateKeys.CALIBRATION_OUTPUT_PATH not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_OUTPUT_PATH] = "artifacts/confidence_calibrator.json"
    if StateKeys.CALIBRATION_DOCKQ_THRESHOLD not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_DOCKQ_THRESHOLD] = 0.23
    if StateKeys.CALIBRATION_MAX_DOCKQ_LOSS not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_MAX_DOCKQ_LOSS] = 0.05
    if StateKeys.CALIBRATION_V5_RISK_TARGET not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_V5_RISK_TARGET] = 0.20
    if StateKeys.CALIBRATION_RISK_CONFIDENCE_LEVEL not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_RISK_CONFIDENCE_LEVEL] = 0.95
    if StateKeys.CALIBRATION_CV_FOLDS not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_CV_FOLDS] = 5
    if StateKeys.CALIBRATION_SEED not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_SEED] = 2026
    calibration_min_complexes = st.session_state.get(
        StateKeys.CALIBRATION_MIN_COMPLEXES
    )
    calibration_policy_version = st.session_state.get(
        StateKeys.CALIBRATION_MIN_COMPLEXES_POLICY_VERSION
    )
    if calibration_policy_version != CALIBRATION_MIN_COMPLEXES_POLICY_VERSION:
        # Migrate old calibration defaults once to the frozen V5 minimum.
        if (
            calibration_min_complexes is None
            or int(calibration_min_complexes) < 21
            or int(calibration_min_complexes) == 25
        ):
            st.session_state[StateKeys.CALIBRATION_MIN_COMPLEXES] = 21
        st.session_state[
            StateKeys.CALIBRATION_MIN_COMPLEXES_POLICY_VERSION
        ] = CALIBRATION_MIN_COMPLEXES_POLICY_VERSION
    elif calibration_min_complexes is None or int(calibration_min_complexes) < 21:
        st.session_state[StateKeys.CALIBRATION_MIN_COMPLEXES] = 21
    if StateKeys.CALIBRATION_LAST_RESULT not in st.session_state:
        st.session_state[StateKeys.CALIBRATION_LAST_RESULT] = None


def get_state() -> AppState:
    """Return the current session state mapped to AppState."""

    report_runs = st.session_state.get(StateKeys.REPORT_RUNS, [])
    if report_runs and not isinstance(report_runs[0], Path):
        report_runs = [Path(path) for path in report_runs]
    return AppState(
        page=st.session_state.get(StateKeys.PAGE, "Início"),
        cfg_source=st.session_state.get(StateKeys.CONFIG_SOURCE_LABEL, "Padrão (configs/default.yaml)"),
        cfg_dict=st.session_state.get(StateKeys.LOADED_CONFIG, {}),
        config_overrides=st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}),
        last_out_dir=st.session_state.get(StateKeys.LAST_OUT_DIR) or None,
        prepared_receptor_path=st.session_state.get(StateKeys.PREPARED_RECEPTOR_PATH, ""),
        prepared_peptide_path=st.session_state.get(StateKeys.PREPARED_PEPTIDE_PATH, ""),
        prepared_pdb_text=st.session_state.get(StateKeys.PREPARED_PDB_TEXT, ""),
        prepared_pdb_name=st.session_state.get(StateKeys.PREPARED_PDB_NAME, ""),
        pdb_prep_outdir=st.session_state.get(StateKeys.PDB_CLEAN_OUTDIR, "datasets/cleaned"),
        default_out_dir=st.session_state.get(StateKeys.DEFAULT_OUT_DIR, "runs"),
        protein_name=st.session_state.get(StateKeys.PROTEIN_NAME, ""),
        reports_root=st.session_state.get(StateKeys.REPORTS_ROOT, "runs"),
        report_runs=report_runs,
    )


def set_state(**kwargs: Any) -> None:
    """Set session state keys via keyword arguments."""

    for key, value in kwargs.items():
        st.session_state[key] = value
