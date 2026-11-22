"""Configuration editing page."""

from __future__ import annotations

import streamlit as st
import yaml

from dockingpp.data.pdb_validation import resolve_peptide_residue_limits
from dockingpp.gui.pages.base import BasePage
from dockingpp.gui.state import AppState, StateKeys
from dockingpp.gui.utils.confidence_policy_labels import (
    build_confidence_policy_override,
    confidence_policy_mode_labels,
    confidence_policy_mode_to_label,
    resolve_confidence_policy_state,
)


class ConfigPage(BasePage):
    id = "Configurações"
    title = "Configurações"

    def render(self, state: AppState) -> None:
        st.header("Configurações")
        config_data = state.cfg_dict or {}
        source_label = state.cfg_source or "Padrão (configs/default.yaml)"

        st.write(f"Configuração carregada: {source_label}")
        if config_data:
            st.code(yaml.safe_dump(config_data, sort_keys=False), language="yaml")
        else:
            st.warning("Nenhuma configuração carregada no momento.")

        st.subheader("Sobrescrever parâmetros")
        st.write("As alterações abaixo são aplicadas apenas na execução atual (em memória).")

        st.session_state.setdefault(StateKeys.OVERRIDE_SEED, config_data.get("seed", 7))
        st.session_state.setdefault(StateKeys.OVERRIDE_GENERATIONS, config_data.get("generations", 100))
        st.session_state.setdefault(StateKeys.OVERRIDE_POP_SIZE, config_data.get("pop_size", 100))
        st.session_state.setdefault(StateKeys.OVERRIDE_TOPK, config_data.get("topk", 5))
        peptide_min_cfg, peptide_max_cfg = resolve_peptide_residue_limits(config_data)
        st.session_state.setdefault(StateKeys.OVERRIDE_PEPTIDE_MIN_RESIDUES, peptide_min_cfg)
        st.session_state.setdefault(StateKeys.OVERRIDE_PEPTIDE_MAX_RESIDUES, peptide_max_cfg)
        st.session_state.setdefault(StateKeys.OVERRIDE_FULL_SEARCH, config_data.get("full_search", True))
        st.session_state.setdefault(
            StateKeys.OVERRIDE_RESCORING_ENABLED,
            bool(config_data.get("rescoring_enabled", True)),
        )
        st.session_state.setdefault(
            StateKeys.OVERRIDE_RESCORING_TOP_K,
            int(config_data.get("rescoring_top_k", 1) or 1),
        )
        initial_scope = str(config_data.get("rescoring_scope", "global") or "global").strip().lower()
        if initial_scope not in {"global", "per_pocket"}:
            initial_scope = "global"
        st.session_state.setdefault(StateKeys.OVERRIDE_RESCORING_SCOPE, initial_scope)

        confidence_policy_enabled_cfg, confidence_policy_mode_cfg = resolve_confidence_policy_state(
            config_data
        )
        st.session_state.setdefault(
            StateKeys.OVERRIDE_CONFIDENCE_POLICY_ENABLED,
            bool(confidence_policy_enabled_cfg),
        )
        st.session_state.setdefault(
            StateKeys.OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL,
            confidence_policy_mode_to_label(confidence_policy_mode_cfg),
        )
        st.session_state.setdefault(
            StateKeys.OVERRIDE_CONFIDENCE_CALIBRATOR_PATH,
            str(config_data.get("confidence_calibrator_path") or ""),
        )

        seed = st.number_input(
            "seed",
            value=int(st.session_state[StateKeys.OVERRIDE_SEED]),
            step=1,
            key=StateKeys.OVERRIDE_SEED,
        )
        generations = st.number_input(
            "generations",
            value=int(st.session_state[StateKeys.OVERRIDE_GENERATIONS]),
            step=1,
            key=StateKeys.OVERRIDE_GENERATIONS,
        )
        pop_size = st.number_input(
            "pop_size",
            value=int(st.session_state[StateKeys.OVERRIDE_POP_SIZE]),
            step=1,
            key=StateKeys.OVERRIDE_POP_SIZE,
        )
        topk = st.number_input(
            "topk",
            value=int(st.session_state[StateKeys.OVERRIDE_TOPK]),
            step=1,
            key=StateKeys.OVERRIDE_TOPK,
        )
        peptide_min_residues = st.number_input(
            "peptide_min_residues",
            min_value=1,
            value=int(st.session_state[StateKeys.OVERRIDE_PEPTIDE_MIN_RESIDUES]),
            step=1,
            key=StateKeys.OVERRIDE_PEPTIDE_MIN_RESIDUES,
        )
        if int(st.session_state[StateKeys.OVERRIDE_PEPTIDE_MAX_RESIDUES]) < int(peptide_min_residues):
            st.session_state[StateKeys.OVERRIDE_PEPTIDE_MAX_RESIDUES] = int(peptide_min_residues)
        peptide_max_residues = st.number_input(
            "peptide_max_residues",
            min_value=int(peptide_min_residues),
            value=int(st.session_state[StateKeys.OVERRIDE_PEPTIDE_MAX_RESIDUES]),
            step=1,
            key=StateKeys.OVERRIDE_PEPTIDE_MAX_RESIDUES,
        )
        full_search = st.checkbox(
            "full_search",
            value=bool(st.session_state[StateKeys.OVERRIDE_FULL_SEARCH]),
            key=StateKeys.OVERRIDE_FULL_SEARCH,
        )
        rescoring_enabled = st.checkbox(
            "rescoring_enabled",
            value=bool(st.session_state[StateKeys.OVERRIDE_RESCORING_ENABLED]),
            key=StateKeys.OVERRIDE_RESCORING_ENABLED,
            help="Aplica final_rescore seletivo apenas nos melhores candidatos.",
        )
        rescoring_top_k = st.number_input(
            "rescoring_top_k",
            min_value=1,
            step=1,
            value=int(st.session_state[StateKeys.OVERRIDE_RESCORING_TOP_K]),
            key=StateKeys.OVERRIDE_RESCORING_TOP_K,
            disabled=not rescoring_enabled,
        )
        scope_options = ["global", "per_pocket"]
        scope_value = str(st.session_state[StateKeys.OVERRIDE_RESCORING_SCOPE] or "global").strip().lower()
        if scope_value not in scope_options:
            scope_value = "global"
        rescoring_scope = st.selectbox(
            "rescoring_scope",
            options=scope_options,
            index=scope_options.index(scope_value),
            key=StateKeys.OVERRIDE_RESCORING_SCOPE,
            disabled=not rescoring_enabled,
        )

        confidence_policy_enabled = st.checkbox(
            "confidence_policy.enabled",
            value=bool(st.session_state[StateKeys.OVERRIDE_CONFIDENCE_POLICY_ENABLED]),
            key=StateKeys.OVERRIDE_CONFIDENCE_POLICY_ENABLED,
            help=(
                "Sinal auxiliar auditável. Não altera score científico final; "
                "apoia decisão de seleção/fallback."
            ),
        )
        confidence_mode_options = confidence_policy_mode_labels()
        confidence_mode_value = str(
            st.session_state[StateKeys.OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL] or ""
        )
        if confidence_mode_value not in confidence_mode_options:
            confidence_mode_value = confidence_policy_mode_to_label("disabled")
        confidence_policy_mode_label = st.selectbox(
            "confidence_policy.mode",
            options=confidence_mode_options,
            index=confidence_mode_options.index(confidence_mode_value),
            key=StateKeys.OVERRIDE_CONFIDENCE_POLICY_MODE_LABEL,
            disabled=not confidence_policy_enabled,
            help="Escolha como a confidence_policy auxilia a decisão no pipeline.",
        )
        confidence_calibrator_path = st.text_input(
            "confidence_calibrator_path",
            value=str(
                st.session_state[StateKeys.OVERRIDE_CONFIDENCE_CALIBRATOR_PATH]
                or ""
            ),
            key=StateKeys.OVERRIDE_CONFIDENCE_CALIBRATOR_PATH,
            help=(
                "Artefato JSON versionado. É obrigatório para executar fallback "
                "automático nos modos assist_fallback e assist_all."
            ),
        )
        confidence_policy_override = build_confidence_policy_override(
            enabled=bool(confidence_policy_enabled),
            selected_mode_or_label=confidence_policy_mode_label,
        )
        calibrator_required = bool(
            confidence_policy_override.get("mode")
            in {"assist_fallback", "assist_all"}
        )
        if calibrator_required and not str(confidence_calibrator_path).strip():
            st.warning(
                "Este modo exige um calibrador treinado; a execução será bloqueada "
                "até confidence_calibrator_path ser informado."
            )

        top_pockets = None
        if not full_search:
            st.session_state.setdefault(StateKeys.OVERRIDE_TOP_POCKETS, config_data.get("top_pockets", 3))
            top_pockets = st.number_input(
                "top_pockets",
                value=int(st.session_state[StateKeys.OVERRIDE_TOP_POCKETS]),
                step=1,
                key=StateKeys.OVERRIDE_TOP_POCKETS,
            )

        current_overrides = dict(st.session_state.get(StateKeys.CONFIG_OVERRIDES, {}))
        current_overrides.update(
            {
                "seed": int(seed),
                "generations": int(generations),
                "pop_size": int(pop_size),
                "topk": int(topk),
                "peptide_min_residues": int(peptide_min_residues),
                "peptide_max_residues": int(peptide_max_residues),
                "full_search": bool(full_search),
                "top_pockets": int(top_pockets) if top_pockets is not None else None,
                "rescoring_enabled": bool(rescoring_enabled),
                "rescoring_top_k": int(rescoring_top_k),
                "rescoring_scope": str(rescoring_scope),
                "confidence_policy": confidence_policy_override,
                "confidence_calibrator_path": (
                    str(confidence_calibrator_path).strip() or None
                ),
                "confidence_calibrator_required": calibrator_required,
            }
        )
        st.session_state[StateKeys.CONFIG_OVERRIDES] = current_overrides
