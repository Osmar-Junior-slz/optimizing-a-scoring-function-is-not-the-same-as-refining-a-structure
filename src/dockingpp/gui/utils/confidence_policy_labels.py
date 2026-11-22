"""Friendly labels for confidence_policy values shown in the GUI."""

from __future__ import annotations

from typing import Any, Mapping


MODE_TO_LABEL = {
    "disabled": "Desligada",
    "observe": "Somente observar",
    "assist_selection": "Ajudar seleção",
    "assist_fallback": "Ajudar retorno para busca completa",
    "assist_all": "Ajudar seleção e retorno",
}

LABEL_TO_MODE = {label: mode for mode, label in MODE_TO_LABEL.items()}

DEFAULT_CONFIDENCE_POLICY_MODE = "disabled"


def confidence_policy_mode_labels() -> list[str]:
    """Return labels in stable display order."""

    return [MODE_TO_LABEL[mode] for mode in MODE_TO_LABEL]


def normalize_confidence_policy_mode(mode: Any) -> str:
    """Normalize internal confidence policy mode to a known value."""

    if isinstance(mode, str):
        normalized = mode.strip().lower()
        if normalized in MODE_TO_LABEL:
            return normalized
    return DEFAULT_CONFIDENCE_POLICY_MODE


def confidence_policy_mode_to_label(mode: Any) -> str:
    """Convert an internal mode value to its friendly label."""

    normalized = normalize_confidence_policy_mode(mode)
    return MODE_TO_LABEL[normalized]


def confidence_policy_label_to_mode(label_or_mode: Any) -> str:
    """Convert a GUI label back to internal mode (accepts labels or internal values)."""

    if isinstance(label_or_mode, str):
        if label_or_mode in LABEL_TO_MODE:
            return LABEL_TO_MODE[label_or_mode]
        normalized_mode = normalize_confidence_policy_mode(label_or_mode)
        if normalized_mode in MODE_TO_LABEL:
            return normalized_mode
    return DEFAULT_CONFIDENCE_POLICY_MODE


def resolve_confidence_policy_state(
    config_data: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    """Resolve enabled/mode from raw config data with backward-compatible defaults."""

    policy_cfg = {}
    if isinstance(config_data, Mapping):
        raw_policy = config_data.get("confidence_policy")
        if isinstance(raw_policy, Mapping):
            policy_cfg = raw_policy
    enabled = bool(policy_cfg.get("enabled", False))
    mode = normalize_confidence_policy_mode(policy_cfg.get("mode", DEFAULT_CONFIDENCE_POLICY_MODE))
    if not enabled:
        mode = DEFAULT_CONFIDENCE_POLICY_MODE
    return enabled, mode


def build_confidence_policy_override(
    *,
    enabled: bool,
    selected_mode_or_label: Any,
) -> dict[str, Any]:
    """Build config override payload preserving internal canonical values."""

    if not enabled:
        return {
            "enabled": False,
            "mode": DEFAULT_CONFIDENCE_POLICY_MODE,
            "apply_fallback": False,
        }
    mode = confidence_policy_label_to_mode(selected_mode_or_label)
    policy_enabled = mode != DEFAULT_CONFIDENCE_POLICY_MODE
    return {
        "enabled": policy_enabled,
        "mode": mode,
        "apply_fallback": bool(
            policy_enabled and mode in {"assist_fallback", "assist_all"}
        ),
    }
