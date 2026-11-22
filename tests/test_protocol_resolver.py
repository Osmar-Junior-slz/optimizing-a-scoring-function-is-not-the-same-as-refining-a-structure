import pytest

from docking_confidence.application.protocols import ProtocolResolver
from docking_confidence.domain.execution import BaseSearchConfig, DockingTask, ProtocolMode


def test_reduced_applies_25_percent_generations_and_50_percent_pockets():
    base = BaseSearchConfig(generations=100, population=100, top_pockets=32)
    profile = ProtocolResolver().resolve(ProtocolMode.REDUCED, base)
    assert profile.generations == 75
    assert profile.top_pockets == 16
    assert profile.population == 100
    assert profile.evaluation_budget == 7600


def test_full_keeps_base_configuration():
    base = BaseSearchConfig(generations=100, population=100, top_pockets=32)
    profile = ProtocolResolver().resolve(ProtocolMode.FULL, base)
    assert profile.generations == 100
    assert profile.top_pockets == 32
    assert profile.evaluation_budget == 10100


@pytest.mark.parametrize("mode", [ProtocolMode.FULL, ProtocolMode.REDUCED])
def test_default_comparison_refines_the_same_input_task(mode):
    profile = ProtocolResolver().resolve(mode, BaseSearchConfig())
    assert profile.docking_task is DockingTask.LOCAL_REFINEMENT


@pytest.mark.parametrize("mode", [ProtocolMode.FULL, ProtocolMode.REDUCED])
def test_legacy_surface_task_requires_explicit_selection(mode):
    base = BaseSearchConfig(docking_task="surface_discovery")
    profile = ProtocolResolver().resolve(mode, base)
    assert profile.docking_task is DockingTask.SURFACE_DISCOVERY


def test_invalid_task_is_rejected_before_planning():
    with pytest.raises(ValueError):
        ProtocolResolver().resolve(ProtocolMode.REDUCED, BaseSearchConfig(docking_task="unknown"))
