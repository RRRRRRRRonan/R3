"""Shared scale presets for ALNS optimisation regression suites.

Centralises the scenario overrides and iteration budgets used by the Minimal
ALNS, Matheuristic ALNS, and Matheuristic + Q-learning tests.  Keeping the
numbers in a single place ensures that when we tune runtimes or difficulty for
one solver we automatically propagate the change to the other variants, keeping
comparisons fair and maintenance light.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping


@dataclass(frozen=True)
class IterationPreset:
    """Number of optimisation iterations each solver variant should run."""

    minimal: int
    matheuristic: int
    q_learning: int


@dataclass(frozen=True)
class ScalePreset:
    """Scenario adjustments and iteration budgets for a single scale."""

    scenario_overrides: Mapping[str, object]
    iterations: IterationPreset


ALNS_TEST_PRESETS: Dict[str, ScalePreset] = {
    "small": ScalePreset(
        scenario_overrides={"num_tasks": 10, "num_charging": 1, "seed": 11},
        iterations=IterationPreset(minimal=16, matheuristic=28, q_learning=18),
    ),
    "medium": ScalePreset(
        scenario_overrides={"num_tasks": 20, "num_charging": 2, "seed": 13},
        iterations=IterationPreset(minimal=24, matheuristic=36, q_learning=24),
    ),
    "large": ScalePreset(
        scenario_overrides={"num_tasks": 30, "num_charging": 3, "seed": 17},
        iterations=IterationPreset(minimal=32, matheuristic=44, q_learning=30),
    ),
}


def get_scale_preset(scale: str) -> ScalePreset:
    """Return the preset for the requested optimisation scale."""

    try:
        return ALNS_TEST_PRESETS[scale]
    except KeyError as exc:  # pragma: no cover - guardrail
        raise ValueError(f"Unknown ALNS test scale: {scale}") from exc


def build_override_mapping(scale: str) -> MutableMapping[str, object]:
    """Return a mutable copy of the scenario overrides for ``scale``."""

    preset = get_scale_preset(scale)
    return dict(preset.scenario_overrides)


__all__ = [
    "ALNS_TEST_PRESETS",
    "IterationPreset",
    "ScalePreset",
    "build_override_mapping",
    "get_scale_preset",
]
