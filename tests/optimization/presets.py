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
        # Increased difficulty: 10→15 tasks for meaningful optimization
        # Previous: 10 tasks had limited optimization space (greedy was already good)
        scenario_overrides={"num_tasks": 15, "num_charging": 1, "seed": 11},
        # CRITICAL: Use SAME iterations for fair comparison
        # Previous q_learning=50 caused worse performance (over-learning)
        iterations=IterationPreset(minimal=24, matheuristic=40, q_learning=40),
    ),
    "medium": ScalePreset(
        scenario_overrides={"num_tasks": 24, "num_charging": 1, "seed": 19},
        # CRITICAL: Use SAME iterations for fair comparison
        # Previous q_learning=55 but only medium performed well
        iterations=IterationPreset(minimal=32, matheuristic=44, q_learning=44),
    ),
    "large": ScalePreset(
        scenario_overrides={"num_tasks": 30, "num_charging": 3, "seed": 17},
        # Phase 1.4: Further increased iterations for pure learning with zero bias
        # With epsilon_min=0.35 and complete Q-value equality (all 10.0)
        # Need extensive iterations for Q-learning to discover operator values purely through experience
        # 35% exploration means 35 random selections per 100 iterations - need more total iterations
        iterations=IterationPreset(minimal=32, matheuristic=44, q_learning=100),
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
