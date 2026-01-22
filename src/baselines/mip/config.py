"""Configuration for the MIP baseline model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MIPBaselineScale:
    """Define the smallest target scale used by the MIP baseline."""

    max_tasks: int = 8
    max_vehicles: int = 2
    max_charging_stations: int = 2
    max_rules: int = 6
    max_decision_epochs: int = 3


@dataclass(frozen=True)
class MIPBaselineSolverConfig:
    """Default solver choice and switches for the baseline model."""

    solver_name: str = "pulp_cbc"
    time_limit_s: float = 30.0
    mip_gap: float = 0.0
    enable_rule_selection: bool = True
    enable_conflict: bool = True
    enable_partial_charging: bool = True
