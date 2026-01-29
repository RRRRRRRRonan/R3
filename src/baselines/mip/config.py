"""Configuration for the MIP baseline model."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import CostParameters


@dataclass(frozen=True)
class ScenarioSynthConfig:
    """Configuration for synthetic scenario generation."""

    num_scenarios: int = 3
    availability_prob: float = 1.0
    release_jitter_s: float = 0.0
    demand_noise_ratio: float = 0.0
    travel_time_factor_range: tuple[float, float] = (1.0, 1.0)
    queue_time_range_s: tuple[float, float] = (0.0, 0.0)
    charging_available_prob: float = 1.0


@dataclass(frozen=True)
class MIPBaselineScale:
    """Define the smallest target scale used by the MIP baseline."""

    max_tasks: int = 8
    max_vehicles: int = 2
    max_charging_stations: int = 2
    max_rules: int = 13
    max_decision_epochs: int = 3
    max_scenarios: int = 3


@dataclass(frozen=True)
class MIPBaselineSolverConfig:
    """Default solver choice and switches for the baseline model."""

    solver_name: str = "ortools"
    time_limit_s: float = 30.0
    mip_gap: float = 0.0
    enable_rule_selection: bool = True
    enable_conflict: bool = True
    enable_partial_charging: bool = True
    scenario_mode: str = "minimal" # scenario changes for minimal or medium instances
    charging_level_ratios: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    rule_candidate_top_k: int = 2
    rule6_charge_level_ratios: tuple[float, ...] = (0.3, 0.5, 0.8)
    rule7_min_charge_ratio: float = 0.8
    rule5_soc_threshold: float = 0.2
    min_soc_threshold: float | None = None  # Optional hard SOC floor for simulation/execution layer. can change None to configure the value.
    standby_beta: float = 1.0
    charging_queue_default_s: float = 0.0
    charging_queue_estimates_s: dict[int, float] = field(default_factory=dict)
    wait_weight_default: float = 1.0
    wait_weight_charging: float = 3.0
    wait_weight_depot: float = 0.5
    wait_weight_scale: float = 1.0
    cost_params: CostParameters = field(default_factory=CostParameters)
    auto_synthesize_scenarios: bool = True
    scenario_synth_config: ScenarioSynthConfig = field(default_factory=ScenarioSynthConfig)
    scenario_synth_seed: int | None = None
