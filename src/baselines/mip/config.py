"""Configuration for the MIP baseline model."""

from __future__ import annotations

from dataclasses import dataclass, field

from config import CostParameters, DEFAULT_TIME_SYSTEM, DEFAULT_VEHICLE_DEFAULTS


@dataclass(frozen=True)
class ScenarioSynthConfig:
    """Configuration for synthetic scenario generation."""

    # Episode / horizon
    episode_length_s: float = 28_800.0  # 8h (paper Section 5.1)

    # Arrival model (NHPP with 3 segments, peak:normal:off-peak = 3:1:0.5)
    use_nhpp_arrivals: bool = True
    # Release-time sampling mode:
    # - "fixed_count": keep task count fixed, sample release-time density via NHPP.
    # - "thinning": sample via NHPP thinning first, then map to tasks (best-effort).
    arrival_time_sampling_mode: str = "fixed_count"
    nhpp_base_rate_per_s: float = 0.0005
    nhpp_peak_multiplier: float = 3.0
    nhpp_normal_multiplier: float = 1.0
    nhpp_offpeak_multiplier: float = 0.5
    nhpp_segment_fractions: tuple[float, float, float, float] = (
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0,
    )

    # Demand model (truncated normal)
    use_truncnorm_demands: bool = True
    demand_mean_kg: float = 75.0
    demand_std_kg: float = 22.5
    demand_min_kg: float = 0.0
    demand_max_kg: float = DEFAULT_VEHICLE_DEFAULTS.capacity_kg

    # Time windows anchored to task release time (seconds)
    use_release_time_windows: bool = True
    pickup_tw_width_s: float = 1800.0
    delivery_tw_width_s: float = 3600.0
    vehicle_speed_m_s: float = DEFAULT_TIME_SYSTEM.vehicle_speed_m_s

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

    # Covers benchmark scales S/M/L/XL (up to 100 tasks, 12 vehicles, 6 chargers).
    max_tasks: int = 100
    max_vehicles: int = 12
    max_charging_stations: int = 6
    max_rules: int = 13
    max_decision_epochs: int = 128
    max_scenarios: int = 16


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
