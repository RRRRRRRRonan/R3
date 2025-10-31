"""Central repository for tunable planner, AMR, and factory defaults.

All frequently adjusted numerical values that influence optimisation behaviour,
vehicle simulation, or charging heuristics are collected here so they can be
updated from a single location without touching algorithmic code.  The
constants are exposed as frozen dataclasses to provide structure and discover-
ability while keeping them easily serialisable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class LPRepairParams:
    """Parameters steering the LP-based repair operator inspired by Singh et al."""

    time_limit_s: float = 0.5
    max_plans_per_task: int = 6
    improvement_tolerance: float = 1e-4
    skip_penalty: float = 5_000.0
    fractional_threshold: float = 1e-3


@dataclass(frozen=True)
class SegmentOptimizationParams:
    """Configuration for the MILP-inspired segment optimiser in Matheuristic ALNS."""

    max_segment_tasks: int = 4
    candidate_pool_size: int = 5
    improvement_tolerance: float = 1e-3
    max_permutations: int = 24
    lookahead_window: int = 3


@dataclass(frozen=True)
class MatheuristicParams:
    """Higher-level knobs steering matheuristic intensification behaviour."""

    elite_pool_size: int = 6
    intensification_interval: int = 40
    segment_frequency: int = 4
    max_elite_trials: int = 3
    segment_optimization: SegmentOptimizationParams = field(default_factory=SegmentOptimizationParams)
    lp_repair: LPRepairParams = field(default_factory=LPRepairParams)


@dataclass(frozen=True)
class VehicleDefaults:
    """Baseline AMR characteristics used by vehicle factories."""

    capacity_kg: float = 150.0
    battery_capacity_kwh: float = 100.0
    cruise_speed_m_s: float = 2.0
    depot_location: Tuple[float, float] = (0.0, 0.0)
    initial_battery_ratio: float = 1.0
    initial_load_kg: float = 0.0


@dataclass(frozen=True)
class EnergySystemDefaults:
    """Factory floor energy model parameters shared across modules."""

    consumption_rate: float = 0.5
    charging_rate: float = 50.0
    charging_efficiency: float = 0.9
    max_charging_time_s: float = 3600.0
    max_charging_amount: float = 100.0
    battery_capacity_kwh: float = 100.0
    safety_threshold: float = 0.05
    warning_threshold: float = 0.15
    comfort_threshold: float = 0.25
    critical_battery_threshold: float = 0.0


@dataclass(frozen=True)
class TimeSystemDefaults:
    """Global timing assumptions for AMR movement and service."""

    vehicle_speed_m_s: float = 15.0
    default_service_time_s: float = 30.0
    tardiness_penalty: float = 1.0


@dataclass(frozen=True)
class CostParameters:
    """Weight configuration for the multi-objective route cost model."""

    C_tr: float = 1.0
    C_ch: float = 0.6
    C_time: float = 0.1
    C_delay: float = 2.0
    C_wait: float = 0.05

    C_missing_task: float = 10000.0
    C_infeasible: float = 10000.0

    def get_total_cost(self, distance: float, charging: float,
                      time: float, delay: float, waiting: float) -> float:
        """Return the weighted sum of the route performance components."""

        return (
            self.C_tr * distance
            + self.C_ch * charging
            + self.C_time * time
            + self.C_delay * delay
            + self.C_wait * waiting
        )


@dataclass(frozen=True)
class FactoryParameters:
    """Bundle of defaults covering AMR, energy, time, and cost settings."""

    vehicle: VehicleDefaults = field(default_factory=VehicleDefaults)
    energy: EnergySystemDefaults = field(default_factory=EnergySystemDefaults)
    time: TimeSystemDefaults = field(default_factory=TimeSystemDefaults)
    costs: CostParameters = field(default_factory=CostParameters)


@dataclass(frozen=True)
class AdaptiveSelectorParams:
    """Weight update configuration for adaptive operator selection."""

    initial_weight: float = 1.0
    decay_factor: float = 0.8
    sigma_best: int = 33
    sigma_accept: int = 9
    sigma_improve: int = 13


@dataclass(frozen=True)
class SimulatedAnnealingParams:
    """Cooling schedule used by the simulated annealing acceptance rule."""

    initial_temperature: float = 100.0
    cooling_rate: float = 0.995


@dataclass(frozen=True)
class DestroyRepairParams:
    """Defaults that govern destroy/repair neighbourhood usage."""

    random_removal_q: int = 2
    partial_removal_q: int = 2
    remove_cs_probability: float = 0.3


@dataclass(frozen=True)
class ChargingDefaults:
    """Fallback charging assumptions for penalty and duration estimates."""

    penalty_per_station: float = 100.0
    fallback_duration_s: float = 10.0


@dataclass(frozen=True)
class VehicleDynamics:
    """Nominal vehicle behaviour used in deterministic simulations."""

    cruise_speed_m_s: float = 1.5
    max_energy_adjustment_iterations: int = 10


@dataclass(frozen=True)
class ALNSHyperParameters:
    """Bundle of configuration blocks used by the ALNS planner."""

    adaptive_selector: AdaptiveSelectorParams = field(default_factory=AdaptiveSelectorParams)
    simulated_annealing: SimulatedAnnealingParams = field(default_factory=SimulatedAnnealingParams)
    destroy_repair: DestroyRepairParams = field(default_factory=DestroyRepairParams)
    charging: ChargingDefaults = field(default_factory=ChargingDefaults)
    vehicle: VehicleDynamics = field(default_factory=VehicleDynamics)
    matheuristic: MatheuristicParams = field(default_factory=MatheuristicParams)
    q_learning: "QLearningParams" = field(default_factory=lambda: QLearningParams())


@dataclass(frozen=True)
class QLearningParams:
    """Hyper-parameters for the Q-learning operator agent."""

    alpha: float = 0.1
    gamma: float = 0.9
    initial_epsilon: float = 0.4
    epsilon_decay: float = 0.92
    epsilon_min: float = 0.05
    reward_new_best: float = 50.0
    reward_improvement: float = 20.0
    reward_accepted: float = 5.0
    reward_rejected: float = -2.0
    stagnation_threshold: int = 200
    deep_stagnation_threshold: int = 800
    stagnation_ratio: float = 0.25
    deep_stagnation_ratio: float = 0.6


@dataclass(frozen=True)
class OptimizationScenarioDefaults:
    """Baseline optimisation scenario parameters used in regression tests."""

    num_tasks: int
    num_charging: int
    area_size: Tuple[float, float]
    vehicle_capacity: float
    battery_capacity: float
    consumption_per_km: float
    charging_rate: float
    num_vehicles: int = 1
    seed: int = 42
    service_time: float = 45.0
    pickup_tw_width: float = 120.0
    delivery_gap: float = 30.0
    vehicle_speed: float = VehicleDynamics().cruise_speed_m_s


DEFAULT_VEHICLE_DEFAULTS = VehicleDefaults()
DEFAULT_ENERGY_SYSTEM = EnergySystemDefaults()
DEFAULT_TIME_SYSTEM = TimeSystemDefaults()
DEFAULT_COST_PARAMETERS = CostParameters()
DEFAULT_FACTORY_PARAMETERS = FactoryParameters()
DEFAULT_ADAPTIVE_SELECTOR_PARAMS = AdaptiveSelectorParams()
DEFAULT_SIMULATED_ANNEALING_PARAMS = SimulatedAnnealingParams()
DEFAULT_DESTROY_REPAIR_PARAMS = DestroyRepairParams()
DEFAULT_CHARGING_DEFAULTS = ChargingDefaults()
DEFAULT_VEHICLE_DYNAMICS = VehicleDynamics()
DEFAULT_LP_REPAIR_PARAMS = LPRepairParams()
DEFAULT_SEGMENT_OPTIMIZATION_PARAMS = SegmentOptimizationParams()
DEFAULT_MATHEURISTIC_PARAMS = MatheuristicParams()
DEFAULT_Q_LEARNING_PARAMS = QLearningParams()
DEFAULT_ALNS_HYPERPARAMETERS = ALNSHyperParameters()
DEFAULT_OPTIMIZATION_SCENARIO = OptimizationScenarioDefaults(
    num_tasks=10,
    num_charging=1,
    area_size=(1000.0, 1000.0),
    vehicle_capacity=200.0,
    battery_capacity=1.2,
    consumption_per_km=0.4,
    charging_rate=5.0 / 3600.0,
    seed=7,
)
OPTIMIZATION_SCENARIO_PRESETS: Dict[str, OptimizationScenarioDefaults] = {
    "small": DEFAULT_OPTIMIZATION_SCENARIO,
    "medium": OptimizationScenarioDefaults(
        num_tasks=30,
        num_charging=2,
        area_size=(2000.0, 2000.0),
        vehicle_capacity=220.0,
        battery_capacity=1.6,
        consumption_per_km=0.45,
        charging_rate=5.5 / 3600.0,
        seed=11,
    ),
    "large": OptimizationScenarioDefaults(
        num_tasks=50,
        num_charging=3,
        area_size=(3000.0, 3000.0),
        vehicle_capacity=240.0,
        battery_capacity=2.0,
        consumption_per_km=0.5,
        charging_rate=6.0 / 3600.0,
        seed=17,
    ),
}
