"""Shared factories and helpers for optimisation-scale ALNS tests.

Utilities in this module generate deterministic demand scenarios, construct
routes and vehicles, and wrap ALNS execution so the small/medium/large
optimisation tests can focus on assertions instead of boilerplate setup.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Sequence, Tuple
from copy import deepcopy
import random

from core.node import DepotNode, ChargingNode, NodeType, create_task_node_pair
from core.route import create_empty_route
from core.task import Task, TaskPool
from core.vehicle import Vehicle, create_vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeWindow, TimeWindowType
from config import (
    CostParameters,
    DestroyRepairParams,
    LPRepairParams,
    MatheuristicParams,
    OptimizationScenarioDefaults,
    SegmentOptimizationParams,
    DEFAULT_ALNS_HYPERPARAMETERS,
    DEFAULT_OPTIMIZATION_SCENARIO,
    OPTIMIZATION_SCENARIO_PRESETS,
)
from planner.alns import MinimalALNS
from planner.alns_matheuristic import MatheuristicALNS
from planner.fleet import FleetPlanner
from tests.optimization.presets import get_scale_preset


@dataclass
class ScenarioConfig:
    """High level parameters for building reproducible scenarios."""

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
    vehicle_speed: float = DEFAULT_OPTIMIZATION_SCENARIO.vehicle_speed

    @classmethod
    def from_defaults(
        cls, defaults: OptimizationScenarioDefaults, **overrides
    ) -> "ScenarioConfig":
        """Create a scenario configuration from shared defaults."""

        params = {**asdict(defaults), **overrides}
        return cls(**params)


@dataclass
class Scenario:
    """Bundle of objects required to run an ALNS optimization test."""

    depot: DepotNode
    tasks: List[Task]
    distance: DistanceMatrix
    vehicles: List[Vehicle]
    energy: EnergyConfig

    def create_task_pool(self) -> TaskPool:
        pool = TaskPool()
        pool.add_tasks(self.tasks)
        return pool

    def create_vehicles(self) -> List[Vehicle]:
        return [deepcopy(vehicle) for vehicle in self.vehicles]


def build_scenario(config: ScenarioConfig) -> Scenario:
    """Generate a deterministic scenario for a given configuration."""

    rng = random.Random(config.seed)

    depot = DepotNode(coordinates=(0.0, 0.0))
    coordinates: Dict[int, Tuple[float, float]] = {0: depot.coordinates}

    width, height = config.area_size

    base_cs_id = config.num_tasks * 2 + 1
    for idx in range(config.num_charging):
        node_id = base_cs_id + idx
        offset_x = (idx + 1) / (config.num_charging + 1)
        coordinate = (width * offset_x, height * (0.3 + 0.4 * offset_x))
        station = ChargingNode(
            node_id=node_id,
            coordinates=coordinate,
            node_type=NodeType.CHARGING,
        )
        coordinates[node_id] = station.coordinates

    tasks: List[Task] = []
    for task_id in range(1, config.num_tasks + 1):
        pickup_coords = (
            rng.uniform(width * 0.1, width * 0.9),
            rng.uniform(height * 0.1, height * 0.9),
        )
        delivery_coords = (
            rng.uniform(width * 0.1, width * 0.9),
            rng.uniform(height * 0.1, height * 0.9),
        )

        pickup_tw_start = task_id * config.service_time
        pickup_tw_end = pickup_tw_start + config.pickup_tw_width
        delivery_tw_start = pickup_tw_end + config.delivery_gap
        delivery_tw_end = delivery_tw_start + config.pickup_tw_width

        pickup, delivery = create_task_node_pair(
            task_id=task_id,
            pickup_id=task_id * 2 - 1,
            delivery_id=task_id * 2,
            pickup_coords=pickup_coords,
            delivery_coords=delivery_coords,
            demand=config.vehicle_capacity * 0.08,
            service_time=config.service_time,
            pickup_time_window=TimeWindow(
                earliest=pickup_tw_start,
                latest=pickup_tw_end,
                window_type=TimeWindowType.SOFT,
            ),
            delivery_time_window=TimeWindow(
                earliest=delivery_tw_start,
                latest=delivery_tw_end,
                window_type=TimeWindowType.SOFT,
            ),
        )

        task = Task(
            task_id=task_id,
            pickup_node=pickup,
            delivery_node=delivery,
            demand=pickup.demand,
        )
        tasks.append(task)
        coordinates[pickup.node_id] = pickup.coordinates
        coordinates[delivery.node_id] = delivery.coordinates

    distance = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=config.num_tasks,
        num_charging_stations=config.num_charging,
    )

    if config.num_vehicles <= 0:
        raise ValueError("ScenarioConfig.num_vehicles must be at least 1")

    vehicles: List[Vehicle] = []
    for idx in range(config.num_vehicles):
        vehicle = create_vehicle(
            vehicle_id=idx + 1,
            capacity=config.vehicle_capacity,
            battery_capacity=config.battery_capacity,
            initial_battery=config.battery_capacity,
        )
        vehicle.speed = config.vehicle_speed
        vehicles.append(vehicle)

    reference_vehicle = vehicles[0]

    energy = EnergyConfig(
        consumption_rate=config.consumption_per_km / 1000.0 * reference_vehicle.speed,
        charging_rate=config.charging_rate,
        battery_capacity=config.battery_capacity,
    )

    return Scenario(
        depot=depot,
        tasks=tasks,
        distance=distance,
        vehicles=vehicles,
        energy=energy,
    )


def get_scale_config(scale: str, **overrides) -> ScenarioConfig:
    """Return a ``ScenarioConfig`` tuned for the requested optimisation scale."""

    preset = get_scale_preset(scale)
    defaults = OPTIMIZATION_SCENARIO_PRESETS[scale]
    scenario_kwargs: Dict[str, object] = dict(preset.scenario_overrides)
    scenario_kwargs.update(overrides)
    return ScenarioConfig.from_defaults(defaults, **scenario_kwargs)


def get_solver_iterations(scale: str, solver: str) -> int:
    """Return the unified iteration budget for ``solver`` on ``scale`` tests."""

    preset = get_scale_preset(scale)
    try:
        return getattr(preset.iterations, solver)
    except AttributeError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Unknown solver key '{solver}' for scale '{scale}'") from exc


def run_minimal_trial(
    scenario: Scenario,
    *,
    iterations: int,
    seed: int = 2024,
    repair_mode: str = "adaptive",
    cost_params: CostParameters | None = None,
) -> Tuple[float, float]:
    """Run the Minimal ALNS solver and return the baseline/optimised costs."""

    rng = random.Random(seed)
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        task_pool = scenario.create_task_pool()
        hyper_params = replace(
            DEFAULT_ALNS_HYPERPARAMETERS,
            destroy_repair=DestroyRepairParams(
                random_removal_q=1,
                partial_removal_q=1,
                remove_cs_probability=0.1,
            ),
        )

        alns = MinimalALNS(
            distance_matrix=scenario.distance,
            task_pool=task_pool,
            repair_mode=repair_mode,
            cost_params=cost_params or CostParameters(),
            charging_strategy=None,
            use_adaptive=True,
            verbose=False,
            hyper_params=hyper_params,
        )
        alns.vehicle = deepcopy(scenario.vehicles[0])
        alns.energy_config = deepcopy(scenario.energy)

        initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
        removed_task_ids = [task.task_id for task in scenario.tasks]
        baseline = alns.greedy_insertion(initial_route, removed_task_ids)
        optimised = alns.optimize(baseline, max_iterations=iterations)

        return alns.evaluate_cost(baseline), alns.evaluate_cost(optimised)
    finally:
        random.setstate(state)


def run_matheuristic_trial(
    scenario: Scenario,
    strategy,
    *,
    iterations: int,
    seed: int = 2024,
    cost_params: CostParameters | None = None,
) -> Tuple[float, float]:
    """Run Matheuristic ALNS with FULL matheuristic capabilities enabled.

    This function uses tuned but fully-enabled matheuristic parameters for
    fair comparison with Q-learning variant.
    """

    rng = random.Random(seed)
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        task_pool = scenario.create_task_pool()
        # Use tuned matheuristic parameters that are strong but not extreme
        hyper_params = replace(
            DEFAULT_ALNS_HYPERPARAMETERS,
            destroy_repair=DestroyRepairParams(
                random_removal_q=2,
                partial_removal_q=2,
                remove_cs_probability=0.2,
            ),
            matheuristic=MatheuristicParams(
                elite_pool_size=4,
                intensification_interval=25,
                segment_frequency=6,
                max_elite_trials=2,
                segment_optimization=SegmentOptimizationParams(
                    max_segment_tasks=3,
                    candidate_pool_size=3,
                    improvement_tolerance=1e-3,
                    max_permutations=12,
                    lookahead_window=2,
                ),
                lp_repair=LPRepairParams(
                    time_limit_s=0.3,
                    max_plans_per_task=4,
                    improvement_tolerance=1e-4,
                    skip_penalty=5_000.0,
                    fractional_threshold=1e-3,
                ),
            ),
        )

        if len(scenario.vehicles) == 1:
            alns = MatheuristicALNS(
                distance_matrix=scenario.distance,
                task_pool=task_pool,
                repair_mode="adaptive",
                cost_params=cost_params or CostParameters(),
                charging_strategy=strategy,
                use_adaptive=True,
                verbose=False,
                adaptation_mode="roulette",  # Use roulette wheel for matheuristic baseline
                hyper_params=hyper_params,
            )
            alns.vehicle = deepcopy(scenario.vehicles[0])
            alns.energy_config = deepcopy(scenario.energy)

            initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
            removed_task_ids = [task.task_id for task in scenario.tasks]
            baseline = alns.greedy_insertion(initial_route, removed_task_ids)

            if hasattr(alns, "_segment_optimizer"):
                alns._segment_optimizer._ensure_schedule(baseline)
                baseline_cost = alns._safe_evaluate(baseline)
            else:
                baseline_cost = alns.evaluate_cost(baseline)

            optimized = alns.optimize(baseline, max_iterations=iterations)

            if hasattr(alns, "_segment_optimizer"):
                alns._segment_optimizer._ensure_schedule(optimized)
                optimized_cost = alns._safe_evaluate(optimized)
            else:
                optimized_cost = alns.evaluate_cost(optimized)

            return baseline_cost, optimized_cost

        planner = FleetPlanner(
            distance_matrix=scenario.distance,
            depot=scenario.depot,
            vehicles=scenario.create_vehicles(),
            task_pool=task_pool,
            energy_config=deepcopy(scenario.energy),
            cost_params=cost_params,
            charging_strategy=strategy,
            repair_mode="adaptive",
            use_adaptive=True,
            verbose=False,
            alns_class=MatheuristicALNS,
            alns_hyper_params=hyper_params,
        )
        result = planner.plan_routes(max_iterations=iterations)

        return result.initial_cost, result.optimised_cost
    finally:
        random.setstate(state)


def run_alns_trial(
    scenario: Scenario,
    strategy,
    *,
    iterations: int,
    seed: int = 2024,
    cost_params: CostParameters | None = None,
) -> Tuple[float, float]:
    """Run ALNS for a given strategy and return (initial_cost, optimized_cost).

    NOTE: This function uses MINIMAL matheuristic settings for backwards
    compatibility. For full matheuristic capabilities, use run_matheuristic_trial().
    """

    rng = random.Random(seed)
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        task_pool = scenario.create_task_pool()
        hyper_params = replace(
            DEFAULT_ALNS_HYPERPARAMETERS,
            destroy_repair=DestroyRepairParams(
                random_removal_q=1,
                partial_removal_q=1,
                remove_cs_probability=0.1,
            ),
            matheuristic=MatheuristicParams(
                elite_pool_size=1,
                intensification_interval=0,
                segment_frequency=0,
                max_elite_trials=0,
                lp_repair=LPRepairParams(time_limit_s=0.01, max_plans_per_task=1),
            ),
        )

        if len(scenario.vehicles) == 1:
            # 使用升级后的 Matheuristic ALNS，以覆盖 LP 修复算子等改进
            alns = MatheuristicALNS(
                distance_matrix=scenario.distance,
                task_pool=task_pool,
                repair_mode="adaptive",
                cost_params=cost_params or CostParameters(),
                charging_strategy=strategy,
                use_adaptive=True,
                verbose=False,
                hyper_params=hyper_params,
            )
            alns.vehicle = deepcopy(scenario.vehicles[0])
            alns.energy_config = deepcopy(scenario.energy)

            initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
            removed_task_ids = [task.task_id for task in scenario.tasks]
            baseline = alns.greedy_insertion(initial_route, removed_task_ids)
            optimized = alns.optimize(baseline, max_iterations=iterations)

            return alns.evaluate_cost(baseline), alns.evaluate_cost(optimized)

        planner = FleetPlanner(
            distance_matrix=scenario.distance,
            depot=scenario.depot,
            vehicles=scenario.create_vehicles(),
            task_pool=task_pool,
            energy_config=deepcopy(scenario.energy),
            cost_params=cost_params,
            charging_strategy=strategy,
            repair_mode="adaptive",
            use_adaptive=True,
            verbose=False,
            alns_class=MatheuristicALNS,
            alns_hyper_params=hyper_params,
        )
        result = planner.plan_routes(max_iterations=iterations)

        return result.initial_cost, result.optimised_cost
    finally:
        random.setstate(state)


def summarize_improvements(cost_pairs: Sequence[Tuple[float, float]]) -> float:
    """Return the average relative improvement across all trials."""

    deltas = [
        (initial - optimized) / initial
        for initial, optimized in cost_pairs
        if initial > 0.0
    ]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)
