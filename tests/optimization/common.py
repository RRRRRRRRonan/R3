"""Shared factories and helpers for optimisation-scale ALNS tests.

Utilities in this module generate deterministic demand scenarios, construct
routes and vehicles, and wrap ALNS execution so the small/medium/large
optimisation tests can focus on assertions instead of boilerplate setup.
"""
from __future__ import annotations

from dataclasses import dataclass
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
from planner.alns import CostParameters, MinimalALNS


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
    seed: int = 42
    service_time: float = 45.0
    pickup_tw_width: float = 120.0
    delivery_gap: float = 30.0


@dataclass
class Scenario:
    """Bundle of objects required to run an ALNS optimization test."""

    depot: DepotNode
    tasks: List[Task]
    distance: DistanceMatrix
    vehicle: Vehicle
    energy: EnergyConfig

    def create_task_pool(self) -> TaskPool:
        pool = TaskPool()
        pool.add_tasks(self.tasks)
        return pool


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

    vehicle = create_vehicle(
        vehicle_id=1,
        capacity=config.vehicle_capacity,
        battery_capacity=config.battery_capacity,
        initial_battery=config.battery_capacity,
    )
    vehicle.speed = 1.5

    energy = EnergyConfig(
        consumption_rate=config.consumption_per_km / 1000.0 * vehicle.speed,
        charging_rate=config.charging_rate,
        battery_capacity=config.battery_capacity,
    )

    return Scenario(
        depot=depot,
        tasks=tasks,
        distance=distance,
        vehicle=vehicle,
        energy=energy,
    )


def run_alns_trial(
    scenario: Scenario,
    strategy,
    *,
    iterations: int,
    seed: int = 2024,
    cost_params: CostParameters | None = None,
) -> Tuple[float, float]:
    """Run ALNS for a given strategy and return (initial_cost, optimized_cost)."""

    rng = random.Random(seed)
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        task_pool = scenario.create_task_pool()
        alns = MinimalALNS(
            distance_matrix=scenario.distance,
            task_pool=task_pool,
            repair_mode="adaptive",
            cost_params=cost_params or CostParameters(),
            charging_strategy=strategy,
            use_adaptive=True,
            verbose=False,
        )
        alns.vehicle = deepcopy(scenario.vehicle)
        alns.energy_config = deepcopy(scenario.energy)

        initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
        removed_task_ids = [task.task_id for task in scenario.tasks]
        baseline = alns.greedy_insertion(initial_route, removed_task_ids)
        optimized = alns.optimize(baseline, max_iterations=iterations)

        return alns.evaluate_cost(baseline), alns.evaluate_cost(optimized)
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
