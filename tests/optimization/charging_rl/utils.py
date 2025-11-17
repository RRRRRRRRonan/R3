"""Helpers for exercising the charging Q-learning integration on scenarios."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

from config import QLearningParams
from core.node import ChargingNode, NodeType
from core.route import Route, RouteNodeVisit, create_empty_route
from core.route_executor import RouteExecutor
from core.task import Task
from core.vehicle import Vehicle
from planner.q_learning import ChargingQLearningAgent
from strategy.charging_strategies import PartialRechargeMinimalStrategy
from tests.optimization.common import ScenarioConfig, build_scenario, get_scale_config


@dataclass
class ChargingRLTrialResult:
    """Container returned by ``run_contextual_charging_trial`` for assertions."""

    executed_route: Route
    vehicle: Vehicle
    agent: ChargingQLearningAgent
    charging_visits: List[RouteNodeVisit]


def run_contextual_charging_trial(scale: str) -> ChargingRLTrialResult:
    """Build a deterministic scenario and execute a contextual charging run."""

    config: ScenarioConfig = get_scale_config(scale)
    scenario = build_scenario(config)

    route = _build_two_task_route(scenario.tasks, scenario.depot, scenario.distance.id_helper.get_all_charging_ids()[0], scenario.distance.coordinates)

    vehicle = deepcopy(scenario.vehicles[0])
    limited_capacity = max(20.0, scenario.energy.battery_capacity * 0.45)
    vehicle.battery_capacity = limited_capacity
    vehicle.initial_battery = limited_capacity

    energy_config = deepcopy(scenario.energy)
    energy_config.battery_capacity = limited_capacity

    params = QLearningParams(
        alpha=0.4,
        gamma=0.9,
        initial_epsilon=0.0,
        epsilon_decay=1.0,
        epsilon_min=0.0,
    )

    agent = ChargingQLearningAgent(
        params,
        initial_q_values=_uniform_state_preferences(best_index=2),
    )

    strategy = PartialRechargeMinimalStrategy(
        safety_margin=0.04,
        min_margin=0.01,
        charging_agent=agent,
        energy_config=energy_config,
    )

    executor = RouteExecutor(scenario.distance, energy_config=energy_config)
    executed_route = executor.execute(route=route, vehicle=vehicle, charging_strategy=strategy)
    charging_visits = [visit for visit in executed_route.visits if visit.node.is_charging_station()]

    return ChargingRLTrialResult(
        executed_route=executed_route,
        vehicle=vehicle,
        agent=agent,
        charging_visits=charging_visits,
    )


def _build_two_task_route(tasks: List[Task], depot, charging_node_id: int, coordinates: Dict[int, tuple]) -> Route:
    route = create_empty_route(vehicle_id=1, depot_node=depot)
    for task in tasks[:2]:
        route.add_node(task.pickup_node)
        route.add_node(task.delivery_node)

    charging_node = ChargingNode(
        node_id=charging_node_id,
        coordinates=coordinates[charging_node_id],
        node_type=NodeType.CHARGING,
    )
    route.insert_node(charging_node, 3)
    return route


def _uniform_state_preferences(best_index: int) -> Dict[str, List[float]]:
    q_values: Dict[str, List[float]] = {}
    for battery in range(4):
        for slack in range(3):
            for density in range(3):
                state = f"b{battery}|s{slack}|d{density}"
                scores = [0.0 for _ in ChargingQLearningAgent.ACTION_LEVELS]
                scores[best_index] = 1.0
                q_values[state] = scores
    return q_values
