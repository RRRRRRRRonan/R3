"""Scenario builders and epoch recorder bridging RL and MIP."""

from __future__ import annotations

import random
from typing import Callable, Dict, Iterable, List, Optional, Sequence

from baselines.mip.model import MIPBaselineScenario
from core.node import ChargingNode
from core.task import TaskPool
from core.vehicle import Vehicle
from coordinator.traffic_manager import TrafficManager
from strategy.execution_layer import ExecutionLayer
from strategy.rule_env import RuleSelectionEnv
from strategy.simulator import EventDrivenSimulator


def build_scenario_from_task_pool(
    task_pool: TaskPool,
    *,
    scenario_id: int = 0,
    probability: float = 1.0,
    chargers: Optional[Sequence[ChargingNode]] = None,
    queue_estimates: Optional[Dict[int, float]] = None,
    travel_time_factor: float = 1.0,
    charging_availability: Optional[Dict[int, int]] = None,
    decision_epoch_times: Optional[List[float]] = None,
) -> MIPBaselineScenario:
    tasks = task_pool.get_all_tasks()
    task_availability = {task.task_id: 1 for task in tasks}
    task_release_times = {task.task_id: float(task.arrival_time) for task in tasks}
    task_demands = {task.task_id: float(task.demand) for task in tasks}
    node_time_windows: Dict[int, tuple[float, float]] = {}
    node_service_times: Dict[int, float] = {}
    for task in tasks:
        pickup = task.pickup_node
        delivery = task.delivery_node
        if pickup.time_window:
            node_time_windows[pickup.node_id] = (
                pickup.time_window.earliest,
                pickup.time_window.latest,
            )
        if delivery.time_window:
            node_time_windows[delivery.node_id] = (
                delivery.time_window.earliest,
                delivery.time_window.latest,
            )
        node_service_times[pickup.node_id] = pickup.service_time
        node_service_times[delivery.node_id] = delivery.service_time

    queue_estimates = queue_estimates or {}
    charging_availability = charging_availability or {}
    if chargers:
        for charger in chargers:
            queue_estimates.setdefault(charger.node_id, 0.0)
            charging_availability.setdefault(charger.node_id, 1)

    if not decision_epoch_times:
        decision_epoch_times = sorted(set(task_release_times.values()))
        if 0.0 not in decision_epoch_times:
            decision_epoch_times.insert(0, 0.0)

    return MIPBaselineScenario(
        scenario_id=scenario_id,
        probability=probability,
        task_availability=task_availability,
        task_release_times=task_release_times,
        node_time_windows=node_time_windows,
        node_service_times=node_service_times,
        task_demands=task_demands,
        queue_estimates_s=queue_estimates,
        travel_time_factor=travel_time_factor,
        charging_availability=charging_availability,
        decision_epoch_times=decision_epoch_times,
    )


def record_event_epochs(
    task_pool: TaskPool,
    vehicles: Sequence[Vehicle],
    chargers: Optional[Sequence[ChargingNode]] = None,
    *,
    max_steps: int = 200,
    seed: int = 0,
    policy_fn: Optional[
        Callable[[Sequence[bool], dict, dict, RuleSelectionEnv], int]
    ] = None,
) -> List[float]:
    """Run a light simulation to capture event-driven decision epochs."""

    rng = random.Random(seed)
    traffic = TrafficManager()
    simulator = EventDrivenSimulator(
        task_pool=task_pool,
        vehicles=vehicles,
        chargers=chargers,
        traffic_manager=traffic,
    )
    executor = ExecutionLayer(task_pool=task_pool, simulator=simulator, traffic_manager=traffic)
    env = RuleSelectionEnv(
        simulator=simulator,
        execution_layer=executor,
        max_decision_steps=max_steps,
    )

    obs, info = env.reset(seed=seed)
    epoch_times: List[float] = []
    if info.get("event_time") is not None:
        epoch_times.append(float(info["event_time"]))

    for _ in range(max_steps):
        mask = env.action_masks()
        if policy_fn is None:
            valid_actions = [idx for idx, allowed in enumerate(mask) if allowed]
            action = rng.choice(valid_actions) if valid_actions else 0
        else:
            action = policy_fn(mask, obs, info, env)
        result = env.step(action)
        obs, info = result.obs, result.info
        if info.get("event_time") is not None:
            epoch_times.append(float(info["event_time"]))
        if result.terminated or result.truncated:
            break

    return sorted(set(epoch_times)) or [0.0]


def attach_epoch_times(
    scenario: MIPBaselineScenario,
    epoch_times: Iterable[float],
) -> MIPBaselineScenario:
    return MIPBaselineScenario(
        scenario_id=scenario.scenario_id,
        probability=scenario.probability,
        task_availability=dict(scenario.task_availability),
        task_release_times=dict(scenario.task_release_times),
        node_time_windows=dict(scenario.node_time_windows),
        node_service_times=dict(scenario.node_service_times),
        task_demands=dict(scenario.task_demands),
        arrival_time_shift_s=scenario.arrival_time_shift_s,
        time_window_scale=scenario.time_window_scale,
        priority_boost=scenario.priority_boost,
        queue_estimates_s=dict(scenario.queue_estimates_s),
        travel_time_factor=scenario.travel_time_factor,
        charging_availability=dict(scenario.charging_availability),
        decision_epoch_times=sorted(set(float(t) for t in epoch_times)),
    )


def apply_scenario_to_simulator(
    simulator: EventDrivenSimulator,
    scenario: MIPBaselineScenario,
) -> None:
    """Apply scenario perturbations to the event-driven simulator."""
    simulator.apply_scenario(
        queue_estimates=scenario.queue_estimates_s or None,
        charging_availability=scenario.charging_availability or None,
        travel_time_factor=scenario.travel_time_factor,
        task_availability=scenario.task_availability or None,
        task_release_times=scenario.task_release_times or None,
        task_demands=scenario.task_demands or None,
    )


__all__ = [
    "build_scenario_from_task_pool",
    "record_event_epochs",
    "attach_epoch_times",
    "apply_scenario_to_simulator",
]
