"""Action masking / shielding for rule-selection policies."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from physics.energy import EnergyConfig, calculate_energy_consumption
from physics.time import TimeWindow, TimeWindowType, calculate_travel_time
from strategy.rule_gating import (
    ACCEPT_RULES,
    CHARGE_RULES,
    DISPATCH_RULES,
    STANDBY_RULES,
    RULE_ACCEPT_FEASIBLE,
    RULE_ACCEPT_VALUE,
    RULE_CHARGE_OPPORTUNITY,
    RULE_CHARGE_TARGET_HIGH,
    RULE_CHARGE_TARGET_LOW,
    RULE_CHARGE_TARGET_MED,
    RULE_CHARGE_URGENT,
    RULE_EDD,
    RULE_HPF,
    RULE_INSERT_MIN_COST,
    RULE_MST,
    RULE_STANDBY_HEATMAP,
    RULE_STANDBY_LAZY,
    RULE_STANDBY_LOW_COST,
    RULE_STTF,
    get_available_rules,
)
from strategy.simulator import (
    EVENT_CHARGE_DONE,
    EVENT_DEADLOCK_RISK,
    EVENT_ROBOT_IDLE,
    EVENT_SOC_LOW,
    EVENT_TASK_ARRIVAL,
    Event,
)
from strategy.state import SimulatorState

ALL_RULES: List[int] = [
    RULE_STTF,
    RULE_EDD,
    RULE_MST,
    RULE_HPF,
    RULE_CHARGE_URGENT,
    RULE_CHARGE_TARGET_LOW,
    RULE_CHARGE_TARGET_MED,
    RULE_CHARGE_TARGET_HIGH,
    RULE_CHARGE_OPPORTUNITY,
    RULE_STANDBY_LOW_COST,
    RULE_STANDBY_LAZY,
    RULE_STANDBY_HEATMAP,
    RULE_ACCEPT_FEASIBLE,
    RULE_ACCEPT_VALUE,
    RULE_INSERT_MIN_COST,
]


def action_masks(
    event: Optional[Event],
    state: SimulatorState,
    *,
    soc_threshold: float = 0.2,
    energy_config: Optional[EnergyConfig] = None,
    return_numpy: bool = False,
) -> Sequence[bool]:
    """Return a rule-level action mask (length=15) after gating + feasibility."""

    available = set(get_available_rules(event, state, soc_threshold=soc_threshold))
    mask: List[bool] = []
    for rule_id in ALL_RULES:
        allowed = rule_id in available
        if allowed:
            allowed = rule_feasible(
                rule_id,
                event,
                state,
                soc_threshold=soc_threshold,
                energy_config=energy_config,
            )
        mask.append(allowed)

    fallback_used = False
    if not any(mask):
        fallback_used = True
        if available:
            mask = [rule_id in available for rule_id in ALL_RULES]
        else:
            mask = [True] * len(ALL_RULES)

    _update_mask_metrics(state, available, mask, fallback_used)

    if return_numpy:
        try:
            import numpy as np
        except Exception:
            return mask
        return np.array(mask, dtype=bool)

    return mask


def rule_feasible(
    rule_id: int,
    event: Optional[Event],
    state: SimulatorState,
    *,
    soc_threshold: float = 0.2,
    energy_config: Optional[EnergyConfig] = None,
) -> bool:
    """Quick feasibility screen for a given rule (permissive shield)."""

    tasks = _candidate_tasks(event, state)
    vehicles = _candidate_vehicles(event, state)

    if rule_id in ACCEPT_RULES:
        return bool(tasks)

    if rule_id in DISPATCH_RULES:
        if not tasks or not vehicles:
            return False
        return _any_task_feasible(tasks, vehicles, state, energy_config)

    if rule_id in CHARGE_RULES:
        if not state.chargers or not vehicles:
            return False
        if rule_id == RULE_CHARGE_URGENT:
            return any(
                _vehicle_soc(vehicle) <= soc_threshold
                and _can_reach_any_charger(vehicle, state, energy_config)
                for vehicle in vehicles
            )
        if rule_id in {RULE_CHARGE_TARGET_LOW, RULE_CHARGE_TARGET_MED, RULE_CHARGE_TARGET_HIGH}:
            return any(
                vehicle.current_battery < vehicle.battery_capacity - 1e-6
                and _can_reach_any_charger(vehicle, state, energy_config)
                for vehicle in vehicles
            )
        if rule_id == RULE_CHARGE_OPPORTUNITY:
            return any(_can_reach_any_charger(vehicle, state, energy_config) for vehicle in vehicles)
        return False

    if rule_id in STANDBY_RULES:
        return bool(vehicles)

    return True


def _candidate_tasks(event: Optional[Event], state: SimulatorState) -> List[object]:
    if event and event.event_type == EVENT_TASK_ARRIVAL:
        task_id = int(event.payload.get("task_id", -1))
        if task_id in state.open_tasks:
            return [state.open_tasks[task_id]]
    return list(state.open_tasks.values())


def _candidate_vehicles(event: Optional[Event], state: SimulatorState) -> List[object]:
    if event and event.event_type in {
        EVENT_ROBOT_IDLE,
        EVENT_CHARGE_DONE,
        EVENT_SOC_LOW,
        EVENT_DEADLOCK_RISK,
    }:
        vehicle_id = int(event.payload.get("vehicle_id", -1))
        if vehicle_id in state.robots:
            return [state.robots[vehicle_id]]
    return list(state.robots.values())


def _any_task_feasible(
    tasks: Iterable[object],
    vehicles: Iterable[object],
    state: SimulatorState,
    energy_config: Optional[EnergyConfig],
) -> bool:
    for task in tasks:
        for vehicle in vehicles:
            if _vehicle_can_serve_task(vehicle, task, state, energy_config):
                return True
    return False


def _vehicle_can_serve_task(
    vehicle: object,
    task: object,
    state: SimulatorState,
    energy_config: Optional[EnergyConfig],
) -> bool:
    if task.demand > vehicle.capacity:
        return False
    if not _energy_to_pickup_feasible(vehicle, task, energy_config):
        return False
    config = energy_config or EnergyConfig()
    min_battery = max(0.0, config.safety_threshold) * vehicle.battery_capacity
    distance_pickup = _euclidean(vehicle.current_location, task.pickup_coordinates)
    distance_delivery = _euclidean(task.pickup_coordinates, task.delivery_coordinates)
    energy_to_pickup = calculate_energy_consumption(
        distance=distance_pickup,
        load=vehicle.current_load,
        config=config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    energy_to_delivery = calculate_energy_consumption(
        distance=distance_delivery,
        load=vehicle.current_load + task.demand,
        config=config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    remaining_after_delivery = vehicle.current_battery - energy_to_pickup - energy_to_delivery
    if remaining_after_delivery < min_battery - 1e-6:
        return False
    if not _hard_time_windows_feasible(vehicle, task, state):
        return False
    return True


def _energy_to_pickup_feasible(
    vehicle: object,
    task: object,
    energy_config: Optional[EnergyConfig],
) -> bool:
    config = energy_config or EnergyConfig()
    distance = _euclidean(vehicle.current_location, task.pickup_coordinates)
    required = calculate_energy_consumption(
        distance=distance,
        load=vehicle.current_load,
        config=config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    min_battery = max(0.0, config.safety_threshold) * vehicle.battery_capacity
    return vehicle.current_battery - required >= min_battery - 1e-6


def _hard_time_windows_feasible(
    vehicle: object,
    task: object,
    state: SimulatorState,
) -> bool:
    speed = vehicle.speed
    if speed <= 0:
        return False

    pickup_node = task.pickup_node
    delivery_node = task.delivery_node

    arrival_pickup = state.t + calculate_travel_time(
        _euclidean(vehicle.current_location, pickup_node.coordinates), speed
    )
    start_pickup = _apply_time_window(arrival_pickup, pickup_node.time_window)
    if start_pickup is None:
        return False
    depart_pickup = start_pickup + pickup_node.service_time

    arrival_delivery = depart_pickup + calculate_travel_time(
        _euclidean(pickup_node.coordinates, delivery_node.coordinates), speed
    )
    start_delivery = _apply_time_window(arrival_delivery, delivery_node.time_window)
    if start_delivery is None:
        return False
    return True


def _apply_time_window(arrival_time: float, window: Optional[TimeWindow]) -> Optional[float]:
    if window is None:
        return arrival_time
    if window.window_type == TimeWindowType.SOFT:
        return max(arrival_time, window.earliest)
    if arrival_time > window.latest:
        return None
    return max(arrival_time, window.earliest)


def _can_reach_any_charger(
    vehicle: object,
    state: SimulatorState,
    energy_config: Optional[EnergyConfig],
) -> bool:
    if not state.chargers:
        return False
    config = energy_config or EnergyConfig()
    min_battery = max(0.0, config.safety_threshold) * vehicle.battery_capacity
    min_required = None
    for charger in state.chargers.values():
        if not charger.is_available:
            continue
        distance = _euclidean(vehicle.current_location, charger.coordinates)
        required = calculate_energy_consumption(
            distance=distance,
            load=vehicle.current_load,
            config=config,
            vehicle_speed=vehicle.speed,
            vehicle_capacity=vehicle.capacity,
        )
        if min_required is None or required < min_required:
            min_required = required
    if min_required is None:
        return False
    return vehicle.current_battery - min_required >= min_battery - 1e-6


def _vehicle_soc(vehicle: object) -> float:
    if vehicle.battery_capacity <= 0:
        return 0.0
    return vehicle.current_battery / vehicle.battery_capacity


def _update_mask_metrics(
    state: SimulatorState,
    available: Sequence[int],
    mask: Sequence[bool],
    fallback_used: bool,
) -> None:
    if state.metrics is None:
        return
    available_count = len(available)
    enabled_count = sum(1 for allowed in mask if allowed)
    if available_count > 0:
        blocked = max(0, available_count - enabled_count)
    else:
        blocked = 0
    state.metrics.mask_total += 1
    state.metrics.mask_blocked += blocked
    if fallback_used:
        state.metrics.mask_fallbacks += 1
        state.metrics.infeasible_actions += 1


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


__all__ = [
    "ALL_RULES",
    "action_masks",
    "rule_feasible",
]
