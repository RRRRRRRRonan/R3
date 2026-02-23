"""Instantiate rule IDs into atomic actions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from config import DEFAULT_COST_PARAMETERS
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


@dataclass(frozen=True)
class AtomicAction:
    kind: str
    payload: Dict[str, object]


def apply(
    rule_id: int,
    event: Optional[Event],
    state: SimulatorState,
    *,
    soc_threshold: float = 0.2,
    energy_config: Optional[EnergyConfig] = None,
    charge_level_ratios: Sequence[float] = (0.3, 0.5, 0.8),
    rule7_min_charge_ratio: float = 0.8,
    standby_beta: float = 1.0,
    heatmap_scores: Optional[Dict[int, float]] = None,
) -> AtomicAction:
    """Return an atomic action for the given rule ID."""

    if rule_id in ACCEPT_RULES:
        return _apply_accept_rule(
            rule_id,
            event,
            state,
            soc_threshold=soc_threshold,
            energy_config=energy_config,
        )

    if rule_id in DISPATCH_RULES:
        return _apply_dispatch_rule(
            rule_id,
            event,
            state,
        )

    if rule_id in CHARGE_RULES:
        return _apply_charge_rule(
            rule_id,
            event,
            state,
            soc_threshold=soc_threshold,
            energy_config=energy_config,
            charge_level_ratios=charge_level_ratios,
            rule7_min_charge_ratio=rule7_min_charge_ratio,
        )

    if rule_id in STANDBY_RULES:
        return _apply_standby_rule(
            rule_id,
            event,
            state,
            standby_beta=standby_beta,
            heatmap_scores=heatmap_scores,
        )

    return AtomicAction(kind="DWELL", payload=_fallback_dwell_payload(state, event))


def _apply_accept_rule(
    rule_id: int,
    event: Optional[Event],
    state: SimulatorState,
    *,
    soc_threshold: float,
    energy_config: Optional[EnergyConfig],
) -> AtomicAction:
    task = _select_task_for_event(event, state)
    task_id = task.task_id if task is not None else -1

    accept = 0
    if task is not None and state.robots:
        if rule_id == RULE_ACCEPT_FEASIBLE:
            accept = 1 if _any_vehicle_feasible(task, state, energy_config) else 0
        elif rule_id == RULE_ACCEPT_VALUE:
            accept = 1 if _value_accept(task, state, energy_config) else 0
        else:
            accept = 1

    return AtomicAction(kind="ACCEPT", payload={"task_id": task_id, "accept": accept})


def _apply_dispatch_rule(
    rule_id: int,
    event: Optional[Event],
    state: SimulatorState,
) -> AtomicAction:
    vehicles = _candidate_vehicles(event, state)
    tasks = _candidate_tasks(event, state)

    if not vehicles or not tasks:
        return AtomicAction(kind="DWELL", payload=_fallback_dwell_payload(state, event))

    if rule_id == RULE_STTF:
        vehicle, task = _select_pair_min_travel(vehicles, tasks)
        mode = "uncommitted"
    elif rule_id == RULE_EDD:
        task = _select_task_earliest_due(tasks)
        vehicle = _nearest_vehicle(task, vehicles)
        mode = "dispatch"
    elif rule_id == RULE_MST:
        task, vehicle = _select_task_min_slack(tasks, vehicles, state)
        mode = "dispatch"
    elif rule_id == RULE_HPF:
        task = _select_task_highest_priority(tasks)
        vehicle = _nearest_vehicle(task, vehicles)
        mode = "dispatch"
    elif rule_id == RULE_INSERT_MIN_COST:
        vehicle, task = _select_pair_min_incremental_cost(vehicles, tasks)
        mode = "insert"
    else:
        vehicle, task = _select_pair_min_travel(vehicles, tasks)
        mode = "dispatch"

    return AtomicAction(
        kind="DISPATCH",
        payload={"robot_id": vehicle.vehicle_id, "task_id": task.task_id, "mode": mode},
    )


def _apply_charge_rule(
    rule_id: int,
    event: Optional[Event],
    state: SimulatorState,
    *,
    soc_threshold: float,
    energy_config: Optional[EnergyConfig],
    charge_level_ratios: Sequence[float],
    rule7_min_charge_ratio: float,
) -> AtomicAction:
    vehicles = _candidate_vehicles(event, state)
    if not vehicles:
        return AtomicAction(kind="DWELL", payload=_fallback_dwell_payload(state, event))

    charger_id, chosen_vehicle, travel_time, queue_time = _select_charger_and_vehicle(
        vehicles,
        state,
        energy_config,
        prefer_low_soc=(rule_id == RULE_CHARGE_URGENT),
        soc_threshold=soc_threshold,
    )

    if charger_id is None or chosen_vehicle is None:
        return AtomicAction(kind="DWELL", payload=_fallback_dwell_payload(state, event))

    current_soc = _vehicle_soc(chosen_vehicle)
    ratios = sorted(charge_level_ratios)
    if not ratios:
        ratios = [0.3, 0.5, 0.8]

    if rule_id == RULE_CHARGE_URGENT:
        soc_target = ratios[0]
    elif rule_id == RULE_CHARGE_TARGET_LOW:
        soc_target = ratios[0] if len(ratios) > 0 else 0.3
    elif rule_id == RULE_CHARGE_TARGET_MED:
        soc_target = ratios[1] if len(ratios) > 1 else 0.5
    elif rule_id == RULE_CHARGE_TARGET_HIGH:
        soc_target = ratios[2] if len(ratios) > 2 else 0.8
    elif rule_id == RULE_CHARGE_OPPORTUNITY:
        if queue_time <= 0.0:
            soc_target = max(ratios)
        else:
            soc_target = max(rule7_min_charge_ratio, _select_target_ratio(current_soc, ratios))
    else:
        soc_target = ratios[0]

    return AtomicAction(
        kind="CHARGE",
        payload={"robot_id": chosen_vehicle.vehicle_id, "charger_id": charger_id, "soc_target": soc_target},
    )


def _apply_standby_rule(
    rule_id: int,
    event: Optional[Event],
    state: SimulatorState,
    *,
    standby_beta: float,
    heatmap_scores: Optional[Dict[int, float]] = None,
) -> AtomicAction:
    vehicle = _select_vehicle_for_event(event, state)
    if vehicle is None:
        return AtomicAction(kind="DWELL", payload=_fallback_dwell_payload(state, event))

    if rule_id == RULE_STANDBY_LAZY:
        return AtomicAction(kind="DWELL", payload={"robot_id": vehicle.vehicle_id, "node_id": -1})

    if rule_id == RULE_STANDBY_HEATMAP:
        node_id = _select_heatmap_node(state, vehicle, heatmap_scores)
        return AtomicAction(kind="DWELL", payload={"robot_id": vehicle.vehicle_id, "node_id": node_id})

    node_id = _select_low_cost_node(state, vehicle, standby_beta)
    return AtomicAction(kind="DWELL", payload={"robot_id": vehicle.vehicle_id, "node_id": node_id})


def _select_task_for_event(event: Optional[Event], state: SimulatorState):
    if event and event.event_type == EVENT_TASK_ARRIVAL:
        task_id = int(event.payload.get("task_id", -1))
        if task_id in state.open_tasks:
            return state.open_tasks[task_id]
    return _select_task_earliest_due(list(state.open_tasks.values()))


def _candidate_tasks(event: Optional[Event], state: SimulatorState):
    if event and event.event_type == EVENT_TASK_ARRIVAL:
        task_id = int(event.payload.get("task_id", -1))
        if task_id in state.open_tasks:
            return [state.open_tasks[task_id]]
    return list(state.open_tasks.values())


def _candidate_vehicles(event: Optional[Event], state: SimulatorState):
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


def _select_vehicle_for_event(event: Optional[Event], state: SimulatorState):
    candidates = _candidate_vehicles(event, state)
    if candidates:
        return candidates[0]
    if state.robots:
        return list(state.robots.values())[0]
    return None


def _any_vehicle_feasible(task, state: SimulatorState, energy_config: Optional[EnergyConfig]) -> bool:
    for vehicle in state.robots.values():
        if _vehicle_can_serve_task(vehicle, task, state, energy_config):
            return True
    return False


def _value_accept(task, state: SimulatorState, energy_config: Optional[EnergyConfig]) -> bool:
    best_cost = None
    for vehicle in state.robots.values():
        incremental = _estimate_incremental_cost(vehicle, task)
        if best_cost is None or incremental < best_cost:
            best_cost = incremental
    rejection_cost = DEFAULT_COST_PARAMETERS.C_missing_task
    if best_cost is None:
        return False
    return best_cost <= rejection_cost


def _select_pair_min_travel(vehicles, tasks):
    best = None
    for task in tasks:
        for vehicle in vehicles:
            dist = _euclidean(vehicle.current_location, task.pickup_coordinates)
            if best is None or dist < best[0]:
                best = (dist, vehicle, task)
    if best is None:
        return vehicles[0], tasks[0]
    return best[1], best[2]


def _select_pair_min_incremental_cost(vehicles, tasks):
    best = None
    for task in tasks:
        for vehicle in vehicles:
            cost = _estimate_incremental_cost(vehicle, task)
            if best is None or cost < best[0]:
                best = (cost, vehicle, task)
    if best is None:
        return vehicles[0], tasks[0]
    return best[1], best[2]


def _select_task_earliest_due(tasks):
    if not tasks:
        return None
    return min(tasks, key=_task_due_time)


def _select_task_highest_priority(tasks):
    if not tasks:
        return None
    return max(tasks, key=lambda task: (task.priority, -_task_due_time(task)))


def _select_task_min_slack(tasks, vehicles, state: SimulatorState):
    best = None
    for task in tasks:
        vehicle = _nearest_vehicle(task, vehicles)
        slack = _task_slack(task, vehicle, state)
        if best is None or slack < best[0]:
            best = (slack, task, vehicle)
    if best is None:
        return tasks[0], vehicles[0]
    return best[1], best[2]


def _nearest_vehicle(task, vehicles):
    return min(
        vehicles,
        key=lambda vehicle: _euclidean(vehicle.current_location, task.pickup_coordinates),
    )


def _task_due_time(task) -> float:
    due_candidates = []
    if task.pickup_time_window:
        due_candidates.append(task.pickup_time_window.latest)
    if task.delivery_time_window:
        due_candidates.append(task.delivery_time_window.latest)
    return min(due_candidates) if due_candidates else float("inf")


def _task_slack(task, vehicle, state: SimulatorState) -> float:
    due = _task_due_time(task)
    if due == float("inf"):
        return float("inf")
    travel_to_pickup = _travel_time(vehicle.current_location, task.pickup_coordinates, vehicle.speed)
    pickup_service = task.pickup_node.service_time
    travel_to_delivery = _travel_time(task.pickup_coordinates, task.delivery_coordinates, vehicle.speed)
    return due - (state.t + travel_to_pickup + pickup_service + travel_to_delivery)


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
    if not _hard_time_windows_feasible(vehicle, task, state):
        return False
    return True


def _energy_to_pickup_feasible(vehicle, task, energy_config: Optional[EnergyConfig]) -> bool:
    config = energy_config or EnergyConfig()
    distance = _euclidean(vehicle.current_location, task.pickup_coordinates)
    required = calculate_energy_consumption(
        distance=distance,
        load=vehicle.current_load,
        config=config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    return required <= vehicle.current_battery + 1e-6


def _hard_time_windows_feasible(vehicle, task, state: SimulatorState) -> bool:
    pickup_node = task.pickup_node
    delivery_node = task.delivery_node
    arrival_pickup = state.t + _travel_time(vehicle.current_location, pickup_node.coordinates, vehicle.speed)
    start_pickup = _apply_time_window(arrival_pickup, pickup_node.time_window)
    if start_pickup is None:
        return False
    depart_pickup = start_pickup + pickup_node.service_time
    arrival_delivery = depart_pickup + _travel_time(
        pickup_node.coordinates, delivery_node.coordinates, vehicle.speed
    )
    start_delivery = _apply_time_window(arrival_delivery, delivery_node.time_window)
    return start_delivery is not None


def _apply_time_window(arrival_time: float, window: Optional[TimeWindow]) -> Optional[float]:
    if window is None:
        return arrival_time
    if window.window_type == TimeWindowType.SOFT:
        return max(arrival_time, window.earliest)
    if arrival_time > window.latest:
        return None
    return max(arrival_time, window.earliest)


def _estimate_incremental_cost(vehicle, task) -> float:
    dist_to_pickup = _euclidean(vehicle.current_location, task.pickup_coordinates)
    dist_pickup_delivery = _euclidean(task.pickup_coordinates, task.delivery_coordinates)
    return DEFAULT_COST_PARAMETERS.C_tr * (dist_to_pickup + dist_pickup_delivery)


def _select_charger_and_vehicle(
    vehicles,
    state: SimulatorState,
    energy_config: Optional[EnergyConfig],
    *,
    prefer_low_soc: bool,
    soc_threshold: float,
):
    if not state.chargers:
        return None, None, 0.0, 0.0

    candidate_vehicles = list(vehicles)
    if prefer_low_soc:
        low_soc = [v for v in candidate_vehicles if _vehicle_soc(v) <= soc_threshold]
        if low_soc:
            candidate_vehicles = low_soc
    candidate_vehicles.sort(key=_vehicle_soc)

    best = None
    for vehicle in candidate_vehicles:
        for charger in state.chargers.values():
            if not charger.is_available:
                continue
            travel = _travel_time(vehicle.current_location, charger.coordinates, vehicle.speed)
            queue = charger.estimated_wait_s
            if energy_config is not None:
                required = calculate_energy_consumption(
                    distance=_euclidean(vehicle.current_location, charger.coordinates),
                    load=vehicle.current_load,
                    config=energy_config,
                    vehicle_speed=vehicle.speed,
                    vehicle_capacity=vehicle.capacity,
                )
                if required > vehicle.current_battery + 1e-6:
                    continue
            score = travel + queue
            if best is None or score < best[0]:
                best = (score, charger.node_id, vehicle, travel, queue)

    if best is None:
        return None, None, 0.0, 0.0
    return best[1], best[2], best[3], best[4]


def _select_target_ratio(current_soc: float, ratios: Sequence[float]) -> float:
    for ratio in ratios:
        if ratio >= current_soc + 1e-6:
            return ratio
    return max(ratios)


def _select_low_cost_node(state: SimulatorState, vehicle, standby_beta: float) -> int:
    best = None
    for task in state.open_tasks.values():
        travel = _travel_time(vehicle.current_location, task.pickup_coordinates, vehicle.speed)
        score = travel + standby_beta
        if best is None or score < best[0]:
            best = (score, task.pickup_node.node_id)

    if best is None:
        return 0
    return best[1]


def _select_heatmap_node(
    state: SimulatorState,
    vehicle,
    heatmap_scores: Optional[Dict[int, float]] = None,
) -> int:
    if heatmap_scores:
        candidates = _collect_candidate_nodes(state, exclude_chargers=True)
        best = None
        for node_id in candidates:
            score = heatmap_scores.get(node_id)
            if score is None:
                continue
            if best is None or score > best[0]:
                best = (score, node_id)
        if best is not None:
            return best[1]

    if not state.open_tasks:
        return 0
    best_task = max(
        state.open_tasks.values(),
        key=lambda task: (task.priority, -_task_due_time(task)),
    )
    return best_task.pickup_node.node_id


def _collect_candidate_nodes(state: SimulatorState, *, exclude_chargers: bool = False) -> List[int]:
    nodes = {0}
    for task in state.open_tasks.values():
        nodes.add(task.pickup_node.node_id)
        nodes.add(task.delivery_node.node_id)
    if not exclude_chargers:
        for charger in state.chargers.values():
            nodes.add(charger.node_id)
    return list(nodes)


def _fallback_dwell_payload(state: SimulatorState, event: Optional[Event]) -> Dict[str, object]:
    vehicle = _select_vehicle_for_event(event, state)
    robot_id = vehicle.vehicle_id if vehicle is not None else -1
    return {"robot_id": robot_id, "node_id": 0}


def _vehicle_soc(vehicle) -> float:
    if vehicle.battery_capacity <= 0:
        return 0.0
    return vehicle.current_battery / vehicle.battery_capacity


def _travel_time(a: Tuple[float, float], b: Tuple[float, float], speed: float) -> float:
    return calculate_travel_time(_euclidean(a, b), speed)


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


__all__ = ["AtomicAction", "apply"]
