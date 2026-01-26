"""State construction helpers for RL-ready observations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.task import Task, TaskPool, TaskStatus
from core.vehicle import Vehicle
from core.node import ChargingNode


@dataclass
class ChargerState:
    node_id: int
    coordinates: Tuple[float, float] = (0.0, 0.0)
    estimated_wait_s: float = 0.0
    queue_length: int = 0
    is_occupied: bool = False


@dataclass
class TrafficState:
    node_occupancy: Dict[int, float] = field(default_factory=dict)
    edge_occupancy: Dict[Tuple[int, int], float] = field(default_factory=dict)


@dataclass
class EpisodeMetrics:
    total_distance: float = 0.0
    total_charging: float = 0.0
    total_delay: float = 0.0
    total_conflict_waiting: float = 0.0
    total_standby: float = 0.0
    rejected_tasks: int = 0
    mask_total: int = 0
    mask_blocked: int = 0
    mask_fallbacks: int = 0


@dataclass
class SimulatorState:
    t: float
    robots: Dict[int, Vehicle]
    open_tasks: Dict[int, Task]
    accepted_tasks: Set[int]
    rejected_tasks: Set[int]
    chargers: Dict[int, ChargerState]
    traffic: TrafficState
    metrics: EpisodeMetrics


def build_simulator_state(
    *,
    task_pool: TaskPool,
    vehicles: Iterable[Vehicle],
    t: float,
    pending_task_ids: Optional[Iterable[int]] = None,
    chargers: Optional[Sequence[ChargingNode]] = None,
    queue_estimates: Optional[Dict[int, float]] = None,
    traffic: Optional[TrafficState] = None,
    metrics: Optional[EpisodeMetrics] = None,
) -> SimulatorState:
    vehicles_by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}

    pending_ids = set(pending_task_ids) if pending_task_ids is not None else None
    open_tasks: Dict[int, Task] = {}
    accepted_tasks: Set[int] = set()
    rejected_tasks: Set[int] = set()

    for task in task_pool.get_all_tasks():
        tracker = task_pool.get_tracker(task.task_id)
        if tracker is None:
            continue
        if tracker.status == TaskStatus.REJECTED:
            rejected_tasks.add(task.task_id)
            continue
        if tracker.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS, TaskStatus.COMPLETED):
            accepted_tasks.add(task.task_id)
        if tracker.status != TaskStatus.COMPLETED:
            if pending_ids is None or task.task_id in pending_ids:
                open_tasks[task.task_id] = task

    charger_states: Dict[int, ChargerState] = {}
    if chargers:
        for charger in chargers:
            estimate = 0.0
            if queue_estimates is not None:
                estimate = queue_estimates.get(charger.node_id, 0.0)
            charger_states[charger.node_id] = ChargerState(
                node_id=charger.node_id,
                coordinates=charger.coordinates,
                estimated_wait_s=estimate,
            )

    return SimulatorState(
        t=t,
        robots=vehicles_by_id,
        open_tasks=open_tasks,
        accepted_tasks=accepted_tasks,
        rejected_tasks=rejected_tasks,
        chargers=charger_states,
        traffic=traffic or TrafficState(),
        metrics=metrics or EpisodeMetrics(),
    )


def env_get_obs(
    state: SimulatorState,
    last_event: Optional[object],
    *,
    top_k_tasks: int = 5,
    top_k_chargers: int = 3,
) -> Dict[str, List[float]]:
    """Build a fixed-size observation dict for RL agents."""

    vehicle_features: List[float] = []
    for vehicle_id in sorted(state.robots.keys()):
        vehicle = state.robots[vehicle_id]
        soc = (
            vehicle.current_battery / vehicle.battery_capacity
            if vehicle.battery_capacity > 0
            else 0.0
        )
        idle_flag = 1.0 if vehicle.is_idle() else 0.0
        x, y = vehicle.current_location
        load_ratio = vehicle.current_load / vehicle.capacity if vehicle.capacity > 0 else 0.0
        dist_to_charger = _distance_to_nearest_charger(vehicle.current_location, state)
        vehicle_features.extend([soc, idle_flag, x, y, load_ratio, dist_to_charger])

    task_features: List[float] = []
    tasks = list(state.open_tasks.values())
    tasks.sort(key=lambda task: _task_due_time(task))
    for task in tasks[:top_k_tasks]:
        due_time = _task_due_time(task)
        min_dist = _distance_to_nearest_vehicle(task, state)
        priority = float(task.priority)
        feasible = 1.0 if _task_capacity_feasible(task, state) else 0.0
        task_features.extend([due_time, min_dist, priority, feasible])

    missing_tasks = top_k_tasks - len(tasks)
    if missing_tasks > 0:
        task_features.extend([0.0] * missing_tasks * 4)

    charger_features: List[float] = []
    chargers = list(state.chargers.values())
    chargers.sort(key=lambda c: c.node_id)
    for charger in chargers[:top_k_chargers]:
        charger_features.extend(
            [float(charger.estimated_wait_s), float(charger.queue_length), float(charger.is_occupied)]
        )
    missing_chargers = top_k_chargers - len(chargers)
    if missing_chargers > 0:
        charger_features.extend([0.0] * missing_chargers * 3)

    event_one_hot = _encode_event(last_event)
    meta_features = [state.t] + event_one_hot

    return {
        "vehicles": vehicle_features,
        "tasks": task_features,
        "chargers": charger_features,
        "meta": meta_features,
    }


def _task_due_time(task: Task) -> float:
    due_candidates = []
    if task.pickup_node.time_window:
        due_candidates.append(task.pickup_node.time_window.latest)
    if task.delivery_node.time_window:
        due_candidates.append(task.delivery_node.time_window.latest)
    if not due_candidates:
        return 0.0
    return min(due_candidates)


def _distance_to_nearest_vehicle(task: Task, state: SimulatorState) -> float:
    pickup = task.pickup_node.coordinates
    if not state.robots:
        return 0.0
    return min(
        _euclidean(vehicle.current_location, pickup)
        for vehicle in state.robots.values()
    )


def _task_capacity_feasible(task: Task, state: SimulatorState) -> bool:
    if not state.robots:
        return False
    return any(task.demand <= vehicle.capacity for vehicle in state.robots.values())


def _distance_to_nearest_charger(position: Tuple[float, float], state: SimulatorState) -> float:
    if not state.chargers:
        return 0.0
    return min(
        _euclidean(position, charger_node)
        for charger_node in _charger_positions(state)
    )


def _charger_positions(state: SimulatorState) -> List[Tuple[float, float]]:
    positions = []
    for charger in state.chargers.values():
        positions.append(_infer_charger_position(charger))
    return positions


def _infer_charger_position(charger: ChargerState) -> Tuple[float, float]:
    return charger.coordinates


def _encode_event(event: Optional[object]) -> List[float]:
    event_type = getattr(event, "event_type", "NONE") if event else "NONE"
    mapping = [
        "NONE",
        "TASK_ARRIVAL",
        "ROBOT_IDLE",
        "CHARGE_DONE",
        "CONFLICT_RESOLVED",
    ]
    return [1.0 if event_type == key else 0.0 for key in mapping]


def _euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


__all__ = [
    "ChargerState",
    "TrafficState",
    "EpisodeMetrics",
    "SimulatorState",
    "build_simulator_state",
    "env_get_obs",
]
