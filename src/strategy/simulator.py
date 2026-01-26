"""Event-driven simulator for decision-epoch triggering.

This module implements a minimal event queue that surfaces decision moments
(task arrivals and robot idle events) without prescribing the RL policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.task import TaskPool
from core.vehicle import Vehicle
from core.node import ChargingNode

from strategy.state import EpisodeMetrics, SimulatorState, TrafficState, build_simulator_state
from coordinator.traffic_manager import TrafficManager
from core.vehicle import VehicleStatus

EVENT_TASK_ARRIVAL = "TASK_ARRIVAL"
EVENT_ROBOT_IDLE = "ROBOT_IDLE"
EVENT_CHARGE_DONE = "CHARGE_DONE"
EVENT_SOC_LOW = "SOC_LOW"
EVENT_DEADLOCK_RISK = "DEADLOCK_RISK"
EVENT_CONFLICT_RESOLVED = "CONFLICT_RESOLVED"

DECISION_EVENTS = {
    EVENT_TASK_ARRIVAL,
    EVENT_ROBOT_IDLE,
    EVENT_CHARGE_DONE,
    EVENT_SOC_LOW,
    EVENT_DEADLOCK_RISK,
}


@dataclass(order=True)
class Event:
    """Discrete event used by the event-driven simulator."""

    sort_index: Tuple[float, int, int] = field(init=False, repr=False)
    time: float
    event_type: str
    payload: Dict[str, object] = field(default_factory=dict, compare=False)
    priority: int = field(default=0, compare=False)
    seq: int = field(default=0, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sort_index", (self.time, self.priority, self.seq))


class EventQueue:
    """Min-heap based event queue."""

    def __init__(self) -> None:
        self._heap: List[Event] = []
        self._counter = itertools.count()

    def push(self, event: Event) -> None:
        event.seq = next(self._counter)
        event.sort_index = (event.time, event.priority, event.seq)
        heapq.heappush(self._heap, event)

    def pop(self) -> Optional[Event]:
        if not self._heap:
            return None
        return heapq.heappop(self._heap)

    def peek(self) -> Optional[Event]:
        if not self._heap:
            return None
        return self._heap[0]

    def clear(self) -> None:
        self._heap.clear()

    def __len__(self) -> int:
        return len(self._heap)


@dataclass
class EventDrivenState:
    """Snapshot of the simulator state at a decision epoch."""

    time: float
    pending_task_ids: List[int]
    idle_vehicle_ids: List[int]
    last_event: Optional[Event] = None


class EventDrivenSimulator:
    """Minimal event-driven kernel for decision epochs."""

    def __init__(
        self,
        *,
        task_pool: TaskPool,
        vehicles: Iterable[Vehicle],
        chargers: Optional[Sequence[ChargingNode]] = None,
        queue_estimates: Optional[Dict[int, float]] = None,
        charging_availability: Optional[Dict[int, int]] = None,
        travel_time_factor: float = 1.0,
        soc_low_threshold: float = 0.2,
        deadlock_wait_threshold_s: float = 30.0,
        deadlock_cooldown_s: float = 10.0,
        traffic_manager: Optional[TrafficManager] = None,
    ) -> None:
        self.task_pool = task_pool
        self.vehicles = list(vehicles)
        self.chargers = list(chargers) if chargers else []
        self.queue_estimates = queue_estimates
        self.charging_availability = charging_availability or {}
        self.travel_time_factor = max(0.0, travel_time_factor)
        self.soc_low_threshold = max(0.0, soc_low_threshold)
        self.deadlock_wait_threshold_s = max(0.0, deadlock_wait_threshold_s)
        self.deadlock_cooldown_s = max(0.0, deadlock_cooldown_s)
        self.traffic_manager = traffic_manager
        self.event_queue = EventQueue()
        self.current_time = 0.0
        self.metrics = EpisodeMetrics()
        self._pending_task_ids: Set[int] = set()
        self._idle_vehicle_ids: Set[int] = set()
        self._arrived_task_ids: Set[int] = set()
        self._soc_low_vehicles: Set[int] = set()
        self._last_deadlock_time: Optional[float] = None

    def reset(self) -> EventDrivenState:
        """Reset the simulator and seed the initial events."""
        self.event_queue.clear()
        self.current_time = 0.0
        self.metrics = EpisodeMetrics()
        if self.traffic_manager is not None:
            self.traffic_manager.clear()
        self._pending_task_ids.clear()
        self._idle_vehicle_ids.clear()
        self._arrived_task_ids.clear()
        self._soc_low_vehicles.clear()
        self._last_deadlock_time = None

        for task in self.task_pool.get_all_tasks():
            self.event_queue.push(
                Event(
                    time=float(task.arrival_time),
                    event_type=EVENT_TASK_ARRIVAL,
                    payload={"task_id": task.task_id},
                )
            )

        for vehicle in self.vehicles:
            self.event_queue.push(
                Event(
                    time=0.0,
                    event_type=EVENT_ROBOT_IDLE,
                    payload={"vehicle_id": vehicle.vehicle_id},
                )
            )
            self.update_soc_status(vehicle.vehicle_id, time=0.0)

        return self._snapshot(None)

    def advance_to_next_decision_epoch(self) -> Tuple[Optional[Event], EventDrivenState]:
        """Advance the simulation to the next decision event."""
        while len(self.event_queue) > 0:
            event = self.event_queue.pop()
            if event is None:
                break
            self.current_time = event.time
            self._apply_event(event)
            if event.event_type in DECISION_EVENTS:
                return event, self._snapshot(event)
        return None, self._snapshot(None)

    def mark_task_assigned(self, task_id: int, vehicle_id: int) -> None:
        """Update internal state when a task is assigned to a vehicle."""
        self._pending_task_ids.discard(task_id)
        self._idle_vehicle_ids.discard(vehicle_id)

    def mark_task_rejected(self, task_id: int) -> None:
        """Update internal state when a task is rejected."""
        self._pending_task_ids.discard(task_id)

    def mark_vehicle_idle(self, vehicle_id: int, time: Optional[float] = None) -> None:
        """Schedule a robot idle event at the given time."""
        event_time = self.current_time if time is None else float(time)
        self.event_queue.push(
            Event(
                time=event_time,
                event_type=EVENT_ROBOT_IDLE,
                payload={"vehicle_id": vehicle_id},
            )
        )

    def mark_charge_done(self, vehicle_id: int, time: Optional[float] = None) -> None:
        """Schedule a charging completion event."""
        event_time = self.current_time if time is None else float(time)
        self.event_queue.push(
            Event(
                time=event_time,
                event_type=EVENT_CHARGE_DONE,
                payload={"vehicle_id": vehicle_id},
            )
        )

    def mark_soc_low(self, vehicle_id: int, time: Optional[float] = None) -> None:
        """Schedule a SOC low warning event."""
        event_time = self.current_time if time is None else float(time)
        self.event_queue.push(
            Event(
                time=event_time,
                event_type=EVENT_SOC_LOW,
                payload={"vehicle_id": vehicle_id},
            )
        )

    def mark_deadlock_risk(self, time: Optional[float] = None, payload: Optional[Dict[str, object]] = None) -> None:
        """Schedule a deadlock risk event."""
        event_time = self.current_time if time is None else float(time)
        self.event_queue.push(
            Event(
                time=event_time,
                event_type=EVENT_DEADLOCK_RISK,
                payload=payload or {},
            )
        )

    def mark_vehicle_busy(self, vehicle_id: int) -> None:
        """Remove a vehicle from the idle pool when it starts an action."""
        self._idle_vehicle_ids.discard(vehicle_id)

    def update_soc_status(self, vehicle_id: int, *, time: Optional[float] = None) -> None:
        """Auto-detect low SOC and trigger decision event when needed."""
        vehicle = self._find_vehicle(vehicle_id)
        if vehicle is None or vehicle.battery_capacity <= 0:
            return
        soc = vehicle.current_battery / vehicle.battery_capacity
        if soc <= self.soc_low_threshold:
            if vehicle_id not in self._soc_low_vehicles:
                self._soc_low_vehicles.add(vehicle_id)
                self.mark_soc_low(vehicle_id, time=time)
        else:
            self._soc_low_vehicles.discard(vehicle_id)

    def maybe_trigger_deadlock(self, *, wait_s: float, time: Optional[float] = None, payload: Optional[Dict[str, object]] = None) -> None:
        """Heuristic deadlock-risk trigger based on excessive waiting."""
        if wait_s < self.deadlock_wait_threshold_s:
            return
        event_time = self.current_time if time is None else float(time)
        if self._last_deadlock_time is not None:
            if event_time - self._last_deadlock_time < self.deadlock_cooldown_s:
                return
        self._last_deadlock_time = event_time
        self.mark_deadlock_risk(time=event_time, payload=payload)

    def _apply_event(self, event: Event) -> None:
        if event.event_type == EVENT_TASK_ARRIVAL:
            task_id = int(event.payload.get("task_id", -1))
            if task_id >= 0:
                self._arrived_task_ids.add(task_id)
                self._pending_task_ids.add(task_id)
        elif event.event_type == EVENT_ROBOT_IDLE:
            vehicle_id = int(event.payload.get("vehicle_id", -1))
            if vehicle_id >= 0:
                self._idle_vehicle_ids.add(vehicle_id)
                self._set_vehicle_status(vehicle_id, VehicleStatus.IDLE)
        elif event.event_type == EVENT_CHARGE_DONE:
            vehicle_id = int(event.payload.get("vehicle_id", -1))
            if vehicle_id >= 0:
                self._idle_vehicle_ids.add(vehicle_id)
                self._set_vehicle_status(vehicle_id, VehicleStatus.IDLE)
        elif event.event_type == EVENT_SOC_LOW:
            return
        elif event.event_type == EVENT_DEADLOCK_RISK:
            return
        elif event.event_type == EVENT_CONFLICT_RESOLVED:
            return

    def _snapshot(self, event: Optional[Event]) -> EventDrivenState:
        return EventDrivenState(
            time=self.current_time,
            pending_task_ids=sorted(self._pending_task_ids),
            idle_vehicle_ids=sorted(self._idle_vehicle_ids),
            last_event=event,
        )

    def build_state(self, *, event: Optional[Event] = None) -> SimulatorState:
        """Build a full RL-ready state snapshot."""
        traffic_state = None
        if self.traffic_manager is not None:
            node_occupancy, edge_occupancy = self.traffic_manager.get_snapshot()
            traffic_state = TrafficState(node_occupancy=node_occupancy, edge_occupancy=edge_occupancy)
        return build_simulator_state(
            task_pool=self.task_pool,
            vehicles=self.vehicles,
            t=self.current_time,
            pending_task_ids=self._pending_task_ids,
            chargers=self.chargers,
            queue_estimates=self.queue_estimates,
            charging_availability=self.charging_availability,
            traffic=traffic_state,
            metrics=self.metrics,
        )

    def apply_scenario(
        self,
        *,
        queue_estimates: Optional[Dict[int, float]] = None,
        charging_availability: Optional[Dict[int, int]] = None,
        travel_time_factor: Optional[float] = None,
    ) -> None:
        """Update simulator-level scenario perturbations."""
        if queue_estimates is not None:
            self.queue_estimates = queue_estimates
        if charging_availability is not None:
            self.charging_availability = charging_availability
        if travel_time_factor is not None:
            self.travel_time_factor = max(0.0, travel_time_factor)

    def _set_vehicle_status(self, vehicle_id: int, status: VehicleStatus) -> None:
        for vehicle in self.vehicles:
            if vehicle.vehicle_id == vehicle_id:
                vehicle.status = status
                vehicle.current_time = max(vehicle.current_time, self.current_time)
                break

    def _find_vehicle(self, vehicle_id: int) -> Optional[Vehicle]:
        for vehicle in self.vehicles:
            if vehicle.vehicle_id == vehicle_id:
                return vehicle
        return None


__all__ = [
    "Event",
    "EventQueue",
    "EventDrivenState",
    "EventDrivenSimulator",
    "EVENT_TASK_ARRIVAL",
    "EVENT_ROBOT_IDLE",
    "EVENT_CHARGE_DONE",
    "EVENT_SOC_LOW",
    "EVENT_DEADLOCK_RISK",
    "EVENT_CONFLICT_RESOLVED",
]
