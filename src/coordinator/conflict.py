"""Conflict resolution helpers for multi-vehicle headway constraints.

The utilities here provide a minimal, deterministic way to derive conflict
waiting times that enforce precedence on shared nodes and directed edges.
They do not solve the full multi-agent scheduling problem; instead they compute
lower-bound waiting adjustments that callers can feed into route schedules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from core.route import Route


@dataclass(frozen=True)
class HeadwayPolicy:
    """Default headway policy for node and edge precedence."""

    min_headway_s: float = 2.0


def _priority_for_vehicle(
    vehicle_id: int,
    priorities: Optional[Dict[int, int]],
) -> int:
    if priorities and vehicle_id in priorities:
        return priorities[vehicle_id]
    return 0


def compute_conflict_waits(
    routes: Dict[int, Route],
    *,
    headway: Optional[HeadwayPolicy] = None,
    priorities: Optional[Dict[int, int]] = None,
) -> Dict[int, List[float]]:
    """Compute per-visit conflict waiting times for node and edge headways.

    The returned lists align with each route's node order. Callers should
    re-run scheduling after applying the waits to propagate timing shifts.
    """

    policy = headway or HeadwayPolicy()
    waits: Dict[int, List[float]] = {}

    for vehicle_id, route in routes.items():
        if not route.visits:
            raise ValueError("Routes must have computed visits before conflict analysis.")
        waits[vehicle_id] = [0.0] * len(route.nodes)

    # Node precedence: only one vehicle can occupy a node at a time.
    events_by_node: Dict[int, List[Tuple[int, float, int, int, float]]] = {}
    for vehicle_id, route in routes.items():
        priority = _priority_for_vehicle(vehicle_id, priorities)
        for idx, visit in enumerate(route.visits or []):
            duration = max(0.0, visit.departure_time - visit.start_service_time)
            events_by_node.setdefault(visit.node.node_id, []).append(
                (priority, visit.arrival_time, vehicle_id, idx, duration)
            )

    for _, events in events_by_node.items():
        events.sort(key=lambda item: (-item[0], item[1], item[2]))
        last_departure: Optional[float] = None
        for _, arrival, vehicle_id, idx, duration in events:
            earliest_start = arrival if last_departure is None else max(
                arrival, last_departure + policy.min_headway_s
            )
            extra_wait = max(0.0, earliest_start - arrival)
            waits[vehicle_id][idx] = max(waits[vehicle_id][idx], extra_wait)
            last_departure = earliest_start + duration

    # Edge precedence: maintain headway on shared directed edges.
    edges: Dict[Tuple[int, int], List[Tuple[int, float, int, int]]] = {}
    for vehicle_id, route in routes.items():
        priority = _priority_for_vehicle(vehicle_id, priorities)
        for idx in range(len(route.visits or []) - 1):
            origin = route.nodes[idx].node_id
            dest = route.nodes[idx + 1].node_id
            departure = route.visits[idx].departure_time
            edges.setdefault((origin, dest), []).append(
                (priority, departure, vehicle_id, idx)
            )

    for _, events in edges.items():
        events.sort(key=lambda item: (-item[0], item[1], item[2]))
        last_departure = None
        for _, departure, vehicle_id, idx in events:
            earliest_departure = departure if last_departure is None else max(
                departure, last_departure + policy.min_headway_s
            )
            extra_wait = max(0.0, earliest_departure - departure)
            waits[vehicle_id][idx] = max(waits[vehicle_id][idx], extra_wait)
            last_departure = earliest_departure

    return waits


def apply_conflict_waits(routes: Dict[int, Route], waits: Dict[int, List[float]]) -> None:
    """Apply pre-computed conflict waits to route objects."""

    for vehicle_id, route in routes.items():
        route.set_conflict_waiting_times(waits.get(vehicle_id))


def resolve_conflicts(
    routes: Dict[int, Route],
    *,
    headway: Optional[HeadwayPolicy] = None,
    priorities: Optional[Dict[int, int]] = None,
) -> Dict[int, List[float]]:
    """Compute and apply conflict waits based on headway rules."""

    waits = compute_conflict_waits(
        routes,
        headway=headway,
        priorities=priorities,
    )
    apply_conflict_waits(routes, waits)
    return waits


__all__ = [
    "HeadwayPolicy",
    "compute_conflict_waits",
    "apply_conflict_waits",
    "resolve_conflicts",
]
