"""Traffic manager with reservation-table headway enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Reservation:
    start: float
    end: float
    robot_id: int


@dataclass(frozen=True)
class PathEdge:
    from_node: int
    to_node: int
    travel_time: float
    distance: float


class TrafficManager:
    """Simple reservation table enforcing edge/node headways."""

    def __init__(self, *, headway_s: float = 2.0) -> None:
        self.headway_s = headway_s
        self._edge_table: Dict[Tuple[int, int], List[Reservation]] = {}
        self._node_table: Dict[int, List[Reservation]] = {}

    def reserve_path(
        self,
        robot_id: int,
        path_edges: Sequence[PathEdge],
        earliest_start: float,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """Reserve a sequence of edges, returning per-edge timing and total conflict wait."""
        schedule: List[Tuple[float, float, float]] = []
        current = earliest_start
        total_conflict = 0.0
        for edge in path_edges:
            start, end, wait = self.reserve_edge(
                robot_id,
                (edge.from_node, edge.to_node),
                edge.travel_time,
                current,
            )
            schedule.append((start, end, wait))
            total_conflict += wait
            current = end
        return schedule, total_conflict

    def preview_reserve_path(
        self,
        path_edges: Sequence[PathEdge],
        earliest_start: float,
    ) -> Tuple[List[Tuple[float, float, float]], float]:
        """Preview reservations without mutating the reservation tables."""
        schedule: List[Tuple[float, float, float]] = []
        current = earliest_start
        total_conflict = 0.0
        for edge in path_edges:
            start, end, wait = self.preview_reserve_edge(
                (edge.from_node, edge.to_node),
                edge.travel_time,
                current,
            )
            schedule.append((start, end, wait))
            total_conflict += wait
            current = end
        return schedule, total_conflict

    def preview_reserve_edge(
        self,
        edge_key: Tuple[int, int],
        travel_time: float,
        earliest_start: float,
    ) -> Tuple[float, float, float]:
        """Preview a single edge reservation without mutating state."""
        reservations = self._edge_table.get(edge_key, [])
        start = _find_slot(reservations, earliest_start, travel_time, self.headway_s)
        end = start + max(0.0, travel_time)
        return start, end, max(0.0, start - earliest_start)

    def preview_reserve_node(
        self,
        node_id: int,
        earliest_start: float,
        dwell_time: float,
    ) -> Tuple[float, float, float]:
        """Preview a node reservation without mutating state."""
        reservations = self._node_table.get(node_id, [])
        start = _find_slot(reservations, earliest_start, dwell_time, self.headway_s)
        end = start + max(0.0, dwell_time)
        return start, end, max(0.0, start - earliest_start)

    def reserve_edge(
        self,
        robot_id: int,
        edge_key: Tuple[int, int],
        travel_time: float,
        earliest_start: float,
    ) -> Tuple[float, float, float]:
        reservations = self._edge_table.setdefault(edge_key, [])
        start = _find_slot(reservations, earliest_start, travel_time, self.headway_s)
        end = start + max(0.0, travel_time)
        reservations.append(Reservation(start=start, end=end, robot_id=robot_id))
        return start, end, max(0.0, start - earliest_start)

    def reserve_node(
        self,
        robot_id: int,
        node_id: int,
        earliest_start: float,
        dwell_time: float,
    ) -> Tuple[float, float, float]:
        reservations = self._node_table.setdefault(node_id, [])
        start = _find_slot(reservations, earliest_start, dwell_time, self.headway_s)
        end = start + max(0.0, dwell_time)
        reservations.append(Reservation(start=start, end=end, robot_id=robot_id))
        return start, end, max(0.0, start - earliest_start)

    def reserve_node_with_window(
        self,
        robot_id: int,
        node_id: int,
        arrival_time: float,
        service_time: float,
        *,
        time_window: Optional[object],
    ) -> Tuple[float, float, float, float]:
        """Reserve a node, respecting time windows for earliest service start."""
        waiting = 0.0
        earliest_start = arrival_time
        if time_window is not None:
            earliest = getattr(time_window, "earliest", None)
            if earliest is not None and arrival_time < earliest:
                waiting = earliest - arrival_time
                earliest_start = earliest
        start, end, conflict_wait = self.reserve_node(
            robot_id,
            node_id,
            earliest_start,
            service_time,
        )
        return start, end, conflict_wait, waiting

    def get_snapshot(self) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float]]:
        """Return simple occupancy summaries for RL state construction."""
        node_occupancy = {node_id: float(len(res)) for node_id, res in self._node_table.items()}
        edge_occupancy = {edge: float(len(res)) for edge, res in self._edge_table.items()}
        return node_occupancy, edge_occupancy

    def clear(self) -> None:
        self._edge_table.clear()
        self._node_table.clear()


def _find_slot(
    reservations: Sequence[Reservation],
    earliest_start: float,
    duration: float,
    headway: float,
) -> float:
    if duration <= 0:
        return earliest_start

    start = earliest_start
    ordered = sorted(reservations, key=lambda res: res.start)
    for res in ordered:
        if start + duration <= res.start - headway:
            break
        if start <= res.end + headway:
            start = res.end + headway
    return start


__all__ = ["TrafficManager", "Reservation", "PathEdge"]
