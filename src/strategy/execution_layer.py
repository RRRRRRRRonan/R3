"""Execution layer applying atomic actions to the simulator state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from coordinator.traffic_manager import PathEdge, TrafficManager
from core.node import Node, NodeType
from core.route import Route
from core.route_executor import RouteExecutor
from core.task import TaskPool, TaskStatus
from core.vehicle import Vehicle, VehicleStatus
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig, calculate_energy_consumption
from physics.time import TimeConfig, TimeWindow, calculate_travel_time
from strategy.rules import AtomicAction
from strategy.simulator import Event, EventDrivenSimulator
from strategy.state import SimulatorState


@dataclass
class ExecutionResult:
    end_time: float
    conflict_waiting: float = 0.0
    distance: float = 0.0
    charging: float = 0.0
    delay: float = 0.0
    standby: float = 0.0


class ExecutionLayer:
    """Translate atomic actions into simulator updates."""

    def __init__(
        self,
        *,
        task_pool: TaskPool,
        simulator: EventDrivenSimulator,
        traffic_manager: TrafficManager,
        energy_config: Optional[EnergyConfig] = None,
        time_config: Optional[TimeConfig] = None,
    ) -> None:
        self.task_pool = task_pool
        self.simulator = simulator
        self.traffic_manager = traffic_manager
        self.energy_config = energy_config or EnergyConfig()
        self.time_config = time_config or TimeConfig()
        self._route_executor: Optional[RouteExecutor] = None

    def execute(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        """Execute a single atomic action and update simulator/task state."""

        if action.kind == "ACCEPT":
            return self._execute_accept(action, state, event)
        if action.kind == "DISPATCH":
            return self._execute_dispatch(action, state, event)
        if action.kind == "CHARGE":
            return self._execute_charge(action, state, event)
        if action.kind == "DWELL":
            return self._execute_dwell(action, state, event)

        return ExecutionResult(end_time=state.t)

    def _execute_accept(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        task_id = int(action.payload.get("task_id", -1))
        accept = int(action.payload.get("accept", 0))
        if task_id >= 0 and accept == 0:
            tracker = self.task_pool.get_tracker(task_id)
            if tracker and tracker.status == TaskStatus.PENDING:
                self.task_pool.reject_task(task_id)
                state.metrics.rejected_tasks += 1
                self.simulator.mark_task_rejected(task_id)
        return ExecutionResult(end_time=state.t)

    def _execute_dispatch(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        task_id = int(action.payload.get("task_id", -1))
        robot_id = int(action.payload.get("robot_id", -1))
        task = self.task_pool.get_task(task_id)
        vehicle = _get_vehicle(state, robot_id)

        if task is None or vehicle is None:
            return ExecutionResult(end_time=state.t)

        tracker = self.task_pool.get_tracker(task_id)
        if tracker and tracker.status == TaskStatus.PENDING:
            self.task_pool.assign_task(task_id, robot_id)
            self.simulator.mark_task_assigned(task_id, robot_id)
        else:
            self.simulator.mark_vehicle_busy(robot_id)

        vehicle.status = VehicleStatus.MOVING
        current_time = max(state.t, vehicle.current_time)

        pickup = task.pickup_node
        delivery = task.delivery_node

        route, distance_matrix = _build_dispatch_route(vehicle, task, state)
        executor = self._get_route_executor(distance_matrix)
        executed_route = executor.execute(route, vehicle, start_time=current_time)
        vehicle.assign_route(executed_route)

        start_node_id, start_coord = _infer_start_node(vehicle, state)
        edge_to_pickup = _make_edge(start_node_id, pickup.node_id, start_coord, pickup.coordinates, vehicle.speed)
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge_to_pickup], current_time)
        distance = executed_route.calculate_total_distance(distance_matrix)

        current_time = schedule[-1][1] if schedule else current_time
        conflict_waiting = conflict_wait
        _consume_energy(vehicle, [edge_to_pickup], self.energy_config)

        pickup_result = _visit_node(
            self.traffic_manager,
            robot_id,
            pickup.node_id,
            pickup.time_window,
            pickup.service_time,
            current_time,
            state,
        )
        current_time = pickup_result.end_time
        conflict_waiting += pickup_result.conflict_waiting
        delay = pickup_result.delay

        vehicle.current_load += task.demand
        if tracker and tracker.status == TaskStatus.ASSIGNED:
            tracker.start_execution(pickup_result.end_time)

        edge_to_delivery = _make_edge(
            pickup.node_id,
            delivery.node_id,
            pickup.coordinates,
            delivery.coordinates,
            vehicle.speed,
        )
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge_to_delivery], current_time)
        current_time = schedule[-1][1] if schedule else current_time
        conflict_waiting += conflict_wait
        if distance <= 0.0:
            distance += edge_to_delivery.distance
        _consume_energy(vehicle, [edge_to_delivery], self.energy_config)

        delivery_result = _visit_node(
            self.traffic_manager,
            robot_id,
            delivery.node_id,
            delivery.time_window,
            delivery.service_time,
            current_time,
            state,
        )
        current_time = delivery_result.end_time
        conflict_waiting += delivery_result.conflict_waiting
        delay += delivery_result.delay

        vehicle.current_load = max(0.0, vehicle.current_load - task.demand)
        vehicle.current_location = delivery.coordinates
        vehicle.current_time = current_time

        if tracker and tracker.status == TaskStatus.IN_PROGRESS:
            tracker.complete(current_time)

        state.metrics.total_distance += distance
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_delay += delay

        self.simulator.mark_vehicle_idle(robot_id, time=current_time)

        return ExecutionResult(
            end_time=current_time,
            conflict_waiting=conflict_waiting,
            distance=distance,
            delay=delay,
        )

    def _execute_charge(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        robot_id = int(action.payload.get("robot_id", -1))
        charger_id = int(action.payload.get("charger_id", -1))
        soc_target = float(action.payload.get("soc_target", 0.0))

        vehicle = _get_vehicle(state, robot_id)
        charger = state.chargers.get(charger_id)
        if vehicle is None or charger is None:
            return ExecutionResult(end_time=state.t)

        vehicle.status = VehicleStatus.CHARGING
        self.simulator.mark_vehicle_busy(robot_id)
        current_time = max(state.t, vehicle.current_time)
        start_node_id, start_coord = _infer_start_node(vehicle, state)

        route, distance_matrix = _build_charge_route(vehicle, charger.node_id, charger.coordinates)
        executor = self._get_route_executor(distance_matrix)
        executed_route = executor.execute(route, vehicle, start_time=current_time)
        vehicle.assign_route(executed_route)

        edge = _make_edge(start_node_id, charger.node_id, start_coord, charger.coordinates, vehicle.speed)
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge], current_time)
        distance = executed_route.calculate_total_distance(distance_matrix)
        current_time = schedule[-1][1] if schedule else current_time
        conflict_waiting = conflict_wait

        _consume_energy(vehicle, [edge], self.energy_config)

        queue_wait = max(0.0, charger.estimated_wait_s)
        earliest_charge = current_time + queue_wait

        target_battery = max(0.0, min(vehicle.battery_capacity, soc_target * vehicle.battery_capacity))
        charge_amount = max(0.0, target_battery - vehicle.current_battery)
        charge_time = 0.0
        if charge_amount > 0:
            charge_time = charge_amount / (
                self.energy_config.charging_rate * self.energy_config.charging_efficiency
            )

        charge_start, charge_end, node_wait, _ = self.traffic_manager.reserve_node_with_window(
            robot_id,
            charger.node_id,
            earliest_charge,
            charge_time,
            time_window=None,
        )

        conflict_waiting += node_wait
        standby = queue_wait

        vehicle.charge_battery(charge_amount, charge_end)
        vehicle.current_location = charger.coordinates

        state.metrics.total_distance += distance
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_charging += charge_amount
        state.metrics.total_standby += standby

        self.simulator.mark_vehicle_idle(robot_id, time=charge_end)

        return ExecutionResult(
            end_time=charge_end,
            conflict_waiting=conflict_waiting,
            distance=distance,
            charging=charge_amount,
            standby=standby,
        )

    def _execute_dwell(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        robot_id = int(action.payload.get("robot_id", -1))
        node_id = int(action.payload.get("node_id", -1))

        vehicle = _get_vehicle(state, robot_id)
        if vehicle is None:
            return ExecutionResult(end_time=state.t)

        vehicle.status = VehicleStatus.MOVING
        self.simulator.mark_vehicle_busy(robot_id)
        current_time = max(state.t, vehicle.current_time)

        if node_id < 0:
            dwell_start = current_time
            dwell_time = self.time_config.default_service_time
            dwell_end = dwell_start + dwell_time
            vehicle.current_time = dwell_end
            state.metrics.total_standby += dwell_time
            self.simulator.mark_vehicle_idle(robot_id, time=dwell_end)
            return ExecutionResult(end_time=dwell_end, standby=dwell_time)

        target_coord = _resolve_node_coordinates(state, node_id)
        if target_coord is None:
            target_coord = vehicle.current_location

        edge = _make_edge(
            _infer_start_node(vehicle, state)[0],
            node_id,
            vehicle.current_location,
            target_coord,
            vehicle.speed,
        )
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge], current_time)
        distance = edge.distance
        current_time = schedule[-1][1] if schedule else current_time

        _consume_energy(vehicle, [edge], self.energy_config)

        dwell_time = self.time_config.default_service_time
        dwell_start, dwell_end, node_wait, _ = self.traffic_manager.reserve_node_with_window(
            robot_id,
            node_id,
            current_time,
            dwell_time,
            time_window=None,
        )
        conflict_waiting = conflict_wait + node_wait

        vehicle.current_location = target_coord
        vehicle.current_time = dwell_end

        state.metrics.total_distance += distance
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_standby += dwell_time

        self.simulator.mark_vehicle_idle(robot_id, time=dwell_end)

        return ExecutionResult(
            end_time=dwell_end,
            conflict_waiting=conflict_waiting,
            distance=distance,
            standby=dwell_time,
        )

    def _get_route_executor(self, distance_matrix: DistanceMatrix) -> RouteExecutor:
        if self._route_executor is None:
            self._route_executor = RouteExecutor(
                distance_matrix=distance_matrix,
                energy_config=self.energy_config,
                time_config=self.time_config,
            )
            return self._route_executor
        self._route_executor.distance = distance_matrix
        return self._route_executor


def _get_vehicle(state: SimulatorState, robot_id: int) -> Optional[Vehicle]:
    return state.robots.get(robot_id)


def _infer_start_node(vehicle: Vehicle, state: SimulatorState) -> Tuple[int, Tuple[float, float]]:
    nodes = _collect_known_nodes(state)
    if not nodes:
        return 0, vehicle.current_location
    best = None
    for node_id, coord in nodes.items():
        dist = _euclidean(vehicle.current_location, coord)
        if best is None or dist < best[0]:
            best = (dist, node_id, coord)
    if best is None:
        return 0, vehicle.current_location
    return best[1], best[2]


def _collect_known_nodes(state: SimulatorState) -> Dict[int, Tuple[float, float]]:
    nodes: Dict[int, Tuple[float, float]] = {0: (0.0, 0.0)}
    for task in state.open_tasks.values():
        nodes[task.pickup_node.node_id] = task.pickup_node.coordinates
        nodes[task.delivery_node.node_id] = task.delivery_node.coordinates
    for charger in state.chargers.values():
        nodes[charger.node_id] = charger.coordinates
    return nodes


def _resolve_node_coordinates(state: SimulatorState, node_id: int) -> Optional[Tuple[float, float]]:
    for task in state.open_tasks.values():
        if task.pickup_node.node_id == node_id:
            return task.pickup_node.coordinates
        if task.delivery_node.node_id == node_id:
            return task.delivery_node.coordinates
    charger = state.chargers.get(node_id)
    if charger:
        return charger.coordinates
    if node_id == 0:
        return (0.0, 0.0)
    return None


def _make_edge(
    from_id: int,
    to_id: int,
    from_coord: Tuple[float, float],
    to_coord: Tuple[float, float],
    speed: float,
) -> PathEdge:
    distance = _euclidean(from_coord, to_coord)
    travel_time = calculate_travel_time(distance, max(speed, 1e-6))
    return PathEdge(from_node=from_id, to_node=to_id, travel_time=travel_time, distance=distance)


def _consume_energy(vehicle: Vehicle, edges: Sequence[PathEdge], energy_config: EnergyConfig) -> None:
    for edge in edges:
        required = calculate_energy_consumption(
            distance=edge.distance,
            load=vehicle.current_load,
            config=energy_config,
            vehicle_speed=vehicle.speed,
            vehicle_capacity=vehicle.capacity,
        )
        try:
            vehicle.consume_battery(required)
        except ValueError:
            vehicle.current_battery = max(0.0, vehicle.current_battery - required)


def _visit_node(
    traffic_manager: TrafficManager,
    robot_id: int,
    node_id: int,
    time_window: Optional[TimeWindow],
    service_time: float,
    arrival_time: float,
    state: SimulatorState,
) -> ExecutionResult:
    start_service, end_service, node_wait, _ = traffic_manager.reserve_node_with_window(
        robot_id,
        node_id,
        arrival_time,
        service_time,
        time_window=time_window,
    )

    delay = 0.0
    if time_window is not None and time_window.latest is not None:
        delay = max(0.0, start_service - time_window.latest)

    return ExecutionResult(
        end_time=end_service,
        conflict_waiting=node_wait,
        delay=delay,
    )


def _euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return (dx * dx + dy * dy) ** 0.5


def _build_dispatch_route(vehicle: Vehicle, task, state: SimulatorState) -> Tuple[Route, DistanceMatrix]:
    start_id = _virtual_node_id(vehicle.vehicle_id, 0)
    end_id = _virtual_node_id(vehicle.vehicle_id, 1)
    start_node = Node(node_id=start_id, coordinates=vehicle.current_location, node_type=NodeType.DEPOT)
    end_node = Node(node_id=end_id, coordinates=task.delivery_coordinates, node_type=NodeType.DEPOT)
    route = Route(
        vehicle_id=vehicle.vehicle_id,
        nodes=[start_node, task.pickup_node, task.delivery_node, end_node],
    )
    distance_matrix = _build_distance_matrix([start_node, task.pickup_node, task.delivery_node, end_node], state)
    return route, distance_matrix


def _build_charge_route(
    vehicle: Vehicle,
    charger_id: int,
    charger_coords: Tuple[float, float],
) -> Tuple[Route, DistanceMatrix]:
    start_id = _virtual_node_id(vehicle.vehicle_id, 0)
    end_id = _virtual_node_id(vehicle.vehicle_id, 1)
    start_node = Node(node_id=start_id, coordinates=vehicle.current_location, node_type=NodeType.DEPOT)
    charger_node = Node(node_id=charger_id, coordinates=charger_coords, node_type=NodeType.CHARGING)
    end_node = Node(node_id=end_id, coordinates=charger_coords, node_type=NodeType.DEPOT)
    route = Route(vehicle_id=vehicle.vehicle_id, nodes=[start_node, charger_node, end_node])
    distance_matrix = _build_distance_matrix([start_node, charger_node, end_node], None)
    return route, distance_matrix


def _build_distance_matrix(nodes: Sequence[Node], state: Optional[SimulatorState]) -> DistanceMatrix:
    coordinates = {node.node_id: node.coordinates for node in nodes}
    if state is not None:
        for task in state.open_tasks.values():
            coordinates.setdefault(task.pickup_node.node_id, task.pickup_node.coordinates)
            coordinates.setdefault(task.delivery_node.node_id, task.delivery_node.coordinates)
        for charger in state.chargers.values():
            coordinates.setdefault(charger.node_id, charger.coordinates)
    num_tasks = len(state.open_tasks) if state is not None else 0
    num_charging = len(state.chargers) if state is not None else 0
    return DistanceMatrix(coordinates=coordinates, num_tasks=num_tasks, num_charging_stations=num_charging)


def _virtual_node_id(vehicle_id: int, offset: int) -> int:
    return 10_000_000 + vehicle_id * 10 + offset


__all__ = ["ExecutionLayer", "ExecutionResult"]
