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
    travel_time: float = 0.0
    charging: float = 0.0
    delay: float = 0.0
    waiting: float = 0.0
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
        travel_time_factor: Optional[float] = None,
        wait_weight_default: float = 1.0,
        wait_weight_charging: float = 3.0,
        wait_weight_depot: float = 0.5,
        wait_weight_scale: float = 1.0,
        min_soc_threshold: Optional[float] = None,
    ) -> None:
        self.task_pool = task_pool
        self.simulator = simulator
        self.traffic_manager = traffic_manager
        self.energy_config = energy_config or EnergyConfig()
        self.time_config = time_config or TimeConfig()
        if travel_time_factor is None:
            travel_time_factor = getattr(simulator, "travel_time_factor", 1.0)
        self.travel_time_factor = max(0.0, travel_time_factor)
        scale = max(0.0, wait_weight_scale)
        self.wait_weight_default = max(0.0, wait_weight_default) * scale
        self.wait_weight_charging = max(0.0, wait_weight_charging) * scale
        self.wait_weight_depot = max(0.0, wait_weight_depot) * scale
        if min_soc_threshold is None:
            min_soc_threshold = self.energy_config.safety_threshold
        self.min_soc_threshold = max(0.0, min_soc_threshold)
        self._route_executor: Optional[RouteExecutor] = None

    def execute(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        """Execute a single atomic action and update simulator/task state."""
        self.travel_time_factor = max(
            0.0, getattr(self.simulator, "travel_time_factor", self.travel_time_factor)
        )

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
        mode = str(action.payload.get("mode", "dispatch"))
        task = self.task_pool.get_task(task_id)
        vehicle = _get_vehicle(state, robot_id)

        if task is None or vehicle is None:
            return ExecutionResult(end_time=state.t)
        if task.demand > vehicle.capacity + 1e-6:
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

        tracker = self.task_pool.get_tracker(task_id)
        if tracker and tracker.status in (TaskStatus.REJECTED, TaskStatus.COMPLETED):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)
        if not _dispatch_energy_feasible(vehicle, task, self.energy_config, self.min_soc_threshold):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)
        if mode == "insert" and vehicle.route is not None:
            selection = _select_best_insert_route(
                vehicle,
                task,
                state,
                self.time_config,
                self.energy_config,
                self.traffic_manager,
                self.travel_time_factor,
                self.min_soc_threshold,
            )
            if selection is None:
                state.metrics.infeasible_actions += 1
                return ExecutionResult(end_time=state.t)
            route, distance_matrix = selection
        if tracker and tracker.status == TaskStatus.PENDING:
            self.task_pool.assign_task(task_id, robot_id)
            self.simulator.mark_task_assigned(task_id, robot_id)
        else:
            self.simulator.mark_vehicle_busy(robot_id)

        vehicle.status = VehicleStatus.MOVING
        current_time = max(state.t, vehicle.current_time)

        pickup = task.pickup_node
        delivery = task.delivery_node
        if mode == "insert" and vehicle.route is not None:
            result = self._execute_full_route(
                vehicle,
                route,
                distance_matrix,
                state,
                current_time,
            )
            return result

        route, distance_matrix = _build_dispatch_route(vehicle, task, state)
        executor = self._get_route_executor(distance_matrix)
        executed_route = executor.execute(route, vehicle, start_time=current_time)
        vehicle.assign_route(executed_route)

        start_node_id, start_coord = _infer_start_node(vehicle, state)
        edge_to_pickup = _make_edge(
            start_node_id,
            pickup.node_id,
            start_coord,
            pickup.coordinates,
            vehicle.speed,
            self.travel_time_factor,
        )
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge_to_pickup], current_time)
        travel_time = sum(end - start for start, end, _ in schedule)
        distance = executed_route.calculate_total_distance(distance_matrix)

        current_time = schedule[-1][1] if schedule else current_time
        conflict_waiting = conflict_wait
        waiting_weighted = 0.0
        if not _consume_energy(
            vehicle,
            [edge_to_pickup],
            self.energy_config,
            self.travel_time_factor,
            min_soc_threshold=self.min_soc_threshold,
        ):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

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
        waiting = pickup_result.waiting
        waiting_weighted += pickup_result.waiting * self._waiting_weight(state, pickup.node_id)

        vehicle.current_load += task.demand
        if tracker and tracker.status == TaskStatus.ASSIGNED:
            tracker.start_execution(pickup_result.end_time)

        edge_to_delivery = _make_edge(
            pickup.node_id,
            delivery.node_id,
            pickup.coordinates,
            delivery.coordinates,
            vehicle.speed,
            self.travel_time_factor,
        )
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge_to_delivery], current_time)
        travel_time += sum(end - start for start, end, _ in schedule)
        current_time = schedule[-1][1] if schedule else current_time
        conflict_waiting += conflict_wait
        if distance <= 0.0:
            distance += edge_to_delivery.distance
        if not _consume_energy(
            vehicle,
            [edge_to_delivery],
            self.energy_config,
            self.travel_time_factor,
            min_soc_threshold=self.min_soc_threshold,
        ):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

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
        waiting += delivery_result.waiting
        waiting_weighted += delivery_result.waiting * self._waiting_weight(state, delivery.node_id)

        vehicle.current_load = max(0.0, vehicle.current_load - task.demand)
        vehicle.current_location = delivery.coordinates
        vehicle.current_time = current_time

        if tracker and tracker.status == TaskStatus.IN_PROGRESS:
            tracker.complete(current_time)

        state.metrics.total_distance += distance
        state.metrics.total_travel_time += travel_time
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_delay += delay
        state.metrics.total_waiting += waiting
        state.metrics.total_waiting_weighted += waiting_weighted

        self.simulator.update_soc_status(robot_id, time=current_time)
        self.simulator.maybe_trigger_deadlock(
            wait_s=conflict_waiting,
            time=current_time,
            payload={"robot_id": robot_id, "conflict_waiting": conflict_waiting},
        )
        self.simulator.mark_vehicle_idle(robot_id, time=current_time)

        return ExecutionResult(
            end_time=current_time,
            conflict_waiting=conflict_waiting,
            distance=distance,
            travel_time=travel_time,
            delay=delay,
            waiting=waiting,
        )

    def _execute_full_route(
        self,
        vehicle: Vehicle,
        route: Route,
        distance_matrix: DistanceMatrix,
        state: SimulatorState,
        start_time: float,
    ) -> ExecutionResult:
        robot_id = vehicle.vehicle_id
        executor = self._get_route_executor(distance_matrix)
        executed_route = executor.execute(route, vehicle, start_time=0.0)
        vehicle.assign_route(executed_route)

        conflict_waiting = 0.0
        distance = 0.0
        travel_time = 0.0
        delay = 0.0
        waiting = 0.0
        standby = 0.0
        charging = 0.0
        waiting_weighted = 0.0

        vehicle.status = VehicleStatus.MOVING
        self.simulator.mark_vehicle_busy(robot_id)
        current_time = start_time

        if not executed_route.nodes:
            return ExecutionResult(end_time=current_time)

        prev_node = executed_route.nodes[0]
        prev_coord = prev_node.coordinates
        prev_node_id = prev_node.node_id

        for idx in range(1, len(executed_route.nodes)):
            node = executed_route.nodes[idx]
            edge = _make_edge(
                prev_node_id,
                node.node_id,
                prev_coord,
                node.coordinates,
                vehicle.speed,
                self.travel_time_factor,
            )
            schedule, edge_conflict = self.traffic_manager.reserve_path(
                robot_id, [edge], current_time
            )
            travel_time += sum(end - start for start, end, _ in schedule)
            distance += edge.distance
            current_time = schedule[-1][1] if schedule else current_time
            conflict_waiting += edge_conflict
            if not _consume_energy(
                vehicle,
                [edge],
                self.energy_config,
                self.travel_time_factor,
                min_soc_threshold=self.min_soc_threshold,
            ):
                state.metrics.infeasible_actions += 1
                return ExecutionResult(end_time=state.t)
            self.simulator.advance_until(current_time)

            if node.is_charging_station():
                charger = state.chargers.get(node.node_id)
                if charger is None or not charger.is_available:
                    state.metrics.infeasible_actions += 1
                    return ExecutionResult(end_time=state.t)
                queue_wait = max(0.0, charger.estimated_wait_s)
                earliest_charge = current_time + queue_wait
                charge_amount = 0.0
                charge_time = 0.0
                if executed_route.visits and idx < len(executed_route.visits):
                    visit = executed_route.visits[idx]
                    charge_amount = max(0.0, visit.battery_after_service - visit.battery_after_travel)
                    charge_time = max(0.0, visit.departure_time - visit.start_service_time)
                charge_start, charge_end, node_wait, _ = self.traffic_manager.reserve_node_with_window(
                    robot_id,
                    node.node_id,
                    earliest_charge,
                    charge_time,
                    time_window=None,
                )
                conflict_waiting += node_wait
                waiting += queue_wait
                waiting_weighted += queue_wait * self.wait_weight_charging
                vehicle.charge_battery(charge_amount, charge_end)
                charging += charge_amount
                current_time = charge_end
                self.simulator.advance_until(current_time)
            else:
                time_window = getattr(node, "time_window", None)
                service_time = getattr(node, "service_time", 0.0)
                result = _visit_node(
                    self.traffic_manager,
                    robot_id,
                    node.node_id,
                    time_window,
                    service_time,
                    current_time,
                    state,
                )
                current_time = result.end_time
                conflict_waiting += result.conflict_waiting
                delay += result.delay
                waiting += result.waiting
                waiting_weighted += result.waiting * self._waiting_weight(state, node.node_id)
                self.simulator.advance_until(current_time)

                if node.is_pickup():
                    vehicle.current_load += getattr(node, "demand", 0.0)
                    tracker = self.task_pool.get_tracker(node.task_id) if hasattr(node, "task_id") else None
                    if (
                        tracker
                        and tracker.status == TaskStatus.ASSIGNED
                        and tracker.assigned_vehicle_id == robot_id
                    ):
                        tracker.start_execution(result.end_time)
                elif node.is_delivery():
                    vehicle.current_load = max(0.0, vehicle.current_load - getattr(node, "demand", 0.0))
                    tracker = self.task_pool.get_tracker(node.task_id) if hasattr(node, "task_id") else None
                    if (
                        tracker
                        and tracker.status == TaskStatus.IN_PROGRESS
                        and tracker.assigned_vehicle_id == robot_id
                    ):
                        tracker.complete(result.end_time)

            if executed_route.visits and idx < len(executed_route.visits):
                standby_time = max(0.0, executed_route.visits[idx].standby_time)
                if standby_time > 0.0:
                    standby += standby_time
                    current_time += standby_time
                    self.simulator.advance_until(current_time)

            vehicle.current_location = node.coordinates
            vehicle.current_time = current_time
            prev_node_id = node.node_id
            prev_coord = node.coordinates

        state.metrics.total_distance += distance
        state.metrics.total_travel_time += travel_time
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_delay += delay
        state.metrics.total_waiting += waiting
        state.metrics.total_waiting_weighted += waiting_weighted
        state.metrics.total_standby += standby
        state.metrics.total_charging += charging

        self.simulator.update_soc_status(robot_id, time=current_time)
        self.simulator.maybe_trigger_deadlock(
            wait_s=conflict_waiting,
            time=current_time,
            payload={"robot_id": robot_id, "conflict_waiting": conflict_waiting},
        )
        self.simulator.mark_vehicle_idle(robot_id, time=current_time)

        return ExecutionResult(
            end_time=current_time,
            conflict_waiting=conflict_waiting,
            distance=distance,
            travel_time=travel_time,
            charging=charging,
            delay=delay,
            waiting=waiting,
            standby=standby,
        )

    def _execute_charge(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        robot_id = int(action.payload.get("robot_id", -1))
        charger_id = int(action.payload.get("charger_id", -1))
        soc_target = float(action.payload.get("soc_target", 0.0))

        vehicle = _get_vehicle(state, robot_id)
        charger = state.chargers.get(charger_id)
        if vehicle is None or charger is None:
            return ExecutionResult(end_time=state.t)
        if not _travel_energy_feasible(
            vehicle,
            charger.coordinates,
            vehicle.current_load,
            self.energy_config,
            self.min_soc_threshold,
        ):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

        vehicle.status = VehicleStatus.CHARGING
        self.simulator.mark_vehicle_busy(robot_id)
        current_time = max(state.t, vehicle.current_time)
        start_node_id, start_coord = _infer_start_node(vehicle, state)

        route, distance_matrix = _build_charge_route(vehicle, charger.node_id, charger.coordinates)
        executor = self._get_route_executor(distance_matrix)
        executed_route = executor.execute(route, vehicle, start_time=current_time)
        vehicle.assign_route(executed_route)

        edge = _make_edge(
            start_node_id,
            charger.node_id,
            start_coord,
            charger.coordinates,
            vehicle.speed,
            self.travel_time_factor,
        )
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge], current_time)
        distance = executed_route.calculate_total_distance(distance_matrix)
        current_time = schedule[-1][1] if schedule else current_time
        conflict_waiting = conflict_wait
        travel_time = sum(end - start for start, end, _ in schedule)

        if not _consume_energy(
            vehicle,
            [edge],
            self.energy_config,
            self.travel_time_factor,
            min_soc_threshold=self.min_soc_threshold,
        ):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

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
        standby = 0.0
        waiting = queue_wait
        waiting_weighted = queue_wait * self.wait_weight_charging

        vehicle.charge_battery(charge_amount, charge_end)
        vehicle.current_location = charger.coordinates

        state.metrics.total_distance += distance
        state.metrics.total_travel_time += travel_time
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_charging += charge_amount
        state.metrics.total_standby += standby
        state.metrics.total_waiting += waiting
        state.metrics.total_waiting_weighted += waiting_weighted

        self.simulator.update_soc_status(robot_id, time=charge_end)
        self.simulator.maybe_trigger_deadlock(
            wait_s=conflict_waiting,
            time=charge_end,
            payload={"robot_id": robot_id, "conflict_waiting": conflict_waiting},
        )
        self.simulator.mark_charge_done(robot_id, time=charge_end)

        return ExecutionResult(
            end_time=charge_end,
            conflict_waiting=conflict_waiting,
            distance=distance,
            travel_time=travel_time,
            charging=charge_amount,
            waiting=waiting,
            standby=standby,
        )

    def _execute_dwell(self, action: AtomicAction, state: SimulatorState, event: Optional[Event]) -> ExecutionResult:
        robot_id = int(action.payload.get("robot_id", -1))
        node_id = int(action.payload.get("node_id", -1))

        vehicle = _get_vehicle(state, robot_id)
        if vehicle is None:
            return ExecutionResult(end_time=state.t)
        if node_id in state.chargers:
            state.metrics.infeasible_actions += 1
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
            self.simulator.update_soc_status(robot_id, time=dwell_end)
            self.simulator.maybe_trigger_deadlock(
                wait_s=0.0,
                time=dwell_end,
                payload={"robot_id": robot_id, "conflict_waiting": 0.0},
            )
            self.simulator.mark_vehicle_idle(robot_id, time=dwell_end)
            return ExecutionResult(end_time=dwell_end, standby=dwell_time)

        target_coord = _resolve_node_coordinates(state, node_id)
        if target_coord is None:
            target_coord = vehicle.current_location
        if not _travel_energy_feasible(
            vehicle,
            target_coord,
            vehicle.current_load,
            self.energy_config,
            self.min_soc_threshold,
        ):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

        edge = _make_edge(
            _infer_start_node(vehicle, state)[0],
            node_id,
            vehicle.current_location,
            target_coord,
            vehicle.speed,
            self.travel_time_factor,
        )
        schedule, conflict_wait = self.traffic_manager.reserve_path(robot_id, [edge], current_time)
        distance = edge.distance
        current_time = schedule[-1][1] if schedule else current_time
        travel_time = sum(end - start for start, end, _ in schedule)

        if not _consume_energy(
            vehicle,
            [edge],
            self.energy_config,
            self.travel_time_factor,
            min_soc_threshold=self.min_soc_threshold,
        ):
            state.metrics.infeasible_actions += 1
            return ExecutionResult(end_time=state.t)

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
        state.metrics.total_travel_time += travel_time
        state.metrics.total_conflict_waiting += conflict_waiting
        state.metrics.total_standby += dwell_time

        self.simulator.update_soc_status(robot_id, time=dwell_end)
        self.simulator.maybe_trigger_deadlock(
            wait_s=conflict_waiting,
            time=dwell_end,
            payload={"robot_id": robot_id, "conflict_waiting": conflict_waiting},
        )
        self.simulator.mark_vehicle_idle(robot_id, time=dwell_end)

        return ExecutionResult(
            end_time=dwell_end,
            conflict_waiting=conflict_waiting,
            distance=distance,
            travel_time=travel_time,
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

    def _waiting_weight(self, state: SimulatorState, node_id: int) -> float:
        if node_id in state.chargers:
            return self.wait_weight_charging
        if node_id == 0:
            return self.wait_weight_depot
        return self.wait_weight_default


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
    travel_time_factor: float = 1.0,
) -> PathEdge:
    distance = _euclidean(from_coord, to_coord)
    travel_time = calculate_travel_time(distance, max(speed, 1e-6)) * max(0.0, travel_time_factor)
    return PathEdge(from_node=from_id, to_node=to_id, travel_time=travel_time, distance=distance)


def _consume_energy(
    vehicle: Vehicle,
    edges: Sequence[PathEdge],
    energy_config: EnergyConfig,
    travel_time_factor: float = 1.0,
    *,
    min_soc_threshold: float = 0.0,
) -> bool:
    for edge in edges:
        required = calculate_energy_consumption(
            distance=edge.distance,
            load=vehicle.current_load,
            config=energy_config,
            vehicle_speed=vehicle.speed,
            vehicle_capacity=vehicle.capacity,
        ) * max(0.0, travel_time_factor)
        min_battery = max(0.0, min_soc_threshold) * vehicle.battery_capacity
        if vehicle.current_battery - required < min_battery - 1e-6:
            return False
        vehicle.consume_battery(required)
    return True


def _visit_node(
    traffic_manager: TrafficManager,
    robot_id: int,
    node_id: int,
    time_window: Optional[TimeWindow],
    service_time: float,
    arrival_time: float,
    state: SimulatorState,
) -> ExecutionResult:
    start_service, end_service, node_wait, waiting = traffic_manager.reserve_node_with_window(
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
        waiting=waiting,
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


def _build_insert_base_nodes(
    vehicle: Vehicle,
) -> List[Node]:
    start_id = _virtual_node_id(vehicle.vehicle_id, 0)
    start_node = Node(node_id=start_id, coordinates=vehicle.current_location, node_type=NodeType.DEPOT)
    nodes = [start_node]
    if vehicle.route and vehicle.route.nodes:
        tail = list(vehicle.route.nodes)
        if tail and tail[0].is_depot():
            tail = tail[1:]
        nodes.extend(tail)
    if not nodes[-1].is_depot():
        end_id = _virtual_node_id(vehicle.vehicle_id, 1)
        end_coord = nodes[-1].coordinates
        nodes.append(Node(node_id=end_id, coordinates=end_coord, node_type=NodeType.DEPOT))
    return nodes


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


def _travel_energy_feasible(
    vehicle: Vehicle,
    target: Tuple[float, float],
    load: float,
    energy_config: EnergyConfig,
    min_soc_threshold: float,
) -> bool:
    distance = _euclidean(vehicle.current_location, target)
    required = calculate_energy_consumption(
        distance=distance,
        load=load,
        config=energy_config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    min_battery = max(0.0, min_soc_threshold) * vehicle.battery_capacity
    return vehicle.current_battery - required >= min_battery - 1e-6


def _dispatch_energy_feasible(
    vehicle: Vehicle,
    task,
    energy_config: EnergyConfig,
    min_soc_threshold: float,
) -> bool:
    pickup = task.pickup_node
    delivery = task.delivery_node
    distance_to_pickup = _euclidean(vehicle.current_location, pickup.coordinates)
    required_pickup = calculate_energy_consumption(
        distance=distance_to_pickup,
        load=vehicle.current_load,
        config=energy_config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    min_battery = max(0.0, min_soc_threshold) * vehicle.battery_capacity
    battery_after_pickup = vehicle.current_battery - required_pickup
    if battery_after_pickup < min_battery - 1e-6:
        return False
    distance_to_delivery = _euclidean(pickup.coordinates, delivery.coordinates)
    required_delivery = calculate_energy_consumption(
        distance=distance_to_delivery,
        load=vehicle.current_load + task.demand,
        config=energy_config,
        vehicle_speed=vehicle.speed,
        vehicle_capacity=vehicle.capacity,
    )
    battery_after_delivery = battery_after_pickup - required_delivery
    if battery_after_delivery < min_battery - 1e-6:
        return False
    return True


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


def _select_best_insert_route(
    vehicle: Vehicle,
    task,
    state: SimulatorState,
    time_config: TimeConfig,
    energy_config: EnergyConfig,
    traffic_manager: TrafficManager,
    travel_time_factor: float,
    min_soc_threshold: float,
) -> Optional[Tuple[Route, DistanceMatrix]]:
    base_nodes = _build_insert_base_nodes(vehicle)
    if len(base_nodes) < 2:
        return None

    end_index = len(base_nodes) - 1 if base_nodes[-1].is_depot() else len(base_nodes)
    best = None
    for pickup_pos in range(1, end_index + 1):
        for delivery_pos in range(pickup_pos + 1, end_index + 2):
            nodes = list(base_nodes)
            nodes.insert(pickup_pos, task.pickup_node)
            nodes.insert(delivery_pos, task.delivery_node)
            candidate = _evaluate_insert_candidate(
                nodes,
                vehicle,
                state,
                time_config,
                energy_config,
                traffic_manager,
                travel_time_factor,
                min_soc_threshold,
            )
            if candidate is None:
                continue
            score, route, distance_matrix = candidate
            if best is None or score < best[0]:
                best = (score, route, distance_matrix)
    if best is None:
        return None
    return best[1], best[2]


def _evaluate_insert_candidate(
    nodes: Sequence[Node],
    vehicle: Vehicle,
    state: SimulatorState,
    time_config: TimeConfig,
    energy_config: EnergyConfig,
    traffic_manager: TrafficManager,
    travel_time_factor: float,
    min_soc_threshold: float,
) -> Optional[Tuple[float, Route, DistanceMatrix]]:
    route = Route(vehicle_id=vehicle.vehicle_id, nodes=list(nodes))
    distance_matrix = _build_distance_matrix(nodes, state)
    charging_availability = {node_id: int(ch.is_available) for node_id, ch in state.chargers.items()}
    zero_waits = [0.0] * len(nodes)
    if not route.compute_schedule(
        distance_matrix,
        vehicle_capacity=vehicle.capacity,
        vehicle_battery_capacity=vehicle.battery_capacity,
        initial_battery=vehicle.current_battery,
        time_config=time_config,
        energy_config=energy_config,
        conflict_waiting_times=zero_waits,
        standby_times=zero_waits,
        min_soc_threshold=min_soc_threshold,
        charging_availability=charging_availability,
    ):
        return None

    conflict_waits = _preview_conflict_waits(
        traffic_manager,
        route,
        distance_matrix,
        vehicle.speed,
        max(state.t, vehicle.current_time),
        travel_time_factor,
    )
    if not route.compute_schedule(
        distance_matrix,
        vehicle_capacity=vehicle.capacity,
        vehicle_battery_capacity=vehicle.battery_capacity,
        initial_battery=vehicle.current_battery,
        time_config=time_config,
        energy_config=energy_config,
        conflict_waiting_times=conflict_waits,
        standby_times=zero_waits,
        min_soc_threshold=min_soc_threshold,
        charging_availability=charging_availability,
    ):
        return None

    total_distance = route.calculate_total_distance(distance_matrix)
    total_delay = route.calculate_total_delay()
    score = total_distance + 0.001 * total_delay
    return score, route, distance_matrix


def _preview_conflict_waits(
    traffic_manager: TrafficManager,
    route: Route,
    distance_matrix: DistanceMatrix,
    speed: float,
    earliest_start: float,
    travel_time_factor: float,
) -> List[float]:
    if not route.nodes:
        return []
    waits = [0.0] * len(route.nodes)
    edges: List[PathEdge] = []
    for idx in range(1, len(route.nodes)):
        prev_node = route.nodes[idx - 1]
        node = route.nodes[idx]
        dist = distance_matrix.get_distance(prev_node.node_id, node.node_id)
        travel_time = calculate_travel_time(dist, max(speed, 1e-6)) * max(0.0, travel_time_factor)
        edges.append(PathEdge(from_node=prev_node.node_id, to_node=node.node_id, travel_time=travel_time, distance=dist))
    schedule, _ = traffic_manager.preview_reserve_path(edges, earliest_start)
    for idx, (_, __, wait) in enumerate(schedule, start=1):
        waits[idx] = wait
    if route.visits:
        for idx, visit in enumerate(route.visits):
            node_wait = traffic_manager.preview_reserve_node(
                visit.node.node_id,
                visit.arrival_time + waits[idx],
                visit.get_service_time(),
            )[2]
            waits[idx] += node_wait
    return waits


__all__ = ["ExecutionLayer", "ExecutionResult"]
