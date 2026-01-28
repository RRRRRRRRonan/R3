"""OR-Tools implementation for the MIP baseline solver."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Tuple

from config import CostParameters
from coordinator.conflict import HeadwayPolicy

from baselines.mip.config import MIPBaselineScale, MIPBaselineSolverConfig
from baselines.mip.model import (
    MIPBaselineInstance,
    MIPBaselineResult,
    MIPBaselineScenario,
    build_minimal_instance,
)

RULE_STTF = 1
RULE_EDD = 2
RULE_MST = 3
RULE_HPF = 4
RULE_CHARGE_URGENT = 5
RULE_CHARGE_TARGET = 6
RULE_CHARGE_OPPORTUNITY = 7
RULE_STANDBY_LOW_COST = 8
RULE_STANDBY_LAZY = 9
RULE_STANDBY_HEATMAP = 10
RULE_ACCEPT_FEASIBLE = 11
RULE_ACCEPT_VALUE = 12
RULE_INSERT_MIN_COST = 13

DISPATCH_RULES = {
    RULE_STTF,
    RULE_EDD,
    RULE_MST,
    RULE_HPF,
    RULE_INSERT_MIN_COST,
}
ACCEPT_RULES = {RULE_ACCEPT_FEASIBLE, RULE_ACCEPT_VALUE}
CHARGE_RULES = {
    RULE_CHARGE_URGENT,
    RULE_CHARGE_TARGET,
    RULE_CHARGE_OPPORTUNITY,
}
STANDBY_RULES = {RULE_STANDBY_LOW_COST, RULE_STANDBY_LAZY, RULE_STANDBY_HEATMAP}


@dataclass(frozen=True)
class RulePreferences:
    accept_indicator: Dict[int, int] = field(default_factory=dict)
    candidate_vehicles: Dict[int, List[int]] = field(default_factory=dict)
    preferred_tasks: List[int] = field(default_factory=list)
    preferred_charging_stations: Dict[int, List[int]] = field(default_factory=dict)
    force_charge: Dict[int, int] = field(default_factory=dict)
    charge_level_ratios: Optional[Tuple[float, ...]] = None
    min_charge_ratio: Optional[float] = None
    preferred_standby_nodes: Dict[int, List[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class MIPBaselineSolver:
    """Base class for baseline MIP solvers."""

    solver_config: MIPBaselineSolverConfig

    def solve(
        self,
        instance: MIPBaselineInstance,
        *,
        cost_params: Optional[CostParameters] = None,
    ) -> MIPBaselineResult:
        raise NotImplementedError


class ORToolsSolver(MIPBaselineSolver):
    """Solve the MIP baseline with OR-Tools MIP backend."""

    def __init__(self, solver_config: Optional[MIPBaselineSolverConfig] = None) -> None:
        super().__init__(solver_config or MIPBaselineSolverConfig())

    def solve(
        self,
        instance: MIPBaselineInstance,
        *,
        cost_params: Optional[CostParameters] = None,
    ) -> MIPBaselineResult:
        try:
            from ortools.linear_solver import pywraplp
        except ImportError as exc:  # pragma: no cover - runtime environment check
            raise RuntimeError(
                "OR-Tools is not installed. Run `python3 -m pip install ortools` first."
            ) from exc

        scale = MIPBaselineScale()
        instance.validate_scale(scale)

        cost_params = cost_params or config.cost_params or CostParameters()
        config = self.solver_config

        vehicles = list(instance.vehicles)
        vehicle_ids = [vehicle.vehicle_id for vehicle in vehicles]
        vehicle_by_id = {vehicle.vehicle_id: vehicle for vehicle in vehicles}

        scenarios = _ensure_scenarios(instance)
        scenario_ids = [scenario.scenario_id for scenario in scenarios]
        scenario_prob = {scenario.scenario_id: scenario.probability for scenario in scenarios}

        node_coords: Dict[int, Tuple[float, float]] = {
            instance.depot.node_id: instance.depot.coordinates
        }
        service_time: Dict[int, float] = {instance.depot.node_id: 0.0}
        demand_delta: Dict[int, float] = {instance.depot.node_id: 0.0}
        is_charging: Dict[int, bool] = {instance.depot.node_id: False}
        time_windows: Dict[int, Optional[Tuple[float, float]]] = {
            instance.depot.node_id: None
        }

        tasks = list(instance.tasks)
        task_pairs: Dict[int, Tuple[int, int]] = {}
        pickup_nodes: Dict[int, int] = {}
        delivery_nodes: Dict[int, int] = {}
        for task in tasks:
            pickup = task.pickup_node
            delivery = task.delivery_node
            task_pairs[task.task_id] = (pickup.node_id, delivery.node_id)
            pickup_nodes[task.task_id] = pickup.node_id
            delivery_nodes[task.task_id] = delivery.node_id
            for node in (pickup, delivery):
                node_coords[node.node_id] = node.coordinates
                service_time[node.node_id] = node.service_time
                if node.is_pickup():
                    demand_delta[node.node_id] = node.demand
                elif node.is_delivery():
                    demand_delta[node.node_id] = -node.demand
                else:
                    demand_delta[node.node_id] = 0.0
                is_charging[node.node_id] = node.is_charging_station()
                if node.time_window:
                    time_windows[node.node_id] = (
                        node.time_window.earliest,
                        node.time_window.latest,
                    )
                else:
                    time_windows[node.node_id] = None

        for station in instance.charging_stations:
            node_coords[station.node_id] = station.coordinates
            service_time[station.node_id] = 0.0
            demand_delta[station.node_id] = 0.0
            is_charging[station.node_id] = True
            time_windows[station.node_id] = None

        start_depot_id = instance.depot.node_id
        end_depot_id = max(node_coords.keys()) + 1
        node_coords[end_depot_id] = instance.depot.coordinates
        service_time[end_depot_id] = 0.0
        demand_delta[end_depot_id] = 0.0
        is_charging[end_depot_id] = False
        time_windows[end_depot_id] = None

        node_ids = sorted(node_coords.keys())
        internal_nodes = [
            node_id
            for node_id in node_ids
            if node_id not in (start_depot_id, end_depot_id)
        ]
        charging_nodes = [node_id for node_id in node_ids if is_charging.get(node_id, False)]

        distance_func = getattr(instance.distance_matrix, "distance_func", None)
        if distance_func is None:
            raise ValueError("Distance matrix must define a distance_func for baseline solver.")

        travel_distance: Dict[Tuple[int, int], float] = {}
        travel_time: Dict[Tuple[int, int, int], float] = {}
        travel_energy: Dict[Tuple[int, int, int], float] = {}

        for i in node_ids:
            x1, y1 = node_coords[i]
            for j in node_ids:
                x2, y2 = node_coords[j]
                if i == j:
                    dist = 0.0
                else:
                    dist = distance_func(x1, y1, x2, y2)
                travel_distance[(i, j)] = dist
                for vehicle in vehicles:
                    speed = vehicle.speed
                    if speed <= 0:
                        raise ValueError("Vehicle speed must be positive.")
                    time_val = dist / speed
                    travel_time[(vehicle.vehicle_id, i, j)] = time_val
                    travel_energy[(vehicle.vehicle_id, i, j)] = (
                        instance.energy_config.consumption_rate * time_val
                    )

        max_service = max(service_time.values()) if service_time else 0.0
        max_travel_factor = max(
            1.0,
            max((scenario.travel_time_factor for scenario in scenarios), default=1.0),
        )
        max_travel = (max(travel_time.values()) if travel_time else 0.0) * max_travel_factor
        max_charge_time = instance.energy_config.max_charging_time
        horizon = max(
            1.0,
            (len(node_ids) + 1) * (max_travel + max_service + max_charge_time),
        )
        big_m_time = horizon * 2.0
        max_energy_base = (max(travel_energy.values()) if travel_energy else 0.0) * max_travel_factor
        load_coeff = max(0.0, instance.energy_config.load_factor_coeff)
        max_energy = max_energy_base * (1.0 + load_coeff)
        big_m_energy = max(
            (vehicle.battery_capacity for vehicle in vehicles),
            default=0.0,
        ) + max_energy
        max_demand = max((abs(delta) for delta in demand_delta.values()), default=0.0)
        max_demand = max(
            max_demand,
            max(
                (
                    abs(demand)
                    for scenario in scenarios
                    for demand in scenario.task_demands.values()
                ),
                default=0.0,
            ),
        )
        big_m_load = max(
            (vehicle.capacity for vehicle in vehicles),
            default=0.0,
        ) + max_demand
        headway = HeadwayPolicy().min_headway_s

        epoch_count = max(1, instance.decision_epochs)
        epoch_length = horizon / epoch_count if epoch_count > 0 else horizon
        scenario_time_windows: Dict[int, Dict[int, Optional[Tuple[float, float]]]] = {}
        scenario_service_times: Dict[int, Dict[int, float]] = {}
        scenario_demand_delta: Dict[int, Dict[int, float]] = {}
        scenario_queue_times: Dict[int, Dict[int, float]] = {}
        scenario_release_times: Dict[int, Dict[int, float]] = {}
        scenario_travel_factor: Dict[int, float] = {}
        scenario_charging_available: Dict[int, Dict[int, int]] = {}
        epoch_times_by_scenario: Dict[int, List[float]] = {}
        epoch_ids_by_scenario: Dict[int, List[int]] = {}
        task_epoch_by_scenario: Dict[int, Dict[int, int]] = {}
        rule_prefs_by_scenario: Dict[int, Dict[int, RulePreferences]] = {}

        for scenario in scenarios:
            release_times = _build_scenario_release_times(tasks, scenario)
            scenario_release_times[scenario.scenario_id] = release_times
            epoch_times = _build_scenario_epoch_times(release_times, scenario)
            epoch_times_by_scenario[scenario.scenario_id] = epoch_times
            epoch_ids_by_scenario[scenario.scenario_id] = list(range(len(epoch_times)))
            release_times_by_node = {
                pickup_id: release_times.get(task_id, 0.0)
                for task_id, (pickup_id, _) in task_pairs.items()
            }
            scenario_time_windows[scenario.scenario_id] = _build_scenario_time_windows(
                time_windows,
                scenario,
                config,
                release_times_by_node=release_times_by_node,
            )
            scenario_service_times[scenario.scenario_id] = _build_scenario_service_times(
                service_time,
                scenario,
            )
            scenario_demand_delta[scenario.scenario_id] = _build_scenario_demand_delta(
                demand_delta,
                scenario,
                task_pairs,
            )
            queue_default = config.charging_queue_default_s
            queue_times = {
                node_id: scenario.queue_estimates_s.get(
                    node_id,
                    config.charging_queue_estimates_s.get(node_id, queue_default),
                )
                for node_id in charging_nodes
            }
            scenario_queue_times[scenario.scenario_id] = queue_times
            scenario_travel_factor[scenario.scenario_id] = max(0.0, scenario.travel_time_factor)
            scenario_charging_available[scenario.scenario_id] = scenario.charging_availability or {}

            arrival_shift, _, priority_boost = _get_scenario_modifiers(scenario, config)
            task_epoch_by_scenario[scenario.scenario_id] = _assign_task_epochs(
                tasks,
                epoch_times_by_scenario[scenario.scenario_id],
                release_times=release_times,
            )
            rule_prefs_by_scenario[scenario.scenario_id] = _build_rule_preferences(
                tasks,
                vehicles,
                node_coords,
                distance_func,
                instance,
                cost_params,
                config,
                start_depot_id,
                charging_nodes,
                time_windows=scenario_time_windows[scenario.scenario_id],
                priority_boost=priority_boost,
                queue_default_s=queue_default,
                queue_estimates_s=scenario.queue_estimates_s
                or config.charging_queue_estimates_s,
            )

        rule_ids = list(range(1, max(1, instance.rule_count) + 1))

        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if solver is None:
            raise RuntimeError("Failed to create OR-Tools CBC solver.")
        if config.time_limit_s > 0:
            solver.SetTimeLimit(int(config.time_limit_s * 1000))
        if config.mip_gap > 0:
            solver.SetSolverSpecificParametersAsString(f"ratioGap={config.mip_gap}")

        arcs: List[Tuple[int, int]] = []
        for i in node_ids:
            if i == end_depot_id:
                continue
            for j in node_ids:
                if j == start_depot_id or i == j:
                    continue
                arcs.append((i, j))

        x: Dict[Tuple[int, int, int, int], pywraplp.Variable] = {}
        y: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        for scenario_id in scenario_ids:
            for vehicle_id in vehicle_ids:
                for node_id in node_ids:
                    y[(vehicle_id, node_id, scenario_id)] = solver.BoolVar(
                        f"y_{vehicle_id}_{node_id}_{scenario_id}"
                    )
                for i, j in arcs:
                    x[(vehicle_id, i, j, scenario_id)] = solver.BoolVar(
                        f"x_{vehicle_id}_{i}_{j}_{scenario_id}"
                    )

        z: Dict[Tuple[int, int], pywraplp.Variable] = {}
        for scenario_id in scenario_ids:
            for task in tasks:
                z[(task.task_id, scenario_id)] = solver.BoolVar(
                    f"z_{task.task_id}_{scenario_id}"
                )

        arrival: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        start: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        depart: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        generic_wait: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        conflict_wait: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        standby: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        tardiness: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        charge_time: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        charge_amount: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        battery_arr: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        battery_dep: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        load: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        travel_actual: Dict[Tuple[int, int, int, int], pywraplp.Variable] = {}

        for scenario_id in scenario_ids:
            for vehicle_id in vehicle_ids:
                vehicle = vehicle_by_id[vehicle_id]
                for node_id in node_ids:
                    arrival[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"A_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    start[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"S_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    depart[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"F_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    generic_wait[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"U_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    conflict_wait[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"CW_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    standby[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"SB_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    tardiness[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"L_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    charge_time[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        max_charge_time,
                        f"G_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    charge_amount[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        instance.energy_config.max_charging_amount,
                        f"Q_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    battery_arr[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        vehicle.battery_capacity,
                        f"Barr_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    battery_dep[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        vehicle.battery_capacity,
                        f"Bdep_{vehicle_id}_{node_id}_{scenario_id}",
                    )
                    load[(vehicle_id, node_id, scenario_id)] = solver.NumVar(
                        0.0,
                        vehicle.capacity,
                        f"Ld_{vehicle_id}_{node_id}_{scenario_id}",
                    )

                for i, j in arcs:
                    travel_actual[(vehicle_id, i, j, scenario_id)] = solver.NumVar(
                        0.0,
                        big_m_time,
                        f"DT_{vehicle_id}_{i}_{j}_{scenario_id}",
                    )

        edge_order: Dict[Tuple[int, int, int, int, int], pywraplp.Variable] = {}
        edge_wait: Dict[Tuple[int, int, int, int, int], pywraplp.Variable] = {}
        node_order: Dict[Tuple[int, int, int, int], pywraplp.Variable] = {}

        if config.enable_conflict and len(vehicle_ids) > 1:
            for scenario_id in scenario_ids:
                for vehicle_a, vehicle_b in combinations(vehicle_ids, 2):
                    for node_id in internal_nodes:
                        node_order[(vehicle_a, vehicle_b, node_id, scenario_id)] = solver.BoolVar(
                            f"node_order_{vehicle_a}_{vehicle_b}_{node_id}_{scenario_id}"
                        )
                        node_order[(vehicle_b, vehicle_a, node_id, scenario_id)] = solver.BoolVar(
                            f"node_order_{vehicle_b}_{vehicle_a}_{node_id}_{scenario_id}"
                        )
                    for i, j in arcs:
                        if i in (start_depot_id, end_depot_id) or j in (
                            start_depot_id,
                            end_depot_id,
                        ):
                            continue
                        edge_order[(vehicle_a, vehicle_b, i, j, scenario_id)] = solver.BoolVar(
                            f"edge_order_{vehicle_a}_{vehicle_b}_{i}_{j}_{scenario_id}"
                        )
                        edge_order[(vehicle_b, vehicle_a, i, j, scenario_id)] = solver.BoolVar(
                            f"edge_order_{vehicle_b}_{vehicle_a}_{i}_{j}_{scenario_id}"
                        )
                        edge_wait[(vehicle_a, vehicle_b, i, j, scenario_id)] = solver.NumVar(
                            0.0,
                            big_m_time,
                            f"t_{vehicle_a}_{vehicle_b}_{i}_{j}_{scenario_id}",
                        )
                        edge_wait[(vehicle_b, vehicle_a, i, j, scenario_id)] = solver.NumVar(
                            0.0,
                            big_m_time,
                            f"t_{vehicle_b}_{vehicle_a}_{i}_{j}_{scenario_id}",
                        )

        rule_select: Dict[Tuple[int, int, int], pywraplp.Variable] = {}
        rule_active: Dict[Tuple[int, int], pywraplp.Variable] = {}
        if config.enable_rule_selection and instance.rule_count:
            for scenario_id in scenario_ids:
                epoch_ids = epoch_ids_by_scenario.get(scenario_id, [0])
                for epoch in epoch_ids:
                    choices = []
                    for rule_id in rule_ids:
                        var = solver.BoolVar(f"pi_{epoch}_{rule_id}_{scenario_id}")
                        rule_select[(epoch, rule_id, scenario_id)] = var
                        choices.append(var)
                    solver.Add(solver.Sum(choices) == 1)
                for rule_id in rule_ids:
                    active = solver.BoolVar(f"pi_active_{rule_id}_{scenario_id}")
                    rule_active[(rule_id, scenario_id)] = active
                    for epoch in epoch_ids:
                        solver.Add(active >= rule_select[(epoch, rule_id, scenario_id)])
                    solver.Add(
                        active
                        <= solver.Sum(
                            rule_select[(epoch, rule_id, scenario_id)]
                            for epoch in epoch_ids
                        )
                    )

        scenario_availability = _build_task_availability(scenarios, tasks)

        for scenario_id in scenario_ids:
            for vehicle_id in vehicle_ids:
                solver.Add(y[(vehicle_id, start_depot_id, scenario_id)] == 1)
                solver.Add(y[(vehicle_id, end_depot_id, scenario_id)] == 1)

                out_start = [
                    x[(vehicle_id, start_depot_id, j, scenario_id)]
                    for j in node_ids
                    if (start_depot_id, j) in arcs
                ]
                solver.Add(solver.Sum(out_start) == 1)

                in_end = [
                    x[(vehicle_id, i, end_depot_id, scenario_id)]
                    for i in node_ids
                    if (i, end_depot_id) in arcs
                ]
                solver.Add(solver.Sum(in_end) == 1)

                in_start = [
                    x[(vehicle_id, i, start_depot_id, scenario_id)]
                    for i in node_ids
                    if (i, start_depot_id) in arcs
                ]
                solver.Add(solver.Sum(in_start) == 0)

                out_end = [
                    x[(vehicle_id, end_depot_id, j, scenario_id)]
                    for j in node_ids
                    if (end_depot_id, j) in arcs
                ]
                solver.Add(solver.Sum(out_end) == 0)

                for node_id in internal_nodes:
                    outgoing = [
                        x[(vehicle_id, node_id, j, scenario_id)]
                        for j in node_ids
                        if (node_id, j) in arcs
                    ]
                    incoming = [
                        x[(vehicle_id, j, node_id, scenario_id)]
                        for j in node_ids
                        if (j, node_id) in arcs
                    ]
                    solver.Add(solver.Sum(outgoing) == y[(vehicle_id, node_id, scenario_id)])
                    solver.Add(solver.Sum(incoming) == y[(vehicle_id, node_id, scenario_id)])

        for scenario_id in scenario_ids:
            for task_id, (pickup_id, delivery_id) in task_pairs.items():
                solver.Add(
                    solver.Sum(
                        y[(vehicle_id, pickup_id, scenario_id)]
                        for vehicle_id in vehicle_ids
                    )
                    == z[(task_id, scenario_id)]
                )
                solver.Add(
                    solver.Sum(
                        y[(vehicle_id, delivery_id, scenario_id)]
                        for vehicle_id in vehicle_ids
                    )
                    == z[(task_id, scenario_id)]
                )
                for vehicle_id in vehicle_ids:
                    solver.Add(
                        y[(vehicle_id, pickup_id, scenario_id)]
                        == y[(vehicle_id, delivery_id, scenario_id)]
                    )
                availability = scenario_availability[(task_id, scenario_id)]
                solver.Add(z[(task_id, scenario_id)] <= availability)

        for scenario_id in scenario_ids:
            for vehicle_id in vehicle_ids:
                vehicle = vehicle_by_id[vehicle_id]
                solver.Add(arrival[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(start[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(depart[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(generic_wait[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(conflict_wait[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(standby[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(charge_time[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(charge_amount[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(tardiness[(vehicle_id, start_depot_id, scenario_id)] == 0.0)
                solver.Add(
                    battery_arr[(vehicle_id, start_depot_id, scenario_id)]
                    == vehicle.initial_battery
                )
                solver.Add(
                    battery_dep[(vehicle_id, start_depot_id, scenario_id)]
                    == vehicle.initial_battery
                )
                solver.Add(load[(vehicle_id, start_depot_id, scenario_id)] == vehicle.initial_load)

                solver.Add(generic_wait[(vehicle_id, end_depot_id, scenario_id)] == 0.0)
                solver.Add(conflict_wait[(vehicle_id, end_depot_id, scenario_id)] == 0.0)
                solver.Add(standby[(vehicle_id, end_depot_id, scenario_id)] == 0.0)
                solver.Add(charge_time[(vehicle_id, end_depot_id, scenario_id)] == 0.0)
                solver.Add(charge_amount[(vehicle_id, end_depot_id, scenario_id)] == 0.0)
                solver.Add(tardiness[(vehicle_id, end_depot_id, scenario_id)] == 0.0)
                solver.Add(
                    start[(vehicle_id, end_depot_id, scenario_id)]
                    == arrival[(vehicle_id, end_depot_id, scenario_id)]
                )
                solver.Add(
                    depart[(vehicle_id, end_depot_id, scenario_id)]
                    == arrival[(vehicle_id, end_depot_id, scenario_id)]
                )
                solver.Add(
                    battery_dep[(vehicle_id, end_depot_id, scenario_id)]
                    == battery_arr[(vehicle_id, end_depot_id, scenario_id)]
                )

                for node_id in internal_nodes:
                    solver.Add(
                        start[(vehicle_id, node_id, scenario_id)]
                        == arrival[(vehicle_id, node_id, scenario_id)]
                        + generic_wait[(vehicle_id, node_id, scenario_id)]
                        + conflict_wait[(vehicle_id, node_id, scenario_id)]
                    )
                    solver.Add(
                        depart[(vehicle_id, node_id, scenario_id)]
                        == start[(vehicle_id, node_id, scenario_id)]
                        + scenario_service_times[scenario_id][node_id]
                        + charge_time[(vehicle_id, node_id, scenario_id)]
                        + standby[(vehicle_id, node_id, scenario_id)]
                    )
                    solver.Add(
                        arrival[(vehicle_id, node_id, scenario_id)]
                        <= big_m_time * y[(vehicle_id, node_id, scenario_id)]
                    )
                    solver.Add(
                        generic_wait[(vehicle_id, node_id, scenario_id)]
                        <= big_m_time * y[(vehicle_id, node_id, scenario_id)]
                    )
                    solver.Add(
                        conflict_wait[(vehicle_id, node_id, scenario_id)]
                        <= big_m_time * y[(vehicle_id, node_id, scenario_id)]
                    )
                    solver.Add(
                        standby[(vehicle_id, node_id, scenario_id)]
                        <= big_m_time * y[(vehicle_id, node_id, scenario_id)]
                    )
                    solver.Add(
                        tardiness[(vehicle_id, node_id, scenario_id)]
                        <= big_m_time * y[(vehicle_id, node_id, scenario_id)]
                    )

                    if is_charging[node_id]:
                        if scenario_charging_available[scenario_id].get(node_id, 1) == 0:
                            solver.Add(charge_time[(vehicle_id, node_id, scenario_id)] == 0.0)
                            solver.Add(charge_amount[(vehicle_id, node_id, scenario_id)] == 0.0)
                            continue
                        queue_time = scenario_queue_times[scenario_id].get(node_id, 0.0)
                        if queue_time > 0:
                            solver.Add(
                                generic_wait[(vehicle_id, node_id, scenario_id)]
                                >= queue_time * y[(vehicle_id, node_id, scenario_id)]
                            )
                        solver.Add(
                            charge_time[(vehicle_id, node_id, scenario_id)]
                            <= max_charge_time * y[(vehicle_id, node_id, scenario_id)]
                        )
                        solver.Add(
                            charge_amount[(vehicle_id, node_id, scenario_id)]
                            <= instance.energy_config.max_charging_amount
                            * y[(vehicle_id, node_id, scenario_id)]
                        )
                        solver.Add(
                            charge_time[(vehicle_id, node_id, scenario_id)]
                            == charge_amount[(vehicle_id, node_id, scenario_id)]
                            / (
                                instance.energy_config.charging_rate
                                * instance.energy_config.charging_efficiency
                            )
                        )
                    else:
                        solver.Add(charge_time[(vehicle_id, node_id, scenario_id)] == 0.0)
                        solver.Add(charge_amount[(vehicle_id, node_id, scenario_id)] == 0.0)

                    scenario_window = scenario_time_windows[scenario_id].get(node_id)
                    if scenario_window:
                        earliest, latest = scenario_window
                        solver.Add(
                            start[(vehicle_id, node_id, scenario_id)]
                            >= earliest - big_m_time * (1 - y[(vehicle_id, node_id, scenario_id)])
                        )
                        solver.Add(
                            start[(vehicle_id, node_id, scenario_id)]
                            <= latest
                            + tardiness[(vehicle_id, node_id, scenario_id)]
                            + big_m_time * (1 - y[(vehicle_id, node_id, scenario_id)])
                        )
                        solver.Add(
                            tardiness[(vehicle_id, node_id, scenario_id)]
                            >= start[(vehicle_id, node_id, scenario_id)] - latest
                        )

                    solver.Add(
                        battery_dep[(vehicle_id, node_id, scenario_id)]
                        == battery_arr[(vehicle_id, node_id, scenario_id)]
                        + charge_amount[(vehicle_id, node_id, scenario_id)]
                    )

                # Enforce task release times on pickup nodes even if no time window is defined.
                release_times = scenario_release_times.get(scenario_id, {})
                for task_id, (pickup_id, _) in task_pairs.items():
                    release_time = release_times.get(task_id, 0.0)
                    if release_time > 0:
                        solver.Add(
                            start[(vehicle_id, pickup_id, scenario_id)]
                            >= release_time - big_m_time * (1 - y[(vehicle_id, pickup_id, scenario_id)])
                        )

        for scenario_id in scenario_ids:
            for vehicle_id in vehicle_ids:
                for i, j in arcs:
                    travel_time_factor = max(0.0, scenario_travel_factor.get(scenario_id, 1.0))
                    scenario_travel_time = (
                        travel_time[(vehicle_id, i, j)] * travel_time_factor
                    )
                    scenario_travel_energy = (
                        travel_energy[(vehicle_id, i, j)] * travel_time_factor
                    )
                    load_coeff = max(0.0, instance.energy_config.load_factor_coeff)
                    vehicle_capacity = max(0.0, vehicle_by_id[vehicle_id].capacity)
                    if load_coeff > 0.0 and vehicle_capacity > 0.0:
                        load_energy_coeff = scenario_travel_energy * load_coeff / vehicle_capacity
                        energy_used = scenario_travel_energy + load_energy_coeff * load[
                            (vehicle_id, i, scenario_id)
                        ]
                    else:
                        energy_used = scenario_travel_energy
                    solver.Add(
                        travel_actual[(vehicle_id, i, j, scenario_id)]
                        >= scenario_travel_time * x[(vehicle_id, i, j, scenario_id)]
                    )
                    solver.Add(
                        travel_actual[(vehicle_id, i, j, scenario_id)]
                        <= scenario_travel_time + big_m_time * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )
                    solver.Add(
                        travel_actual[(vehicle_id, i, j, scenario_id)]
                        <= big_m_time * x[(vehicle_id, i, j, scenario_id)]
                    )

                    solver.Add(
                        arrival[(vehicle_id, j, scenario_id)]
                        >= depart[(vehicle_id, i, scenario_id)]
                        + travel_actual[(vehicle_id, i, j, scenario_id)]
                        - big_m_time * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )
                    solver.Add(
                        arrival[(vehicle_id, j, scenario_id)]
                        <= depart[(vehicle_id, i, scenario_id)]
                        + travel_actual[(vehicle_id, i, j, scenario_id)]
                        + big_m_time * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )

                    solver.Add(
                        battery_arr[(vehicle_id, j, scenario_id)]
                        >= battery_dep[(vehicle_id, i, scenario_id)]
                        - energy_used
                        - big_m_energy * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )
                    solver.Add(
                        battery_arr[(vehicle_id, j, scenario_id)]
                        <= battery_dep[(vehicle_id, i, scenario_id)]
                        - energy_used
                        + big_m_energy * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )

                    solver.Add(
                        load[(vehicle_id, j, scenario_id)]
                        >= load[(vehicle_id, i, scenario_id)]
                        + scenario_demand_delta[scenario_id][j]
                        - big_m_load * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )
                    solver.Add(
                        load[(vehicle_id, j, scenario_id)]
                        <= load[(vehicle_id, i, scenario_id)]
                        + scenario_demand_delta[scenario_id][j]
                        + big_m_load * (1 - x[(vehicle_id, i, j, scenario_id)])
                    )

        for scenario_id in scenario_ids:
            for vehicle_id in vehicle_ids:
                for task_id, (pickup_id, delivery_id) in task_pairs.items():
                    solver.Add(
                        start[(vehicle_id, delivery_id, scenario_id)]
                        >= depart[(vehicle_id, pickup_id, scenario_id)]
                        - big_m_time * (1 - y[(vehicle_id, pickup_id, scenario_id)])
                    )

        if config.enable_conflict and len(vehicle_ids) > 1:
            for scenario_id in scenario_ids:
                for vehicle_a, vehicle_b in combinations(vehicle_ids, 2):
                    for node_id in internal_nodes:
                        forward_key = (vehicle_a, vehicle_b, node_id, scenario_id)
                        backward_key = (vehicle_b, vehicle_a, node_id, scenario_id)
                        if forward_key in node_order and backward_key in node_order:
                            solver.Add(
                                node_order[forward_key]
                                + node_order[backward_key]
                                >= y[(vehicle_a, node_id, scenario_id)]
                                + y[(vehicle_b, node_id, scenario_id)]
                                - 1
                            )
                            solver.Add(
                                node_order[forward_key]
                                + node_order[backward_key]
                                <= y[(vehicle_a, node_id, scenario_id)]
                            )
                            solver.Add(
                                node_order[forward_key]
                                + node_order[backward_key]
                                <= y[(vehicle_b, node_id, scenario_id)]
                            )

                            solver.Add(
                                start[(vehicle_b, node_id, scenario_id)]
                                >= depart[(vehicle_a, node_id, scenario_id)]
                                + headway
                                - big_m_time * (1 - node_order[forward_key])
                                - big_m_time
                                * (
                                    2
                                    - y[(vehicle_a, node_id, scenario_id)]
                                    - y[(vehicle_b, node_id, scenario_id)]
                                )
                            )
                            solver.Add(
                                start[(vehicle_a, node_id, scenario_id)]
                                >= depart[(vehicle_b, node_id, scenario_id)]
                                + headway
                                - big_m_time * (1 - node_order[backward_key])
                                - big_m_time
                                * (
                                    2
                                    - y[(vehicle_a, node_id, scenario_id)]
                                    - y[(vehicle_b, node_id, scenario_id)]
                                )
                            )

                    for i, j in arcs:
                        if i in (start_depot_id, end_depot_id) or j in (
                            start_depot_id,
                            end_depot_id,
                        ):
                            continue
                        forward_key = (vehicle_a, vehicle_b, i, j, scenario_id)
                        backward_key = (vehicle_b, vehicle_a, i, j, scenario_id)
                        if forward_key in edge_order and backward_key in edge_order:
                            solver.Add(
                                edge_order[forward_key]
                                + edge_order[backward_key]
                                >= x[(vehicle_a, i, j, scenario_id)]
                                + x[(vehicle_b, i, j, scenario_id)]
                                - 1
                            )
                            solver.Add(
                                edge_order[forward_key]
                                + edge_order[backward_key]
                                <= x[(vehicle_a, i, j, scenario_id)]
                            )
                            solver.Add(
                                edge_order[forward_key]
                                + edge_order[backward_key]
                                <= x[(vehicle_b, i, j, scenario_id)]
                            )

                            solver.Add(
                                edge_wait[forward_key]
                                >= depart[(vehicle_a, i, scenario_id)]
                                + headway
                                - arrival[(vehicle_b, i, scenario_id)]
                                - big_m_time * (1 - edge_order[forward_key])
                                - big_m_time
                                * (
                                    2
                                    - x[(vehicle_a, i, j, scenario_id)]
                                    - x[(vehicle_b, i, j, scenario_id)]
                                )
                            )
                            solver.Add(
                                edge_wait[backward_key]
                                >= depart[(vehicle_b, i, scenario_id)]
                                + headway
                                - arrival[(vehicle_a, i, scenario_id)]
                                - big_m_time * (1 - edge_order[backward_key])
                                - big_m_time
                                * (
                                    2
                                    - x[(vehicle_a, i, j, scenario_id)]
                                    - x[(vehicle_b, i, j, scenario_id)]
                                )
                            )
                            solver.Add(
                                edge_wait[forward_key]
                                <= big_m_time * x[(vehicle_a, i, j, scenario_id)]
                            )
                            solver.Add(
                                edge_wait[forward_key]
                                <= big_m_time * x[(vehicle_b, i, j, scenario_id)]
                            )
                            solver.Add(
                                edge_wait[backward_key]
                                <= big_m_time * x[(vehicle_a, i, j, scenario_id)]
                            )
                            solver.Add(
                                edge_wait[backward_key]
                                <= big_m_time * x[(vehicle_b, i, j, scenario_id)]
                            )

                for vehicle_b in vehicle_ids:
                    for node_id in internal_nodes:
                        related_waits = []
                        for vehicle_a in vehicle_ids:
                            if vehicle_a == vehicle_b:
                                continue
                            for j in node_ids:
                                key = (vehicle_a, vehicle_b, node_id, j, scenario_id)
                                if key in edge_wait:
                                    related_waits.append(edge_wait[key])
                        if related_waits:
                            solver.Add(
                                conflict_wait[(vehicle_b, node_id, scenario_id)]
                                >= solver.Sum(related_waits)
                            )
                        else:
                            solver.Add(conflict_wait[(vehicle_b, node_id, scenario_id)] == 0.0)
        else:
            for scenario_id in scenario_ids:
                for vehicle_id in vehicle_ids:
                    for node_id in internal_nodes:
                        solver.Add(conflict_wait[(vehicle_id, node_id, scenario_id)] == 0.0)

        if config.enable_rule_selection and instance.rule_count:
            for scenario_id in scenario_ids:
                epoch_ids = epoch_ids_by_scenario.get(scenario_id, [0])
                for epoch in epoch_ids:
                    for rule_id in rule_ids:
                        pi_var = rule_select[(epoch, rule_id, scenario_id)]
                        tasks_in_epoch = [
                            task.task_id
                            for task in tasks
                            if task_epoch_by_scenario[scenario_id][task.task_id] == epoch
                        ]
                        prefs = rule_prefs_by_scenario[scenario_id].get(
                            rule_id, RulePreferences()
                        )
                        if rule_id in ACCEPT_RULES:
                            for task_id in tasks_in_epoch:
                                indicator = prefs.accept_indicator.get(task_id, 0)
                                availability = scenario_availability[(task_id, scenario_id)]
                                indicator = int(indicator and availability)
                                solver.Add(
                                    z[(task_id, scenario_id)]
                                    <= indicator + (1 - pi_var)
                                )
                                solver.Add(
                                    z[(task_id, scenario_id)]
                                    >= indicator - (1 - pi_var)
                                )
                        if rule_id in (RULE_STTF, RULE_INSERT_MIN_COST):
                            for task_id in tasks_in_epoch:
                                candidates = prefs.candidate_vehicles.get(task_id, [])
                                if not candidates:
                                    continue
                                pickup_id = pickup_nodes[task_id]
                                solver.Add(
                                    solver.Sum(
                                        y[(vehicle_id, pickup_id, scenario_id)]
                                        for vehicle_id in candidates
                                    )
                                    >= z[(task_id, scenario_id)] - (1 - pi_var)
                                )
                        if rule_id in (RULE_EDD, RULE_MST, RULE_HPF):
                            preferred_tasks = set(prefs.preferred_tasks)
                            for task_id in tasks_in_epoch:
                                if task_id in preferred_tasks:
                                    continue
                                solver.Add(z[(task_id, scenario_id)] <= 1 - pi_var)

                for rule_id in rule_ids:
                    if (rule_id, scenario_id) not in rule_active:
                        continue
                    pi_active = rule_active[(rule_id, scenario_id)]
                    prefs = rule_prefs_by_scenario[scenario_id].get(
                        rule_id, RulePreferences()
                    )
                    if rule_id == RULE_CHARGE_URGENT:
                        for vehicle_id in vehicle_ids:
                            preferred = prefs.preferred_charging_stations.get(vehicle_id, [])
                            if not preferred:
                                continue
                            for node_id in charging_nodes:
                                if node_id in preferred:
                                    continue
                                solver.Add(
                                    y[(vehicle_id, node_id, scenario_id)]
                                    <= 1 - pi_active
                                )
                            if prefs.force_charge.get(vehicle_id, 0) == 1:
                                solver.Add(
                                    solver.Sum(
                                        y[(vehicle_id, node_id, scenario_id)]
                                        for node_id in preferred
                                    )
                                    >= pi_active
                                )
                    if rule_id == RULE_CHARGE_TARGET and prefs.charge_level_ratios:
                        for vehicle_id in vehicle_ids:
                            preferred = prefs.preferred_charging_stations.get(vehicle_id, [])
                            if preferred:
                                for node_id in charging_nodes:
                                    if node_id in preferred:
                                        continue
                                    solver.Add(
                                        y[(vehicle_id, node_id, scenario_id)]
                                        <= 1 - pi_active
                                    )
                        for vehicle_id in vehicle_ids:
                            for node_id in charging_nodes:
                                levels = []
                                for idx, ratio in enumerate(prefs.charge_level_ratios):
                                    level = solver.BoolVar(
                                        f"rule6_level_{vehicle_id}_{node_id}_{scenario_id}_{idx}"
                                    )
                                    levels.append((ratio, level))
                                level_sum = solver.Sum(level for _, level in levels)
                                solver.Add(level_sum <= y[(vehicle_id, node_id, scenario_id)])
                                solver.Add(
                                    level_sum
                                    >= y[(vehicle_id, node_id, scenario_id)]
                                    - (1 - pi_active)
                                )
                                weighted_sum = solver.Sum(
                                    ratio * level for ratio, level in levels
                                )
                                solver.Add(
                                    charge_amount[(vehicle_id, node_id, scenario_id)]
                                    <= instance.energy_config.max_charging_amount
                                    * weighted_sum
                                    + instance.energy_config.max_charging_amount
                                    * (1 - pi_active)
                                )
                                solver.Add(
                                    charge_amount[(vehicle_id, node_id, scenario_id)]
                                    >= instance.energy_config.max_charging_amount
                                    * weighted_sum
                                    - instance.energy_config.max_charging_amount
                                    * (1 - pi_active)
                                )
                    if rule_id == RULE_CHARGE_OPPORTUNITY and prefs.min_charge_ratio is not None:
                        min_charge = prefs.min_charge_ratio
                        for vehicle_id in vehicle_ids:
                            for node_id in charging_nodes:
                                solver.Add(
                                    charge_amount[(vehicle_id, node_id, scenario_id)]
                                    >= instance.energy_config.max_charging_amount
                                    * min_charge
                                    * y[(vehicle_id, node_id, scenario_id)]
                                    - instance.energy_config.max_charging_amount
                                    * (1 - pi_active)
                                )
                    if rule_id in STANDBY_RULES:
                        for vehicle_id in vehicle_ids:
                            preferred_nodes = set(
                                prefs.preferred_standby_nodes.get(vehicle_id, [])
                            )
                            if not preferred_nodes:
                                continue
                            for node_id in internal_nodes:
                                if node_id in preferred_nodes:
                                    continue
                                solver.Add(
                                    standby[(vehicle_id, node_id, scenario_id)]
                                    <= big_m_time * (1 - pi_active)
                                )

        if config.enable_partial_charging and not config.enable_rule_selection:
            for scenario_id in scenario_ids:
                for vehicle_id in vehicle_ids:
                    for node_id in charging_nodes:
                        levels = []
                        for idx, ratio in enumerate(config.charging_level_ratios):
                            level = solver.BoolVar(
                                f"charge_level_{vehicle_id}_{node_id}_{scenario_id}_{idx}"
                            )
                            levels.append(level)
                        solver.Add(
                            solver.Sum(levels) == y[(vehicle_id, node_id, scenario_id)]
                        )
                        solver.Add(
                            charge_amount[(vehicle_id, node_id, scenario_id)]
                            == instance.energy_config.max_charging_amount
                            * solver.Sum(
                                ratio * level
                                for ratio, level in zip(config.charging_level_ratios, levels)
                            )
                        )

        distance_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                travel_distance[(i, j)] * x[(vehicle_id, i, j, scenario_id)]
                for vehicle_id in vehicle_ids
                for i, j in arcs
            )
            for scenario_id in scenario_ids
        )
        time_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                travel_actual[(vehicle_id, i, j, scenario_id)]
                for vehicle_id in vehicle_ids
                for i, j in arcs
            )
            for scenario_id in scenario_ids
        )
        charging_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                charge_amount[(vehicle_id, node_id, scenario_id)]
                for vehicle_id in vehicle_ids
                for node_id in node_ids
            )
            for scenario_id in scenario_ids
        )
        delay_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                tardiness[(vehicle_id, node_id, scenario_id)]
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        wait_weight_scale = max(0.0, config.wait_weight_scale)
        wait_weight_default = max(0.0, config.wait_weight_default) * wait_weight_scale
        wait_weight_charging = max(0.0, config.wait_weight_charging) * wait_weight_scale
        wait_weight_depot = max(0.0, config.wait_weight_depot) * wait_weight_scale
        node_wait_weight = {
            node_id: wait_weight_default for node_id in internal_nodes
        }
        for node_id in charging_nodes:
            if node_id in node_wait_weight:
                node_wait_weight[node_id] = wait_weight_charging
        if start_depot_id in node_wait_weight:
            node_wait_weight[start_depot_id] = wait_weight_depot
        if end_depot_id in node_wait_weight:
            node_wait_weight[end_depot_id] = wait_weight_depot

        wait_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                node_wait_weight[node_id]
                * generic_wait[(vehicle_id, node_id, scenario_id)]
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        conflict_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                conflict_wait[(vehicle_id, node_id, scenario_id)]
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        standby_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(
                standby[(vehicle_id, node_id, scenario_id)]
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        rejection_expr = solver.Sum(
            scenario_prob[scenario_id]
            * solver.Sum(1 - z[(task.task_id, scenario_id)] for task in tasks)
            for scenario_id in scenario_ids
        )

        solver.Minimize(
            cost_params.C_tr * distance_expr
            + cost_params.C_time * time_expr
            + cost_params.C_ch * charging_expr
            + cost_params.C_delay * delay_expr
            + cost_params.C_wait * wait_expr
            + cost_params.C_conflict * conflict_expr
            + cost_params.C_missing_task * rejection_expr
            + cost_params.C_infeasible * rejection_expr
            + cost_params.C_standby * standby_expr
        )

        status_code = solver.Solve()
        status = _status_label(status_code, pywraplp)

        if status not in ("OPTIMAL", "FEASIBLE"):
            return MIPBaselineResult(status=status)

        total_distance = sum(
            scenario_prob[scenario_id]
            * sum(
                travel_distance[(i, j)] * x[(vehicle_id, i, j, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for i, j in arcs
            )
            for scenario_id in scenario_ids
        )
        total_time = sum(
            scenario_prob[scenario_id]
            * sum(
                travel_actual[(vehicle_id, i, j, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for i, j in arcs
            )
            for scenario_id in scenario_ids
        )
        total_charging = sum(
            scenario_prob[scenario_id]
            * sum(
                charge_amount[(vehicle_id, node_id, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for node_id in node_ids
            )
            for scenario_id in scenario_ids
        )
        total_delay = sum(
            scenario_prob[scenario_id]
            * sum(
                tardiness[(vehicle_id, node_id, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        total_waiting = sum(
            scenario_prob[scenario_id]
            * sum(
                generic_wait[(vehicle_id, node_id, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        total_waiting_weighted = sum(
            scenario_prob[scenario_id]
            * sum(
                node_wait_weight[node_id]
                * generic_wait[(vehicle_id, node_id, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        total_conflict_waiting = sum(
            scenario_prob[scenario_id]
            * sum(
                var.solution_value()
                for key, var in edge_wait.items()
                if key[-1] == scenario_id
            )
            for scenario_id in scenario_ids
        )
        total_standby = sum(
            scenario_prob[scenario_id]
            * sum(
                standby[(vehicle_id, node_id, scenario_id)].solution_value()
                for vehicle_id in vehicle_ids
                for node_id in internal_nodes
            )
            for scenario_id in scenario_ids
        )
        rejected_tasks = sum(
            scenario_prob[scenario_id]
            * sum(1 - z[(task.task_id, scenario_id)].solution_value() for task in tasks)
            for scenario_id in scenario_ids
        )
        num_charging_stops = sum(
            scenario_prob[scenario_id]
            * sum(
                1
                for vehicle_id in vehicle_ids
                for node_id in node_ids
                if is_charging.get(node_id, False)
                and charge_amount[(vehicle_id, node_id, scenario_id)].solution_value()
                > 1e-6
            )
            for scenario_id in scenario_ids
        )

        details = {
            "total_distance": total_distance,
            "total_time": total_time,
            "total_charging": total_charging,
            "total_delay": total_delay,
            "total_waiting": total_waiting,
            "total_waiting_weighted": total_waiting_weighted,
            "total_conflict_waiting": total_conflict_waiting,
            "total_standby": total_standby,
            "rejected_tasks": rejected_tasks,
            "num_charging_stops": num_charging_stops,
            "distance_cost": total_distance * cost_params.C_tr,
            "time_cost": total_time * cost_params.C_time,
            "charging_cost": total_charging * cost_params.C_ch,
            "delay_cost": total_delay * cost_params.C_delay,
            "waiting_cost": total_waiting_weighted * cost_params.C_wait,
            "conflict_waiting_cost": total_conflict_waiting * cost_params.C_conflict,
            "standby_cost": total_standby * cost_params.C_standby,
            "rejection_cost": rejected_tasks * cost_params.C_missing_task,
            "infeasible_cost": rejected_tasks * cost_params.C_infeasible,
        }
        total_cost = sum(
            details[key]
            for key in (
                "distance_cost",
                "time_cost",
                "charging_cost",
                "delay_cost",
                "waiting_cost",
                "conflict_waiting_cost",
                "standby_cost",
                "rejection_cost",
                "infeasible_cost",
            )
        )
        details["total_cost"] = total_cost

        return MIPBaselineResult(
            status=status,
            objective_value=solver.Objective().Value(),
            details=details,
        )


def _ensure_scenarios(instance: MIPBaselineInstance) -> List[MIPBaselineScenario]:
    if instance.scenarios:
        return instance.scenarios
    return [
        MIPBaselineScenario(
            scenario_id=0,
            probability=1.0,
            task_availability={task.task_id: 1 for task in instance.tasks},
            task_release_times={task.task_id: task.arrival_time for task in instance.tasks},
        )
    ]


def _build_task_availability(
    scenarios: Iterable[MIPBaselineScenario],
    tasks: Iterable,
) -> Dict[Tuple[int, int], int]:
    availability: Dict[Tuple[int, int], int] = {}
    for scenario in scenarios:
        for task in tasks:
            availability[(task.task_id, scenario.scenario_id)] = scenario.task_availability.get(
                task.task_id, 0
            )
    return availability


def _assign_task_epochs(
    tasks: Iterable,
    epoch_times: List[float],
    *,
    release_times: Dict[int, float],
) -> Dict[int, int]:
    mapping: Dict[int, int] = {}
    if not epoch_times:
        for task in tasks:
            mapping[task.task_id] = 0
        return mapping
    sorted_times = sorted(set(epoch_times))
    for task in tasks:
        release_time = release_times.get(task.task_id, task.arrival_time)
        epoch_index = 0
        for idx, epoch_time in enumerate(sorted_times):
            if release_time >= epoch_time:
                epoch_index = idx
            else:
                break
        mapping[task.task_id] = epoch_index
    return mapping


def _build_scenario_release_times(
    tasks: Iterable,
    scenario: MIPBaselineScenario,
) -> Dict[int, float]:
    release_times: Dict[int, float] = {}
    for task in tasks:
        base_release = scenario.task_release_times.get(
            task.task_id,
            task.arrival_time,
        )
        release_times[task.task_id] = base_release + scenario.arrival_time_shift_s
    return release_times


def _build_scenario_epoch_times(
    release_times: Dict[int, float],
    scenario: MIPBaselineScenario,
) -> List[float]:
    if scenario.decision_epoch_times:
        times = sorted(set(scenario.decision_epoch_times))
    else:
        times = sorted(set(release_times.values()))
    if 0.0 not in times:
        times.insert(0, 0.0)
    return times


def _build_scenario_service_times(
    base_service_times: Dict[int, float],
    scenario: MIPBaselineScenario,
) -> Dict[int, float]:
    service_times = dict(base_service_times)
    for node_id, value in scenario.node_service_times.items():
        service_times[node_id] = value
    return service_times


def _build_scenario_demand_delta(
    base_demand_delta: Dict[int, float],
    scenario: MIPBaselineScenario,
    task_pairs: Dict[int, Tuple[int, int]],
) -> Dict[int, float]:
    demand_delta = dict(base_demand_delta)
    if not scenario.task_demands:
        return demand_delta
    for task_id, demand in scenario.task_demands.items():
        pickup_id, delivery_id = task_pairs.get(task_id, (None, None))
        if pickup_id is None:
            continue
        demand_delta[pickup_id] = abs(demand)
        demand_delta[delivery_id] = -abs(demand)
    return demand_delta


def _get_scenario_modifiers(
    scenario: MIPBaselineScenario,
    config: MIPBaselineSolverConfig,
) -> Tuple[float, float, int]:
    if config.scenario_mode.lower() == "medium":
        return (
            scenario.arrival_time_shift_s,
            max(0.0, scenario.time_window_scale),
            scenario.priority_boost,
        )
    return 0.0, 1.0, 0


def _build_scenario_time_windows(
    base_time_windows: Dict[int, Optional[Tuple[float, float]]],
    scenario: MIPBaselineScenario,
    config: MIPBaselineSolverConfig,
    *,
    release_times_by_node: Optional[Dict[int, float]] = None,
) -> Dict[int, Optional[Tuple[float, float]]]:
    shift, scale, _ = _get_scenario_modifiers(scenario, config)
    adjusted: Dict[int, Optional[Tuple[float, float]]] = {}
    for node_id, window in base_time_windows.items():
        override = scenario.node_time_windows.get(node_id)
        if override is not None:
            window = override
        if window is None:
            adjusted[node_id] = None
            continue
        earliest, latest = window
        earliest += shift
        latest += shift
        if release_times_by_node is not None:
            release_time = release_times_by_node.get(node_id)
            if release_time is not None:
                earliest = max(earliest, release_time)
        width = max(0.0, latest - earliest)
        latest = earliest + width * scale
        adjusted[node_id] = (earliest, max(earliest, latest))
    return adjusted


def _build_rule_preferences(
    tasks: List,
    vehicles: List,
    node_coords: Dict[int, Tuple[float, float]],
    distance_func,
    instance: MIPBaselineInstance,
    cost_params: CostParameters,
    config: MIPBaselineSolverConfig,
    depot_id: int,
    charging_nodes: List[int],
    *,
    time_windows: Optional[Dict[int, Optional[Tuple[float, float]]]] = None,
    priority_boost: int = 0,
    queue_default_s: Optional[float] = None,
    queue_estimates_s: Optional[Dict[int, float]] = None,
) -> Dict[int, RulePreferences]:
    travel_time_cache: Dict[Tuple[Tuple[float, float], Tuple[float, float], float], float] = {}

    def travel_time(a: Tuple[float, float], b: Tuple[float, float], speed: float) -> float:
        safe_speed = max(speed, 1e-6)
        key = (a, b, safe_speed)
        if key in travel_time_cache:
            return travel_time_cache[key]
        dist = distance_func(a[0], a[1], b[0], b[1])
        time_val = dist / safe_speed
        travel_time_cache[key] = time_val
        return time_val

    depot_coords = node_coords[depot_id]

    task_due: Dict[int, float] = {}
    task_slack: Dict[int, float] = {}
    task_priority: Dict[int, int] = {}
    min_vehicle_travel: Dict[int, float] = {}
    min_incremental_cost: Dict[int, float] = {}
    feasible_indicator: Dict[int, int] = {}

    for task in tasks:
        pickup = task.pickup_node
        delivery = task.delivery_node
        due_candidates = []
        pickup_window = None
        delivery_window = None
        if time_windows is not None:
            pickup_window = time_windows.get(pickup.node_id)
            delivery_window = time_windows.get(delivery.node_id)
        if pickup_window:
            due_candidates.append(pickup_window[1])
        elif pickup.time_window:
            due_candidates.append(pickup.time_window.latest)
        if delivery_window:
            due_candidates.append(delivery_window[1])
        elif delivery.time_window:
            due_candidates.append(delivery.time_window.latest)
        due = min(due_candidates) if due_candidates else instance.time_config.default_service_time
        task_due[task.task_id] = due
        task_priority[task.task_id] = task.priority + priority_boost

        pickup_coords = node_coords[pickup.node_id]
        delivery_coords = node_coords[delivery.node_id]
        fleet_speed = (
            sum(vehicle.speed for vehicle in vehicles) / len(vehicles)
            if vehicles
            else instance.time_config.vehicle_speed
        )
        base_time = travel_time(pickup_coords, delivery_coords, fleet_speed)
        slack = due - (base_time + pickup.service_time)
        task_slack[task.task_id] = slack

        best_travel = float("inf")
        best_incremental = float("inf")
        is_feasible = False
        for vehicle in vehicles:
            start_coords = vehicle.initial_location
            to_pickup = travel_time(start_coords, pickup_coords, vehicle.speed)
            to_delivery = travel_time(pickup_coords, delivery_coords, vehicle.speed)
            to_depot = travel_time(delivery_coords, depot_coords, vehicle.speed)
            travel_total = to_pickup + to_delivery + to_depot
            best_travel = min(best_travel, to_pickup)
            best_incremental = min(best_incremental, travel_total)

            load_coeff = max(0.0, instance.energy_config.load_factor_coeff)
            load_factor = 1.0
            if vehicle.capacity > 0:
                load_factor += load_coeff * (task.demand / vehicle.capacity)
            energy_needed = instance.energy_config.consumption_rate * travel_total * load_factor
            capacity_ok = task.demand <= vehicle.capacity
            battery_ok = energy_needed <= vehicle.battery_capacity
            pickup_ok = True
            delivery_ok = True
            if pickup_window:
                pickup_ok = to_pickup <= pickup_window[1]
            elif pickup.time_window:
                pickup_ok = to_pickup <= pickup.time_window.latest
            if delivery_window:
                delivery_ok = to_pickup + pickup.service_time + to_delivery <= delivery_window[1]
            elif delivery.time_window:
                delivery_ok = (
                    to_pickup + pickup.service_time + to_delivery
                    <= delivery.time_window.latest
                )
            if capacity_ok and battery_ok and pickup_ok and delivery_ok:
                is_feasible = True

        min_vehicle_travel[task.task_id] = best_travel
        min_incremental_cost[task.task_id] = best_incremental
        feasible_indicator[task.task_id] = 1 if is_feasible else 0

    preferences: Dict[int, RulePreferences] = {}

    if tasks:
        sorted_due = sorted(tasks, key=lambda t: task_due[t.task_id])
        sorted_slack = sorted(tasks, key=lambda t: task_slack[t.task_id])
        sorted_priority = sorted(tasks, key=lambda t: (-task_priority[t.task_id], t.task_id))
        top_k = max(1, config.rule_candidate_top_k)
        preferences[RULE_EDD] = RulePreferences(
            preferred_tasks=[task.task_id for task in sorted_due[:top_k]]
        )
        preferences[RULE_MST] = RulePreferences(
            preferred_tasks=[task.task_id for task in sorted_slack[:top_k]]
        )
        preferences[RULE_HPF] = RulePreferences(
            preferred_tasks=[task.task_id for task in sorted_priority[:top_k]]
        )

    candidate_vehicles_sttf: Dict[int, List[int]] = {}
    candidate_vehicles_insert: Dict[int, List[int]] = {}
    for task in tasks:
        pickup_coords = node_coords[task.pickup_node.node_id]
        candidates = []
        for vehicle in vehicles:
            start_coords = vehicle.initial_location
            to_pickup = travel_time(start_coords, pickup_coords, vehicle.speed)
            candidates.append((vehicle.vehicle_id, to_pickup))
        candidates.sort(key=lambda item: item[1])
        top_k = max(1, config.rule_candidate_top_k)
        candidate_vehicles_sttf[task.task_id] = [
            vehicle_id for vehicle_id, _ in candidates[:top_k]
        ]

        insert_candidates = []
        delivery_coords = node_coords[task.delivery_node.node_id]
        for vehicle in vehicles:
            start_coords = vehicle.initial_location
            cost = (
                travel_time(start_coords, pickup_coords, vehicle.speed)
                + travel_time(pickup_coords, delivery_coords, vehicle.speed)
                + travel_time(delivery_coords, depot_coords, vehicle.speed)
            )
            insert_candidates.append((vehicle.vehicle_id, cost))
        insert_candidates.sort(key=lambda item: item[1])
        candidate_vehicles_insert[task.task_id] = [
            vehicle_id for vehicle_id, _ in insert_candidates[:top_k]
        ]

    preferences[RULE_STTF] = RulePreferences(candidate_vehicles=candidate_vehicles_sttf)
    preferences[RULE_INSERT_MIN_COST] = RulePreferences(
        candidate_vehicles=candidate_vehicles_insert
    )

    preferences[RULE_ACCEPT_FEASIBLE] = RulePreferences(
        accept_indicator=feasible_indicator
    )

    accept_by_value: Dict[int, int] = {}
    for task in tasks:
        estimated_cost = min_incremental_cost[task.task_id] * cost_params.C_tr
        accept_by_value[task.task_id] = 1 if cost_params.C_missing_task > estimated_cost else 0
    preferences[RULE_ACCEPT_VALUE] = RulePreferences(accept_indicator=accept_by_value)

    preferred_charging = {}
    force_charge = {}
    for vehicle in vehicles:
        start_coords = vehicle.initial_location
        closest_station = []
        best_dist = float("inf")
        for node_id in charging_nodes:
            dist = distance_func(
                start_coords[0],
                start_coords[1],
                node_coords[node_id][0],
                node_coords[node_id][1],
            )
            if dist < best_dist - 1e-6:
                closest_station = [node_id]
                best_dist = dist
            elif abs(dist - best_dist) <= 1e-6:
                closest_station.append(node_id)
        if closest_station:
            preferred_charging[vehicle.vehicle_id] = closest_station
        battery_ratio = 0.0
        if vehicle.battery_capacity > 0:
            battery_ratio = vehicle.initial_battery / vehicle.battery_capacity
        force_charge[vehicle.vehicle_id] = 1 if battery_ratio <= config.rule5_soc_threshold else 0

    preferences[RULE_CHARGE_URGENT] = RulePreferences(
        preferred_charging_stations=preferred_charging,
        force_charge=force_charge,
    )
    preferred_charging_rule6: Dict[int, List[int]] = {}
    queue_default = (
        config.charging_queue_default_s if queue_default_s is None else queue_default_s
    )
    queue_estimates = queue_estimates_s if queue_estimates_s is not None else config.charging_queue_estimates_s
    for vehicle in vehicles:
        start_coords = vehicle.initial_location
        best_value = float("inf")
        best_nodes: List[int] = []
        for node_id in charging_nodes:
            travel = travel_time(start_coords, node_coords[node_id], vehicle.speed)
            queue_time = queue_estimates.get(node_id, queue_default)
            total = travel + queue_time
            if total < best_value - 1e-6:
                best_value = total
                best_nodes = [node_id]
            elif abs(total - best_value) <= 1e-6:
                best_nodes.append(node_id)
        if best_nodes:
            preferred_charging_rule6[vehicle.vehicle_id] = best_nodes

    preferences[RULE_CHARGE_TARGET] = RulePreferences(
        preferred_charging_stations=preferred_charging_rule6,
        charge_level_ratios=config.rule6_charge_level_ratios,
    )
    preferences[RULE_CHARGE_OPPORTUNITY] = RulePreferences(
        min_charge_ratio=config.rule7_min_charge_ratio
    )

    standby_low_cost: Dict[int, List[int]] = {}
    standby_lazy: Dict[int, List[int]] = {}
    standby_heatmap: Dict[int, List[int]] = {}
    standby_candidates = [depot_id] + charging_nodes

    pickup_coords_list = [node_coords[task.pickup_node.node_id] for task in tasks]

    for vehicle in vehicles:
        start_coords = vehicle.initial_location
        best_node = depot_id
        best_cost = float("inf")
        for node_id in standby_candidates:
            travel_cost = travel_time(start_coords, node_coords[node_id], vehicle.speed)
            total_cost = travel_cost + config.standby_beta
            if total_cost < best_cost - 1e-6:
                best_cost = total_cost
                best_node = node_id
        standby_low_cost[vehicle.vehicle_id] = [best_node]
        standby_lazy[vehicle.vehicle_id] = [depot_id, best_node]

        heatmap_best = depot_id
        heatmap_cost = float("inf")
        for node_id in standby_candidates:
            if not pickup_coords_list:
                break
            avg_dist = sum(
                distance_func(
                    node_coords[node_id][0],
                    node_coords[node_id][1],
                    pickup[0],
                    pickup[1],
                )
                for pickup in pickup_coords_list
            ) / len(pickup_coords_list)
            if avg_dist < heatmap_cost - 1e-6:
                heatmap_cost = avg_dist
                heatmap_best = node_id
        standby_heatmap[vehicle.vehicle_id] = [heatmap_best]

    preferences[RULE_STANDBY_LOW_COST] = RulePreferences(
        preferred_standby_nodes=standby_low_cost
    )
    preferences[RULE_STANDBY_LAZY] = RulePreferences(
        preferred_standby_nodes=standby_lazy
    )
    preferences[RULE_STANDBY_HEATMAP] = RulePreferences(
        preferred_standby_nodes=standby_heatmap
    )

    return preferences


def _status_label(status_code: int, pywraplp_module) -> str:
    if status_code == pywraplp_module.Solver.OPTIMAL:
        return "OPTIMAL"
    if status_code == pywraplp_module.Solver.FEASIBLE:
        return "FEASIBLE"
    if status_code == pywraplp_module.Solver.INFEASIBLE:
        return "INFEASIBLE"
    if status_code == pywraplp_module.Solver.UNBOUNDED:
        return "UNBOUNDED"
    if status_code == pywraplp_module.Solver.ABNORMAL:
        return "ABNORMAL"
    if status_code == pywraplp_module.Solver.NOT_SOLVED:
        return "NOT_SOLVED"
    return "UNKNOWN"


def get_default_solver(
    solver_config: Optional[MIPBaselineSolverConfig] = None,
) -> MIPBaselineSolver:
    config = solver_config or MIPBaselineSolverConfig()
    name = config.solver_name.lower()
    if name in ("ortools", "or-tools", "or_tools"):
        return ORToolsSolver(config)
    raise ValueError(f"Unsupported MIP baseline solver: {config.solver_name}")


def solve_minimal_instance(
    *,
    solver_config: Optional[MIPBaselineSolverConfig] = None,
    cost_params: Optional[CostParameters] = None,
) -> MIPBaselineResult:
    instance = build_minimal_instance()
    solver = get_default_solver(solver_config)
    return solver.solve(instance, cost_params=cost_params)
