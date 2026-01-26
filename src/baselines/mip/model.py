"""Model specification for the MIP baseline benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from config import CostParameters
from core.node import (
    ChargingNode,
    DepotNode,
    create_charging_node,
    create_depot,
    create_task_node_pair,
)
from core.task import Task, TaskPool
from core.vehicle import Vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeConfig

from baselines.mip.config import MIPBaselineScale, MIPBaselineSolverConfig
from physics.distance import NodeIDHelper


@dataclass
class MIPBaselineInstance:
    """Container holding all inputs required to build the baseline model."""

    tasks: List[Task]
    vehicles: List[Vehicle]
    depot: DepotNode
    charging_stations: List[ChargingNode]
    distance_matrix: DistanceMatrix
    time_config: TimeConfig
    energy_config: EnergyConfig
    rule_count: int = 0
    decision_epochs: int = 1
    scenarios: List["MIPBaselineScenario"] = field(default_factory=list)

    def validate_scale(self, scale: MIPBaselineScale) -> None:
        """Ensure the instance fits inside the baseline scale limits."""

        if len(self.tasks) > scale.max_tasks:
            raise ValueError(
                f"MIP baseline supports at most {scale.max_tasks} tasks, got {len(self.tasks)}."
            )
        if len(self.vehicles) > scale.max_vehicles:
            raise ValueError(
                f"MIP baseline supports at most {scale.max_vehicles} vehicles, got {len(self.vehicles)}."
            )
        if len(self.charging_stations) > scale.max_charging_stations:
            raise ValueError(
                f"MIP baseline supports at most {scale.max_charging_stations} charging stations, "
                f"got {len(self.charging_stations)}."
            )
        if self.rule_count > scale.max_rules:
            raise ValueError(
                f"MIP baseline supports at most {scale.max_rules} rules, got {self.rule_count}."
            )
        if self.decision_epochs > scale.max_decision_epochs:
            raise ValueError(
                f"MIP baseline supports at most {scale.max_decision_epochs} decision epochs, "
                f"got {self.decision_epochs}."
            )
        if self.scenarios and len(self.scenarios) > scale.max_scenarios:
            raise ValueError(
                f"MIP baseline supports at most {scale.max_scenarios} scenarios, "
                f"got {len(self.scenarios)}."
            )


@dataclass(frozen=True)
class MIPBaselineScenario:
    """Scenario definition for dynamic task availability."""

    scenario_id: int
    probability: float
    task_availability: Dict[int, int]
    task_release_times: Dict[int, float] = field(default_factory=dict)
    node_time_windows: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    node_service_times: Dict[int, float] = field(default_factory=dict)
    task_demands: Dict[int, float] = field(default_factory=dict)
    arrival_time_shift_s: float = 0.0
    time_window_scale: float = 1.0
    priority_boost: int = 0
    queue_estimates_s: Dict[int, float] = field(default_factory=dict)
    travel_time_factor: float = 1.0
    charging_availability: Dict[int, int] = field(default_factory=dict)
    decision_epoch_times: List[float] = field(default_factory=list)


@dataclass
class MIPBaselineModelSpec:
    """Human-readable representation of the baseline model."""

    sets: List[str]
    parameters: List[str]
    variables: List[str]
    objective_terms: List[str]
    constraints: List[str]


@dataclass
class MIPBaselineResult:
    """Placeholder result for future solver integration."""

    status: str
    objective_value: Optional[float] = None
    details: Optional[Dict[str, float]] = None


def build_instance(
    task_pool: TaskPool,
    *,
    vehicles: Iterable[Vehicle],
    depot: DepotNode,
    charging_stations: Iterable[ChargingNode],
    distance_matrix: DistanceMatrix,
    time_config: Optional[TimeConfig] = None,
    energy_config: Optional[EnergyConfig] = None,
    rule_count: int = 0,
    decision_epochs: int = 1,
    scenarios: Optional[Iterable[MIPBaselineScenario]] = None,
) -> MIPBaselineInstance:
    """Build a baseline instance from core objects."""

    tasks = list(task_pool.get_all_tasks())
    scenario_list = list(scenarios or [])
    if not scenario_list:
        scenario_list = [
            MIPBaselineScenario(
                scenario_id=0,
                probability=1.0,
                task_availability={task.task_id: 1 for task in tasks},
            )
        ]
    else:
        inferred_epochs = []
        for scenario in scenario_list:
            if scenario.decision_epoch_times:
                inferred_epochs.append(len(set(scenario.decision_epoch_times)))
            elif scenario.task_release_times:
                inferred_epochs.append(len(set(scenario.task_release_times.values())))
        if inferred_epochs:
            decision_epochs = max(decision_epochs, max(inferred_epochs))

    instance = MIPBaselineInstance(
        tasks=tasks,
        vehicles=list(vehicles),
        depot=depot,
        charging_stations=list(charging_stations),
        distance_matrix=distance_matrix,
        time_config=time_config or TimeConfig(),
        energy_config=energy_config or EnergyConfig(),
        rule_count=rule_count,
        decision_epochs=decision_epochs,
        scenarios=scenario_list,
    )
    return instance


def build_model_spec(
    instance: MIPBaselineInstance,
    cost_params: Optional[CostParameters] = None,
    *,
    scale: Optional[MIPBaselineScale] = None,
    solver_config: Optional[MIPBaselineSolverConfig] = None,
) -> MIPBaselineModelSpec:
    """Create a textual model spec aligned with the baseline formulation."""

    scale = scale or MIPBaselineScale()
    solver_config = solver_config or MIPBaselineSolverConfig()
    instance.validate_scale(scale)

    cost_params = cost_params or CostParameters()

    sets = [
        "A: AMR set",
        "R: request set (pickup/delivery pairs)",
        "C: charging station set",
        "N: all nodes (including depot)",
        "E: directed edges over N",
        "W: demand scenarios",
        "E_dec: decision epochs",
        "H: scheduling rules",
    ]

    parameters = [
        "d_ij: travel distance",
        "tau_ij: travel time",
        "E_i, Ehat_i: time window bounds",
        "rho: energy consumption per time",
        "kappa, eta: charging rate/efficiency",
        "T_ch_max, E_ch_max: charging caps",
        "q_i^{queue}: queue waiting estimate at charging stations",
        "Delta_t_safe: headway safety margin",
        "delta_r^w: dynamic task availability",
        "tau_r^w: task release time in scenario w",
        "p_w: scenario probability",
        "gamma_w: travel time factor (scenario congestion)",
        "a_i^w: charging availability indicator",
        f"C_tr={cost_params.C_tr}, C_ch={cost_params.C_ch}",
        f"C_delay={cost_params.C_delay}, C_conflict={cost_params.C_conflict}",
        f"C_missing_task={cost_params.C_missing_task}, C_standby={cost_params.C_standby}",
        f"solver={solver_config.solver_name}",
    ]

    variables = [
        "x_ij^{a,w} in {0,1}: arc usage",
        "y_i^{a,w} in {0,1}: node service indicator",
        "z_r^w in {0,1}: request acceptance",
        "T_i^{a,w}, F_i^{a,w}: service start / departure",
        "L_i^{a,w}: tardiness",
        "u_i^{a,w}: generic waiting",
        "w_i^{a,w}: aggregated conflict waiting (node/edge)",
        "t_ij^{a,b,w}: edge-level conflict waiting (lower bound)",
        "s_i^{a,w}: standby dwell",
        "b_arr,i^{a,w}, b_dep,i^{a,w}: battery levels",
        "q_i^{a,w}, g_i^{a,w}: charging amount/time",
        "l_i^{a,w}: vehicle load",
        "m_ij^{a,b,w}, n_k^{a,b,w}: precedence variables",
        "pi_{e,h}^w: rule selection",
    ]

    objective_terms = [
        "C_tr * sum d_ij * x_ij^{a,w}",
        "C_ch * sum q_i^{a,w}",
        "C_delay * sum L_i^{a,w}",
        "C_conflict * sum w_i^{a,w}",
        "C_missing_task * sum (1 - z_r^w)",
        "C_standby * sum s_i^{a,w}",
    ]

    constraints = [
        "Accepted requests are served exactly once; pickup and delivery on same AMR.",
        "Flow conservation at intermediate nodes; depot departure/arrival once.",
        "Time propagation: arrival, waiting, service, charging, standby.",
        "Pickup precedes delivery for each request.",
        "Time windows with tardiness capturing lateness; pickup release times enforced.",
        "Battery initialization and flow; charge only at stations.",
        "Charging time/energy coupling with caps.",
        "Partial charging (optional discrete levels).",
        "Load propagation and capacity bound.",
        "Collision avoidance with headway precedence on nodes/edges.",
        "Rule selection: exactly one rule per decision epoch.",
    ]

    return MIPBaselineModelSpec(
        sets=sets,
        parameters=parameters,
        variables=variables,
        objective_terms=objective_terms,
        constraints=constraints,
    )


def build_minimal_instance(
    *,
    scale: Optional[MIPBaselineScale] = None,
    rule_count: int = 13,
    decision_epochs: int = 2,
) -> MIPBaselineInstance:
    """Create a deterministic minimal instance for solver smoke tests."""

    scale = scale or MIPBaselineScale()
    num_tasks = scale.max_tasks
    num_charging = scale.max_charging_stations
    node_helper = NodeIDHelper(num_tasks, num_charging)

    depot = create_depot((0.0, 0.0))
    coordinates: Dict[int, tuple[float, float]] = {depot.node_id: depot.coordinates}

    tasks: List[Task] = []
    task_pool = TaskPool()
    for task_id in range(1, num_tasks + 1):
        pickup_id = task_id
        delivery_id = task_id + num_tasks
        pickup_coords = (float(task_id * 6), 0.0)
        delivery_coords = (float(task_id * 6), 8.0)
        pickup, delivery = create_task_node_pair(
            task_id=task_id,
            pickup_id=pickup_id,
            delivery_id=delivery_id,
            pickup_coords=pickup_coords,
            delivery_coords=delivery_coords,
        )
        task = Task(task_id=task_id, pickup_node=pickup, delivery_node=delivery, demand=pickup.demand)
        tasks.append(task)
        task_pool.add_task(task)
        coordinates[pickup_id] = pickup_coords
        coordinates[delivery_id] = delivery_coords

    charging_stations: List[ChargingNode] = []
    for idx, station_id in enumerate(node_helper.get_all_charging_ids()):
        coords = (float(idx * 15), 15.0)
        station = create_charging_node(node_id=station_id, coordinates=coords)
        charging_stations.append(station)
        coordinates[station_id] = coords

    vehicles = [Vehicle(vehicle_id=1), Vehicle(vehicle_id=2)]

    distance_matrix = DistanceMatrix(
        coordinates=coordinates,
        num_tasks=num_tasks,
        num_charging_stations=num_charging,
    )

    instance = build_instance(
        task_pool,
        vehicles=vehicles,
        depot=depot,
        charging_stations=charging_stations,
        distance_matrix=distance_matrix,
        time_config=TimeConfig(),
        energy_config=EnergyConfig(),
        rule_count=rule_count,
        decision_epochs=decision_epochs,
    )
    instance.validate_scale(scale)
    return instance
