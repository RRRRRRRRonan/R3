"""Model specification for the MIP baseline benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from config import CostParameters
from core.node import ChargingNode, DepotNode
from core.task import Task, TaskPool
from core.vehicle import Vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from physics.time import TimeConfig

from baselines.mip.config import MIPBaselineScale, MIPBaselineSolverConfig


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
) -> MIPBaselineInstance:
    """Build a baseline instance from core objects."""

    instance = MIPBaselineInstance(
        tasks=list(task_pool.get_all_tasks()),
        vehicles=list(vehicles),
        depot=depot,
        charging_stations=list(charging_stations),
        distance_matrix=distance_matrix,
        time_config=time_config or TimeConfig(),
        energy_config=energy_config or EnergyConfig(),
        rule_count=rule_count,
        decision_epochs=decision_epochs,
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
        "Delta_t_safe: headway safety margin",
        "delta_r^w: dynamic task availability",
        "p_w: scenario probability",
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
        "w_i^{a,w}: conflict waiting",
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
        "Time windows with tardiness capturing lateness.",
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
