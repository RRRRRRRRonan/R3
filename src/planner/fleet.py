"""Fleet-level planner that orchestrates multiple single-vehicle ALNS runs.

The legacy codebase focused on optimising a single AMR route via the
``MinimalALNS`` solver.  This module adds a thin coordination layer that
distributes tasks across several vehicles and reuses the well-tested single
vehicle planner to optimise each sub-problem.  The goal is not to provide a
full cooperative search (that remains the responsibility of the CBS layer),
but to deliver a practical way to generate one route per vehicle so the rest of
the stack can simulate multi-AMR operations.

Design choices
--------------
* **Task partitioning.**  Tasks are split into roughly even batches using a
  deterministic round allocation (ceil division).  This keeps the
  implementation simple while ensuring balanced workloads when the fleet has a
  homogeneous configuration.  The partitioning helper is isolated so smarter
  heuristics (distance-based clustering, dynamic assignment, etc.) can be added
  later without touching the optimisation loop.
* **Solver reuse.**  For each vehicle the planner instantiates ``MinimalALNS``
  with a scoped ``TaskPool`` that only exposes the batch assigned to that
  vehicle.  The same cost parameters, charging strategy, and energy settings are
  forwarded so every route remains consistent with the single vehicle
  behaviour.
* **State propagation.**  After a batch is assigned the method updates the
  master task pool trackers and calls ``vehicle.assign_route`` so execution
  modules immediately know which route belongs to which AMR.

The resulting interface mirrors ``MinimalALNS`` closely but returns a mapping
from vehicle ids to their optimised ``Route`` objects, enabling the rest of the
codebase to reason about multiple robots executing in parallel.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

from core.node import DepotNode
from core.route import Route, create_empty_route
from core.task import TaskPool
from core.vehicle import Vehicle
from physics.distance import DistanceMatrix
from physics.energy import EnergyConfig
from config import CostParameters, DEFAULT_COST_PARAMETERS
from planner.alns import MinimalALNS


@dataclass
class FleetPlanResult:
    """Container storing the outcome of the fleet planner.

    Attributes
    ----------
    routes:
        Mapping from vehicle id to the optimised route assigned to that AMR.
    initial_routes:
        Mapping from vehicle id to the seeded greedy route before the ALNS
        optimisation loop runs.  Useful for callers that want to compare the
        improvement delivered by the local search phase.
    initial_cost:
        Sum of the individual route costs before optimisation.  Uses the same
        weighted objective as ``MinimalALNS``.
    optimised_cost:
        Sum of the route costs after optimisation.  Allows multi-vehicle tests
        to reuse the legacy assertions that check for overall cost reductions.
    unassigned_tasks:
        List of task ids that were not assigned because the fleet ran out of
        vehicles.  The current implementation aims to allocate every task, so
        this list is expected to be empty; the attribute exists to keep the
        return type extensible.
    """

    routes: Dict[int, Route]
    initial_routes: Dict[int, Route]
    initial_cost: float
    optimised_cost: float
    unassigned_tasks: List[int]


class FleetPlanner:
    """Coordinate multiple ``MinimalALNS`` runs to plan routes for a fleet."""

    def __init__(
        self,
        *,
        distance_matrix: DistanceMatrix,
        depot: DepotNode,
        vehicles: Sequence[Vehicle],
        task_pool: TaskPool,
        energy_config: EnergyConfig,
        cost_params: CostParameters | None = None,
        charging_strategy=None,
        repair_mode: str = "mixed",
        use_adaptive: bool = True,
        verbose: bool = False,
    ) -> None:
        if not vehicles:
            raise ValueError("FleetPlanner requires at least one vehicle")

        self.distance = distance_matrix
        self.depot = depot
        self.vehicles = sorted(vehicles, key=lambda v: v.vehicle_id)
        self.task_pool = task_pool
        self.energy_config = energy_config
        self.cost_params = cost_params or DEFAULT_COST_PARAMETERS
        self.charging_strategy = charging_strategy
        self.repair_mode = repair_mode
        self.use_adaptive = use_adaptive
        self.verbose = verbose

    def plan_routes(self, *, max_iterations: int = 100) -> FleetPlanResult:
        """Generate a route for each vehicle in the fleet.

        The algorithm assigns tasks to vehicles via a simple round allocation
        (ceil division) and, for every vehicle, spins up an independent
        ``MinimalALNS`` instance scoped to that subset.  Routes are seeded using
        the greedy insertion heuristic before running the full ALNS loop.
        """

        remaining_task_ids = [task.task_id for task in self.task_pool.get_all_tasks()]
        routes: Dict[int, Route] = {}
        initial_routes: Dict[int, Route] = {}
        initial_cost = 0.0
        optimised_cost = 0.0

        if not remaining_task_ids:
            for vehicle in self.vehicles:
                routes[vehicle.vehicle_id] = create_empty_route(vehicle.vehicle_id, self.depot)
                initial_routes[vehicle.vehicle_id] = routes[vehicle.vehicle_id]
            return FleetPlanResult(
                routes=routes,
                initial_routes=initial_routes,
                initial_cost=0.0,
                optimised_cost=0.0,
                unassigned_tasks=[],
            )

        remaining_vehicles = len(self.vehicles)

        for vehicle in self.vehicles:
            if not remaining_task_ids:
                routes[vehicle.vehicle_id] = create_empty_route(vehicle.vehicle_id, self.depot)
                initial_routes[vehicle.vehicle_id] = routes[vehicle.vehicle_id]
                continue

            batch_size = math.ceil(len(remaining_task_ids) / remaining_vehicles)
            batch_task_ids = remaining_task_ids[:batch_size]
            remaining_task_ids = remaining_task_ids[batch_size:]
            remaining_vehicles -= 1

            sub_pool = TaskPool()
            for task_id in batch_task_ids:
                task = self.task_pool.get_task(task_id)
                if task is None:
                    raise ValueError(f"Task {task_id} not found in the master pool")
                sub_pool.add_task(task)
                # Mark the task as assigned in the master pool for downstream modules.
                self.task_pool.assign_task(task_id, vehicle.vehicle_id)

            planner = MinimalALNS(
                self.distance,
                sub_pool,
                repair_mode=self.repair_mode,
                cost_params=self.cost_params,
                charging_strategy=self.charging_strategy,
                use_adaptive=self.use_adaptive,
                verbose=self.verbose,
            )
            planner.vehicle = vehicle
            planner.energy_config = self.energy_config

            initial_route = create_empty_route(vehicle.vehicle_id, self.depot)
            seeded_route = planner.greedy_insertion(initial_route, batch_task_ids)
            initial_routes[vehicle.vehicle_id] = seeded_route
            initial_cost += planner.evaluate_cost(seeded_route)

            if max_iterations > 0 and not seeded_route.is_empty():
                optimised_route = planner.optimize(seeded_route, max_iterations=max_iterations)
            else:
                optimised_route = seeded_route

            routes[vehicle.vehicle_id] = optimised_route
            optimised_cost += planner.evaluate_cost(optimised_route)
            vehicle.assign_route(optimised_route)

        return FleetPlanResult(
            routes=routes,
            initial_routes=initial_routes,
            initial_cost=initial_cost,
            optimised_cost=optimised_cost,
            unassigned_tasks=remaining_task_ids,
        )


__all__ = ["FleetPlanner", "FleetPlanResult"]
