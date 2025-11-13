"""Regression tests for the matheuristic ALNS variant."""

from copy import deepcopy
from dataclasses import replace

from config import (
    CostParameters,
    MatheuristicParams,
    SegmentOptimizationParams,
    VehicleDynamics,
    DEFAULT_OPTIMIZATION_SCENARIO,
)
from core.route import create_empty_route
from planner.alns_matheuristic import MatheuristicALNS

from tests.optimization.common import ScenarioConfig, build_scenario


def test_matheuristic_alns_populates_elite_pool_and_improves_cost():
    """The matheuristic solver should outperform its greedy seed and keep elites."""

    scenario = build_scenario(
        ScenarioConfig.from_defaults(
            DEFAULT_OPTIMIZATION_SCENARIO,
            num_tasks=6,
            num_charging=2,
            seed=21,
        )
    )

    mat_params = MatheuristicParams(
        elite_pool_size=4,
        intensification_interval=10,
        segment_frequency=1,
        max_elite_trials=2,
        segment_optimization=SegmentOptimizationParams(
            max_segment_tasks=3,
            candidate_pool_size=3,
            improvement_tolerance=1e-4,
            max_permutations=6,
            lookahead_window=2,
        ),
    )

    task_pool = scenario.create_task_pool()

    alns = MatheuristicALNS(
        distance_matrix=scenario.distance,
        task_pool=task_pool,
        repair_mode="adaptive",
        cost_params=CostParameters(),
        charging_strategy=None,
        use_adaptive=True,
        verbose=False,
        matheuristic_params=mat_params,
    )
    alns.vehicle = deepcopy(scenario.vehicles[0])
    alns.energy_config = deepcopy(scenario.energy)

    initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
    removed_task_ids = [task.task_id for task in scenario.tasks]
    baseline = alns.greedy_insertion(initial_route, removed_task_ids)
    baseline_cost = alns._safe_evaluate(baseline)

    optimized = alns.optimize(baseline, max_iterations=4)
    optimized_cost = alns._safe_evaluate(optimized)

    assert optimized_cost <= baseline_cost
    assert len(optimized.get_served_tasks()) == len(baseline.get_served_tasks())
    assert len(alns._elite_pool) > 0


def test_matheuristic_optimize_emits_progress_events():
    """A progress callback should receive start/end/complete notifications."""

    scenario = build_scenario(
        ScenarioConfig.from_defaults(
            DEFAULT_OPTIMIZATION_SCENARIO,
            num_tasks=5,
            num_charging=2,
            seed=7,
        )
    )
    task_pool = scenario.create_task_pool()

    alns = MatheuristicALNS(
        distance_matrix=scenario.distance,
        task_pool=task_pool,
        repair_mode="adaptive",
        cost_params=CostParameters(),
        charging_strategy=None,
        use_adaptive=True,
        verbose=False,
    )
    alns.vehicle = deepcopy(scenario.vehicles[0])
    alns.energy_config = deepcopy(scenario.energy)

    baseline = alns.greedy_insertion(
        create_empty_route(vehicle_id=1, depot_node=scenario.depot),
        [task.task_id for task in scenario.tasks],
    )

    events = []

    def progress(iteration, total, best_cost, event, is_new_best):
        events.append((event, iteration, is_new_best, best_cost))

    iterations = 3
    alns.optimize(baseline, max_iterations=iterations, progress_callback=progress)

    starts = [event for event in events if event[0] == "start"]
    ends = [event for event in events if event[0] == "end"]

    assert len(starts) == iterations
    assert len(ends) == iterations
    assert events[-1][0] == "complete"


def test_lp_repair_operator_improves_weighted_tardiness():
    """The LP-based repair should not regress compared to regret-2 insertion."""

    scenario = build_scenario(
        ScenarioConfig.from_defaults(
            DEFAULT_OPTIMIZATION_SCENARIO,
            num_tasks=4,
            num_charging=1,
            pickup_tw_width=60.0,
            delivery_gap=15.0,
            service_time=75.0,
            seed=17,
        )
    )

    prioritized_tasks = [
        replace(task, priority=(3 if task.task_id >= 3 else 1)) for task in scenario.tasks
    ]
    scenario.tasks = prioritized_tasks
    task_pool = scenario.create_task_pool()

    cost_params = CostParameters(
        C_tr=1.0,
        C_ch=0.0,
        C_time=0.0,
        C_delay=8.0,
        C_wait=0.0,
        C_missing_task=5000.0,
        C_infeasible=10000.0,
    )

    alns = MatheuristicALNS(
        distance_matrix=scenario.distance,
        task_pool=task_pool,
        repair_mode="lp",
        cost_params=cost_params,
        charging_strategy=None,
        use_adaptive=False,
        verbose=False,
    )
    alns.vehicle = deepcopy(scenario.vehicles[0])
    alns.energy_config = deepcopy(scenario.energy)
    alns.hyper = replace(
        alns.hyper,
        vehicle=VehicleDynamics(
            cruise_speed_m_s=alns.vehicle.speed,
            max_energy_adjustment_iterations=alns.hyper.vehicle.max_energy_adjustment_iterations,
        ),
    )

    initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
    all_task_ids = [task.task_id for task in prioritized_tasks]
    seeded_route = alns.greedy_insertion(initial_route, all_task_ids)

    removed_ids = [prioritized_tasks[-1].task_id, prioritized_tasks[-2].task_id]
    partial_route = seeded_route.copy()
    for task_id in removed_ids:
        task = task_pool.get_task(task_id)
        partial_route.remove_task(task)

    fallback_route = alns.regret2_insertion(partial_route.copy(), removed_ids)
    lp_route = alns.lp_insertion(partial_route.copy(), removed_ids)

    fallback_cost = alns.evaluate_cost(fallback_route)
    lp_cost = alns.evaluate_cost(lp_route)

    def total_tardiness(route):
        alns.ensure_route_schedule(route)
        tardiness = 0.0
        for visit in route.visits:
            if hasattr(visit.node, "time_window") and visit.node.time_window:
                tardiness += max(0.0, visit.start_service_time - visit.node.time_window.latest)
        return tardiness

    assert lp_cost <= fallback_cost + 1e-6
    assert total_tardiness(lp_route) <= total_tardiness(fallback_route) + 1e-6
