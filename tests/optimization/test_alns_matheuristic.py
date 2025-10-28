"""Regression tests for the matheuristic ALNS variant."""

from copy import deepcopy

from config import (
    CostParameters,
    MatheuristicParams,
    SegmentOptimizationParams,
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

    optimized = alns.optimize(baseline, max_iterations=30)
    optimized_cost = alns._safe_evaluate(optimized)

    assert optimized_cost <= baseline_cost
    assert len(optimized.get_served_tasks()) == len(baseline.get_served_tasks())
    assert len(alns._elite_pool) > 0
