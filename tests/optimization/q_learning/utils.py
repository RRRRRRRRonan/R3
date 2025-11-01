"""Helpers for running matheuristic ALNS with Q-learning enabled in tests."""

from __future__ import annotations

from copy import deepcopy
from typing import Tuple
import random
from dataclasses import replace

from core.route import create_empty_route
from planner.alns_matheuristic import MatheuristicALNS
from planner.q_learning import QLearningOperatorAgent
from tests.optimization.common import ScenarioConfig, build_scenario
from config import (
    DEFAULT_ALNS_HYPERPARAMETERS,
    DestroyRepairParams,
    LPRepairParams,
    MatheuristicParams,
    QLearningParams,
    SegmentOptimizationParams,
)


def run_q_learning_trial(
    config: ScenarioConfig,
    *,
    iterations: int,
    seed: int = 2025,
) -> Tuple[MatheuristicALNS, float, float]:
    """Build a scenario and execute matheuristic ALNS with Q-learning enabled.

    Returns the solver instance together with the baseline and optimised costs
    so tests can assert on both solution quality and the learning signals.
    """
    from strategy.charging_strategies import PartialRechargeMinimalStrategy
    from config import CostParameters

    scenario = build_scenario(config)
    task_pool = scenario.create_task_pool()

    # Use SAME configuration as matheuristic baseline for fair comparison
    # The ONLY difference should be that we explicitly use Q-learning
    # (Matheuristic also uses Q-learning by default, but we make it explicit here)
    tuned_hyper = replace(
        DEFAULT_ALNS_HYPERPARAMETERS,
        destroy_repair=DestroyRepairParams(
            random_removal_q=2,
            partial_removal_q=2,
            remove_cs_probability=0.2,
        ),
        matheuristic=MatheuristicParams(
            elite_pool_size=4,
            intensification_interval=25,
            segment_frequency=6,
            max_elite_trials=2,
            segment_optimization=SegmentOptimizationParams(
                max_segment_tasks=3,
                candidate_pool_size=3,
                improvement_tolerance=1e-3,
                max_permutations=12,
                lookahead_window=2,
            ),
            lp_repair=LPRepairParams(
                time_limit_s=0.3,
                max_plans_per_task=4,
                improvement_tolerance=1e-4,
                skip_penalty=5_000.0,
                fractional_threshold=1e-3,
            ),
        ),
        # Use DEFAULT Q-learning parameters - don't override!
        # Matheuristic performs well with defaults, so we should too
    )

    # CRITICAL FIX: Use the SAME charging strategy and cost params as matheuristic
    # This ensures fair comparison with identical baseline quality
    charging_strategy = PartialRechargeMinimalStrategy(safety_margin=0.02, min_margin=0.0)
    cost_params = CostParameters()

    # Don't provide custom initial Q-values - let it use defaults
    # Matheuristic doesn't provide custom Q-values, so we shouldn't either

    rng = random.Random(seed)
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        # Create MatheuristicALNS with EXACTLY the same config as matheuristic trial
        # The only difference is we don't override anything - both use default Q-learning
        alns = MatheuristicALNS(
            distance_matrix=scenario.distance,
            task_pool=task_pool,
            repair_mode="adaptive",
            cost_params=cost_params,
            charging_strategy=charging_strategy,
            use_adaptive=True,
            verbose=False,
            hyper_params=tuned_hyper,
        )
        # Don't manually create Q-agent - let MinimalALNS.__init__ do it with defaults
        alns.vehicle = deepcopy(scenario.vehicles[0])
        alns.energy_config = deepcopy(scenario.energy)

        initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
        removed_task_ids = [task.task_id for task in scenario.tasks]
        baseline = alns.greedy_insertion(initial_route, removed_task_ids)
        if hasattr(alns, "_segment_optimizer"):
            alns._segment_optimizer._ensure_schedule(baseline)
            baseline_cost = alns._safe_evaluate(baseline)
        else:
            baseline_cost = alns.evaluate_cost(baseline)

        optimised_route = alns.optimize(baseline, max_iterations=iterations)
        if hasattr(alns, "_segment_optimizer"):
            alns._segment_optimizer._ensure_schedule(optimised_route)
            optimised_cost = alns._safe_evaluate(optimised_route)
        else:
            optimised_cost = alns.evaluate_cost(optimised_route)
    finally:
        random.setstate(state)

    return alns, baseline_cost, optimised_cost
