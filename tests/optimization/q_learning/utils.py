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

    scenario = build_scenario(config)
    task_pool = scenario.create_task_pool()

    # Use SAME matheuristic strength as the pure matheuristic baseline
    # This ensures fair comparison - Q-learning's value is in WHEN to use
    # these operators, not in having weaker operators
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
                time_limit_s=0.3,  # Increased from 0.06 to 0.3 (5x stronger)
                max_plans_per_task=4,  # Increased from 1 to 4 (4x stronger)
                improvement_tolerance=1e-4,
                skip_penalty=5_000.0,
                fractional_threshold=1e-3,
            ),
        ),
        q_learning=QLearningParams(
            alpha=0.1,
            gamma=0.9,
            # Reduced initial exploration for faster convergence
            initial_epsilon=0.2,  # Reduced from 0.4
            epsilon_decay=0.85,   # Faster decay from 0.92
            epsilon_min=0.05,
            enable_online_updates=True,
            # Reward structure optimized for ROI-aware learning
            reward_new_best=50.0,
            reward_improvement=20.0,
            reward_accepted=5.0,
            reward_rejected=-2.0,
            # Time penalty thresholds
            time_penalty_threshold=0.1,
            time_penalty_positive_scale=2.5,  # Moderate penalty for expensive+good
            time_penalty_negative_scale=7.5,  # Heavy penalty for expensive+bad
            standard_time_penalty_scale=0.75, # Light penalty for cheap operators
            # State transition thresholds (scaled by iterations)
            stagnation_threshold=200,
            deep_stagnation_threshold=800,
            stagnation_ratio=0.25,
            deep_stagnation_ratio=0.6,
        ),
    )

    rng = random.Random(seed)
    state = random.getstate()
    random.setstate(rng.getstate())
    try:
        alns = MatheuristicALNS(
            distance_matrix=scenario.distance,
            task_pool=task_pool,
            repair_mode="adaptive",
            cost_params=None,
            charging_strategy=None,
            use_adaptive=True,
            verbose=False,
            hyper_params=tuned_hyper,
        )
        if getattr(alns, "_use_q_learning", False):
            alns._q_agent = QLearningOperatorAgent(
                destroy_operators=alns._destroy_operators,
                repair_operators=alns.repair_operators,
                params=alns._q_params,
            )
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
