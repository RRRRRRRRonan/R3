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
            # CRITICAL FIX: Faster learning for online setting
            alpha=0.25,  # Increased from 0.1 - learn faster in limited iterations
            gamma=0.9,

            # CRITICAL FIX 2.0: Balanced exploration that persists into stuck phase
            # Previous problem: epsilon=0.05 with decay=0.5 meant exploration ended
            # at iteration 3, BEFORE Q-learning could learn LP's value in stuck state
            #
            # New strategy: Higher initial epsilon + slower decay
            # - Iterations 1-5: 15-9% exploration (learn LP works)
            # - Iterations 6-10: 7-5% exploration (refine strategy)
            # - Iterations 11-20: 4-1.5% exploration (exploit learned policy)
            initial_epsilon=0.15,  # Increased from 0.05 - need real exploration
            epsilon_decay=0.88,    # Increased from 0.5 - sustain exploration longer
            epsilon_min=0.01,      # Lower floor for minimal exploration
            enable_online_updates=True,

            # Reward structure optimized for ROI-aware learning
            reward_new_best=50.0,
            reward_improvement=20.0,
            reward_accepted=5.0,
            reward_rejected=-2.0,

            # Time penalty thresholds - encourage matheuristic when it delivers
            time_penalty_threshold=0.1,
            time_penalty_positive_scale=2.0,   # Reduced from 2.5 - less penalty for good results
            time_penalty_negative_scale=10.0,  # Increased from 7.5 - heavier penalty for waste
            standard_time_penalty_scale=0.5,   # Reduced from 0.75 - cheaper ops are cheap

            # CRITICAL FIX: Earlier state transitions for large-scale problems
            # Now that LP is available in explore phase, we can be more conservative
            # with state transitions - let Q-learning learn naturally
            stagnation_threshold=200,          # Legacy absolute threshold (unused if ratio set)
            deep_stagnation_threshold=800,     # Legacy absolute threshold (unused if ratio set)
            stagnation_ratio=0.1,              # Reduced from 0.15 - stuck after 4 iterations
            deep_stagnation_ratio=0.35,        # Reduced from 0.4 - deep_stuck after 14-15 iterations
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
