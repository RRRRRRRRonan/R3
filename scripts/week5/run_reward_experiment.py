#!/usr/bin/env python3
"""
Week 5 Experiment Runner: Scale-Aware Reward Normalization

Tests OLD (baseline ROI-aware) vs. NEW (scale-aware) reward functions.

Usage:
    python scripts/week5/run_reward_experiment.py --scale small --reward old --seed 2025
    python scripts/week5/run_reward_experiment.py --scale medium --reward new --seed 2026
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Ensure the repository's ``src`` package and root directory are importable
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dataclasses import replace

from tests.optimization.common import build_scenario, get_scale_config, get_solver_iterations
from planner.alns_matheuristic import MatheuristicALNS
from config import (
    ALNSHyperParameters,
    CostParameters,
    DestroyRepairParams,
    MatheuristicParams,
    SegmentOptimizationParams,
    LPRepairParams,
    DEFAULT_ALNS_HYPERPARAMETERS,
)
from strategy.charging_strategies import PartialRechargeMinimalStrategy
from core.route import create_empty_route
import random


def run_single_experiment(
    scenario_scale: str,
    reward_type: str,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Run a single Week 5 experiment.

    Args:
        scenario_scale: "small", "medium", or "large"
        reward_type: "old" (ROI-aware baseline) or "new" (scale-aware)
        seed: Random seed
        verbose: Print debug output

    Returns:
        Dictionary with results
    """
    print(f"[Week 5] Running {scenario_scale} scale, {reward_type} reward, seed {seed}...")

    # Build scenario
    config = get_scale_config(scenario_scale)
    scenario = build_scenario(config)
    task_pool = scenario.create_task_pool()
    num_requests = len(scenario.tasks)

    # Get appropriate iterations for scale
    iterations = get_solver_iterations(scenario_scale, "q_learning")

    if verbose:
        print(f"[Week 5 Experiment]")
        print(f"  Scenario: {scenario_scale}")
        print(f"  Reward Type: {reward_type}")
        print(f"  Seed: {seed}")
        print(f"  Iterations: {iterations}")

    # Configure hyperparameters - same as Week 1/2 for consistency
    tuned_hyper = replace(
        DEFAULT_ALNS_HYPERPARAMETERS,
        destroy_repair=DestroyRepairParams(
            random_removal_q=2,
            partial_removal_q=2,
            remove_cs_probability=0.2,
        ),
        matheuristic=MatheuristicParams(
            elite_pool_size=2,
            intensification_interval=50,
            segment_frequency=0,  # DISABLED
            max_elite_trials=1,
            segment_optimization=SegmentOptimizationParams(
                max_segment_tasks=2,
                candidate_pool_size=2,
                improvement_tolerance=1e-3,
                max_permutations=4,
                lookahead_window=1,
            ),
            lp_repair=LPRepairParams(
                time_limit_s=2.0,
                max_plans_per_task=2,
                improvement_tolerance=1e-3,
                skip_penalty=5_000.0,
                fractional_threshold=1e-3,
            ),
        ),
    )

    # Setup charging strategy and cost parameters
    charging_strategy = PartialRechargeMinimalStrategy(safety_margin=0.02, min_margin=0.0)
    cost_params = CostParameters()

    # Set random seed for reproducibility
    random.seed(seed)

    # Determine reward type
    use_scale_aware_reward = (reward_type == "new")

    start_time = time.time()

    # Create ALNS with specified reward strategy
    alns = MatheuristicALNS(
        distance_matrix=scenario.distance,
        task_pool=task_pool,
        repair_mode="adaptive",
        cost_params=cost_params,
        charging_strategy=charging_strategy,
        use_adaptive=True,
        verbose=verbose,
        adaptation_mode="q_learning",
        hyper_params=tuned_hyper,
        adapt_matheuristic_params=True,
        use_scale_aware_reward=use_scale_aware_reward,
    )

    # Set vehicle and energy config
    alns.vehicle = scenario.vehicles[0]
    alns.energy_config = scenario.energy

    # Create baseline solution
    initial_route = create_empty_route(vehicle_id=1, depot_node=scenario.depot)
    removed_task_ids = [task.task_id for task in scenario.tasks]
    baseline = alns.greedy_insertion(initial_route, removed_task_ids)

    if hasattr(alns, "_segment_optimizer"):
        alns._segment_optimizer._ensure_schedule(baseline)
        baseline_cost = alns._safe_evaluate(baseline)
    else:
        baseline_cost = alns.evaluate_cost(baseline)

    if verbose:
        print(f"  Baseline cost: {baseline_cost:.2f}")

    # Optimize
    optimised_route = alns.optimize(
        baseline,
        max_iterations=iterations,
    )

    elapsed_time = time.time() - start_time

    # Collect results
    if hasattr(alns, "_segment_optimizer"):
        final_cost = alns._safe_evaluate(optimised_route)
    else:
        final_cost = alns.evaluate_cost(optimised_route)

    improvement_ratio = (baseline_cost - final_cost) / baseline_cost * 100

    # Get Q-learning diagnostics
    if alns._q_agent:
        q_agent = alns._q_agent
        q_diagnostics = {
            "final_q_values": {
                str(state): {str(action): float(q) for action, q in actions.items()}
                for state, actions in q_agent.q_table.items()
            },
            "operator_counts": {str(k): int(v) for k, v in q_agent.action_counts.items()},
            "final_epsilon": float(q_agent.epsilon),
        }
    else:
        q_diagnostics = {}

    # Get scale info (if using scale-aware reward)
    if use_scale_aware_reward and hasattr(alns, '_reward_calculator') and alns._reward_calculator is not None:
        scale_info = alns._reward_calculator.get_scale_info()
    else:
        scale_info = None

    # Collect anytime performance (cost at specific iterations)
    cost_history = getattr(alns, '_cost_history', [])
    anytime_checkpoints = [100, 250, 500, 750, 1000]
    anytime_costs = {}
    for checkpoint in anytime_checkpoints:
        if checkpoint <= len(cost_history):
            anytime_costs[f"cost_at_{checkpoint}"] = float(cost_history[checkpoint - 1])

    # Build result dictionary
    result = {
        "scenario_scale": scenario_scale,
        "num_requests": num_requests,
        "reward_type": reward_type,
        "seed": seed,
        "baseline_cost": float(baseline_cost),
        "final_cost": float(final_cost),
        "improvement_ratio": float(improvement_ratio),
        "iterations_to_best": int(getattr(alns, '_iteration_of_best', 0)),
        "total_iterations": iterations,
        "elapsed_time": float(elapsed_time),
        "anytime_costs": anytime_costs,
        "q_diagnostics": q_diagnostics,
        "scale_info": scale_info,
    }

    print(f"[Week 5] Completed: improvement={improvement_ratio:.2f}%, "
          f"baseline={baseline_cost:.1f}, final={final_cost:.1f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Week 5 Reward Normalization Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/week5/run_reward_experiment.py --scale small --reward old --seed 2025
  python scripts/week5/run_reward_experiment.py --scale large --reward new --seed 2034

Reward types:
  old = ROI-aware baseline (current implementation)
  new = Scale-aware normalized rewards (Week 5 innovation)
        """
    )
    parser.add_argument(
        "--scale",
        required=True,
        choices=["small", "medium", "large"],
        help="Problem scale"
    )
    parser.add_argument(
        "--reward",
        required=True,
        choices=["old", "new"],
        help="Reward function type"
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed"
    )
    parser.add_argument(
        "--output-dir",
        default="results/week5/reward_experiments",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run experiment
    try:
        result = run_single_experiment(
            scenario_scale=args.scale,
            reward_type=args.reward,
            seed=args.seed,
            verbose=args.verbose,
        )

        # Save result
        output_file = output_dir / f"reward_{args.reward}_{args.scale}_seed{args.seed}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"\n[Week 5] Results saved to: {output_file}")
        print(f"[Week 5] Summary: {args.scale} scale, {args.reward} reward, "
              f"improvement={result['improvement_ratio']:.2f}%")

        return 0

    except Exception as e:
        print(f"\n[ERROR] Experiment failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
