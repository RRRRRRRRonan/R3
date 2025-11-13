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

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scenario.scenario_builder import build_scenario, get_scale_config
from planner.alns_matheuristic import MatheuristicALNS
from planner.alns import ALNSHyperParameters
from cost.cost_calculator import CostParameters
from charging.adaptive_charging import AdaptiveChargingStrategy


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
    config["seed"] = seed
    scenario = build_scenario(config)
    task_pool = scenario.create_task_pool()
    num_requests = len(scenario.tasks)

    # Cost parameters
    cost_params = CostParameters(
        distance_weight=1.0,
        time_weight=0.1,
        battery_weight=0.05,
    )

    # ALNS hyperparameters (tuned configuration)
    tuned_hyper = ALNSHyperParameters(
        max_iterations=1000,
        time_limit=300,
        destroy_pct=0.25,
        temperature_start=10000,
        temperature_decay=0.995,
        temperature_threshold=0.01,
    )

    # Charging strategy
    charging_strategy = AdaptiveChargingStrategy(
        soc_threshold=0.2,
        charge_to=0.8,
        enable_partial=True,
    )

    # Determine reward type
    use_scale_aware_reward = (reward_type == "new")

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

    # Run optimization
    start_time = time.time()
    best_solution = alns.optimize()
    elapsed_time = time.time() - start_time

    # Collect results
    baseline_cost = alns.baseline_cost
    final_cost = best_solution.total_cost
    improvement_ratio = (baseline_cost - final_cost) / baseline_cost * 100

    # Get Q-learning diagnostics
    q_agent = alns._q_agent
    q_diagnostics = {
        "final_q_values": {
            str(state): {str(action): float(q) for action, q in actions.items()}
            for state, actions in q_agent.q_table.items()
        },
        "operator_counts": {str(k): int(v) for k, v in q_agent.action_counts.items()},
        "final_epsilon": float(q_agent.epsilon),
    }

    # Get scale info (if using scale-aware reward)
    if use_scale_aware_reward and alns._reward_calculator is not None:
        scale_info = alns._reward_calculator.get_scale_info()
    else:
        scale_info = None

    # Collect anytime performance (cost at specific iterations)
    cost_history = getattr(alns, 'cost_history', [])
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
        "iterations_to_best": int(getattr(alns, 'iteration_of_best', 0)),
        "total_iterations": int(getattr(alns, 'iteration', 0)),
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
