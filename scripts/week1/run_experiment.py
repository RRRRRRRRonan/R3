"""Week 1 Experiment Runner: Q-table Initialization Strategies

This script runs a single Q-learning experiment with a specified
initialization strategy and saves the results to JSON.

Usage:
    python scripts/week1/run_experiment.py \
        --scenario small \
        --init_strategy uniform \
        --seed 2025 \
        --output results/week1/experiment.json
"""

import argparse
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any

from core.route import create_empty_route
from planner.alns_matheuristic import MatheuristicALNS
from planner.q_learning_init import QInitStrategy
from strategy.charging_strategies import PartialRechargeMinimalStrategy
from config import (
    DEFAULT_ALNS_HYPERPARAMETERS,
    CostParameters,
    DestroyRepairParams,
    LPRepairParams,
    MatheuristicParams,
    SegmentOptimizationParams,
)
from tests.optimization.common import (
    get_scale_config,
    build_scenario,
    get_solver_iterations,
)


def run_single_experiment(
    scenario_scale: str,
    init_strategy: str,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a single Q-learning experiment.

    Args:
        scenario_scale: Scenario size ("small", "medium", "large")
        init_strategy: Q-table initialization strategy
        seed: Random seed for reproducibility
        verbose: Whether to print progress

    Returns:
        Dictionary containing experiment results
    """

    # Parse initialization strategy
    try:
        q_init = QInitStrategy(init_strategy)
    except ValueError:
        raise ValueError(
            f"Invalid init_strategy '{init_strategy}'. "
            f"Valid options: {[s.value for s in QInitStrategy]}"
        )

    # Build scenario
    config = get_scale_config(scenario_scale)
    scenario = build_scenario(config)
    task_pool = scenario.create_task_pool()

    # Get appropriate iterations for scale
    iterations = get_solver_iterations(scenario_scale, "q_learning")

    if verbose:
        print(f"[Week 1 Experiment]")
        print(f"  Scenario: {scenario_scale}")
        print(f"  Init Strategy: {init_strategy}")
        print(f"  Seed: {seed}")
        print(f"  Iterations: {iterations}")

    # Configure hyperparameters - aggressively simplified for stable execution
    # Focus on Q-learning comparison, minimize matheuristic complexity
    tuned_hyper = replace(
        DEFAULT_ALNS_HYPERPARAMETERS,
        destroy_repair=DestroyRepairParams(
            random_removal_q=2,
            partial_removal_q=2,
            remove_cs_probability=0.2,
        ),
        matheuristic=MatheuristicParams(
            elite_pool_size=2,
            intensification_interval=50,  # Increased to reduce frequency
            segment_frequency=0,  # DISABLED - segment optimization can be slow
            max_elite_trials=1,
            segment_optimization=SegmentOptimizationParams(
                max_segment_tasks=2,
                candidate_pool_size=2,
                improvement_tolerance=1e-3,
                max_permutations=4,  # Further reduced
                lookahead_window=1,
            ),
            lp_repair=LPRepairParams(
                time_limit_s=2.0,  # Increased timeout to prevent hanging
                max_plans_per_task=2,  # Reduced complexity
                improvement_tolerance=1e-3,  # Relaxed tolerance
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

    start_time = time.time()

    # Create ALNS with Q-learning
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
    )

    # CRITICAL: Inject initialization strategy into Q-agent
    # The Q-agent is created in MatheuristicALNS.__init__, so we need to
    # recreate it with our custom initialization
    from planner.q_learning import QLearningOperatorAgent

    alns._q_agent = QLearningOperatorAgent(
        destroy_operators=alns._destroy_operators,
        repair_operators=alns.repair_operators,
        params=alns.hyper.q_learning,
        init_strategy=q_init,  # Use our custom init strategy
    )

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
    optimised_route = alns.optimize(baseline, max_iterations=iterations)

    if hasattr(alns, "_segment_optimizer"):
        alns._segment_optimizer._ensure_schedule(optimised_route)
        optimised_cost = alns._safe_evaluate(optimised_route)
    else:
        optimised_cost = alns.evaluate_cost(optimised_route)

    runtime = time.time() - start_time

    if verbose:
        print(f"  Optimised cost: {optimised_cost:.2f}")
        print(f"  Improvement: {(1 - optimised_cost / baseline_cost) * 100:.2f}%")
        print(f"  Runtime: {runtime:.2f}s")

    # Collect Q-learning statistics
    q_agent = alns._q_agent
    q_stats = q_agent.statistics()

    # Extract Q-values
    q_values_by_state = {}
    for state, stats_list in q_stats.items():
        q_values_by_state[state] = {
            str(stat.action): stat.average_q_value for stat in stats_list
        }

    # Calculate improvement ratio
    improvement_ratio = (baseline_cost - optimised_cost) / baseline_cost

    # Prepare results
    results = {
        "scenario": scenario_scale,
        "init_strategy": init_strategy,
        "seed": seed,
        "iterations": iterations,
        "baseline_cost": float(baseline_cost),
        "optimised_cost": float(optimised_cost),
        "improvement_ratio": float(improvement_ratio),
        "runtime": float(runtime),
        "final_epsilon": float(q_agent.epsilon),
        "q_values": q_values_by_state,
        "experiment_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Week 1 Q-learning initialization experiment"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["small", "medium", "large"],
        help="Scenario scale",
    )
    parser.add_argument(
        "--init_strategy",
        type=str,
        required=True,
        choices=["zero", "uniform", "action_specific", "state_specific"],
        help="Q-table initialization strategy",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSON file path",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress information",
    )

    args = parser.parse_args()

    # Run experiment
    results = run_single_experiment(
        scenario_scale=args.scenario,
        init_strategy=args.init_strategy,
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if args.verbose:
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
