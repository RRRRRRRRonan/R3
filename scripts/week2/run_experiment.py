"""Week 2 Experiment Runner: Adaptive Epsilon Strategies

This script runs a single Q-learning experiment with a specified
epsilon strategy and saves the results to JSON.

Usage:
    python scripts/week2/run_experiment.py \
        --scenario small \
        --epsilon_strategy scale_adaptive \
        --seed 2025 \
        --output results/week2/experiment.json
"""

import argparse
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, Any, Optional, Callable


# Ensure the repository's ``src`` package and root directory are importable when
# the script is executed directly (e.g. ``python scripts/week2/run_experiment.py``).
# On Windows the working directory is often the repo root, which means only
# ``scripts/week2`` ends up on ``sys.path`` by default, so modules like
# ``core`` (under ``src``) and ``tests`` would be missing.  By injecting both
# directories explicitly we avoid requiring callers to pre-set ``PYTHONPATH``.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for _path in (str(_REPO_ROOT), str(_SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


class _NonVerboseProgressPrinter:
    """Minimal progress reporter used when ``--verbose`` is not supplied."""

    def __init__(self, scenario: str, seed: int) -> None:
        self._scenario = scenario
        self._seed = seed
        self._iteration_start: Optional[float] = None

    @property
    def _prefix(self) -> str:
        return f"[Week 2] {self._scenario} seed={self._seed}"

    def __call__(
        self,
        iteration: int,
        total: int,
        best_cost: float,
        event: str,
        is_new_best: bool,
    ) -> None:
        now = time.time()
        if event == "start":
            self._iteration_start = now
            print(
                f"{self._prefix} starting iteration {iteration + 1}/{total}",
                flush=True,
            )
            return

        if event == "end":
            elapsed = (
                now - self._iteration_start
                if self._iteration_start is not None
                else 0.0
            )
            status = "new best" if is_new_best else "best so far"
            print(
                f"{self._prefix} iteration {iteration + 1}/{total} finished in "
                f"{elapsed:.1f}s ({status}: {best_cost:.2f}m)",
                flush=True,
            )
            return

        if event == "complete":
            print(
                f"{self._prefix} completed {total} iterations; "
                f"best cost {best_cost:.2f}m",
                flush=True,
            )
            return

from core.route import create_empty_route
from planner.alns_matheuristic import MatheuristicALNS
from planner.epsilon_strategy import EpsilonStrategy
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
    epsilon_strategy_name: str,
    seed: int,
    verbose: bool = False,
    adapt_matheuristic: bool = True,
) -> Dict[str, Any]:
    """Run a single Q-learning experiment with epsilon strategy.

    Args:
        scenario_scale: Scenario size ("small", "medium", "large")
        epsilon_strategy_name: Epsilon strategy name
        seed: Random seed for reproducibility
        verbose: Whether to print progress
        adapt_matheuristic: If ``True`` (default), allow the solver to expand
            matheuristic hyper-parameters based on the detected scenario scale.
            Set to ``False`` to keep the lightweight parameters defined in
            this script for faster but potentially lower-quality runs.

    Returns:
        Dictionary containing experiment results
    """

    # Build scenario
    config = get_scale_config(scenario_scale)
    scenario = build_scenario(config)
    task_pool = scenario.create_task_pool()
    num_requests = len(scenario.requests)

    # Create epsilon strategy
    try:
        epsilon_strategy = EpsilonStrategy.from_name(
            epsilon_strategy_name, num_requests=num_requests
        )
    except ValueError as e:
        raise ValueError(f"Invalid epsilon_strategy: {e}")

    # Get appropriate iterations for scale
    iterations = get_solver_iterations(scenario_scale, "q_learning")

    if verbose:
        print(f"[Week 2 Experiment]")
        print(f"  Scenario: {scenario_scale}")
        print(f"  Epsilon Strategy: {epsilon_strategy}")
        print(f"  Seed: {seed}")
        print(f"  Iterations: {iterations}")
        if not adapt_matheuristic:
            print("  Matheuristic adaptation: disabled (lightweight tuning)")

    # Configure hyperparameters - same as Week 1 for consistency
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

    start_time = time.time()

    progress_callback: Optional[Callable[[int, int, float, str, bool], None]] = None
    progress_printer: Optional[_NonVerboseProgressPrinter] = None
    if not verbose:
        progress_printer = _NonVerboseProgressPrinter(
            scenario=scenario_scale,
            seed=seed,
        )
        progress_callback = progress_printer

    # Create ALNS with Q-learning and epsilon strategy
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
        adapt_matheuristic_params=adapt_matheuristic,
        epsilon_strategy=epsilon_strategy,  # NEW: Pass epsilon strategy
    )

    # NOTE: Week 1 finding - Q-init doesn't matter, so we use ZERO (default)
    # The epsilon_strategy was already passed to MatheuristicALNS.__init__
    # and will be forwarded to the Q-learning agent automatically

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
        progress_callback=progress_callback,
    )

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
        "epsilon_strategy": epsilon_strategy_name,
        "epsilon_config": {
            "initial": epsilon_strategy.initial_epsilon,
            "decay": epsilon_strategy.decay_rate,
            "min": epsilon_strategy.min_epsilon,
        },
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
        description="Run Week 2 Q-learning epsilon strategy experiment"
    )
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=["small", "medium", "large"],
        help="Scenario scale",
    )
    parser.add_argument(
        "--epsilon_strategy",
        type=str,
        required=True,
        choices=["current", "scale_adaptive", "high_uniform"],
        help="Epsilon exploration strategy",
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
    parser.add_argument(
        "--disable_matheuristic_adaptation",
        action="store_true",
        help="Keep the lightweight matheuristic hyper-parameters without "
        "scale-based expansion (speeds up medium runs at the cost of "
        "solution quality).",
    )

    args = parser.parse_args()

    if not args.verbose:
        print(
            f"[Week 2] Running scenario={args.scenario} seed={args.seed} "
            f"epsilon={args.epsilon_strategy} → {args.output}"
        )

    # Run experiment
    results = run_single_experiment(
        scenario_scale=args.scenario,
        epsilon_strategy_name=args.epsilon_strategy,
        seed=args.seed,
        verbose=args.verbose,
        adapt_matheuristic=not args.disable_matheuristic_adaptation,
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    if args.verbose:
        print(f"\nResults saved to: {args.output}")
    else:
        print(
            f"[Week 2] Completed scenario={args.scenario} seed={args.seed} "
            f"epsilon={args.epsilon_strategy} (runtime: {results['runtime']:.1f}s)"
        )


if __name__ == "__main__":
    main()
